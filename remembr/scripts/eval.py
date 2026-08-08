import json
import numpy as np

from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from time import strftime, localtime
import numpy as np
import tqdm

import re
import time
import uuid
import sys
import os, sys
import pickle as pkl
from PIL import Image as PILImage
import glob

from dataclasses import asdict
import argparse
import traceback 

# load this directory
sys.path.append(sys.path[0] + '/..')

from agents.remembr_agent import ReMEmbRAgent
from agents.non_agent import NonAgent
from agents.retrieve_agent import RetrieveAgent
from agents.vlm_non_agent import VLMNonAgent

from memory.memory import MemoryItem
from memory.milvus_memory import MilvusMemory
from memory.text_memory import TextMemory
from memory.video_memory import VideoMemory, ImageMemoryItem

from tools.tools import format_docs
from remembr.utils import trace


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def _num_or_none(v):
    """Parse a predicted numeric field (coordinate/minutes). Returns None if the
    model gave null/None or an unparseable string, so the caller can mark the
    question 'unanswered' instead of crashing on `float - None` (TypeError) or
    `eval('null')` (NameError)."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = eval(v)
        except Exception:  # noqa: BLE001  ('null', prose, etc.)
            return None
    return v


def evaluate_output(qa_instance, predicted):

    out_error = {}

    q_type = qa_instance['type']
    if 'position' in q_type:
        pred = _num_or_none(predicted.get('position'))
        if pred is None:
            return {'unanswered': True}
        answer = np.array(qa_instance['answers']['position'])
        out_error['position_error'] = float(np.linalg.norm(answer - np.array(pred)))

    elif 'binary' in q_type:
        # binary answer is the second one (assumption kept from other examples)
        answer = qa_instance['answers']['text'][1]
        pb = predicted.get('binary')
        if isinstance(pb, str) and pb.lower() in ("yes", "no"):
            out_error['binary_iscorrect'] = 1 if pb.lower() == answer.lower() else 0
        else:
            return {'unanswered': True}

    elif 'time' in q_type:
        pred = _num_or_none(predicted.get('time'))
        if pred is None:
            return {'unanswered': True}
        answer = np.array(qa_instance['answers']['time'])
        out_error['time_error'] = float(np.ravel(np.abs(answer - np.array(pred)))[0])

    elif 'duration' in q_type:
        pred = _num_or_none(predicted.get('duration'))
        if pred is None:
            return {'unanswered': True}
        answer = np.array(qa_instance['answers']['duration'])
        out_error['duration_error'] = float(np.ravel(np.abs(answer - np.array(pred)))[0])

    elif 'text' in q_type:
        out_error = {'answer': qa_instance['answers']['text']}

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


# Cap retries per question. The 3B often can't fill the typed field for a
# position/time question and returns null; without a cap the retry loop below
# would re-ask forever. After MAX_ATTEMPTS we accept the last answer (marked
# 'unanswered' by evaluate_output) and move on.
MAX_ATTEMPTS = 5

# Helper function for wrapping memory search methods and appending per-call latency records.
def instrument_memory_search(memory, search_events):
    for method_name in ("search_by_text", "search_by_position", "search_by_time"):
        if not hasattr(memory, method_name):
            continue

        original = getattr(memory, method_name)

        # check what memory search tool was used and records its latency
        def wrapped(query, _original=original, _method_name=method_name):
            start = time.perf_counter()
            record = {
                "method": _method_name,
                "latency_s": None,
                "status": None,
                "error": None,
            }
            try:
                result = _original(query)
                record["status"] = "ok"
                return result
            except Exception as e:
                record["status"] = "exception"
                record["error"] = repr(e)
                raise
            finally:
                record["latency_s"] = time.perf_counter() - start
                search_events.append(record)

        setattr(memory, method_name, wrapped)

# Track retrieval latency for each query attempt.
def answer_squad_question(model, question, qa_instance, search_events=None):

    # search_events is shared with memory search wrappers.
    if search_events is None:
        search_events = []

    trace.plain(f'Question: {question}')

    parsed = None
    out_error = None
    elapsed = 0.0
    last_attempt_elapsed = 0.0
    attempts = 0
    attempt_latencies = []
    total_db_search_latency = 0.0

    for attempt in range(MAX_ATTEMPTS):
        attempts = attempt + 1
        search_start_idx = len(search_events)
        attempt_start = time.perf_counter()
        db_search_latency = 0.0
        llm_latency = None

        attempt_record = {
            'attempt': attempts,
            'latency_s': None,
            'status': None,
            'unanswered': None,
            'error': None,
            'db_search_latency_s': 0.0,
            'llm_latency_s': None,
        }

        try:
            response = model.query(question)
            last_attempt_elapsed = time.perf_counter() - attempt_start

            attempt_search_events = search_events[search_start_idx:]
            db_search_latency = sum((e.get("latency_s") or 0.0) for e in attempt_search_events)
            llm_latency = max(0.0, last_attempt_elapsed - db_search_latency)

            parsed = asdict(response)
            out_error = evaluate_output(qa_instance, parsed)

            attempt_record['status'] = 'ok'
            attempt_record['unanswered'] = bool(out_error.get('unanswered'))

            if trace.ENABLED:
                mark = "미답변" if out_error.get("unanswered") else " ".join(
                    f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in out_error.items())
                trace.step("채점", f"{mark}  attempts={attempts}  {last_attempt_elapsed:.1f}s", arrow=" ")
            else:
                print("Attempt elapsed", last_attempt_elapsed)
                print("Query elapsed total", elapsed + last_attempt_elapsed)

        except Exception as e:
            last_attempt_elapsed = time.perf_counter() - attempt_start

            attempt_search_events = search_events[search_start_idx:]
            db_search_latency = sum((event.get("latency_s") or 0.0) for event in attempt_search_events)
            llm_latency = max(0.0, last_attempt_elapsed - db_search_latency)

            attempt_record['status'] = 'exception'
            attempt_record['error'] = repr(e)

            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info())
            out_error = None

        attempt_record['latency_s'] = last_attempt_elapsed
        attempt_record['db_search_latency_s'] = db_search_latency
        attempt_record['llm_latency_s'] = llm_latency

        elapsed += last_attempt_elapsed
        total_db_search_latency += db_search_latency
        attempt_latencies.append(attempt_record)

        # a clean evaluation: stop if the model actually answered, else retry
        # (a different sample may fill the field) until the attempt cap.
        if out_error is not None and not out_error.get('unanswered'):
            break

    if out_error is None:
        out_error = {'unanswered': True}

    return_dict = {"response": parsed}
    if isinstance(parsed, dict):
        return_dict.update(parsed)
    return_dict['error'] = out_error
    return_dict['elapsed'] = elapsed
    return_dict['query_latency_s'] = elapsed
    return_dict['db_search_latency_s'] = total_db_search_latency
    return_dict['llm_latency_s'] = max(0.0, elapsed - total_db_search_latency)
    return_dict['last_attempt_latency_s'] = last_attempt_elapsed
    return_dict['attempts'] = attempts
    return_dict['attempt_latencies'] = attempt_latencies

    return return_dict


def load_memory(args, qa_instance, use_milvus=True, use_optimal_context=False, ip_address='127.0.0.1'):
    # Here we load everything needed to load a MilvusDB instance neatly
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']


    if use_milvus:
        # milv = MilvusWrapper(ip_address=ip_address)
        memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time)
    elif 'vlm' in args.model:
        memory = VideoMemory()
    else:
        memory = TextMemory()

    memory.reset()


    captions_path = os.path.join(args.data_dir, 'captions', str(args.sequence_id), 'captions', f'{args.caption_file}.json')

    with open(captions_path, 'r') as f:
        out = json.load(f)


    outputs = []

    # Compute start idx
    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))

    # Compute end idx
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))


    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    for i in range(start_idx, end_idx+1):

        item = out[i]
        entity = {
            'position': item['position'],
            'theta': item['theta'], # ignoring rotation
            'time': item['time'], 
            'caption': item['caption'],
        }

        outputs.append(entity)

        if type(memory) == VideoMemory:

            qa_start_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i]['file_start'])
            qa_end_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i+1]['file_start'])

            qa_start_idx = pkl_files.index(qa_start_path)
            qa_end_idx = pkl_files.index(qa_end_path)
            idxs = np.linspace(qa_start_idx, qa_end_idx, 6, dtype=int)

            for pkl_idx in idxs:
                # pkl_path = os.path.join(args.coda_dir, str(args.sequence_id), item['file_start'])
                pkl_path = pkl_files[pkl_idx]
                with open(pkl_path, 'rb') as f:
                    pkl_data = pkl.load(f)
                entity['image'] = PILImage.fromarray(pkl_data['cam0'].astype('uint8'), 'RGB')

            entity = ImageMemoryItem.from_dict(entity)
        else:
            entity = MemoryItem.from_dict(entity)

        if use_milvus:
            memory.insert(entity, text_embedding=item['text_embedding'])
        else:
            memory.insert(entity)

    if use_optimal_context:
        # then replace the full memory with the optimal context
        memory = TextMemory()
        memory.insert(qa_instance['context'])


    return memory, outputs

def main(args):
    eval_run_start = time.perf_counter()

    use_milvus = False
    use_optimal_context = False
    if 'remembr' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = ReMEmbRAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True

    elif 'retrieve' in args.model:
        # Real Milvus retrieval like remembr+, but without the tool-calling loop
        # the planner cannot drive on this board: one plain query, one search,
        # then NonAgent's answer stage. Directly comparable to optimal+, which
        # shares that answer stage and differs only in getting oracle context.
        base_llm = args.model.split('+')[-1]
        agent = RetrieveAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True

    elif 'optimal' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = NonAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_optimal_context = True
    elif 'vlm' in args.model:
        agent = VLMNonAgent(llm_type='gpt-4o')

    else:
        agent = NonAgent(llm_type=args.model, num_ctx=args.num_ctx*4, temperature=args.temperature)


    data_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), args.qa_file+'.json')

    data = json.load(open(data_path, 'r'))
    data = data['data']


    running_successes = 0
    num_binary = 0

    running_pos_error = 0
    num_position = 0

    running_time_error = 0
    num_time = 0

    running_duration_error = 0
    num_duration = 0
    
    responses = []
    for i in tqdm.tqdm(range(0, len(data)), total=len(data), disable=trace.ENABLED):

        eval_latency_start = time.perf_counter()

        qa_instance = data[i]
        question = qa_instance['question']
        context = qa_instance['context']
        start_time = qa_instance['start_time']
        answers = qa_instance['answers']
        id = qa_instance['id']
        asked = [line.strip() for line in question.strip().splitlines() if line.strip()]
        trace.question(i + 1, len(data), qa_instance['type'], asked[-1] if asked else question)
        trace.plain(f"Evaluating {i} out of {len(data)}")

        if (qa_instance['type'] == 'text'):
            print("Skipping text questions for now")
            responses.append({
                'unscored': True,
                'skipped': True,
                'skip_reason': 'text_question',
                'eval_latency_s': time.perf_counter() - eval_latency_start,
                'question': question,
                'id': id,
            })
            continue

        memory, instance_captions = load_memory(args, data[i], use_milvus=use_milvus, use_optimal_context=use_optimal_context, ip_address=args.db_ip)
        if len(instance_captions) == 0: # ISSUE
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        trace.note(f"캡션 {len(instance_captions)}개 적재") if trace.ENABLED else print("HISTORY LENGTH", len(instance_captions))

        search_events = []
        instrument_memory_search(memory, search_events)

        # model.update_for_instance(captions=instance_captions, ref_time=start_time)
        agent.set_memory(memory)


        out_dict = answer_squad_question(agent, question, qa_instance, search_events=search_events)

        out_dict['eval_latency_s'] = time.perf_counter() - eval_latency_start

        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id


        error_dict = out_dict['error']

        # keep track of how many of each. usually all CSVs are one type only
        if qa_instance['type'] == 'position':
            num_position += 1
            if 'position_error' in error_dict:
                running_pos_error += error_dict['position_error']
        elif qa_instance['type'] == 'binary':
            num_binary += 1
            if 'binary_iscorrect' in error_dict:
                running_successes += error_dict['binary_iscorrect']
        elif qa_instance['type'] == 'time':
            num_time += 1
            if 'time_error' in error_dict:
                running_time_error += error_dict['time_error']
        elif qa_instance['type'] == 'duration':
            num_duration += 1
            if 'duration_error' in error_dict:
                running_duration_error += error_dict['duration_error']

        trace.plain("Question:", question)
        if 'response' in out_dict:
            trace.plain("Response:", out_dict['response'])
        trace.plain("Running Binary QA accuracy", running_successes/(num_binary+1))
        trace.plain("Running Spatial Error", running_pos_error/(num_position+1))
        trace.plain("Running Temporal Error", running_time_error/(num_time+1))
        trace.plain("Running Duration Error", running_duration_error/(num_duration+1))

        print()


        responses.append(out_dict)


    # save all_questions into json
    out_json = {
        "version": 0.1,
        "eval_total_latency_s": time.perf_counter() - eval_run_start,
        "responses": responses
    }

    # save the outputs
    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
    os.makedirs(out_path, exist_ok=True)

    # sanitize '/' (e.g. from 'tt/Llama-3.2-3B-Instruct') so it doesn't get
    # treated as a path separator and write into a non-existent subdirectory.
    name = (args.model+'__'+args.caption_file+args.postfix).replace('/', '_')
    with open(os.path.join(out_path, f'{name}.json'), 'w') as f:
        # to_save = json.dumps(out_json, indent=4)
        json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument("--model", type=str, default="remembr+llama3")
    parser.add_argument("--qa_file", type=str, default="human_qa")
    parser.add_argument("--caption_file", type=str, default="captions_VILA1.5-13b_3_secs")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--coda_dir", type=str, default="./coda_data/")

    parser.add_argument("--out_dir", type=str, default="./out/")

    parser.add_argument("--postfix", type=str, default='_0')


    # all model args
    # parser.add_argument("--use_gt_context", type=bool, default=False)


    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_ctx", type=int, default=8192*8)

    # remembr specific args
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--db_name", type=str, default='test')
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')


    args = parser.parse_args()
    main(args)
