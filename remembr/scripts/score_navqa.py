"""Score an eval.py result file the way the ReMEmbR paper scores NaVQA.

eval.py records a raw error per question (L2 metres for position, L1 minutes for
time/duration, a 0/1 for binary) but never aggregates them: its running printouts
divide by n+1, and it exits after dumping JSON. The paper instead reports

  - Descriptive Accuracy : binary + free-text questions judged correct
  - Positional Error (m) : mean L2
  - Temporal Error       : mean L1 over point-in-time AND duration questions
  - Overall Correctness  : every question type folded into one accuracy, using
                           "correct" = within 15 m / within 2 minutes

so this reads the result JSON back and computes those. Keeping it separate leaves
eval.py's own numbers untouched and lets any past result file be re-scored.

A note on units: the paper's table heads the temporal column "(s)", but its
threshold is "within 2 minutes", the captions are 3 s apart, and NaVQA stores its
answers in minutes ("-0.36 minutes ago" -> -0.36). A 0.3 s mean error is not
reachable at 3 s caption granularity, so the values are minutes and the header
looks like a slip. eval.py subtracts the stored values directly, so its errors are
in minutes and THRESH_TIME_MIN applies as-is.

Usage (remembr venv):

    python scripts/score_navqa.py --results out/0/human_qa/<file>.json --sequence_id 0
    python scripts/score_navqa.py --results ... --judge none   # skip the text judge
"""

import argparse
import json
import os
import sys

import numpy as np
import requests

THRESH_POS_M = 15.0     # "a spatial question ... is correct if it is within 15 meters of the goal"
THRESH_TIME_MIN = 2.0   # "a temporal question ... is correct if it is within 2 minutes of the goal"

JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {'correct': {'type': 'string', 'enum': ['yes', 'no']}},
    'required': ['correct'],
    'additionalProperties': False,
}

JUDGE_PROMPT = """You are grading a robot's answer to a question about what it saw.

Question: {question}
Reference answer: {reference}
Robot's answer: {predicted}

Does the robot's answer agree with the reference answer on the substance of the
question? Ignore wording, phrasing, and extra detail. Answer only "yes" or "no".
Reply as JSON: {{"correct": "yes"}} or {{"correct": "no"}}"""


def judge_text(question, reference, predicted, base_url, model):
    """Ask an LLM whether a free-text answer matches the reference."""
    if not predicted or not str(predicted).strip():
        return False
    if isinstance(reference, list):
        reference = reference[0] if reference else ''
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': JUDGE_PROMPT.format(
            question=question, reference=reference, predicted=predicted)}],
        'temperature': 0.0,
        'max_tokens': 32,
        'response_format': {'type': 'json_schema',
                            'json_schema': {'name': 'judgement', 'schema': JUDGE_SCHEMA}},
    }
    r = requests.post(f'{base_url}/chat/completions', json=body, timeout=300)
    r.raise_for_status()
    content = r.json()['choices'][0]['message']['content']
    return json.loads(content)['correct'] == 'yes'


def main(args):
    with open(args.results) as f:
        results = json.load(f)
    responses = results['responses']

    qa_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), args.qa_file + '.json')
    with open(qa_path) as f:
        data = json.load(f)['data']

    if len(responses) != len(data):
        print(f"warning: {len(responses)} responses vs {len(data)} questions -- "
              f"scoring the {min(len(responses), len(data))} that line up")

    pos_errors, time_errors = [], []
    descriptive = []       # (is_correct, kind) over binary + text
    correct_flags = []     # one bool per question -> Overall Correctness
    unanswered = 0
    judge_failed = 0

    for qa, resp in zip(data, responses):
        q_type = qa['type']
        # An empty dict is a question eval.py skipped; 'unscored' is one the model
        # never answered in a scoreable shape. Both are wrong answers, not absent
        # data -- counting them any other way would flatter the score.
        missing = (not resp) or ('unscored' in resp)
        err = (resp or {}).get('error') or {}

        if q_type == 'position':
            if 'position_error' in err:
                e = float(err['position_error'])
                pos_errors.append(e)
                correct_flags.append(e <= THRESH_POS_M)
            else:
                unanswered += 1
                correct_flags.append(False)

        elif q_type in ('time', 'duration'):
            key = f'{q_type}_error'
            if key in err:
                e = float(err[key])
                time_errors.append(e)
                correct_flags.append(e <= THRESH_TIME_MIN)
            else:
                unanswered += 1
                correct_flags.append(False)

        elif q_type == 'binary':
            if 'binary_iscorrect' in err:
                ok = bool(err['binary_iscorrect'])
            else:
                unanswered += 1
                ok = False
            descriptive.append((ok, 'binary'))
            correct_flags.append(ok)

        elif q_type == 'text':
            if missing or args.judge == 'none':
                if missing:
                    unanswered += 1
                ok = False
            else:
                try:
                    ok = judge_text(qa['question'], qa['answers'].get('text'),
                                    resp.get('text'), args.judge_url, args.judge)
                except Exception as e:
                    print(f"  judge failed ({type(e).__name__}: {e}) -- counting as incorrect")
                    judge_failed += 1
                    ok = False
            descriptive.append((ok, 'text'))
            correct_flags.append(ok)

    n = len(correct_flags)
    print(f"\n=== NaVQA, sequence {args.sequence_id} ({n} questions) ===")
    print(f"    {os.path.basename(args.results)}\n")

    if descriptive:
        acc = sum(ok for ok, _ in descriptive) / len(descriptive)
        n_bin = sum(1 for _, k in descriptive if k == 'binary')
        n_txt = len(descriptive) - n_bin
        note = "" if args.judge != 'none' else "  (text counted wrong: --judge none)"
        print(f"Descriptive Accuracy   {acc:.2f}   over {len(descriptive)} "
              f"({n_bin} binary + {n_txt} text){note}")
    if pos_errors:
        print(f"Positional Error       {np.mean(pos_errors):.2f} m   over {len(pos_errors)} answered")
    if time_errors:
        print(f"Temporal Error         {np.mean(time_errors):.2f} min over {len(time_errors)} answered")

    overall = sum(correct_flags) / n if n else 0.0
    print(f"\nOverall Correctness    {overall:.2f}   "
          f"({sum(correct_flags)}/{n} within {THRESH_POS_M:g} m / {THRESH_TIME_MIN:g} min)")

    if unanswered:
        print(f"\n{unanswered} of {n} questions had no usable answer; they count as incorrect in "
              f"Overall Correctness and are left out of the mean errors above (so those means "
              f"describe only the questions the model did answer).")
    if judge_failed:
        print(f"{judge_failed} text questions could not be judged and were counted incorrect.")

    if args.out:
        summary = {
            'results_file': args.results,
            'sequence_id': args.sequence_id,
            'n_questions': n,
            'descriptive_accuracy': (sum(ok for ok, _ in descriptive) / len(descriptive)) if descriptive else None,
            'positional_error_m': float(np.mean(pos_errors)) if pos_errors else None,
            'temporal_error_min': float(np.mean(time_errors)) if time_errors else None,
            'overall_correctness': overall,
            'n_unanswered': unanswered,
            'thresholds': {'position_m': THRESH_POS_M, 'time_min': THRESH_TIME_MIN},
        }
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nsummary -> {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='score_navqa',
        description="Score eval.py output with the ReMEmbR paper's NaVQA metrics.")
    parser.add_argument('--results', type=str, required=True, help='eval.py output JSON')
    parser.add_argument('--sequence_id', type=int, default=0)
    parser.add_argument('--qa_file', type=str, default='human_qa')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--out', type=str, default=None, help='write the summary to this JSON too')
    # The paper judges free text with an LLM. Ours is the local NPU model, which is
    # also the model being graded -- state that when reporting, or pass 'none'.
    parser.add_argument('--judge', type=str, default='tt/Llama-3.2-3B-Instruct',
                        help="model to judge text answers, or 'none' to count them incorrect")
    parser.add_argument('--judge_url', type=str,
                        default=os.environ.get('TT_LLM_BASE_URL', 'http://localhost:8100/v1'))
    args = parser.parse_args()
    if args.judge.startswith('tt/'):
        args.judge = args.judge[len('tt/'):]
    main(args)
