import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
from PIL import Image
import numpy as np
import sys
# added to get captioning time
import time

# load this directory
sys.path.append(sys.path[0] + '/..')
# NOTE: VILACaptioner is imported lazily inside build_captioner() so this script
# also runs in the (VILA-free) remembr_venv when using the Tenstorrent NPU
# captioner. Importing VILA at module load would hard-fail without the VILA stack.
from utils.util import get_frames
import pickle as pkl
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
from scipy.spatial.transform import Rotation
import shutil
import json

import tqdm


# A caption is an entry in a search index, not a reply to a person. The paper's prompt
# asks the VLM to "describe in detail" and "think step by step", and both Qwen2.5-VL-3B
# and Qwen3-VL-8B answer it as an assistant would: 271/272 captions opened with
# "Certainly! Let's break down the image description step by step:" and ran to a median
# 1182 characters of markdown lists (#50). Every caption then shared the same opening,
# which is dead weight in the embedding and dead tokens in the answer context.
#
# No worked example here on purpose. A one-shot example of a campus scene would name
# objects that really are in this dataset (a bench, a utility box, a drainage grate),
# and #27 measured this model copying example values verbatim into its output. So the
# shape is given as a skeleton and the content is left to the frames.
TERSE_QUERY = (
    "You are the camera on a robot moving through a university campus. "
    "Write what is in view as one entry in a search index.\n"
    "\n"
    "Shape: \"<where this is>. <what is there and where it sits relative to the rest>. "
    "<people and what they are doing>.\"\n"
    "\n"
    "Rules:\n"
    "- Two or three plain sentences. Under 60 words total.\n"
    "- Name concrete things: what they are, what they are made of, where they are.\n"
    "- Say how many people and what they are doing, or say there are none.\n"
    "- No greeting, no preamble, no markdown, no headings, no numbered or bulleted lists.\n"
    "- Do not write \"the image\", \"the photo\", \"appears to be\", \"likely\", "
    "\"step by step\", or anything about yourself.\n"
    "- Start with the place. First word is a noun."
)

CAPTION_STYLES = {'paper': None, 'terse': TERSE_QUERY}


def build_captioner(args):
    """Pick the captioner backend. Both implement Captioner.caption(images)->str,
    so the rest of the preprocessing pipeline is identical regardless of choice.

    - 'vila'        : original ReMEmbR captioner (needs the VILA/torch stack).
    - 'tenstorrent' : Qwen2.5-VL on the Tenstorrent NPU, via the local inference
                      server (talks HTTP; no VILA deps in-process).
    """
    if args.captioner == 'tenstorrent':
        from captioners.tenstorrent_captioner import TenstorrentCaptioner
        # Keep the paper's original --query verbatim, but neutralize the VILA
        # *video* wording: Qwen2.5-VL is given a still frame, so "<video>" / "the
        # few seconds of the video" make it refuse ("I can't see the video you're
        # referring to"). Only those phrases are rewritten -> everything else
        # ("Think step by step", the focus list, ...) stays faithful to the paper.
        # url/timeout come from TT_VLM_URL / TT_VLM_TIMEOUT env if set.
        prompt = (args.query or "")
        prompt = prompt.replace("<video>", "")
        prompt = prompt.replace("the few seconds of the video", "the image")
        prompt = prompt.replace("the video", "the image")
        prompt = prompt.strip() or None
        return TenstorrentCaptioner(prompt=prompt) if prompt else TenstorrentCaptioner()
    elif args.captioner == 'vila':
        from captioners.vila_captioner import VILACaptioner  # lazy: heavy deps
        return VILACaptioner(args)
    else:
        raise ValueError(f"Unknown --captioner {args.captioner!r} (vila|tenstorrent)")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_video_in_segs(args):

    SEQUENCE_ID=args.seq_id

    # load folders
    pkl_files = glob.glob(os.path.join(args.data_path, str(SEQUENCE_ID), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    times = [float(x.split('/')[-1][:-4]) for x in pkl_files]

    segments = []
    current_segment = []
    time_start = times[0]
    for t, file in zip(times, pkl_files):
        if t - time_start > args.seconds_per_caption:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    captioner_model = build_captioner(args)

    # captions_location = f'./data/{SEQUENCE_ID}/captions'
    captions_location = args.out_path
    out_file = os.path.join(
        captions_location,
        f'captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json',
    )
    # Skip only if the FINAL captions JSON already exists (a completed run).
    # The original code exited whenever the *directory* existed, which meant a
    # killed/interrupted run left an empty dir and permanently blocked re-runs.
    if os.path.exists(out_file):
        print(f"Captions already exist, nothing to do: {out_file}")
        exit()
    os.makedirs(captions_location, exist_ok=True)

    # --- incremental checkpoint / resume ---
    # Each finished caption is appended to this JSONL sidecar immediately, so an
    # interrupted run (kill, server hiccup) keeps everything done so far and can
    # resume instead of recomputing every caption. On full completion we write the
    # final JSON and delete the sidecar.
    partial_file = out_file + '.partial.jsonl'
    outputs = []
    done_ids = set()
    if os.path.exists(partial_file):
        with open(partial_file) as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                try:
                    ent = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a mid-write kill; drop it
                outputs.append(ent)
                done_ids.add(ent['id'])
        print(f"[resume] {len(done_ids)} captions loaded from checkpoint; continuing")
    partial_fh = open(partial_file, 'a')

    if args.max_segments:
        segments = segments[:args.max_segments]
        print(f"[probe] 앞 {len(segments)}개 구간만 캡션한다")

    for i, file_names in tqdm.tqdm(enumerate(segments), total=len(segments)):

        if file_names[0] in done_ids:
            continue  # already captioned in a previous run -> skip

        images = []
        # depth = []
        # bboxes = []
        position = []
        rotation = []
        timestamp = []


        for file in file_names:
            with open(file, 'rb') as f:
                data = pkl.load(f)
                data['cam0'] = data['cam0'][:, :, ::-1]

                images.append(PILImage.fromarray(data['cam0'].astype('uint8'), 'RGB'))
                # depth.append(data['stereo'])
                # bboxes.append(data['bbox_3d'])
                position.append(data['position'])
                rotation.append(data['rotation'])
                timestamp.append(data['timestamp'])

        
        position = np.array(position)
        rotation = np.array(rotation)
        rotation = Rotation.from_quat(rotation).as_euler('xyz', degrees=True)
        timestamp = np.array(timestamp)

        # let's sample the images down to args.num_video_frames
        images = images[::30//args.num_video_frames]

        # start time
        caption_start = time.perf_counter()

        out_text = captioner_model.caption(images)

        # captioning duration
        caption_latency_s = time.perf_counter() - caption_start

        print(out_text)
        filename_start = os.path.basename(file_names[0])
        filename_end = os.path.basename(file_names[1])


        text_embedding = embedder.embed_query(out_text)

        
        entity = {
            'id': file_names[0],
            'position': position.mean(axis=0),
            'theta': 3.14, # TEMPORARY: We are not using rotation information yet, so just leaving a placeholder
            'time': timestamp.mean(),
            'caption': out_text,
            'caption_latency_s': caption_latency_s,
            'file_start': filename_start,
            'file_end': filename_end,
            'text_embedding': text_embedding
        }

        outputs.append(entity)

        # checkpoint this caption immediately (durable against interruption)
        partial_fh.write(json.dumps(entity, cls=NumpyEncoder) + '\n')
        partial_fh.flush()
        os.fsync(partial_fh.fileno())



    partial_fh.close()

    # A probe is not a run. Writing the final JSON from 3 captions would make the top-of-
    # function "captions already exist" check skip the real run forever; leaving the
    # checkpoint in place instead lets the full run pick up the probe's captions.
    if args.max_segments:
        print(f"[probe] {len(outputs)}개 캡션을 체크포인트에 남겼다 -> {partial_file}")
        print("[probe] 최종 JSON 은 쓰지 않는다. 전체 실행이 여기서 이어받는다.")
        return

    # now save the outputs into a json (same path checked at the top)
    with open(out_file, 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder)

    # completed successfully -> drop the checkpoint sidecar
    if os.path.exists(partial_file):
        os.remove(partial_file)
    print(f"Wrote {len(outputs)} captions -> {out_file}")


if __name__ == "__main__":

    # default_query = "<video>\n You are a wandering around a university campus.\
    #     Please describe in detail what you see in the few seconds of the video. \
    #     Format it to focus on a few key elements. Provide details about each of these in the following format: \
    #     People: \n \
    #     Objects: \n \
    #     Environmental Features: \n \
    #     Activities/Events: \n \
    #     Other details: \n"
    

    default_query = "<video>\n You are a wandering around a university campus.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific."

    # default_query = "<video>\n What is the color of the floor?"

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-13b")
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions")
    parser.add_argument("--captioner_name", type=str, default="Llama-3-VILA1.5-8b")
    # which captioner backend to run: 'vila' (original) or 'tenstorrent' (NPU).
    parser.add_argument("--captioner", type=str, default="vila", choices=["vila", "tenstorrent"])

    parser.add_argument("--seconds_per_caption", type=int, default=3)
    # 'paper' keeps the original --query verbatim; 'terse' swaps in TERSE_QUERY (#50).
    parser.add_argument("--caption_style", type=str, default="paper",
                        choices=sorted(CAPTION_STYLES))
    # Caption only the first N segments and stop. For probing a prompt before
    # committing the board: 272 captions at 92 s each is about 7 hours.
    parser.add_argument("--max_segments", type=int, default=0)

    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()


    # add some rules here
    if 'Efficient-Large-Model/VILA1.5-40b' in args.model_path:
        args.conv_mode = 'hermes-2'
    elif 'Efficient-Large-Model/VILA1.5' in args.model_path:
        args.conv_mode = 'vicuna_v1'
    elif 'Llama' in args.model_path:
        args.conv_mode = 'llama_3'
    else:
        # trust the default conv_mode
        args.conv_mode = args.conv_mode

    styled = CAPTION_STYLES[args.caption_style]
    if styled is not None:
        args.query = styled

    run_video_in_segs(args)


# python -W ignore caption_segments.py --video-file "/home/aanwar/projects/memory_nav/foundation-nav/tools/isaac/data/102344094/path_vis/output.avi"     --query "<video>\n Please describe what you see in the few seconds of the video." 
