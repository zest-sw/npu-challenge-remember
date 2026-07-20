import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
from PIL import Image
import numpy as np
import sys

# load this directory
sys.path.append(sys.path[0] + '/..')
# NOTE: VILACaptioner is imported lazily inside build_captioner() so this script
# also runs in the (VILA-free) remembr_venv when using the Tenstorrent NPU
# captioner. Importing VILA at module load would hard-fail without the VILA stack.
from captioners.captioner import (BALANCED_CAPTION_PROMPT, DENSE_CAPTION_PROMPT,
                                  clean_balanced_caption, clean_dense_caption)
from utils.util import get_frames
import pickle as pkl
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
from scipy.spatial.transform import Rotation
import shutil
import json

import tqdm


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
        # (With --caption_style dense the query is DENSE_CAPTION_PROMPT, which has
        # no video wording, so these replacements are no-ops.)
        # url/timeout come from TT_VLM_URL / TT_VLM_TIMEOUT env if set.
        prompt = (args.query or "")
        prompt = prompt.replace("<video>", "")
        prompt = prompt.replace("the few seconds of the video", "the image")
        prompt = prompt.replace("the video", "the image")
        prompt = prompt.strip() or None
        # --temperature is honored just like the VILA path (paper default 0.2).
        # Note: the original tt captions run predates this and decoded greedily.
        if prompt:
            return TenstorrentCaptioner(prompt=prompt, temperature=args.temperature)
        return TenstorrentCaptioner(temperature=args.temperature)
    elif args.captioner == 'vila':
        from captioners.vila_captioner import VILACaptioner  # lazy: heavy deps
        return VILACaptioner(args)
    else:
        raise ValueError(f"Unknown --captioner {args.captioner!r} (vila|tenstorrent)")


def caption_segment(captioner_model, images, args, max_attempts=3):
    """Caption one segment, cleaning dense output and retrying degenerate results.

    On a frame it cannot read, the 3B sometimes recites the prompt's format example
    verbatim and nothing else ("outdoor gravel path, orange cone on curb, low wall
    behind parked bus"). clean_dense_caption strips those phrases, which correctly
    leaves nothing — but an empty caption embeds to a meaningless vector and drops
    that place out of memory entirely. It is a sampling accident, not a property of
    the frame, so retry at a higher temperature; the last raw output is kept as a
    floor so a segment is never recorded blank.
    """
    if args.caption_style == 'balanced':
        # Prose, so no example phrases to copy and no repetition collapse to retry --
        # only the fixed announcing clause needs removing.
        return clean_balanced_caption(captioner_model.caption(images))
    if args.caption_style != 'dense':
        return captioner_model.caption(images)

    base_temp = getattr(captioner_model, 'temperature', None)
    raw = ''
    try:
        for attempt in range(max_attempts):
            if base_temp is not None:
                captioner_model.temperature = base_temp + 0.3 * attempt
            raw = captioner_model.caption(images)
            cleaned = clean_dense_caption(raw)
            if cleaned:
                return cleaned
            print(f"[dense] attempt {attempt+1}/{max_attempts} produced only example "
                  f"copies; retrying at higher temperature")
    finally:
        if base_temp is not None:
            captioner_model.temperature = base_temp
    return raw.strip()


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

        # Sample the segment down to args.num_video_frames evenly spaced frames.
        # The original stride form (images[::30//n]) assumes every segment holds
        # exactly 30 frames; CODa segments hold 30 OR 31, so 40 of the 272 seq0
        # segments yielded n+1 images instead of n. That silently inflated the
        # request: with the dense prompt the extra ~66 vision tokens pushed the
        # encoding past the VLM's 256-token prefill shape and the server 500'd on
        # segment 55. Slicing to a count makes --num-video-frames mean what it says
        # and keeps every request the same size.
        if len(images) > args.num_video_frames:
            take = np.linspace(0, len(images) - 1, args.num_video_frames, dtype=int)
            images = [images[i] for i in take]

        out_text = caption_segment(captioner_model, images, args)

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

    # 'paper'    keeps the original ReMEmbR query verbatim (fidelity baseline).
    # 'dense'    comma-separated observation phrases. Best embedding space, but starves
    #            _select_memory of the words it needs -- measured worse end to end.
    # 'balanced' prose minus the preamble and the speculation: keeps the vocabulary the
    #            selector runs on while dropping the boilerplate that dominated the
    #            paper captions' vectors. See captioners/captioner.py for the numbers.
    parser.add_argument("--caption_style", type=str, default="paper",
                        choices=["paper", "dense", "balanced"])

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

    # Suffix the captions filename with the style so a run can never collide with (or be
    # mistaken for) the paper-style captions existing results came from.
    if args.caption_style in ('dense', 'balanced'):
        args.query = (DENSE_CAPTION_PROMPT if args.caption_style == 'dense'
                      else BALANCED_CAPTION_PROMPT)
        if args.caption_style not in args.captioner_name:
            args.captioner_name += '-' + args.caption_style

    run_video_in_segs(args)


# python -W ignore caption_segments.py --video-file "/home/aanwar/projects/memory_nav/foundation-nav/tools/isaac/data/102344094/path_vis/output.avi"     --query "<video>\n Please describe what you see in the few seconds of the video." 
