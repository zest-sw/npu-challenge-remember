"""ROS-free CODa -> .pkl extractor.

The original scripts/preprocess_coda.py needs a full ROS stack (rospy, cv_bridge,
tf2_ros, sensor_msgs) — but all of that is only used for RViz visualization /
publishing. The actual data written into each per-frame .pkl (the RGB image,
robot position, orientation, timestamp) comes from plain files:

  - poses/dense/{seq}.txt          rows of [ts, x, y, z, qw, qx, qy, qz]
  - 2d_rect/cam0/{seq}/2d_rect_cam0_{seq}_{frame}.png   rectified RGB frames

so we can produce the exact same .pkl files that scripts/preprocess_captions.py
consumes, with no ROS at all.

Output: {out_dir}/{timestamp}.pkl each holding
  {'cam0': HxWx3 uint8 (BGR, as cv2.imread returns), 'position': [x,y,z],
   'rotation': [qx,qy,qz,qw] (scipy Rotation.from_quat order), 'timestamp': float}

Usage:
  python scripts/preprocess_coda_rosfree.py \
      --seq_id 0 \
      --coda_root /home/user/coda_raw/0     (dir containing 2d_rect/, poses/, ...) \
      --out_dir  ./coda_data/0 \
      --frame_skip 1
"""
import os
import argparse
import glob
import pickle as pkl

import numpy as np
import cv2
from tqdm import tqdm


def main(args):
    seq = str(args.seq_id)

    pose_file = os.path.join(args.coda_root, "poses", "dense", f"{seq}.txt")
    assert os.path.exists(pose_file), f"pose file not found: {pose_file}"
    # rows: ts, x, y, z, qw, qx, qy, qz
    pose_np = np.fromfile(pose_file, sep=" ").reshape(-1, 8)

    cam0_dir = os.path.join(args.coda_root, "2d_rect", "cam0", seq)
    assert os.path.isdir(cam0_dir), f"cam0 dir not found: {cam0_dir}"

    # Map frame index -> cam0 png path (frame parsed from the filename suffix).
    cam0_by_frame = {}
    for p in glob.glob(os.path.join(cam0_dir, f"2d_rect_cam0_{seq}_*.png")):
        frame = int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0])
        cam0_by_frame[frame] = p

    os.makedirs(args.out_dir, exist_ok=True)

    n_written = 0
    n_missing = 0
    # In CODa the dense pose file has one row per LiDAR frame, and 2d_rect/cam0
    # provides one rectified image per frame, so pose row index == frame index.
    for frame in tqdm(range(0, len(pose_np), args.frame_skip), desc=f"seq {seq}"):
        img_path = cam0_by_frame.get(frame)
        if img_path is None:
            n_missing += 1
            continue

        ts, x, y, z, qw, qx, qy, qz = pose_np[frame]
        # Same as the original preprocess_coda.py: cv2.imread returns BGR, and
        # preprocess_captions.py flips it back to RGB before captioning.
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            n_missing += 1
            continue

        entity = {
            "cam0": img,
            "position": np.array([x, y, z], dtype=float),
            # scipy Rotation.from_quat expects [x, y, z, w]
            "rotation": np.array([qx, qy, qz, qw], dtype=float),
            "timestamp": float(ts),
        }

        out_path = os.path.join(args.out_dir, f"{ts:.6f}.pkl")
        with open(out_path, "wb") as f:
            pkl.dump(entity, f)
        n_written += 1

    print(f"[done] seq {seq}: wrote {n_written} .pkl, skipped {n_missing} "
          f"(poses={len(pose_np)}, cam0_frames={len(cam0_by_frame)}) -> {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--coda_root", type=str, required=True,
                        help="Dir containing 2d_rect/, poses/, calibrations/ for this sequence")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output dir for {timestamp}.pkl (e.g. ./coda_data/0)")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Keep every Nth frame (1 = all; 10Hz data)")
    main(parser.parse_args())
