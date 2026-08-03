"""Stage 2: pair each LiDAR pose with the nearest camera image -> CODa .pkl set.

Poses (from 01_lidar_odometry.py) are 10 Hz; color images are 30 Hz. For each pose
we take the image closest in time (<=~16 ms away), so no pose interpolation is
needed and the output lands at CODa's own 10 Hz. Each frame is written as a
{timestamp}.pkl with the exact keys ReMEmbR's preprocess/eval expect:

    {'cam0': HxWx3 uint8 BGR, 'position': [x,y,z] float64,
     'rotation': [qx,qy,qz,qw] float64 (scipy order), 'timestamp': float}

Usage:
    python 02_build_coda.py /home/user/ycc/samsung_demo3 poses_demo3.npz coda_data/0
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

COLOR_TOPIC = "/camera/camera/color/image_raw/compressed"


def load_images(bag_dir):
    """(timestamps sorted, list of raw JPEG bytes aligned to them)."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    stamps, blobs = [], []
    with AnyReader([Path(bag_dir)], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == COLOR_TOPIC]
        for conn, _t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            blobs.append(bytes(m.data))
    order = np.argsort(stamps)
    return np.array(stamps)[order], [blobs[i] for i in order]


def main(bag_dir, poses_path, out_dir):
    d = np.load(poses_path)
    pstamps, poses = d["stamps"], d["poses"]  # (N,), (N,4,4)
    img_stamps, img_blobs = load_images(bag_dir)
    print(f"{len(poses)} poses, {len(img_stamps)} images")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    max_gap = 0.0
    for pose_t, T in zip(pstamps, poses):
        j = int(np.argmin(np.abs(img_stamps - pose_t)))
        gap = abs(img_stamps[j] - pose_t)
        max_gap = max(max_gap, gap)
        img = cv2.imdecode(np.frombuffer(img_blobs[j], np.uint8), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            continue
        stamp = float(img_stamps[j])
        entry = {
            "cam0": img,
            "position": T[:3, 3].astype(np.float64),
            "rotation": Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float64),  # xyzw
            "timestamp": stamp,
        }
        with open(out / f"{stamp:.6f}.pkl", "wb") as f:
            pickle.dump(entry, f)
        written += 1
        if written % 200 == 0:
            print(f"  {written}/{len(poses)}")

    print(f"\n{written} frames -> {out}/")
    print(f"  max pose<->image time gap: {max_gap*1000:.0f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
