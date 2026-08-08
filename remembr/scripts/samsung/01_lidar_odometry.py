"""Stage 1: LiDAR odometry over the Samsung rosbag -> per-scan poses.

Reads /velodyne_points from a ROS2 sqlite bag (no ROS install; rosbags + a Humble
typestore), runs KISS-ICP scan-to-map registration, and writes each scan's
timestamp + 4x4 world pose. The bag has no GPS/odometry, so this IS the position
source. Poses are relative to the first scan (0,0,0), which is what the ReMEmbR
eval scores against.

Needs rosbags + kiss-icp, which are NOT in the repo venv. Use the dedicated venv:
    /home/user/ycc/samsung_dataset/.venv/bin/python

Usage:
    .../.venv/bin/python 01_lidar_odometry.py /home/user/ycc/samsung_demo3 poses_demo3.npz

Then stage 2 (02_build_coda.py) turns those poses into CODa-format .pkl frames.
"""
import sys
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import load_config

LIDAR_TOPIC = "/velodyne_points"


def scan_xyz_time(msg):
    """(Nx3 xyz float64, per-point time float64) from a PointCloud2, invalid points dropped.

    Velodyne layout verified on this bag: x,y,z float32 at byte 0/4/8, per-point
    'time' float32 at byte 18, point_step 22.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    xyz = buf[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    tim = buf[:, 18:22].copy().view(np.float32).reshape(-1)
    good = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.1)
    return xyz[good].astype(np.float64), tim[good].astype(np.float64)


def main(bag_dir, out_path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    kiss = KissICP(load_config(None))

    stamps, poses = [], []
    with AnyReader([Path(bag_dir)], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
        total = sum(c.msgcount for c in conns)
        for i, (conn, t_ns, raw) in enumerate(reader.messages(connections=conns)):
            msg = reader.deserialize(raw, conn.msgtype)
            xyz, tim = scan_xyz_time(msg)
            # normalize per-point time to [0,1] for motion deskew
            if tim.size and tim.max() > tim.min():
                tim = (tim - tim.min()) / (tim.max() - tim.min())
            kiss.register_frame(xyz, tim)
            # header stamp is the scan's own clock; use it to sync images later
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            poses.append(kiss.last_pose.copy())
            if i % 200 == 0:
                print(f"  {i}/{total}")

    stamps = np.array(stamps)
    poses = np.array(poses)  # (N,4,4)
    np.savez(out_path, stamps=stamps, poses=poses)

    xy = poses[:, :2, 3]
    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    print(f"\n{len(poses)} scans -> {out_path}")
    print(f"  duration    {stamps[-1]-stamps[0]:.1f}s")
    print(f"  path length {dist:.1f}m")
    print(f"  x {xy[:,0].min():.1f}..{xy[:,0].max():.1f}m  y {xy[:,1].min():.1f}..{xy[:,1].max():.1f}m  "
          f"z {poses[:,2,3].min():.2f}..{poses[:,2,3].max():.2f}m")
    if np.abs(xy).max() > 1000:
        print("  WARNING: trajectory diverged (>1km) -- registration likely failed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "poses.npz")
