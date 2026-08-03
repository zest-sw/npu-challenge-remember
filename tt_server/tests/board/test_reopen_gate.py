"""Gate test: does repeated in-process close_mesh_device + open_mesh_device wedge
the board? This is THE decision point for the single-process mode-switch design.

Mirrors the server's open params exactly (tt_infer_server._open_device):
  open_mesh_device(mesh_shape, trace_region_size=30M, num_command_queues=2)

Each cycle: open -> tiny op (forces FW/cores active) -> synchronize -> close.
If reopen re-runs FW init reliably, all N cycles pass. If the board wedges, a
cycle raises "failed to initialize FW" and we stop with the cycle number.

Run: python test_reopen_gate.py [N]
"""
import os
import sys
import time
import ttnn


def open_device():
    mapping = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}
    param = mapping.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    device_ids = ttnn.get_device_ids()
    if isinstance(param, tuple):
        mesh_shape = ttnn.MeshShape(*param)
    else:
        mesh_shape = ttnn.MeshShape(1, min(param, len(device_ids)))
    trace_region_size = int(os.environ.get("TT_TRACE_REGION_SIZE", str(30_000_000)))
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, trace_region_size=trace_region_size,
                                 num_command_queues=2)


def tiny_op(mesh_device):
    import torch
    t = ttnn.from_torch(torch.randn(64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    r = ttnn.add(t, t)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(r)
    ttnn.deallocate(t)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"=== reopen gate: {n} cycles of open -> op -> synchronize -> close ===", flush=True)
    for i in range(1, n + 1):
        t0 = time.time()
        try:
            md = open_device()
            tiny_op(md)
            ttnn.synchronize_device(md)
            ttnn.close_mesh_device(md)
            print(f"  cycle {i:>3}/{n}: OK  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  cycle {i:>3}/{n}: FAIL after {time.time()-t0:.1f}s -- {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
            print("REOPEN GATE FAILED -- single-process close+reopen is NOT safe on this board.",
                  flush=True)
            sys.exit(1)
    print(f"REOPEN GATE PASSED -- {n} in-process close+reopen cycles, no wedge.", flush=True)


if __name__ == "__main__":
    main()
