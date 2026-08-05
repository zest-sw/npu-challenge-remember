"""How close is decode to this board's memory-bandwidth ceiling?

Why this exists
---------------
At batch=1 an autoregressive decode step reads every weight from DRAM once, so the
step time has a hard floor:

    t_token >= weight_bytes / memory_bandwidth

For 8B at bfloat8_b that is ~8 GB per token. Measured decode on this board is
~250 ms/token, which implies an *effective* ~32 GB/s. That number alone does not
say whether decode is optimal or terrible -- it depends on the peak the memory
system can actually deliver:

    utilisation = 32 GB/s / peak

If peak is ~35 GB/s, decode is already at the ceiling and kernel work is pointless;
the only remaining levers are shrinking the weights or breaking the one-sweep-per-
token rule (issue #39). If peak is ~200 GB/s, decode is at 16% and there is a lot
left on the table. `npu_info.py` reports grid sizes but no bandwidth, so measure it.

Method: time a large elementwise add over DRAM-resident tensors. Each iteration
moves 3 x tensor_bytes (two reads, one write), which is pure streaming with no
reuse -- the same access pattern a decode weight sweep has. Sizes are swept so the
trend is visible; a single size can sit in a cache-friendly corner and mislead.

    # the inference server must be STOPPED -- the device is exclusive
    source $TT_METAL_HOME/python_env/bin/activate
    python tt_server/tests/board/bench_dram_bandwidth.py

Board safety: opens the device once, closes it in a finally, and does nothing else.
Start small -- if a size hangs, everything after it is unreachable, so the sweep
prints each result as it lands rather than at the end.
"""
import argparse
import time

# MiB per tensor. Small first: a hang on a big size must not hide the small results.
DEFAULT_SIZES = (16, 64, 256, 512)
WARMUP = 2
ITERS = 10


def bench_one(ttnn, device, mib, iters, warmup):
    """Effective GB/s for an elementwise add over two DRAM tensors."""
    import torch

    elements = mib * 1024 * 1024 // 2  # bfloat16
    side = 1
    while side * side < elements:
        side *= 2
    shape = (1, 1, side, elements // side)

    host = torch.ones(shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        for _ in range(warmup):
            ttnn.deallocate(ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.synchronize_device(device)

        start = time.perf_counter()
        for _ in range(iters):
            ttnn.deallocate(ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
    finally:
        ttnn.deallocate(a)
        ttnn.deallocate(b)

    tensor_bytes = a.volume() * 2 if hasattr(a, "volume") else mib * 1024 * 1024
    moved = 3 * tensor_bytes * iters  # two reads + one write per iteration
    return moved / elapsed / 1e9, elapsed / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="MiB per tensor, ascending")
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--weight_gb", type=float, default=8.0,
                        help="weight bytes swept per decode token (8B at bfloat8_b ~ 8)")
    parser.add_argument("--decode_ms", type=float, default=250.0,
                        help="measured ms per decoded token, for the utilisation line")
    args = parser.parse_args()

    import ttnn

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    best = 0.0
    try:
        print(f"{'MiB/tensor':>11s} {'GB moved/iter':>14s} {'ms/iter':>9s} {'GB/s':>8s}")
        for mib in sorted(args.sizes):
            gbps, per_iter = bench_one(ttnn, device, mib, args.iters, WARMUP)
            best = max(best, gbps)
            print(f"{mib:>11d} {3*mib/1024:>14.2f} {per_iter*1e3:>9.2f} {gbps:>8.1f}", flush=True)
    finally:
        ttnn.close_mesh_device(device)

    decode_gbps = args.weight_gb / (args.decode_ms / 1e3)
    print(f"\npeak measured here      : {best:.1f} GB/s")
    print(f"decode implies          : {decode_gbps:.1f} GB/s "
          f"({args.weight_gb:.1f} GB / {args.decode_ms:.0f} ms per token)")
    if best > 0:
        print(f"decode utilisation      : {100*decode_gbps/best:.0f}% of measured peak")
        print(f"floor at measured peak  : {1e3*args.weight_gb/best:.0f} ms/token "
              f"({best/args.weight_gb:.1f} tok/s)")
    print("\nHigh utilisation means kernel work is pointless and the levers are a smaller\n"
          "weight footprint or breaking one-sweep-per-token (issue #39). Low utilisation\n"
          "means there is room in the decode path itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
