#!/usr/bin/env python3
"""Dump NPU (BOS) hardware / spec info.

Two kinds of info:
  - HOST side (device node, kernel module, PCIe, hugepages) -> always works.
  - ttnn/device spec (arch, Tensix grid, DRAM, L1, clock) -> needs to OPEN the
    device, which is EXCLUSIVE. Stop the inference server first (Ctrl-C) and run
    this in the tt-metal python_env.

Usage:
    source /home/user/tt-metal/env_set.sh        # activates ttnn's python_env
    python /home/user/remembr/tt_server/npu_info.py

The ttnn section is written defensively: device introspection method names vary
between tt-metal/bos-metal versions, so each field tries several candidates and
prints "(n/a)" instead of crashing when none exist.
"""
import os
import subprocess

DEV_NODE = os.environ.get("BOS_DEV_NODE", "/dev/bos/0")


def sh(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(err: {e})"


def hdr(title: str) -> None:
    print(f"\n=== {title} ===")


def kv(label: str, value) -> None:
    print(f"  {label:28}: {value}")


# ---------------------------------------------------------------------------
# 1) Host / driver (no device open required)
# ---------------------------------------------------------------------------
def host_info() -> None:
    hdr("Host / driver")
    kv("device node", sh(f"ls -l {DEV_NODE} 2>/dev/null") or "(missing)")
    kv("kernel module", sh("lsmod | grep -w bos") or "(bos not loaded)")
    kv("module version", sh("modinfo bos 2>/dev/null | awk -F': *' '/^version/{print $2}'") or "(n/a)")
    kv("host RAM", sh("awk '/MemTotal/{printf \"%.1f GB\", $2/1048576}' /proc/meminfo"))
    kv("hugepages(1G)", sh("ls /dev/hugepages-1G/ 2>/dev/null | tr '\\n' ' '") or "(none)")
    kv("CPU cores", sh("nproc"))

    # PCIe device the bos driver is bound to. The symlink NAMES in the driver
    # dir ARE the device BDFs (using their targets would match the parent bridge).
    bdf = sh("ls /sys/bus/pci/drivers/bos/ 2>/dev/null | "
             "grep -oE '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\\.[0-9a-f]$' | head -1")
    if bdf and "err" not in bdf:
        kv("PCIe BDF", bdf)
        kv("PCIe link", sh(f"lspci -vv -s {bdf} 2>/dev/null | grep -iE 'LnkSta:' | head -1") or "(need root for -vv)")
        kv("PCIe device", sh(f"lspci -s {bdf} 2>/dev/null") or "(n/a)")
    else:
        kv("PCIe BDF", "(bos driver sysfs not found)")


# ---------------------------------------------------------------------------
# 2) Device availability
# ---------------------------------------------------------------------------
def availability() -> bool:
    hdr("Device availability")
    usage = sh("lsmod | awk '/^bos /{print $3}'")
    kv("bos usage count", usage or "?")
    holder = sh(f"fuser -v {DEV_NODE} 2>&1")
    kv("holder (fuser)", holder or "(free)")
    srv = sh("pgrep -af '[t]t_infer_server'")
    free = (usage in ("0", "")) and not srv
    if srv:
        print("  ⚠️  inference server is running and holds the device:")
        print("     ", srv)
        print("     -> stop it (Ctrl-C) before the ttnn spec section can open the device.")
    return free


# ---------------------------------------------------------------------------
# 3) ttnn / device spec (needs the device free)
# ---------------------------------------------------------------------------
def _first(dev, names):
    """Call the first existing no-arg method/attr from `names`; raise if none."""
    for n in names:
        if hasattr(dev, n):
            attr = getattr(dev, n)
            return attr() if callable(attr) else attr
    raise AttributeError(f"none of {names}")


def _probe(dev, label, names, post=lambda x: x):
    try:
        print(f"  {label:28}: {post(_first(dev, names))}")
    except Exception as e:  # noqa: BLE001
        print(f"  {label:28}: (n/a: {type(e).__name__})")


def ttnn_info() -> None:
    hdr("ttnn / device spec")
    try:
        import ttnn
    except Exception as e:  # noqa: BLE001
        print("  ttnn import failed -> run inside the tt-metal python_env")
        print("  (source /home/user/tt-metal/env_set.sh):", e)
        return

    # device count (module-level, no open)
    for nm in ("get_device_ids", "GetNumAvailableDevices", "get_num_devices"):
        if hasattr(ttnn, nm):
            try:
                kv(f"ttnn.{nm}", getattr(ttnn, nm)())
            except Exception as e:  # noqa: BLE001
                kv(f"ttnn.{nm}", f"(n/a: {type(e).__name__})")
            break

    try:
        dev = ttnn.open_device(device_id=int(os.environ.get("BOS_DEVICE_ID", "0")))
    except Exception as e:  # noqa: BLE001
        print("  open_device failed (device busy/held by server, or wedged):", e)
        return

    try:
        _probe(dev, "arch", ["arch", "get_arch"])
        _probe(dev, "device id", ["id", "get_id"])
        _probe(dev, "compute grid (Tensix)",
               ["compute_with_storage_grid_size", "logical_grid_size", "grid_size"])
        _probe(dev, "dram channels", ["num_dram_channels", "get_num_dram_channels"])
        _probe(dev, "dram grid", ["dram_grid_size"])
        _probe(dev, "dram / channel",
               ["dram_size_per_channel"], post=lambda b: f"{b/1e9:.2f} GB ({b} B)")
        _probe(dev, "L1 / core",
               ["l1_size_per_core", "get_l1_size_per_core"], post=lambda b: f"{b/1024:.0f} KB")
        _probe(dev, "worker grid", ["worker_grid_size"])
        _probe(dev, "num program cache", ["num_program_cache_entries"])

        # total DRAM = channels * size/channel, if both available
        try:
            ch = _first(dev, ["num_dram_channels", "get_num_dram_channels"])
            per = _first(dev, ["dram_size_per_channel"])
            kv("total DRAM", f"{ch * per / 1e9:.2f} GB ({ch} x {per/1e9:.2f} GB)")
        except Exception:  # noqa: BLE001
            pass

        # allocator / free memory snapshot, if the version exposes it
        for nm in ("dump_device_memory_state", "get_memory_view"):
            if hasattr(ttnn, nm):
                print(f"  --- ttnn.{nm} ---")
                try:
                    getattr(ttnn, nm)(dev)
                except Exception as e:  # noqa: BLE001
                    print(f"      (n/a: {type(e).__name__})")
                break
    finally:
        try:
            ttnn.close_device(dev)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    print("NPU (BOS) info dump")
    host_info()
    free = availability()
    ttnn_info()
    if not free:
        print("\nNote: device was held -> the ttnn spec section may be partial. "
              "Stop the server and re-run for full specs.")
