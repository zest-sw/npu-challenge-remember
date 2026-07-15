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
import contextlib
import glob
import os
import re
import subprocess
import sys

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


@contextlib.contextmanager
def quiet(enabled: bool = True):
    """Silence ttnn's import/device chatter.

    The noise comes from two places: python-side loguru records ("DEBUG | ttnn:
    ... Initial ttnn.CONFIG") and C++ logs written straight to the process's
    stdout/stderr ("info | UMD | Initializing TLB mappings", "Metal | Profiler
    started"). tt-metal has no log-level env for these, so redirect the actual
    file descriptors — that catches the native writes too, which a Python-level
    redirect would miss. Only the ttnn import / device open+close are wrapped;
    our own printing happens outside.
    """
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def soc_descriptor_specs(arch: str) -> dict:
    """DRAM / L1 sizes from tt-metal's SoC descriptor.

    MeshDevice only exposes grid sizes (dram_grid_size / compute_with_storage_
    grid_size) — there is no DRAM-size method to query — but the authoritative
    numbers are in $TT_METAL_HOME/tt_metal/soc_descriptors/<arch>_*.yaml.
    """
    home = os.environ.get("TT_METAL_HOME", "/home/user/tt-metal")
    pat = os.path.join(home, "tt_metal", "soc_descriptors", f"{arch.lower()}*.yaml")
    files = sorted(glob.glob(pat))
    if not files:
        return {}
    text = open(files[0]).read()
    out = {"_file": files[0]}

    def num(key):
        # values sit on the line after the key, e.g. "dram_bank_size:\n  0xFFFFFFC0"
        m = re.search(rf"^{key}:\s*\n\s*\[?\s*(0x[0-9A-Fa-f]+|\d+)", text, re.M)
        return int(m.group(1), 0) if m else None

    out["dram_bank_size"] = num("dram_bank_size")
    out["worker_l1_size"] = num("worker_l1_size")
    out["eth_l1_size"] = num("eth_l1_size")
    out["dram_channels"] = len(re.findall(r"^\s*channel:\s*\d+", text, re.M)) or None
    return out


def ttnn_info(verbose: bool = False) -> None:
    hdr("ttnn / device spec")
    specs, err = {}, None
    try:
        with quiet(not verbose):
            import ttnn

            specs["device_ids"] = ttnn.get_device_ids()
            dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
            try:
                # MeshDevice exposes only these; DRAM/L1 sizes come from the
                # SoC descriptor below.
                specs["arch"] = dev.arch()
                specs["compute_grid"] = dev.compute_with_storage_grid_size()
                specs["dram_grid"] = dev.dram_grid_size()
                specs["core_grid"] = getattr(dev, "core_grid", None)
                specs["num_devices"] = dev.get_num_devices()
            finally:
                ttnn.close_mesh_device(dev)
    except Exception as e:  # noqa: BLE001
        err = e

    if err is not None:
        print(f"  unavailable: {type(err).__name__}: {err}")
        print("  (run in the tt-metal python_env with TT_METAL_HOME set, and the")
        print("   inference server stopped — the device is exclusive)")
        return

    kv("arch", specs["arch"])
    kv("devices", f'{specs["num_devices"]}  (ids={specs["device_ids"]})')
    kv("compute grid (Tensix)", specs["compute_grid"])
    kv("dram grid", specs["dram_grid"])

    arch_name = str(specs["arch"]).split(".")[-1]
    soc = soc_descriptor_specs(arch_name)
    if soc:
        bank, ch = soc.get("dram_bank_size"), soc.get("dram_channels")
        if bank:
            kv("DRAM / bank", f"{bank/2**30:.2f} GiB (0x{bank:X})")
        if bank and ch:
            kv("DRAM total", f"{bank*ch/2**30:.2f} GiB  ({ch} banks x {bank/2**30:.2f} GiB)")
        if soc.get("worker_l1_size"):
            kv("L1 / worker core", f"{soc['worker_l1_size']/1024:.0f} KB")
        if soc.get("eth_l1_size"):
            kv("L1 / eth core", f"{soc['eth_l1_size']/1024:.0f} KB")
        kv("(soc descriptor)", soc["_file"])


if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    print("NPU (BOS) info dump")
    host_info()
    free = availability()
    ttnn_info(verbose=verbose)
    if not free:
        print("\nNote: device was held -> the ttnn spec section may be partial. "
              "Stop the server and re-run for full specs.")
    if not verbose:
        print("\n(ttnn/UMD logs suppressed; pass -v to see them)")
        # ttnn tears the device down at interpreter exit, which prints one last
        # native line ("Closing user mode device drivers") after we're done. Keep
        # the fd redirect in place through exit so the report stays clean.
        sys.stdout.flush()
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
