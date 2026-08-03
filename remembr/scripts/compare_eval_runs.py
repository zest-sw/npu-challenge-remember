"""Compare two eval.py result JSONs question-by-question.

Built to measure one change at a time (e.g. the space/time calculator on vs off)
on the SAME question set, so it pairs responses by question id rather than
reporting two independent averages — an average alone hides that a change helped
some questions and hurt others.

Usage:
    python remembr/scripts/compare_eval_runs.py BEFORE.json AFTER.json
    python remembr/scripts/compare_eval_runs.py BEFORE.json AFTER.json --per-question
"""
import argparse
import json
import os


ERROR_KEYS = {
    "position": "position_error",
    "time": "time_error",
    "duration": "duration_error",
}


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    by_id = {}
    for item in payload.get("responses", []):
        by_id[item.get("id")] = item
    return payload, by_id


def question_type(item):
    """Infer the scored type from which error key the run produced."""
    error = item.get("error") or {}
    for qtype, key in ERROR_KEYS.items():
        if key in error:
            return qtype
    if "binary_iscorrect" in error:
        return "binary"
    if "answer" in error:
        return "text"
    return "unanswered"


def summarize(by_id, ids):
    """Mean error per type + unanswered/attempt counts over the given ids."""
    totals = {}
    for qtype in list(ERROR_KEYS) + ["binary", "text", "unanswered"]:
        totals[qtype] = {"n": 0, "sum": 0.0}
    unanswered = 0
    attempts = 0
    latency = 0.0

    for qid in ids:
        item = by_id.get(qid)
        if item is None:
            continue
        error = item.get("error") or {}
        attempts += item.get("attempts") or 0
        latency += item.get("query_latency_s") or 0.0
        if error.get("unanswered"):
            unanswered += 1
            totals["unanswered"]["n"] += 1
            continue
        for qtype, key in ERROR_KEYS.items():
            if key in error:
                totals[qtype]["n"] += 1
                totals[qtype]["sum"] += float(error[key])
        if "binary_iscorrect" in error:
            totals["binary"]["n"] += 1
            totals["binary"]["sum"] += float(error["binary_iscorrect"])
        if "answer" in error:
            totals["text"]["n"] += 1
    return {"totals": totals, "unanswered": unanswered, "attempts": attempts, "latency": latency}


def mean(entry):
    return (entry["sum"] / entry["n"]) if entry["n"] else None


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def arrow(before, after, lower_is_better=True):
    """Direction of change, guarding against a missing side."""
    if before is None or after is None:
        return ""
    if abs(after - before) < 1e-9:
        return "="
    improved = (after < before) if lower_is_better else (after > before)
    return "improved" if improved else "WORSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--per-question", action="store_true",
                        help="list every question whose scored value changed")
    args = parser.parse_args()

    before_payload, before = load(args.before)
    after_payload, after = load(args.after)

    shared = [qid for qid in before if qid in after]
    only_before = [qid for qid in before if qid not in after]
    only_after = [qid for qid in after if qid not in before]

    print(f"BEFORE : {os.path.basename(args.before)}  ({len(before)} questions)")
    print(f"AFTER  : {os.path.basename(args.after)}  ({len(after)} questions)")
    print(f"paired : {len(shared)}")
    if only_before or only_after:
        # Never average over a different denominator without saying so.
        print(f"  ! unpaired -> only in before: {len(only_before)}, only in after: {len(only_after)}")
    print()

    b = summarize(before, shared)
    a = summarize(after, shared)

    print(f"{'metric':<22} {'before':>10} {'after':>10}   direction")
    print("-" * 60)
    rows = [
        ("position error (m)", "position", True),
        ("time error (min)", "time", True),
        ("duration error (min)", "duration", True),
    ]
    for label, key, lower in rows:
        mb, ma = mean(b["totals"][key]), mean(a["totals"][key])
        n = f"n={a['totals'][key]['n']}"
        print(f"{label:<22} {fmt(mb):>10} {fmt(ma):>10}   {arrow(mb, ma, lower):<9} {n}")

    mb, ma = mean(b["totals"]["binary"]), mean(a["totals"]["binary"])
    print(f"{'binary accuracy':<22} {fmt(mb):>10} {fmt(ma):>10}   "
          f"{arrow(mb, ma, lower_is_better=False):<9} n={a['totals']['binary']['n']}")

    print(f"{'unanswered':<22} {b['unanswered']:>10} {a['unanswered']:>10}   "
          f"{arrow(b['unanswered'], a['unanswered']):<9}")
    print(f"{'total attempts':<22} {b['attempts']:>10} {a['attempts']:>10}   "
          f"{arrow(b['attempts'], a['attempts']):<9}")
    print(f"{'total latency (s)':<22} {fmt(b['latency'], 1):>10} {fmt(a['latency'], 1):>10}")

    # Per-question movement: an average can hide wins cancelling losses.
    moved = []
    for qid in shared:
        eb = (before[qid].get("error") or {})
        ea = (after[qid].get("error") or {})
        for qtype, key in ERROR_KEYS.items():
            if key in eb or key in ea:
                vb = eb.get(key)
                va = ea.get(key)
                if vb != va:
                    moved.append((qid, key, vb, va))
        if bool(eb.get("unanswered")) != bool(ea.get("unanswered")):
            moved.append((qid, "unanswered", eb.get("unanswered", False), ea.get("unanswered", False)))

    better = sum(1 for _, k, vb, va in moved
                 if k != "unanswered" and vb is not None and va is not None and va < vb)
    worse = sum(1 for _, k, vb, va in moved
                if k != "unanswered" and vb is not None and va is not None and va > vb)
    print()
    print(f"changed questions: {len(moved)}  (better {better}, worse {worse})")

    if args.per_question and moved:
        print()
        for qid, key, vb, va in moved:
            question = (after.get(qid) or before.get(qid) or {}).get("question", "")
            asked = question.strip().splitlines()[-1].strip() if question else ""
            print(f"  [{key}] id={qid}  {vb} -> {va}")
            if asked:
                print(f"        {asked[:100]}")


if __name__ == "__main__":
    main()
