"""Set parked plan rows back to built, so the next run gates them again, optionally relaxing one
gate check on them with a reason (docs/adding-a-country.md 6).

    setbuilt.py <ids> [<check> "<reason>"] [--dry] [--tw <trainworld folder>]

    setbuilt.py abc,def drive_factor_median "delta town, roads wind round the channels"
    setbuilt.py @ids.txt top_zone_jobs_share "under 0.2M, one big employer" --dry

<ids> is a comma list, or @file (ids separated by commas or white space). A row is set to built only
if it is parked and its pack (data/packs/<id>.json and .bin) is on disk; its run reason is cleared.
With <check> and <reason> the row's `gate_relax[<check>]` is set to the reason (that check's FAIL then
counts as a WARN), whatever the row's status. Rows not in the plan are listed. The plan is written
under the `plan` lock, atomically. --dry prints the same lines and writes nothing (no lock taken).
Then rerun the rows: `world_run.py run --rows <ids> --no-join`.

    setbuilt.py <ids> --hold "<reason, with its task id>" [--dry]

(`--unhold` sets held rows back to planned) holds planned or parked rows instead (status "held", the reason in `hold_reason`): no run selects them by
--countries, nor the join; for rows whose data is wrong at the source (Cambodia's GHSL phantom towns, T-2000).
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402


def read_ids(text):
    if text.startswith("@"):
        with open(text[1:], encoding="utf8") as f:
            text = f.read()
    return [i for i in re.split(r"[,\s]+", text) if i]


def apply(doc, ids, relax, tw, say=print, hold=None, unhold=False):
    seen = set()
    changed = 0
    for r in doc["metros"]:
        if r.get("id") in ids and r.get("units") and unhold:
            seen.add(r["id"])
            will = r["status"] == "held"
            say(f"{r['id']} {r['status']}" + (" -> planned" if will else " (left: not held)"))
            if will:
                r["status"] = "planned"
                r.pop("hold_reason", None)
                changed += 1
        elif r.get("id") in ids and r.get("units") and hold:
            seen.add(r["id"])
            will = r["status"] in ("planned", "parked")
            say(f"{r['id']} {r['status']}" + (" -> held" if will else " (left: only planned or parked rows are held)"))
            if will:
                r["status"], r["hold_reason"] = "held", hold
                changed += 1
        elif r.get("id") in ids and r.get("units"):
            seen.add(r["id"])
            ok = all(os.path.exists(os.path.join(tw, "data", "packs", f"{r['id']}.{e}")) for e in ("json", "bin"))
            will = ok and r["status"] == "parked"
            say(f"{r['id']} {r['status']}" + (" -> built" if will else "" if ok else " (pack missing, left)")
                + (f"; relax {relax[0]}" if relax else ""))
            if relax:
                r.setdefault("gate_relax", {})[relax[0]] = relax[1]
                changed += 1
            if will:
                r["status"] = "built"
                (r.get("run") or {}).pop("reason", None)
                changed += 1
    missing = sorted(ids - seen)
    if missing:
        say("not in the plan (or no units): " + ",".join(missing))
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids")
    ap.add_argument("check", nargs="?", help="gate check to relax, e.g. drive_factor_median")
    ap.add_argument("reason", nargs="?", help="why, naming the ground")
    ap.add_argument("--dry", action="store_true", help="print what would change, write nothing")
    ap.add_argument("--hold", help="hold the rows instead, with this reason")
    ap.add_argument("--unhold", action="store_true", help="held rows back to planned (Anita lifted the hold)")
    ap.add_argument("--tw", default=TW, help="trainworld folder (default: this tree's)")
    a = ap.parse_args()
    if (a.check is None) != (a.reason is None):
        ap.error("give both <check> and <reason>, or neither")
    if (a.hold or a.unhold) and a.check:
        ap.error("--hold and --unhold take no check")
    tw = os.path.abspath(a.tw)
    ids = set(read_ids(a.ids))
    relax = [a.check, a.reason] if a.check else None
    plan = os.path.join(tw, "pipeline", "world_plan.json")
    if a.dry:
        apply(shared_lock.read_json(plan), ids, relax, tw, hold=a.hold, unhold=a.unhold)
        print("dry: plan not written")
        return
    with shared_lock.held("plan", "setbuilt.py", root=tw):
        doc = shared_lock.read_json(plan)
        apply(doc, ids, relax, tw, hold=a.hold, unhold=a.unhold)
        shared_lock.write_atomic(plan, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
