"""T-3506: markdown tables of cases2.json, by group, with the call per id (default by group)."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
cases = json.load(open(os.path.join(HERE, "cases2.json"), encoding="utf8"))
CALL = {
    "bug": "shift (queued, T-3499)", "arv": "merge (queued, T-3488)",
    "slr": "shift", "dir": "shift", "per": "shift", "roc": "shift (with per)", "lal": "shift",
    "suc": "merge", "yhu": "merge",
}
STALE = {"GT", "BZ", "SV", "HN", "NI", "CR", "PA", "VE"}


def pop(c):
    p = c["pop"] or c["plan_people"] or 0
    return f"{p / 1e6:.2f}M" if p >= 1e6 else f"{p / 1e3:,.0f}k"


for g in ("inland", "edge", "coast", "foreign"):
    rows = [c for c in cases if c["group"] == g]
    print(f"\n### {g} ({len(rows)})\n")
    print("| id | zone | country | people | inside | call |")
    print("|---|---|---|---|---|---|")
    for c in rows:
        call = CALL.get(c["id"], "keep")
        if c["country"] in STALE:
            call += " (mid-build, recheck)"
        print(f"| {c['id']} | {c['name']} | {c['country']} | {pop(c)} | {c['by']} {c['by_name']} | {call} |")

