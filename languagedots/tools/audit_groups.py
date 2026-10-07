"""Who is drawn on a GROUP node (one with children), and which source labels put them there.

    python tools/audit_groups.py                 ranked table, every drawn country
    python tools/audit_groups.py --min 50000     only rows over 50k people or 2% of the country
    python tools/audit_groups.py --cc in,af      only those countries

The viewer draws a group node washed out as "X, not named", so a NAMED census label must never
resolve to one (AGENT_BRIEF §3; India's Rajasthani, 26M, 2026-10). An unnamed remainder ("Others
under X", "Other Bantu") belongs there. This lists both, for a person to tell apart.

People per node come from data/processed/counts.json (dots x dot_value, plus rings), so they are
the last scatter's, rounded to the dot. Labels come from each mapping module's NAMES / CODES dict
(and resolve() for labels the dict lacks), with each label's people summed from the country's
data/normalized/<cc>*.csv files where they carry `source_category` or `source_code` (the largest
geo_level's sum, so a file with state and district rows is not double counted). A label with no
normalized count is still listed, with "?".
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "taxonomy"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def label_counts(cc):
    """{label or code: people} from data/normalized/<cc>*.csv, best effort."""
    import pandas as pd
    out = {}
    for p in sorted((ROOT / "data" / "normalized").glob(f"{cc}*.csv")):
        if p.stem != cc and not p.stem.startswith(cc + "_"):
            continue
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if "count" not in df.columns:
            continue
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
        lvl = "geo_level" if "geo_level" in df.columns else None
        for col in ("source_category", "source_code"):
            if col not in df.columns:
                continue
            if lvl:
                g = df.groupby([lvl, col])["count"].sum().reset_index()
                best = g.groupby(col)["count"].max()
            else:
                best = df.groupby(col)["count"].sum()
            for k, v in best.items():
                out[(p.stem, str(k))] = max(out.get((p.stem, str(k)), 0), float(v))
        # code -> label, so a CODES mapping can print words
        if {"source_code", "source_category"} <= set(df.columns):
            for c, lab in df[["source_code", "source_category"]].drop_duplicates().values:
                out.setdefault(("__label__", str(c)), lab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0)
    ap.add_argument("--share", type=float, default=0.02)
    ap.add_argument("--cc", default="")
    ap.add_argument("--top", type=int, default=0)
    a = ap.parse_args()

    nodes = json.loads((ROOT / "taxonomy" / "languages.json").read_text(encoding="utf-8"))["nodes"]
    label = {n["id"]: n["label"] for n in nodes}
    groups = {n["parent"] for n in nodes if n["parent"]}
    cj = json.loads((ROOT / "data" / "processed" / "counts.json").read_text(encoding="utf-8"))
    dv = cj["dot_value"]
    only = set(a.cc.split(",")) - {""}

    from countries import load_one
    from regroup import move
    rows = []
    for cc, c in sorted(cj["countries"].items()):
        if only and cc not in only:
            continue
        people = {k: v * dv for k, v in c.get("dots", {}).items()}
        for k, v in (c.get("rings") or {}).items():
            people[k] = people.get(k, 0) + v
        total = sum(people.values()) or 1
        hits = {k: v for k, v in people.items() if k in groups}
        if not hits:
            continue
        try:
            mappings = load_one(cc)["mappings"]
        except Exception:  # noqa: BLE001
            mappings = []
        lc = None
        by_node = {}
        for m in mappings:
            try:
                mod = importlib.import_module(m)
            except Exception:  # noqa: BLE001
                continue
            for attr in ("NAMES", "CODES"):
                d = getattr(mod, attr, None)
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    v = move(v)       # written id -> drawn id (taxonomy/regroup.py)
                    if isinstance(v, str) and v in hits:
                        by_node.setdefault(v, []).append((m, k))
        for node, v in hits.items():
            if v < a.min and v / total < a.share:
                continue
            labs = []
            for m, k in by_node.get(node, []):
                if lc is None:
                    lc = label_counts(cc)
                n = sum(val for (stem, kk), val in lc.items() if stem != "__label__" and kk == str(k))
                words = lc.get(("__label__", str(k)))
                labs.append((n, f"{k}" + (f" [{words}]" if words and words != k else "")))
            labs.sort(key=lambda t: -t[0])
            rows.append((v, cc, node, v / total, labs))
    rows.sort(key=lambda r: -r[0])
    if a.top:
        rows = rows[:a.top]
    for v, cc, node, sh, labs in rows:
        print(f"{v:>12,.0f} {100 * sh:5.1f}%  {cc}  {node}  ({label.get(node, '?')})")
        for n, k in labs[:12]:
            print(f"{'':22}{n:>12,.0f}  {k}" if n else f"{'':22}{'?':>12}  {k}")
        if len(labs) > 12:
            print(f"{'':22}... {len(labs) - 12} more labels")
        if not labs:
            print(f"{'':22}(no NAMES/CODES label maps here: resolve() or counts() puts them on it)")


if __name__ == "__main__":
    main()
