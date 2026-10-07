"""Check one country end to end before scattering it.

    python tools/check_country.py <cc>

Stops (exit 1) on: an entry that does not load, a node not in taxonomy/languages.json (run
taxonomy/build.py), a counted unit with no polygon in the placement layer, a negative or missing
count. Prints the total, the unit count and the ten largest languages, so a reconciliation
against the source's own national table is one glance.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    cc = sys.argv[1].lower()
    from countries import load_one
    try:
        e = load_one(cc)
    except Exception as ex:  # noqa: BLE001
        raise SystemExit(f"FAIL countries/{cc}.py: {ex}")
    df = e["counts"]()
    bad = []
    if not {"unit", "node", "count"} <= set(df.columns):
        raise SystemExit(f"FAIL counts() columns {list(df.columns)}; need unit, node, count")
    if df["count"].isna().any() or (df["count"] < 0).any():
        bad.append("missing or negative counts")
    nodes = {n["id"] for n in json.loads((ROOT / "taxonomy" / "languages.json").read_text(encoding="utf-8"))["nodes"]}
    unknown = sorted(set(df["node"].dropna()) - nodes)
    if unknown:
        bad.append(f"nodes not in languages.json (run taxonomy/build.py?): {unknown[:10]}")
    if df["node"].isna().any():
        bad.append(f"{int(df['node'].isna().sum())} rows with no node: unmapped labels")

    import pyogrio
    place = pyogrio.read_dataframe(e["place"], read_geometry=False)
    units = set(e["place_unit"](place))
    missing = sorted(set(df["unit"].astype(str)) - units)
    if missing:
        lost = df[df["unit"].astype(str).isin(missing)]["count"].sum()
        bad.append(f"{len(missing)} counted units have no polygon ({lost:,.0f} people): {missing[:6]}")

    tot = df["count"].sum()
    print(f"{e['name']}: {tot:,.0f} people, {df['unit'].nunique():,} units, {df['node'].nunique()} languages; "
          f"placement layer {len(place):,} polygons over {len(units):,} units")
    top = df.groupby("node")["count"].sum().sort_values(ascending=False)
    for n, v in top.head(10).items():
        print(f"  {v:>14,.0f}  {100 * v / tot:5.1f}%  {n}")
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
