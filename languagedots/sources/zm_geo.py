"""Zambia placement layer: religiondots' Kontur hexes for the 156 constituencies, each
constituency cut into a town part and a countryside part -> data/geo/zm/zm_hexes.gpkg.

    python sources/zm_geo.py          (needs data/normalized/zm.csv from sources/zm_census.py)

WHY. The census prints every constituency's nine language groups for its rural and its urban
people separately (Tables C2.3 and C2.4), and the languages inside a group by province, rural and
urban (C1). Urban and rural mixes differ a lot: English is 97% urban, Nyanja 85% of its speakers
in towns, Chewa 73% rural. Spread over a whole constituency, Kasama's town English would be
scattered over its villages. So each constituency is two units, `<id>-U` and `<id>-R`.

HOW. religiondots' layer (sources/zm_grid.py there: Kontur 2023-11 400 m hexes keyed to COD-AB
constituencies) is read, never written. Inside each constituency the hexes are ranked by Kontur
population (all hexes are one size, so this is density), and the densest are labelled urban
until they hold the census's urban share of the constituency's people; the hex that crosses the
share goes to the town too. A constituency the census counts as all rural has no `-U` unit; one
counted all urban (the Lusaka and Copperbelt city constituencies) has no `-R` unit.

This decides only where inside a constituency each residence's dots go, never how many.

CHECKS: every (constituency, residence) with people in zm.csv has hexes; the achieved urban
share of Kontur population per constituency, printed against the census share; the ids match
religiondots' 156.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

SRC = RD_GEO / "zm" / "zm_hexes.gpkg"
NORM = HERE / "data" / "normalized" / "zm.csv"
OUT = HERE / "data" / "geo" / "zm" / "zm_hexes.gpkg"


def main():
    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    hx = pyogrio.read_dataframe(SRC)
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    res = df.groupby(["geo_id", "residence"])["count"].sum().unstack(fill_value=0)
    res["share_u"] = res["U"] / (res["U"] + res["R"])
    report(set(hx["unit"]) == set(res.index) and len(res) == 156,
           f"religiondots' {hx['unit'].nunique()} constituencies are zm.csv's {len(res)}")

    hx["con"] = hx["unit"].astype(str)
    hx = hx.sort_values(["con", "pop"], ascending=[True, False], kind="mergesort")
    lab = np.empty(len(hx), dtype=object)
    rows = []
    for con, idx in hx.groupby("con", sort=False).indices.items():
        p = hx["pop"].to_numpy()[idx]
        share = res.at[con, "share_u"]
        if share <= 0:
            k = 0
        elif share >= 1:
            k = len(idx)
        else:
            cum = np.cumsum(p) / p.sum()
            k = int(np.searchsorted(cum, share) + 1)
            k = min(max(k, 1), len(idx) - 1)          # keep at least one hex for each side
        lab[idx[:k]] = "U"
        lab[idx[k:]] = "R"
        rows.append((con, share, p[:k].sum() / p.sum() if p.sum() else 0, k, len(idx)))
    hx["unit"] = hx["con"] + "-" + lab
    hx = hx.sort_index()

    chk = pd.DataFrame(rows, columns=["con", "census_u", "kontur_u", "hexes_u", "hexes"])
    need = {f"{c}-{r}" for c in res.index for r in "UR" if res.at[c, r] > 0}
    have = set(hx["unit"])
    report(need <= have, f"every constituency x residence with people has hexes "
                         f"({len(need)} needed, {len(need - have)} missing {sorted(need - have)[:4]})")
    d = (chk["kontur_u"] - chk["census_u"]).abs()
    report(d.max() < 0.05,
           f"town share of Kontur people against the census's urban share: median gap "
           f"{d.median():.3f}, largest {d.max():.3f} ({chk.loc[d.idxmax(), 'con']})")
    mixed = chk[(chk["census_u"] > 0) & (chk["census_u"] < 1)]
    print(f"     {len(mixed)} constituencies split, {int((chk['census_u'] == 0).sum())} all rural, "
          f"{int((chk['census_u'] >= 1).sum())} all urban; {int((lab == 'U').sum()):,} town hexes "
          f"of {len(hx):,}")
    if not ok:
        raise SystemExit("FAILED; nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = hx[["unit", "pop", "geometry"]]
    if OUT.exists():
        OUT.unlink()
    pyogrio.write_dataframe(out, OUT, layer="hexes")
    print(f"wrote {OUT}: {len(out):,} hexes, {out['unit'].nunique()} units")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
