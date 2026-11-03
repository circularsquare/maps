"""T-2111: the two files `admin_totals.CONFIGS["IN"]` reads, made from the downloads in
data/raw/admin_totals/IN/:

- DDW_PCA0000_2011_Indiastatedist.xlsx: Census 2011 Primary Census Abstract, India, states and districts
  (censusindia.gov.in/nada catalog 6191, download 9268; `curl -k`, the host's certificate chain is broken)
- 2011_Dist.shp/.shx/.dbf/.prj: Datameet's 2011 districts (github.com/datameet/maps, Districts/Census_2011,
  CC BY 4.0), whose `censuscode` is the PCA's district code

Datameet's code carries no state, and the reader takes a unit's parent from its code, so both files get the
code state (2 digits) + district (3): `in_districts_2011.gpkg` (code, name) and `in_pca_2011.csv` (code, pop;
"0" India, "SS" the states, "SSDDD" the districts). It stops if a polygon's state differs from its census row's.
Polygons are clipped to the lines India holds (world_plan.de_facto), as the plan's units are.

    python trainworld/tools/country/india_census.py
"""
import os
import sys

import pandas as pd
import pyogrio

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "raw", "admin_totals", "IN")


def main():
    g = pyogrio.read_dataframe(os.path.join(D, "2011_Dist.shp"))
    p = pd.read_excel(os.path.join(D, "DDW_PCA0000_2011_Indiastatedist.xlsx"), "Sheet1", dtype=str)
    p = p[p.TRU == "Total"].copy()
    dist = p[p.Level == "DISTRICT"].set_index(p[p.Level == "DISTRICT"].District.astype(int))
    st = g.ST_CEN_CD.astype(int)
    known = g.censuscode.isin(dist.index)
    bad = g[known & (st != dist.State.reindex(g.censuscode).astype(float).to_numpy())]
    if len(bad):
        raise SystemExit(f"polygons whose state differs from the census row's: {bad[['DISTRICT', 'censuscode']].to_dict('records')}")
    g["code"] = [f"{s:02d}{c:03d}" for s, c in zip(st, g.censuscode)]
    g["name"] = g.DISTRICT.str.strip()
    # Datameet draws J&K on India's claimed lines; a district cut by the Line of Control would count GHSL's
    # people across it in its state's total and its own (Punch 1.01M against a census of 0.48M)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
    import world_plan as wp
    clipped = [wp.de_facto("IND", wp.to_moll(x)) for x in g.geometry]
    g["geometry"] = [wp.to_ll(c) if c is not None and not c.is_empty else None for c in clipped]
    lost = g.geometry.isna()
    print(f"clipped to the de facto lines; wholly across them: {list(g.name[lost])}")
    g = g[~lost]
    out = os.path.join(D, "in_districts_2011.gpkg")
    pyogrio.write_dataframe(g[["code", "name", "geometry"]], out + ".part.gpkg", layer="districts")
    os.replace(out + ".part.gpkg", out)
    code = p.apply(lambda r: "0" if r.Level == "India" else r.State if r.Level == "STATE" else r.State + r.District, axis=1)
    t = pd.DataFrame({"code": code, "name": p.Name.str.strip(), "pop": p.TOT_P.astype(int)})
    t.to_csv(os.path.join(D, "in_pca_2011.csv"), index=False)
    print(f"{len(g)} polygons ({(~known).sum()} without a census row), {len(t)} census rows, India {t['pop'][t.code == '0'].iloc[0]:,}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
