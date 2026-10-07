"""China: split the counties the Language Atlas itself splits between two dialect groups.

    python sources/cn_dlac.py      (needs data/normalized/cn_dialect.csv from sources/cn_dialect.py,
                                    and data/raw/cn/dlac/ACSN_MajorLanguages_pgn.zip)

Anita, 2026-10-06: Chinese dialect groups snapping at county borders were one of the map's most
visible false edges. sources/cn_dialect.py gives each county ONE group (dan-qqq's county table of
the Language Atlas of China, 2nd ed., 2012), so a county the atlas draws half Hakka, half Cantonese
(Dongguan, Bobai, Xinyi, Shenzhen's Longgang) is drawn wholly one or the other.

THE POLYGONS: Crissman, "Digital Language Atlas of China" (ACASIAN, Griffith University, 1995),
Harvard Dataverse doi:10.7910/DVN/OHYYXH, CC0. 307 polygons vectorised from the 1st edition of the
atlas (Longman, 1987), field CHINESE_GR: Mandarin, Jin, Wu, Hui, Gan, Xiang, Hakka, Yue, Min,
Pinghua, Shaozhou Tuhua, Danzhou, and three mixed classes (Hakka and Yue, Hakka and Min, Mandarin and
Pinghua). Min is not subdivided and Yue has no Sze Yap, so the polygons only say WHICH GROUP, and
the county table's subgroups are kept wherever they apply.

THE RULE: each 3 km grid cell (data/geo/cn/cn_grid_3km.gpkg, with population) takes the group of
the polygon its representative point falls in (a mixed class counts half for each). For a county
with one table row (share 1, basis direct or code history), with T its table group:
  - if the polygons put at least OWN_MIN of the county's people in T, and at least OTHER_MIN in
    another group O, the county's Chinese is split between T and every such O by those people
    (renormalised over the groups kept);
  - otherwise the county is left whole. Where the polygons disagree with the table outright (T
    under OWN_MIN) it is the 1987 and 2012 editions classifying differently (Guangxi's Yue against
    Pinghua; the Jin border), or digitising slop at the county line, not a split; the 2012 table
    wins.
  - EXCEPT a unit with no table row of its own (inherited(): carved out of a county after the table
    was compiled, e.g. Shenzhen's Pingshan and Guangming): the table never classified it, so the
    polygons decide, every group at or over OTHER_MIN kept (2026-10-06, session 5d7dac7e-mcp).
O's node is the table's own group and subgroup of the nearest county (population-weighted
centroids) that the table files under O, so Min in a Teochew-speaking corner is Teochew and Yue in
Taishan's neighbour is Sze Yap.
The county's total drawn as Chinese is unchanged (asserted); only its split between dialect groups
changes. countries/cn.py then places each group's dots on the cells the polygons give that group
(it repeats the cell join on the placement layer it is handed, with POLY and coarse() from here).

Output: data/normalized/cn_dialect_dlac.csv, cn_dialect.csv's columns with the split counties'
rows replaced (basis "atlas 1987 split", from_code the county that lent O's subgroup).

A QUIRK OF THE POLYGONS: where the atlas overlays a minority language on Chinese, the polygon often
carries only the minority (central Hunan's Xiang country is filed "Miao-Yao Languages", no Chinese
group). Such cells have no group and are left out of the shares, so they neither split a county
nor stop one splitting; in a split county they take any of its groups.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ZIP = HERE / "data" / "raw" / "cn" / "dlac" / "ACSN_MajorLanguages_pgn.zip"
GRID = HERE / "data" / "geo" / "cn" / "cn_grid_3km.gpkg"
DIALECT = HERE / "data" / "normalized" / "cn_dialect.csv"
OUT = HERE / "data" / "normalized" / "cn_dialect_dlac.csv"

OWN_MIN = 0.25
OTHER_MIN = 0.15

POLY = {   # DLAC CHINESE_GR -> coarse group(s); "Manderin" is the file's own spelling
    "Mandarin Supergroup": "Mandarin", "Manderin Supergroup": "Mandarin", "Jin Group": "Jin",
    "Wu Group": "Wu", "Hui Group": "Hui", "Gan Group": "Gan", "Xiang Group": "Xiang",
    "Hakka Group": "Hakka", "Yue (Cantonese) Group": "Yue", "Min Supergroup": "Min",
    "Pinghua Group": "Pinghua", "Shaozhou Tuhua": "Pinghua",
    "Danzhou Dialect (Unclassified)": "Yue",          # the table files 儋州话 under Yue too
    "Hakka and Yue (Cantonese) Grou": "Hakka+Yue", "Hakka and Min Groups": "Hakka+Min",
    "Mandarin Sg and Shaozhou Tuhua": "Mandarin+Pinghua", "Mandarin and Pinghua Groups": "Mandarin+Pinghua",
    "Mandarin Supergroup and Pinghu": "Mandarin+Pinghua",
}
MANDARIN = {"Southwestern", "Zhongyuan", "Northeastern", "Jilu", "Jianghuai", "Lanyin", "Jiaoliao", "Beijing"}


def coarse(group):
    """the table's group -> the polygons' coarse group"""
    if group in MANDARIN:
        return "Mandarin"
    return {"Pinghua and Tuhua": "Pinghua"}.get(group, group)


def cell_groups(geoms):
    """coarse group(s) of each polygon's representative point ("" outside every group polygon,
    "Hakka+Yue" for a mixed class), as a Series on geoms' index"""
    poly = gpd.read_file(f"zip://{ZIP}")
    poly["cg"] = poly["CHINESE_GR"].map(POLY)
    odd = sorted(set(poly.loc[poly["cg"].isna() & poly["CHINESE_GR"].notna(), "CHINESE_GR"]))
    if odd:
        raise SystemExit(f"DLAC groups with no mapping: {odd}")
    poly = poly[poly["cg"].notna()].to_crs(4326)
    pts = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms).representative_point()).set_crs(4326, allow_override=True)
    j = gpd.sjoin(pts, poly[["cg", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]          # overlapping slivers: first polygon
    return j["cg"].reindex(pts.index).fillna("")


def inherited(dia):
    """Units with no table row of their own: every row they have is a code that also reaches another
    unit (cn_dialect.py's CARVED districts, and districts that took part of a county the table lists
    whole, e.g. 钱塘区 from 江干区, 光明区 from 宝安区). Added 2026-10-06 (session 5d7dac7e-mcp)."""
    users = dia.groupby("from_code")["unit"].nunique()
    out = set()
    for u, g in dia[dia["basis"].isin(["direct", "code history"])].groupby("unit"):
        if (g["from_code"] != u).all() and (g["from_code"].map(users) > 1).all():
            out.add(u)
    return out


def main():
    grid = gpd.read_file(GRID).to_crs(4326)
    pts = gpd.GeoDataFrame({"unit": grid["unit"].astype(str), "pop": grid["pop"].astype(float)},
                           geometry=grid.geometry.representative_point(), crs=4326)
    cell_cg = cell_groups(grid.geometry)
    print(f"{len(pts):,} cells; {int((cell_cg == '').sum()):,} outside every Chinese-group polygon "
          f"({pts.loc[cell_cg == '', 'pop'].sum():,.0f} people)")

    # people per (county, coarse group); a mixed cell counts half for each
    rec = []
    for cg, sub in pts.assign(cg=cell_cg).groupby("cg"):
        if not cg:
            continue
        parts = cg.split("+")
        for p in parts:
            rec.append(sub.groupby("unit")["pop"].sum().div(len(parts)).rename(p))
    by = pd.concat(rec, axis=1).fillna(0).T.groupby(level=0).sum().T      # unit x group
    share = by.div(by.sum(1), axis=0)

    dia = pd.read_csv(DIALECT, dtype={"unit": str, "from_code": str})
    dia["cg"] = dia["group"].map(coarse)
    one = dia.groupby("unit").filter(lambda g: len(g) == 1)
    one = one[one["basis"].isin(["direct", "code history"])].set_index("unit")

    # population-weighted county centroids, for "the nearest county the table files under O"
    lon = pts.geometry.x * np.cos(np.radians(pts.geometry.y))
    cen = pd.DataFrame({"x": lon * pts["pop"], "y": pts.geometry.y * pts["pop"], "w": pts["pop"],
                        "unit": pts["unit"]}).groupby("unit").sum()
    cen = pd.DataFrame({"x": cen["x"] / cen["w"].clip(lower=1), "y": cen["y"] / cen["w"].clip(lower=1)})

    inh = inherited(dia)
    print(f"{len(inh)} units inherit a row from a county they were carved out of: polygons decide there")
    rows, n_split, moved = [], 0, 0.0
    for u, r in one.iterrows():
        if u not in share.index:
            continue
        s = share.loc[u]
        own = s.get(r["cg"], 0.0)
        others = s[(s.index != r["cg"]) & (s >= OTHER_MIN)]
        if u in inh:
            # the table never classified this unit, so the polygons are not overruled by it
            if others.empty:
                continue
            keep = s[s >= OTHER_MIN]
            print(f"  inherited {u}: table {r['cg']}, polygons {s[s > 0.005].round(2).to_dict()}")
        else:
            if own < OWN_MIN or others.empty:
                continue
            keep = pd.concat([pd.Series({r["cg"]: own}), others])
        keep = keep / keep.sum()
        n_split += 1
        moved += 1 - keep.get(r["cg"], 0.0)
        if r["cg"] in keep.index:
            rows.append(dict(unit=u, sgroup=r["sgroup"], group=r["group"], subgroup=r["subgroup"],
                             share=keep[r["cg"]], basis="atlas 1987 split", from_code=r["from_code"]))
        for o, sh in keep.drop(r["cg"], errors="ignore").items():
            cand = one[(one["cg"] == o) & (one.index != u)]
            cand = cand[cand.index.isin(cen.index)]
            d = np.hypot(cen.loc[cand.index, "x"] - cen.loc[u, "x"], cen.loc[cand.index, "y"] - cen.loc[u, "y"])
            near = d.idxmin()
            c = one.loc[near]
            rows.append(dict(unit=u, sgroup=c["sgroup"], group=c["group"], subgroup=c["subgroup"],
                             share=sh, basis="atlas 1987 split", from_code=near))
    split = pd.DataFrame(rows)
    out = pd.concat([dia[~dia["unit"].isin(set(split["unit"]))].drop(columns="cg"), split],
                    ignore_index=True).sort_values(["unit", "share"], ascending=[True, False])
    s = out.groupby("unit")["share"].sum()
    if (s.sub(1).abs() > 1e-9).any() or set(out["unit"]) != set(dia["unit"]):
        raise SystemExit("cn_dlac: shares do not cover every county once")
    out.to_csv(OUT, index=False)

    han = pd.read_csv(HERE / "data" / "normalized" / "cn.csv", dtype={"geo_id": str})
    han = han[han["source_category"] == "han"].groupby("geo_id")["count"].sum()
    sp = split.assign(cg=split["group"].map(coarse))
    first = sp.groupby("unit").head(1).set_index("unit")["cg"]
    sp["pair"] = sp["unit"].map(first) + " -> " + sp["cg"]
    sp = sp[sp["unit"].map(first) != sp["cg"]]
    sp["han"] = sp["share"] * sp["unit"].map(han).fillna(0)
    print(f"{n_split} counties split (of {len(one):,} single-group counties); Han moved to another "
          f"group: {sp['han'].sum():,.0f}")
    print(sp.groupby("pair").agg(counties=("unit", "nunique"), han=("han", "sum"))
          .sort_values("han", ascending=False).round(0).to_string())
    print(f"wrote {OUT.name} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
