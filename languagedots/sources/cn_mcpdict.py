"""China: dialect groups the county table misses, from MCPDict's located dialect points.

    python sources/cn_mcpdict.py   (needs data/normalized/cn_dialect_dlac.csv from sources/cn_dlac.py,
                                    data/normalized/cn.csv, and data/raw/cn/mcpdict/info.geojson)

Anita, 2026-10-06: soften the sharp Hakka / Cantonese / Pinghua / Min / Xiang edges in western and
northern Guangdong and Guangxi, where whole counties flip from one group to another ("not too sure
how the mcpdict blend would work really but if you think you can make it work lets do it").

THE POINTS: 汉字音典 (MCPDict, github.com/osfans/MCPDict, MIT licence, copy in data/raw/cn/mcpdict/
LICENSE), tools/info.geojson: 3,147 documented dialect points, each at a township or village, each
filed under its Language Atlas of China 2nd ed. (2012) group (地圖集二分區). They say WHERE a group is
spoken, never how many speak it, and linguists record rare varieties far more often than common ones
(Guangxi Pinghua has about as many points as Yue). So:

(a) COUNTS, only in SCOPE (the southern provinces where non-Mandarin groups meet). In a county, a
    point whose node is not among the county's groups (cn_dialect_dlac.csv) is evidence that some
    of its people speak that node. Each such node gets
        CAP * (1 - exp(-E_edge / E0)) + ISLAND_CAP * (1 - exp(-E_pocket / E0))
    of the county's Chinese, all added nodes together at most CAP, taken from the county's own groups
    in proportion. An EDGE point is one whose node is the atlas group of a neighbouring county and
    that MCPDict does not mark 方言島: the group's area runs over the county line, which is the case
    Anita asked about. Anything else is a POCKET (a dialect island, a migrant village far from its
    group), worth at most ISLAND_CAP; Mandarin pockets, mostly garrison 军话, ISLAND_CAP_MANDARIN.
    E is points weighted per GROUP, not per point: each node's points are weighted by its people per
    point across SCOPE against the median node, clipped to [W_MIN, W_MAX] (so only the oversampled
    groups shrink: a Pinghua point counts 0.25, a Yue point 1). The county's own points are ignored:
    the atlas already says its own group is there.
(b) HELD TOWARDS PUBLISHED PROVINCE TOTALS, Guangdong and Guangxi (PUB): each coarse group the
    gazetteer counts may move only towards its published share of the province's Chinese (target =
    its total after (a), clipped between today's total and the published share); groups it does not
    count keep what (a) gave them. The province's county x node table is then raked (IPF) to its
    county totals and those targets. A full rake of GUANGXI to the gazetteer's shares (FULL below)
    was tried and reverted at Anita's request; it is off (GX_FULL_RAKE = False; sources/cn.md §12).
Each county's Chinese total is unchanged (asserted in countries/cn.py).

PLACEMENT (Placer, used by countries/cn.py): in every county with more than one local group
(nationally, not only SCOPE: this moves no counts), each group's dots lean towards its points.
For cell c and node g, seed = base + K * exp(-d^2 / 2 SIGMA^2), d the distance to g's nearest point
(any county's); base is 1 for the county's atlas groups (in a county sources/cn_dlac.py split, 1
inside the 1987 polygons of that group or of none, OFF outside) and OFF for a node only the points
gave it. The county's cell x node matrix is raked from seed x pop to the cells' population and the
nodes' shares, so the cells near a Hakka point fill with Hakka first and every cell still holds its
population. A node's weights are its column.

Outputs: data/normalized/cn_dialect_mcp.csv (cn_dialect_dlac.csv's columns; an added row has basis
"MCPDict points" and from_code the county lending the subgroup) and data/normalized/cn_mcp_points.csv
(every point in a county: unit, node, coarse group, island, lon, lat).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "taxonomy"))
sys.path.insert(0, str(HERE / "sources"))
import cn2020  # noqa: E402
from cn_dlac import coarse  # noqa: E402

GEOJSON = HERE / "data" / "raw" / "cn" / "mcpdict" / "info.geojson"
COUNTIES = HERE.parent / "chinaethnicity" / "data" / "geo" / "counties.gpkg"
DLAC = HERE / "data" / "normalized" / "cn_dialect_dlac.csv"
OUT = HERE / "data" / "normalized" / "cn_dialect_mcp.csv"
POINTS_OUT = HERE / "data" / "normalized" / "cn_mcp_points.csv"

SCOPE = {"44": "Guangdong", "45": "Guangxi", "35": "Fujian", "36": "Jiangxi", "43": "Hunan",
         "33": "Zhejiang"}
CAP = 0.25          # at most this share of a county's Chinese goes to groups only the points show
E0 = 1.0            # weighted points at which the share reaches 63% of CAP (1 - 1/e)
ISLAND_CAP = 0.12   # 方言島 points (outside the group's main area): a pocket, at most this share
ISLAND_CAP_MANDARIN = 0.03   # Mandarin islands in the south are mostly 军话, a few garrison villages
W_MIN, W_MAX = 0.25, 1.0  # only shrink the oversampled groups; never inflate a big group's point

# Published speaker figures (sources/cn_dialect_sources.md lead 4), used as SHARES of their sum, so
# their vintage (1990s, 2000) and the "+" lower bounds drop out:
# Guangxi, 《广西通志·汉语方言志》 (1998) via chinanews 2011-04-12: Yue 12M+, Hakka 5.6M+, Southwestern
#   Mandarin 5M+, Pinghua ~4M, Xiang ~1.5M, Min ~0.25M.
# Guangdong, 《广东省志·方言志》 (2004) via secondary summaries: speakers Yue nearly 40M, Hakka about
#   15M, Min about 17M; Shaozhou Tuhua about 0.8M (Zhuang Chusheng).
PUB = {"45": ("Guangxi", {"Yue": 12.0, "Hakka": 5.6, "Mandarin": 5.0, "Pinghua": 4.0, "Xiang": 1.5,
                          "Min": 0.25}),
       "44": ("Guangdong", {"Yue": 40.0, "Hakka": 15.0, "Min": 17.0, "Pinghua": 0.8})}

# GUANGXI, RAKED FULLY to the gazetteer's shares (Anita, 2026-10-06: "we can push guangxi if you think
# gazetteer is somewhat trustworthy"; sources/cn.md §12). Guangxi takes no combined CAP in (a), and in
# place of (b)'s partway hold its county x node table is raked to its county totals and the published
# shares. The rake keeps zeros at zero, so the extra Hakka, Pinghua and Min need somewhere to go first:
#   - SEEDS from the 1987 atlas polygons (sources/cn_dlac.py's cell_groups on the 3 km grid): a group's
#     share of a county's people inside its polygons, times GAMMA (GAMMA_CORE in the city-core
#     districts, where the polygons paint the villages' speech over the city; GAMMA_WEAK for Pinghua
#     or Min polygons in a county that neither the gazetteer's list below nor an MCPDict point names);
#   - FLOORS from the gazetteer's own county lists (the 2011 Guangxi Daily summary of it): Hakka in
#     every county but 全州, 兴安, 资源 and 凤山; Pinghua in its named 桂南 and 桂北 counties; Min Nan in
#     its ten named counties;
#   - the counties' own table groups and MCPDict points, as (a) left them.
# Xiang is not raked to its share: the gazetteer puts all of it in four counties (全州, 灌阳, 资源,
# 兴安) that have lost people since, so its target is those four at their polygon share (or what they
# have, if more) plus Xiang elsewhere as (a) left it; the other groups share the rest by the gazetteer.
FULL = "45"
GX_FULL_RAKE = False   # tried 2026-10-06 and reverted at Anita's request (sources/cn.md §12); Guangxi keeps (b)
GAMMA = 0.5
GAMMA_WEAK = 0.15  # Pinghua or Min polygons in a county neither the gazetteer's list nor a point names
GAMMA_CORE = 0.1
CORE = {"450102", "450103", "450105", "450107", "450108", "450109",      # Nanning's city districts
        "450202", "450203", "450204", "450205",                          # Liuzhou's
        "450302", "450303", "450304", "450305", "450311"}                # Guilin's
FLOOR = {
    "Hakka": (0.02, None),                       # None: every county but HAKKA_NONE
    "Pinghua": (0.05, {
        "450102", "450103", "450105", "450107", "450108", "450109", "450110",   # 南宁郊区 (now its districts)
        "450126", "450127", "450125", "450124", "450722", "451421", "451422", "451423",  # 宾阳 横县 上林 马山 浦北 扶绥 宁明 龙州
        "451002", "451082", "451022", "451003",                                 # 百色郊区 (右江) 平果 田东 田阳
        "450206", "450222", "450225", "450224", "450226", "451225",             # 柳江 柳城 融水 融安 三江 罗城
        "450302", "450303", "450304", "450305", "450311",                       # 桂林郊区 (now its districts)
        "450312", "450323", "450326", "450328",                                 # 临桂 灵川 永福 龙胜
        "451102", "451103", "451123", "451122"}),                               # 八步 (and 平桂, carved 2013) 富川 钟山
    "Min": (0.01, {"450821", "450981", "450702", "450703", "450330", "450332", "451102", "451122",
                   "450206", "451202", "450621"}),   # 平南 北流 钦州 (钦南 钦北) 平乐 恭城 八步 钟山 柳江 金城江 上思
}
HAKKA_NONE = {"450324", "450325", "450329", "451223"}       # 全州 兴安 资源 凤山
XIANG_COUNTIES = {"450324", "450327", "450329", "450325"}   # 全州 灌阳 资源 兴安
SEED_ROW = {"Hakka": ("Non-Mandarin", "Hakka", ""), "Pinghua": ("Non-Mandarin", "Pinghua and Tuhua", ""),
            "Min": ("Non-Mandarin", "Min", "Minnan"), "Xiang": ("Non-Mandarin", "Xiang", "")}
GX_BASIS = "Guangxi gazetteer rake"

# LAC2 group (the label's first part) -> the county table's (group, subgroup) vocabulary
GROUP = {"吳語": "Wu", "晉語": "Jin", "徽語": "Hui", "贛語": "Gan", "湘語": "Xiang", "客家話": "Hakka",
         "平話和土話": "Pinghua and Tuhua"}
MIN_SUB = {"閩南片": "Minnan", "閩東片": "Mindong", "閩北片": "Minbei", "閩中片": "Minzhong",
           "莆仙片": "Puxian", "邵將片": "Shaojiang", "雷州片": "Leizhou", "瓊文片": "Qiongwen"}
SKIP = {"域外方音", "戲劇", "民族語", "現代標準漢語", "鄉話"}   # foreign readings, opera, minority
# languages, the standard, and Waxiang (unclassified; no node; western Hunan only)


def table_group(label):
    """MCPDict 地圖集二分區 -> (group, subgroup) as cn_dialect.csv spells them, or None"""
    top = label.split(",")[0]
    if top in SKIP:
        return None
    parts = top.split("－")
    g = parts[0]
    if g.endswith("官話"):
        return ("Southwestern", "")          # any Mandarin: dialect_node puts all eight on one leaf
    if g == "粤語":
        return ("Yue", "Siyi") if len(parts) > 1 and parts[1] == "四邑片" else ("Yue", "")
    if g == "閩語":
        sub = MIN_SUB.get(parts[1]) if len(parts) > 1 else None
        if sub is None:
            raise SystemExit(f"MCPDict: Min label with no subgroup: {top}")
        return ("Min", sub)
    if g in GROUP:
        return (GROUP[g], "")
    raise SystemExit(f"MCPDict: unknown group label {top}")


def chinese_by_county():
    """people each county draws as Chinese (cn.csv x cn2020.shares), before dialects and migrants"""
    df = pd.read_csv(HERE / "data" / "normalized" / "cn.csv", dtype={"geo_id": str})
    out = []
    for (cat, prov), g in df.groupby(["source_category", "prov"]):
        for node, share in cn2020.shares(cat, int(prov)):
            if node == cn2020.CHINESE:
                out.append(pd.Series((g["count"] * share).to_numpy(), index=g["geo_id"].to_numpy()))
    return pd.concat(out).groupby(level=0).sum()


def load_points():
    d = json.loads(GEOJSON.read_text(encoding="utf-8"))
    rows = []
    for f in d["features"]:
        p = f["properties"]
        if p.get("歷史音", "0") != "0":
            continue
        tg = table_group(p["地圖集二分區"])
        if tg is None:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        rows.append(dict(title=p["title"], place=p.get("地點", ""), label=p["地圖集二分區"].split(",")[0],
                         group=tg[0], subgroup=tg[1], island=bool(p.get("方言島")) or "方言島" in p["地圖集二分區"], lon=lon, lat=lat))
    pts = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy(
        [r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326)
    cty = gpd.read_file(COUNTIES)[["adcode", "geometry"]].to_crs(4326)
    cty["adcode"] = cty["adcode"].astype(str)
    j = gpd.sjoin(pts, cty, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    n_out = int(j["adcode"].isna().sum())
    j = j[j["adcode"].notna()].rename(columns={"adcode": "unit"})
    j["node"] = [cn2020.dialect_node(u, g, s) for u, g, s in zip(j["unit"], j["group"], j["subgroup"])]
    j["coarse"] = j["group"].map(coarse).where(j["group"] != "Southwestern", "Mandarin")
    print(f"{len(rows):,} usable points, {n_out} outside every county (Hong Kong, Macau, Taiwan, "
          f"abroad), {len(j):,} in counties; {int(j['island'].sum())} dialect islands")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


SIGMA_KM = 8.0      # about a township's radius
K = 4.0             # a cell at a point of g seeds g 5x a cell with only the atlas's say-so
OFF = 0.05          # seed of a node where neither the atlas nor a point puts it


class Placer:
    """scatter.py weighter: see PLACEMENT in the module docstring. Rows on a node the county has no
    local share of (migrants' home dialects) and every minority language follow population."""

    def __init__(self, place):
        from cn_dlac import cell_groups
        if not place.index.equals(pd.RangeIndex(len(place))):
            raise SystemExit("cn Placer: the placement layer's index must be 0..n-1")
        self.pop = place["pop"].to_numpy(dtype=float)
        unit = place["unit"].astype(str).to_numpy()
        rp = place.geometry.representative_point()
        lon, lat = rp.x.to_numpy(), rp.y.to_numpy()
        dia = pd.read_csv(OUT, dtype={"unit": str})
        dia["node"] = [cn2020.dialect_node(u, g, s) for u, g, s in zip(dia["unit"], dia["group"], dia["subgroup"])]
        multi = dia.groupby("unit")["node"].transform("nunique") > 1
        dia = dia[multi]
        pts = pd.read_csv(POINTS_OUT, dtype={"unit": str})
        cg = cell_groups(place.geometry).to_numpy()
        pos = pd.Series(np.arange(len(unit))).groupby(unit).apply(lambda s: s.to_numpy())
        self.w = {}          # (unit, node) -> weights over that unit's cells, in place order
        self.cells = {}
        for u, g in dia.groupby("unit"):
            if u not in pos.index:
                continue
            idx = pos[u]
            p = self.pop[idx]
            if p.sum() <= 0:
                continue
            share = g.groupby("node")["share"].sum()
            split = (g["basis"] == "atlas 1987 split").any()
            seeds = []
            for node in share.index:
                bas = g.loc[g["node"] == node, "basis"]
                if (bas == "MCPDict points").all():
                    base = np.full(len(idx), OFF)
                elif split or (bas == GX_BASIS).all():     # Guangxi's seeds came from the polygons
                    c = coarse(g.loc[g["node"] == node, "group"].iloc[0])
                    ok = np.array([x == "" or c in x.split("+") for x in cg[idx]])
                    base = np.where(ok, 1.0, OFF)
                else:
                    base = np.ones(len(idx))
                q = pts[pts["node"] == node]
                ker = np.zeros(len(idx))
                if len(q):
                    x0 = np.radians(lon[idx])[:, None] - np.radians(q["lon"].to_numpy())[None, :]
                    y0 = np.radians(lat[idx])[:, None] - np.radians(q["lat"].to_numpy())[None, :]
                    x0 *= np.cos(np.radians(lat[idx]))[:, None]
                    d = 6371.0 * np.sqrt(x0 ** 2 + y0 ** 2)
                    ker = np.exp(-(d.min(1) ** 2) / (2 * SIGMA_KM ** 2))
                seeds.append(base + K * ker)
            m = np.column_stack(seeds) * p[:, None]
            col = share.to_numpy() / share.sum() * p.sum()
            for _ in range(100):
                m *= (col / np.maximum(m.sum(0), 1e-12))[None, :]
                m *= (p / np.maximum(m.sum(1), 1e-12))[:, None]
            for j, node in enumerate(share.index):
                self.w[(u, node)] = m[:, j]
            self.cells[u] = idx
        self.unit = unit
        self.n_lean = self.n_pop = 0

    def weights(self, node, idx, count, plain=False):
        u = self.unit[idx[0]]
        w = self.w.get((u, node))
        if w is not None and w.sum() > 0:
            if not np.array_equal(self.cells[u], idx):        # scatter's cell order, to be safe
                w = pd.Series(w, index=self.cells[u]).reindex(idx).fillna(0).to_numpy()
            self.n_lean += 1
            return w
        self.n_pop += 1
        p = self.pop[idx]
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n_lean:,} (county, dialect group) rows leaned towards MCPDict points and the "
                f"atlas's area, {self.n_pop:,} on population")


def neighbours():
    """adcode -> the adcodes of the counties whose polygons touch it (within ~100 m)"""
    cty = gpd.read_file(COUNTIES)[["adcode", "geometry"]].to_crs(3857)
    cty["adcode"] = cty["adcode"].astype(str)
    buf = cty.assign(geometry=cty.geometry.buffer(100))
    j = gpd.sjoin(buf, cty, how="inner", predicate="intersects")
    j = j[j["adcode_left"] != j["adcode_right"]]
    return j.groupby("adcode_left")["adcode_right"].apply(set).to_dict()


def rake(m, rows, cols, iters=1000):
    """iterative proportional fitting of matrix m (units x nodes, zeros stay zero)"""
    m = m.copy()
    for _ in range(iters):
        m = m.mul(rows / m.sum(1).replace(0, np.nan), axis=0).fillna(0)
        m = m.mul(cols / m.sum(0).replace(0, np.nan), axis=1).fillna(0)
    return m


def poly_shares(prov):
    """unit x coarse group: each group's share of a county's people inside the 1987 atlas polygons
    (cells outside every Chinese polygon left out; a mixed polygon counts half for each group)"""
    from cn_dlac import GRID, cell_groups
    grid = gpd.read_file(GRID)
    grid = grid[grid["unit"].astype(str).str[:2] == prov].to_crs(4326).reset_index(drop=True)
    cg = cell_groups(grid.geometry).to_numpy()
    rec = []
    for c, sub in pd.DataFrame({"unit": grid["unit"].astype(str), "pop": grid["pop"].astype(float),
                                "cg": cg}).groupby("cg"):
        if not c:
            continue
        for part in c.split("+"):
            rec.append(sub.groupby("unit")["pop"].sum().div(len(c.split("+"))).rename(part))
    by = pd.concat(rec, axis=1).fillna(0).T.groupby(level=0).sum().T
    return by.div(by.sum(1).replace(0, np.nan), axis=0).fillna(0)


def gx_rake(sub, zh, dia, pubg):
    """Guangxi: seed, then rake to county totals and the gazetteer's shares (see FULL above)"""
    zero = sub[sub["unit"].map(zh).fillna(0) <= 0]       # no Chinese: shares as they are
    sub = sub[sub["unit"].map(zh).fillna(0) > 0].copy()
    units = sorted(set(sub["unit"]))
    poly = poly_shares(FULL).reindex(units).fillna(0)
    m = sub.pivot_table(index="unit", columns="node", values="count", aggfunc="sum", fill_value=0)
    meta = sub.drop_duplicates(["unit", "node"]).set_index(["unit", "node"])
    node_cg = sub.groupby("node")["group"].first().map(coarse).where(
        sub.groupby("node")["group"].first() != "Southwestern", "Mandarin").to_dict()
    basis = {}                       # (unit, node) -> basis of a seeded or re-seeded cell
    pts = pd.read_csv(POINTS_OUT, dtype={"unit": str})
    pt_cells = set(zip(pts["unit"], pts["node"]))
    z = pd.Series(zh).reindex(units)
    for cg, (sg, grp, subg) in SEED_ROW.items():
        node = cn2020.dialect_node("450000", grp, subg)
        node_cg[node] = cg
        if node not in m.columns:
            m[node] = 0.0
        floor_share, where = FLOOR.get(cg, (0.0, set()))
        for u in units:
            p = poly.at[u, cg] if cg in poly else 0.0
            if cg == "Xiang":
                share = p if u in XIANG_COUNTIES else 0.0          # the polygons' share, undiscounted
            else:
                named = (where is None and u not in HAKKA_NONE) or (where is not None and u in where)
                pointed = (u, node) in pt_cells
                share = p * (GAMMA_CORE if u in CORE else GAMMA if (named or pointed) else GAMMA_WEAK)
                if named:
                    share = max(share, floor_share)
            seed = share * z[u]
            if seed > m.at[u, node]:
                if m.at[u, node] == 0 or (u, node) not in meta.index or \
                        meta.loc[(u, node), "basis"] == "MCPDict points":
                    basis[(u, node)] = GX_BASIS
                m.at[u, node] = seed
    cgs = pd.Series(node_cg).reindex(m.columns)
    before = dia[dia["unit"].str[:2] == FULL].groupby("node")["count"].sum().groupby(
        lambda n: node_cg.get(n, n)).sum()
    after_a = sub.groupby("node")["count"].sum().groupby(lambda n: node_cg[n]).sum()
    seeded = m.sum(0).groupby(cgs).sum()
    total = z.sum()
    # Xiang fixed (above); the other groups share the rest by the gazetteer
    x_cols = cgs.index[cgs == "Xiang"]
    xiang = min(m[x_cols].sum().sum(), pubg["Xiang"] / sum(pubg.values()) * total)
    rest = {k: v for k, v in pubg.items() if k != "Xiang"}
    tgt = pd.Series(rest) / sum(rest.values()) * (total - xiang)
    tgt["Xiang"] = xiang
    missing = set(cgs) - set(tgt.index)
    if missing:
        raise SystemExit(f"Guangxi: groups the gazetteer does not count: {missing}")
    tgt_n = m.sum(0) * cgs.map(tgt / seeded)           # a group's nodes in proportion
    r = rake(m, z, tgt_n, iters=3000)
    err = max((r.sum(0) - tgt_n).abs().max(), (r.sum(1) - z).abs().max())
    if err > 1:
        raise SystemExit(f"Guangxi: rake did not converge ({err:,.0f})")
    long = r.stack().rename("count").reset_index()
    long = long[long["count"] > 0.5]
    rows = []
    for x in long.itertuples(index=False):
        if (x.unit, x.node) in meta.index:
            d = meta.loc[(x.unit, x.node)].to_dict()
        else:
            cg = node_cg[x.node]
            sg, grp, subg = SEED_ROW[cg]
            d = dict(sgroup=sg, group=grp, subgroup=subg, from_code=x.unit)
        if (x.unit, x.node) in basis:
            d["basis"] = basis[(x.unit, x.node)]
        rows.append(dict(unit=x.unit, node=x.node, count=x.count, **{k: d[k] for k in
                         ("sgroup", "group", "subgroup", "basis", "from_code")}))
    out = pd.DataFrame(rows)
    out["share"] = out["count"] / out["unit"].map(zh)
    pub = pd.Series(pubg) / sum(pubg.values()) * total
    print("\nGuangxi, millions (pre-migrant): before = the table and §7; points = after (a) with no cap;"
          " seeded = the rake's starting table; published = the gazetteer's share; raked = the result")
    print(pd.DataFrame({"before": before / 1e6, "points": after_a / 1e6, "seeded": seeded / 1e6,
                        "published": pub / 1e6, "raked": r.sum(0).groupby(cgs).sum() / 1e6})
          .round(2).sort_values("before", ascending=False).to_string())
    return pd.concat([zero, out], ignore_index=True)


def main():
    zh = chinese_by_county()
    dia = pd.read_csv(DLAC, dtype={"unit": str, "from_code": str})
    dia["node"] = [cn2020.dialect_node(u, g, s) for u, g, s in zip(dia["unit"], dia["group"], dia["subgroup"])]
    pts = load_points()
    pts.to_csv(POINTS_OUT, index=False, columns=["unit", "node", "coarse", "island", "label", "title",
                                                  "place", "lon", "lat"])

    dia["count"] = dia["share"] * dia["unit"].map(zh).fillna(0)
    prov = dia["unit"].str[:2]
    sc = dia[prov.isin(SCOPE)]
    pin = pts[pts["unit"].str[:2].isin(SCOPE)]

    # per-group weight: the node's people per point across SCOPE, against the median node
    ppl = sc.groupby("node")["count"].sum()
    npt = pin.groupby("node").size()
    per = (ppl.reindex(npt.index).fillna(0) / npt)
    w = (per / per[npt >= 5].median()).clip(W_MIN, W_MAX)
    print("people per point and weight, by node (SCOPE):")
    print(pd.DataFrame({"points": npt, "people_M": (ppl.reindex(npt.index) / 1e6).round(2),
                        "weight": w.round(2)}).sort_values("points", ascending=False).to_string())
    pin = pin.assign(w=pin["node"].map(w))

    own = dia.groupby("unit")["node"].apply(set)
    nbr = neighbours()
    added, rows = {}, []
    n_edge = n_pocket = 0
    for u, g in pin.groupby("unit"):
        mine = own.get(u, set())
        g = g[~g["node"].isin(mine)]
        if g.empty:
            continue
        # an EDGE point: its group is the atlas's group of a neighbouring county and MCPDict does not
        # call it an island, so the group's area runs across the county line; anything else is a
        # POCKET (a 方言島, or a group no neighbour has: 龍游's Min Dong village, Guangxi's Hakka)
        near = set().union(*[own.get(v, set()) for v in nbr.get(u, ())])
        edge = ~g["island"] & g["node"].isin(near)
        n_edge += int(edge.sum())
        n_pocket += int((~edge).sum())
        # per node: edge points up to CAP, pockets up to ISLAND_CAP (Mandarin's garrison islands up
        # to ISLAND_CAP_MANDARIN); then all of the county's added groups together to CAP
        a = {}
        for node, h in g.assign(edge=edge).groupby("node"):
            cap_i = ISLAND_CAP_MANDARIN if h["coarse"].iloc[0] == "Mandarin" else ISLAND_CAP
            e_edge, e_pocket = h.loc[h["edge"], "w"].sum(), h.loc[~h["edge"], "w"].sum()
            a[node] = CAP * (1 - np.exp(-e_edge / E0)) + cap_i * (1 - np.exp(-e_pocket / E0))
        a = pd.Series(a)
        added[u] = a if (GX_FULL_RAKE and u[:2] == FULL) else a * min(1.0, CAP / a.sum())   # Guangxi: no combined cap
    print(f"points in SCOPE of a group their county lacks: {n_edge} at an edge, {n_pocket} pockets")

    # the new shares
    new = []
    for u, g in dia.groupby("unit"):
        a = added.get(u)
        if a is None:
            new.append(g)
            continue
        keep = g.copy()
        keep["share"] = keep["share"] * (1 - a.sum())
        new.append(keep)
        for node, s in a.items():
            # the subgroup: this county's points of that node, the most common label's
            p = pin[(pin["unit"] == u) & (pin["node"] == node)].iloc[0]
            new.append(pd.DataFrame([dict(unit=u, sgroup="Mandarin" if p["group"] == "Southwestern" else p["group"],
                                          group=p["group"], subgroup=p["subgroup"] or "", share=s,
                                          basis="MCPDict points", from_code=u, node=node)]))
    new = pd.concat(new, ignore_index=True)
    new["count"] = new["share"] * new["unit"].map(zh).fillna(0)

    # hold the province totals: each coarse group the gazetteer counts may move only towards its
    # published share (target = the total after the points, clipped between today's and that share);
    # a group the gazetteer does not count keeps what the points gave it
    held = []
    for p, (name, pubg) in PUB.items():
        if GX_FULL_RAKE and p == FULL:
            held.append(gx_rake(new[new["unit"].str[:2] == p], zh, dia, pubg))
            continue
        sub = new[new["unit"].str[:2] == p]
        held.append(sub[sub["count"] <= 0])          # counties with no Chinese: shares as they are
        sub = sub[sub["count"] > 0]
        m = sub.pivot_table(index="unit", columns="node", values="count", aggfunc="sum", fill_value=0)
        node_grp = sub.groupby("node")["group"].first()
        node_cg = {n: coarse(node_grp[n]) for n in m.columns}
        cur_n = dia[dia["unit"].str[:2] == p].groupby("node")["count"].sum().reindex(m.columns).fillna(0)
        aft_n = m.sum(0)
        cur, aft = cur_n.groupby(node_cg).sum(), aft_n.groupby(node_cg).sum()
        pub = pd.Series(pubg) / sum(pubg.values()) * aft.sum()
        tgt = aft.copy()
        for cg in pub.index.intersection(aft.index):
            lo, hi = min(cur.get(cg, 0), pub[cg]), max(cur.get(cg, 0), pub[cg])
            tgt[cg] = min(max(aft[cg], lo), hi)
        free = (tgt == aft)
        tgt[free] += (aft.sum() - tgt.sum()) * aft[free] / aft[free].sum()
        tgt_n = aft_n * pd.Series(node_cg).map(tgt / aft)          # a group's nodes in proportion
        r = rake(m, m.sum(1), tgt_n)
        err = (r.sum(0) - tgt_n).abs().max()
        if err > 1:
            raise SystemExit(f"{name}: rake did not converge ({err:,.0f})")
        long = r.stack().rename("count").reset_index()
        long = long[long["count"] > 0]
        sub = sub.drop_duplicates(["unit", "node"]).drop(columns=["share", "count"]).merge(long, on=["unit", "node"])
        sub["share"] = sub["count"] / sub["unit"].map(zh)
        held.append(sub)
        print(f"\n{name}, millions (pre-migrant; published = the gazetteer's share of the same total)")
        print(pd.DataFrame({"before": cur / 1e6, "points": aft / 1e6, "published": pub / 1e6,
                            "held": r.sum(0).groupby(node_cg).sum() / 1e6})
              .round(2).sort_values("before", ascending=False).to_string())
    new = pd.concat([new[~new["unit"].str[:2].isin(["44", "45"])]] + held, ignore_index=True)

    s = new.groupby("unit")["share"].sum()
    if (s.sub(1).abs() > 1e-3).any() or set(new["unit"]) != set(dia["unit"]):
        print(s[s.sub(1).abs() > 1e-6].head(), sorted(set(dia["unit"]) ^ set(new["unit"]))[:10])
        raise SystemExit("cn_mcpdict: shares do not cover every county once")
    new["share"] = new["share"] / new["unit"].map(s)
    cols = ["unit", "sgroup", "group", "subgroup", "share", "basis", "from_code"]
    out = new.sort_values(["unit", "share"], ascending=[True, False])
    out[cols].to_csv(OUT, index=False)

    # report
    b = dia.groupby(["unit", "node"])["count"].sum()
    a = new.groupby(["unit", "node"])["count"].sum()
    d = pd.concat([b.rename("before"), a.rename("after")], axis=1).fillna(0)
    d["moved"] = (d["after"] - d["before"]).clip(lower=0)
    print(f"\n{len(added)} counties given groups only the points show; people moved between groups "
          f"(all of SCOPE, after the hold): {d['moved'].sum():,.0f}")
    by = d.reset_index()
    by["prov"] = by["unit"].str[:2].map(SCOPE)
    print(by.groupby("prov")["moved"].sum().div(1e6).round(2).to_string())
    print(f"wrote {OUT.name} ({len(out):,} rows), {POINTS_OUT.name} ({len(pts):,} points)")


if __name__ == "__main__":
    main()
