"""West Africa: the undivided Christian cell split into churches, seven countries.

    python sources/wa_christian.py            # all seven
    python sources/wa_christian.py sl gn      # some of them

Anita, 2026-10-16, reversing the habit of leaving a census's undivided `Christian` on the bare
`christianity` node: a flat colour that stops at a border makes neighbours look as if they
hold different religions when they do not. unspecified_splits/westafrica.md is the research;
this file is the build. Each country writes `data/normalized/<cc>_split.csv`, which
countries/<cc>.py reads IN PLACE OF that country's Christian rows, and the per-country share
tables it used to `data/raw/wa_christian/`.

    cc  bucket replaced            split from                                   tier
    ng  `Christian` (37.0%)        GFS 2023 by geopolitical zone, Afrobarometer  modelled
                                   as the zone-rank witness
    lr  `Christian` (84.9%)        Afrobarometer R4-R9 named answers; Lutheran,  modelled
                                   Baptist, Catholic by county
    sl  `Christianity` (21.9%)     the 2015 census's own national split          derived
    gn  `Chrétienne` (6.8%)        Afrobarometer R5-R9 named + Pew 2010          derived
    sn  `Chrétiens` (4.5%)         1988 census's own Diourbel ratio + Pew 2010   derived
    gm  `Christianity` (3.75%)     Afrobarometer R7-R9, Pew 2010, Church claim   derived
    gw  `Cristão` (22.1%)          the 1991 census's own national split         derived

`derived` (spec §2.7a): a census counted the Christian cell at the unit and only its division
is borrowed, so every split row carries `parent_column=<the cell>` and rolls back to it when
inferred dots are hidden. Nigeria and Liberia were already `modelled` throughout (nobody
counted religion at the unit), and the split rows stay `modelled` with them.

Every country's split rows sum to the cell they replace, unit by unit, by largest-remainder
rounding, and that is asserted. Per-country checks are printed and the load-bearing ones
asserted; each is described where it runs.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

NORMDIR = os.path.join(ROOT, "data", "normalized")
RAWDIR = os.path.join(ROOT, "data", "raw", "wa_christian")
AB_DIR = os.path.join(ROOT, "data", "raw", "afrobarometer")
GFS_CSV = os.path.join(ROOT, "data", "raw", "gfs", "gfs_all_countries_wave2.csv")

# Pew-Templeton, *Global Christianity* (December 2011), country table, 2010 estimates, % of the
# total population: https://www.pewresearch.org/religion/2011/12/19/
# table-christian-population-as-percentages-of-total-population-by-country/
# A compiler's estimate (§3.1 `estimate`), used ONLY as a witness to a share, never as a level.
PEW_2010 = {  # (Catholic, all Christians)
    "ng": (12.6, 50.8), "sn": (3.5, 3.7), "gm": (3.1, 4.7), "gw": (17.9, 20.1),
    "gn": (7.5, 11.0), "sl": (7.1, 21.1), "lr": (7.2, 85.6),
}


def pew_catholic(cc):
    c, t = PEW_2010[cc]
    return c / t


# ---------------------------------------------------------------------------------------------
# shared helpers

def largest_remainder(total, weights):
    """Integers summing to `total` exactly, in proportion to `weights`."""
    w = np.asarray(weights, dtype=float)
    if total == 0 or w.sum() == 0:
        return np.zeros(len(w), dtype="int64")
    raw = total * w / w.sum()
    base = np.floor(raw).astype("int64")
    short = int(total - base.sum())
    if short:
        order = np.argsort(-(raw - base))
        base[order[:short]] += 1
    return base


def split_rows(base, parent, shares, source_id, year, how):
    """Replace each row of `base` whose category is `parent` by one row per share.

    `shares` is {category: weight} for every unit alike, or {geo_id: {category: weight}}.
    """
    rows = []
    par = base[base["source_category"] == parent]
    if par.empty:
        raise SystemExit(f"no `{parent}` rows in the base file")
    for _, r in par.iterrows():
        sh = shares.get(r["geo_id"], shares) if isinstance(next(iter(shares.values())), dict) \
            else shares
        cats = list(sh)
        counts = largest_remainder(int(r["count"]), [sh[c] for c in cats])
        for c, n in zip(cats, counts):
            rows.append({
                "geo_id": r["geo_id"], "geo_level": r["geo_level"], "geo_name": r["geo_name"],
                "source_category": c, "count": int(n), "basis": r["basis"], "year": year,
                "source_id": source_id,
                # parent_column LAST, which is where countries/_shared.py::_column_of reads it.
                "note": f"{how}; {r['geo_name']}'s `{parent}` {int(r['count'])} x "
                        f"{sh[c] / sum(sh.values()):.4f}; parent_column={parent}",
            })
    out = pd.DataFrame(rows)
    # THE CHECK EVERY COUNTRY GETS: each unit's split sums to the cell it replaces.
    a = par.set_index("geo_id")["count"].astype(int)
    b = out.groupby("geo_id")["count"].sum().reindex(a.index)
    if not (a == b).all():
        raise SystemExit(f"split does not close on `{parent}`:\n{pd.concat([a, b], axis=1)}")
    print(f"  `{parent}` {int(a.sum()):,} people on {len(a)} units -> "
          f"{out['source_category'].nunique()} rows a unit; every unit closes exactly")
    for c, n in out.groupby("source_category")["count"].sum().sort_values(ascending=False).items():
        print(f"    {c:58s} {n:>12,}  {100 * n / a.sum():5.1f}% of the cell")
    return out


def write(cc, df):
    p = os.path.join(NORMDIR, f"{cc}_split.csv")
    df.to_csv(p + ".part", index=False)
    os.replace(p + ".part", p)
    print(f"  -> {os.path.relpath(p, ROOT)} ({len(df)} rows)")


def write_raw(name, df):
    os.makedirs(RAWDIR, exist_ok=True)
    p = os.path.join(RAWDIR, name)
    df.to_csv(p + ".part", index=True)
    os.replace(p + ".part", p)
    print(f"  -> {os.path.relpath(p, ROOT)}")


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def exact_perm_p(a, b):
    """One-sided p of a Spearman rho over every ordering (for six units, 720)."""
    obs = spearman(a, b)
    hits = tot = 0
    for perm in itertools.permutations(list(b)):
        tot += 1
        hits += spearman(a, list(perm)) >= obs - 1e-9
    return obs, hits / tot


def sampled_perm_p(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearman(a, b)
    null = np.array([spearman(a, rng.permutation(np.asarray(b))) for _ in range(n)])
    return obs, float((null >= obs - 1e-9).mean()), float(np.quantile(null, 0.95))


# ---------------------------------------------------------------------------------------------
# the Afrobarometer, read once for every country here

_AB = None


def ab_rows():
    """Every West African respondent in R4-R9: round, country, region, answer, weight.

    Reads only four columns of each merged .sav (sources/afrobarometer.py's ROUNDS names the
    religion and weight column per round; the religion VARIABLE NAME moves, see that file).
    """
    global _AB
    if _AB is not None:
        return _AB
    import pyreadstat
    import afrobarometer as ab

    want = {"nigeria", "senegal", "gambia", "guinea", "sierra leone", "liberia"}
    frames = []
    for rnd, fn, _url, rel, wt in ab.ROUNDS:
        p = os.path.join(AB_DIR, fn)
        try:
            _, meta = pyreadstat.read_sav(p, metadataonly=True)
            enc = None
        except Exception:
            _, meta = pyreadstat.read_sav(p, metadataonly=True, encoding="LATIN1")
            enc = "LATIN1"
        up = {c.upper(): c for c in meta.column_names}
        use = [up[n.upper()] for n in ("COUNTRY", "REGION", rel, wt)]
        kw = {"usecols": use, **({"encoding": enc} if enc else {})}
        df, meta = pyreadstat.read_sav(p, **kw)
        c, g, r, w = use
        lab = str(meta.column_names_to_labels.get(r, ""))
        if "religion of respondent" not in lab.casefold():
            raise SystemExit(f"R{rnd} {r} is labelled {lab!r}, not the religion question")
        cname = df[c].map(meta.variable_value_labels.get(c, {})).astype(str).map(ab.fold)
        sub = df[cname.isin(want)]
        frames.append(pd.DataFrame({
            "round": rnd,
            "country": cname[sub.index],
            "region": sub[g].map(meta.variable_value_labels.get(g, {})).astype(str),
            "rel": sub[r].map(meta.variable_value_labels.get(r, {})),
            "w": pd.to_numeric(sub[w], errors="coerce"),
        }))
    _AB = pd.concat(frames, ignore_index=True)
    _AB = _AB[_AB["rel"].notna()]
    print(f"  Afrobarometer R4-R9: {len(_AB):,} West African respondents read")
    return _AB


def ab_church(label):
    """An Afrobarometer answer -> the church group used here; None if not Christian."""
    s = str(label).replace("’", "'").casefold().split("(")[0].strip()
    if s.startswith("christian only"):
        return "UNNAMED"
    table = {
        "roman catholic": "Catholic",
        "pentecostal": "Pentecostal", "assemblies of god": "Pentecostal",
        "methodist": "Methodist", "lutheran": "Lutheran", "baptist": "Baptist",
        "anglican": "Anglican",
        "evangelical": "Evangelical", "cma": "Evangelical",
        "presbyterian": "Presbyterian", "calvinist": "Presbyterian",
        "dutch reformed": "Presbyterian",
        "independent": "Independent", "zionist christian church": "Independent",
        "jehovah's witness": "Other", "seventh day adventist": "Other", "mormon": "Other",
        "church of christ": "Other", "orthodox": "Other", "coptic": "Other",
        "new apostolic church": "Other", "mennonite": "Other", "quaker/friends": "Other",
    }
    if s.startswith("alliance chr"):
        return "Evangelical"
    return table.get(s)


def ab_named(country):
    d = ab_rows()
    d = d[d["country"] == country].copy()
    d["g"] = d["rel"].map(ab_church)
    return d[d["g"].notna()]


# ---------------------------------------------------------------------------------------------
# Nigeria

NG_GROUP = {  # GFS REL3_Y1 code -> group (codebook wave 2, OSF 285w7, p.21; sources/ng.py)
    8: "Pentecostal",
    7: "Protestant", 6: "Protestant", 4: "Protestant", 5: "Protestant",
    2: "Protestant",    # `Orthodox`: Nigeria's "orthodox churches", the old mission churches
    9: "Protestant",    # `Independent/Evangelical`: ECWA and COCIN in the Middle Belt
    11: "Other Christian", 12: "Other Christian", 10: "Other Christian", 13: "Other Christian",
    96: "Other Christian",
    97: "Christian",    # "none in particular": stays on the bare node (§2.7a)
}
NG_AB_GROUP = {
    "Pentecostal": "Pentecostal",
    "Baptist": "Protestant", "Methodist": "Protestant", "Presbyterian": "Protestant",
    "Lutheran": "Protestant", "Evangelical": "Protestant", "Other": None,
}
NG_LABEL = {
    "Pentecostal": "Pentecostal",
    "Protestant": "Protestant (Baptist, Methodist, Presbyterian, ECWA, COCIN and others)",
    "Other Christian": "Other Christian (Witnesses, Adventists, Latter-day Saints, African "
                       "Initiated)",
    "Christian": "Christian",
}
NG_P_MAX = 0.05     # zone rank agreement with the Afrobarometer, one-sided exact p


def build_ng():
    import ng

    print("\n=== Nigeria ===")
    base = pd.read_csv(os.path.join(NORMDIR, "ng.csv"), dtype={"geo_id": str})
    s2z = {s: z for z, ss in ng.ZONES.items() for s in ss}
    zone = base["geo_name"].map(s2z)
    if zone.isna().any():
        raise SystemExit(f"states with no zone: {sorted(set(base.loc[zone.isna(), 'geo_name']))}")

    # GFS 2023: the bucket's people are the non-Catholic, non-Anglican Christians.
    g = pd.read_csv(GFS_CSV, usecols=["COUNTRY", "REGION1_Y1", "REL3_Y1", "ANNUAL_WEIGHT_C1"],
                    low_memory=False)
    g = g[g["COUNTRY"] == ng.GFS_COUNTRY].copy()
    g["REL3_Y1"] = pd.to_numeric(g["REL3_Y1"], errors="coerce")
    b = g[g["REL3_Y1"].isin(list(NG_GROUP))].copy()
    b["zone"] = b["REGION1_Y1"].astype(int).map(ng.GFS_REGION).map(s2z)
    b["grp"] = b["REL3_Y1"].astype(int).map(NG_GROUP)
    w = "ANNUAL_WEIGHT_C1"
    gt = b.groupby(["zone", "grp"])[w].sum().unstack(fill_value=0.0)
    gt = gt.div(gt.sum(axis=1), axis=0)
    gt["n"] = b.groupby("zone").size()
    print(f"  GFS 2023: {len(b):,} non-Catholic, non-Anglican Christians; shares of them by zone")
    print((gt * [100, 100, 100, 100, 1]).round(1).to_string())
    write_raw("ng_gfs_zone_shares.csv", gt)

    # The witness: the Afrobarometer's NAMED answers in the same bucket, all six rounds.
    d = ab_named("nigeria")
    d["state"] = d["region"].map(lambda s: ng.NORM.get(ng.ckey(s)))
    if d["state"].isna().any():
        raise SystemExit(f"Afrobarometer regions with no state: "
                         f"{sorted(set(d.loc[d['state'].isna(), 'region']))}")
    d["zone"] = d["state"].map(s2z)
    d["grp"] = d["g"].map(NG_AB_GROUP)
    a = d[d["grp"].notna()]
    at = a.groupby(["zone", "grp"])["w"].sum().unstack(fill_value=0.0)
    at = at.div(at.sum(axis=1), axis=0)
    at["n"] = a.groupby("zone").size()
    write_raw("ng_afrobarometer_zone_shares.csv", at)
    named = gt[["Pentecostal", "Protestant", "Other Christian"]]
    named = named.div(named.sum(axis=1), axis=0)
    print("  zone rank agreement, GFS 2023 against the Afrobarometer R4-R9 (named answers):")
    for grp in ("Pentecostal", "Protestant"):
        z = named.index
        rho, p = exact_perm_p(list(named.loc[z, grp]), list(at.loc[z, grp]))
        print(f"    {grp:12s} rho {rho:+.3f}, exact one-sided p {p:.3f} over 720 orderings")
        if p > NG_P_MAX:
            raise SystemExit(f"{grp}: the two surveys no longer rank the zones alike "
                             f"(p {p:.3f} > {NG_P_MAX}); the zone split is unsupported")
    gp = float((b.loc[b.grp == "Pentecostal", w].sum()) / b.loc[b.grp != "Christian", w].sum())
    ap = float(a.loc[a.grp == "Pentecostal", "w"].sum() / a["w"].sum())
    print(f"  level: Pentecostal {100 * gp:.1f}% of the named bucket in GFS, "
          f"{100 * ap:.1f}% in the Afrobarometer's named answers (its `Christian only` "
          f"swing keeps it from levelling); GFS sets the level")

    shares = {}
    for gid, z in zip(base["geo_id"], zone):
        shares[gid] = {NG_LABEL[k]: float(gt.loc[z, k]) for k in NG_LABEL}
    out = split_rows(base, "Christian", shares, "ng_gfs2023_zone_split", "2023",
                     "GFS 2023 wave 1 REL3, share of non-Catholic non-Anglican Christians in the "
                     "state's geopolitical zone")
    write("ng", out)


# ---------------------------------------------------------------------------------------------
# Liberia

LR_COUNTY = ["Lutheran", "Baptist", "Catholic"]   # carried on their own county shares
LR_FLAT = ["Pentecostal", "Methodist", "Protestant", "Other Christian"]
LR_AB_GROUP = {
    "Catholic": "Catholic", "Pentecostal": "Pentecostal", "Methodist": "Methodist",
    "Lutheran": "Lutheran", "Baptist": "Baptist",
    "Evangelical": "Protestant", "Presbyterian": "Protestant", "Anglican": "Protestant",
    "Independent": "Other Christian", "Other": "Other Christian",
}
LR_LABEL = {
    "Pentecostal": "Pentecostal", "Methodist": "Methodist", "Lutheran": "Lutheran",
    "Baptist": "Baptist", "Catholic": "Roman Catholic",
    "Protestant": "Protestant (Evangelical, Presbyterian, Episcopal)",
    "Other Christian": "Other Christian",
}
LR_MIN_NAMED = 30            # a county with fewer named Christians takes the national share
LR_SPLIT_MIN_N = 15          # per half, for the split-half
LR_SPLIT_P_MAX = 0.05
LR_PEW_GAP_MAX = 0.05        # the Catholic share against Pew 2010's 8.4% of Christians
LR_CHURCH_CLAIM = 361_000 / 4_928_000 / 0.849   # gcatholic.org 2023, on the census's 84.9%


def build_lr():
    import lr

    print("\n=== Liberia ===")
    base = pd.read_csv(os.path.join(NORMDIR, "lr.csv"), dtype={"geo_id": str})
    lut = pd.read_csv(os.path.join(ROOT, "data", "geo", "lr", "lr_lookup.csv"), dtype=str)
    name_to_id = dict(zip(lut["name"], lut["geo_id"]))

    d = ab_named("liberia")
    d["county"] = d["region"].map(lambda s: lr.NORM.get(lr.ckey(s)))
    if d["county"].isna().any():
        raise SystemExit(f"regions with no county: {sorted(set(d.loc[d.county.isna(), 'region']))}")
    d["geo_id"] = d["county"].map(name_to_id)
    unnamed = (d["w"].where(d["g"] == "UNNAMED", 0.0).groupby(d["round"]).sum()
               / d.groupby("round")["w"].sum())
    n = d[d["g"] != "UNNAMED"].copy()
    n["grp"] = n["g"].map(LR_AB_GROUP)
    if n["grp"].isna().any():
        raise SystemExit(f"church answers with no group: {sorted(set(n.loc[n.grp.isna(), 'g']))}")

    # 1. The mix of NAMED Christians does not move with how many name a church.
    by_round = n.groupby(["round", "grp"])["w"].sum().unstack(fill_value=0.0)
    by_round = by_round.div(by_round.sum(axis=1), axis=0)
    by_round["unnamed"] = unnamed
    by_round["n_named"] = n.groupby("round").size()
    print("  named Christians by round (shares), with the share who named no church:")
    print((by_round * 100).round(1).assign(n_named=by_round["n_named"]).to_string())
    for grp in LR_COUNTY + LR_FLAT:
        r = float(np.corrcoef(by_round["unnamed"], by_round[grp])[0, 1])
        print(f"    {grp:16s} range {100 * by_round[grp].min():4.1f}-"
              f"{100 * by_round[grp].max():4.1f}%; r against the unnamed share {r:+.2f}")
    write_raw("lr_afrobarometer_named_by_round.csv", by_round)

    nat = n.groupby("grp")["w"].sum()
    nat = nat / nat.sum()
    print("  pooled national shares of named Christians: "
          + ", ".join(f"{k} {100 * v:.1f}" for k, v in nat.sort_values(ascending=False).items()))

    # 2. The Catholic level against two outside witnesses.
    pew = pew_catholic("lr")
    print(f"  Catholic: {100 * nat['Catholic']:.1f}% of Christians here; Pew 2010 "
          f"{100 * pew:.1f}%; the Church's own 2023 claim {100 * LR_CHURCH_CLAIM:.1f}%")
    if abs(nat["Catholic"] - pew) > LR_PEW_GAP_MAX:
        raise SystemExit("the Catholic share has drifted more than 5 points from Pew 2010")

    # 3. Which churches have a county geography: split-half, even rounds against odd.
    halves = {}
    for h, sel in (("even", n["round"] % 2 == 0), ("odd", n["round"] % 2 == 1)):
        x = n[sel]
        t = x.groupby(["geo_id", "grp"])["w"].sum().unstack(fill_value=0.0)
        halves[h] = (t.div(t.sum(axis=1), axis=0), x.groupby("geo_id").size())
    keep = [k for k in halves["even"][0].index.intersection(halves["odd"][0].index)
            if halves["even"][1][k] >= LR_SPLIT_MIN_N and halves["odd"][1][k] >= LR_SPLIT_MIN_N]
    print(f"  county split-half, even rounds against odd, {len(keep)} counties with "
          f"{LR_SPLIT_MIN_N}+ named Christians in each half:")
    for grp in LR_COUNTY + LR_FLAT:
        rho, p, q = sampled_perm_p(list(halves["even"][0].loc[keep, grp]),
                                   list(halves["odd"][0].loc[keep, grp]))
        carried = grp in LR_COUNTY
        print(f"    {grp:16s} rho {rho:+.3f}  p {p:.3f}  (null 95th {q:+.3f})  "
              f"{'county shares' if carried else 'flat'}")
        if carried and p > LR_SPLIT_P_MAX:
            raise SystemExit(f"{grp} no longer replicates across counties (p {p:.3f}); "
                             "move it to LR_FLAT")

    # 4. County shares for the carried three, raked to the national level.
    t = n.groupby(["geo_id", "grp"])["w"].sum().unstack(fill_value=0.0)
    cs = t.div(t.sum(axis=1), axis=0)
    cn = n.groupby("geo_id").size()
    chr_ = base[base["source_category"] == "Christian"].set_index("geo_id")["count"].astype(float)
    total = chr_.sum()
    sh = pd.DataFrame(index=chr_.index, columns=LR_COUNTY + LR_FLAT, dtype=float)
    for grp in LR_COUNTY:
        own = cs[grp].reindex(chr_.index)
        thin = cn.reindex(chr_.index).fillna(0) < LR_MIN_NAMED
        own[thin] = nat[grp]
        cnt = chr_ * own
        sh[grp] = own * (nat[grp] * total / cnt.sum())
    left = 1.0 - sh[LR_COUNTY].sum(axis=1)
    if (left < 0).any():
        raise SystemExit(f"county shares over 100%: {left[left < 0].to_dict()}")
    flat = nat[LR_FLAT] / nat[LR_FLAT].sum()
    for grp in LR_FLAT:
        sh[grp] = left * flat[grp]
    sh["n_named"] = cn.reindex(chr_.index)
    sh["county"] = base.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    print("  county shares of Christians as drawn:")
    print((sh[LR_COUNTY + LR_FLAT] * 100).round(1).assign(county=sh["county"],
                                                           n=sh["n_named"]).to_string())
    write_raw("lr_afrobarometer_county_shares.csv", sh)

    shares = {gid: {LR_LABEL[k]: float(sh.loc[gid, k]) for k in LR_COUNTY + LR_FLAT}
              for gid in sh.index}
    out = split_rows(base, "Christian", shares, "lr_afrobarometer_2008_2022_church_split",
                     "2008-2022",
                     "Afrobarometer R4-R9 named Christians; Lutheran, Baptist and Catholic at "
                     "the county's own share raked to the national level, the rest national")
    write("lr", out)


# ---------------------------------------------------------------------------------------------
# the five census countries: one national split, applied evenly

# Sierra Leone 2015 PHC, National Analytical Report Table 3.25, % of the household population;
# the census's own six Christian codes. Sum 21.9 against Table 5.3's 21.86.
SL_SHARES = {"Catholic": 7.0, "Pentecostal": 5.3, "Other Christian": 4.7, "Methodist": 3.0,
             "Anglican": 1.2, "SDA": 0.7}
# Guinea: Afrobarometer R5-R9 named Christians 64.8 Catholic / 24.3 Evangelical (Eglise
# Protestante Evangelique, CMA-founded) / the rest; Pew 2010 68 / 31.
GN_SHARES = {"Catholique": 65.0, "Protestante ou évangélique": 30.0, "Autre chrétienne": 5.0}
# Senegal 1988: Diourbel's own codes 6 and 7 are 3,775 Catholic of 3,997 (94.4%); Pew 2010 95%;
# the Afrobarometer's named Christians 83%.
SN_SHARES = {"Catholiques": 92.0, "Autres chrétiens": 8.0}
# The Gambia: Afrobarometer R7-R9 named Christians 87.7% Catholic (n=54); Pew 2010 66%; the
# Church's 2023 claim, 54,000, is about 64% of the census's Christians.
GM_SHARES = {"Catholic": 75.0, "Protestant": 25.0}
# Guinea-Bissau RGPH 1991, Resultados, Tabela 6.6 (PDF p61): Cristã Católica 125,637, Outras
# Cristãs 15,310, of 979,203 residents.
GW_SHARES = {"Cristã Católica": 125_637, "Outras Cristãs": 15_310}

BAND = {
    "sl": ("sl.csv", "Christianity", SL_SHARES, "sl_phc2015_nar_t325_split", "2015",
           "2015 PHC National Analytical Report Table 3.25, national shares of Christians"),
    "gn": ("gn.csv", "Chrétienne", GN_SHARES, "gn_afrobarometer_pew_split", "2011-2022",
           "Afrobarometer R5-R9 named Christians and Pew 2010, national shares of Christians"),
    "sn": ("sn.csv", "Chrétiens", SN_SHARES, "sn_rgph1988_diourbel_split", "1988",
           "1988 census codes 6 and 7 as Diourbel reports them, with Pew 2010, national"),
    "gm": ("gm.csv", "Christianity", GM_SHARES, "gm_afrobarometer_pew_split", "2016-2022",
           "Afrobarometer R7-R9, Pew 2010 and the Church's 2023 claim, national"),
    "gw": ("gw.csv", "Cristão", GW_SHARES, "gw_rgph1991_t66_split", "1991",
           "1991 census Tabela 6.6, national Catholic and other Christian"),
}
# The Afrobarometer witness per country (its country label) and the Catholic key in SHARES.
BAND_AB = {"sl": ("sierra leone", "Catholic"), "gn": ("guinea", "Catholique"),
           "sn": ("senegal", "Catholiques"), "gm": ("gambia", "Catholic"), "gw": (None, None)}
BAND_GAP_MAX = 0.25   # the drawn Catholic share against Pew 2010, a loose sanity bound


def build_band(cc):
    fn, parent, shares, sid, year, how = BAND[cc]
    print(f"\n=== {cc} ===")
    base = pd.read_csv(os.path.join(NORMDIR, fn), dtype={"geo_id": str})
    cat_key = BAND_AB[cc][1] or "Cristã Católica"
    drawn = shares[cat_key] / sum(shares.values())
    pew = pew_catholic(cc)
    line = f"  Catholic as drawn {100 * drawn:.1f}% of Christians; Pew 2010 {100 * pew:.1f}%"
    country = BAND_AB[cc][0]
    if country:
        nm = ab_named(country)
        nm = nm[nm["g"] != "UNNAMED"]
        abc = float(nm.loc[nm.g == "Catholic", "w"].sum() / nm["w"].sum())
        line += f"; Afrobarometer named {100 * abc:.1f}% (n={len(nm)})"
    print(line)
    if abs(drawn - pew) > BAND_GAP_MAX:
        raise SystemExit(f"{cc}: drawn Catholic share {drawn:.2f} is far from Pew's {pew:.2f}")
    write_raw(f"{cc}_shares.csv", pd.Series(shares, name="weight").to_frame())
    out = split_rows(base, parent, shares, sid, year, how)
    write(cc, out)


def main():
    todo = sys.argv[1:] or ["ng", "lr", "sl", "gn", "sn", "gm", "gw"]
    for cc in todo:
        if cc == "ng":
            build_ng()
        elif cc == "lr":
            build_lr()
        elif cc in BAND:
            build_band(cc)
        else:
            raise SystemExit(f"unknown country {cc!r}")


if __name__ == "__main__":
    main()
