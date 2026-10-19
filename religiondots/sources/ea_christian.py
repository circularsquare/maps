"""East Africa: a census's undivided Christian cell split into churches, Kenya and Malawi.

    python sources/ea_christian.py --labels   # fetch the GFS .sav header, check Kenya's county codes
    python sources/ea_christian.py            # both countries
    python sources/ea_christian.py ke         # one of them

Anita, 2026-10-16, the same ruling sources/wa_christian.py builds for West Africa: a flat colour that
stops at a border makes neighbours look as if they hold different religions when they do not.
unspecified_splits/eastafrica.md is the research; this file is the build. Each country writes
`data/normalized/<cc>_split.csv`, which countries/<cc>.py reads IN PLACE OF the cells it replaces
(countries/_shared.py::_swap_christian_split), and its share tables to `data/raw/ea_christian/`.

    cc  cell replaced                               split from                              tier
    ke  `Protestant` (15.78M, 33.4%)                GFS wave 1 (2023) REL3, by county       derived
    mw  `Seventh Day Adventist/Baptist/Apostolic`   Afrobarometer R4-R9, by region          derived
        (1.64M) and `Other Christian
        Denominations` (4.67M)

`derived` (spec §2.7a): the census counted each cell at the unit and only its division is
borrowed, so every split row carries `parent_column=<the cell>` and rolls back to it when inferred
dots are hidden. Every unit's split rows sum to the cell they replace (largest remainder), and
that is asserted here and again in _swap_christian_split.

KENYA. What KNBS's `Protestant` holds is set by its 2019 Enumerator's Instructions Manual (IPUMS
copy, PDF p.40): "Anglican Church of Kenya (ACK), Presbyterian Church of East Africa (PCEA),
African Inland Church (AIC), Lutheran, Quakers (Friends), Methodists, Seventh Day Adventists
(SDA), Salvation Army, Baptists, etc." The GFS boxes that fall in it are Anglican, Presbyterian,
Lutheran, Methodist, Baptist, Adventist and Independent/Evangelical (the Africa Inland Church: it
peaks in Kitui, Kericho, Makueni and Machakos). Five are drawn as churches; Independent/Evangelical
and Lutheran stay on `Protestant`. GFS has no Friends box, and in Kakamega, Bungoma and Vihiga (the
Friends heartland, where the Afrobarometer finds 1.0% Presbyterian and 4.4% Quaker) its
"Presbyterian" is left on `Protestant` too: Anita's call, 2026-10-16.

MALAWI. Afrobarometer answers are mapped onto the census's ten cells; inside the two cells, each
region's (North, Central, South; R4-R9 pooled) shares are applied to its districts. A region with
fewer than MW_MIN_N respondents in a cell takes the national shares. Independent churches and
`Christian only` trade places between rounds (R5: 43% and 6.5% of the cell; R6: 29% and 39%), so
per the Afrobarometer playbook they are placed as one box and split at the late rounds' (R7-R9)
national ratio.
"""
import os
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

NORMDIR = os.path.join(ROOT, "data", "normalized")
RAWDIR = os.path.join(ROOT, "data", "raw", "ea_christian")
GFS_CSV = os.path.join(ROOT, "data", "raw", "gfs", "gfs_all_countries_wave2.csv")
GFS_HEAD = os.path.join(ROOT, "data", "raw", "gfs", "gfs_wave2_header.sav")
GFS_SAV_URL = "https://osf.io/download/eadfm/"
GFS_HEAD_BYTES = 40_000_000
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

from wa_christian import largest_remainder, split_rows, write_raw  # noqa: E402


def write(cc, df):
    p = os.path.join(NORMDIR, f"{cc}_split.csv")
    df.to_csv(p + ".part", index=False)
    os.replace(p + ".part", p)
    print(f"  -> {os.path.relpath(p, ROOT)} ({len(df)} rows)")


def _raw(name, df):
    global RAWDIR
    import wa_christian
    old = wa_christian.RAWDIR
    wa_christian.RAWDIR = RAWDIR
    try:
        write_raw(name, df)
    finally:
        wa_christian.RAWDIR = old


# ---------------------------------------------------------------------------------------------
# Kenya

KE_GFS_COUNTRY = 10
KE_GFS_N = 11_389
KE_GFS_CHRISTIANS = 10_192
# REGION1_Y1 1001-1047 are the 47 counties in alphabetical order; `--labels` asserts it against
# the .sav's own value labels. Names are KNBS's spelling in ke.csv.
KE_ALPHA = ["BARINGO", "BOMET", "BUNGOMA", "BUSIA", "ELGEYO/MARAKWET", "EMBU", "GARISSA",
            "HOMA BAY", "ISIOLO", "KAJIADO", "KAKAMEGA", "KERICHO", "KIAMBU", "KILIFI",
            "KIRINYAGA", "KISII", "KISUMU", "KITUI", "KWALE", "LAIKIPIA", "LAMU", "MACHAKOS",
            "MAKUENI", "MANDERA", "MARSABIT", "MERU", "MIGORI", "MOMBASA", "MURANG'A", "NAIROBI",
            "NAKURU", "NANDI", "NAROK", "NYAMIRA", "NYANDARUA", "NYERI", "SAMBURU", "SIAYA",
            "TAITA/TAVETA", "TANA RIVER", "THARAKA-NITHI", "TRANS NZOIA", "TURKANA",
            "UASIN GISHU", "VIHIGA", "WAJIR", "WEST POKOT"]
KE_GFS_REGION = {1001 + i: n for i, n in enumerate(KE_ALPHA)}
# The former provinces by KNBS county code, the fallback for a thin county.
KE_PROVINCE = [(1, 6, "Coast"), (7, 9, "North Eastern"), (10, 17, "Eastern"), (18, 22, "Central"),
               (23, 36, "Rift Valley"), (37, 40, "Western"), (41, 46, "Nyanza"), (47, 47, "Nairobi")]
# GFS REL3_Y1 (codebook wave 2, OSF 285w7, p.21) -> the label drawn; `Protestant` stays on the cell.
KE_REL3 = {3: "Anglican", 12: "Seventh-day Adventist", 4: "Presbyterian", 7: "Baptist",
           6: "Methodist", 5: "Protestant", 9: "Protestant"}
KE_DRAWN = ["Anglican", "Seventh-day Adventist", "Presbyterian", "Baptist", "Methodist"]
KE_FRIENDS = {"KAKAMEGA", "BUNGOMA", "VIHIGA"}    # GFS `Presbyterian` here stays on Protestant
KE_MIN_N = 40               # GFS respondents in the pool; below it, the province, then the nation
KE_LEVEL_GAP_MAX = 3.0      # points: GFS pool level against the census's Protestant level
KE_SPLIT_HALF_MIN = 0.5     # median Spearman over 200 halvings, every drawn church


def ke_province(code):
    c = int(code)
    for lo, hi, name in KE_PROVINCE:
        if lo <= c <= hi:
            return name
    raise SystemExit(f"KNBS county code {code} in no province")


def fetch_labels():
    """The GFS .sav's dictionary (first 40 MB of OSF eadfm, CC BY) and Kenya's REGION1 labels."""
    import pyreadstat

    if not (os.path.exists(GFS_HEAD) and os.path.getsize(GFS_HEAD) >= GFS_HEAD_BYTES):
        req = urllib.request.Request(GFS_SAV_URL, headers={"User-Agent": UA})
        got = 0
        with urllib.request.urlopen(req, timeout=600) as r, open(GFS_HEAD + ".part", "wb") as f:
            while got < GFS_HEAD_BYTES:
                b = r.read(min(1 << 20, GFS_HEAD_BYTES - got))
                if not b:
                    break
                f.write(b)
                got += len(b)
        with open(GFS_HEAD + ".part", "rb") as f:
            if f.read(4)[:2] != b"$F":
                raise SystemExit("the OSF download is not a .sav")
        os.replace(GFS_HEAD + ".part", GFS_HEAD)
        print(f"  fetched {got:,} bytes of the GFS .sav header")
    _, meta = pyreadstat.read_sav(GFS_HEAD, metadataonly=True)
    labs = meta.variable_value_labels.get("REGION1_Y1") or {}
    ke = {int(k): str(v) for k, v in labs.items() if 1001 <= int(k) <= 1099}
    print(f"  REGION1_Y1 labels for codes 1001-1099: {len(ke)}")

    def key(s):
        return "".join(ch for ch in s.upper() if ch.isalpha())

    bad = []
    for code, name in sorted(KE_GFS_REGION.items()):
        lab = ke.get(code)
        ok = lab is not None and (key(lab) == key(name) or key(name).startswith(key(lab))
                                  or key(lab).startswith(key(name)))
        print(f"    {code}  {str(lab):24s} {name:18s} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append((code, lab, name))
    extra = sorted(set(ke) - set(KE_GFS_REGION))
    if bad or extra:
        raise SystemExit(f"Kenya's REGION1_Y1 codes do not match the alphabetical reading: "
                         f"{bad} extra {extra}")
    print("  every Kenyan code matches its county")


def build_ke():
    from scipy.stats import spearmanr

    print("\n=== Kenya ===")
    base = pd.read_csv(os.path.join(NORMDIR, "ke.csv"), dtype={"geo_id": str},
                       keep_default_na=False, na_values=[""])
    base = base[base["geo_level"] == "county"].copy()
    prot = base[base["source_category"] == "Protestant"].set_index("geo_name")
    tot = base[base["source_category"] == "Total"].set_index("geo_name")["count"].astype(float)
    if len(prot) != 47 or set(prot.index) != set(KE_ALPHA):
        raise SystemExit(f"ke.csv counties do not match KE_ALPHA: "
                         f"{sorted(set(prot.index) ^ set(KE_ALPHA))}")

    g = pd.read_csv(GFS_CSV, usecols=["COUNTRY", "REGION1_Y1", "REL2_Y1", "REL3_Y1",
                                      "ANNUAL_WEIGHT_C1"], low_memory=False)
    g = g.apply(pd.to_numeric, errors="coerce")
    g = g[g["COUNTRY"] == KE_GFS_COUNTRY].copy()
    if len(g) != KE_GFS_N:
        raise SystemExit(f"GFS Kenya has {len(g):,} rows, written against {KE_GFS_N:,}")
    if int((g["REL2_Y1"] == 1).sum()) != KE_GFS_CHRISTIANS:
        raise SystemExit("GFS Kenya's Christian count has changed")
    stray = sorted(set(g["REGION1_Y1"].dropna().astype(int)) - set(KE_GFS_REGION))
    if stray:
        raise SystemExit(f"GFS REGION1_Y1 codes with no county: {stray}")
    g["county"] = g["REGION1_Y1"].astype(int).map(KE_GFS_REGION)
    g["w"] = g["ANNUAL_WEIGHT_C1"]
    code = dict(zip(base.drop_duplicates("geo_name")["geo_name"],
                    base.drop_duplicates("geo_name")["geo_id"]))
    g["province"] = g["county"].map(lambda c: ke_province(code[c]))
    # THE COUNTY READING, CHECKED WITHOUT THE .sav. GFS also carries REGION2_Y1, the 8 former
    # provinces. Under the alphabetical reading every sampled county must fall in exactly the
    # province KNBS's own code puts it in; 42 counties nesting into 8 groups by chance is not a
    # thing that happens. `--labels` is the direct check against the value labels (OSF was
    # unreachable on 2026-10-16, so this was the check the build first ran on).
    g2 = pd.read_csv(GFS_CSV, usecols=["COUNTRY", "REGION2_Y1"], low_memory=False)
    g2 = pd.to_numeric(g2.loc[pd.to_numeric(g2["COUNTRY"], errors="coerce") == KE_GFS_COUNTRY,
                              "REGION2_Y1"], errors="coerce")
    nest = pd.crosstab(g["province"], g2.reindex(g.index))
    if not ((nest > 0).sum(axis=1) == 1).all() or not ((nest > 0).sum(axis=0) == 1).all():
        raise SystemExit(f"GFS counties do not nest in GFS provinces under the alphabetical "
                         f"reading:\n{nest}")
    print(f"  {g['county'].nunique()} sampled counties nest exactly in the 8 GFS provinces "
          "under the alphabetical reading")

    p = g[(g["REL2_Y1"] == 1) & g["REL3_Y1"].isin(list(KE_REL3))].copy()
    p["c"] = p["REL3_Y1"].astype(int).map(KE_REL3)
    p.loc[p["county"].isin(KE_FRIENDS) & (p["c"] == "Presbyterian"), "c"] = "Protestant"
    cats = KE_DRAWN + ["Protestant"]

    # 1. Level: the GFS pool against the census column, both weighted by county population.
    pool = g.groupby("county").apply(
        lambda d: d.loc[(d["REL2_Y1"] == 1) & d["REL3_Y1"].isin(list(KE_REL3)), "w"].sum()
        / d["w"].sum(), include_groups=False)
    have = pool.index
    cen = (prot["count"].astype(float) / tot)
    lv_g = float((pool * tot[have]).sum() / tot[have].sum() * 100)
    lv_c = float((cen[have] * tot[have]).sum() / tot[have].sum() * 100)
    print(f"  level over the {len(have)} sampled counties: census Protestant {lv_c:.1f}%, "
          f"GFS pool {lv_g:.1f}%")
    if abs(lv_g - lv_c) > KE_LEVEL_GAP_MAX:
        raise SystemExit("the GFS pool no longer matches the census Protestant level")
    n_pool = p.groupby("county").size()
    big = n_pool[n_pool >= 30].index
    rho = spearmanr(cen[big], pool[big]).correlation
    print(f"  county rank, census Protestant share against the GFS pool ({len(big)} counties "
          f"with 30+): Spearman {rho:+.3f}")

    # 2. Split-half: each drawn church as a share of the pool, county by county.
    rng = np.random.default_rng(0)
    keys = n_pool[n_pool >= KE_MIN_N].index
    for ch in KE_DRAWN:
        rs = []
        for _ in range(200):
            half = rng.random(len(p)) < 0.5
            a, b = (p[h].groupby("county").apply(
                lambda d: d.loc[d["c"] == ch, "w"].sum() / d["w"].sum(), include_groups=False)
                .reindex(keys) for h in (half, ~half))
            rs.append(spearmanr(a, b, nan_policy="omit").correlation)
        med = float(np.median(rs))
        print(f"  split-half {ch:22s} median Spearman {med:+.2f} over {len(keys)} counties")
        if med < KE_SPLIT_HALF_MIN:
            raise SystemExit(f"{ch} no longer replicates across counties")

    # 3. Shares: county pool if 40+ respondents, else province, else nation.
    def mix(d):
        t = d.groupby("c")["w"].sum().reindex(cats, fill_value=0.0)
        return t / t.sum()

    nat = mix(p)
    prov_mix = {pr: mix(d) for pr, d in p.groupby("province")}
    prov_n = p.groupby("province").size()
    rows, shares = [], {}
    for county in KE_ALPHA:
        n = int(n_pool.get(county, 0))
        pr = ke_province(code[county])
        if n >= KE_MIN_N:
            sh, used = mix(p[p["county"] == county]), "county"
        elif prov_n.get(pr, 0) >= KE_MIN_N:
            sh, used = prov_mix[pr].copy(), f"province {pr}"
        else:
            sh, used = nat.copy(), "national"
        if county in KE_FRIENDS and sh["Presbyterian"] > 0:
            sh["Protestant"] += sh["Presbyterian"]
            sh["Presbyterian"] = 0.0
        shares[code[county]] = {c: float(sh[c]) for c in cats}
        rows.append({"county": county, "n": n, "from": used, **{c: 100 * sh[c] for c in cats}})
    tab = pd.DataFrame(rows).set_index("county")
    print("  shares of each county's Protestants as drawn (%):")
    print(tab.round(1).to_string())
    _raw("ke_gfs_county_shares.csv", tab)

    labels = {"Anglican": "Protestant: Anglican", "Seventh-day Adventist":
              "Protestant: Seventh-day Adventist", "Presbyterian": "Protestant: Presbyterian",
              "Baptist": "Protestant: Baptist", "Methodist": "Protestant: Methodist",
              "Protestant": "Protestant"}
    shares = {gid: {labels[c]: v for c, v in sh.items()} for gid, sh in shares.items()}
    out = split_rows(base, "Protestant", shares, "ke_gfs2023_protestant_split", "2023",
                     "GFS wave 1 (2023) REL3, share of the county's Protestant-cell churches")
    write("ke", out)


# ---------------------------------------------------------------------------------------------
# Malawi

MW_CELLS = ("Seventh Day Adventist/Baptist/Apostolic", "Other Christian Denominations")
MW_MIN_N = 100              # respondents in a cell in a region; below it, national shares
MW_LEVEL_GAP_MAX = 5.0      # points: the Afrobarometer's cell level against the census's
MW_LATE = (7, 8, 9)         # the rounds whose ratio splits independents from `Christian only`
# Afrobarometer answer (lower-case, before any bracket) -> (census cell, group)
MW_AB = {
    "seventh day adventist": (0, "Seventh-day Adventist"), "baptist": (0, "Baptist"),
    "apostolic church": (0, "Apostolic"), "apostolic faith/new_united": (0, "Apostolic"),
    "church of christ": (1, "Church of Christ"), "jehovah's witness": (1, "Jehovah's Witnesses"),
    "evangelical": (1, "Evangelical"),
    "independent": (1, "Independent"), "zionist christian church": (1, "Independent"),
    "last church": (1, "Independent"), "last church of god": (1, "Independent"),
    "last church/reform": (1, "Independent"), "african abraham church": (1, "Independent"),
    "african abraham": (1, "Independent"), "abraham church": (1, "Independent"),
    "african international": (1, "Independent"), "utopia church": (1, "Independent"),
    "nationality": (1, "Independent"),
    "christian only": (1, "Christian only"),
    "lutheran": (1, "Other"), "new apostolic church": (1, "Other"), "methodist": (1, "Other"),
    "orthodox": (1, "Other"), "coptic": (1, "Other"), "mennonite": (1, "Other"),
    "covenant church": (1, "Other"), "emmanuel": (1, "Other"), "bible believers": (1, "Other"),
    "quaker / friends": (1, "Other"),
}
# Answers in other census cells, listed so an unknown answer stops the build.
MW_ELSEWHERE = {"roman catholic", "presbyterian", "dutch reformed", "calvinist", "anglican",
                "pentecostal", "muslim only", "sunni only", "shia", "shia only", "ismaeli",
                "qadiriya brotherhood", "sukuti", "traditional / ethnic religion",
                "traditional/ethnic religion", "agnostic", "atheist", "bahai", "hindu", "jewish",
                "other", "none"}
MW_LABEL = {
    "Seventh-day Adventist": "Seventh Day Adventist/Baptist/Apostolic: Seventh-day Adventist",
    "Baptist": "Seventh Day Adventist/Baptist/Apostolic: Baptist",
    "Apostolic": "Seventh Day Adventist/Baptist/Apostolic",
    "Independent": "Other Christian Denominations: independent churches",
    "Church of Christ": "Other Christian Denominations: Church of Christ",
    "Jehovah's Witnesses": "Other Christian Denominations: Jehovah's Witnesses",
    "Evangelical": "Other Christian Denominations: Evangelical",
    "Christian only": "Other Christian Denominations",
    "Other": "Other Christian Denominations",
}


def mw_answer(label):
    s = "".join(ch if ord(ch) < 128 else "'" for ch in str(label))
    return s.split("(")[0].strip().lower()


def build_mw():
    import afrobarometer as ab

    print("\n=== Malawi ===")
    base = pd.read_csv(os.path.join(NORMDIR, "mw.csv"), dtype={"geo_id": str},
                       keep_default_na=False, na_values=[""])
    base = base[base["geo_level"] == "district"].copy()
    base["region"] = base["geo_id"].str[:3].map({"MW1": "North", "MW2": "Central",
                                                 "MW3": "South"})
    if base["region"].isna().any():
        raise SystemExit("a district geo_id outside MW1-MW3")

    d = ab.load("Malawi", expect_rounds=[4, 5, 6, 7, 8, 9], regroup=True)
    d["a"] = d["category"].map(mw_answer)
    unknown = sorted(set(d["a"]) - set(MW_AB) - MW_ELSEWHERE)
    if unknown:
        raise SystemExit(f"Afrobarometer answers with no census cell: {unknown}")
    d["region"] = d["geo_raw"].astype(str).str[0].map({"N": "North", "C": "Central",
                                                       "S": "South"})
    if d["region"].isna().any():
        raise SystemExit(f"REGION labels with no region: {sorted(set(d.loc[d.region.isna(), 'geo_raw']))}")
    d["cell"] = d["a"].map(lambda a: MW_AB[a][0] if a in MW_AB else None)
    d["grp"] = d["a"].map(lambda a: MW_AB[a][1] if a in MW_AB else None)

    # 1. Level: each cell's share of respondents against the census, nationally and by region.
    tot = base[base["source_category"] == "Total"]
    for i, cell in enumerate(MW_CELLS):
        cen = base[base["source_category"] == cell]
        cen_nat = 100 * cen["count"].sum() / tot["count"].sum()
        ab_nat = 100 * d.loc[d["cell"] == i, "w"].sum() / d["w"].sum()
        by_r = []
        for r in ("North", "Central", "South"):
            c = 100 * cen.loc[cen.region == r, "count"].sum() / tot.loc[
                tot["geo_id"].str[:3] == {"North": "MW1", "Central": "MW2", "South": "MW3"}[r],
                "count"].sum()
            dr = d[d["region"] == r]
            a = 100 * dr.loc[dr["cell"] == i, "w"].sum() / dr["w"].sum()
            by_r.append(f"{r} {c:.1f}/{a:.1f}")
        print(f"  `{cell}`: census {cen_nat:.1f}%, Afrobarometer {ab_nat:.1f}% of respondents; "
              f"by region census/survey {', '.join(by_r)}")
        if abs(cen_nat - ab_nat) > MW_LEVEL_GAP_MAX:
            raise SystemExit(f"`{cell}`: the survey's level is off the census by more than "
                             f"{MW_LEVEL_GAP_MAX} points")

    # 2. Independents and `Christian only` trade places between rounds: shown, then joined.
    oc = d[d["cell"] == 1]
    by_round = oc.groupby(["round", "grp"])["w"].sum().unstack(fill_value=0.0)
    by_round = 100 * by_round.div(by_round.sum(axis=1), axis=0)
    print("  inside `Other Christian Denominations`, % by round:")
    print(by_round.round(1).to_string())
    r = float(np.corrcoef(by_round["Independent"], by_round["Christian only"])[0, 1])
    print(f"  independents against `Christian only` across rounds: r {r:+.2f}")
    late = oc[oc["round"].isin(MW_LATE)]
    li = late.loc[late["grp"] == "Independent", "w"].sum()
    lc = late.loc[late["grp"] == "Christian only", "w"].sum()
    ind_ratio = float(li / (li + lc))
    print(f"  late rounds R7-R9: independents are {100 * ind_ratio:.1f}% of the joined box")
    d["grp2"] = d["grp"].replace({"Independent": "JOINED", "Christian only": "JOINED"})

    # 3. Shares per region inside each cell.
    shares = {}
    tabs = []
    for i, cell in enumerate(MW_CELLS):
        x = d[d["cell"] == i]
        groups = sorted(set(x["grp2"]))

        def mix(y):
            t = y.groupby("grp2")["w"].sum().reindex(groups, fill_value=0.0)
            return t / t.sum()

        nat = mix(x)
        for reg in ("North", "Central", "South"):
            y = x[x["region"] == reg]
            sh = mix(y) if len(y) >= MW_MIN_N else nat
            out = {}
            for k, v in sh.items():
                if k == "JOINED":
                    out["Independent"] = v * ind_ratio
                    out["Christian only"] = v * (1 - ind_ratio)
                else:
                    out[k] = v
            lab = {}
            for k, v in out.items():
                lab[MW_LABEL[k]] = lab.get(MW_LABEL[k], 0.0) + float(v)
            for gid in base.loc[(base.region == reg) & (base.source_category == cell), "geo_id"]:
                shares.setdefault(cell, {})[gid] = lab
            tabs.append(pd.Series({**{k: 100 * v for k, v in out.items()}, "n": len(y),
                                   "from": "region" if len(y) >= MW_MIN_N else "national"},
                                  name=f"{cell[:24]} | {reg}"))
    tab = pd.DataFrame(tabs)
    print("  shares of each cell as drawn (%), by region:")
    print(tab.to_string())
    _raw("mw_afrobarometer_region_shares.csv", tab)

    parts = [split_rows(base, cell, shares[cell], "mw_afrobarometer_2008_2022_split",
                        "2008-2022", "Afrobarometer R4-R9 by region, share of the census cell")
             for cell in MW_CELLS]
    write("mw", pd.concat(parts, ignore_index=True))


def main():
    args = sys.argv[1:]
    if "--labels" in args:
        fetch_labels()
        return
    for cc in args or ["ke", "mw"]:
        {"ke": build_ke, "mw": build_mw}[cc]()


if __name__ == "__main__":
    main()
