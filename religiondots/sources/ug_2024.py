"""Uganda: the 2024 census, read from UBOS's 10% population microdata sample, drawn on subcounties.

Writes data/normalized/ug.csv. The 2002 build's file is data/normalized/ug2002.csv from
2026-09-15 (sources/ug.py), the way Pakistan's 2017 file became pk2017.csv: the drawn vintage
owns the plain name.

WHAT THE FILE IS. Anita registered with UBOS and downloaded `NPHC 2024-Users  File using
cpro_extract_Population_record_data.rar` (ask 008). The one member is a Stata 118 file of
4,693,190 person records, 10.22% of the 45,905,417 counted, with no weight variable. Every
record carries district, county, subcounty and parish codes, and the county, subcounty and
parish names; `HH_P9` is religion in twelve codes. Every household record answers it and every
non-household record (152,385, 3.25%) is blank. Two streaming passes over the RAR keep only
counts (data/raw/ug/nphc2024_aggregate_script.py and nphc2024_waves_script.py); no person or
household record is on disk.

IT IS A FLAT TEN PERCENT OF HOUSEHOLDS, CHECKED PER UNIT. Joined by name to the census's own
full-count workbook (`NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx`, Table 2 household
population), every one of the 2,207 subcounties has a sampling fraction between 8.2% and 12.5%
and every one of the 10,852 sample parishes joins. So shares come from the sample and people
from the full count, unit by unit, and no weight is needed.

THE GRAIN IS THE STABILITY TEST'S, NOT THE FILE'S. The sample's unit is the household, so
households are dealt into six waves (a hash order inside each parish, round-robin) and the test
halves the waves, never persons. District against the nation is the shared construction
(`stability.median_rho`, `wave_null`, `permutation_p`, `chi2_p`). Each finer tier is tested
against its PARENT: each unit's share minus its parent's share in the same half, correlated
between the halves across units, median over the ten halvings, with a null that shuffles units
within their parent separately in every wave. Significance is not enough at 2,207 units, where
almost anything passes; a tier is taken for a category only if the median half-sample Pearson
correlation of those departures is at least 1/3. At 1/3 the full sample's share for the unit
has the same expected squared error as its parent's share (Spearman-Brown: reliability 1/2), so
above it the finer figure is the better estimate and below it the parent's is.

Result (EXPECT_TIER, asserted): Catholic, Anglican, SDA, Islam, Pentecostal/Evangelical,
Traditional, No Religion and Others pass to parish; Orthodox and Jehovah's Witnesses to county;
Bahai and Buddhist to district. The drawn unit is the SUBCOUNTY because no 2024 parish polygon
is published anywhere (sources/ug_2024_geo.py lists what was asked).

THE COMPOSITION. Inside each subcounty, Orthodox and Witnesses take their county's share and
Bahai and Buddhist their district's (Sweden's "each category at its own level"), and the eight
subcounty categories share the rest in the proportions the subcounty's own sample gives them.
Nothing can go negative and every row closes.

THREE UNITS ARE MERGED. Yumbe's Bidi Bidi refugee settlement is three census subcounties (one in
each of three counties) with no polygon of their own on the portal; each is drawn together with
the host subcounties of its own county whose Kontur population exceeds the census household
count by more than 1.2 times (CAMP_HOSTS, checked in sources/ug_2024_geo.py). The Lobule
refugee camp polygon in Koboko has no population row and joins Lobule subcounty's polygon.

WHO IS NOT DRAWN: the 1,517,205 people counted outside households in the 146 districts, whose
sample records carry no religion, and Apaa, 9,456 people the census counts as a unit of its own
(the disputed land on the Adjumani and Amuru border), which has no subcounty and no sample
record. 1,526,661 people, 3.33%.

Usage:
    python sources/ug_2024.py            rebuild from data/raw/ug/ (a few minutes)
    python sources/ug_2024.py --fetch    fetch the portal files first (sources/ug_2024_geo.py)
    python sources/ug_2024.py --parish   also test the parish tier (about ten more minutes)
"""
import io
import json
import os
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")  # [[reference_scipy_eats_all_cores]]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stability as st  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "ug")
AGG = os.path.join(RAW, "nphc2024_aggregate")
PORTAL = os.path.join(RAW, "nphc2024_portal")
XLSX = os.path.join(PORTAL, "NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx")
OUT = os.path.join(ROOT, "data", "normalized", "ug.csv")
LEVEL = "subcounty_2024"
SOURCE_ID = "ug_nphc_2024_10pct_sample"

NATIONAL = dict(total=45_905_417, hhpop=44_387_526, households=10_698_913)
SAMPLE_N = 4_693_190
APAA = dict(name="APAA", total=9_456, hhpop=8_770, households=2_551)
EXPECT_UNITS = dict(district=146, county=312, subcounty=2_207, parish=10_854)
EXPECT_SAMPLE_PARISHES = 10_852
# Parishes in the workbook with no household in the sample: household population, printed.
WORKBOOK_ONLY_PARISHES = {
    ("KARENGA", "NAPORE WEST COUNTY", "KIDEPO TOWN COUNCIL", "KIDEPO WARD"): 6,
    ("KWEEN", "SOI COUNTY", "GREEK RIVER (KIRIKI)", "ALALAM"): 31,
}
FRACTION_BAND = (0.08, 0.13)
# The portal's subcounty population rows: five Karamoja subcounties read `#N/A` there and have
# figures in the workbook, and one row has no workbook subcounty and no sample record.
PORTAL_NA = {"311101", "311102", "311103", "311104", "311105"}
PORTAL_EXTRA = {"319107": "LOBULE REFUGEE CAMP"}
# Subcounties where a household's records carry two parish codes (district, county, subcounty).
PARISH_SPLIT_SUBCOUNTIES = [(417, 2, 3), (419, 2, 12)]
# Bidi Bidi's three census subcounties have no polygon; each is drawn with its county's hosts.
CAMP_HOSTS = {
    "UG3130106": ["UG3130102"],                            # county 1: Kululu
    "UG3130207": ["UG3130203", "UG3130205", "UG3130206"],  # county 2: Barakala TC, Lori, Romogi
    "UG3130409": ["UG3130402", "UG3130403", "UG3130408"],  # county 4: Ariwa, Drajini, Odravu West
}
HOST_RATIO = 1.2

RELS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 96]
TIERS = [("district", ["district"]), ("county", ["district", "county"]),
         ("subcounty", ["district", "county", "subcounty"]),
         ("parish", ["district", "county", "subcounty", "parish"])]
BAR = 1.0 / 3.0
EXPECT_TIER = {11: "parish", 12: "parish", 13: "parish", 14: "parish", 15: "parish",
               16: "county", 17: "district", 18: "district", 19: "county", 20: "parish",
               21: "parish", 96: "parish"}
# Final Report Volume 1, Table 3.1, as transcribed in sources.md §11b (rounded there).
REPORT_2024 = {11: 16_600_000, 12: 13_300_000, 15: 6_500_000, 14: 6_100_000, 13: 911_000,
               16: 65_000, 19: 46_000, 20: 56_000, 21: 85_559}


def norm(x):
    return " ".join(unicodedata.normalize("NFC", str(x)).replace(" ", " ").upper().split())


def say(ok, msg, failures):
    print(f"  {'ok ' if ok else 'BAD'} {msg}")
    if not ok:
        failures.append(msg)


def unit_id(d, c, s):
    return f"UG{int(d):03d}{int(c):02d}{int(s):02d}"


def drawn_unit(uid):
    """The unit a census subcounty is drawn as: itself, or its Bidi Bidi merge."""
    for camp, hosts in CAMP_HOSTS.items():
        if uid == camp or uid in hosts:
            return hosts[0] + "BB"
    return uid


# ---------------------------------------------------------------------------------------------
# the labels, the full-count workbook, the portal
# ---------------------------------------------------------------------------------------------

def religion_labels():
    """HH_P9's value labels, verbatim, from the label block the scout's pass decoded."""
    lab, on = {}, False
    with open(os.path.join(AGG, "ug2024_value_labels.txt"), encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("## "):
                on = line == "## HH_P9_VS1"
                continue
            if on and line:
                code, text = line.split("\t", 1)
                lab[int(code)] = unicodedata.normalize("NFC", text.strip())
    if sorted(lab) != RELS:
        raise SystemExit(f"HH_P9_VS1 codes {sorted(lab)}, expected {RELS}")
    return lab


def _workbook():
    import openpyxl

    src = zipfile.ZipFile(XLSX)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for it in src.infolist():
            if it.filename != "xl/styles.xml":  # [[reference_openpyxl_stylesheet]]
                dst.writestr(it, src.read(it.filename))
    buf.seek(0)
    return openpyxl.load_workbook(buf, read_only=True, data_only=True)


def profiles():
    """Tables 1 and 2 as parish rows, by sum closure: each parent row is the sum of its children.

    The indentation that shows the levels is cell style, which a value read does not see, so
    the tree is rebuilt from the totals: district = its counties = their subcounties = their
    parishes, in document order. Apaa is one row after the last district, with no children.
    """
    wb = _workbook()

    def rows(name):
        return [r for i, r in enumerate(wb[name].iter_rows(values_only=True))
                if i >= 6 and r[0] is not None]

    t1, t2 = rows("Table1"), rows("Table2")
    if len(t1) != len(t2):
        raise SystemExit(f"Table1 has {len(t1)} rows and Table2 {len(t2)}")
    recs = []
    for a, b in zip(t1, t2):
        if norm(a[0]) != norm(b[0]):
            raise SystemExit(f"Table1 {a[0]!r} beside Table2 {b[0]!r}")
        vals = (a[1], a[2], a[3], b[1], b[2])
        if not all(isinstance(v, (int, float)) and float(v).is_integer() for v in vals):
            raise SystemExit(f"a non-integer count on {a[0]!r}: {vals}")
        recs.append(dict(name=str(a[0]).strip(), male=int(a[1]), female=int(a[2]),
                         total=int(a[3]), hhpop=int(b[1]), households=int(b[2])))
    national = recs.pop()
    if norm(national["name"]) != "NATIONAL":
        raise SystemExit(f"last row is {national['name']!r}, not National")
    apaa = recs.pop()
    if norm(apaa["name"]) != APAA["name"] or any(apaa[k] != APAA[k]
                                                  for k in ("total", "hhpop", "households")):
        raise SystemExit(f"the row before National is {apaa}, expected Apaa {APAA}")

    pos = 0
    out, lv = [], {"district": [], "county": [], "subcounty": []}

    def take():
        nonlocal pos
        if pos >= len(recs):
            raise SystemExit("the workbook's sums ran off the end: a unit with no children")
        pos += 1
        return recs[pos - 1]

    while pos < len(recs):
        d = take()
        lv["district"].append(d)
        cs = 0
        while cs < d["total"]:
            c = take()
            cs += c["total"]
            lv["county"].append(dict(c, district=d["name"]))
            ss = 0
            while ss < c["total"]:
                s = take()
                ss += s["total"]
                lv["subcounty"].append(dict(s, district=d["name"], county=c["name"]))
                ps = 0
                while ps < s["total"]:
                    p = take()
                    ps += p["total"]
                    out.append(dict(district=d["name"], county=c["name"],
                                    subcounty=s["name"], parish=p["name"], total=p["total"],
                                    hhpop=p["hhpop"], households=p["households"]))
                if ps != s["total"]:
                    raise SystemExit(f"parishes sum to {ps} under {s}")
            if ss != c["total"]:
                raise SystemExit(f"subcounties sum to {ss} under {c}")
        if cs != d["total"]:
            raise SystemExit(f"counties sum to {cs} under {d}")
    return pd.DataFrame(out), lv, national, apaa


def district_names():
    raw = open(os.path.join(PORTAL, "district.json"), "rb").read()
    d = json.loads(raw[raw.find(b"{"):])
    return {int(r["code"]): norm(r["name"]) for r in d["population_data"]}


def portal_subcounties():
    raw = open(os.path.join(PORTAL, "counties_with_subcounty_populations.json"), "rb").read()
    rows = json.loads(raw[raw.find(b"{"):])["population_data"]
    out = []
    for r in rows:
        dc, cc, code = r["district_code"], r["county_code"], r["code"]
        if not (cc.startswith(dc) and code.startswith(cc)):
            raise SystemExit(f"portal code {code} is not district {dc} + county {cc} + subcounty")
        tot = str(r["total_popn"])
        out.append(dict(code=code, district=int(dc), county=int(cc[len(dc):]),
                        subcounty=int(code[len(cc):]), name=norm(r["name"]),
                        total=int(tot) if tot.isdigit() else None))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# the sample and the join
# ---------------------------------------------------------------------------------------------

def load_sample(failures):
    """The scout's parish x religion counts, the waves pass, and their agreement."""
    smp = pd.read_csv(os.path.join(AGG, "ug2024_parish_religion.csv"), keep_default_na=False)
    say(int(smp["persons"].sum()) == SAMPLE_N,
        f"the scout's pass holds {int(smp['persons'].sum()):,} persons, N {SAMPLE_N:,}", failures)
    W = pd.read_csv(os.path.join(AGG, "ug2024_parish_wave_religion.csv"))
    geo = ["district", "county", "subcounty", "parish"]
    a = (smp[smp["qrtype"] == 1].astype({"religion": "int64"})
         .groupby(geo + ["religion"])["persons"].sum().rename("scout"))
    b = W.astype({"religion": "int64"}).groupby(geo + ["religion"])["persons"].sum().rename("waves")
    both = pd.concat([a, b], axis=1).fillna(0)
    bad = both[both["scout"] != both["waves"]]
    # The waves pass puts a whole household in the parish of its first record; a few households'
    # records carry two parish codes inside one subcounty, so parish cells move and no more.
    moved = sorted({k[:3] for k in bad.index})
    sub = both.groupby(level=[0, 1, 2, 4]).sum()
    say(bool((sub["scout"] == sub["waves"]).all()) and moved == PARISH_SPLIT_SUBCOUNTIES,
        f"the two passes over the RAR agree on every subcounty x religion household count "
        f"({int(b.sum()):,} persons); {len(bad)} parish cells differ, inside {moved}", failures)
    say(set(smp.loc[smp["qrtype"] == 2, "religion"].astype(str)) == {"-9223372036854775808"},
        "every non-household record has a blank religion", failures)
    g = (smp.groupby(geo + ["county_name", "subcounty_name", "parish_name", "qrtype"])["persons"]
         .sum().unstack("qrtype", fill_value=0).rename(columns={1: "hh", 2: "nonhh"})
         .reset_index())
    return g, W


def join(g, P, lv, dnames, failures):
    g = g.copy()
    g["district_name"] = g["district"].map(dnames)
    say(g["district_name"].notna().all(), "every sample district code has a portal name", failures)
    for c in ("county_name", "subcounty_name", "parish_name"):
        g[c] = g[c].map(norm)
    # the workbook's printed names would collide with the sample's code columns in the merge
    P = P.rename(columns={c: c + "_printed" for c in ("district", "county", "subcounty", "parish")})
    for c in ("district", "county", "subcounty", "parish"):
        P[c + "_n"] = P[c + "_printed"].map(norm)
    ks = ["district_name", "county_name", "subcounty_name", "parish_name"]
    kw = ["district_n", "county_n", "subcounty_n", "parish_n"]
    say(not g.duplicated(ks).any() and not P.duplicated(kw).any(),
        "no repeated name key inside a subcounty on either side", failures)
    m = g.merge(P, left_on=ks, right_on=kw, how="outer", indicator=True)
    lo = m[m["_merge"] == "left_only"]
    ro = {tuple(r[kw]): int(r["hhpop"]) for _, r in m[m["_merge"] == "right_only"].iterrows()}
    say(len(g) == EXPECT_SAMPLE_PARISHES and lo.empty,
        f"all {len(g):,} sample parishes join the workbook by name", failures)
    say(ro == WORKBOOK_ONLY_PARISHES,
        f"workbook parishes with no sample household are the pinned two: {ro}", failures)
    J = m[m["_merge"] == "both"].drop(columns="_merge")
    # the witness neither key determines: a wrong twin shows as a sampling fraction far off 10%
    sc = J.groupby(["district", "county", "subcounty"])[["hh", "hhpop"]].sum()
    frac = sc["hh"] / sc["hhpop"]
    lo_f, hi_f = float(frac.min()), float(frac.max())
    say(len(sc) == EXPECT_UNITS["subcounty"] and FRACTION_BAND[0] <= lo_f and hi_f <= FRACTION_BAND[1],
        f"{len(sc):,} subcounties, sampling fraction {lo_f:.4f} to {hi_f:.4f} "
        f"(median {frac.median():.4f}), inside {FRACTION_BAND}", failures)
    pj = J[J["hhpop"] >= 1000]
    pf = pj["hh"] / pj["hhpop"]
    print(f"     parishes of 1,000+ in households: fraction {pf.min():.4f} to {pf.max():.4f}")
    return J


def subcounty_table(J, P):
    """One row per census subcounty: codes, unit, drawn unit, names, full-count populations.

    Populations come from every workbook parish, not only the joined ones: two parishes have no
    sample household (WORKBOOK_ONLY_PARISHES) and their people still live in a subcounty.
    """
    S = (J.groupby(["district", "county", "subcounty"])
         .agg(district_name=("district_name", "first"), county_name=("county_name", "first"),
              subcounty_name=("subcounty_name", "first")).reset_index())
    pop = (P.assign(district_name=P["district"].map(norm), county_name=P["county"].map(norm),
                    subcounty_name=P["subcounty"].map(norm))
           .groupby(["district_name", "county_name", "subcounty_name"])[["hhpop", "total"]].sum())
    S = S.merge(pop, left_on=["district_name", "county_name", "subcounty_name"],
                right_index=True, how="left")
    if S["hhpop"].isna().any() or len(pop) != len(S):
        raise SystemExit("workbook subcounties and sample subcounties do not pair one to one")
    S["unit"] = [unit_id(*k) for k in zip(S["district"], S["county"], S["subcounty"])]
    S["drawn"] = S["unit"].map(drawn_unit)
    return S


# ---------------------------------------------------------------------------------------------
# the stability test
# ---------------------------------------------------------------------------------------------

def _cube(W, keys):
    g = W.groupby(keys + ["w6", "religion"])["persons"].sum()
    wide = g.unstack(["w6", "religion"], fill_value=0)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([range(6), RELS]), fill_value=0)
    cube = wide.to_numpy(dtype=float).reshape(len(wide), 6, len(RELS)).transpose(1, 0, 2)
    return wide.index, cube


def _pearson_cols(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    sx, sy = np.sqrt((x ** 2).sum(axis=0)), np.sqrt((y ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((sx > 0) & (sy > 0), (x * y).sum(axis=0) / (sx * sy), np.nan)


def _spearman_cols(x, y):
    from scipy.stats import rankdata

    return _pearson_cols(rankdata(x, axis=0), rankdata(y, axis=0))


def _departures(cube, pidx, n_par, a, b):
    out = []
    for h in (a, b):
        u = cube[list(h)].sum(axis=0)
        par = np.zeros((n_par, u.shape[1]))
        np.add.at(par, pidx, u)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(u / u.sum(axis=1, keepdims=True) - (par / par.sum(axis=1, keepdims=True))[pidx])
    return out


def _stat(cube, pidx, n_par, keep, splits):
    ps, ss = [], []
    for a, b in splits:
        da, db = _departures(cube, pidx, n_par, a, b)
        ps.append(_pearson_cols(da[keep], db[keep]))
        ss.append(_spearman_cols(da[keep], db[keep]))
    return np.nanmedian(np.array(ps), axis=0), np.nanmedian(np.array(ss), axis=0)


def stability(W, lab, tiers, failures):
    """Each religion code at each tier against its parent. Returns the verdict table."""
    W = W.assign(w6=W["wave"] // 2)
    splits = st.halvings(6)
    rows, prev = [], None
    for tier, keys in tiers:
        index, cube = _cube(W, keys)
        n_u = len(index)
        if tier == "district":
            pidx, n_par = np.zeros(n_u, dtype=int), 1
        else:
            pos = {k: i for i, k in enumerate(prev)}
            parent = [k[0] if len(k) == 2 else tuple(k[:-1]) for k in index]
            pidx = np.array([pos[p] for p in parent])
            n_par = len(prev)
        full = (cube.sum(axis=2) > 0).all(axis=0)
        sibs = np.bincount(pidx[full], minlength=n_par)
        keep = full if tier == "district" else full & (sibs[pidx] >= 2)
        print(f"\n  {tier}: {n_u:,} units; {int((~full).sum()):,} with an empty wave and "
              f"{int((full & ~keep).sum()):,} only children left out; {int(keep.sum()):,} tested")
        obs_p, obs_s = _stat(cube, pidx, n_par, keep, splits)
        if tier == "district":
            shared = st.median_rho(cube[:, keep], splits)
            say(float(np.nanmax(np.abs(shared - obs_s))) <= 1e-9,
                "the vectorised Spearman equals stability.median_rho at district", failures)
        rng = np.random.default_rng(st.STAB_SEED)
        kidx = np.flatnonzero(keep)
        kp = pidx[kidx]
        base = kidx[np.lexsort((kidx, kp))]
        null_p = np.full((st.STAB_PERM, len(RELS)), np.nan)
        null_s = np.full((st.STAB_PERM, len(RELS)), np.nan)
        for i in range(st.STAB_PERM):
            pc = cube.copy()
            for w in range(6):
                pc[w, base] = cube[w, kidx[np.lexsort((rng.random(len(kidx)), kp))]]
            null_p[i], null_s[i] = _stat(pc, pidx, n_par, keep, splits)
        tot = cube.sum(axis=0)
        for j, r in enumerate(RELS):
            p_s, q_s = st.permutation_p(obs_s[j], null_s[:, j])
            p_p, q_p = st.permutation_p(obs_p[j], null_p[:, j])
            chi = st.chi2_p(tot[keep, j], tot[keep].sum(axis=1)) if tier == "district" else np.nan
            ok = bool(p_s < st.STAB_ALPHA and obs_p[j] >= BAR
                      and (tier != "district" or chi < st.STAB_ALPHA))
            rows.append(dict(tier=tier, code=r, label=lab[r], pearson=obs_p[j],
                             null95_pearson=q_p, p_pearson=p_p, spearman=obs_s[j],
                             null95_spearman=q_s, p_spearman=p_s, chi2=chi, passed=ok))
            print(f"    {lab[r][:28]:<28} pearson {obs_p[j]:+.3f} (null95 {q_p:+.3f})  "
                  f"spearman {obs_s[j]:+.3f} (null95 {q_s:+.3f}, p {p_s:.4f})"
                  + (f"  chi2 {chi:.1e}" if tier == "district" else "")
                  + ("  PASS" if ok else ""))
        prev = index
    return pd.DataFrame(rows)


def cell_cap(lab, failures):
    """CELL_CAP on the sampling cell, the household: no one household holds half of an answer."""
    hm = pd.read_csv(os.path.join(AGG, "ug2024_parish_hhmax.csv"))
    worst = hm.groupby("religion").agg(big=("max_in_one_household", "max"), n=("persons", "sum"))
    share = worst["big"] / worst["n"]
    say(bool((share <= st.CELL_CAP).all()),
        "no household holds more than half of any answer's sample persons (largest: "
        + ", ".join(f"{lab[r].split(' /')[0]} {share[r]:.4f}" for r in share.sort_values().index[-3:])
        + ")", failures)


# ---------------------------------------------------------------------------------------------
# composition, scaling, output
# ---------------------------------------------------------------------------------------------

def compose(W, tier_of, S):
    """Per drawn unit: coarse categories at their parent's share, the rest by the unit's sample."""
    counts = {t: W.groupby(k + ["religion"])["persons"].sum().unstack(fill_value=0)
                  .reindex(columns=RELS, fill_value=0) for t, k in TIERS[:3]}
    shares = {t: c.div(c.sum(axis=1), axis=0) for t, c in counts.items()}
    drawn = S.set_index(["district", "county", "subcounty"])["drawn"]
    sub = counts["subcounty"]
    ids = drawn.reindex(sub.index).to_numpy()
    if pd.isna(ids).any():
        raise SystemExit("a sample subcounty with no drawn unit")
    sc = sub.groupby(ids).sum()
    par = S.groupby("drawn")[["district", "county"]].agg(["min", "max"])
    if ((par[("district", "min")] != par[("district", "max")])
            | (par[("county", "min")] != par[("county", "max")])).any():
        raise SystemExit("a merged unit spans two counties")
    par = pd.DataFrame({"district": par[("district", "min")], "county": par[("county", "min")]})
    par = par.reindex(sc.index)
    drawn_tier = {r: (t if t in ("district", "county") else "subcounty") for r, t in tier_of.items()}
    fine = [r for r in RELS if drawn_tier[r] == "subcounty"]
    coarse = [r for r in RELS if drawn_tier[r] != "subcounty"]
    comp = pd.DataFrame(index=sc.index, columns=RELS, dtype=float)
    for r in coarse:
        if drawn_tier[r] == "county":
            idx = pd.MultiIndex.from_arrays([par["district"], par["county"]])
        else:
            idx = par["district"]
        comp[r] = shares[drawn_tier[r]][r].reindex(idx).to_numpy()
    rest = 1.0 - comp[coarse].sum(axis=1)
    fs = sc[fine].sum(axis=1)
    if (fs <= 0).any() or (rest <= 0).any():
        raise SystemExit("a unit with no room for its own categories")
    for r in fine:
        comp[r] = rest * sc[r] / fs
    err = float((comp.sum(axis=1) - 1).abs().max())
    if err > 1e-9 or (comp < 0).to_numpy().any():
        raise SystemExit(f"composition does not close: worst {err:.2e}")
    return comp, drawn_tier, sc


def main():
    failures = []
    if "--fetch" in sys.argv:
        import ug_2024_geo

        ug_2024_geo.fetch()
    lab = religion_labels()

    print("1. the full-count workbook")
    P, lv, national, apaa = profiles()
    for k, want in EXPECT_UNITS.items():
        got = len(P) if k == "parish" else len(lv[k])
        say(got == want, f"{got:,} {k} rows (expected {want:,})", failures)
    for col in ("total", "hhpop", "households"):
        s = int(P[col].sum()) + apaa[col]
        say(s == NATIONAL[col] == national[col],
            f"parishes plus Apaa give the National row's {col}: {s:,}", failures)
    for lvl, key in (("subcounty", ["district", "county", "subcounty"]),
                     ("county", ["district", "county"]), ("district", ["district"])):
        agg = P.groupby(key, sort=False)[["hhpop", "households"]].sum()
        ref = pd.DataFrame(lv[lvl])[["hhpop", "households"]].to_numpy()
        say(bool((agg.to_numpy() == ref).all()),
            f"household population and households close at {lvl} too", failures)

    print("\n2. the sample, and the join to the workbook")
    g, W = load_sample(failures)
    J = join(g, P, lv, district_names(), failures)
    S = subcounty_table(J, P)
    k3 = ["district", "county", "subcounty"]
    nonhh_census = 1 - J.groupby(k3)["hhpop"].sum() / J.groupby(k3)["total"].sum()
    nonhh_sample = J.groupby(k3)["nonhh"].sum() / J.groupby(k3)[["hh", "nonhh"]].sum().sum(axis=1)
    rr = float(np.corrcoef(nonhh_census, nonhh_sample)[0, 1])
    say(rr > 0.9, f"non-household share per subcounty, census against sample, r = {rr:.3f}",
        failures)

    print("\n3. the portal's subcounty codes, names and totals against both")
    ps = portal_subcounties()
    ps["unit"] = [unit_id(d, c, s) for d, c, s in zip(ps["district"], ps["county"], ps["subcounty"])]
    mm = S.merge(ps, on="unit", how="outer", indicator=True, suffixes=("", "_portal"))
    extra = {r["code"]: r["name"] for _, r in mm[mm["_merge"] == "right_only"].iterrows()}
    say(extra == PORTAL_EXTRA and not (mm["_merge"] == "left_only").any(),
        f"portal subcounty codes cover the sample's; the portal's extra rows are {extra}", failures)
    b = mm[mm["_merge"] == "both"]
    say(bool((b["subcounty_name"] == b["name"]).all()),
        f"the portal's name for each code is the sample's name for it (all {len(b):,})", failures)
    na = set(b.loc[b["total_portal"].isna(), "code"])
    tb = b[b["total_portal"].notna()]
    say(na == PORTAL_NA and bool((tb["total"] == tb["total_portal"]).all()),
        f"portal totals equal the workbook's for {len(tb):,} subcounties; `#N/A` for {sorted(na)}",
        failures)

    print("\n4. the stability test, households dealt into six waves")
    cell_cap(lab, failures)
    tiers = TIERS if "--parish" in sys.argv else TIERS[:3]
    V = stability(W, lab, tiers, failures)
    order = [t for t, _ in TIERS]
    tier_of = {}
    for r in RELS:
        passed = V[(V["code"] == r) & V["passed"]]["tier"].tolist()
        tier_of[r] = max(passed, key=order.index) if passed else "national"
    tested = [t for t, _ in tiers]
    expect = {r: (t if t in tested else tested[-1]) for r, t in EXPECT_TIER.items()}
    say(tier_of == expect, "each code's finest passing tier is EXPECT_TIER's: "
        + ", ".join(f"{lab[r].split(' /')[0]} {tier_of[r]}" for r in RELS), failures)
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed; nothing written")

    print("\n5. composition on the drawn units, scaled to their household population")
    comp, drawn_tier, sc = compose(W, tier_of, S)
    D = S.groupby("drawn").agg(hhpop=("hhpop", "sum"), total=("total", "sum"),
                               members=("unit", list), district_name=("district_name", "first"),
                               county_name=("county_name", "first"),
                               names=("subcounty_name", list)).reindex(comp.index)
    counts = comp.mul(D["hhpop"], axis=0)
    nat = counts.sum()
    print(f"     drawn {nat.sum():,.0f} people on {len(comp):,} units "
          f"({len(S):,} census subcounties; household population {int(D['hhpop'].sum()):,})")
    smp_nat = sc.sum()
    for r in RELS:
        rep = REPORT_2024.get(r)
        scaled = smp_nat[r] / smp_nat.sum() * D["hhpop"].sum()
        print(f"     {lab[r][:40]:<40} {nat[r]:>12,.0f}  sample x one factor {scaled:>12,.0f}"
              + (f"  report {rep:>11,}  {nat[r] / rep:5.3f}" if rep else "")
              + f"   [{drawn_tier[r]}]")
    off = [lab[r] for r in (11, 12, 14, 15) if abs(nat[r] / REPORT_2024[r] - 1) > 0.03]
    say(not off, "the four largest answers land within 3% of Final Report Table 3.1", failures)
    # Latvia's check: a coarse category drawn well under what its own unit's sample shows.
    own = sc.div(sc.sum(axis=1), axis=0)
    for r in [r for r in RELS if drawn_tier[r] != "subcounty"]:
        rev = own[r][(sc[r] >= 10) & (own[r] >= 2 * comp[r])]
        worst = (own[r] - comp[r]).idxmax()
        print(f"     {lab[r]:<20} at {drawn_tier[r]}: {len(rev)} units where 10+ sample persons give "
              f"twice the drawn share or more; largest gap {worst} "
              f"{own[r][worst]:.4f} sampled against {comp[r][worst]:.4f} drawn")
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed; nothing written")

    rows = []
    for uid in comp.index:
        dr = D.loc[uid]
        nm = " and ".join(n.title() for n in dr["names"])
        geo_name = f"{nm}, {dr['county_name'].title()}, {dr['district_name'].title()}"
        base = dict(geo_id=uid, geo_level=LEVEL, geo_name=geo_name, basis="self_id", year=2024,
                    source_id=SOURCE_ID)
        extra_note = f"; census subcounties {'+'.join(dr['members'])}" if len(dr["members"]) > 1 else ""
        for r in RELS:
            rows.append(dict(base, source_category=lab[r], count=round(float(counts.loc[uid, r]), 2),
                             note=f"level={LEVEL}; tier={drawn_tier[r]}; "
                                  f"sample_persons={int(sc.loc[uid, r])}{extra_note}"))
        rows.append(dict(base, source_category="Total population", count=int(dr["total"]),
                         note=f"level={LEVEL}; workbook Table 1{extra_note}"))
        rows.append(dict(base, source_category="Not in a household",
                         count=int(dr["total"] - dr["hhpop"]),
                         note=f"level={LEVEL}; workbook Table 1 minus Table 2; no religion "
                              f"recorded{extra_note}"))
    df = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                     "count", "basis", "year", "source_id", "note"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".part", index=False, lineterminator="\n")
    os.replace(OUT + ".part", OUT)
    vt = os.path.join(AGG, "ug2024_stability.csv")
    V.to_csv(vt + ".part", index=False, lineterminator="\n")
    os.replace(vt + ".part", vt)
    print(f"\nwrote {OUT} ({len(df):,} rows, {df['geo_id'].nunique():,} units) and {vt}")


if __name__ == "__main__":
    main()
