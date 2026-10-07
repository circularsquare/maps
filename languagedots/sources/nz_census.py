"""New Zealand, Census 2023: languages spoken by SA1 -> data/normalized/nz.csv.

    python sources/nz_census.py --fetch     # data/raw/nz/ (~10 MB, no key)
    python sources/nz_census.py             # data/raw/nz/ -> data/normalized/nz.csv

THE QUESTION. "In which language(s) could you have a conversation about a lot of everyday
things?" Every language a person can hold a conversation in, as many as they like. Not a
first language: 95% name English, and most who name another language name English too.

SOURCE. Stats NZ Geospatial's "2023 Census totals by topic for individuals" feature services
(ArcGIS Online, CC BY 4.0, no key), part 1, unclipped layer, at SA1 (33,000 units of ~150
people) and SA2 (2,395). Both carry the same two level-1 variables:
  Languages spoken (total responses): English, Maori, Samoan, NZ Sign Language, Other,
      None (too young to talk), Not elsewhere included, Total, Total stated
  Official language indicator: 13 classes of person by which of {Maori, English, NZSL, Other}
      they named (English only; Maori and English only; English and Other only; ...)
Field names are opaque (VAR_1_205); the categories are read from the layers' own aliases.

THE OTHER SPLIT (SA2). Aotearoa Data Explorer's dataflow CEN23_ECI_011 (2023, Total ethnicity,
Total gender), downloaded by hand into data/raw/nz/CEN23_ECI_011_sa2_2023.csv because ADE's data
API needs a key, names 11 more languages at SA2: Northern Chinese, Hindi, Tagalog, Sinitic not
further defined, Yue, French, Panjabi, Afrikaans, Spanish, German, Tongan, and a smaller Other.
Each SA1's Other persons are split over those 12 by its SA2's mentions (applying the SA2's mix to
each of its SA1s; the SA1 layer has no such detail). An SA2 with any of the 12 cells confidential
keeps its Other unsplit, as does one with no Other mentions (124 small SA2s, 21 people). Without the file, nz.csv is
written as before, Other unsplit.

SHARING EACH PERSON (spec §3.6). The indicator is the cross-table that gives the combinations,
so the split between English, Maori, NZSL and "some other language" is exact: an "English and
Maori only" person gives 1/2 to each, a "Maori, English and Other" person 1/3 to each.
Two parts are not exact:
  * class 52, "other combination" (Maori+NZSL, Maori+Other, NZSL+Other, all three without
    English; 0.1% of people): shared over Maori, NZSL and Other by what each language's
    mentions leave unaccounted for after the named classes, assuming two languages a person.
  * the "Other" slot (a person counts once there however many other languages they named) is
    split between Samoan and the remaining Other by their mentions in the unit.
Every row is tier `derived`. "No language" (too young to talk) and "not elsewhere included"
are not drawn: they are the gap.

Counts are randomly rounded to base 3 per cell by Stats NZ, so a unit's classes do not add to
its total exactly; suppressed cells carry -999 (confidential) or -997 (not available).

CHECKS (all must pass, or nothing is written):
  1. per SA1, each of English, Maori and NZSL mentions equals the indicator classes holding it,
     within random rounding (|diff| small; the distribution is printed)
  2. SA1s sum to their SA2 per category, within accumulated rounding, and the SA2 layer's
     totals agree with the SA1 layer's nationally
  3. the persons drawn plus the gap sum to the units' indicator totals
  4. the ADE table is CEN23_ECI_011, 2023, Total ethnicity and gender, every SA2 of the layer, and
     its English, Maori, Samoan, NZSL, None, Total and Total stated equal the SA2 layer's exactly
  5. per SA2, ADE's 11 + Other mentions are at least the layer's level-1 Other, within rounding
     (level 1 counts a person once however many other languages they named)
  6. the split keeps every SA1's Other persons, and every non-Other row is unchanged
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nz"
OUT = ROOT / "data" / "normalized" / "nz.csv"
SVC = "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/"
LAYERS = {
    "sa1": (SVC + "2023_Census_totals_by_topic_for_individuals_by_SA1/FeatureServer/1",
            "SA12023_V1_00"),
    "sa2": (SVC + "2023_Census_totals_by_topic_for_individuals_by_SA2/FeatureServer/1",
            "SA22023_V1_00"),
}
PAGE = 2000
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
ALIAS = re.compile(r"Year:\s*2023,\s*Measure:\s*Count,\s*Var1:\s*"
                   r"(?P<var>Languages spoken|Official language indicator|"
                   r"Census usually resident population count) \((?P<cat>.+)\)\s*$")

# indicator classes -> the slots each person in it named (E, M, N = NZSL, O = other slot)
CLASSES = {
    "No language": "",
    "Māori only": "M",
    "English only": "E",
    "NZ Sign Language only": "N",
    "Māori and English only (not NZ Sign Language)": "ME",
    "English and NZ Sign Language only (not Māori)": "EN",
    "English and Other only (not Māori or NZ Sign Language)": "EO",
    "Māori, English and NZ Sign Language (not Other)": "MEN",
    "Māori, English and Other (not NZ Sign Language)": "MEO",
    "English, NZ Sign Language and Other (not Māori)": "ENO",
    "Māori, English, NZ Sign Language and Other": "MENO",
    "Other languages only (neither English, Māori nor NZ Sign Language)": "O",
    "Other combination of Māori, English, NZ Sign Language and Other": "*",
    "Not elsewhere included": "",
}
LANGS = {"English": "E", "Māori": "M", "Samoan": "S", "New Zealand Sign Language": "N",
         "Other": "O"}
SENTINELS = (-999, -997)
ADE = RAW / "CEN23_ECI_011_sa2_2023.csv"
# ADE's labels for the languages it names inside level 1's Other, largest first; its own "Other"
# is what remains
OTHER_SPLIT = ["Northern Chinese", "Hindi", "Tagalog", "Sinitic not further defined", "Yue",
               "French", "Panjabi", "Afrikaans", "Spanish", "German", "Tongan"]
ADE_SAME = {"English": "L:English", "Māori": "L:Māori", "Samoan": "L:Samoan",
            "New Zealand Sign Language": "L:New Zealand Sign Language",
            "None (eg too young to talk)": "L:None (eg too young to talk)",
            "Total - languages spoken": "L:Total", "Total stated - languages spoken": "L:Total stated"}


def get(url, params):
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def fields(level):
    meta = json.loads((RAW / f"{level}_layer.json").read_text(encoding="utf-8"))
    out = {}
    for f in meta["fields"]:
        m = ALIAS.search(f.get("alias") or "")
        if m:
            out[f["name"]] = (m.group("var"), m.group("cat").strip())
    want = {("Census usually resident population count", "Total")}
    want |= {("Languages spoken", c) for c in LANGS}
    want |= {("Official language indicator", c) for c in CLASSES}
    missing = want - set(out.values())
    if missing:
        raise SystemExit(f"{level}: fields not found in the aliases: {sorted(missing)}")
    return out


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for level, (url, key) in LAYERS.items():
        if (RAW / f"{level}_languages.csv").exists():
            print(f"  {level}: already fetched")
            continue
        (RAW / f"{level}_layer.json").write_bytes(get(url, {"f": "json"}))
        meta = json.loads((RAW / f"{level}_layer.json").read_text(encoding="utf-8"))
        have = {f["name"] for f in meta["fields"]}
        names = [key] + (["LANDWATER"] if "LANDWATER" in have else []) + sorted(fields(level))
        rows, offset = [], 0
        while True:
            body = json.loads(get(url + "/query", {
                "where": "1=1", "outFields": ",".join(names), "returnGeometry": "false",
                "orderByFields": key, "resultOffset": offset, "resultRecordCount": PAGE,
                "f": "json"}))
            if "features" not in body:
                raise SystemExit(f"{level}: {str(body)[:300]}")
            feats = [f["attributes"] for f in body["features"]]
            rows += feats
            offset += len(feats)
            print(f"  {level}: {offset:,} rows", flush=True)
            if len(feats) < PAGE and not body.get("exceededTransferLimit"):
                break
        pd.DataFrame(rows).to_csv(RAW / f"{level}_languages.csv", index=False)


def load(level):
    f = fields(level)
    key = LAYERS[level][1]
    df = pd.read_csv(RAW / f"{level}_languages.csv", dtype={key: str, "LANDWATER": str})
    cols = {}
    for name, (var, cat) in f.items():
        tag = {"Languages spoken": "L", "Official language indicator": "I",
               "Census usually resident population count": "P"}[var]
        cols[name] = f"{tag}:{cat}"
    df = df.rename(columns=cols).rename(columns={key: "geo_id"})
    if "LANDWATER" not in df.columns:
        df["LANDWATER"] = ""
    return df[["geo_id", "LANDWATER"] + sorted(cols.values())]


def share(df):
    """Persons per unit on English, Maori, NZSL, Samoan, Other (spec §3.6, docstring)."""
    n = {c: df[f"I:{c}"].clip(lower=0).to_numpy(float) for c in CLASSES}
    men = {k: df[f"L:{c}"].clip(lower=0).to_numpy(float) for c, k in LANGS.items()}
    slot = {s: np.zeros(len(df)) for s in "EMNO"}
    held = {s: np.zeros(len(df)) for s in "EMNO"}
    for c, slots in CLASSES.items():
        if slots in ("", "*"):
            continue
        for s in slots:
            slot[s] += n[c] / len(slots)
            held[s] += n[c]
    c52 = n["Other combination of Māori, English, NZ Sign Language and Other"]
    rM = np.clip(men["M"] - held["M"], 0, None)
    rN = np.clip(men["N"] - held["N"], 0, None)
    rO = np.clip(2 * c52 - rM - rN, 0, None)
    tot = rM + rN + rO
    w = {"M": np.divide(rM, tot, out=np.zeros_like(tot), where=tot > 0),
         "N": np.divide(rN, tot, out=np.zeros_like(tot), where=tot > 0),
         "O": np.divide(rO, tot, out=np.ones_like(tot), where=tot > 0)}
    for s in "MNO":
        slot[s] += c52 * w[s]
    so = men["S"] + men["O"]
    fs = np.divide(men["S"], so, out=np.zeros_like(so), where=so > 0)
    out = pd.DataFrame({
        "English": slot["E"], "Māori": slot["M"], "New Zealand Sign Language": slot["N"],
        "Samoan": slot["O"] * fs, "Other": slot["O"] * (1 - fs)}, index=df.index)
    return out, held, men


def load_ade(sa2):
    """ADE's CEN23_ECI_011 at SA2, as SA2 x label (NaN = confidential); checks 4 and 5."""
    df = pd.read_csv(ADE, dtype=str, encoding="utf-8-sig")
    one = {"STRUCTURE_ID": "STATSNZ:CEN23_ECI_011(1.0)", "CEN23_YEAR_001": "2023",
           "CEN23_ETH_003": "9999", "CEN23_GEN_002": "99"}
    for c, v in one.items():
        if set(df[c]) != {v}:
            raise SystemExit(f"{ADE.name}: {c} is {sorted(set(df[c]))}, expected only {v}")
    want = set(OTHER_SPLIT) | {"Other"} | set(ADE_SAME)
    if not want <= set(df["Languages spoken"]):
        raise SystemExit(f"{ADE.name}: labels missing: {sorted(want - set(df['Languages spoken']))}")
    # the file also holds regions, TALBs and health areas (codes of 2 to 5 digits) and the
    # 999999 national row; SA2 codes are six digits
    df = df[(df["CEN23_GEO_002"].str.len() == 6) & (df["CEN23_GEO_002"] != "999999")]
    df["v"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    if (df["v"].isna() != (df["OBS_STATUS"] == "c")).any():
        raise SystemExit(f"{ADE.name}: an empty value that is not marked confidential")
    ade = df.pivot(index="CEN23_GEO_002", columns="Languages spoken", values="v")
    if set(ade.index) != set(sa2["geo_id"]):
        raise SystemExit(f"{ADE.name}: SA2s differ from the layer's "
                         f"({len(set(ade.index) - set(sa2['geo_id']))} extra, "
                         f"{len(set(sa2['geo_id']) - set(ade.index))} missing)")
    s2 = sa2.set_index("geo_id")
    ok = True
    for a, lay in ADE_SAME.items():
        both = ade[a].notna() & (s2[lay].reindex(ade.index) >= 0)
        d = (ade.loc[both, a] - s2[lay].reindex(ade.index)[both]).abs()
        print(f"check 4 {a}: ADE vs SA2 layer on {both.sum():,} SA2s, max |diff| {d.max():.0f}, "
              f"national {ade.loc[both, a].sum():,.0f}")
        if d.max() > 0:
            ok = False
    mix = ade[OTHER_SPLIT + ["Other"]]
    lo = s2["L:Other"].reindex(ade.index)
    both = mix.notna().all(axis=1) & (lo >= 0)
    d = mix[both].sum(axis=1) - lo[both]
    print(f"check 5: ADE's 11 + Other minus the layer's Other per SA2 on {both.sum():,} SA2s: "
          f"min {d.min():.0f}, median {d.median():.0f}; nationally "
          f"{mix[both].sum().sum() / lo[both].sum():.3f} mentions a level-1 Other response")
    if d.min() < -12:
        ok = False
    if not ok:
        raise SystemExit("ADE checks failed; nothing written")
    return ade


def split_other(long, ade):
    """Each SA1's Other persons over ADE's 11 languages and its residual Other, by the SA2's
    mentions; check 6."""
    mix = ade[OTHER_SPLIT + ["Other"]]
    usable = mix.notna().all(axis=1) & (mix.sum(axis=1) > 0)
    share = mix[usable].div(mix[usable].sum(axis=1), axis=0)
    is_o = long["source_category"] == "Other"
    go = is_o & long["sa2"].isin(share.index)
    keep, oth = long[~go], long[go]
    parts = []
    for lab in share.columns:
        p = oth.copy()
        p["count"] = p["count"].to_numpy() * share.loc[p["sa2"], lab].to_numpy()
        p["source_category"] = lab
        p["mentions"] = np.nan      # the mix is the SA2's; the SA1 has no such detail
        parts.append(p)
    out = pd.concat([keep] + parts, ignore_index=True)
    before = long[is_o].groupby("geo_id")["count"].sum()
    after = out[out["source_category"].isin(share.columns)].groupby("geo_id")["count"].sum()
    resid = (after.reindex(before.index) - before).abs().max()
    print(f"check 6: Other split on {usable.sum():,} SA2s ({(~usable).sum()} kept whole, "
          f"{long.loc[is_o & ~go, 'count'].sum():,.1f} people); per-SA1 Other kept to "
          f"{resid:.2e}")
    if resid > 1e-6:
        raise SystemExit("the Other split lost people; nothing written")
    return out


def normalize():
    sa1, sa2 = load("sa1"), load("sa2")
    for name, d in (("sa1", sa1), ("sa2", sa2)):
        num = d.drop(columns=["geo_id", "LANDWATER"])
        bad = num.isin(SENTINELS).sum().sum()
        neg = (num < 0).sum().sum()
        print(f"{name}: {len(d):,} units, {bad:,} suppressed cells, {neg:,} negative cells, "
              f"population {d['P:Total'].clip(lower=0).sum():,.0f}")
    lut = pd.read_csv(ROOT.parent / "religiondots" / "data" / "geo" / "nz" /
                      "sa1_2023_to_sa2_2023.csv", dtype=str)
    sa1["sa2"] = sa1["geo_id"].map(dict(zip(lut["SA12023_V1_00"], lut["SA22023_V1_00"])))
    # religiondots' SA1 layer (the placement layer) holds the land SA1s and the 71 inland-water
    # ones, which it drops; SA1s of inlets and ocean are not in it at all. Their people have no
    # land to stand on and go to the gap.
    water = sa1["sa2"].isna() | (sa1["LANDWATER"] == "21")
    print(f"{water.sum()} SA1s of water ({sorted(sa1.loc[water, 'LANDWATER'].unique())}), "
          f"{sa1.loc[water, 'P:Total'].sum():,} people: gap")
    # Small units have cells suppressed (-999): under about six people every cell, up to about
    # thirty most of the indicator's classes. Only the population is always printed. Any SA1
    # with a suppressed cell is drawn on its SA2's shares instead.
    full = ~water & (sa1["L:Total stated"] < 0)
    supp = ~water & (sa1[[c for c in sa1.columns if c[:2] in ("L:", "I:")]] < 0).any(axis=1)
    print(f"{supp.sum():,} SA1s with suppressed cells ({full.sum():,} entirely), "
          f"{sa1.loc[supp, 'P:Total'].sum():,} people: drawn on their SA2's shares")
    sa1_all = sa1
    sa1 = sa1[~water & ~supp].reset_index(drop=True)

    # check 1: mentions of E, M, N equal the indicator classes holding them, per SA1
    persons, held, men = share(sa1)
    ok = True
    for s, lab in (("E", "English"), ("M", "Māori"), ("N", "NZSL")):
        d = men[s] - held[s]
        q = np.percentile(np.abs(d), [50, 99, 100])
        print(f"check 1 {lab}: mentions - classes per SA1, |diff| median {q[0]:.0f}, "
              f"99th {q[1]:.0f}, max {q[2]:.0f}; national {men[s].sum():,.0f} vs "
              f"{held[s].sum():,.0f}")
        if s == "E" and abs(men[s].sum() - held[s].sum()) > 0.001 * men[s].sum():
            ok = False
    # Maori's and NZSL's mentions also hold class 52 people (two or three slots each, no English)
    c52 = sa1["I:Other combination of Māori, English, NZ Sign Language and Other"].clip(lower=0)
    extra = men["M"].sum() - held["M"].sum() + men["N"].sum() - held["N"].sum()
    print(f"check 1 class 52: {c52.sum():,.0f} people; Maori + NZSL mentions beyond the named "
          f"classes {extra:,.0f}, must lie within 0 and {3 * c52.sum():,.0f} (plus rounding)")
    if not -1000 <= extra <= 3 * c52.sum() + 1000:
        ok = False

    # check 2: SA1s sum to their SA2s, and the layers agree nationally
    cats = [f"L:{c}" for c in LANGS] + ["P:Total", "I:Total"]
    agg = sa1_all[cats].clip(lower=0).assign(sa2=sa1_all["sa2"]).groupby("sa2")[cats].sum()
    s2 = sa2.set_index("geo_id")[cats]
    s2 = s2[s2.index.isin(agg.index) & (s2 >= 0).all(axis=1)]   # SA2s with no suppressed cell
    print(f"check 2 on {len(s2):,} SA2s with no suppressed cell")
    diff = (agg.reindex(s2.index) - s2).abs()
    for c in cats:
        q = np.percentile(diff[c], [50, 99, 100])
        nat1, nat2 = sa1_all[c].clip(lower=0).sum(), sa2[c].clip(lower=0).sum()
        print(f"check 2 {c}: SA2 vs sum of its SA1s |diff| median {q[0]:.0f}, 99th {q[1]:.0f},"
              f" max {q[2]:.0f}; national SA1 {nat1:,.0f} vs SA2 {nat2:,.0f}")
        if abs(nat1 - nat2) > 0.005 * nat2 + 500:
            ok = False
    only2 = sa2[~sa2["geo_id"].isin(agg.index)]
    only1 = set(agg.index) - set(sa2["geo_id"])
    print(f"check 2: {len(only2)} SA2s have no land SA1 in the lookup "
          f"({only2['P:Total'].clip(lower=0).sum():,} people, water SA2s); "
          f"{len(only1)} lookup SA2s missing from the SA2 layer")
    if only1:
        ok = False

    # check 3: drawn persons + gap = indicator totals
    drawn = persons.sum(axis=1)
    gap = (sa1["I:No language"].clip(lower=0) + sa1["I:Not elsewhere included"].clip(lower=0))
    classes = sum(sa1[f"I:{c}"].clip(lower=0) for c in CLASSES)
    resid = (drawn + gap - classes).abs().max()
    print(f"check 3: drawn {drawn.sum():,.1f} + gap {gap.sum():,.0f} = classes "
          f"{classes.sum():,.0f} (max per-SA1 residual {resid:.6f}); indicator total "
          f"{sa1['I:Total'].clip(lower=0).sum():,.0f}; no language "
          f"{sa1['I:No language'].clip(lower=0).sum():,.0f}, not elsewhere included "
          f"{sa1['I:Not elsewhere included'].clip(lower=0).sum():,.0f}")
    if resid > 1e-6:
        ok = False
    print("persons drawn, national:")
    for c in persons.columns:
        print(f"  {c:28s} {persons[c].sum():12,.1f}   (mentions "
              f"{sa1['L:' + c].clip(lower=0).sum():,.0f})")
    if not ok:
        raise SystemExit("checks failed; nothing written")

    long = persons.assign(geo_id=sa1["geo_id"], sa2=sa1["sa2"]).melt(
        id_vars=["geo_id", "sa2"], var_name="source_category", value_name="count")
    m = sa1.set_index("geo_id")[[f"L:{c}" for c in LANGS]].clip(lower=0)
    m.columns = list(LANGS)
    long["mentions"] = [m.at[g, c] for g, c in zip(long["geo_id"], long["source_category"])]

    # the fully suppressed SA1s: population x their SA2's persons-per-head, from the SA2 layer
    # (which includes them); an SA2 itself suppressed falls back to its other SA1s, then to NZ
    p2, _, _ = share(sa2)
    p2.index = sa2["geo_id"]
    rate = p2.div(sa2.set_index("geo_id")["P:Total"].where(lambda s: s > 0), axis=0)
    p1 = persons.assign(sa2=sa1["sa2"]).groupby("sa2").sum()
    pop1 = sa1.groupby("sa2")["P:Total"].sum()
    rate1 = p1.div(pop1.where(lambda s: s > 0), axis=0)
    nat = persons.sum() / sa1["P:Total"].sum()
    sup = sa1_all[supp & (sa1_all["P:Total"] > 0)]
    rows, fell = [], 0
    for g, s2, pop in zip(sup["geo_id"], sup["sa2"], sup["P:Total"]):
        r = rate.loc[s2] if s2 in rate.index and rate.loc[s2].notna().all() \
            and rate.loc[s2].sum() > 0 else None
        if r is None:
            fell += 1
            r = rate1.loc[s2] if s2 in rate1.index and rate1.loc[s2].notna().all() \
                and rate1.loc[s2].sum() > 0 else nat
        for c in persons.columns:
            if pop * r[c] > 0:
                rows.append((g, s2, c, pop * r[c], np.nan))
    print(f"suppressed SA1s: {sup['P:Total'].sum():,} people, {sum(x[3] for x in rows):,.1f} "
          f"drawn; {fell} on a fallback rate (their SA2 suppressed too)")
    long = pd.concat([long, pd.DataFrame(rows, columns=long.columns)], ignore_index=True)
    if ADE.exists():
        long = split_other(long, load_ade(sa2))
        print("persons drawn, national, after the Other split:")
        for c, v in long.groupby("source_category")["count"].sum().sort_values(
                ascending=False).items():
            print(f"  {c:28s} {v:12,.1f}")
    else:
        print(f"{ADE.name} not in data/raw/nz/: Other left unsplit (sources/nz.md)")
    long = long[long["count"] > 0]
    long.insert(1, "geo_level", "sa1")
    long["tier"] = "derived"
    long.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(long):,} rows, {long['count'].sum():,.1f} people")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    normalize()
