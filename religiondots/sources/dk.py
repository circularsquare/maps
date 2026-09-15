"""Denmark — a church roll the state keeps, and a survey that asks what people are.

Writes data/normalized/dk.csv (citizens) and dk_foreign.csv (foreign residents).

Usage:
    python sources/dk.py --fetch     # ESS tabulations, Eurostat census, Pew, DST's roll
    python sources/dk.py             # rebuild from data/raw/dk/

THE OFFICE WAS CHECKED FIRST (§9cu), AND WHAT IT HAS IS A ROLL OF ONE CHURCH. Danmarks Statistik
publishes Church of Denmark membership from the population register (CPR) and nothing else about
religion: its Statbank has 5,721 tables and the only ones on belief are KM1/KM5 (members per
parish), KM6 (members per kommune, 2011-2026), KM2/KM22 (joining and leaving) and KM4/KM44
(church ceremonies). Its own information page on religion says it holds no figures on other
faith communities and sends readers to Aarhus University's Center for Samtidsreligion. No public
shelf of commissioned tables was found by two web searches; sources/dk.md §1 says how thin that
negative is.

**KM6 is a `roll` (spec §3.1) and this map does not draw it.** Finland (§9by), Sweden (§9cz) and
Norway (§9dd) are all drawn from self-identification with their rolls printed beside, and
Denmark on its roll would put a step of more than twenty points in the Lutheran share at the
border with each of them. build() prints KM6 beside the survey at the same 11 landsdele.

SO THE TWO HALVES ARE THE OTHER NORDIC COUNTRIES':

    Danish citizens      5.3M   ESS rounds 5, 6, 7 and 9 pooled, `ctzcntr = Yes`
    foreign residents    0.5M   Eurostat `cens_21ctz_r3` x Pew's origin compositions

DENMARK IS IN FOUR ROUNDS WITH A REGION, AND NONE AFTER 2018-19. Rounds 1-4 carry no `region`;
Denmark is absent from rounds 8, 10 and 11. So the citizen half is 2010-2019 and `grain` says so.

THE VARIABLE IS `rlgdnm`. `rlgdndk` exists in round 5 only and is the harmonised card in Danish,
code for code (`Protestantisk` is `Protestant`), which is Belgium's case; `_check_card` proves it.
`rlgdnadk` and `rlgdnbdk` do not exist. So the Church of Denmark and the free churches are one
answer, which Norway's card split and Denmark's does not.

THE LEVEL IS NUTS 2, THE 5 REGIONER, IN EVERY ROUND, AND ROUND 9 LABELS THEM IN A DIFFERENT ORDER.
Rounds 5-7 give DK01 Hovedstaden ... DK05 Nordjylland as NUTS does. Round 9's DK01-DK05 are
Danmarks Statistik's own region numbers 1081-1085 read in order: Nordjylland, Midtjylland,
Syddanmark, Hovedstaden, Sjaelland. Nothing errors; the sample shares, the urban profile, the
religious composition and the party vote all come out in the wrong regions (published DK04 is
35% big-city and has 16 of the round's Muslims; published DK01 is 11.5% of the sample). And
ESS's own post-stratification weight `pspwght` was raked to those wrong labels, so round 9 is
weighted by its design weight instead (Denmark's sample is a simple random draw from CPR, so
that is equal weights). `_check_recode` asserts the recode against the data on every build.

COUNTED AT THE 11 LANDSDELE (NUTS 3), where the census counts citizenship and so where the foreign
half is measured. They nest in the regions exactly, so every landsdel takes its region's citizen
composition, and a category the test does not license is drawn at the national rate inside each
landsdel's residual (§9bi).
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "dk")
OUT = os.path.join(ROOT, "data", "normalized", "dk.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "dk_foreign.csv")
LAU = os.path.join(ROOT, "data", "geo", "dk", "dk_lau.gpkg")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# THE STATISTIC IS IMPORTED, NOT COPIED: Norway's `_stability` is Belgium's median split-half
# Spearman with its per-round permutation null plus Sweden's spatial chi-square gate, and its
# alpha, draw count and seed are be.py's. Denmark must not run a different test from its
# neighbours. Norway's Eurostat helper and query template come along for the same reason.
import be  # noqa: E402
import no as _no  # noqa: E402

# --- ESS --------------------------------------------------------------------------------
ESS_API = _no.ESS_API
ESS_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
}
ESS_YEARS = {5: "2010-11", 6: "2012-13", 7: "2014-15", 9: "2018-19"}
ESS_BREAK = ["region", "ctzcntr", "rlgblg", "rlgdnm"]
CARD_ROUND = 5
CARD_BREAK = ["rlgdndk", "rlgdnm"]
FETCH_WEIGHTS = {"n": None, "w": "pspwght", "d": "dweight"}

# Which file weights each round's composition. Round 9's pspwght was raked to mislabelled regions.
ROUND_WEIGHT = {5: "w", 6: "w", 7: "w", 9: "d"}


def _q(weight):
    return _no._TAB % (f' weightVariable:"{weight}",' if weight else "")


# --- Eurostat and Pew ---------------------------------------------------------------------
EU_CTZ = "cens_21ctz_r3"
PEW_ZIP = _no.PEW_ZIP

# --- DST, the roll, which is printed and not drawn -----------------------------------------
DST_DATA = "https://api.statbank.dk/v1/data"
DST_INFO = "https://api.statbank.dk/v1/tableinfo/KM6?format=JSON&lang=da"
ROLL_YEARS = ["2014", "2021"]      # the survey pool's midpoint, and the census year

# --- the geography ------------------------------------------------------------------------
NUTS2 = {
    "DK01": "Hovedstaden", "DK02": "Sjælland", "DK03": "Syddanmark", "DK04": "Midtjylland",
    "DK05": "Nordjylland",
}
NUTS3 = {
    "DK011": "Byen København", "DK012": "Københavns omegn", "DK013": "Nordsjælland",
    "DK014": "Bornholm", "DK021": "Østsjælland", "DK022": "Vest- og Sydsjælland",
    "DK031": "Fyn", "DK032": "Sydjylland", "DK041": "Vestjylland", "DK042": "Østjylland",
    "DK050": "Nordjylland",
}
assert {u[:4] for u in NUTS3} == set(NUTS2)

# ROUND 9's `region` IS DST's REGION NUMBER ORDER, 1081 Nordjylland, 1082 Midtjylland, 1083
# Syddanmark, 1084 Hovedstaden, 1085 Sjaelland, published under DK01-DK05. published -> true.
RECODE = {9: {"DK01": "DK05", "DK02": "DK04", "DK03": "DK03", "DK04": "DK01", "DK05": "DK02"}}

# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"
REFUSAL = "__refused__"
PROTESTANT = "Protestant"
ISLAM = "Islam"

# rlgdndk (round 5) against rlgdnm, the nesting `_check_card` asserts. It is one to one.
CARD_NEST = {
    "Romersk-katolsk": "Roman Catholic",
    "Protestantisk": "Protestant",
    "Ortodoks": "Eastern Orthodox",
    "Andre kristne religioner": "Other Christian denomination",
    "Jødisk": "Jewish",
    "Islam": "Islam",
    "Østlige religioner": "Eastern religions",
    "Andre ikke kristne religioner": "Other Non-Christian religions",
}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Denmark's own totals, asserted so a re-fetch against a new release fails here rather than
# quietly redrawing the map. None prints the value instead.
POP_2021 = 5_840_046          # cens_21ctz_r3 TOTAL = NAT + FOR + STLS 8,556 + UNK 51
N_CITIZENS = 6_041             # unweighted answered citizens with a region, four rounds

# `No religion` stays the residual, spec §12 (nested units), as in Sweden and Norway. Fixed at its
# region's share it would never go negative here (worst +2.82%), so the rule costs nothing.
KEEP_AS_RESIDUAL = {NO_RELIGION}

# What the 5-region test selects, asserted, so a re-fetch that changes a verdict stops the build.
# ISLAM FAILS THE RANK TEST (p 0.32) WITH A CHI-SQUARE OF 4e-05, AND IS NOT OVERRIDDEN. The regions
# differ because Hovedstaden is 2.98% Muslim among citizens and the other four are 0.80-1.35% in
# no stable order, and a rank correlation over five units cannot see one unit standing apart.
# Norway's override (no.py) had a rank p of 0.0555 and an independent roll ordering all seven
# regions at +0.929; Denmark has neither, and the residual construction already draws the capital
# region's citizens at about 2.1% Muslim against about 1.1-1.6% elsewhere. sources/dk.md §5.
EXPECT_PASS = {PROTESTANT, NO_RELIGION}

# Categories drawn against the test, with the reason printed on every build (gt.py's convention).
OVERRIDE = {}


def _key(s):
    return " ".join(str(s).split())


# =======================================================================================
# fetch
# =======================================================================================

def _ess(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(ESS_API, data=body, headers={
        "Content-Type": "application/json", **UA})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if "errors" in r:
        sys.exit(f"!! ESS API: {r['errors'][0].get('message')} "
                 f"{r['errors'][0].get('extensions', {}).get('code', '')}")
    return r["data"]


def _ess_fetch(rnd, bv, weight, dest):
    if os.path.exists(dest):
        return
    fid, ver = ESS_ROUNDS[rnd]
    d = _ess(_q(weight), {"id": fid, "v": ver, "bv": bv})
    hit = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
           if x["by"][0]["value"] == "DK"]
    if not hit:
        sys.exit(f"!! ESS round {rnd} has no DK response")
    _no._save(hit[0]["response"], dest)


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd in sorted(ESS_ROUNDS):
        for tag, w in FETCH_WEIGHTS.items():
            _ess_fetch(rnd, ESS_BREAK, w, os.path.join(RAW, f"ess_r{rnd}_{tag}.json"))
        _ess_fetch(rnd, ["region", "domicil"], None, os.path.join(RAW, f"ess_r{rnd}_domicil.json"))
        n = sum(c["count"] for c in json.load(
            open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")
    _ess_fetch(CARD_ROUND, CARD_BREAK, None, os.path.join(RAW, f"ess_r{CARD_ROUND}_card.json"))
    _ess_fetch(9, ["region", "prtvtddk"], None, os.path.join(RAW, "ess_r9_party.json"))

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_dk.json")
    if not os.path.exists(dest):
        d = _no._eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T",
                          geo=sorted(NUTS3) + ["DK"])
        _no._save(d, dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("Pew…")
    dest = os.path.join(RAW, "pew.zip")
    if not os.path.exists(dest):
        with urllib.request.urlopen(urllib.request.Request(PEW_ZIP, headers=UA),
                                    timeout=600) as r, open(dest + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(dest + ".tmp", dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("DST KM6, the Church of Denmark roll (for the comparison, not for the map)…")
    dest = os.path.join(RAW, "dst_km6_info.json")
    if not os.path.exists(dest):
        _no._save(json.load(urllib.request.urlopen(DST_INFO, timeout=120)), dest)
    dest = os.path.join(RAW, "dst_km6.csv")
    if not os.path.exists(dest):
        body = json.dumps({
            "table": "KM6", "format": "CSV", "valuePresentation": "Code", "lang": "da",
            "variables": [{"code": "KOMK", "values": ["*"]}, {"code": "FKMED", "values": ["*"]},
                          {"code": "Tid", "values": ROLL_YEARS}]}).encode()
        req = urllib.request.Request(DST_DATA, data=body, headers={
            "Content-Type": "application/json", **UA})
        raw = urllib.request.urlopen(req, timeout=300).read()
        with open(dest + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(dest + ".tmp", dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")


# =======================================================================================
# reading
# =======================================================================================

def _read(path, rnd):
    """One saved ESS response as a tidy frame: region CODE, everything else as a LABEL.

    `path` indexes into `codeList`, it is not a code value (fi.py's trap).
    """
    d = json.load(open(path, encoding="utf-8"))
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rows = []
    for cell in d["table"]:
        rec = {"round": rnd, "count": float(cell["count"])}
        for i, n in enumerate(order):
            c = codes[n][cell["path"][i]]
            rec[n] = c["value"] if n == "region" else _key(c["label"])
            rec[n + "_miss"] = bool(c["isMissing"])
        rows.append(rec)
    return pd.DataFrame(rows)


def _pool(rounds, tag_of):
    """Citizens with a region, region recoded, category resolved (refusals kept as REFUSAL).

    `tag_of` is a file tag, or a {round: tag} dict, which is how round 9 gets its design weight.
    """
    frames = []
    for r in sorted(rounds):
        tag = tag_of[r] if isinstance(tag_of, dict) else tag_of
        df = _read(os.path.join(RAW, f"ess_r{r}_{tag}.json"), r)
        if r in RECODE:
            df["region"] = df["region"].map(lambda x: RECODE[r].get(x, x))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["ctzcntr"] == "Yes") & (~df["region_miss"])].copy()
    df["cat"] = np.where(
        df["rlgblg"] == "No", NO_RELIGION,
        np.where((df["rlgblg"] != "Yes") | df["rlgdnm_miss"], REFUSAL, df["rlgdnm"]))
    return df


def _check_level():
    """The region code set of every round, asserted (Italy's lesson, §9as)."""
    for rnd in sorted(ESS_ROUNDS):
        d = _read(os.path.join(RAW, f"ess_r{rnd}_n.json"), rnd)
        codes = set(d.loc[~d["region_miss"], "region"])
        if codes != set(NUTS2):
            sys.exit(f"!! round {rnd}: regions {sorted(codes)}, expected {sorted(NUTS2)}")
    print(f"  rounds {sorted(ESS_ROUNDS)}: the 5 NUTS 2 codes in every round")


def _check_recode():
    """ROUND 9's LABELS, CHECKED AGAINST THE DATA RATHER THAN TRUSTED.

    After recoding, in every round: Hovedstaden has the highest share living in a big city or its
    suburbs, and Sjaelland the lowest share in a big city (it has none; Roskilde and Naestved are
    towns). In round 9 alone the published labels must FAIL that test, so the recode cannot be
    left on after ESS fixes the file. And round 9's recalled 2015 vote must put Dansk Folkeparti
    higher in Syddanmark than in Midtjylland and Enhedslisten highest in Hovedstaden, which is
    what separates the two Jutland regions (sources/dk.md §3 has the table).
    """
    def profile(rnd, recode):
        d = _read(os.path.join(RAW, f"ess_r{rnd}_domicil.json"), rnd)
        d = d[~d["region_miss"] & ~d["domicil_miss"]].copy()
        if recode and rnd in RECODE:
            d["region"] = d["region"].map(RECODE[rnd])
        t = d.groupby("region")["count"].sum()
        big = d[d["domicil"] == "A big city"].groupby("region")["count"].sum() / t
        sub = d[d["domicil"] == "Suburbs or outskirts of big city"].groupby("region")["count"].sum() / t
        return (big.fillna(0) + sub.fillna(0)).reindex(sorted(NUTS2)).fillna(0), \
            big.reindex(sorted(NUTS2)).fillna(0), t.reindex(sorted(NUTS2)).fillna(0) / t.sum()

    ok = lambda urban, big: urban.idxmax() == "DK01" and big.idxmin() == "DK02"
    for rnd in sorted(ESS_ROUNDS):
        urban, big, share = profile(rnd, True)
        print(f"  round {rnd}: big city or suburbs " + ", ".join(
            f"{NUTS2[r]} {100 * urban[r]:.0f}%" for r in urban.index)
              + "; sample " + ", ".join(f"{100 * share[r]:.0f}%" for r in share.index))
        if not ok(urban, big):
            sys.exit(f"!! round {rnd}: after recoding, the urban profile does not put Hovedstaden "
                     "first and Sjaelland last; the region labels have changed")
    for rnd in RECODE:
        urban, big, _ = profile(rnd, False)
        if ok(urban, big):
            sys.exit(f"!! round {rnd}'s PUBLISHED labels now pass the urban check; ESS has fixed "
                     "the file and RECODE must be removed")
        print(f"  round {rnd} as published fails the same check (big city or suburbs highest in "
              f"{urban.idxmax()}), so RECODE is needed")

    d = _read(os.path.join(RAW, "ess_r9_party.json"), 9)
    d = d[~d["region_miss"] & ~d["prtvtddk_miss"]].copy()
    d["region"] = d["region"].map(RECODE[9])
    t = d.groupby("region")["count"].sum()
    share = lambda p: (d[d["prtvtddk"].str.startswith(p)].groupby("region")["count"].sum()
                       / t).reindex(sorted(NUTS2)).fillna(0)
    df_, el = share("Dansk Folkeparti"), share("Enhedslisten")
    print("  round 9 recalled 2015 vote, recoded: Dansk Folkeparti " + ", ".join(
        f"{NUTS2[r]} {100 * df_[r]:.1f}%" for r in df_.index))
    if not (df_["DK03"] > df_["DK04"] and el.idxmax() == "DK01"):
        sys.exit("!! round 9's party vote does not separate Syddanmark from Midtjylland as RECODE "
                 "says; check the DK02/DK03 assignment")


def _check_card():
    """`rlgdndk` is `rlgdnm` in Danish, answer for answer, proved from round 5's cross-tab."""
    d = _read(os.path.join(RAW, f"ess_r{CARD_ROUND}_card.json"), CARD_ROUND)
    d = d[d["count"] > 0]
    seen = set()
    for _, r in d.iterrows():
        a, b = r["rlgdndk"], r["rlgdnm"]
        if r["rlgdndk_miss"] or r["rlgdnm_miss"]:
            if r["rlgdndk_miss"] != r["rlgdnm_miss"]:
                sys.exit(f"!! `{a}` / `{b}` missing on one card only ({r['count']:.0f})")
            continue
        if CARD_NEST.get(a) != b:
            sys.exit(f"!! rlgdndk `{a}` sits in rlgdnm `{b}` for {r['count']:.0f} respondents")
        seen.add(a)
    if seen != set(CARD_NEST):
        sys.exit(f"!! rlgdndk answers nobody gave: {sorted(set(CARD_NEST) - seen)}")
    print(f"  rlgdndk is rlgdnm in Danish, one to one, {len(seen)} answers (round {CARD_ROUND}, "
          "the only round that carries it)")


# =======================================================================================
# the citizen half
# =======================================================================================

def _assert_source(pool, label):
    import dk2024
    unknown = sorted(set(pool["cat"]) - dk2024.SOURCE)
    if unknown:
        sys.exit(f"!! {label}: source categories with no mapping: {unknown}")
    vanished = sorted(dk2024.SOURCE - set(pool["cat"]))
    if vanished:
        sys.exit(f"!! {label}: dk2024.SOURCE categories nobody answered: {vanished}")


def _composition(df):
    t = df.groupby("cat")["count"].sum()
    return t / t.sum()


def _citizen_shares():
    raw = _pool(ESS_ROUNDS, "n")
    wtd = _pool(ESS_ROUNDS, ROUND_WEIGHT)
    answered = 1.0 - wtd.loc[wtd["cat"] == REFUSAL, "count"].sum() / wtd["count"].sum()
    print(f"  {raw['count'].sum():,.0f} citizens with a region; {100 * (1 - answered):.2f}% "
          "declined (weighted) and are not drawn")
    raw, wtd = raw[raw["cat"] != REFUSAL], wtd[wtd["cat"] != REFUSAL]
    n = int(round(raw["count"].sum()))
    if N_CITIZENS is None:
        print(f"  !! N_CITIZENS is unset; this build has {n:,}")
    elif n != N_CITIZENS:
        sys.exit(f"!! {n:,} answered citizens, expected {N_CITIZENS:,}")
    _assert_source(wtd, "rounds 5-7, 9")

    # The weighting call, printed: round 9 on its design weight against on its broken pspwght.
    alt = _pool(ESS_ROUNDS, "w")
    alt = alt[alt["cat"] != REFUSAL]
    a, b = _composition(wtd), _composition(alt)
    print("  national citizen composition, round 9 on dweight (drawn) against on pspwght:")
    for c in a.sort_values(ascending=False).index:
        print(f"    {c[:40]:<42}{100 * a[c]:6.2f}%  {100 * b.get(c, 0):6.2f}%")

    print("  by round, weighted:")
    per = {r: _composition(wtd[wtd["round"] == r]) for r in sorted(ESS_ROUNDS)}
    cats0 = a.sort_values(ascending=False).index
    print("    " + " " * 42 + "".join(f"{'r' + str(r) + ' ' + ESS_YEARS[r]:>14}" for r in per))
    for c in cats0:
        print(f"    {c[:40]:<42}" + "".join(f"{100 * per[r].get(c, 0):13.2f}%" for r in per))

    tab = wtd.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    tab = tab.reindex(index=sorted(NUTS2), fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)
    nat = tab.sum() / tab.sum().sum()
    cats = sorted(nat.index, key=lambda c: -nat[c])
    share = share.reindex(columns=cats, fill_value=0.0)
    passed = _no._stability(raw, cats, ESS_ROUNDS, NUTS2, "5 NUTS 2 regions")

    if EXPECT_PASS is not None and set(passed) != EXPECT_PASS:
        sys.exit(f"!! the 5-region test now selects {sorted(passed)}, not {sorted(EXPECT_PASS)}; "
                 "OVERRIDE, note_public and sources/dk.md quote the old verdicts")
    for c, why in OVERRIDE.items():
        if c in passed:
            sys.exit(f"!! `{c}` now passes on its own; take it out of OVERRIDE")
        print(f"\n  OVERRIDE, drawn at the 5 regions against the test: `{c}`\n    {why}")
    fine = [c for c in cats if (c in passed or c in OVERRIDE) and c not in KEEP_AS_RESIDUAL]
    small = [c for c in cats if c not in fine]
    nraw = raw.groupby("cat")["count"].sum()
    print(f"\n  {len(fine)} categories at the 5 regions, {len(small)} at the national rate inside "
          "each landsdel's residual")
    for c in cats:
        print(f"    {100 * nat[c]:>6.2f}%  {int(nraw.get(c, 0)):>6,}  {c}   "
              f"[{'5 regions' if c in fine else 'national rate (residual)'}]")
    print("\n  shares by region, weighted:")
    print("    " + " " * 34 + "".join(f"{NUTS2[r][:11]:>13}" for r in share.index))
    for c in cats:
        print(f"    {c[:32]:<34}" + "".join(f"{100 * share.loc[r, c]:12.2f}%" for r in share.index))
    return share, nat, fine, small, answered, n


def _compose(share, nat, fine, small):
    """Per landsdel: a fine category takes its region's share; the rest share the residual at
    the national proportions (§9bi)."""
    out = pd.DataFrame(index=sorted(NUTS3), columns=list(nat.index), dtype=float)
    for u in out.index:
        for c in fine:
            out.loc[u, c] = float(share.loc[u[:4], c])
    residual = 1.0 - out[fine].sum(axis=1) if fine else pd.Series(1.0, index=out.index)
    for c in sorted(KEEP_AS_RESIDUAL & set(small)):
        own = pd.Series({u: float(share.loc[u[:4], c]) for u in out.index})
        alt = residual - own
        print(f"  if `{c}` were fixed at its region's share, the tail left would be negative in "
              f"{int((alt < 0).sum())} of {len(alt)} landsdele (worst {alt.min():+.2%})")
    if (residual <= 0).any():
        sys.exit(f"!! landsdele with no room for the tail: {sorted(residual[residual <= 0].index)}")
    tot = float(nat[small].sum())
    for c in small:
        out[c] = residual * (float(nat[c]) / tot)
    first = {r: min(u for u in out.index if u[:4] == r) for r in sorted(NUTS2)}
    print("  residual categories as drawn among citizens (the survey's own regional share in "
          "brackets):")
    for c in small[:4]:
        print(f"    {c[:30]:<32}" + "  ".join(
            f"{NUTS2[r][:11]} {100 * out.loc[u, c]:.2f}% ({100 * float(share.loc[r, c]):.2f}%)"
            for r, u in first.items()))
    if (out.sum(axis=1) - 1.0).abs().max() > 1e-9:
        sys.exit("!! composition does not sum to 1")
    return out


# =======================================================================================
# the foreign half
# =======================================================================================

def _census():
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_dk.json"), encoding="utf-8"))
    dims = d["id"]
    cats = [list(d["dimension"][x]["category"]["index"]) for x in dims]
    sizes = d["size"]
    rows = []
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rows.append([cats[j][idx[j]] for j in range(len(dims))] + [v])
    df = pd.DataFrame(rows, columns=dims + ["value"])
    df = df[df["geo"].isin(NUTS3)].copy()
    df["unit"] = df["geo"]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign citizens at NUTS 3 (no.py's, other.dk)."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "DK")]
    total_for = cen[cen["citizen"] == "FOR"]["value"].sum()
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{total_for:,.0f} foreign residents ({100 * covered / total_for:.2f}%)")
    # EUROSTAT'S `FOR` CAN HOLD THE COUNTRY'S OWN RECOGNISED NON-CITIZENS (Latvia's 190,544
    # `RNC`), and scaling the named citizenships up to `FOR` would hand them to the named
    # countries (spec §12, "EUROSTAT'S `FOR`"). Denmark's `RNC` is 0 (2026-09-14); asserted, with
    # `lv.py::_foreign_half`'s coverage band.
    rnc = cen[cen["citizen"] == "RNC"]["value"].sum()
    if rnc:
        sys.exit(f"!! cens_21ctz_r3 has {rnc:,.0f} recognised non-citizens (RNC) inside FOR; scale "
                 "to FOR - RNC and draw RNC separately, as lv.py::_foreign_half does")
    if not 0.9 < covered / total_for <= 1.0001:
        sys.exit("!! the named citizenships do not match FOR; read the census file before scaling")

    comp, unmapped = {}, []
    for iso in sorted(leaf["citizen"].unique()):
        pn = origin.PEW_BY_ISO.get(iso, "MISSING")
        if pn == "MISSING":
            unmapped.append(iso)
            continue
        row = origin.REGIONAL.get(iso) if pn is None else (
            {f: float(pew.loc[pn, f]) for f in fams} if pn in pew.index else None)
        if row is None:
            unmapped.append(f"{iso} ({pn})")
            continue
        comp[iso] = origin.composition(iso, row, "other.dk")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    rows = []
    for unit, g in leaf.groupby("unit"):
        base = g["value"].sum()
        if base <= 0:
            continue
        target = cen[(cen["unit"] == unit) & (cen["citizen"] == "FOR")]["value"].sum()
        scale = target / base
        acc = {}
        for iso, v in g.groupby("citizen")["value"].sum().items():
            if v <= 0:
                continue
            for node, s in comp[iso].items():
                acc[node] = acc.get(node, 0.0) + v * scale * s
        for node, c in acc.items():
            rows.append((unit, node, c))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    print(f"  foreign half: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes")
    return df


# =======================================================================================
# the roll, printed beside the survey
# =======================================================================================

def _roll():
    """{year: DataFrame[unit, members, pop]} from KM6, kommune joined to landsdel by code."""
    import geopandas as gpd
    info = json.load(open(os.path.join(RAW, "dst_km6_info.json"), encoding="utf-8"))
    fk = {v["id"]: v["text"] for x in info["variables"] if x["id"] == "FKMED" for v in x["values"]}
    member = [k for k, t in fk.items() if t.startswith("Medlem")]
    if len(member) != 1:
        sys.exit(f"!! KM6 FKMED codes {fk}")
    d = pd.read_csv(os.path.join(RAW, "dst_km6.csv"), sep=";", dtype=str)
    d.columns = [c.strip().upper() for c in d.columns]
    d["INDHOLD"] = pd.to_numeric(d["INDHOLD"], errors="coerce")
    d["lau"] = d["KOMK"].str.strip().str.zfill(3)
    lau = gpd.read_file(LAU, ignore_geometry=True)
    unit_of = dict(zip(lau["lau"], lau["unit"]))
    stray = sorted(set(d["lau"]) - set(unit_of))
    if stray:
        sys.exit(f"!! KM6 kommune codes not in dk_lau.gpkg: {stray}")
    d["unit"] = d["lau"].map(unit_of)
    out = {}
    for y in ROLL_YEARS:
        g = d[d["TID"] == y]
        pop = g.groupby("unit")["INDHOLD"].sum()
        mem = g[g["FKMED"] == member[0]].groupby("unit")["INDHOLD"].sum()
        out[y] = pd.DataFrame({"members": mem, "pop": pop})
    return out


# =======================================================================================
# build
# =======================================================================================

def build():
    import dk2024

    print("checking the region set, the round 9 recode and the Danish card…")
    _check_level()
    _check_recode()
    _check_card()

    print("\nESS…")
    share, nat, fine, small, answered, n = _citizen_shares()

    print("\nEurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat_cit = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    other_codes = sorted(set(cen["citizen"]) - {"TOTAL", "NAT", "FOR"}
                         - set(cen.loc[cen["citizen"].str.fullmatch(r"[A-Z]{2}"), "citizen"]))
    print(f"  {pop.sum():,.0f} people = {nat_cit.sum():,.0f} Danish citizens + {for_.sum():,.0f} "
          f"foreign + {pop.sum() - nat_cit.sum() - for_.sum():,.0f} other, over {len(pop)} "
          f"landsdele; aggregate codes present: {other_codes[:12]}")
    for code in ("STLS", "UNK"):
        if code in set(cen["citizen"]):
            print(f"    {code}: {cen[cen['citizen'] == code]['value'].sum():,.0f}")
    if sorted(pop.index) != sorted(NUTS3):
        sys.exit(f"!! census landsdele {sorted(pop.index)}")
    if POP_2021 is None:
        print(f"  !! POP_2021 is unset; this census has {pop.sum():,.0f}")
    elif int(pop.sum()) != POP_2021:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Denmark's {POP_2021:,}")

    print("\ncomposing the citizen half…")
    comp = _compose(share, nat, fine, small)

    rows = []
    for u in sorted(NUTS3):
        n_cit = float(nat_cit[u]) * answered
        r = u[:4]
        for cat, s in comp.loc[u].items():
            if s <= 0:
                continue
            if cat in fine:
                note = (f"ESS rounds 5-7 and 9 at NUTS 2 ({r} {NUTS2[r]}); round 9's region codes "
                        "recoded from DST's order"
                        + ("; drawn by OVERRIDE, see sources/dk.py" if cat in OVERRIDE else ""))
            else:
                note = ("share of the landsdel's residual at the national proportions "
                        "(§9bi, split-half not passed)")
            for sub, w in dk2024.split(cat).items():
                rows.append((u, NUTS3[u], sub, s * w * n_cit,
                             note + ("" if sub == cat else f"; {dk2024.SPLIT_NOTE}")))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count", "note"])
    cit["geo_level"] = "nuts3"
    cit["basis"] = "self_id"
    cit["year"] = 2021
    cit["source_id"] = "ess_r5_r7_r9_nuts2"
    cit = cit[COLUMNS]
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat_cit.sum():.2f}% of citizens; the rest declined)")
    unknown = sorted(set(cit["source_category"]) - set(dk2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    print("\nforeign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts3"
    ext["geo_name"] = ext["geo_id"].map(NUTS3)
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_derived"
    ext["year"] = 2021
    ext["source_id"] = "cens_21ctz_r3_x_pew2020"
    ext = ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
               "source_id"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cit.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    ext.to_csv(OUT_FOREIGN + ".tmp", index=False)
    os.replace(OUT_FOREIGN + ".tmp", OUT_FOREIGN)

    drawn = cit["count"].sum() + ext["count"].sum()
    print(f"\nwrote {OUT}  ({len(cit):,} rows, {cit['source_category'].nunique()} source categories)")
    print(f"wrote {OUT_FOREIGN}  ({len(ext):,} rows, {ext['node'].nunique()} nodes)")
    print(f"drawn {drawn:,.0f} of {pop.sum():,.0f}, {100 * drawn / pop.sum():.2f}%")

    both = pd.concat([
        cit.assign(node=cit["source_category"].map(dk2024.resolve))[["geo_id", "node", "count"]],
        ext[["geo_id", "node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(22).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    byu = both.groupby(["geo_id", "node"])["count"].sum().unstack(fill_value=0.0)
    tot = byu.sum(axis=1)
    fam = lambda pre: byu[[c for c in byu.columns if c == pre or c.startswith(pre + ".")]].sum(axis=1)
    print("\nper landsdel (drawn, both halves):")
    print(f"    {'landsdel':<24}{'lutheran':>10}{'unaffil':>9}{'islam':>8}{'catholic':>9}"
          f"{'orthodox':>9}{'foreign':>9}")
    for u in sorted(NUTS3):
        t = tot[u]
        print(f"    {NUTS3[u]:<24}{100 * byu.loc[u].get('christianity.lutheran', 0) / t:9.2f}%"
              f"{100 * byu.loc[u].get('unaffiliated', 0) / t:8.2f}%{100 * fam('islam')[u] / t:7.2f}%"
              f"{100 * fam('christianity.catholic')[u] / t:8.2f}%"
              f"{100 * (fam('christianity.orthodox')[u] + fam('christianity.oriental')[u]) / t:8.2f}%"
              f"{100 * for_[u] / pop[u]:8.2f}%")

    # ---- THE ROLL BESIDE THE SURVEY. Not a validation (sources/se.md §4, sources/no.md §8).
    from scipy.stats import spearmanr
    roll = _roll()
    r21 = roll["2021"]
    for u in sorted(NUTS3):
        if abs(r21.loc[u, "pop"] / pop[u] - 1.0) > 0.02:
            sys.exit(f"!! {u}: KM6's 2021 population {r21.loc[u, 'pop']:,.0f} against the census's "
                     f"{pop[u]:,.0f}; the kommune join is wrong")
    print("\nTHE ROLL (DST KM6, 1 January) BESIDE THE SURVEY:")
    for y, r in roll.items():
        print(f"  {y}: {100 * r['members'].sum() / r['pop'].sum():.2f}% of Denmark in the Church of "
              f"Denmark ({r['members'].sum():,.0f} of {r['pop'].sum():,.0f})")
    lut = byu.get("christianity.lutheran", pd.Series(0.0, index=byu.index))
    print(f"  on this map: christianity.lutheran {100 * lut.sum() / drawn:.2f}%, "
          f"unaffiliated {100 * byu.get('unaffiliated', pd.Series(0.0)).sum() / drawn:.2f}%")
    r14 = roll["2014"]
    reg = pd.DataFrame({
        "roll14": r14.groupby(lambda u: u[:4]).sum().pipe(lambda x: 100 * x["members"] / x["pop"]),
        "survey": 100 * share[PROTESTANT],
        "survey_none": 100 * share[NO_RELIGION],
    })
    print(f"    {'region':<14}{'roll 2014':>10}{'survey Prot':>13}{'survey none':>13}")
    for r, x in reg.sort_values("roll14", ascending=False).iterrows():
        print(f"    {NUTS2[r]:<14}{x['roll14']:9.1f}%{x['survey']:12.2f}%{x['survey_none']:12.2f}%")
    sp = spearmanr(reg["roll14"], reg["survey"])
    print(f"  Spearman, 5 regions, roll 2014 against the survey's Protestant share among citizens: "
          f"{sp.correlation:+.3f}")
    land = pd.DataFrame({"roll14": 100 * r14["members"] / r14["pop"],
                         "roll21": 100 * r21["members"] / r21["pop"]})
    print(f"    {'landsdel':<24}{'roll 2014':>10}{'roll 2021':>10}")
    for u, x in land.sort_values("roll14", ascending=False).iterrows():
        print(f"    {NUTS3[u]:<24}{x['roll14']:9.1f}%{x['roll21']:9.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
