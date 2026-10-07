"""United Arab Emirates: nobody is asked their religion. Emiratis are drawn on Islam, and everyone
else by country of origin, in each of the seven emirates, around 2024.

Reads, with everything pinned below and in `sources/ae.md`:

  * each emirate's own statistics office for its population, and its latest count of Emiratis,
    which are of mixed years (`TOTALS`, `EMIRATIS`);
  * the Federal Competitiveness and Statistics Centre (FCSC) for the national population at the
    end of 2024, and GLMM's FCSC-based series of Emiratis 2010-2022 (`EMIRATI_SERIES`);
  * UN DESA, *International Migrant Stock 2024* (data/raw/mr/, shared): the UAE's migrants by
    origin, mid-2024;
  * Pew Research Center, *Religious Composition 2010-2020* (data/raw/estimates/pew.zip).

Writes data/normalized/ae.csv (Emiratis, one row per emirate) and data/normalized/ae_foreign.csv
(non-Emiratis, already on nodes). `sources/ae.md` is the record in prose.

## NOBODY IS ASKED

The 2005 census form has no religion item (sources.md §11ao's Gulf re-check, UNSD `ARE2005en.pdf`),
and it was the last federal census. The emirate censuses since (Abu Dhabi 2023, Sharjah 2015 and
2022, Ajman 2017, Ras Al Khaimah 2023) publish no religion table found, the UAE is in no Arab
Barometer wave, and the UNSD religion table has no UAE row. Nothing like Kuwait's PACI
religion-by-nationality answer was found (sources/ae.md §1 lists what was searched). Built the
Oman and Saudi way, on Anita's priority line (2026-09-15) and the Maghreb and Mauritania rulings.

## EMIRATIS ARE ALL DRAWN ON ISLAM

No source counts an Emirati who is not Muslim. No Sunni/Shia split (asks 040 and 043: Gulf
citizens stay on one Islam where nothing places a sect).

## THE POPULATION, EMIRATE BY EMIRATE

Three emirates publish a 2024 total (Abu Dhabi, Dubai, Fujairah). The other four's newest totals
are older (Sharjah 2022, Ajman 2017, Ras Al Khaimah 2015, Umm Al Quwain 2005), so they are scaled
together, one factor for all four, until the seven sum to FCSC's 2024 national total. That keeps
their shares of each other as their offices last printed them and makes the country total the
federal one.

Emiratis: each emirate's newest published count, grown to 2024 at the national growth of Emiratis
(GLMM's series, built by FCSC births less deaths from the 2005 census; 2023 and 2024 extended at
2022's rate). Non-Emiratis are each emirate's 2024 total less its Emiratis. The grown counts sum
to 9% more Emiratis than GLMM's national series; printed, not forced (sources/ae.md §2).

## NON-EMIRATIS: ONE NATIONAL MIX OF ORIGINS

UN DESA 2024 names 33 origins for the UAE's 8,157,000 migrants; its `Others` (248,004, 3.0%) are
taken to be like the named ones. Every emirate's non-Emiratis take that one mix: DESA has nothing
by emirate. Saudi Arabia's split by sex was tried and dropped: DESA gives the UAE's men and women
nearly the same mix (Christians 9.0% of the men's, 9.7% of the women's), and the emirates' sex
splits are of mixed years, so it moved almost nothing and added a fitted Dubai residual.

Religion by origin: Pew 2020 through `taxonomy/origin_religion.py`, Muslim branches folded to
`islam`, Pew's unplaced `Other religions` on `other.ae`. Two named corrections:

  * **India's Hindu share comes from Pew's own UAE estimate**, as Saudi Arabia and Oman: Pew's
    *Faith on the Move* (2012) takes Egypt's census as its guide and counts more of the Indians in
    Muslim-majority Middle Eastern countries as Muslim than India's own share. India's Hindu share
    is set so the layer's Hindus equal Pew 2020's UAE share of the 2024 total; the Hindus removed
    are drawn on Islam.
  * **The Gulf rule** (2026-10-03, sources.md §gulf-2026-10-03, `origin_religion.gulf_christian_hindu`):
    then Indians are moved from Hindu to Christian until the layer's Christians / (Christians +
    Hindus) equals Pew's UAE row (0.549; the layer gave 0.405). 321,606 move; Indians end 20.6%
    Hindu, 10.2% Christian, 65.9% Muslim. The non-Muslim total and every other family are unchanged.
  * **Origins in the other Gulf states (Bahrain, Kuwait, Qatar, Saudi Arabia) are drawn on Islam.**
    Pew's rows for them are mostly their own foreign residents, while a migrant from one of them is
    most likely its citizen, and Kuwait's register counts its citizens 99.978% Muslim
    (sources/kw.md §1). 73,719 people in DESA.

`christian_witness`: the layer's Christians against Pew 2020's UAE share, inside `CHRISTIAN_BAND`,
which only catches an error of several times (Oman's band; before the Gulf rule Oman came out 0.31
and the UAE 0.56, now 0.69 and 0.76).

Usage:
    python sources/ae.py --fetch    nothing to fetch beyond the shared files; checks they exist
    python sources/ae.py            rebuild data/normalized/ae.csv and ae_foreign.csv
"""

import io
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from afrobarometer import round_within_rows
import origin_religion as origin

PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
DESA = os.path.join(ROOT, "data", "raw", "mr",
                    "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
OUT = os.path.join(ROOT, "data", "normalized", "ae.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "ae_foreign.csv")
OTHER_NODE = "other.ae"

EMIRATES = {"AE01": "Abu Dhabi", "AE02": "Dubai", "AE03": "Sharjah", "AE04": "Ajman",
            "AE05": "Umm Al Quwain", "AE06": "Ras Al Khaimah", "AE07": "Fujairah"}

# FCSC, 2024 (Gulf News, "UAE population hits 11.3 million in 2024", 2025): total, men, women
NATIONAL_2024 = (11_294_243, 7_235_074, 4_059_169)
if NATIONAL_2024[1] + NATIONAL_2024[2] != NATIONAL_2024[0]:
    raise SystemExit("FCSC's 2024 men and women do not sum to its total")

# Each emirate's newest total: (year, people, office and where read). The 2024 ones are kept as
# printed; the rest are scaled together to close on NATIONAL_2024 (`totals_2024`).
TOTALS = {
    "AE01": (2024, 4_135_985, "Statistics Centre Abu Dhabi, release 'Abu Dhabi population in 2024 "
                              "grows 7.5% to reach 4.14m' (scad.gov.ae)"),
    "AE02": (2024, 3_863_600, "Dubai Statistics Center, end of 2024 (dubai.ae, Population and "
                              "Vital Statistics)"),
    "AE07": (2024, 314_829, "Fujairah Statistics Centre, mid-2024 (u.ae, Fujairah)"),
    "AE03": (2022, 1_800_000, "Department of Statistics and Community Development, Sharjah Census "
                              "2022, as released (rounded; Gulf News, 2023)"),
    "AE04": (2017, 504_846, "Statistics and Competitiveness Centre, Ajman, 2017 census (u.ae, Ajman)"),
    "AE06": (2015, 345_000, "Ras Al Khaimah Government Media Office, 2015 (u.ae, Ras Al Khaimah)"),
    "AE05": (2005, 49_159, "2005 census, December (u.ae, Umm Al Quwain; GLMM's NBS table)"),
}
CURRENT = {"AE01", "AE02", "AE07"}

# Each emirate's newest count of Emiratis: (year, count).
EMIRATIS = {
    "AE01": (2016, 551_535),    # SCAD, mid-2016 (GLMM, estimates as of March 2018)
    "AE02": (2016, 233_430),    # DSC, 2016 (GLMM, estimates as of March 2018)
    "AE03": (2022, 208_000),    # Sharjah Census 2022, as released (rounded)
    "AE04": (2005, 39_231),     # 2005 census (GLMM's NBS table); nothing newer found
    "AE05": (2010, 17_482),     # mid-2010 (u.ae, Umm Al Quwain)
    "AE06": (2015, 127_000),    # RAK Government Media Office, 2015 (u.ae)
    "AE07": (2016, 87_814),     # Fujairah, 2016 (GLMM, estimates as of March 2018)
}

# GLMM, "UAE - A methodology for estimating the Emirati and non-Emirati populations (2010-2022)",
# from FCSC births and deaths; 2005 is the census.
EMIRATI_SERIES = {2005: 825_495, 2010: 962_348, 2011: 992_898, 2012: 1_024_316, 2013: 1_055_978,
                  2014: 1_087_864, 2015: 1_120_010, 2016: 1_152_057, 2017: 1_183_086,
                  2018: 1_213_794, 2019: 1_244_638, 2020: 1_274_435, 2021: 1_303_383,
                  2022: 1_331_683}

DESA_YEAR = 2024.0
DESA_WORLD = (8_157_000, 5_491_000, 2_666_000)
DESA_ORIGINS = {
    "Eritrea": "ER", "Ethiopia": "ET", "Somalia": "SO", "South Sudan": "SS", "Chad": "TD",
    "Egypt": "EG", "Morocco": "MA", "Sudan": "SD", "Tunisia": "TN", "Nigeria": "NG",
    "Afghanistan": "AF", "Bangladesh": "BD", "India": "IN", "Nepal": "NP", "Pakistan": "PK",
    "Sri Lanka": "LK", "Indonesia": "ID", "Philippines": "PH", "Thailand": "TH",
    "Bahrain": "BH", "Jordan": "JO", "Kuwait": "KW", "Lebanon": "LB", "Qatar": "QA",
    "Saudi Arabia": "SA", "State of Palestine": "PS", "Syrian Arab Republic": "SY",
    "Türkiye": "TR", "Yemen": "YE", "United Kingdom": "GB", "France": "FR",
    "Netherlands": "NL", "United States of America": "US"}
DESA_UNNAMED = "Others"
GULF_ON_ISLAM = {"BH", "KW", "QA", "SA"}
INDIA = "IN"

CHRISTIAN_BAND = (0.2, 2.0)
# note_public's figures, measured 2026-10-03 under the Gulf rule, asserted
NOTE = dict(emiratis=1519227, non_emiratis=9775016, foreign_muslim=7218209, non_muslim=2556807,
            christians=1224962, hindus=1005866, buddhists=152171, india_hindu=0.2062,
            india_christian=0.1022)


def growth_to_2024(year):
    s = dict(EMIRATI_SERIES)
    rate = s[2022] / s[2021]
    s[2023] = s[2022] * rate
    s[2024] = s[2023] * rate
    if year not in s:
        raise SystemExit(f"no Emirati series value for {year}")
    return s[2024] / s[year], s[2024]


def totals_2024():
    fixed = sum(TOTALS[c][1] for c in CURRENT)
    old = sum(v[1] for c, v in TOTALS.items() if c not in CURRENT)
    k = (NATIONAL_2024[0] - fixed) / old
    out = {}
    for c, (_y, n, _s) in TOTALS.items():
        out[c] = float(n) if c in CURRENT else n * k
    print(f"  totals: Abu Dhabi, Dubai and Fujairah as printed for 2024 ({fixed:,}); the other four "
          f"({old:,} at their own years) scaled by {k:.4f} to close on FCSC's {NATIONAL_2024[0]:,}")
    if not 1.0 <= k <= 1.3:
        raise SystemExit("the four older totals need a factor outside 1.0-1.3; read sources/ae.md §2")
    return out


def emiratis_2024():
    out = {}
    for c, (y, n) in EMIRATIS.items():
        g, _ = growth_to_2024(y)
        out[c] = (n * g,)
    _, nat = growth_to_2024(2022)
    tot = sum(v[0] for v in out.values())
    print(f"  Emiratis grown to 2024: {tot:,.0f}; GLMM's series extended gives {nat:,.0f} "
          f"(ratio {tot / nat:.3f}; printed, not forced)")
    return out


def desa_mixes():
    with zipfile.ZipFile(DESA) as z:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for n in z.namelist():
                if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                    out.writestr(n, z.read(n))
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name="Table 1", header=None, engine="openpyxl")
    hdr = next(i for i in range(2, 20)
               if any("of destination" in str(x) for x in df.iloc[i])
               and any("of origin" in str(x) for x in df.iloc[i]))
    cols = list(df.iloc[hdr])
    names = [str(x).strip() for x in cols]
    dcol = next(i for i, x in enumerate(names) if "of destination" in x)
    ocol = next(i for i, x in enumerate(names) if "of origin" in x)
    ccol = next(i for i, x in enumerate(names) if x == "Location code of origin")
    ycols = [i for i, x in enumerate(cols) if str(x) in (str(DESA_YEAR), str(int(DESA_YEAR)))]
    if len(ycols) != 3:
        raise SystemExit(f"expected three {DESA_YEAR:g} columns (both sexes, men, women), got {ycols}")
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "United Arab Emirates"]
    name = m[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    world = tuple(int(pd.to_numeric(m.loc[name == "World", y]).iloc[0]) for y in ycols)
    if world != DESA_WORLD:
        raise SystemExit(f"DESA 2024 world stock for the UAE is {world}, pinned {DESA_WORLD}")
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[(code < 900) | (name == DESA_UNNAMED)]
    cname = ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    if set(cname) != set(DESA_ORIGINS) | {DESA_UNNAMED}:
        raise SystemExit(f"DESA's origins for the UAE changed: "
                         f"{sorted(set(cname) ^ (set(DESA_ORIGINS) | {DESA_UNNAMED}))}")
    mixes = []
    for j, y in enumerate(ycols):
        stock = dict(zip(cname, pd.to_numeric(ctry[y]).astype(int)))
        if abs(sum(stock.values()) - world[j]) > 2:
            raise SystemExit(f"DESA's origins sum to {sum(stock.values()):,}, world {world[j]:,}")
        mixes.append({DESA_ORIGINS[k]: v for k, v in stock.items() if k != DESA_UNNAMED})
        if j == 0:
            named = sum(mixes[0].values())
            print(f"  UN DESA 2024: {world[0]:,} migrants in the UAE ({world[1]:,} men, {world[2]:,} "
                  f"women); 33 named origins {named:,}, `Others` {stock[DESA_UNNAMED]:,} "
                  f"({stock[DESA_UNNAMED] / world[0]:.1%}) taken to be like them")
    both, men, women = mixes
    for k in both:
        if abs(men[k] + women[k] - both[k]) > 2:
            raise SystemExit(f"DESA {k}: men {men[k]:,} + women {women[k]:,} != {both[k]:,}")
    return both, men, women


def pew_table():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    return t[t["Year"] == 2020].set_index("Country")


def fold(comp):
    out = {}
    for node, s in comp.items():
        node = "islam" if node.startswith("islam") else node
        out[node] = out.get(node, 0.0) + s
    return out


def composition(pew, iso):
    if iso in GULF_ON_ISLAM:
        return {"islam": 1.0}
    pn = origin.PEW_BY_ISO[iso]
    if pn is None or pn not in pew.index:
        raise SystemExit(f"Pew has no row for {iso}")
    return fold(origin.composition(iso, {f: float(pew.loc[pn, f]) for f in origin.FAMILIES},
                                   OTHER_NODE))


def pew_share(pew, family):
    return float(pew.loc["United Arab Emirates", family]) / float(pew.loc["United Arab Emirates",
                                                                         "Population"])


def mix(weights, comps):
    tot = sum(weights.values())
    out = {}
    for k, w in weights.items():
        for node, s in comps[k].items():
            out[node] = out.get(node, 0.0) + w / tot * s
    return out


def main():
    for p in (PEW, DESA):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (shared; see sources/mr.py and estimates.py)")
    if "--fetch" in sys.argv:
        print("nothing to fetch: every figure is pinned here, and the two shared files exist")

    tot = totals_2024()
    emi = emiratis_2024()
    non = {c: tot[c] - emi[c][0] for c in EMIRATES}
    if min(non.values()) <= 0:
        raise SystemExit(f"an emirate has more Emiratis than people: {non}")

    print("\n  per emirate, 2024: total, Emiratis, non-Emiratis")
    for c in EMIRATES:
        print(f"      {EMIRATES[c]:<16} {tot[c]:>11,.0f} {emi[c][0]:>10,.0f} {non[c]:>11,.0f}   "
              f"total {TOTALS[c][0]}, Emiratis {EMIRATIS[c][0]}")

    # ---- compositions ----
    pew = pew_table()
    db, dm, dw = desa_mixes()
    keys = set(db)
    comps = {k: composition(pew, k) for k in keys}
    gulf = sum(db[k] for k in GULF_ON_ISLAM)
    print(f"  Gulf origins on Islam: {gulf:,} in DESA")

    non_all = sum(non.values())
    people = {k: non_all * v / sum(db.values()) for k, v in db.items()}
    pop = NATIONAL_2024[0]
    target = pew_share(pew, "Hindus") * pop
    others = sum(people[k] * comps[k].get("hinduism", 0.0) for k in keys if k != INDIA)
    h_row = comps[INDIA].get("hinduism", 0.0)
    h = (target - others) / people[INDIA]
    print(f"\n  India ({people[INDIA]:,.0f} in the layer): Pew 2020's UAE Hindus are "
          f"{pew_share(pew, 'Hindus'):.3%}, {target:,.0f} of the 2024 total; other origins give "
          f"{others:,.0f}, so Indians are {h:.2%} Hindu (India's own row {h_row:.2%}, which would give "
          f"{others + people[INDIA] * h_row:,.0f})")
    if not 0.0 < h < h_row:
        raise SystemExit("India's Hindu share from Pew's UAE estimate is not between 0 and India's row")
    comps[INDIA]["islam"] = comps[INDIA].get("islam", 0.0) + (h_row - h)
    comps[INDIA]["hinduism"] = h
    # The Gulf rule (origin_religion.gulf_christian_hindu): Christians / (Christians + Hindus) of the
    # layer raked to Pew 2020's UAE row, on Indians
    ratio_cw = pew_share(pew, "Christians") / (pew_share(pew, "Christians") + pew_share(pew, "Hindus"))
    moved = origin.gulf_christian_hindu(people, comps, INDIA, ratio_cw)
    if moved < origin.GULF_MATERIAL * pop:
        raise SystemExit(f"the Gulf rule moves {moved:,.0f}, under {origin.GULF_MATERIAL:.0%} of the "
                         "country; it should not be applied here")
    c_in = sum(s for n, s in comps[INDIA].items() if n.startswith("christianity"))
    print(f"  Gulf rule: Christians / (Christians + Hindus) raked to Pew's UAE {ratio_cw:.3f}; "
          f"{moved:,.0f} Indians moved from Hindu to Christian (Indians now {comps[INDIA]['hinduism']:.2%} "
          f"Hindu, {c_in:.2%} Christian, {comps[INDIA]['islam']:.2%} Muslim)")
    for k, c in comps.items():
        if abs(sum(c.values()) - 1) > 1e-9:
            raise SystemExit(f"composition {k} sums to {sum(c.values())}")
    mix_b, mix_m, mix_f = mix(db, comps), mix(dm, comps), mix(dw, comps)
    nodes = sorted(set(mix_b))
    chr_nodes = [n for n in nodes if n.startswith("christianity")]
    # By sex, as Saudi Arabia was built, the two mixes barely differ here (DESA's UAE origins are
    # close to the same shares for both sexes), so one mix is drawn; printed so a later reader sees why.
    print("  Christian share: non-Emiratis {:.2%} (men's mix {:.2%}, women's {:.2%}; one mix drawn)".format(
        *(sum(x.get(n, 0) for n in chr_nodes) for x in (mix_b, mix_m, mix_f))))

    # ---- per emirate ----
    tot_int = {c: int(round(tot[c])) for c in EMIRATES}
    emi_int = {c: int(round(emi[c][0])) for c in EMIRATES}
    non_int = {c: tot_int[c] - emi_int[c] for c in EMIRATES}
    em = pd.DataFrame({c: {n: non_int[c] * mix_b.get(n, 0) for n in nodes}
                       for c in EMIRATES}).T[nodes]
    fcounts = round_within_rows(em)
    for c in EMIRATES:
        drawn = int(fcounts.loc[c].sum()) + emi_int[c]
        if drawn != tot_int[c]:
            raise SystemExit(f"{c}: drawn {drawn:,} against {tot_int[c]:,}")

    total_drawn = int(fcounts.to_numpy().sum()) + sum(emi_int.values())
    chr_ = int(fcounts[chr_nodes].sum().sum())
    ratio = (chr_ / total_drawn) / pew_share(pew, "Christians")
    print(f"\n  witness: {chr_:,} Christians, {chr_ / total_drawn:.3%} of {total_drawn:,}; Pew 2020 has "
          f"{pew_share(pew, 'Christians'):.3%} for everyone in the UAE; ratio {ratio:.2f}, band "
          f"{CHRISTIAN_BAND}")
    for fam, node in (("Buddhists", "buddhism"), ("Religiously_unaffiliated", "unaffiliated"),
                      ("Jews", "judaism"), ("Hindus", "hinduism"), ("Muslims", "islam")):
        got = int(fcounts[node].sum()) if node in fcounts else 0
        if node == "islam":
            got += sum(emi_int.values())
        print(f"      {fam}: {got:,} drawn against Pew's share of the total, "
              f"{pew_share(pew, fam) * total_drawn:,.0f} (not asserted)")
    if not CHRISTIAN_BAND[0] <= ratio <= CHRISTIAN_BAND[1]:
        raise SystemExit("the layer's Christians are outside the band; read sources/ae.md §3")
    print("  per emirate, % Christian and % non-Muslim of everyone:")
    for c in EMIRATES:
        allp = tot[c]
        cc = float(fcounts.loc[c, chr_nodes].sum())
        nm = float(fcounts.loc[c].sum() - fcounts.loc[c].get("islam", 0))
        print(f"      {EMIRATES[c]:<16} {100 * cc / allp:5.2f}%  {100 * nm / allp:5.2f}%")

    # ---- write ----
    out = pd.DataFrame({"geo_id": list(EMIRATES), "geo_level": "emirate",
                        "geo_name": [EMIRATES[c] for c in EMIRATES],
                        "source_category": "Emirati citizens",
                        "count": [emi_int[c] for c in EMIRATES], "basis": "estimate",
                        "year": 2024, "source_id": "ae_emirates_emiratis_2024",
                        "note": "no source asks religion; every Emirati is drawn on Islam (sources/ae.py), "
                                "each emirate's newest count of Emiratis grown to 2024"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    ext = fcounts.stack().rename("count").reset_index()
    ext.columns = ["geo_id", "node", "count"]
    ext = ext[ext["count"] > 0].copy()
    ext["geo_level"] = "emirate"
    ext["geo_name"] = ext["geo_id"].map(EMIRATES)
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_sex_derived"
    ext["year"] = 2024
    ext["source_id"] = "emirates2024_non_emiratis_x_undesa2024_x_pew2020"
    ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
         "source_id"]].to_csv(OUT_FOREIGN, index=False, encoding="utf-8")

    fx = ext.groupby("node")["count"].sum().sort_values(ascending=False)
    muslim_f = int(fx.get("islam", 0))
    non_muslim = int(ext["count"].sum()) - muslim_f
    emi_tot = sum(emi_int.values())
    print(f"\nwrote {OUT} ({emi_tot:,} Emiratis) and {OUT_FOREIGN} ({int(ext['count'].sum()):,} "
          f"non-Emiratis, {ext['node'].nunique()} nodes)")
    print(f"    drawn Muslim {(emi_tot + muslim_f) / total_drawn:.3%}; non-Muslim {non_muslim:,}; Pew 2020 "
          f"for everyone: {pew_share(pew, 'Muslims'):.3%} Muslim")
    print("    non-Emiratis: " + ", ".join(f"{n} {int(v):,}" for n, v in fx.items()))
    got = dict(emiratis=emi_tot, non_emiratis=int(ext["count"].sum()), foreign_muslim=muslim_f,
               non_muslim=non_muslim, christians=chr_, hindus=int(fx.get("hinduism", 0)),
               buddhists=int(fx.get("buddhism", 0)), india_hindu=round(comps[INDIA]["hinduism"], 4),
               india_christian=round(c_in, 4))
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
