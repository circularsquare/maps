"""Sierra Leone 2015 PHC, main language, by district -> data/normalized/sl.csv, plus CLEAR
Global's district shares of the same census -> data/normalized/sl_clear.csv.

    python sources/sl_census.py [--fetch]

WHAT STATISTICS SIERRA LEONE PUBLISHES. The 2015 Population and Housing Census asked every
household member's main language (P10, "main language NAME speaks") and a secondary language
(P11). The *National Analytical Report* (584 pp) prints:
  Table 3.22, PDF p134   main language, NATIONAL, in persons: 15 local languages, "Foreign
                         language" and "Other", 6,954,702 people (the household population less
                         those with no answer).                                     -> the counts
  Table 3.23, PDF p135   for each region and district, the three most common main languages
                         with their % of the household population, and "Others".   -> the check
No thematic report covers language (the 15 reports in the Wayback CDX for
statistics.sl/.../Census/2015/ are agriculture, children, disability, economic, education,
elderly, gender, housing, life tables, migration, mortality, nuptiality, population structure,
projections, poverty, Ebola; none has a language table), and the Census Atlas maps ethnicity
by chiefdom but not language. So no full district x language table is published.

CLEAR GLOBAL (https://data.humdata.org/dataset/sierra-leone-languages, CC BY-SA 4.0): "main
language spoken" proportions for the country, 5 regions and 16 districts (the 2017 set, COD-AB
pcodes), made from the IPUMS International 10% sample of this same census (extract
ipumsi_00291). 18 named languages + `Unknown`. It also has a 17th row, SL05XXX "northwestern:
level 2 unknown" (94.5% Temne): sample households CLEAR could not put in a North Western
district. It has no population, so it is left out; check 7 shows what that costs.

THE BUILD. The district x language cells come from CLEAR's shares, fitted to the census's own
published margins (iterative proportional fitting):
  rows     each 2015 district's household population (religiondots' sl_lookup.csv, which is
           Table 2.2 x 7,076,119 / 7,092,113; religiondots/sources/sl.md §2)
  columns  Table 3.22's national count for each language; "Foreign language" split into
           English, French and Arabic by CLEAR's national ratio; "Other" (5,499) and the
           121,417 with no answer (7,076,119 - 6,954,702) share CLEAR's `Unknown` column in
           their national ratio. The no-answer column is then dropped.
The 2015 districts are 14; CLEAR's are 16. Each 2015 district's starting shares are its 2017
pieces' shares weighted by their Kontur population (Bombali = Bombali + Karene's Bombali
chiefdoms, Port Loko = Port Loko + Karene's Port Loko chiefdoms, Koinadugu = Koinadugu +
Falaba). The hex -> 2017 district join is sources/sl_place.py's.

So every language's national total is the census's (Table 3.22) and every district's total is
the census's; only the split of a district among languages comes from the 10% sample, and check
6 holds it to Table 3.23's 42 published district cells.

CHECKS (all must pass):
  1. the PDF is the pinned 584-page report (digest, %%EOF)
  2. Table 3.22 parsed off p134 = the transcription; its rows sum to its 6,954,702
  3. Table 3.23 parsed off p135 = the transcription (19 rows: nation, 4 regions, 14 districts)
  4. the 14 districts' household population sums to 7,076,119 and joins religiondots'
     sl_hexes units both ways
  5. CLEAR: the 16 district pcodes (+ SL05XXX) are COD-AB sle_admin2's; shares sum to 1
  6. the fitted cells against Table 3.23's 42 district cells (and 12 region cells), % of the
     household population: within 2.0 points
  7. before fitting, CLEAR's national shares against Table 3.22: within 1.0 point
"""
import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "sl"
OUT = HERE / "data" / "normalized" / "sl.csv"
OUT_CLEAR = HERE / "data" / "normalized" / "sl_clear.csv"
PLACE = HERE / "data" / "geo" / "sl" / "sl_hexes.gpkg"
NAME = "2015_census_national_analytical_report.pdf"
PDF = RAW / NAME
_PATH = "www.statistics.sl/images/StatisticsSL/Documents/Census/2015/" + NAME
ORIGINAL = "https://" + _PATH
WAYBACK = "https://web.archive.org/web/20201114075521id_/https://" + _PATH
PDF_BYTES = 12_614_169
SHA256 = "02ebfb8eb4b44299ad1032e5c0c385f1418821175b38c4974eeee02270c50077"
PAGES = 584
P_T322, P_T323 = 133, 134          # 0-based
ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "sl" / "shp" / "sle_admin2.shp"

CLEAR_BASE = "https://data.humdata.org/dataset/8da45d51-d496-4dda-87d7-a0b0b52bd62d/resource/"
CLEAR = {
    0: CLEAR_BASE + "92f86a39-1326-4900-aa11-cd16d1eb67f7/download/clearglobal_language_use_sle_admin0.csv",
    1: CLEAR_BASE + "317e8c21-ae7c-409d-a5c9-2a17981c95d3/download/clearglobal_language_use_sle_admin1.csv",
    2: CLEAR_BASE + "15a87a78-69f8-4d41-844d-e3bf9238d73e/download/clearglobal_language_use_sle_admin2.csv",
}

HH_POP = 7_076_119
# Table 3.22 as printed (persons)
T322 = {
    "Mende": 2_065_349, "Temne": 1_851_300, "Krio": 1_265_295, "Limba": 380_060,
    "Kono": 306_824, "Koranko": 277_356, "Fullah": 173_003, "Susu": 155_175,
    "Kissi": 154_341, "Loko": 91_668, "Madingo": 88_650, "Sherbro": 81_304,
    "Yalunka": 44_935, "Krim": 1_669, "Vai": 1_043, "Foreign language": 11_231, "Other": 5_499,
}
T322_TOTAL = 6_954_702
NOT_STATED = HH_POP - T322_TOTAL      # 121,417

# census label -> CLEAR's Glottolog code
CODE = {
    "Mende": "mend1266", "Temne": "timn1235", "Krio": "krio1253", "Limba": "limb1267",
    "Kono": "kono1268", "Koranko": "kura1250", "Fullah": "fula1264", "Susu": "susu1250",
    "Kissi": "kiss1245", "Loko": "loko1255", "Madingo": "mand1436", "Sherbro": "sher1258",
    "Yalunka": "yalu1240", "Krim": "krim1238", "Vai": "vaii1241",
}
FOREIGN = {"English": "stan1293", "French": "stan1290", "Arabic": "stan1318"}

# Table 3.23: unit -> [(language, %) x 3], Others, as printed ("Time" is the page's Temne)
T323 = {
    "Total country": ([("Mende", 29.2), ("Temne", 26.2), ("Krio", 17.9)], 26.8),
    "Eastern": ([("Mende", 53.0), ("Kono", 17.6), ("Kissi", 9.1)], 20.3),
    "Kailahun": ([("Mende", 69.8), ("Kissi", 22.0), ("Krio", 3.0)], 5.2),
    "Kenema": ([("Mende", 80.2), ("Krio", 7.2), ("Temne", 3.9)], 8.7),
    "Kono": ([("Kono", 56.0), ("Krio", 11.1), ("Temne", 7.4)], 25.6),
    "Northern": ([("Temne", 55.3), ("Limba", 11.4), ("Koranko", 9.5)], 23.8),
    "Bombali": ([("Temne", 42.8), ("Limba", 20.0), ("Krio", 11.8)], 25.4),
    "Kambia": ([("Temne", 53.7), ("Susu", 20.6), ("Limba", 15.7)], 10.0),
    "Koinadugu": ([("Koranko", 49.3), ("Limba", 14.2), ("Fullah", 11.2)], 25.2),
    "Port Loko": ([("Temne", 81.7), ("Krio", 6.2), ("Susu", 3.7)], 8.3),
    "Tonkolili": ([("Temne", 80.3), ("Limba", 6.9), ("Koranko", 6.4)], 6.3),
    "Southern": ([("Mende", 76.4), ("Temne", 6.9), ("Krio", 6.0)], 10.2),
    "Bo": ([("Mende", 76.5), ("Krio", 10.7), ("Temne", 5.0)], 7.8),
    "Bonthe": ([("Mende", 80.4), ("Sherbro", 12.0), ("Krio", 2.8)], 4.7),
    "Moyamba": ([("Mende", 54.8), ("Temne", 20.7), ("Sherbro", 12.5)], 12.0),
    "Pujehun": ([("Mende", 93.8), ("Krio", 2.2), ("Temne", 0.6)], 3.4),
    "Western": ([("Krio", 59.9), ("Temne", 20.3), ("Mende", 5.3)], 14.5),
    "Western Area Rural": ([("Krio", 38.4), ("Temne", 34.6), ("Mende", 7.1)], 19.8),
    "Western Area Urban": ([("Krio", 69.0), ("Temne", 14.2), ("Mende", 4.6)], 12.2),
}
REGIONS = {"Eastern": ["Kailahun", "Kenema", "Kono"],
           "Northern": ["Bombali", "Kambia", "Koinadugu", "Port Loko", "Tonkolili"],
           "Southern": ["Bo", "Bonthe", "Moyamba", "Pujehun"],
           "Western": ["Western Area Rural", "Western Area Urban"]}
DISTRICTS = [d for ds in REGIONS.values() for d in ds]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        return r.read()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == PDF_BYTES:
        print("already have", PDF)
    else:
        for url in (ORIGINAL, WAYBACK):
            try:
                body = get(url)
            except Exception as e:  # noqa: BLE001
                print(f"  {url}: {e}")
                continue
            if body[:4] == b"%PDF" and body.rstrip().endswith(b"%%EOF"):
                PDF.write_bytes(body)
                print(f"wrote {PDF} ({len(body):,} bytes)")
                break
        else:
            raise SystemExit("could not fetch the national analytical report")
    for level, url in CLEAR.items():
        out = RAW / f"clearglobal_sle_admin{level}.csv"
        body = get(url)
        if not body.startswith(b"location_code"):
            raise SystemExit(f"{url}: not CLEAR's CSV")
        out.write_bytes(body)
        print(f"wrote {out} ({len(body):,} bytes)")


def lines_of(doc, page):
    return [ln.strip() for ln in doc.load_page(page).get_text().splitlines() if ln.strip()]


def read_t322(doc):
    ls = lines_of(doc, P_T322)
    i = ls.index("All languages")
    total = int(ls[i + 1].replace(",", ""))
    out, j = {}, ls.index("Mende")
    while ls[j] != "Other" or "Other" not in out:
        lab = ls[j]
        if re.fullmatch(r"[\d,]+", ls[j + 1]):
            out[lab] = int(ls[j + 1].replace(",", ""))
            j += 3
        else:
            j += 1
        if lab == "Other":
            break
    return total, out


def read_t323(doc):
    ls = lines_of(doc, P_T323)
    j = ls.index("Total country")
    out = {}
    while j < len(ls) and not ls[j].startswith("Source"):
        unit = ls[j]
        trio = [(ls[j + 1 + 2 * k].replace("Time", "Temne"), float(ls[j + 2 + 2 * k]))
                for k in range(3)]
        others, total = float(ls[j + 7]), float(ls[j + 8])
        assert total == 100.0, (unit, total)
        out[unit] = (trio, others)
        j += 9
    return out


def ipf(m, rows, cols, blocks=(), iters=5000):
    """Fit the free cells `m` (fixed cells are 0 in it) to row and column targets, and to
    `blocks`: (districts, language, target) sums of a language's free cells over a region."""
    m = m.copy()
    for _ in range(iters):
        for ds, lang, t in blocks:
            cur = m.loc[ds, lang].sum()
            if cur > 0:
                m.loc[ds, lang] *= t / cur
        m = m.mul(rows / m.sum(axis=1), axis=0)
        m = m.mul(cols / m.sum(axis=0), axis=1)
        if (m.sum(axis=1) - rows).abs().max() < 0.01 and all(
                abs(m.loc[ds, lang].sum() - t) < 1 for ds, lang, t in blocks):
            break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()

    import fitz
    import geopandas as gpd
    body = PDF.read_bytes()
    doc = fitz.open(PDF)
    say(len(body) == PDF_BYTES and hashlib.sha256(body).hexdigest() == SHA256
        and body.rstrip().endswith(b"%%EOF") and doc.page_count == PAGES,
        f"1. {NAME}: {len(body):,} bytes, sha256 pinned, %%EOF, {doc.page_count} pages")

    total, t322 = read_t322(doc)
    say(t322 == T322 and total == T322_TOTAL and sum(T322.values()) == T322_TOTAL,
        f"2. Table 3.22: {len(t322)} rows parsed off p134 = the transcription, summing to "
        f"{total:,}")

    t323 = read_t323(doc)
    say(t323 == T323, f"3. Table 3.23: {len(t323)} rows parsed off p135 = the transcription")

    lut = pd.read_csv(RD_GEO / "sl" / "sl_lookup.csv")
    hh = lut.set_index("unit")["census_hh_pop_2015"].astype(float)
    hexes = gpd.read_file(RD_GEO / "sl" / "sl_hexes.gpkg", ignore_geometry=True)
    say(int(hh.sum()) == HH_POP and set(hh.index) == set(DISTRICTS)
        and set(hexes["unit"]) == set(DISTRICTS),
        f"4. 14 districts, household population {hh.sum():,.0f}, = religiondots' hex units "
        "both ways")

    c2 = pd.read_csv(RAW / "clearglobal_sle_admin2.csv")
    adm2 = gpd.read_file(ADM2, engine="fiona", ignore_geometry=True)
    pcs = set(c2["location_code"]) - {"SL05XXX"}
    sums = c2.groupby("location_code")["proportion_value"].sum()
    say(pcs == set(adm2["adm2_pcode"]) and len(pcs) == 16 and (sums - 1).abs().max() < 1e-6,
        "5. CLEAR: 16 districts = COD-AB sle_admin2 pcodes both ways (+ SL05XXX, left out); "
        "shares sum to 1")
    share = c2[c2["location_code"] != "SL05XXX"].pivot_table(
        index="location_code", columns="language_code", values="proportion_value",
        aggfunc="sum").fillna(0.0)

    # 2015 district starting shares: its 2017 pieces weighted by Kontur population
    place = gpd.read_file(PLACE, ignore_geometry=True)
    w = place.groupby(["unit", "pcode17"])["pop"].sum()
    start = {}
    for d in DISTRICTS:
        pieces = w.loc[d]
        start[d] = (share.loc[pieces.index].mul(pieces, axis=0).sum() / pieces.sum())
    start = pd.DataFrame(start).T
    pieces_txt = "; ".join(
        f"{d}: " + ", ".join(f"{p} {v / w.loc[d].sum():.2f}" for p, v in w.loc[d].items())
        for d in DISTRICTS if len(w.loc[d]) > 1)
    print("  2015 districts mixed from 2017 pieces (Kontur weight): " + pieces_txt)

    # columns
    c0 = pd.read_csv(RAW / "clearglobal_sle_admin0.csv").set_index("language_code")[
        "proportion_value"]
    fsum = sum(c0[c] for c in FOREIGN.values())
    cols, m = {}, {}
    for lab, code in CODE.items():
        cols[lab] = T322[lab]
        m[lab] = start[code]
    for lab, code in FOREIGN.items():
        cols[lab] = T322["Foreign language"] * c0[code] / fsum
        m[lab] = start[code]
    unk = HH_POP - T322_TOTAL + T322["Other"]
    cols["Other"] = T322["Other"]
    m["Other"] = start["Unknown"] * T322["Other"] / unk
    cols["_not_stated"] = NOT_STATED
    m["_not_stated"] = start["Unknown"] * NOT_STATED / unk
    m = pd.DataFrame(m).mul(hh, axis=0)
    cols = pd.Series(cols)
    say(abs(cols.sum() - HH_POP) < 1, f"   columns sum to the household population {HH_POP:,}")

    # 7. CLEAR national (before fitting) vs Table 3.22
    nat = m.sum() / m.sum().sum() * 100
    want = cols / HH_POP * 100
    d7 = (nat - want).abs().sort_values(ascending=False)
    say(d7.iloc[0] <= 1.0, "7. CLEAR's shares x district population vs Table 3.22, worst: "
        + ", ".join(f"{k} {nat[k]:.2f} vs {want[k]:.2f}" for k in d7.index[:4]))

    # 6. CLEAR alone (before any fitting) against Table 3.23's district cells: the sample's
    #    honesty, and what fixing the printed cells moves
    raw = m.div(m.sum(axis=1), axis=0) * 100
    d6 = sorted(((abs(raw.loc[d, lang] - p), d, lang, raw.loc[d, lang], p)
                 for d in DISTRICTS for lang, p in T323[d][0]), reverse=True)
    # Port Loko and Bombali are the worst: Karene's shares are used for both its halves, but its
    # Port Loko chiefdoms are Temne and its Bombali ones Limba and Loko (and SL05XXX, 94.5%
    # Temne, is probably Port Loko's). The twelve other districts are within 1.7 points.
    say(d6[0][0] <= 8.0, "6. CLEAR alone vs Table 3.23's 42 district cells, worst four: "
        + "; ".join(f"{lang} in {d} {g:.1f} vs {p:.1f}" for _, d, lang, g, p in d6[:4])
        + f"; mean {sum(x[0] for x in d6) / len(d6):.2f}")

    # Table 3.23's 42 printed district cells are FIXED (share x household population); the
    # free cells are fitted to what is left of each district and of each Table 3.22 total.
    fixed = pd.DataFrame(0.0, index=m.index, columns=m.columns)
    for d in DISTRICTS:
        trio, others = T323[d]
        say(abs(sum(p for _, p in trio) + others - 100) <= 0.15,
            f"   3.23 {d}: top three + Others = 100 within rounding")
        for lang, p in trio:
            fixed.loc[d, lang] = p / 100 * hh[d]
    free = m.where(fixed == 0, 0.0)
    rows_free = hh - fixed.sum(axis=1)
    cols_free = cols - fixed.sum()
    say((cols_free > 0).all(), "   every Table 3.22 total exceeds its fixed district cells")
    # each region's printed cells, less the fixed district cells inside it, bind that
    # language's free cells in the region (Krio in Southern pins Moyamba's Krio, for one)
    blocks = []
    for unit, ds in REGIONS.items():
        for lang, p in T323[unit][0]:
            t = p / 100 * hh[ds].sum() - fixed.loc[ds, lang].sum()
            fr = [d for d in ds if fixed.loc[d, lang] == 0]
            if fr:
                say(t > 0, f"   3.23 {unit} {lang}: the printed region cell exceeds its "
                    f"districts' printed cells ({t:,.0f} left for {', '.join(fr)})")
                blocks.append((fr, lang, t))
    print(f"  {len(blocks)} region cells bind free district cells")
    fit = fixed + ipf(free, rows_free, cols_free, blocks)
    say((fit.sum(axis=1) - hh).abs().max() < 1 and (fit.sum() - cols).abs().max() < 1,
        "   fitted: district totals and Table 3.22 totals both met within 1 person")

    # 8. the printed three are still each district's top three (no free cell above the third)
    worst = []
    for d in DISTRICTS:
        trio = [lang for lang, _ in T323[d][0]]
        third = fit.loc[d, trio[-1]]
        rest = fit.loc[d].drop(trio + ["_not_stated"])
        worst.append((rest.max() / hh[d] * 100 - third / hh[d] * 100, d, rest.idxmax()))
    worst.sort(reverse=True)
    say(worst[0][0] <= 0.1, "8. the printed three stay each district's top three; closest "
        "fourth: " + "; ".join(f"{d} {lang} {g:+.1f} points vs the third"
                               for g, d, lang in worst[:3]))

    # 9. the regions' printed cells (fitted where a district cell was free; the rest a check)
    d9 = []
    for unit, ds in REGIONS.items():
        for lang, p in T323[unit][0]:
            got = fit.loc[ds, lang].sum() / hh[ds].sum() * 100
            d9.append((abs(got - p), unit, lang, got, p))
    d9.sort(reverse=True)
    say(d9[0][0] <= 0.06, "9. region cells vs Table 3.23's 12 printed region cells, worst: "
        + "; ".join(f"{lang} in {u} {g:.2f} vs {p:.1f}" for _, u, lang, g, p in d9[:3]))

    out = []
    for d in DISTRICTS:
        for lab in cols.index:
            if lab == "_not_stated":
                continue
            v = fit.loc[d, lab]
            if round(v) <= 0:
                continue
            out.append(dict(geo_id=d, geo_level="district", geo_name=d, source_category=lab,
                            count=int(round(v)), tier="derived"))
    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,} people "
          f"(Table 3.22: {T322_TOTAL:,})")

    c2.rename(columns={"location_code": "pcode17", "location_name": "name17",
                       "language_code": "clear_code", "language_name": "clear_name",
                       "proportion_value": "share"})[
        ["pcode17", "name17", "clear_code", "clear_name", "share"]].to_csv(
        OUT_CLEAR, index=False, encoding="utf-8")
    print(f"wrote {OUT_CLEAR}")


if __name__ == "__main__":
    main()
