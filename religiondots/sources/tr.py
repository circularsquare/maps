"""Türkiye — the Diyanet's own religious-life survey, at the twelve İBBS-1 regions.

Writes data/normalized/tr.csv: 11 categories x 12 statistical regions, as PEOPLE.

Usage:
    python sources/tr.py --fetch     # the 293-page report and OCHA's province populations
    python sources/tr.py             # rebuild tr.csv from data/raw/tr/

THE SOURCE IS NOT THE STATISTICAL OFFICE, AND THAT IS THE WHOLE FINDING (sources.md §11ac).
`sources.md` §11r closed Türkiye in September 2026 on the sentence *"Türkiye's own publication
of religion is nothing since 1965"*, having asked TÜİK and asked the census. Religion in
Türkiye is published by the **Diyanet İşleri Başkanlığı** — the Presidency of Religious
Affairs — and TÜİK ran the fieldwork for it. `Türkiye'de Dinî Hayat Araştırması`, Ankara 2014:

    fieldwork      15 May - 20 September 2013, CAPI on tablets, face to face
    respondents    21,632 adults, one per sampled household
    design         three-stage stratified cluster, 2,019 clusters (1,330 urban / 689 rural),
                   20 households per urban cluster and 16 per rural
    frame          the Ulusal Adres Veri Tabanı of February 2013, ADNKS's own address base
    strata         İBBS Düzey 1 crossed with urban/rural at a 20,000 threshold
    weighting      inverse selection probability, non-response correction, then integrated
                   calibration to projected populations at the fieldwork midpoint
    estimates at   "Türkiye total, Türkiye urban/rural, and İBBS-1 region totals", its words

**TWELVE REGIONS IS THE SOURCE'S OWN CEILING, not a choice made here**, and it is why §14.4's
rule 2 is satisfied by construction rather than argued around: the resolution IS the state's
own publication.

WHAT IS PARSED. Two tables, both read off the PDF by word position rather than retyped:

    Table 4, page 42  `Ameli mezhep mensubiyetine göre kişi oranı (İBBS, 1. Düzey)`
                      Hanefi / Şafi / Maliki / Hanbeli / Caferi / Diğer / Hiçbiri /
                      Bilmiyorum / Cevap vermeyen, for TR and for each of TR1..TRC.
                      **Shares of those who answered Islam at Q10**, not of everybody.
    Table 1, page 38  `Dini mensubiyetine göre kişi oranı` — İslamiyet 99.2, Diğer 0.4,
                      Cevap vermeyen 0.5. **National only.** There is no regional religion
                      table anywhere in the 293 pages.

So a region's people are built as `population x 99.2% x its own madhhab shares`, plus 0.4%
and 0.5% laid on at the national rate. The madhhab structure varies by region because that is
what the source measures; the religion split does not, because the source publishes one
number for the country. Every row is `modelled` in spec §7 regardless — Guatemala's test
(§7b): the tiers are about whether anybody was COUNTED, and nobody counted religion here.

THERE IS NO ALEVI BOX, AND THE QUESTIONNAIRE IS REPRINTED IN THE REPORT SO THIS IS NOT AN
INFERENCE. Question 11 reads, in full: *Kendinizi hangi mezhebe ait hissediyorsunuz?* —
Hanefi (1), Şafi (2), Maliki (3), Hanbeli (4), Caferi (5), Nusayri (6), Bilmiyorum (7),
Diğer (belirtiniz…) (98), Hiçbiri (90), Cevap vermek istemiyorum (99). Four Sunni schools,
Ja'fari, Nusayri, and nothing else. **The word `Alevi` does not appear once in 293 pages.**
An Alevi respondent's honest options are Hiçbiri, Diğer, Bilmiyorum or a refusal, and all
four of those land on `islam` here — Russia's precedent, where *"Muslim, but neither Sunni
nor Shia"* stays on the parent. See taxonomy/tr2014.py and countries.py's note.

**Do not try to recover Alevis from `Hiçbiri`.** It is 20.8% in Batı Marmara and 11.9% in the
Aegean against 6.5% in Orta Anadolu, which holds Sivas and Yozgat and is the Alevi heartland
by every settlement count there is. That distribution is secularity, not Alevism, and §14.12
already found which way a fractional split over a bucket like this fails.

THE MAGNITUDE IS OCHA'S COD-PS, WHICH IS TÜİK'S OWN REGISTER. `tur_admpop_adm1_2022.csv`,
81 provinces, 85,279,553 people, resident population from ADNKS. TÜİK's own portal is a
single-page app whose table API returns the shell on every path and real 404s under `/api/`,
so the COD is the reachable form of the same numbers. Provinces are summed to İBBS-1 through
`IBBS1` below, which is written out in full and asserted exhaustive both ways — §12's second
shape of failure, a confident wrong pairing, cannot arise from a partial map.

THE 2013 SHARES SIT ON 2022 POPULATIONS, which is spec §3.4's rule (structure from the
detailed source, totals from the recent one) and Russia's and Guatemala's practice. Türkiye
grew about 8% over those nine years and the growth was not even — Şanlıurfa and Istanbul
against the Black Sea — so this carries that shift rather than freezing it.

AND THE SURVEY IS ADULTS WHILE THE DOTS ARE EVERYBODY. The Diyanet interviewed people 18 and
over; the shares are applied to the whole population, which assumes Turkish children are
distributed like their parents. Drawing only adults would leave a quarter of the country
blank, and §6.12 is about how badly a blank reads on a dot map.
"""

import argparse
import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tr")
OUT = os.path.join(ROOT, "data", "normalized", "tr.csv")

# The 293-page report. diyanet.gov.tr no longer serves it; this is CEID's library copy,
# complete and with a valid %%EOF. The Ankara University copy that also turns up in search
# is a 23-page summary deck WITHOUT Table 4 -- see sources/tr.md.
REPORT_URL = "https://www.ceidizleme.org/ekutuphaneresim/dosya/914_1.pdf"
REPORT_PDF = os.path.join(RAW, "diyanet_dini_hayat_2014.pdf")

POP_URL = ("https://data.humdata.org/dataset/79d8aa16-f74e-4881-99a7-81cbd976b8c6/"
           "resource/6ddc2d5c-711c-40d1-acc4-951ff93f4cf1/download/tur_admpop_adm1_2022.csv")
POP_CSV = os.path.join(RAW, "tur_admpop_adm1_2022.csv")

SOURCE_ID = "tr_diyanet_2014"
YEAR = 2013            # fieldwork, not publication
POP_YEAR = 2022

MEZHEP_PAGE = 42       # 1-based, `4. Ameli mezhep ... (İBBS, 1. Düzey)`
DIN_PAGE = 38          # 1-based, `Grafik 1. Dini mensubiyetine göre kişi oranı`

# Table 4's columns, left to right, exactly as its header row prints them.
MEZHEP_COLS = ["Hanefi", "Şafi", "Maliki", "Hanbeli", "Caferi", "Diğer",
               "Hiçbiri", "Bilmiyorum", "Cevap vermeyen"]

# The row labels, as the table prints them. TR is the national row and is kept for the
# cross-check only; countries.py reads geo_level == 'region'.
IBBS_NAMES = {
    "TR1": "İstanbul",
    "TR2": "Batı Marmara",
    "TR3": "Ege",
    "TR4": "Doğu Marmara",
    "TR5": "Batı Anadolu",
    "TR6": "Akdeniz",
    "TR7": "Orta Anadolu",
    "TR8": "Batı Karadeniz",
    "TR9": "Doğu Karadeniz",
    "TRA": "Kuzeydoğu Anadolu",
    "TRB": "Ortadoğu Anadolu",
    "TRC": "Güneydoğu Anadolu",
}

# Table 1, page 38. National only; there is no regional version in the report.
DIN_ISLAM = 99.2
DIN_OTHER = 0.4        # "İslam dini dışındaki diğer dinlere mensup YA DA herhangi bir dine
                       #  mensup olmadığını ifade etmiştir" -- other religion OR none, one cell
DIN_NOANSWER = 0.5

# ---------------------------------------------------------------------------------------
# The 81 provinces of Türkiye grouped into İBBS Düzey 1, using OCHA COD-PS's own ASCII
# spellings of ADM1_EN. Written out rather than derived: the İBBS-1 grouping has been fixed
# since 2002, and a crosswalk that is asserted exhaustive in both directions cannot silently
# drop a province the way a fuzzy name match can ([[reference_name_join_wrong_neighbour]]).
# ---------------------------------------------------------------------------------------
IBBS1 = {
    "TR1": ["Istanbul"],
    "TR2": ["Tekirdag", "Edirne", "Kirklareli", "Balikesir", "Canakkale"],
    "TR3": ["Izmir", "Aydin", "Denizli", "Mugla", "Manisa", "Afyonkarahisar", "Kutahya",
            "Usak"],
    "TR4": ["Bursa", "Eskisehir", "Bilecik", "Kocaeli", "Sakarya", "Duzce", "Bolu",
            "Yalova"],
    "TR5": ["Ankara", "Konya", "Karaman"],
    "TR6": ["Antalya", "Isparta", "Burdur", "Adana", "Mersin", "Hatay", "Kahramanmaras",
            "Osmaniye"],
    "TR7": ["Kirikkale", "Aksaray", "Nigde", "Nevsehir", "Kirsehir", "Kayseri", "Sivas",
            "Yozgat"],
    "TR8": ["Zonguldak", "Karabuk", "Bartin", "Kastamonu", "Cankiri", "Sinop", "Samsun",
            "Tokat", "Corum", "Amasya"],
    "TR9": ["Trabzon", "Ordu", "Giresun", "Rize", "Artvin", "Gumushane"],
    "TRA": ["Erzurum", "Erzincan", "Bayburt", "Agri", "Kars", "Igdir", "Ardahan"],
    "TRB": ["Malatya", "Elazig", "Bingol", "Tunceli", "Van", "Mus", "Bitlis", "Hakkari"],
    "TRC": ["Gaziantep", "Adiyaman", "Kilis", "Sanliurfa", "Diyarbakir", "Mardin", "Batman",
            "Sirnak", "Siirt"],
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dest in ((REPORT_URL, REPORT_PDF), (POP_URL, POP_CSV)):
        if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
            print("already have", os.path.basename(dest))
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.basename(dest)}  {len(r.content):,} bytes")

    # [[reference_pdf_truncated_at_source]]: Content-Length can match a damaged file, so the
    # trailer is what says the PDF arrived whole.
    with open(REPORT_PDF, "rb") as fh:
        head = fh.read(5)
        fh.seek(-2048, os.SEEK_END)
        tail = fh.read()
    if head != b"%PDF-" or b"%%EOF" not in tail:
        raise SystemExit(f"{REPORT_PDF} is not a complete PDF (no %%EOF in the last 2 KB)")


def _rows_by_line(page):
    """Words grouped into visual lines, so a table row reads left to right."""
    from collections import defaultdict

    lines = defaultdict(list)
    for x0, y0, _x1, _y1, word, *_ in page.get_text("words"):
        lines[round(y0 / 3)].append((x0, word))
    return [" ".join(w for _, w in sorted(v)) for _, v in sorted(lines.items())]


def read_mezhep():
    """Table 4 -> {ibbs_code: {column: share}}, plus the national row.

    Parsed off the page rather than retyped. The row label and nine numbers are on one
    visual line, except TR6, whose `100` floats one line above it in this PDF -- so the
    parser takes the LAST nine numeric tokens of a labelled line and ignores the total
    column entirely, which is robust to that and to any similar drift.
    """
    import fitz

    if not os.path.exists(REPORT_PDF):
        raise SystemExit(f"missing {REPORT_PDF} -- run sources/tr.py --fetch first")
    doc = fitz.open(REPORT_PDF)
    if doc.page_count != 293:
        raise SystemExit(f"the report has {doc.page_count} pages, expected 293 -- this is "
                         "probably the 23-page summary deck, which has no Table 4")

    page = doc[MEZHEP_PAGE - 1]
    text = page.get_text()
    if "Ameli mezhep mensubiyetine göre kişi oranı" not in text or "İBBS" not in text:
        raise SystemExit(f"page {MEZHEP_PAGE} is not Table 4 -- the PDF changed")

    def numbers(line):
        out = []
        for tok in line.split():
            t = tok.replace(",", ".")
            if t == "-":
                out.append(0.0)
            elif t.replace(".", "", 1).isdigit():
                out.append(float(t))
        return out

    want = {"TR": "Türkiye"}
    want.update(IBBS_NAMES)

    got = {}
    for line in _rows_by_line(page):
        for code, name in want.items():
            if code in got:
                continue
            head = f"{code} {name}"
            if not line.startswith(head):
                continue
            nums = numbers(line[len(head):])
            if len(nums) < 9:
                raise SystemExit(f"{code}: found {len(nums)} numbers on its line, need 9 "
                                 f"-- {line!r}")
            got[code] = dict(zip(MEZHEP_COLS, nums[-9:]))

    missing = sorted(set(want) - set(got))
    if missing:
        raise SystemExit(f"Table 4 rows not found: {missing} -- the page layout changed")

    for code, shares in got.items():
        total = sum(shares.values())
        # The report's own NOT 1 says the totals may miss 100 through rounding; it is never
        # more than 0.2 here. Anything wider is a mis-parse, not a rounding artefact.
        if abs(total - 100.0) > 0.5:
            raise SystemExit(f"{code} sums to {total:.2f}, not 100 -- wrong columns?")

    # Two values read off the page by eye, asserted so a silent column shift cannot pass.
    if abs(got["TR"]["Hanefi"] - 77.5) > 0.01 or abs(got["TRB"]["Şafi"] - 48.7) > 0.01:
        raise SystemExit("Table 4's anchors moved: national Hanefi should be 77.5 and "
                         "TRB Şafi 48.7")

    # And the religion table on its own page, checked rather than trusted to the constants.
    din = doc[DIN_PAGE - 1].get_text()
    if "Dini mensubiyetine göre kişi oranı" not in din or "99,2" not in din:
        raise SystemExit(f"page {DIN_PAGE} is not Table 1 / Grafik 1 -- the PDF changed")

    national = got.pop("TR")
    return got, national


def read_population():
    """OCHA COD-PS 2022 -> {ibbs_code: people}, summed from the 81 provinces."""
    if not os.path.exists(POP_CSV):
        raise SystemExit(f"missing {POP_CSV} -- run sources/tr.py --fetch first")

    prov = {}
    with open(POP_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("T_TL", "").strip():
                continue                       # the file carries blank template rows
            prov[row["ADM1_EN"].strip()] = int(row["T_TL"])
    if len(prov) != 81:
        raise SystemExit(f"COD-PS has {len(prov)} provinces with a total, expected 81")

    mapped = {p for names in IBBS1.values() for p in names}
    if mapped != set(prov):
        raise SystemExit(
            "the İBBS-1 crosswalk and COD-PS disagree about the provinces.\n"
            f"  in the crosswalk, not in COD-PS: {sorted(mapped - set(prov))}\n"
            f"  in COD-PS, not in the crosswalk: {sorted(set(prov) - mapped)}")
    if sum(len(v) for v in IBBS1.values()) != 81:
        raise SystemExit("the crosswalk lists a province twice")

    pop = {code: sum(prov[p] for p in names) for code, names in IBBS1.items()}
    print(f"  population: {len(pop)} regions, {sum(pop.values()):,} people "
          f"(COD-PS {POP_YEAR}, from ADNKS)")
    return pop


def to_people(shares, population):
    """shares (summing to ~100) x population -> whole people summing EXACTLY to population.

    Largest remainder, and ru.py's note applies unchanged: this apportions CATEGORIES within
    one region, so no geography is being decided and only the last few people move. The
    shares are renormalised first because the report rounds each cell independently.
    """
    scale = sum(shares.values())
    if scale <= 0:
        raise SystemExit("a region has no shares at all")
    exact = {k: v / scale * population for k, v in shares.items()}
    floors = {k: int(v) for k, v in exact.items()}
    short = population - sum(floors.values())
    order = sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in order[:short]:
        floors[k] += 1
    return floors


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the report and COD-PS")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    if args.fetch:
        fetch()

    import pandas as pd

    mezhep, national = read_mezhep()
    pop = read_population()
    print(f"  Table 4: {len(mezhep)} regions x {len(MEZHEP_COLS)} columns, "
          f"survey n=21,632 over 12 regions (~1,800 each)")

    rows = []
    for code in sorted(mezhep):
        people = pop[code]
        # Table 1's split is national and is laid on every region at the same rate, because
        # the report publishes no regional version of it. Table 4's shares are OF MUSLIMS,
        # so they are scaled into the Islam slice rather than applied to the whole region.
        block = {f"Mezhep: {c}" if c in ("Diğer", "Hiçbiri", "Bilmiyorum", "Cevap vermeyen")
                 else c: v * DIN_ISLAM / 100.0
                 for c, v in mezhep[code].items()}
        block["Din: Diğer"] = DIN_OTHER
        block["Din: Cevap vermeyen"] = DIN_NOANSWER

        counts = to_people(block, people)
        if sum(counts.values()) != people:
            raise SystemExit(f"{code}: apportionment lost people")
        for cat, n in counts.items():
            rows.append({
                "geo_id": code, "geo_level": "region", "geo_name": IBBS_NAMES[code],
                "source_category": cat, "count": n,
                "basis": "self_id", "year": YEAR, "source_id": SOURCE_ID,
                "note": (f"share={block[cat]:.4f}% of the region; population={people} "
                         f"(COD-PS {POP_YEAR}); Diyanet/TÜİK survey, n~1,800 in this region"),
            })

    # The national row, for cross-checking only.
    for cat, share in national.items():
        rows.append({
            "geo_id": "TR", "geo_level": "country", "geo_name": "Türkiye",
            "source_category": f"Mezhep: {cat}" if cat in (
                "Diğer", "Hiçbiri", "Bilmiyorum", "Cevap vermeyen") else cat,
            "count": round(share / 100.0 * DIN_ISLAM / 100.0 * sum(pop.values())),
            "basis": "self_id", "year": YEAR, "source_id": SOURCE_ID,
            "note": f"share={share:.4f}% of Muslims; the report's own national row, n=21,632",
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    drawn = out.loc[out["geo_level"] == "region", "count"].sum()
    if drawn != sum(pop.values()):
        raise SystemExit(f"the region rows total {drawn:,}, not {sum(pop.values()):,}")
    print(f"  wrote {OUT}  {len(out):,} rows")
    print(f"  drawn: {drawn:,} people on 12 regions (the `country` rows are the report's "
          f"own national column, the same people again, and countries.py ignores them)")

    # ---- the readable check: the region-weighted shares against the report's own national
    # row. They are separately weighted, so this is a relationship rather than an identity.
    print("\n  region-weighted vs the report's own national row (% of Muslims):")
    total = sum(pop.values())
    for col in MEZHEP_COLS:
        w = sum(mezhep[c][col] * pop[c] for c in mezhep) / total
        flag = "" if abs(w - national[col]) < 2.0 else "   <-- check"
        print(f"    {col:14s} {w:6.2f}%  vs {national[col]:6.2f}%{flag}")

    print("\n  and what the map will draw, by region (% of Muslims):")
    print(f"    {'':22s} {'Hanefi':>8s} {'Şafi':>8s} {'Caferi':>8s} {'no school':>10s}")
    for code in sorted(mezhep):
        m = mezhep[code]
        none = m["Hiçbiri"] + m["Bilmiyorum"] + m["Diğer"] + m["Cevap vermeyen"]
        print(f"    {code} {IBBS_NAMES[code]:<18s} {m['Hanefi']:8.1f} {m['Şafi']:8.1f} "
              f"{m['Caferi']:8.1f} {none:10.1f}")


if __name__ == "__main__":
    main()
