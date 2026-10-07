"""Solomon Islands, 2019 National Population and Housing Census: first language learnt as a child,
national counts, plus the province tables sources/sb_model.py places them with
-> data/normalized/sb.csv and data/normalized/sb_inputs.csv.

    python sources/sb_census.py [--fetch]

THE QUESTION. "What is the first language this person (name) learnt as a child?", asked of
everyone aged 5 and over (questionnaire P18a, Volume 2's appendix: 1 Pidjin, 2 English, 3 Local
language (specify), 4 Other (specify)). 631,061 people aged 5+.

THE LANGUAGE TABLES are in the *National Report, Volume 1* (SINSO, 2023), section 9.6, PDF pages
149-152 (printed 112-115), and they are national only:
  * Table 9.6.1 "Larger local languages by province": Pidgin and 24 local languages, grouped
    under the province each is spoken in (the heading is the language's home, not where its
    speakers were counted; every figure is a national total). 1976, 1999 and 2019 columns.
  * Table 9.6.2 "Larger local languages": 21 more rows, three of them repeating 9.6.1 (Lau,
    Marovo, and RenBell = 9.6.1's Rennell-Bellona row).
  * Table 9.6.3 "Endangered and new listed languages": 17 small ones, grouped by province, Anuta
    repeating 9.6.2.
English and "Other" are not printed anywhere, and neither are the many local languages too small
or too unremarkable for these three tables (Tikopia, Bauro, Natugu, Bughotu, Savosavo...). Their
total is the census's 5+ population less everything named: carried as one category, REMAINDER.
Volume 2 (Basic Tables) has no first-language table at all (its language tables are literacy).

THE PROVINCE TABLES, for the placement model (all census, all printed):
  * Figure 9.6.1 (vol 1, PDF p150) "Pidgin as first-learnt language by province": a bar chart
    printed as an IMAGE, so its ten labels are transcribed below (PIDGIN_FIG) from the rendered
    page. They sum to 100.1, so they are each province's share of the 101,588 Pidgin speakers,
    not Pidgin's share of each province (that reading gives 106,276 against 101,588).
  * Table 9.2.2 (vol 1, PDF p135): population aged 5+ by province, sums to 631,061.
  * Table P7.2 (vol 2, PDF pp133-135): population by province of enumeration x province of
    birth, all ages, 720,956.
  * Table 8.4.3 (vol 1, PDF p131): ethnic origin by province, for the Micronesian row (where
    the Kiribati-speaking resettlement communities live).

CHECKS, all asserted:
  1. the three language tables agree wherever they repeat a language (2019 column);
  2. Pidgin 101,588 is 16.1% of 631,061, as printed in the text;
  3. Table 9.2.2's provinces sum to 631,061, its national row;
  4. P7.2: every row's birth columns sum to its Total, the column totals are the sums of the ten
     province blocks, and each province's Total equals Table 8.4.3's Total for it;
  5. Table 8.4.3's Micronesian row sums to its printed 8,647;
  6. the named languages sum to less than 631,061 (the remainder is 90,859, 14.4%).

RAW FILES. religiondots' cached copies (../religiondots/data/raw/sb/), read-only; with --fetch,
and no copy there, they are downloaded to data/raw/sb/ (checked for the %%EOF trailer).
"""
import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RD_RAW = ROOT.parent / "religiondots" / "data" / "raw" / "sb"
RAW = ROOT / "data" / "raw" / "sb"
OUT = ROOT / "data" / "normalized" / "sb.csv"
OUT_IN = ROOT / "data" / "normalized" / "sb_inputs.csv"

FILES = {
    "vol1": ("sb_2019_national_report_vol1.pdf",
             "https://solomons.gov.sb/wp-content/uploads/2023/09/Solomon-Islands-2019-Population-"
             "and-Housing-Census_National-Report-Vol-1.pdf"),
    "vol2": ("sb_2019_basic_tables_vol2.pdf",
             "https://solomons.gov.sb/wp-content/uploads/2023/09/Solomon-Islands-2019-Population-"
             "Census-Report_Basic-Tables_Operations_Vol2.pdf"),
}

PROVINCES = ["Choiseul", "Western", "Isabel", "Central", "Rennell-Bellona", "Guadalcanal",
             "Malaita", "Makira-Ulawa", "Temotu", "Honiara"]
POP5_TOTAL = 631_061
PIDGIN = 101_588
REMAINDER = "Not named in the report (other local languages, English, other)"

# Figure 9.6.1, read off the rendered chart (an image; no text layer). Share of all Pidgin
# first-language speakers living in each province, percent.
PIDGIN_FIG = {"Honiara": 47.2, "Guadalcanal": 20.1, "Western": 15.2, "Malaita": 5.6,
              "Makira-Ulawa": 3.1, "Temotu": 2.7, "Central": 2.5, "Isabel": 1.9,
              "Choiseul": 1.4, "Rennell-Bellona": 0.4}

NUM = re.compile(r"^-?[\d,]+(\.\d+)?%?$")


def pdf(key, fetch=False):
    name, url = FILES[key]
    for d in (RD_RAW, RAW):
        if (d / name).exists():
            return d / name
    if not fetch:
        raise SystemExit(f"{name} not found in {RD_RAW} or {RAW}; run with --fetch")
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
                     "Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0"})
    r.raise_for_status()
    if not r.content.rstrip().endswith(b"%%EOF"):
        raise SystemExit(f"{url}: no %%EOF trailer, truncated")
    (RAW / name).write_bytes(r.content)
    return RAW / name


def lines(doc, pno, start, stop):
    """Stripped non-empty lines of PDF page `pno` (1-based) between two marker substrings."""
    out, on = [], False
    for ln in doc[pno - 1].get_text().splitlines():
        s = ln.strip()
        if not on and start in s:
            on = True
            continue
        if on and stop in s:
            break
        if on and s:
            out.append(s)
    if not out:
        raise SystemExit(f"page {pno}: nothing between {start!r} and {stop!r}")
    return out


def num(s):
    return float(s.replace(",", "").rstrip("%"))


def rows(toks, skip=()):
    """[(label, [numbers], heading)] from a token stream: a label followed by numbers is a row;
    a label followed by another label is a heading (the province a language is listed under)."""
    out, heading, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if NUM.match(t) or t in skip:
            i += 1
            continue
        j = i + 1
        vals = []
        while j < len(toks) and NUM.match(toks[j]):
            vals.append(num(toks[j]))
            j += 1
        if vals:
            out.append((t, vals, heading))
        else:
            heading = t
        i = j
    return out


def read_languages(doc):
    head = ["Language", "1976", "1999", "2019", "Percent (%)", "increase,1999-2019", "increase,",
            "1999-2019", "5 years +", "Local Language", "Census", "Rate of Growth", "1999 to 2019",
            "Census Years", "Endangered languages", "& New Languages"]
    t1 = (lines(doc, 149, "Table 9.6.1: Larger local languages by province", "Note that this")
          + lines(doc, 150, "Table 9.6.1 (Cont", "Figure 9.6.1"))
    t1 = [t for t in t1 if t not in head and not t.startswith("41 ")]
    t2 = lines(doc, 151, "Table 9.6.2: Larger local languages", "Surprisingly")
    t2 = [t for t in t2 if t not in head and not t.startswith("1976")]
    t3 = lines(doc, 152, "Table 9.6.3. Endangered", "* New language name")
    t3 = [t for t in t3 if t not in head]

    r1 = rows(t1, skip=("Solomon Islands: 1976, 1999, 2019", "1976, 1999, 2019"))
    r2 = rows(t2, skip=("1976, 1999, 2019",))
    r3 = rows(t3)
    # the province headings of 9.6.1 and 9.6.3 are the languages' homes
    out = {}
    for tab, rr in (("9.6.1", r1), ("9.6.2", r2), ("9.6.3", r3)):
        for label, vals, heading in rr:
            label = label.rstrip("*").strip()
            # 9.6.1: 1976, 1999, 2019, % increase (Tairaha prints 2019 only); 9.6.2: the same with
            # a growth rate last; 9.6.3: whichever census years it has, 2019 always last
            if tab == "9.6.3" or len(vals) < 4:
                v2019 = vals[-1]
            else:
                v2019 = vals[-2]
            if label == "Rennell-Bellona":
                # 9.6.1 prints the language as a row named for its province, directly under the
                # Central heading; 9.6.2 calls it RenBell (same 2019 figure, asserted below)
                label, heading = "RenBell", "Rennell-Bellona"
            home = {"Choisuel": "Choiseul"}.get(heading, heading)
            if label in out:
                if out[label]["count"] != v2019:
                    raise SystemExit(f"{label}: {tab} has {v2019:,.0f}, "
                                     f"{out[label]['table']} has {out[label]['count']:,.0f}")
                out[label]["table"] += f"+{tab}"
                continue
            out[label] = dict(count=v2019, table=tab, home=home if tab != "9.6.2" else None)
    return out


def read_pop5(doc):
    t = lines(doc, 135, "Table 9.2.2:", "Source: 2019 Solomon Islands Census")
    i = t.index("Total", t.index("Never been", t.index("Left school")))
    out = {}
    for name in ["Total"] + PROVINCES:
        k = t.index(name, i)
        out[name] = int(num(t[k + 1]))
    if sum(out[p] for p in PROVINCES) != out["Total"] or out["Total"] != POP5_TOTAL:
        raise SystemExit(f"Table 9.2.2 does not sum: {out}")
    return out


def read_birth(doc2):
    cols = ["Total"] + PROVINCES + ["Oversea"]
    toks = []
    for p in (133, 134, 135):
        page = doc2[p - 1].get_text()
        a = page.index("P7.2: Total population")
        b = page.index("Province of birth", a + 100) if "Province of birth" in page[a + 100:] \
            else len(page)
        toks += [s.strip() for s in page[a:b].splitlines() if s.strip()]
    enum_names = {"Makira": "Makira-Ulawa", "Solomomon Islands": "Solomon Islands"}
    out, cur, i = {}, None, 0
    while i < len(toks):
        s = toks[i]
        if s in PROVINCES or s in enum_names:
            cur = enum_names.get(s, s)
        elif s == "Total" and cur and i + 1 < len(toks) and NUM.match(toks[i + 1]):
            vals = [int(num(x)) for x in toks[i + 1:i + 1 + len(cols)]]
            out[cur] = dict(zip(cols, vals))
            cur = None
            i += len(cols)
        i += 1
    nat = out.pop("Solomon Islands")
    if sorted(out) != sorted(PROVINCES):
        raise SystemExit(f"P7.2 provinces: {sorted(out)}")
    for p, r in out.items():
        if sum(r[c] for c in cols[1:]) != r["Total"]:
            raise SystemExit(f"P7.2 {p}: birth columns do not sum to Total")
    for c in cols:
        if sum(out[p][c] for p in PROVINCES) != nat[c]:
            raise SystemExit(f"P7.2 column {c}: provinces sum {sum(out[p][c] for p in PROVINCES)}"
                             f", national row {nat[c]}")
    return out


def read_ethnic(doc):
    t = lines(doc, 131, "Table 8.4.3:", "The median age")
    cut = t.index("Total", t.index("Honiara"))
    def row(label):
        k = t.index(label, cut)
        return [int(num(x)) for x in t[k + 1:k + 12]]
    tot, mic = row("Total"), row("Micronesian")
    if sum(mic[1:]) != mic[0] or mic[0] != 8647 or sum(tot[1:]) != tot[0]:
        raise SystemExit(f"Table 8.4.3 does not sum: {tot} {mic}")
    return dict(zip(PROVINCES, tot[1:])), dict(zip(PROVINCES, mic[1:]))


def main():
    import fitz
    fetch = "--fetch" in sys.argv
    doc = fitz.open(pdf("vol1", fetch))
    doc2 = fitz.open(pdf("vol2", fetch))

    langs = read_languages(doc)
    if langs.get("Pidgin", {}).get("count") != PIDGIN or round(100 * PIDGIN / POP5_TOTAL, 1) != 16.1:
        raise SystemExit("Pidgin is not 101,588 / 16.1%")
    named = sum(v["count"] for v in langs.values())
    rem = POP5_TOTAL - named
    if rem <= 0:
        raise SystemExit(f"named languages {named:,.0f} exceed the 5+ population")
    pop5 = read_pop5(doc)
    birth = read_birth(doc2)
    tot843, mic = read_ethnic(doc)
    for p in PROVINCES:
        if birth[p]["Total"] != tot843[p]:
            raise SystemExit(f"{p}: P7.2 {birth[p]['Total']:,} vs Table 8.4.3 {tot843[p]:,}")
    if abs(sum(PIDGIN_FIG.values()) - 100) > 0.15 or sorted(PIDGIN_FIG) != sorted(PROVINCES):
        raise SystemExit("Figure 9.6.1 transcription does not sum to 100 or misses a province")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "note"])
        for label, v in sorted(langs.items(), key=lambda kv: -kv[1]["count"]):
            w.writerow(["SB", "national", "Solomon Islands", label, int(v["count"]),
                        f"table={v['table']}; home={v['home'] or ''}"])
        w.writerow(["SB", "national", "Solomon Islands", REMAINDER, int(rem),
                    "5+ population 631,061 less every language the report names"])
    with open(OUT_IN, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "row", "col", "value"])
        for p in PROVINCES:
            w.writerow(["pop5_9.2.2", p, "", pop5[p]])
            w.writerow(["pidgin_share_fig9.6.1", p, "", PIDGIN_FIG[p]])
            w.writerow(["micronesian_8.4.3", p, "", mic[p]])
            for b in PROVINCES + ["Oversea"]:
                w.writerow(["birth_P7.2", p, b, birth[p][b]])   # row = enumerated in, col = born in

    print(f"{len(langs)} named answers, {named:,.0f} people; remainder {rem:,.0f} "
          f"({100 * rem / POP5_TOTAL:.1f}%) of {POP5_TOTAL:,} aged 5+")
    for label, v in sorted(langs.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {label:14s} {v['count']:>9,.0f}  {v['table']:13s} {v['home'] or ''}")
    pid = {p: PIDGIN * PIDGIN_FIG[p] / 100.1 for p in PROVINCES}
    print("Pidgin by province (Figure 9.6.1 x 101,588), and as a share of the province's 5+:")
    for p in PROVINCES:
        print(f"  {p:16s} {pid[p]:>8,.0f}  {100 * pid[p] / pop5[p]:5.1f}%  of {pop5[p]:,}")
    print(f"wrote {OUT} and {OUT_IN}")


if __name__ == "__main__":
    main()
