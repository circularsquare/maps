"""Singapore: Census of Population 2020, language most frequently spoken at home, by planning area.

    python sources/sg_census.py --fetch    download into data/raw/sg/, then normalise
    python sources/sg_census.py            normalise what is on disk

Three SingStat TableBuilder tables, republished on data.gov.sg (Singapore Open Data Licence, no
key, no login), all "Resident Population Aged 5 Years and Over by ... Language Most / Second
Most Frequently Spoken at Home (Census of Population 2020)":

  CT/17596  d_21f546492a87dec38391fc72eb4c7890   by PLANNING AREA OF RESIDENCE. Seven groups for
            the language most frequently spoken: English, Mandarin, Chinese Dialects, Malay,
            Tamil, Other Indian Languages, Other Languages (each also split by the second
            language, which this map does not use). 30 named planning areas plus `Others`, the
            25 URA Master Plan 2019 areas with almost nobody living in them: the same 31 rows as
            the religion table religiondots drew, so its placement layer serves as it is.
  CT/17439  d_2a1a771fe8515745d21ee64e43d6b46d   NATIONAL, by ethnic group and sex. Same seven
            groups; a second table of the same census, used only as a check.
  (Table 41 of Statistical Release 1)
            d_ad4a8ccbdab03d16c486a9ee6988289d   NATIONAL, by age group. The only table that opens
            "Chinese Dialects" into Hokkien, Teochew, Cantonese and Other Chinese Dialects.
            Nothing finer than national splits the dialects (the 2010 and GHS 2015 planning-area
            tables carry the same seven groups), so countries/sg.py shares each planning area's
            Chinese Dialects out by this national mix, tier `derived`.

THE UNIVERSE. Residents (citizens and PRs) aged 5 and over, EXCLUDING "persons who were unable
to speak, and those in one-person households and households comprising only unrelated persons"
(every table's footnote: the question is about the language spoken to other household
members). 3,596,284 people against a June 2020 population of 5,685,800 (Table 1.1): the
1,641,590 non-residents are not asked at all, and 448,126 residents fall outside the question
(under 5, living alone or with unrelated people, unable to speak). Not scaled up; see
sources/sg.md.

CHECKS:
  * `-` is the report's "nil or negligible" and reads as 0; anything else non-numeric raises.
  * every planning area's seven groups sum to its Total, and English's "English only / English
    & X" sub-columns sum to English, within SingStat's random rounding (ROUNDING per cell; the
    other groups' sub-columns omit some second languages by footnote and are not checked);
  * the 31 rows sum to the table's own Total row, per group, within the same rounding;
  * the Total row equals CT/17439's national row EXACTLY, group by group (a second table);
  * the four dialects sum to Chinese Dialects exactly, and every national figure equals the
    printed report (Table 41, page 151) EXACTLY;
  * the 2010 census's national dialect mix (CT/5980) is printed beside 2020's: the borrowed
    shape has to be stable for the national split to be defensible.
The planning-area join is asserted in countries/sg.py against religiondots' layer.
"""
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "sg"
OUT = ROOT / "data" / "normalized" / "sg.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

DATASETS = {
    "pa": "d_21f546492a87dec38391fc72eb4c7890",        # CT/17596, by planning area
    "nat_eth": "d_2a1a771fe8515745d21ee64e43d6b46d",   # CT/17439, by ethnic group and sex
    "nat_age": "d_ad4a8ccbdab03d16c486a9ee6988289d",   # Table 41, by age, dialects opened
    "nat_2010": "d_cde207636c5c3a4e76b796fa9ba17971",  # 2010 CT/5980, for the drift check
}

# planning-area table column -> label (the language MOST frequently spoken)
PA_COLS = {
    "English_Total": "English",
    "Mandarin_Total1": "Mandarin",
    "ChineseDialects_Total1": "Chinese Dialects",
    "Malay_Total1": "Malay",
    "IndianLanguages_Tamil_Total1": "Tamil",
    "IndianLanguages_OtherIndianLanguages_Total1": "Other Indian Languages",
    "OtherLanguages_Total1": "Other Languages",
}
# national tables' row labels -> the same labels
# (the "1/" footnote mark is stripped before matching)
NAT_ROWS = {k: k for k in ["English", "Mandarin", "Chinese Dialects", "Malay", "Tamil",
                           "Other Indian Languages", "Other Languages"]}
DIALECTS = {k: k for k in ["Hokkien", "Teochew", "Cantonese", "Other Chinese Dialects"]}

# Statistical Release 1, Table 41, Total column (printed page 151, PDF pages 167-168).
PRINTED = {"Total": 3_596_284, "English": 1_735_242, "Mandarin": 1_075_172,
           "Chinese Dialects": 313_258, "Hokkien": 157_259, "Teochew": 59_424,
           "Cantonese": 79_216, "Other Chinese Dialects": 17_359, "Malay": 332_256,
           "Tamil": 89_946, "Other Indian Languages": 23_818, "Other Languages": 26_592}

TOTAL_POPULATION = 5_685_800      # Table 1.1, June 2020
NON_RESIDENTS = 1_641_590
ROUNDING = 10                     # SingStat's random rounding, per cell; religiondots saw <= 4
NIL = "-"


def _num(v, where):
    v = (v or "").strip()
    if v == NIL:
        return 0
    try:
        return int(v.replace(",", ""))
    except ValueError:
        raise SystemExit(f"!! {where}: {v!r} is neither a number nor {NIL!r}")


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for tag, ds in DATASETS.items():
        dest = RAW / f"{tag}_{ds}.csv"
        if dest.exists() and dest.stat().st_size > 500:
            print("have", dest.name)
            continue
        # data.gov.sg hands out a signed, expiring S3 URL: poll for it every time. Anonymous
        # callers are rate-limited (code 24, "try again in 10 seconds").
        for attempt in range(6):
            j = requests.get(f"https://api-open.data.gov.sg/v1/public/api/datasets/{ds}/"
                             "poll-download", headers=UA, timeout=60).json()
            if j.get("code") != 24:
                break
            time.sleep(12)
        url = (j.get("data") or {}).get("url")
        if j.get("code") != 0 or not url:
            raise SystemExit(f"!! poll-download for {ds} returned no URL: {j}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if not r.content.lstrip(b"\xef\xbb\xbf").startswith(b"Number,"):
            raise SystemExit(f"!! {ds}: not the expected CSV")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        os.replace(tmp, dest)
        print("got", dest.name, len(r.content), "bytes")


def _read(tag):
    path = RAW / f"{tag}_{DATASETS[tag]}.csv"
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _national(rows, col, labels, tag):
    out = {}
    for row in rows:
        # the "1/" footnote mark is on some tables' labels and not others'
        lab = row["Number"].strip().removesuffix("1/")
        if lab == "Total":
            out["Total"] = _num(row[col], f"{tag} Total")
        elif lab in labels:
            out[labels[lab]] = _num(row[col], f"{tag} {lab}")
    missing = (set(labels.values()) | {"Total"}) - set(out)
    if missing:
        raise SystemExit(f"!! {tag}: rows missing {sorted(missing)}")
    return out


def normalise():
    pa = _read("pa")
    header = list(pa[0].keys())
    for c in PA_COLS:
        if c not in header:
            raise SystemExit(f"!! planning-area table lacks column {c}")
    # English's "English only / English & X" sub-columns must sum to it. The other groups carry
    # the footnote "1/ Data include other categories of language second most frequently spoken
    # at home which are not shown", so their sub-columns fall short by design.
    groups = {"English_Total": [h for h in header
                                if h.startswith("English_") and h != "English_Total"]}
    if len(groups["English_Total"]) != 7:
        raise SystemExit(f"!! English sub-columns: {groups['English_Total']}")

    rows, total_row, worst = [], None, 0
    for r in pa:
        name = r["Number"].strip()
        vals = {lab: _num(r[c], f"{name} {c}") for c, lab in PA_COLS.items()}
        tot = _num(r["Total"], f"{name} Total")
        d = abs(sum(vals.values()) - tot)
        worst = max(worst, d)
        if d > ROUNDING:
            raise SystemExit(f"!! {name}: groups sum {sum(vals.values())} vs Total {tot}")
        for c, subs in groups.items():
            if not subs:
                continue
            s = sum(_num(r[h], f"{name} {h}") for h in subs)
            if abs(s - _num(r[c], name)) > ROUNDING * len(subs):
                raise SystemExit(f"!! {name} {c}: sub-columns {subs} sum {s}")
        if name == "Total":
            total_row = dict(vals, Total=tot)
        else:
            rows.append((name, vals, tot))
    if total_row is None or len(rows) != 31:
        raise SystemExit(f"!! expected a Total row and 31 units, got {len(rows)}")
    if not any(n == "Others" for n, _, _ in rows):
        raise SystemExit("!! no `Others` row")
    print(f"31 units; worst unit-vs-Total gap {worst} (random rounding)")

    for lab in PA_COLS.values():
        s = sum(v[lab] for _, v, _ in rows)
        if abs(s - total_row[lab]) > ROUNDING * 3:
            raise SystemExit(f"!! {lab}: units sum {s} vs Total row {total_row[lab]}")
        print(f"  {lab:24s} units {s:>10,}  Total row {total_row[lab]:>10,}  diff {s - total_row[lab]:+d}")

    eth = _national(_read("nat_eth"), "Total_Total", NAT_ROWS, "CT/17439")
    age = _national(_read("nat_age"), "Total_Total", {**NAT_ROWS, **DIALECTS}, "Table 41")
    for k, v in eth.items():
        if total_row[k] != v:
            raise SystemExit(f"!! {k}: planning-area Total row {total_row[k]} vs CT/17439 {v}")
    for k, v in PRINTED.items():
        if age[k] != v:
            raise SystemExit(f"!! {k}: Table 41 file {age[k]} vs printed {v}")
    if sum(age[v] for v in DIALECTS.values()) != age["Chinese Dialects"]:
        raise SystemExit("!! the four dialects do not sum to Chinese Dialects")
    print("national: planning-area Total row == CT/17439 == printed Table 41, exactly")

    # the borrowed shape: 2010's national dialect mix beside 2020's
    old = _national(_read("nat_2010"), "Total_Total",
                    {"Chinese Dialects": "Chinese Dialects", "Hokkien": "Hokkien",
                     "Teochew": "Teochew", "Cantonese": "Cantonese",
                     "Other Chinese Dialects": "Other Chinese Dialects"}, "CT/5980")
    print("dialect mix, share of Chinese Dialects:   2010     2020")
    for v in DIALECTS.values():
        a, b = old[v] / old["Chinese Dialects"], age[v] / age["Chinese Dialects"]
        print(f"  {v:24s} {a:7.1%}  {b:7.1%}")
        if abs(a - b) > 0.03:
            raise SystemExit(f"!! {v}'s share of the dialects moved {a:.1%} -> {b:.1%}; "
                             "the national split needs rethinking")

    covered = total_row["Total"]
    print(f"universe {covered:,} of {TOTAL_POPULATION:,} ({covered / TOTAL_POPULATION:.1%}); "
          f"non-residents {NON_RESIDENTS:,}, residents outside the question "
          f"{TOTAL_POPULATION - NON_RESIDENTS - covered:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_level", "geo_id", "geo_name", "source_category", "count"])
        for name, vals, tot in rows:
            for lab, n in vals.items():
                w.writerow(["planning_area", name, name, lab, n])
        for k in ["Total", *NAT_ROWS.values(), *DIALECTS.values()]:
            w.writerow(["country", "SG", "Singapore", k, age[k]])
    os.replace(tmp, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    normalise()
