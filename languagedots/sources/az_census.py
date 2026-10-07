"""Azerbaijan, Population Census 2019: mother tongue by nationality -> data/normalized/az.csv.

    python sources/az_census.py [--fetch]

THE TABLE. *Population Census in the Republic of Azerbaijan 2019*, Volume B (State Statistical
Committee, 2022), Table 30, "National (ethnic) composition and mother tongue of population"
(Ehalinin milli (etnik) terkibi ve ana dili), printed pp.415-429. One answer per person ("ana dili",
mother tongue, the ex-USSR "native language" question). For each of 20 nationalities plus "other",
and for the whole country, it prints the population and its split over 13 mother-tongue columns:

    "language of the nationality (ethnic group) that it belongs", then Azerbaijani, Turkish,
    Russian, Talish, Lezgi, Tat, Kurd, Georgian, Avar, Sakhur, Udin, other languages

with sub-rows for men, women, urban and rural (each split by sex). An "x" marks the column that is
the row's own language (Azerbaijanis' "Azerbaijani" column). THE GRAIN IS THE COUNTRY: no volume
of the 2019 census prints mother tongue below it (Volume A has no language table; Volume B's
language tables, 30 and 31, are national), and the 2009 census's language volume (XIX cild) is not
online. So `geo_id` is "AZ" and `area` is total / urban / rural.

THE POPULATION is the PERMANENT (de jure) one, 9,951,409, which counts the people displaced from
the districts outside government control in 2019 in their district of origin; the census did not
enumerate the territory itself. Nationally the two differ by 7,451 (the EXISTING population is
9,943,958; religiondots `sources/az_geo.py`).

CATEGORIES. `source_category` is the column's language, and for the own-language column
"own language: <nationality>", so taxonomy/az2019.py can name the language each nationality's own
column means (Lezgins -> Lezgian, Ingiloys -> Ingilo, Jews -> the census's "Jewish").

CHECKS, all asserted:
  1. every row's 13 columns sum to its population; men + women and urban + rural equal the total;
  2. the nationality rows sum to the national row, column by column;
  3. the national population is 9,951,409 and each nationality's population equals table 1.11
     (stat.gov.az 001_11-12en.xls, 2019 column, thousands) to 0.1 thousand;
  4. A SECOND TABLE OF THE SAME CENSUS: table 1.12's share of each nationality who "consider the
     language of their nationality native" equals Table 30's own column over its population, to
     the 0.1 point printed.

RAW FILES. The volume is a 29 MB zip from stat.gov.az; religiondots already holds the PDF
(`religiondots/data/raw/az/`), which is read in place, read-only. `--fetch` downloads the zip into
languagedots/data/raw/az/ only if neither copy exists. stat.gov.az's certificate chain is
incomplete, so the fetch does not verify it (as religiondots' az_geo.py).
"""
import os
import re
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD  # noqa: E402

RAW = ROOT / "data" / "raw" / "az"
RD_RAW = RD / "data" / "raw" / "az"
VOL_B_NAME = "Siyahiyaalinma-2019, Cild B.pdf"
VOL_B_URL = ("https://www.stat.gov.az/menu/6/statistical_yearbooks/source/"
             "Siyahiyaalinma-2019,%20Cild%20B.zip")
T111_NAME = "001_11-12en.xls"
T111_URL = "https://www.stat.gov.az/source/demoqraphy/en/001_11-12en.xls"
OUT = ROOT / "data" / "normalized" / "az.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

PAGES = range(414, 429)            # 0-based: printed pp.415-429
NATIONAL = 9_951_409
COLS = ["population", "own", "Azerbaijani", "Turkish", "Russian", "Talysh", "Lezgi", "Tat",
        "Kurdish", "Georgian", "Avar", "Tsakhur", "Udi", "Other languages"]
SUBROWS = {"men", "women", "urban population", "rural population"}
# Table 30's English nationality label -> table 1.11/1.12's label.
NATS = {"Azerbaijani": "Azerbaijanis", "Lezgi": "Lezgins", "Talish": "Talysh",
        "Russian": "Russians", "Ukrainian": "Ukrainians", "Avar": "Avars", "Turkish": "Turks",
        "Tat": "Tats", "Sakhur": "Tsakhurs", "Georgian": "Georgians", "Ingiloy": "Ingiloys",
        "Kurd": "Kurds", "Tatarian": "Tatars", "Griz": "Grysz", "Jews": "Jews", "Udin": "Udins",
        "Khinalig": "Khynalygs", "Budug": "Buduqlus", "Armenian": "Armenians",
        "Khaput": "Haputs", "Other": "other nationalities"}


def path(name):
    for d in (RAW, RD_RAW):
        if (d / name).exists():
            return d / name
    return None


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for name, url, is_zip in ((VOL_B_NAME, VOL_B_URL, True), (T111_NAME, T111_URL, False)):
        if path(name):
            print(f"  have {path(name)}")
            continue
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900,
                                      context=ctx).read()
        dst = RAW / (name.replace(".pdf", ".zip") if is_zip else name)
        dst.with_suffix(".part").write_bytes(data)
        os.replace(dst.with_suffix(".part"), dst)
        print(f"  got  {dst} ({len(data):,} bytes)")
        if is_zip:
            with zipfile.ZipFile(dst) as z:
                z.extractall(RAW)


def read_table30():
    """[(nationality, area, sex, {col: value})] in printed order."""
    import fitz

    doc = fitz.open(path(VOL_B_NAME))
    lines = []
    for p in PAGES:
        lines += [s.strip() for s in doc[p].get_text().split("\n")]
    num = re.compile(r"^(\d+|x|-)$")
    rows, run, label = [], [], None
    nat, area = None, "total"

    def flush():
        nonlocal nat, area
        if len(run) < 14 or run[:14] == [str(i) for i in range(1, 15)]:
            return
        if len(run) not in (14, 15):         # 15: a page number printed after the row
            raise SystemExit(f"a run of {len(run)} values after {label!r}")
        vals = [0 if v in ("x", "-") else int(v) for v in run[:14]]
        lab = (label or "").strip().replace("İ", "I")     # the English line prints "İngiloy"
        if lab == "Republic of Azerbaijan":
            nat, area, sex = "Republic of Azerbaijan", "total", "all"
        elif lab in NATS:
            nat, area, sex = lab, "total", "all"
        elif lab == "urban population":
            area, sex = "urban", "all"
        elif lab == "rural population":
            area, sex = "rural", "all"
        elif lab in ("men", "women"):
            sex = lab
        else:
            raise SystemExit(f"a row labelled {lab!r}")
        if run[2] == "x" and nat != "Azerbaijani":
            raise SystemExit(f"{nat}: an 'x' in the Azerbaijani column")
        rows.append((nat, area, sex, dict(zip(COLS, vals))))

    for s in lines:
        if num.match(s):
            run.append(s)
            continue
        if run:
            flush()
            run = []
        if s:
            label = s
    if run:
        flush()
    return rows


def main():
    if "--fetch" in sys.argv or not (path(VOL_B_NAME) and path(T111_NAME)):
        fetch()
    rows = read_table30()
    langs = COLS[1:]

    # 1. rows sum, sexes and areas sum
    bad = [(n, a, s) for n, a, s, v in rows if sum(v[c] for c in langs) != v["population"]]
    if bad:
        raise SystemExit(f"rows whose columns do not sum to the population: {bad}")
    tab = {(n, a, s): v for n, a, s, v in rows}
    nats = list(dict.fromkeys(n for n, *_ in rows))
    if nats[0] != "Republic of Azerbaijan" or set(nats[1:]) != set(NATS):
        raise SystemExit(f"nationality rows read: {nats}")
    for n in nats:
        for a in ("total", "urban", "rural"):
            if (n, a, "all") not in tab:
                raise SystemExit(f"{n}: no {a} row")
            for c in COLS:
                if tab[(n, a, "men")][c] + tab[(n, a, "women")][c] != tab[(n, a, "all")][c]:
                    raise SystemExit(f"{n} {a} {c}: men + women != total")
        for c in COLS:
            if tab[(n, "urban", "all")][c] + tab[(n, "rural", "all")][c] != tab[(n, "total", "all")][c]:
                raise SystemExit(f"{n} {c}: urban + rural != total")
    print(f"Table 30: {len(rows)} rows, {len(nats) - 1} nationalities; every row sums, men + women "
          "and urban + rural equal the total")

    # 2. nationalities sum to the national row
    for a in ("total", "urban", "rural"):
        for c in COLS:
            s = sum(tab[(n, a, "all")][c] for n in NATS)
            if s != tab[("Republic of Azerbaijan", a, "all")][c]:
                raise SystemExit(f"{a} {c}: nationalities sum to {s:,}, the national row "
                                 f"{tab[('Republic of Azerbaijan', a, 'all')][c]:,}")
    if tab[("Republic of Azerbaijan", "total", "all")]["population"] != NATIONAL:
        raise SystemExit("national population is not 9,951,409")
    print("  the 21 nationality rows sum to the national row in every column, total, urban, rural")

    # 3 and 4. table 1.11 (population) and 1.12 (own-language share)
    t = pd.read_excel(path(T111_NAME), sheet_name=None, header=None)
    t11, t12 = t["1.11"], t["1.12"]
    pop11 = {str(r[1]).strip(): r[10] for _i, r in t11.iloc[7:28].iterrows()}
    own12 = {str(r[1]).strip(): r[3] for _i, r in t12.iloc[7:28].iterrows()}
    bad = []
    for n, lab in NATS.items():
        v = tab[(n, "total", "all")]
        if abs(float(pop11[lab]) * 1000 - v["population"]) > 100:
            bad.append((n, "1.11", v["population"], pop11[lab]))
        share = 100 * v["own"] / v["population"]
        if abs(share - float(own12[lab])) > 0.051:
            bad.append((n, "1.12", round(share, 2), own12[lab]))
    if bad:
        raise SystemExit(f"Table 30 against tables 1.11/1.12: {bad}")
    print("  witness: every nationality's population equals table 1.11 and its own-language share "
          "equals table 1.12, to the rounding printed")

    out = []
    for n in NATS:
        for a in ("total", "urban", "rural"):
            v = tab[(n, a, "all")]
            for c in langs:
                if v[c] == 0:
                    continue
                cat = f"own language: {n}" if c == "own" else c
                out.append(("AZ", "country", a, n, cat, v[c]))
    df = pd.DataFrame(out, columns=["geo_id", "geo_level", "area", "nationality",
                                    "source_category", "count"])
    df["tier"] = "measured"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    tot = df[df["area"] == "total"]
    print(f"\nwrote {OUT}: {len(df):,} rows, {int(tot['count'].sum()):,} people")
    print("  mother tongue x nationality, the largest 30 cells:")
    for _i, r in tot.sort_values("count", ascending=False).head(30).iterrows():
        print(f"    {r['nationality']:<12} {r['source_category']:<28} {r['count']:>10,}")


if __name__ == "__main__":
    main()
