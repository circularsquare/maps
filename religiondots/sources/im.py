"""Isle of Man: 2021 Isle of Man Census Report Part I, Table 2.12, residents by religious faith.

Fetches the two census reports into data/raw/im/ and Kontur's IM extract into data/raw/micro/,
and writes

    data/normalized/im.csv       the island's six answers, geo_level `country`, plus the resident
                                 population and the non-answer as EXCLUDED rows
    data/geo/im/im_hexes.gpkg    Kontur 400 m population hexagons, unit = IM

`sources/im.md` is the write-up; `taxonomy/im2021.py` is the mapping.

## THE TABLE

Statistics Isle of Man (Cabinet Office), *2021 Isle of Man Census Report Part I*, 42 pages.
Table 2.12, *Residents by Religious Faith and Age*, printed p.27: Buddhism, Christianity,
Hinduism, Islam, Judaism, Other, No Religion and Total, by 18 quinary age bands. Island-wide
only; no religion table by area is in Part I or Part II. Footnote 14: "This question was
voluntary. The total is that of all people who chose to provide an answer." p.13: the first Manx
census to ask religion.

The resident population is 84,069 (Table 1.1, Table 2.1, Table 2.4), so 9,582 residents (11.4%)
are not in the table. Part II's form facsimile (Q8, p.43) offers Christianity, Judaism, Buddhism,
Sikhism, Hinduism, No religion, Islam and Other (please specify below). **Table 2.12 has no Sikhism
row and prints Other as 0 in every cell**, so Sikh and write-in answers are either coded into the
listed faiths or left out of the total; the report does not say which.

## DRAWN AS ONE UNIT ON KONTUR

No finer table exists, so this is the microstate shape (`countries/_shared.py::_micro_counts`):
one unit, every row measured, placed on Kontur hexes. Kontur holds 84,591 people against the
census's 84,069; no calibration to Table 2.4's 21 areas was made (no area polygons in
geoBoundaries, Overpass answered 504 on 2026-09-15), and a rough town-buffer witness is printed.

## THE CHECKS

Both PDFs pinned by digest. Table 2.12 parsed off the page equals the transcription; every row's
age bands sum to its total; the rows sum to the total row in every band; Table 2.1's total row
parsed and equal to 84,069; answered <= residents in every age band; p.13's figures (74,487
answering, 54.7% Christian of them, 38.8% no religion of the resident population, and the six
printed shares) recompute; Q8 on the form offers the eight boxes including Sikhism and Other.

Usage:
    python sources/im.py --fetch
    python sources/im.py
"""

import csv
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "im")
OUT = os.path.join(ROOT, "data", "normalized", "im.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import micro                                                   # noqa: E402  Kontur, COLUMNS, UA
from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402

SOURCE_ID = "iom_census2021_t2_12"
YEAR = 2021
BASIS = "self_id"
COLUMNS = micro.COLUMNS

# (url, local name, size, digest, pages). gov.im/media/ gives a 269-byte "Request Rejected" page
# to a bare `Mozilla/5.0` user agent AND to urllib with micro.UA (2026-09-15), and served the PDF
# to curl with a Chrome/124 string. The files in data/raw/im/ came from curl; fetch() skips them on
# size, and the digest pins in check() are the verdict on the bytes.
FILES = {
    "part1": ("https://www.gov.im/media/1375604/2021-01-27-census-report-part-i-final-2.pdf",
              "2021-isle-of-man-census-report-part-i.pdf", 1_691_173,
              "QVOV43VF23Y4GK26O2U25C6UMWBJU4SV", 42),
    "part2": ("https://www.gov.im/media/1376421/2021-isle-of-man-census-report-part-ii_11052022.pdf",
              "2021-isle-of-man-census-report-part-ii.pdf", 482_408,
              "CHTCZ3EG5BDN4SR6PTBEYQSIA3EDTVEH", 49),
}

PAGE_PROSE = 13      # 0-based index = printed page; p.13 religion prose
PAGE_T21 = 18        # p.18 Table 2.1
PAGE_T212 = 27       # p.27 Table 2.12
PAGE_FORM = 42       # Part II p.43, individual questions Q5-Q8

BANDS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
         "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
# Table 2.12, transcribed: label -> (total, 18 age bands)
T212 = {
    "Buddhism": (390, 5, 15, 15, 12, 12, 27, 45, 43, 44, 44, 38, 19, 28, 22, 14, 2, 2, 3),
    "Christianity": (40725, 1074, 1647, 1955, 1807, 1520, 1309, 1490, 1762, 2102, 2664, 3304,
                     3640, 3508, 3219, 3543, 2645, 1870, 1666),
    "Hinduism": (263, 17, 31, 19, 9, 10, 8, 26, 43, 42, 10, 13, 16, 7, 3, 4, 1, 1, 3),
    "Islam": (393, 22, 20, 16, 21, 34, 38, 40, 30, 39, 40, 36, 22, 19, 8, 3, 1, 1, 3),
    "Judaism": (113, 6, 7, 9, 8, 1, 5, 3, 7, 8, 12, 7, 8, 6, 7, 10, 5, 2, 2),
    "Other": (0,) * 19,
    "No Religion": (32603, 1944, 2046, 2237, 2061, 2212, 2451, 2595, 2531, 2423, 2363, 2504,
                    2210, 1588, 1171, 1051, 650, 339, 227),
}
T212_TOTAL = (74487, 3068, 3766, 4251, 3918, 3789, 3838, 4199, 4416, 4658, 5133, 5902, 5915,
              5156, 4430, 4625, 3304, 2215, 1904)
T21_TOTAL = (84069, 3483, 4245, 4786, 4448, 4309, 4364, 4741, 4954, 5242, 5798, 6648, 6713,
             5769, 4898, 5156, 3671, 2489, 2355)
RESIDENTS = 84_069
ANSWERED = 74_487
FORM_BOXES = ["Christianity", "Judaism", "Buddhism", "Sikhism", "Hinduism", "No religion",
              "Islam", "Other (please specify below)"]
P13_SHARES = {"Christianity": "54.7%", "No Religion": "43.8%", "Islam": "0.5%",
              "Buddhism": "0.5%", "Hinduism": "0.4%", "Judaism": "0.2%"}

UNIVERSE = "Resident population (Table 2.1 total)"
NO_ANSWER = "Did not answer (resident population less Table 2.12 total)"

# Rough town-buffer witness for Kontur: (name, lon, lat, radius km, census area population).
TOWNS = (("Douglas and Onchan", -4.4820, 54.1520, 3.0, 26677 + 9039),
         ("Ramsey", -4.3830, 54.3220, 1.5, 8288),
         ("Peel", -4.6960, 54.2230, 1.2, 5710),
         ("Castletown", -4.6530, 54.0740, 1.0, 3206),
         ("Port Erin and Port St Mary", -4.7400, 54.0800, 2.0, 3730 + 1989))


def _path(key):
    return os.path.join(RAW, FILES[key][1])


def _num(tok):
    if not re.fullmatch(r"\d{1,3}(?:,\d{3})*", tok):
        raise ValueError(f"not a count: {tok!r}")
    return int(tok.replace(",", ""))


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    for key, (url, _name, size, dig, _pages) in FILES.items():
        path = _path(key)
        if os.path.exists(path) and os.path.getsize(path) == size:
            print("already have", path)
            continue
        req = urllib.request.Request(url, headers=micro.UA)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            check_body(body, "pdf", where=url, pin_digest=dig)
        except (FetchCheckError, OSError) as e:
            raise SystemExit(f"{url}: {e}; give Anita the URL (AGENT_BRIEF, blocked downloads) "
                             f"and save it as {path}")
        with open(path + ".part", "wb") as fh:
            fh.write(body)
        os.replace(path + ".part", path)                     # [[reference_wb_truncates]]
        print(f"wrote {path} ({len(body):,} bytes)")
    micro.fetch(["im"])


def _rows(text, start_marker, labels, stop_marker):
    at = text.find(start_marker)
    stop = text.find(stop_marker, at)
    if at < 0 or stop < 0:
        raise SystemExit(f"no {start_marker!r} ... {stop_marker!r}")
    body = text[at + len(start_marker):stop]
    out, pos, spans = {}, 0, []
    for lab in labels:
        m = re.compile(r"(?<![A-Za-z])" + re.escape(lab) + r"(?![A-Za-z])").search(body, pos)
        if not m:
            raise SystemExit(f"no row {lab!r}")
        spans.append((lab, m.start(), m.end()))
        pos = m.end()
    for k, (lab, _s, e) in enumerate(spans):
        end = spans[k + 1][1] if k + 1 < len(spans) else len(body)
        toks = body[e:end].split()
        if len(toks) != 19:
            raise SystemExit(f"row {lab!r}: {len(toks)} tokens, expected 19: {toks}")
        out[lab] = tuple(map(_num, toks))
    return out


def check(p1, p2):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Isle of Man: 2021 Census Report Part I, Table 2.12\n")
    for key, doc in (("part1", p1), ("part2", p2)):
        with open(_path(key), "rb") as fh:
            body = fh.read()
        say(digest(body) == FILES[key][3] and doc.page_count == FILES[key][4],
            f"{FILES[key][1]}: digest {digest(body)}, {doc.page_count} pages")

    t = p1.load_page(PAGE_T212).get_text()
    header = "Table 2.12: Residents by Religious Faith and Age"
    # Stop before the footnote's own marker: "12 Data unavailable ..." sits right after the rows.
    got = _rows(t, "85+", list(T212) + ["Total"], "12 Data unavailable") if header in t else None
    say(got is not None and all(got[k] == v for k, v in T212.items())
        and got["Total"] == T212_TOTAL,
        "Table 2.12 parsed off p.27 equals the transcription: 8 rows x 19 columns")
    say(all(sum(v[1:]) == v[0] for v in list(T212.values()) + [T212_TOTAL]),
        "every row's 18 age bands sum to its total")
    say(all(sum(v[i] for v in T212.values()) == T212_TOTAL[i] for i in range(19)),
        f"the seven rows sum to the total row in all 19 columns; answered {T212_TOTAL[0]:,}")
    say(T212_TOTAL[0] == ANSWERED and "This question was voluntary. The total is that of all "
        "people who chose to provide an answer." in re.sub(r"\s+", " ", t),
        "footnote 14: the question was voluntary and the total is those who answered")
    say(all(v == 0 for v in T212["Other"]) and "Sikh" not in t,
        "Other is 0 in every cell and the table has no Sikhism row")

    t21 = p1.load_page(PAGE_T21).get_text()
    tot21 = _rows(t21, "Santon", ["Total"], "5 For ease of comparison")["Total"]
    say(tot21 == T21_TOTAL and sum(T21_TOTAL[1:]) == RESIDENTS,
        f"Table 2.1's total row (p.18) parsed and closes on {RESIDENTS:,} residents")
    t11 = re.sub(r"\s+", " ", p1.load_page(15).get_text())
    say("Resident population 84,069" in t11, "Table 1.1 (p.15): resident population 84,069")
    gaps = [r - a for r, a in zip(T21_TOTAL, T212_TOTAL)]
    say(all(g >= 0 for g in gaps),
        f"answered <= residents in every age band; not answered {gaps[0]:,} "
        f"({100 * gaps[0] / RESIDENTS:.2f}%), from {min(g / r for g, r in zip(gaps[1:], T21_TOTAL[1:])):.1%} "
        f"to {max(g / r for g, r in zip(gaps[1:], T21_TOTAL[1:])):.1%} by band "
        f"(85+ {gaps[-1] / T21_TOTAL[-1]:.1%})")

    prose = re.sub(r"\s+", " ", p1.load_page(PAGE_PROSE).get_text())
    say("Of the 74,487 residents who answered" in prose
        and f"{100 * 40725 / ANSWERED:.1f}%" == "54.7%" and "54.7% reported that they were Christian" in prose
        and f"{100 * 32603 / RESIDENTS:.1f}%" == "38.8%" and "38.8% of the resident population" in prose,
        "p.13: 74,487 answered; 54.7% Christian of them; no religion 38.8% of all residents")
    shares = {k: f"{100 * T212[k][0] / ANSWERED:.1f}%" for k in P13_SHARES}
    say(shares == P13_SHARES and all(v in prose for v in P13_SHARES.values()),
        f"p.13's six shares of those answering recompute: {shares}")
    say("first Manx census to ask about the religious beliefs" in prose,
        "p.13: the first Manx census to ask religion")

    form = re.sub(r"\s+", " ", p2.load_page(PAGE_FORM).get_text())
    q = form.find("What is your religion? (voluntary)")
    q8 = form[q:form.find("Q9", q)] if q >= 0 else ""
    say(q >= 0 and all(b in q8 for b in FORM_BOXES),
        "Part II form Q8 (p.43), voluntary, offers " + ", ".join(FORM_BOXES))
    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    note = ("2021 Isle of Man Census Report Part I, Table 2.12 (printed p.27), total column; "
            "{:.2f}% of the 74,487 who answered")
    rows = [dict(geo_id="IM", geo_level="country", geo_name="Isle of Man", source_category=c,
                 count=v[0], basis=BASIS, year=YEAR, source_id=SOURCE_ID,
                 note=note.format(100.0 * v[0] / ANSWERED))
            for c, v in sorted(T212.items(), key=lambda kv: -kv[1][0]) if v[0] > 0]
    rows.append(dict(geo_id="IM", geo_level="country", geo_name="Isle of Man",
                     source_category=UNIVERSE, count=RESIDENTS, basis=BASIS, year=YEAR,
                     source_id=SOURCE_ID,
                     note="Part I Tables 1.1, 2.1 and 2.4; universe total, not a religion category"))
    rows.append(dict(geo_id="IM", geo_level="country", geo_name="Isle of Man",
                     source_category=NO_ANSWER, count=RESIDENTS - ANSWERED, basis=BASIS,
                     year=YEAR, source_id=SOURCE_ID,
                     note="84,069 residents less the 74,487 in Table 2.12; not printed as a cell"))
    return rows


def geometry():
    """Kontur hexes -> data/geo/im/im_hexes.gpkg, all on the one unit, with a town witness."""
    import gzip
    import shutil

    import geopandas as gpd
    from shapely.geometry import Point

    from geo_checks import read_layer

    gz, gpkg = micro._kontur_paths("im")
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 0):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    hexes = read_layer(gpkg, "Kontur IM")
    hexes = hexes.rename(columns={"population": "pop"})
    hexes = hexes[hexes["pop"] > 0].copy().to_crs("EPSG:4326").reset_index(drop=True)
    hexes["unit"] = "IM"
    hexes["cellcode"] = "IM:" + hexes.index.astype(str)
    w, s, e, n = hexes.total_bounds
    if not (-4.9 < w < e < -4.25 and 54.0 < s < n < 54.45):
        raise SystemExit(f"im: Kontur bbox {w:.3f},{s:.3f},{e:.3f},{n:.3f} is not the Isle of Man")
    k = float(hexes["pop"].sum())
    ratio = k / RESIDENTS
    # Kontur 2023-11 against a count of 30 May 2021, two and a half years apart.
    if not 0.9 <= ratio <= 1.15:
        raise SystemExit(f"im: Kontur {k:,.0f} against {RESIDENTS:,} residents, ratio {ratio:.3f}")
    m = hexes.to_crs("EPSG:32630")
    cent = m.geometry.centroid
    print(f"\n  Kontur {len(hexes)} hexes, {k:,.0f} people, ratio {ratio:.3f} to the resident "
          "population. Rough witness (buffers, not the areas' boundaries, so Kontur reads low):")
    for name, lon, lat, r_km, census in TOWNS:
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32630").iloc[0]
        kk = float(hexes.loc[(cent.distance(p) <= r_km * 1000).values, "pop"].sum())
        print(f"    {name:<28} within {r_km} km: Kontur {100 * kk / k:5.1f}%   census area "
              f"{100 * census / RESIDENTS:5.1f}%")
    geo = os.path.join(ROOT, "data", "geo", "im")
    os.makedirs(geo, exist_ok=True)
    out = os.path.join(geo, "im_hexes.gpkg")
    hexes[["cellcode", "unit", "pop", "geometry"]].to_file(out + ".part.gpkg", layer="hexes",
                                                           driver="GPKG")
    os.replace(out + ".part.gpkg", out)
    print(f"  wrote {out}")


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    for key in FILES:
        if not os.path.exists(_path(key)):
            raise SystemExit(f"{_path(key)} missing; run: python sources/im.py --fetch")
    check(fitz.open(_path("part1")), fitz.open(_path("part2")))
    rows = emit()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print(f"\nwrote {OUT} ({len(rows)} rows)")
    geometry()


if __name__ == "__main__":
    main()
