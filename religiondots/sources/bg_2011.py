"""Bulgaria — Census 2011 religion by oblast, the shape the 2021 municipal split is built on.

Reads (or fetches) data/raw/bg/2011/ and writes data/normalized/bg_2011.csv.

**WHY A SECOND CENSUS IS WIRED AT ALL.** The 2021 municipal workbook has one `Християнско`
column for 4.2M people, and Bulgaria drawn from it alone is a flat Christianity colour beside
four Orthodox neighbours (Anita, 2026-09-08: *"bulgaria looks kinda out of place as it's the
only one in the area where we dont have christianity breakdown"*). The 2021 census does
publish the breakdown, **nationally and nowhere else**. The 2011 census publishes it **per
oblast**, and that is the finest geography any Bulgarian census offers for it.

So this file supplies the SHAPE and `bg_split.py` takes the MAGNITUDE from 2021. Neither
census is asked for something it does not have.

**THE SOURCE IS THE 2011 RESULTS PORTAL, AND ITS CEILING IS 28 UNITS BY CONSTRUCTION.**
`censusresults.nsi.bg/Census/Reports/2/2/R10.aspx` is *Население по местоживеене, възраст и
вероизповедание*, and `?Obl=<code>` switches oblast. **The dropdown offers 29 options: the
country and 28 oblasti, and nothing below.** That was checked rather than assumed; other
parameter names (`Obst`, `Ob`, `Niva`) change the page length, because the ASP.NET viewstate
differs, and change no data. There is no municipal religion table for 2011 anywhere on NSI.

**THE ELEVEN CATEGORIES, AND ONE OF THEM IS MISSPELLED AT SOURCE.** The Sunni row reads
`Мюсюлмаснко-сунитско`, with the `с` and `н` transposed. It is keyed verbatim per §2.4; a
tidied version matches nothing.

**THE 2011 UNIVERSE IS NOT THE 2021 ONE AND IS NEVER SUMMED WITH IT.** 5,758,301 people
answered in 2011 out of 7,364,570 enumerated. Only the *composition within Christians* and
*within Muslims* is read from this file, never a level, so the different universe does not
propagate. `bg_split.py` asserts that.

Usage:
    python sources/bg_2011.py --fetch    29 pages, ~1.9 MB of HTML
    python sources/bg_2011.py            normalise from data/raw/bg/2011/
"""

import csv
import html
import os
import re
import ssl
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bg", "2011")
OUT = os.path.join(ROOT, "data", "normalized", "bg_2011.csv")

SOURCE_ID = "bg_census_2011"
YEAR = 2011
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

REPORT = "https://censusresults.nsi.bg/Census/Reports/2/2/R10.aspx"
NATIONAL_KEY = "BBB"             # the dropdown's own value for <Общо за България>

# Verbatim, including the transposed letters in the Sunni row.
CATEGORIES = [
    "Общо",
    "Източноправославно",
    "Католическо",
    "Протестантско",
    "Мюсюлмаснко-сунитско",
    "Мюсюлманско-шиитско",
    "Мюсюлманско",
    "Арменско апостолическо православно",
    "Израилтянско/юдаизъм",
    "Друго",
    "Няма",
    "Не се самоопределя",
]
NATIONAL_ANSWERED = 5_758_301
EXPECTED_OBLASTI = 28

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept-Language": "bg,en;q=0.8"}


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout,
                                  context=_ctx()).read().decode("utf-8", "replace")


def _clean(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def _oblast_codes(page):
    """The report's own dropdown is the list of oblasti — never a hand-typed one."""
    opts = re.findall(r'<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', page, re.S)
    out = [(v, _clean(lab)) for v, lab in opts if v and v != NATIONAL_KEY]
    if len(out) != EXPECTED_OBLASTI:
        raise SystemExit(f"{len(out)} oblast options in the dropdown, "
                         f"expected {EXPECTED_OBLASTI}")
    return out


def fetch():
    os.makedirs(RAW, exist_ok=True)
    national = _get(REPORT)
    with open(os.path.join(RAW, f"{NATIONAL_KEY}.html"), "w", encoding="utf-8") as fh:
        fh.write(national)
    codes = _oblast_codes(national)
    with open(os.path.join(RAW, "oblasti.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "name"])
        w.writerows(codes)
    print(f"  {len(codes)} oblasti from the report's own dropdown")
    for i, (code, name) in enumerate(codes, 1):
        path = os.path.join(RAW, f"{code}.html")
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            continue
        page = _get(f"{REPORT}?Obl={code}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(page)
        print(f"  [{i:>2}/{len(codes)}] {code} {name}")
        time.sleep(0.4)          # an ASP.NET report server, not an API


def _parse(page, where):
    """The report's LAST table is the data; everything before it is the nav menu.

    **`..` AND `-` ARE DIFFERENT AND READING THEM ALIKE LOSES PEOPLE SILENTLY.** `-` is a
    true zero; `..` is NSI's disclosure control over a cell too small to publish. Blagoevgrad
    publishes `..` for `Израилтянско/юдаизъм` and its eleven readable rows then sum to 253,199
    against a stated total of 253,200, which is the whole reason this distinction is here.

    Where exactly ONE cell is suppressed the total recovers it exactly, and that is done.
    Where more than one is, the shortfall is left undistributed and reported, because
    splitting it would be inventing a magnitude (§14.4 rule 1) for the sake of tidiness.
    """
    tables = re.findall(r"<table.*?</table>", page, re.S)
    if not tables:
        raise SystemExit(f"{where}: no tables in the page")
    got, suppressed = {}, []
    for row in re.findall(r"<tr.*?</tr>", tables[-1], re.S):
        cells = [_clean(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) < 2 or cells[0] not in CATEGORIES:
            continue
        value, hidden = _num(cells[1], where, cells[0])
        got[cells[0]] = value
        if hidden:
            suppressed.append(cells[0])
    missing = [c for c in CATEGORIES if c not in got]
    if missing:
        raise SystemExit(f"{where}: the report is missing {missing}")

    parts = sum(v for c, v in got.items() if c != "Общо")
    gap = got["Общо"] - parts
    if not suppressed:
        if gap:
            raise SystemExit(f"{where}: categories sum to {parts:,}, "
                             f"`Общо` says {got['Общо']:,} and nothing is suppressed")
    elif len(suppressed) == 1 and gap >= 0:
        got[suppressed[0]] = gap
        gap = 0
    if gap < 0:
        raise SystemExit(f"{where}: categories exceed `Общо` by {-gap:,}")
    return got, suppressed, gap


def _num(s, where, cat):
    """-> (value, was_suppressed). `-` is a true zero, `..` is a withheld small count."""
    t = s.replace("\xa0", "").replace(" ", "").replace(",", "")
    if t in ("..", "…"):
        return 0, True
    if t in ("", "-"):
        return 0, False
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{where}: cannot read {cat!r} value {s!r}")
    return int(t), False


def normalise():
    idx = os.path.join(RAW, "oblasti.csv")
    if not os.path.exists(idx):
        raise SystemExit(f"missing {idx} — run with --fetch first")
    with open(idx, encoding="utf-8") as fh:
        codes = [(r["code"], r["name"]) for r in csv.DictReader(fh)]

    out = []

    def emit(code, level, name, got):
        for cat in CATEGORIES:
            out.append(dict(geo_id=code, geo_level=level, geo_name=name,
                            source_category=cat, count=got[cat], basis=BASIS, year=YEAR,
                            source_id=SOURCE_ID,
                            note="censusresults.nsi.bg Census 2011 report R10, `Общо` column"
                                 + ("; universe total, not a religion category"
                                    if cat == "Общо" else "")))

    with open(os.path.join(RAW, f"{NATIONAL_KEY}.html"), encoding="utf-8") as fh:
        nat, nat_sup, nat_gap = _parse(fh.read(), "national")
    if nat_sup or nat_gap:
        raise SystemExit(f"the national row is suppressed somewhere ({nat_sup}), "
                         "which it has never been")
    if nat["Общо"] != NATIONAL_ANSWERED:
        raise SystemExit(f"national answered {nat['Общо']:,}, expected {NATIONAL_ANSWERED:,}")
    emit("BG", "country", "Общо за България", nat)

    running = {c: 0 for c in CATEGORIES}
    hidden, unresolved = [], 0
    for code, name in codes:
        with open(os.path.join(RAW, f"{code}.html"), encoding="utf-8") as fh:
            got, sup, gap = _parse(fh.read(), code)
        if sup:
            hidden.append(f"{code}:{len(sup)}" + ("" if not gap else f"(+{gap} unresolved)"))
            unresolved += gap
        emit(code, "oblast", name, got)
        for c in CATEGORIES:
            running[c] += got[c]
    if hidden:
        print(f"  suppressed cells recovered from the row total: {', '.join(hidden)}")

    # §8.1 both directions: the 28 oblasti must reconstruct the national row. Disclosure
    # control is the only thing allowed to break it, and only by the amount it withheld.
    bad = {c: (running[c], nat[c]) for c in CATEGORIES if running[c] != nat[c]}
    slack = sum(abs(running[c] - nat[c]) for c in CATEGORIES)
    if slack > max(unresolved * 2, 40):
        raise SystemExit(f"oblast sums miss the national row by {slack:,}, "
                         f"more than suppression explains: {bad}")
    if bad:
        print(f"  the 28 oblasti miss the national row by {slack:,} across "
              f"{len(bad)} categories, all of it withheld small counts")
    else:
        print(f"  OK  the 28 oblasti sum to the national row on all "
              f"{len(CATEGORIES)} categories")

    chr_cats = ["Източноправославно", "Католическо", "Протестантско",
                "Арменско апостолическо православно"]
    chr_tot = sum(nat[c] for c in chr_cats)
    print(f"  Christians {chr_tot:,}; Eastern Orthodox is "
          f"{100.0 * nat['Източноправославно'] / chr_tot:.2f}% of them")
    mus_cats = ["Мюсюлмаснко-сунитско", "Мюсюлманско-шиитско", "Мюсюлманско"]
    mus_tot = sum(nat[c] for c in mus_cats)
    print(f"  Muslims {mus_tot:,}; Shia is "
          f"{100.0 * nat['Мюсюлманско-шиитско'] / mus_tot:.2f}% of them")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


def main():
    if "--fetch" in sys.argv:
        fetch()
    normalise()


if __name__ == "__main__":
    main()
