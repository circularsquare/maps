"""Bulgaria: NSI, Census 2021 (Преброяване 2021), mother tongue by obshtina (municipality).

    python sources/bg_census.py --fetch    download the workbook (79 KB) and press release if missing
    python sources/bg_census.py            normalise from data/raw/bg/

-> data/normalized/bg.csv (levels `country`, `nuts1`, `nuts2`, `oblast`, `obshtina`;
   alternatives, never summed; only `obshtina` is drawn)

THE TABLE. NSI's Census 2021 results page (https://www.nsi.bg/statistical-data/151/1349) lists
nine workbooks. `Census2021_Ethnocultural characteristics_BG.xlsx` has four sheets: 1 ethnic group,
national; 2 ethnic group by obshtina; 3 MOTHER TONGUE BY OBSHTINA (the drawn table); 4 religion
by obshtina (religiondots' table, religiondots/sources/bg.py). Sheet 3 has eight columns besides
the total: Bulgarian, Turkish, Romani, `Друг` (other), cannot determine, do not wish to answer,
and `Непоказан`, people added from administrative registers who were never asked (footnote 1).
Nothing finer than "other" is published at any level: the press release
(https://www.nsi.bg/file/24016/Census2021-ethnos.pdf) prints the same four languages nationally
and by oblast. Mother tongue is defined there as "the first language learned at home in early
childhood"; the question was voluntary.

THE GEO_IDS ARE NSI'S OWN CODES (oblast `VID`, obshtina `VID09`), which religiondots' placement
layer carries as its `unit` (GISCO's LAU_ID is the same code), so the join is the identity.

CHECKS: the header; the level counts (1, 2, 6, 28, 265); the national row against the press
release's figures; the eight categories partition every row; obshtini sum to their oblast and
oblasti to the country in every column; the 28 oblast rows against the press release's Table 2,
parsed from the PDF (a second printing of the same counts); every obshtina's total against
sheets 2 and 4 of the same census, and `Непоказан` against sheet 4's `Непоказано` (the same
register-added people); every obshtina total against religiondots' bg.csv; and, as an
independent-ish sanity check, mother tongue against ethnic group by obshtina (sheet 2).
"""

import csv
import os
import re
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bg")
OUT = os.path.join(ROOT, "data", "normalized", "bg.csv")
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "bg.csv")

SOURCE_ID = "bg_census_2021_mt"
YEAR = 2021

RESULTS_PAGE = "https://www.nsi.bg/statistical-data/151/1349"
WORKBOOK_LABEL = "Census2021_Ethnocultural characteristics_BG.xlsx"
WORKBOOK_FALLBACK = ("https://www.nsi.bg/file/download/"
                     "d6bebedae9d8dc7824e050bfc47124b402d9129b")
WORKBOOK = os.path.join(RAW, "Census2021_Ethnocultural_BG.xlsx")
PRESS_URL = "https://www.nsi.bg/file/24016/Census2021-ethnos.pdf"
PRESS_PDF = os.path.join(RAW, "Census2021-ethnos.pdf")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

TOTAL = "Общо"
MT_CATS = ["Общо", "Български", "Турски", "Ромски", "Друг", "Не мога да определя",
           "Не желая да отговоря", "Непоказан1"]
ETH_CATS = ["Общо", "Българска", "Турска", "Ромска", "Друга", "Не мога да определя",
            "Не желая да отговоря", "Непоказана1"]
REL_CATS = ["Общо", "Християнско", "Мюсюлманско", "Юдейско", "Друго", "Нямам",
            "Не мога да определя", "Не желая да отговоря", "Непоказано1"]
HEADER_ROW = 3

NATIONAL = 6_519_789
# The press release's text, p. 6: the national row, to the person.
PRESS_NATIONAL = {"Български": 5_037_607, "Турски": 514_386, "Ромски": 227_974,
                  "Друг": 62_906, "Не мога да определя": 10_633,
                  "Не желая да отговоря": 49_602, "Непоказан1": 616_681}

EXPECTED = {"country": 1, "nuts1": 2, "nuts2": 6, "oblast": 28, "obshtina": 265}
OBLAST_CODE = re.compile(r"^[A-Z]{3}$")
OBSHTINA_CODE = re.compile(r"^[A-Z]{3}\d{2}$")
NUTS_CODE = re.compile(r"^BG\d*$")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept-Language": "bg,en;q=0.8"}


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, timeout=240):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout, context=_ctx()).read()


def _save(url, path, what, magic):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  have {what} ({os.path.getsize(path):,} bytes)")
        return
    body = _get(url)
    if not body.startswith(magic):
        raise SystemExit(f"{what}: {url} did not return the file (starts {body[:16]!r})")
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, path)
    print(f"  got {what} ({len(body):,} bytes) from {url}")


def _workbook_url():
    """The link labelled as the ethnocultural workbook, off the results page (the download
    hash is opaque and would change on a re-publication)."""
    try:
        page = _get(RESULTS_PAGE, timeout=90).decode("utf-8", "replace")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! could not read {RESULTS_PAGE} ({exc}); using the recorded hash")
        return WORKBOOK_FALLBACK
    for href, label in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        if WORKBOOK_LABEL in re.sub(r"<[^>]+>", "", label):
            return "https://www.nsi.bg" + href if href.startswith("/") else href
    print(f"  !! {RESULTS_PAGE} no longer labels {WORKBOOK_LABEL!r}; using the recorded hash")
    return WORKBOOK_FALLBACK


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _save(_workbook_url(), WORKBOOK, "the ethnocultural workbook", b"PK")
    _save(PRESS_URL, PRESS_PDF, "the ethnocultural press release", b"%PDF")


def _num(v, where):
    """NSI writes an empty cell as '-'. A suppressed '..' would be a loss, so it stops."""
    if v is None:
        return 0
    s = str(v).strip().replace("\xa0", "").replace(" ", "")
    if s in ("", "-"):
        return 0
    if s in ("..", "…"):
        raise SystemExit(f"{where}: suppressed cell {v!r}; decide how to recover it")
    f = float(s)
    if f != int(f):
        raise SystemExit(f"{where}: {v!r} is not a whole number")
    return int(f)


def _sheet(name, cats):
    """{code: (level, name, {category: count})} in sheet order."""
    import openpyxl

    if not os.path.exists(WORKBOOK):
        raise SystemExit(f"missing {WORKBOOK}; run with --fetch first")
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    rows = list(wb[name].iter_rows(values_only=True))
    header = [("" if c is None else str(c)).strip() for c in rows[HEADER_ROW]]
    got = [h for h in header if h]
    if got != cats:
        raise SystemExit(f"sheet {name}: header {got} is not {cats}; the layout changed")
    out = {}
    for i, r in enumerate(rows[HEADER_ROW + 1:], start=HEADER_ROW + 1):
        code = ("" if r[0] is None else str(r[0])).strip()
        if not code:
            if any(r[2:]):
                raise SystemExit(f"sheet {name} row {i}: values with no code: {r}")
            continue
        if NUTS_CODE.match(code):
            level = {2: "country", 3: "nuts1", 4: "nuts2"}[len(code)]
        elif OBLAST_CODE.match(code):
            level = "oblast"
        elif OBSHTINA_CODE.match(code):
            level = "obshtina"
        else:
            continue                                # notes and legend at the foot
        vals = {c: _num(r[2 + j], f"sheet {name} row {i} {c}") for j, c in enumerate(cats)}
        if code in out:
            raise SystemExit(f"sheet {name}: code {code} twice")
        out[code] = (level, str(r[1]).strip(), vals)
    return out


def _press_table2():
    """Table 2 of the press release (p. 14): oblast name -> the 8 counts, in MT_CATS order."""
    import fitz

    doc = fitz.open(PRESS_PDF)
    text = doc[13].get_text()
    if "Население по майчин език и по области" not in text:
        raise SystemExit("press release p. 14 is not Table 2 any more")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # the header is broken across lines; the body starts at the `Общо` row followed by numbers
    start = next(i for i, ln in enumerate(lines)
                 if ln == "Общо" and re.fullmatch(r"\d+", lines[i + 1]))
    out, i = {}, start
    while i + 8 < len(lines) + 1:
        name = lines[i]
        nums = lines[i + 1:i + 9]
        if not all(re.fullmatch(r"\d+", n) for n in nums):
            break
        out[name] = [int(n) for n in nums]
        i += 9
    return out


def read():
    mt = _sheet("3", MT_CATS)
    rows = []
    for code, (level, name, vals) in mt.items():
        for cat, n in vals.items():
            note = f"level={level}"
            if cat == TOTAL:
                note += "; universe total, not a language"
            elif cat == "Непоказан1":
                note += "; added from administrative registers, never asked"
            rows.append(dict(geo_id=code, geo_level=level, geo_name=name, source_category=cat,
                             count=n, tier="measured", year=YEAR, source_id=SOURCE_ID,
                             note=note))
    return rows, mt


def _say(ok, msg):
    print(f"  {'OK ' if ok else 'BAD'} {msg}")
    return ok


def check(mt):
    ok = True
    by = {}
    for code, (level, name, vals) in mt.items():
        by.setdefault(level, []).append(code)
    for lv, want in EXPECTED.items():
        ok &= _say(len(by.get(lv, ())) == want, f"{lv:<9} {len(by.get(lv, ())):>4} rows "
                                                 f"(expected {want})")

    nat = mt["BG"][2]
    ok &= _say(nat[TOTAL] == NATIONAL, f"national total {nat[TOTAL]:,} (published {NATIONAL:,})")
    bad = {k: (nat[k], v) for k, v in PRESS_NATIONAL.items() if nat[k] != v}
    ok &= _say(not bad, "national row equals the press release's text in all 7 categories")
    for k, (a, b) in bad.items():
        print(f"        {k}: {a:,} vs {b:,}")

    bad = [c for c, (_, _, v) in mt.items() if sum(v.values()) - v[TOTAL] != v[TOTAL]]
    ok &= _say(not bad, f"the 7 categories partition all {len(mt)} rows")
    for c in bad[:6]:
        print(f"        {c}")

    # obshtina -> oblast by code prefix; oblast -> country
    bad = []
    for ob in by["oblast"]:
        kids = [c for c in by["obshtina"] if c[:3] == ob]
        for k in MT_CATS:
            s = sum(mt[c][2][k] for c in kids)
            if s != mt[ob][2][k]:
                bad.append((ob, k, s, mt[ob][2][k]))
    ok &= _say(not bad, "obshtini sum to their oblast in every column")
    for x in bad[:6]:
        print("       ", x)
    for lv in ("nuts1", "nuts2", "oblast", "obshtina"):
        bad = [k for k in MT_CATS if sum(mt[c][2][k] for c in by[lv]) != nat[k]]
        ok &= _say(not bad, f"the {len(by[lv])} {lv} rows sum to the country in every column")

    # Table 2 of the press release, parsed from the PDF.
    if os.path.exists(PRESS_PDF):
        t2 = _press_table2()
        names = {mt[c][1]: c for c in by["oblast"]}
        miss = sorted(set(names) ^ (set(t2) - {"Общо"}))
        bad = [(n, mt[c][2][k], t2[n][j]) for n, c in names.items() if n in t2
               for j, k in enumerate(MT_CATS) if mt[c][2][k] != t2[n][j]]
        good = not miss and not bad and t2.get("Общо") == [nat[k] for k in MT_CATS]
        ok &= _say(good, f"press release Table 2: all {len(names)} oblast rows and the "
                         "national row equal the workbook in all 8 columns")
        for m in miss:
            print(f"        only on one side: {m}")
        for x in bad[:6]:
            print("       ", x)
    else:
        ok &= _say(False, f"{PRESS_PDF} missing; run --fetch")

    # Same census, other sheets: the same people per obshtina.
    eth = _sheet("2", ETH_CATS)
    rel = _sheet("4", REL_CATS)
    obs = by["obshtina"]
    bad = [c for c in obs if not (mt[c][2][TOTAL] == eth[c][2][TOTAL] == rel[c][2][TOTAL])]
    ok &= _say(not bad, f"all {len(obs)} obshtina totals equal the ethnicity and religion "
                        "sheets'")
    bad = [c for c in obs if mt[c][2]["Непоказан1"] != rel[c][2]["Непоказано1"]]
    ok &= _say(not bad, "`Непоказан` equals the religion sheet's `Непоказано` in every obshtina "
                        "(the same register-added people)")
    diff_eth = sum(abs(mt[c][2]["Непоказан1"] - eth[c][2]["Непоказана1"]) for c in obs)
    print(f"      (the ethnicity sheet's `Непоказана` is {eth['BG'][2]['Непоказана1']:,}, "
          f"not {nat['Непоказан1']:,}; per-obshtina absolute difference {diff_eth:,}; a "
          "register can carry ethnicity where it does not carry language)")

    if os.path.exists(RD_NORM):
        rd = {}
        with open(RD_NORM, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["geo_level"] == "obshtina" and row["source_category"] == "Общо":
                    rd[row["geo_id"]] = int(row["count"])
        mine = {c: mt[c][2][TOTAL] for c in obs}
        miss = sorted(set(mine) ^ set(rd))
        diff = [c for c in mine if c in rd and mine[c] != rd[c]]
        ok &= _say(not miss and not diff, f"all {len(mine)} obshtina codes match religiondots' "
                                          "bg.csv both ways, with identical totals")
        for c in miss[:6] + diff[:6]:
            print(f"        {c}")
    else:
        ok &= _say(False, f"religiondots' {RD_NORM} is missing; join unchecked")

    # Sanity, not identity: mother tongue against ethnic group, per obshtina.
    print("\n  mother tongue against ethnic group, by obshtina (answers to each, Непоказан "
          "excluded):")
    for lang, grp in (("Турски", "Турска"), ("Ромски", "Ромска"), ("Български", "Българска")):
        xs = [mt[c][2][lang] for c in obs]
        ys = [eth[c][2][grp] for c in obs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        r = sxy / (sxx * syy) ** 0.5
        print(f"    {lang:<10} {sum(xs):>10,}   {grp:<10} {sum(ys):>10,}   ratio "
              f"{sum(xs) / sum(ys):.3f}   r = {r:.4f}")
        if lang != "Български":
            ok &= _say(r > 0.95, f"{lang} mother tongue tracks {grp} ethnicity across "
                                 "obshtini (r > 0.95)")
    # where Turkish mother tongue most exceeds Turkish ethnicity (Roma and Pomak Turkish-speakers)
    over = sorted(obs, key=lambda c: -(mt[c][2]["Турски"] - eth[c][2]["Турска"]))[:5]
    print("    Turkish MT minus Turkish ethnicity, largest:",
          ", ".join(f"{mt[c][1]} {mt[c][2]['Турски'] - eth[c][2]['Турска']:+,}" for c in over))
    under = sorted(obs, key=lambda c: (mt[c][2]["Ромски"] - eth[c][2]["Ромска"]))[:5]
    print("    Romani MT minus Roma ethnicity, most negative:",
          ", ".join(f"{mt[c][1]} {mt[c][2]['Ромски'] - eth[c][2]['Ромска']:+,}" for c in under))

    print("\n  categories, national:")
    for k, v in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>10,}  {100.0 * v / NATIONAL:6.2f}%  {k}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, mt = read()
    check(mt)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
