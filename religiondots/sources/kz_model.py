"""Kazakhstan — the ETHNICITY MODEL. **SUPERSEDED as the build, kept as the validation.**

**This no longer draws Kazakhstan.** `sources/kz.py` does, from BNS's own religion × oblast
figures, which turned out to be published after all — through the census's Qlik engine rather
than in any PDF (`sources.md` §9cd, §11u's addendum). This file is retained because the model
it builds is now the only ethnicity→religion model on this map that can be scored against the
truth, and spec §14.25 is written from what it scores.

It writes `data/raw/kz/kz_modelled_oblast.csv` and, if the measured build is on disk, prints
the comparison. Run it as `python sources/kz_model.py`.

**WHAT §11u GOT RIGHT AND WHAT IT MISSED.** Every one of its four checks was correct and
still is: the 2021 religion volume's three chapters are religion × nationality, × age and ×
education and **not one of its 542 pages names an oblast**; the 2009 volume contains `область`
zero times; the 2009 per-region series has no religion on any page; there is no 2021 regional
series. All four are about **documents**. None of them is about the office's dashboard, which
serves the census microdata itself and will cross-tabulate religion against oblast, rayon,
settlement and urban/rural on request. **The negative was true of every PDF and false of the
office.**

**WHAT IS MODELLED, AND WHAT IS NOT.** Both inputs are the same census, published by the same
office, in the same round:

    magnitude     ethnicity × oblast          sheet 2.1 of the ethnos workbook, 17 regions
    coefficients  share(religion | ethnicity) volume ch.12, national, 18 nationalities
    output        religion × oblast           this file, basis `modelled`

    count(oblast, religion) = Σ_ethnicity  pop(oblast, ethnicity) × share(religion|ethnicity)

**NOTHING IS INVENTED AND NOTHING IS SCALED.** §14.4's rule 1 — never estimate a magnitude a
source does not publish — is satisfied by identity: every person placed is a person BNS counted
in that oblast, and the model only decides which column they go in. Both margins come back
exact, and that is arithmetic rather than luck: because the coefficients are conditional on
ethnicity and the ethnic margins agree between the two publications, **every religion's
national total is reproduced to the person and every oblast's population is reproduced to the
person.** check() asserts both.

**THE COEFFICIENTS ARE THE BEST ANY MODELLED COUNTRY HERE HAS.** Greece, Spain and France
multiply a state count by a THIRD PARTY's national composition — Pew's. Kazakhstan's
coefficients are its own census's own cross-tabulation of the two variables, an exact
partition, collected from the same people in the same interview. §14.10's condition 2
(documented and attributable, not fitted) is met about as strongly as it can be.

**AND THERE IS A REAL HELD-OUT TEST, WHICH IS §14.10's CONDITION 5 AND IS USUALLY THE WEAK
ONE.** The volume publishes religion × nationality separately for URBAN and RURAL Kazakhstan.
That is the model's own assumption — that share(religion|ethnicity) does not vary by place —
stated as a testable claim about a partition the model never sees. So:

  * **the model is built from the NATIONAL coefficients only.** Using the urban/rural sets
    would fit better and would consume the only independent check this country has, which is
    a bad trade — spec §14.10's fifth condition is about what the output was checked against,
    and a check you have spent is not one.
  * predicting religion × urban/rural from ethnicity × urban/rural and the national
    coefficients puts **313,745 people — 1.64% of the country — on the wrong side of the
    town/country line**, and the error is not spread evenly:

        Ислам        +1.7% urban / -2.3% rural     Православие   +0.9% / -2.4%
        Неверующие  -13.0% / +41.8%                Отказались    -7.8% / +15.0%
        Католицизм  +43.5% / -34.3%                Протестантизм -17.1% / +105.6%

    **The three cells that are 86% of the country come back within three percent. The two
    that are about ATTITUDE rather than ancestry do not**, because non-belief and refusal are
    urban behaviours inside every ethnic group at once — Kazakhs are 1.4% non-believing in
    town and 0.6% in the country, Koreans 17.3% and 10.7%. Ethnicity cannot see that and this
    model does not claim to. **Catholics and Protestants are wrong by a third to a double**,
    which matters less than it reads: they are 18,988 and 9,419 people, so the whole error is
    ~9,000 either way, and it is the same effect (Kazakhstan's Catholics are a rural
    Polish-and-German population, its Protestants an urban one).

**AND THE REFUSAL IS DRAWN, WHICH REVERSES THIS FILE'S FIRST VERSION.** `Отказались указать`
— 2,112,653 people, **11.01%** — was excluded on §3.5 and tt2011.py's Trinidad precedent until
Anita asked whether it could be drawn (2026-09-07). **The census form settles it.** Question 11
of `Переписной лист 3-И` offers seven options and the sixth is `Отказываюсь указать` — *"I
decline to state"*, printed, numbered, first person. **It is a chosen answer, not a blank**,
which is exactly what Trinidad's derived `Not Stated` residual is not. It goes to `unknown`,
the node branches.py defines as *the one that reports nothing at all… the claim is only that
these people are here*, and drawing it is §3.5 satisfied rather than bent: nobody is
redistributed into a religion, they are marked in place. **Kazakhstan is 100.00% drawn.**
`Неверующие` goes to `secular` on ru2012.py's precedent. See taxonomy/kz2021.py for both.

Usage:
    python sources/kz.py --fetch    one 7.2 MB xlsx and one 24.3 MB PDF, ~1 min
    python sources/kz.py            re-model from data/raw/kz/
"""

import csv
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kz")
OUT = os.path.join(RAW, "kz_modelled_oblast.csv")
MEASURED = os.path.join(ROOT, "data", "normalized", "kz.csv")

SOURCE_ID = "kz_census_2021"
YEAR = 2021
BASIS = "modelled"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Both files hang off stat.gov.kz/ru/national/2021/ as Bitrix object routes. The anchor TEXT
# on that page is the catalogue — there is no Content-Disposition to read (sources.md §11u).
XLSX_URL = ("https://stat.gov.kz/upload/medialibrary/93d/msehzbw870uc729ejpv5eynfszawxgep/"
            "%D0%A7%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%BD%D0%BE%D1%81%D1%82%D1%8C%20"
            "%D0%BD%D0%B0%D1%81%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F%20"
            "%D0%A0%D0%B5%D1%81%D0%BF%D1%83%D0%B1%D0%BB%D0%B8%D0%BA%D0%B8%20"
            "%D0%9A%D0%B0%D0%B7%D0%B0%D1%85%D1%81%D1%82%D0%B0%D0%BD%20%D0%BF%D0%BE%20"
            "%D1%8D%D1%82%D0%BD%D0%BE%D1%81%D0%B0%D0%BC%2C%20"
            "%D0%BD%D0%B0%D1%81%D0%B5%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%BC%20"
            "%D0%BF%D1%83%D0%BD%D0%BA%D1%82%D0%B0%D0%BC%20%D0%B8%20%D0%BF%D0%BE%20"
            "%D0%B2%D0%BE%D0%B7%D1%80%D0%B0%D1%81%D1%82%D0%B0%D0%BC%20(1).xlsx")
XLSX_NAME = "kz2021_ethnos_settlement.xlsx"
PDF_URL = "https://stat.gov.kz/api/iblock/element/100842/file/ru/"
PDF_NAME = "kz2021_ethnic_religion_language.pdf"

NATIONAL = 19_186_015
URBAN = 11_741_342
RURAL = 7_444_673
EXPECTED_REGIONS = 17          # the 2021 vintage: 14 oblasts + Astana, Almaty, Shymkent

# ---- the coefficient table's eleven numeric columns, by x-band on the page. The PDF's
# header is bilingual and split across five lines, so the columns are taken by POSITION and
# the labels below are this project's canonical names for them, not verbatim strings.
COLS = [("total", 165, 225), ("Ислам", 226, 275), ("Христианство", 276, 340),
        ("Православие", 341, 405), ("Католицизм", 406, 470),
        ("Протестантизм", 471, 530), ("Иудаизм", 531, 580), ("Буддизм", 581, 625),
        ("Другое", 626, 675), ("Отказались указать", 676, 745),
        ("Неверующие", 746, 800)]
# `Христианство` is the parent of the next three and is not a category of its own.
PARTS = ["Ислам", "Христианство", "Иудаизм", "Буддизм", "Другое",
         "Отказались указать", "Неверующие"]
CHRISTIAN_SUB = ["Православие", "Католицизм", "Протестантизм"]
DRAWN = ["Ислам", "Православие", "Католицизм", "Протестантизм", "Иудаизм", "Буддизм",
         "Другое", "Отказались указать", "Неверующие"]

# The volume names 18 nationalities and a residual; the workbook names 44 ethnicities. The
# pairing is Kazakh label -> Russian column and is ASSERTED ON THE COUNTS in check(), which
# is what makes it evidence: two separate publications of the same census have to agree to
# the person on all eighteen before any coefficient is used.
PAIR = {
    "Қазақтар": "Казахи", "Орыстар": "Русские", "Украиндар": "Украинцы",
    "Белорустар": "Белорусы", "Өзбектер": "Узбеки", "Әзірбайжандар": "Азербайджанцы",
    "Қырғыздар": "Кыргызы", "Тәжіктер": "Таджики", "Татарлар": "Татары",
    "Шешендер": "Чеченцы", "Дүнгендер": "Дунгане", "Кәрістер": "Корейцы",
    "Күрттер": "Курды", "Немістер": "Немцы", "Поляктар": "Поляки",
    "Түріктер": "Турки", "Ұйғырлар": "Уйгуры",
}
RESIDUAL = "Басқа ұлттар"

# BNS types some workbook headers with LATIN lookalikes -- `Hемцы` begins with U+0048, not
# Cyrillic U+041D, so a plain dict lookup on the ethnicity name misses Germans and only
# Germans. Folded before anything is matched by name. (§12: a mixed-script header is
# invisible in every printout and fails exactly one row.)
CONFUSABLE = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х", "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у", "i": "і",
})


def fold(s):
    return " ".join(str(s).translate(CONFUSABLE).split())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, name, magic, floor in ((XLSX_URL, XLSX_NAME, b"PK", 1_000_000),
                                    (PDF_URL, PDF_NAME, b"%PDF", 5_000_000)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > floor:
            print("already have", dest)
            continue
        print("GET", url[:110])
        r = requests.get(url, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        # sources.md §5a: assert the type, never the absence of an exception.
        with open(dest, "rb") as fh:
            got = fh.read(len(magic))
        if got != magic:
            raise SystemExit(f"{dest} does not start {magic!r} -- got {got!r}, "
                             f"{os.path.getsize(dest):,} bytes")
        print(f"  {os.path.getsize(dest):,} bytes")


def read_magnitude():
    """sheet 2.1 -> (national row, [17 region rows]), ethnicity columns folded."""
    import openpyxl

    src = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    if "2.1" not in wb.sheetnames:
        raise SystemExit(f"no sheet '2.1' in {src} -- found {wb.sheetnames}")
    rows = list(wb["2.1"].iter_rows(values_only=True))
    wb.close()

    hdr = next((i for i, r in enumerate(rows[:10])
                if r and str(r[0]).strip() == "Уровень"), None)
    if hdr is None:
        raise SystemExit("sheet 2.1 has no `Уровень` header row")
    names = [fold(c) if c is not None else "" for c in rows[hdr]]
    eth = [(j, names[j]) for j in range(3, len(names)) if names[j]]
    if not any(n == "Казахи" for _, n in eth):
        raise SystemExit(f"sheet 2.1's ethnicity header changed: {[n for _, n in eth][:8]}")

    out = []
    for r in rows[hdr + 2:]:
        if not r or r[0] in (None, ""):
            continue
        try:
            lvl = int(str(r[0]).strip())
        except ValueError:
            continue
        rec = {"level": lvl, "kato": str(r[1]).strip(), "name": str(r[2]).strip()}
        for j, n in eth:
            v = r[j] if j < len(r) else None
            rec[n] = int(v) if isinstance(v, (int, float)) else 0
        out.append(rec)

    nat = [r for r in out if r["level"] == 0]
    reg = [r for r in out if r["level"] == 1]
    if len(nat) != 1:
        raise SystemExit(f"{len(nat)} level-0 rows, expected 1")
    if len(reg) != EXPECTED_REGIONS:
        raise SystemExit(f"{len(reg)} level-1 regions, expected {EXPECTED_REGIONS} -- "
                         "has the 2022 three-oblast reform reached this file?")
    return nat[0], reg, [n for _, n in eth]


def read_coefficients():
    """volume ch.12 -> {'total'|'urban'|'rural': {nationality: {category: count}}}."""
    import fitz

    src = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    doc = fitz.open(src)
    if doc.page_count == 0:
        raise SystemExit(f"{src} opened with ZERO pages -- truncated at source")

    def merge(ws, lo, hi):
        toks = [w[4] for w in ws if lo <= w[0] < hi]
        if not toks:
            return None
        s = "".join(toks)
        if s in ("-", "—", "–"):
            return 0
        return int(s) if re.fullmatch(r"\d+", s) else None

    recs, sex, pending = [], None, ""
    for pno in range(505, 515):
        bands = {}
        for w in doc[pno].get_text("words"):
            bands.setdefault(round(w[1] / 3.0), []).append(w)
        for k in sorted(bands):
            ws = sorted(bands[k], key=lambda w: w[0])
            whole = " ".join(w[4] for w in ws)
            label = " ".join(w[4] for w in ws if w[0] < 165).strip()
            # the panel marker is CENTRED, not in the label column
            if re.search(r"Оба\s*пола|Екі\s*жыныс", whole):
                sex = "both"
                continue
            if re.search(r"Мужчины|Ерлер", whole):
                sex = "male"
                continue
            if re.search(r"Женщины|Әйелдер", whole):
                sex = "female"
                continue
            vals = {n: merge(ws, lo, hi) for n, lo, hi in COLS}
            vals = {n: v for n, v in vals.items() if v is not None}
            if len(vals) >= 8:
                name = label or pending
                pending = ""
                if name and sex == "both":
                    recs.append({"label": name, **vals})
            elif label and not vals:
                # a nationality whose name sits on its own line above its figures
                pending = label

    # three blocks in print order: whole country, urban, rural. Segmented on the `Барлығы`
    # rows and then ASSERTED against the three published totals, so a missed block is loud.
    blocks, cur = [], None
    for r in recs:
        if r["label"].startswith("Барлығы"):
            cur = {}
            blocks.append(cur)
        if cur is None:
            raise SystemExit("a nationality row before the first `Барлығы` row")
        cur[r["label"]] = r
    if len(blocks) != 3:
        raise SystemExit(f"{len(blocks)} blocks in chapter 12, expected total/urban/rural")
    named = dict(zip(("total", "urban", "rural"), blocks))
    for key, want in (("total", NATIONAL), ("urban", URBAN), ("rural", RURAL)):
        got = named[key]["Барлығы"]["total"]
        if got != want:
            raise SystemExit(f"the {key} block totals {got:,}, expected {want:,} -- the "
                             "three panels are not in the order this parse assumes")
    return named


def model(nat, regions, coef):
    """The whole of it. See the module docstring."""
    groups = list(PAIR) + [RESIDUAL]
    total = coef["total"]
    shares = {g: {c: total[g][c] / total[g]["total"] for c in DRAWN} for g in groups}

    named_ru = set(PAIR.values())
    rest = [c for c in nat if c not in named_ru and c not in ("level", "kato", "name",
                                                              "Всего")]

    def by_group(row):
        out = {kk: row[ru] for kk, ru in PAIR.items()}
        out[RESIDUAL] = sum(row[c] for c in rest)
        return out

    out = {}
    for r in regions:
        pop = by_group(r)
        out[r["kato"]] = {
            "name": r["name"],
            "pop": r["Всего"],
            "counts": {c: sum(pop[g] * shares[g][c] for g in groups) for c in DRAWN},
        }
    return out, shares, rest


def _largest_remainder(vals, target):
    """Integers that sum to `target` exactly, closest to `vals` (spec §4.1a's rule)."""
    floors = {k: int(v) for k, v in vals.items()}
    short = target - sum(floors.values())
    order = sorted(vals, key=lambda k: (vals[k] - floors[k]), reverse=True)
    for k in order[:short]:
        floors[k] += 1
    return floors


def check(nat, regions, coef, modelled, shares, rest):
    ok = True
    total = coef["total"]

    print("  --- the ethnicity pairing, asserted on the counts of two publications ---")
    bad = []
    for kk, ru in PAIR.items():
        if total[kk]["total"] != nat[ru]:
            bad.append((kk, ru, total[kk]["total"], nat[ru]))
    rest_sum = sum(nat[c] for c in rest)
    if total[RESIDUAL]["total"] != rest_sum:
        bad.append((RESIDUAL, "(the other 26)", total[RESIDUAL]["total"], rest_sum))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(PAIR) + 1} nationalities agree to the "
          f"person between the volume and the workbook ({len(bad)} failures)")
    for kk, ru, a, b in bad[:6]:
        print(f"      {kk} / {ru}: volume {a:,} vs workbook {b:,}")

    print("\n  --- the source's own identities ---")
    bad = [g for g in list(PAIR) + [RESIDUAL, "Барлығы"]
           if sum(total[g][c] for c in PARTS) != total[g]["total"]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 7 top-level cells sum to the row total on "
          f"all {len(PAIR) + 2} rows ({len(bad)} failures) {bad[:3]}")
    bad = [g for g in list(PAIR) + [RESIDUAL, "Барлығы"]
           if sum(total[g][c] for c in CHRISTIAN_SUB) != total[g]["Христианство"]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} Orthodox + Catholic + Protestant == Christian "
          f"on all rows ({len(bad)} failures)")

    print("\n  --- the model's margins, which are exact by construction ---")
    for c in DRAWN:
        got = sum(m["counts"][c] for m in modelled.values())
        want = total["Барлығы"][c]
        good = abs(got - want) < 0.5
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {c:<20} model {got:>12,.0f}  "
              f"published {want:>12,}  diff {got - want:>+7,.0f}")
    got = sum(m["pop"] for m in modelled.values())
    good = got == nat["Всего"] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {'population':<20} model {got:>12,}  "
          f"published {NATIONAL:>12,}")

    # ---- §14.10 condition 5: the held-out test. The urban/rural coefficient sets are NOT
    # used by the model; they are the only thing that can falsify its assumption.
    print("\n  --- HELD-OUT TEST: national coefficients -> urban / rural ---")
    print("      the model never sees the urban and rural coefficient sets")
    groups = list(PAIR) + [RESIDUAL]
    tot_abs = 0
    print(f"      {'category':<20} {'area':<6} {'predicted':>12} {'published':>12} "
          f"{'rel err':>9}")
    for area in ("urban", "rural"):
        blk = coef[area]
        for c in DRAWN:
            pred = sum(blk[g]["total"] * shares[g][c] for g in groups)
            obs = blk["Барлығы"][c]
            tot_abs += abs(pred - obs)
            print(f"      {c:<20} {area:<6} {pred:>12,.0f} {obs:>12,} "
                  f"{(pred - obs) / obs if obs else 0:>+8.1%}")
    share = tot_abs / 2 / NATIONAL
    print(f"\n      {tot_abs / 2:,.0f} people misallocated between town and country "
          f"= {share:.2%} of the population")
    print("      Islam and Christianity — 86% of the country — come back within 3%.")
    print("      `Неверующие` and `Отказались указать` do not, because they are about")
    print("      ATTITUDE rather than ancestry. BOTH ARE DRAWN, and they are the two cells")
    print("      note_public tells the reader to hold most loosely (sources/kz.md §5).")
    if share > 0.06:
        raise SystemExit(f"the held-out error is {share:.1%}, which is too large for this "
                         "model to be worth drawing -- see sources/kz.md")

    # Every one of the seven answers Question 11 offers now lands on the tree, including
    # `Отказываюсь указать` (option 6), so nobody is left off. taxonomy/kz2021.py argues it.
    drawn = sum(total["Барлығы"][c] for c in DRAWN)
    good = drawn == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} drawn: {drawn:,} of {NATIONAL:,} = "
          f"{drawn / NATIONAL:.2%} — every answer on the form is on the tree")
    ref = total["Барлығы"]["Отказались указать"]
    print(f"      of which `Отказались указать` {ref:,} ({ref / NATIONAL:.2%}) -> `unknown`, "
          "and it is\n      the model's SECOND-worst cell (-7.8% urban / +15.0% rural above)")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    nat, regions, eth_cols = read_magnitude()
    coef = read_coefficients()
    modelled, shares, rest = model(nat, regions, coef)
    check(nat, regions, coef, modelled, shares, rest)

    rows = []
    for kato, m in modelled.items():
        ints = _largest_remainder(m["counts"], m["pop"])
        note = ("level=region; MODELLED (spec §14.10) = this oblast's ethnic composition "
                "(census, sheet 2.1) x the national share of each religion within each "
                "ethnicity (census, volume ch.12). No magnitude is estimated: the oblast's "
                "population is BNS's own count and the model only splits it")
        rows.append({"geo_id": kato, "geo_level": "region", "geo_name": m["name"],
                     "source_category": "Всего", "count": m["pop"], "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": note + "; universe total, not a religion category"})
        for c in DRAWN:
            rows.append({"geo_id": kato, "geo_level": "region", "geo_name": m["name"],
                         "source_category": c, "count": ints[c], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows)")
    score(rows)


def score(rows):
    """Score the model against the MEASURED build. This is the point of the file now.

    Both sides are (geo_id, source_category) -> count over the same 17 oblasts and the same
    nine categories, and both reproduce the same national margins to the person, so the only
    thing that differs is WHERE each religion was put. Half the summed absolute difference is
    the number of people the model places in the wrong (oblast, religion) cell -- halved
    because every person the model puts somewhere wrong it also fails to put somewhere right,
    and counting both ends double-counts one displacement.
    """
    if not os.path.exists(MEASURED):
        print("no measured build on disk; skipping the score")
        return
    truth = {}
    with open(MEASURED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            truth[(r["geo_id"], r["source_category"])] = int(r["count"])
    if not truth or next(iter(truth.values())) is None:
        return
    mine = {(r["geo_id"], r["source_category"]): r["count"] for r in rows}

    keys = {k for k in mine if k[1] != "Всего"}
    if not keys <= set(truth):
        print("measured build has a different shape; skipping the score")
        return

    print("\n" + "=" * 74)
    print("THE MODEL SCORED AGAINST THE CENSUS (spec §14.25)")
    print("=" * 74)
    tot_pop = sum(v for k, v in truth.items() if k[1] == "Всего")
    misplaced = sum(abs(mine[k] - truth[k]) for k in keys) / 2
    print(f"people in the wrong (oblast, religion) cell: {int(misplaced):,} "
          f"= {100 * misplaced / tot_pop:.2f}% of the country")
    print(f"\n{'category':<22}{'model':>12}{'census':>12}{'misplaced':>12}{'of that':>9}")
    for c in DRAWN:
        ks = [k for k in keys if k[1] == c]
        m = sum(mine[k] for k in ks)
        t = sum(truth[k] for k in ks)
        mis = sum(abs(mine[k] - truth[k]) for k in ks) / 2
        print(f"{c:<22}{m:>12,}{t:>12,}{int(mis):>12,}{100 * mis / max(t, 1):>8.1f}%")
    print("\nEvery national total agrees to within rounding; every error is geography.")


if __name__ == "__main__":
    main()
