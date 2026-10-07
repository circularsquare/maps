"""Romania: INS, Recensământul Populaţiei şi Locuinţelor 2021, mother tongue, down to UAT.

    python sources/ro_census.py --fetch    download the workbook (0.4 MB) if missing
    python sources/ro_census.py            normalise from data/raw/ro/

-> data/normalized/ro.csv (levels `country`, `judet`, `uat`; alternatives, never summed)

Table 2.3 "Populaţia rezidentă după limba maternă": sheet `Tab.2.3.1` by macroregion, development
region and judeţ; sheet `Tab 2.3.2` by judeţ, municipiu, oraş and comună (3,181 UATs). 22 named
mother tongues, `Alta limba materna` and `Informatie nedisponibila` (2,502,378 people, 13.1%: the
2021 census was built largely from administrative registers, which carry no language).

THE LAYOUT IS RELIGIONDOTS' RELIGION TABLE (2.4) EXACTLY, so this follows
religiondots/sources/ro.py, whose three traps are all here:
  1. `*` is an in-band suppression marker and `-` a true zero, in the numeric columns. Each cell
     is classified; anything else stops the run. Suppressed cells are dropped, never guessed; the
     check prints what they cost.
  2. No codes: a UAT is identified by (judeţ, name), written `JUDEŢ|NAME`, the same key
     religiondots' ro_uat_lookup.csv resolves to SIRUTA (countries/ro.py reads it, read-only).
  3. County headers are bare names like communes. A row is a county header when its name is a
     county's AND its total equals that county's total on Tab.2.3.1 (Călăraşi and Satu Mare are
     also commune names elsewhere, and those communes sort first); Bucharest appears twice running.

CHECKS: units and population per level against INS's 19,053,815; categories never exceed a unit's
total; the shortfall at each level is exactly the suppressed cells; and the UAT rows summed by
judeţ against Tab.2.3.1's own judeţ rows, per language (a second table of the same census: its
judeţ cells are suppressed far less often, so the UAT sums may fall short but never exceed them).
"""

import csv
import os
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ro")
OUT = os.path.join(ROOT, "data", "normalized", "ro.csv")

SOURCE_ID = "ro_rpl_2021_t2.3"
YEAR = 2021

XLSX_NAME = "Tabel-2.03.1-si-Tabel-2.03.2.xlsx"
XLSX_URL = ("https://www.recensamantromania.ro/wp-content/uploads/2023/06/"
            "Tabel-2.03.1-si-Tabel-2.03.2.xlsx")
MIN_BYTES = 300_000

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

NATIONAL = 19_053_815        # INS, resident population, 1 December 2021

SHEET_JUDET = "Tab.2.3.1"
SHEET_UAT = "Tab 2.3.2"
HEADER_ROW = 4
TOTAL_COL = 1
FIRST_CAT, LAST_CAT = 2, 24      # inclusive: 22 languages, other, information not available

TOTAL_LABEL = "POPULATIA REZIDENTA TOTAL"
SUPPRESSED = "*"
TRUE_ZERO = "-"

SUPPRESSED_JUDET = set()     # (judeţ, category) cells `*` on Tab.2.3.1, filled by read()
HIDDEN = {}                  # level -> [(geo_id, category)] for every `*` cell, filled by read()

NOT_COUNTIES = {
    "romania", "macroregiunea 1", "macroregiunea 2", "macroregiunea 3", "macroregiunea 4",
    "nord-vest", "centru", "nord-est", "sud-est", "sud-muntenia", "bucuresti-ilfov",
    "sud-vest oltenia", "vest",
}


def fold(s):
    """ş/ș and ţ/ț are different codepoints (cedilla, comma-below); fold both to ASCII."""
    s = " ".join(str(s).split())
    s = s.replace("ş", "s").replace("Ş", "S").replace("ţ", "t").replace("Ţ", "T")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, XLSX_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
        print("already have", dest)
        return
    print("downloading", XLSX_URL)
    r = requests.get(XLSX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    if os.path.getsize(tmp) < MIN_BYTES:
        raise SystemExit(f"{tmp} is {os.path.getsize(tmp):,} bytes, expected >= {MIN_BYTES:,}")
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def _txt(v):
    return "" if v is None else " ".join(str(v).split())


def _value(cell):
    """(count, kind), kind one of n / suppressed / zero / blank."""
    if cell is None:
        return None, "blank"
    if isinstance(cell, (int, float)):
        return int(cell), "n"
    s = str(cell).strip()
    # `**` occurs once, Oraş Ţăndărei's Polish cell, with no footnote anywhere in the workbook;
    # read as the `*` it sits among (a typo), so it is dropped like any suppressed cell.
    if s in (SUPPRESSED, "**"):
        return None, "suppressed"
    if s in (TRUE_ZERO, "–", "—"):
        return 0, "zero"
    if s == "":
        return None, "blank"
    raise SystemExit(f"unexpected cell value {cell!r} in a count column: a new sentinel?")


def read():
    import openpyxl

    path = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(path) or os.path.getsize(path) < MIN_BYTES:
        raise SystemExit(f"missing or truncated {path}; run with --fetch")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for s in (SHEET_JUDET, SHEET_UAT):
        if s not in wb.sheetnames:
            raise SystemExit(f"missing sheet {s!r}; got {wb.sheetnames}")

    cats = {}
    for s in (SHEET_JUDET, SHEET_UAT):
        header = list(wb[s].iter_rows(min_row=HEADER_ROW, max_row=HEADER_ROW,
                                      values_only=True))[0]
        got = {c: _txt(header[c]) for c in range(FIRST_CAT, LAST_CAT + 1)}
        if not all(got.values()):
            raise SystemExit(f"{s}: blank category header")
        if cats and got != cats:
            raise SystemExit("the two sheets' category headers differ")
        cats = got
    if cats[LAST_CAT] != "Informatie nedisponibila" or cats[FIRST_CAT] != "Româna":
        raise SystemExit(f"category columns moved: {cats}")

    counties = {}
    for r in wb[SHEET_JUDET].iter_rows(min_row=7, values_only=True):
        name = _txt(r[0])
        if name and isinstance(r[TOTAL_COL], (int, float)) and fold(name) not in NOT_COUNTIES:
            counties[fold(name)] = (name, int(r[TOTAL_COL]))
    if len(counties) != 42:
        raise SystemExit(f"found {len(counties)} counties, expected 42")

    rows, stats = [], {}

    def emit_unit(gid, level, gname, r, note):
        tot, kind = _value(r[TOTAL_COL])
        if kind != "n":
            raise SystemExit(f"{gname}: total is not a number ({kind})")
        rows.append(dict(geo_id=gid, geo_level=level, geo_name=gname,
                         source_category=TOTAL_LABEL, count=tot, tier="measured", year=YEAR,
                         source_id=SOURCE_ID, note=note + "; universe total"))
        for c in range(FIRST_CAT, LAST_CAT + 1):
            n, kind = _value(r[c])
            stats[(level, kind)] = stats.get((level, kind), 0) + 1
            if kind == "suppressed":
                HIDDEN.setdefault(level, []).append((gid, cats[c]))
                if level == "judet":
                    SUPPRESSED_JUDET.add((gid, cats[c]))
            if kind == "n" and n > 0:
                rows.append(dict(geo_id=gid, geo_level=level, geo_name=gname,
                                 source_category=cats[c], count=n, tier="measured",
                                 year=YEAR, source_id=SOURCE_ID, note=note))

    for r in wb[SHEET_JUDET].iter_rows(min_row=7, values_only=True):
        name = _txt(r[0])
        if not name or not isinstance(r[TOTAL_COL], (int, float)):
            continue
        f = fold(name)
        if f == "romania":
            emit_unit("RO", "country", "România", r, f"INS RPL 2021 {SHEET_JUDET}")
        elif f in counties:
            emit_unit(name, "judet", name, r, f"INS RPL 2021 {SHEET_JUDET}")

    seen, current, n_uat = set(), None, 0
    for r in wb[SHEET_UAT].iter_rows(min_row=7, values_only=True):
        name = _txt(r[0])
        if not name or not isinstance(r[TOTAL_COL], (int, float)):
            continue
        f = fold(name)
        if f == "romania":
            continue
        if f in counties and f not in seen and int(r[TOTAL_COL]) == counties[f][1]:
            seen.add(f)
            current = counties[f][0]
            continue
        if current is None:
            raise SystemExit(f"UAT row {name!r} before any county header")
        n_uat += 1
        emit_unit(f"{current}|{name}", "uat", name, r, f"INS RPL 2021 {SHEET_UAT}")
    if len(seen) != 42:
        raise SystemExit(f"only {len(seen)} of 42 counties found on {SHEET_UAT}")
    print(f"  parsed {n_uat:,} UAT rows in {len(seen)} counties")
    for lv in ("uat", "judet", "country"):
        print(f"  {lv:<8} cells: " + ", ".join(
            f"{stats.get((lv, k), 0):,} {k}" for k in ("n", "suppressed", "zero")))
    return rows


def _ipf(cells, row_margin, col_margin, rounds=2000):
    """Fill hidden cells so each row and each column sums to its margin.

    cells: [(row, col)]; margins: what the hidden cells of a row (a unit's total less its
    published cells) and of a column (the parent's cell less the published children) must hold.
    Starts every hidden cell at 1 (a `*` hides at least one person) and scales rows and columns
    in turn: the even spread that meets both totals. Returns ({(row, col): value}, row miss,
    column miss).
    """
    v = {c: 1.0 for c in cells}
    by_r, by_c = {}, {}
    for c in cells:
        by_r.setdefault(c[0], []).append(c)
        by_c.setdefault(c[1], []).append(c)
    for _ in range(rounds):
        for grp, margin in ((by_r, row_margin), (by_c, col_margin)):
            for k, cs in grp.items():
                tot = sum(v[c] for c in cs)
                want = max(margin.get(k, 0.0), 0.0)
                for c in cs:
                    v[c] = v[c] * want / tot if tot > 0 else want / len(cs)
    # the column step runs last, so columns are met exactly; a row can miss where the hidden
    # cells' pattern cannot meet both (reported as the summed absolute miss over rows)
    err_r = sum(abs(sum(v[c] for c in cs) - max(row_margin.get(k, 0), 0))
                for k, cs in by_r.items())
    err_c = max((abs(sum(v[c] for c in cs) - max(col_margin.get(k, 0), 0))
                 for k, cs in by_c.items()), default=0.0)
    return v, err_r, err_c


def impute(rows):
    """Estimate every `*` cell from what the margins leave, judeţ first, then UAT.

    INS hides cells of 1-2 people (the smallest published value at either level is 3) and then
    hides more cells, of any size, so that the first cannot be worked out from the row total:
    6,096 hidden UAT cells hold 24,398 people, four a cell. Both margins are known exactly: a
    unit's total less its published cells, and the parent's cell for that language less its
    published children. A judeţ's hidden cell is estimated first from the national row; that
    estimate then stands as the parent cell for the judeţ's UATs. Every estimated row is tier
    `derived`; the national and judeţ totals per language are reproduced (asserted in main).
    """
    tot = {(r["geo_level"], r["geo_id"]): r["count"] for r in rows
           if r["source_category"] == TOTAL_LABEL}
    pub_unit, pub_cell = {}, {}
    for r in rows:
        if r["source_category"] == TOTAL_LABEL:
            continue
        k = (r["geo_level"], r["geo_id"])
        pub_unit[k] = pub_unit.get(k, 0) + r["count"]
        pub_cell[(r["geo_level"], r["geo_id"], r["source_category"])] = r["count"]
    note = "`*` cell, estimated from the margins (sources/ro_census.py impute)"

    out = []
    # ---- judeţ: columns are languages, the column margin the national cell less its judeţe
    cells = HIDDEN.get("judet", [])
    row_m = {g: tot[("judet", g)] - pub_unit.get(("judet", g), 0) for g, _ in cells}
    pub_j = {}
    for (lv, _, c), n in pub_cell.items():
        if lv == "judet":
            pub_j[c] = pub_j.get(c, 0) + n
    col_m = {c: pub_cell[("country", "RO", c)] - pub_j.get(c, 0) for _, c in cells}
    v_j, er, ec = _ipf(cells, row_m, col_m)
    print(f"\n  judeţ: {len(cells)} hidden cells, {sum(v_j.values()):,.1f} people; "
          f"largest miss {er:.3g} on a row, {ec:.3g} on a language")
    for (g, c), n in v_j.items():
        out.append(dict(geo_id=g, geo_level="judet", geo_name=g, source_category=c, count=n,
                        tier="derived", year=YEAR, source_id=SOURCE_ID, note=note))

    # ---- UAT, one judeţ at a time: column margin is the judeţ cell (published or estimated)
    judet_cell = {(g, c): n for (lv, g, c), n in pub_cell.items() if lv == "judet"}
    judet_cell.update(v_j)
    uat_pub_by = {}
    for (lv, g, c), n in pub_cell.items():
        if lv == "uat":
            k = (g.split("|")[0], c)
            uat_pub_by[k] = uat_pub_by.get(k, 0) + n
    by_j = {}
    for g, c in HIDDEN.get("uat", []):
        by_j.setdefault(g.split("|")[0], []).append((g, c))
    miss_r, worst_c, total = {}, 0.0, 0.0
    for j, cs in by_j.items():
        row_m = {g: tot[("uat", g)] - pub_unit.get(("uat", g), 0) for g, _ in cs}
        col_m = {c: judet_cell.get((j, c), 0) - uat_pub_by.get((j, c), 0) for _, c in cs}
        v, er, ec = _ipf(cs, row_m, col_m)
        miss_r[j], worst_c = er, max(worst_c, ec)
        for (g, c), n in v.items():
            total += n
            out.append(dict(geo_id=g, geo_level="uat", geo_name=g.split("|")[1],
                            source_category=c, count=n, tier="derived", year=YEAR,
                            source_id=SOURCE_ID, note=note))
    print(f"  UAT: {len(HIDDEN.get('uat', [])):,} hidden cells, {total:,.1f} people; largest "
          f"miss {worst_c:.3g} on a judeţ language")
    worst = sorted(miss_r.items(), key=lambda kv: -kv[1])[:4]
    print(f"  UAT rows missed by {sum(miss_r.values()):,.1f} people in all (people moved "
          "between UATs of one judeţ, never between judeţe): "
          + "; ".join(f"{j} {m:,.1f}" for j, m in worst))
    if max(worst_c, ec, er) > 0.5 or sum(miss_r.values()) > 1000:
        raise SystemExit("hidden-cell estimate does not meet its margins")
    return out


def check(rows):
    ok = True
    levels, totals, sums = {}, {}, {}
    for r in rows:
        k = (r["geo_level"], r["geo_id"])
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
        if r["source_category"] == TOTAL_LABEL:
            totals[k] = r["count"]
        else:
            sums[k] = sums.get(k, 0) + r["count"]

    expected = {"country": 1, "judet": 42, "uat": 3181}
    print("\n  units and population per level (alternatives, never summed):")
    for lv in ("uat", "judet", "country"):
        n = len(levels.get(lv, ()))
        pop = sum(v for (l, _), v in totals.items() if l == lv)
        good = n == expected[lv] and pop == NATIONAL
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {lv:<8} {n:>6,} units {pop:>12,}")

    over = [k for k in sums if sums[k] > totals.get(k, 0)]
    ok &= not over
    print(f"  {'OK ' if not over else 'BAD'} categories never exceed the unit total "
          f"({len(over)} violations) {over[:3]}")
    for lv in ("uat", "judet", "country"):
        got = sum(v for (l, _), v in sums.items() if l == lv)
        tot = sum(v for (l, _), v in totals.items() if l == lv)
        print(f"    {lv:<8} categories sum to {got:>12,} of {tot:,}: {tot - got:,} "
              "people in suppressed cells")
    nat_short = totals[("country", "RO")] - sums[("country", "RO")]
    ok &= nat_short == 0
    print(f"  {'OK ' if nat_short == 0 else 'BAD'} national row: categories = total exactly")

    # second table: UAT rows summed by judeţ, per language, against Tab.2.3.1's judeţ cells
    uat_by = {}
    jud = {}
    for r in rows:
        if r["source_category"] == TOTAL_LABEL:
            continue
        if r["geo_level"] == "uat":
            key = (r["geo_id"].split("|")[0], r["source_category"])
            uat_by[key] = uat_by.get(key, 0) + r["count"]
        elif r["geo_level"] == "judet":
            jud[(r["geo_id"], r["source_category"])] = r["count"]
    # a judeţ cell can itself be suppressed (Bistriţa-Năsăud's Russian); then there is nothing
    # to compare the UAT sum with, and it is counted apart
    exceed = [(k, v, jud.get(k, 0)) for k, v in uat_by.items()
              if v > jud.get(k, 0) and k not in SUPPRESSED_JUDET]
    n_supp = sum(1 for k in uat_by if k in SUPPRESSED_JUDET)
    print(f"  {'OK ' if not exceed else 'BAD'} UAT sums never exceed the judeţ table's cell "
          f"({len(exceed)} exceed; {n_supp} judeţ cells suppressed, not compared)")
    for k, v, j in exceed[:8]:
        print(f"      {k}: UAT sum {v:,} vs judeţ {j:,}")
    ok &= not exceed
    short = sorted(((jud[k] - uat_by.get(k, 0), k) for k in jud), reverse=True)[:5]
    print("    largest judeţ cells the UAT rows fall short of (suppressed UAT cells): "
          + "; ".join(f"{k[0]} {k[1]} {d:,}" for d, k in short))
    tot_j = sum(jud.values())
    tot_u = sum(uat_by.values())
    print(f"    all languages: UAT sums {tot_u:,} of the judeţ cells' {tot_j:,} "
          f"({100 * tot_u / tot_j:.3f}%)")

    print("\n  national, by category:")
    for r in sorted((r for r in rows if r["geo_level"] == "country"
                     and r["source_category"] != TOTAL_LABEL), key=lambda r: -r["count"]):
        print(f"    {r['count']:>11,}  {100 * r['count'] / NATIONAL:6.3f}%  {r['source_category']}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = read()
    check(rows)
    rows += impute(rows)
    # with the estimates every level holds its whole population, language by language
    nat = {r["source_category"]: r["count"] for r in rows
           if r["geo_level"] == "country" and r["source_category"] != TOTAL_LABEL}
    for lv in ("uat", "judet"):
        by = {}
        for r in rows:
            if r["geo_level"] == lv and r["source_category"] != TOTAL_LABEL:
                by[r["source_category"]] = by.get(r["source_category"], 0) + r["count"]
        miss = max(abs(by.get(c, 0) - n) for c, n in nat.items())
        print(f"  {'OK ' if miss < 1 else 'BAD'} {lv} with estimates: every language within "
              f"{miss:.3g} of the national row")
        if miss >= 1:
            raise SystemExit("estimates do not reproduce the national row")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}: {len(rows):,} rows")


if __name__ == "__main__":
    main()
