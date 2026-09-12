"""South Korea — KOSIS, 인구총조사 2015, religion by si/gun/gu.

Reads data/raw/kr/kosis_DT_1PM1502_2015.csv and writes data/normalized/kr.csv.

KOSIS table **`DT_1PM1502`** — *성별/연령별/종교별 인구-시군구* — org 101, 국가데이터처
(formerly KOSTAT). 12 categories, 49,052,389 people, **229 drawn sigungu**.

**THE 2015 CENSUS IS THE LAST ONE THAT ASKED.** Religion was dropped afterwards, so this is
not a stale vintage that a later release will replace — it is the final measurement, and any
Korean religion map from now on is this table or nothing.

THERE IS NO `--fetch` AND THERE CANNOT BE. KOSIS's metadata endpoint is open — the table id,
its 327 geography items and its 12 categories were all read off `statHtmlContent.do` without
a login — but every DATA endpoint (`html.do`, `downNormal.do`, `downGrid.do`, `downLarge.do`)
answers HTTP 200 with an HTML `alert()` reading *비정상적인 서비스 이용으로 접근이 차단되었습니다*
(access blocked, abnormal service use). That is bot protection; §12 says a stop sign rather
than a puzzle. **Anita downloaded this file through a browser on 2026-09-05** and it is in
`data/raw/kr/`. sources.md §11g has the click path if it ever needs doing again.

TWO STRUCTURAL TRAPS, AND NEITHER HAS A MARKER OF ANY KIND.

1. **45 rows are not places.** `동부`, `읍부` and `면부` (urban / town / rural) appear once
   nationally and again inside most provinces. They are a CROSS-CUTTING partition of the same
   people, so treating them as units double-counts the country. They are dropped — and
   asserted to sum to their province exactly, because that is what proves they are the thing
   this docstring says they are rather than places being discarded by mistake.
2. **12 cities appear twice, as themselves and as their own gu.** Suwon, Seongnam, Anyang,
   Bucheon, Ansan, Goyang, Yongin, Cheongju, Cheonan, Jeonju, Pohang and Changwon are 특정시
   subdivided into 구, and KOSIS lists both tiers in one column with **the same indent, the
   same column and no flag**. Summing the level as delivered overstates Korea by 5,997,676 in
   Gyeonggi alone. This is `spec` §12's Serbia case exactly — an extra level INSIDE the drawn
   tier — and it is found the same way: **a parent's children are the consecutive rows whose
   totals sum to it exactly**, verified afterwards across all twelve categories, not just the
   total.
   **Which of the two tiers to draw is decided by the boundaries, not by the source.** KOSIS's
   own level 2 is the 252 — plain units plus the 35 gu — and every boundary set available
   carries the CITY: geoBoundaries KOR ADM2 is 228 units with Suwon, Changwon and the rest
   whole, and has no 일반구 at all. So the drawn tier is the **229** (plain units plus the 12
   cities) and the 35 gu are emitted at their own `gu` level, undrawn, so nothing published is
   thrown away (§2.4). Both tiers are asserted to partition every province independently —
   two complete partitions of the same people, and checking only one would not notice the
   other going wrong.

THE FILE HAS NO CODES, ONLY NAMES, AND KOREAN DISTRICT NAMES COLLIDE HARD. 중구 appears six
times, 동구 six, 남구 six, 서구 five, 북구 five — a national name join would hand one city's
district to another and every total would still reconcile (§12's Ghana `TMA` failure). Names
ARE unique inside a province, and row order gives the province, so every row carries its
parent and `sources/kr_geo.py` joins within it. Requesting codes in the download would have
avoided this; the option was asked for and did not come through, and it is not worth a second
manual export because the within-parent join is checkable.

**THE FIGURES ARE A 20% SAMPLE GROSSED UP.** 2015 was a register-based census (등록센서스) and
religion was carried on the sample survey, not the register, so every cell here is an
estimate with sampling error attached. It matters most exactly where this table is most
interesting: Daejonggyo is 3,101 people nationally, so its sigungu cells are a handful of
sampled households each. Treat small categories at small units as indicative.

**AND THE UNIVERSE IS 49,052,389 AGAINST A CENSUS POPULATION OF 51,069,375**, a gap of
2,016,986 (3.95%) that this file does not explain and neither does the table. Not silently
absorbed: see `sources/kr.md` §5. The gap is reported, not filled (§3.5).

Usage:
    python sources/kr.py            normalise from data/raw/kr/
"""

import csv
import io
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kr")
SRC = os.path.join(RAW, "kosis_DT_1PM1502_2015.csv")
OUT = os.path.join(ROOT, "data", "normalized", "kr.csv")

SOURCE_ID = "kr_kosis_DT_1PM1502_2015"
YEAR = 2015
BASIS = "self_id"
ENCODING = "cp949"          # KOSIS serves CSV as cp949, never UTF-8

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

NATIONAL_NAME = "전국"
NATIONAL = 49_052_389
CENSUS_POPULATION = 51_069_375      # 2015 census total, for the gap report

# KOSIS's own level-1 geography items, in its order. Taken from the table's metadata rather
# than inferred from the file, so the tier split is the source's and not a guess.
PROVINCES = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
             "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도",
             "충청북도", "충청남도", "전라북도", "전라남도", "경상북도",
             "경상남도", "제주특별자치도"]

# Urban / town / rural. A cross-cutting partition of a province, not places.
URBAN_RURAL = ["동부", "읍부", "면부"]

# The 12 item columns, in file order. The first two are universes.
CATEGORIES = ["계", "종교있음-계", "불교", "기독교(개신교)", "기독교(천주교)", "원불교",
              "유교", "천도교", "대순진리회", "대종교", "기타", "종교없음-계"]
TOTAL_CAT = "계"
HAS_RELIGION = "종교있음-계"
NO_RELIGION = "종교없음-계"
# The nine named religions between the two universe columns.
RELIGIONS = CATEGORIES[2:11]

EXPECTED_ROWS = 327
EXPECTED_SIGUNGU = 229          # 217 plain units + the 12 cities themselves
EXPECTED_GU = 35                # their general gu, emitted at their own level, not drawn
EXPECTED_NESTED = 12


def num(v):
    """A cell -> int. Blank means zero here, and check() proves it rather than assuming."""
    v = (v or "").strip().replace(",", "")
    if v == "":
        return 0
    if not v.lstrip("-").isdigit():
        raise SystemExit(f"unrecognised cell {v!r} -- KOSIS uses e/p/-/.../X as sentinels "
                         "and none has been seen in this table; classify it before running")
    return int(v)


def read():
    if not os.path.exists(SRC):
        raise SystemExit(
            f"missing {SRC}\n"
            "This file cannot be fetched -- KOSIS bot-blocks its data endpoints. Download it\n"
            "through a browser from\n"
            "  https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1PM1502\n"
            "with 행정구역별(시군구) set to its second level (all 306) and all 12 items,\n"
            "then save it here. sources.md §11g has the detail.")

    with open(SRC, "rb") as fh:
        txt = fh.read().decode(ENCODING)
    rows = list(csv.reader(io.StringIO(txt)))

    header = [c.strip() for c in rows[1][3:]]
    if header != CATEGORIES:
        raise SystemExit(f"category columns changed.\n  file: {header}\n  want: {CATEGORIES}")

    data = rows[2:]
    if len(data) != EXPECTED_ROWS:
        raise SystemExit(f"{len(data)} data rows, expected {EXPECTED_ROWS}")

    names = [r[0].strip() for r in data]
    vals = [[num(c) for c in r[3:3 + len(CATEGORIES)]] for r in data]

    if names[0] != NATIONAL_NAME:
        raise SystemExit(f"first row is {names[0]!r}, expected {NATIONAL_NAME!r}")

    pidx = [i for i, n in enumerate(names) if n in PROVINCES]
    if len(pidx) != len(PROVINCES):
        missing = set(PROVINCES) - {names[i] for i in pidx}
        raise SystemExit(f"{len(pidx)} province rows, expected {len(PROVINCES)}; "
                         f"missing {missing}")

    # ---- split each province block into its places, its urban/rural rows, and the
    #      nested si parents hiding among the places -------------------------------------
    blocks = []
    for k, i in enumerate(pidx):
        end = pidx[k + 1] if k + 1 < len(pidx) else len(data)
        kids = [j for j in range(i + 1, end) if names[j] not in URBAN_RURAL]
        subs = [j for j in range(i + 1, end) if names[j] in URBAN_RURAL]

        # Serbia's test: a parent's children are the consecutive rows summing to it exactly.
        nested, pos = {}, 0
        while pos < len(kids):
            j = kids[pos]
            acc, run = 0, []
            for j2 in kids[pos + 1:]:
                acc += vals[j2][0]
                run.append(j2)
                if acc == vals[j][0]:
                    nested[j] = list(run)
                    break
                if acc > vals[j][0]:
                    break
            pos += 1 + (len(nested[j]) if j in nested else 0)

        # Two complete tiers over the same people, and they must never be mixed:
        #   `fine`    the 252 KOSIS publishes as its level 2 — plain units plus the 35 gu,
        #             with the 12 city parents removed. Sums to the province.
        #   `sigungu` the 229 that boundary files exist for — plain units plus the 12
        #             cities themselves, with their 35 gu removed. Also sums to the province.
        gu = {j for run in nested.values() for j in run}
        fine = [j for j in kids if j not in nested]
        sigungu = [j for j in kids if j not in gu]
        blocks.append(dict(prov=i, kids=kids, subs=subs, nested=nested,
                           gu=sorted(gu), fine=fine, sigungu=sigungu))

    return data, names, vals, pidx, blocks


def build(names, vals, pidx, blocks):
    rows = []

    def emit(gid, level, name, i, note):
        for cat, n in zip(CATEGORIES, vals[i]):
            nt = note
            if cat in (TOTAL_CAT, HAS_RELIGION):
                nt += "; universe row, not a religion category"
            rows.append({"geo_id": gid, "geo_level": level, "geo_name": name,
                         "source_category": cat, "count": n, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": nt})

    emit("0", "country", NATIONAL_NAME, 0, "level=country")
    for p, b in enumerate(blocks, start=1):
        i = b["prov"]
        emit(f"{p:02d}", "province", names[i], i, "level=province")

        # THE DRAWN TIER IS `sigungu`, AND IT IS THE 12 NESTED CITIES THEMSELVES, NOT THEIR
        # GU. KOSIS publishes both; every boundary set available publishes only the city
        # (geoBoundaries KOR ADM2 is 228 units and has Suwon, Changwon and the rest whole),
        # so drawing the gu would mean 35 units with no polygon. The gu are emitted anyway
        # at their own level so nothing published is discarded — §2.4's deferred matching,
        # and the upgrade path if a 일반구 boundary file is ever found. Indonesia's
        # regency/kecamatan split is the same shape.
        for c, j in enumerate(b["sigungu"], start=1):
            # No code in the source, so the id is positional: province ordinal + child
            # ordinal in KOSIS's own order. The NAME plus the parent province is what
            # sources/kr_geo.py actually joins on; this is a stable key, not a code.
            emit(f"{p:02d}{c:03d}", "sigungu", names[j], j,
                 f"level=sigungu; parent={names[i]}")
        for parent, run in b["nested"].items():
            for g, j in enumerate(run, start=1):
                emit(f"{p:02d}{parent:03d}g{g:02d}", "gu", names[j], j,
                     f"level=gu; parent={names[parent]}; province={names[i]}; "
                     "record only, not drawn -- no boundary set carries these")
    return rows


def check(names, vals, pidx, blocks):
    ok = True
    nat = vals[0]

    print(f"  national universe {nat[0]:,} (expected {NATIONAL:,})")
    good = nat[0] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total matches the published figure")

    gap = CENSUS_POPULATION - nat[0]
    print(f"      the 2015 census counted {CENSUS_POPULATION:,}; this table's universe is "
          f"{gap:,} ({100.0 * gap / CENSUS_POPULATION:.2f}%) smaller.\n"
          f"      Reported, not filled (§3.5) -- see sources/kr.md §5.")

    # 1. the two universe identities, which also prove blank == 0
    bad = []
    for i in range(len(names)):
        if sum(vals[i][2:11]) != vals[i][1]:
            bad.append(("religions vs 종교있음-계", names[i]))
        if vals[i][1] + vals[i][11] != vals[i][0]:
            bad.append(("종교있음+종교없음 vs 계", names[i]))
    ok &= not bad
    print(f"\n  {'OK ' if not bad else 'BAD'} both universe identities hold on all "
          f"{len(names)} rows ({len(bad)} failures)")
    print("      this is also what proves an EMPTY cell means zero: the identities close "
          "with blanks read as 0")
    for what, nm in bad[:5]:
        print(f"        {nm}: {what}")

    # 2. provinces sum to the nation, per category
    bad = []
    for c in range(len(CATEGORIES)):
        s = sum(vals[b["prov"]][c] for b in blocks)
        if s != nat[c]:
            bad.append((CATEGORIES[c], s, nat[c]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 17 provinces sum to 전국 on all "
          f"{len(CATEGORIES)} categories ({len(bad)} failures)")
    for c, s, n in bad[:5]:
        print(f"        {c}: {s:,} vs {n:,}")

    # 3. BOTH tiers sum to their province, per category. Two complete partitions of the
    #    same people; checking only one would not notice the other being wrong.
    for tier in ("sigungu", "fine"):
        bad = []
        for b in blocks:
            for c in range(len(CATEGORIES)):
                s = sum(vals[j][c] for j in b[tier])
                if s != vals[b["prov"]][c]:
                    bad.append((names[b["prov"]], CATEGORIES[c], s, vals[b["prov"]][c]))
        ok &= not bad
        label = ("the 229 drawn sigungu" if tier == "sigungu"
                 else "KOSIS's own 252 level-2 units")
        print(f"  {'OK ' if not bad else 'BAD'} {label} sum to their province on all "
              f"{len(CATEGORIES)} categories ({len(bad)} failures)")
        for nm, c, s, n in bad[:5]:
            print(f"        {nm} / {c}: {s:,} vs {n:,}")

    # 4. THE NESTED PARENTS. Detected on the total; verified on every category, which is
    #    what turns a plausible grouping into a proof (§12, Serbia).
    nested_total = sum(len(b["nested"]) for b in blocks)
    good = nested_total == EXPECTED_NESTED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {nested_total} nested si-with-gu parents found "
          f"(expected {EXPECTED_NESTED})")
    bad = []
    for b in blocks:
        for p, run in b["nested"].items():
            for c in range(len(CATEGORIES)):
                if sum(vals[j][c] for j in run) != vals[p][c]:
                    bad.append((names[p], CATEGORIES[c]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} each nested parent equals its gu on all "
          f"{len(CATEGORIES)} categories, not just the total ({len(bad)} failures)")
    for nm, c in bad[:5]:
        print(f"        {nm} / {c}")

    # 5. urban/rural rows are a partition of their province, which is why they are dropped
    bad = []
    for b in blocks:
        if not b["subs"]:
            continue
        for c in range(len(CATEGORIES)):
            s = sum(vals[j][c] for j in b["subs"])
            if s != vals[b["prov"]][c]:
                bad.append((names[b["prov"]], CATEGORIES[c], s, vals[b["prov"]][c]))
    ok &= not bad
    n_subs = sum(len(b["subs"]) for b in blocks)
    print(f"  {'OK ' if not bad else 'BAD'} the {n_subs} 동부/읍부/면부 rows partition their "
          f"province on all {len(CATEGORIES)} categories, so they are a universe and not "
          f"places ({len(bad)} failures)")

    # 6. the drawn tier
    drawn = sum(len(b["sigungu"]) for b in blocks)
    good = drawn == EXPECTED_SIGUNGU
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {drawn} sigungu drawn (expected "
          f"{EXPECTED_SIGUNGU})")

    # 7. names must be unique INSIDE a province, since that is the join key
    bad = []
    for b in blocks:
        seen = [names[j] for j in b["sigungu"]]
        dup = {n for n in seen if seen.count(n) > 1}
        if dup:
            bad.append((names[b["prov"]], sorted(dup)))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} district names are unique within every province "
          f"-- the key sources/kr_geo.py joins on ({len(bad)} provinces with duplicates)")
    for nm, d in bad[:5]:
        print(f"        {nm}: {d}")

    allnames = [names[j] for b in blocks for j in b["sigungu"]]
    coll = sorted({n for n in allnames if allnames.count(n) > 1},
                  key=lambda n: -allnames.count(n))
    print(f"      for contrast, {len(coll)} names are NOT unique nationally: "
          + ", ".join(f"{n} x{allnames.count(n)}" for n in coll[:6]))
    print("      which is why the join is inside the parent and never across the country.")

    print(f"\n  categories, national:")
    for c, cat in enumerate(CATEGORIES):
        mark = "  <- universe" if cat in (TOTAL_CAT, HAS_RELIGION) else ""
        print(f"    {nat[c]:>12,}  {100.0 * nat[c] / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    data, names, vals, pidx, blocks = read()
    check(names, vals, pidx, blocks)
    rows = build(names, vals, pidx, blocks)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
