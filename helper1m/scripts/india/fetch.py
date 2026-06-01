"""Fetch India district & state population for helper1m.

Source: IIPS report "Projection of District-Level Annual Population by
Quinquennial Age-Group and Sex from 2012 to 2031 in India" (Dhar 2022),
PDF at maps/data/asia1m/FULL_REPORT_WITH_FINAL_TABLES.pdf.

  Table 4 (printed p.23-27) - Census 2011 actuals, 640 districts in the
      35-state census layout. Its S.N. column is the all-India census
      district code (== SHRUG pc11_d_id).
  Table 8 (printed p.31-43) - projected total population by sex for
      2011/2016/2021/2026/2031, per district.

Quirks handled:
  * Table 8's S.N. is a re-sequenced row counter, not the census code.
  * Table 8 uses a 37-state layout - it splits Telangana out of Andhra
    Pradesh and Ladakh out of J&K - so state numbers diverge from Table 4
    after state 27, and within-state codes are renumbered.
So we join Table 8 -> Table 4 on (census state, district name) to recover
each district's real census code (pc11_d_id).

Writes helper1m/data/india/population.csv  (columns: code, level, year, pop):
  level 1 = state       (code = 2-digit pc11_s_id)
  level 2 = district    (code = 3-digit pc11_d_id)
  level 3 = subdistrict - each subdistrict's 2011 population (SHRUG PC11
            subdistrict weights) scaled by its district's projected growth.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

HELPER = Path(__file__).resolve().parents[2]
PDF = HELPER.parent / "data" / "asia1m" / "FULL_REPORT_WITH_FINAL_TABLES.pdf"
OUT = HELPER / "data" / "india" / "population.csv"
# SHRUG PC11 subdistrict weights - 2011 census population + area per subdistrict.
SUBDIST_CSV = (HELPER.parent / "data" / "asia1m" / "india" /
               "shrug-subdist-wt" / "subdist_pc11_pop_area_key.csv")
# SHRUG subdistrict polygons - the keep-set for level 3 (build_country.py joins
# on pc11_d_id+pc11_sd_id and silently drops weights rows with no polygon, e.g.
# the census 99999 "unclassified" / big-city residuals, ~48M people nationally).
SUBDIST_SHP = HELPER.parent / "data" / "asia1m" / "india" / "subdistrict.shp"

# 0-indexed PDF page ranges (printed page number = index - 7).
TABLE4_PAGES = range(30, 35)   # printed 23-27
TABLE8_PAGES = range(38, 51)   # printed 31-43
YEARS = [2011, 2016, 2021, 2026, 2031]

NN_RE = re.compile(r"^\((\d{1,3})\)$")
STATE_RE = re.compile(r"^(\d{1,2})\s*\S{0,3}\s*([A-Za-z][A-Za-z&.\s]+?)\s*$")
COMMA_NUM = re.compile(r"^\d{1,3}(?:,\d{3})+$")   # Table 8 values: "1,071,637"
PLAIN_NUM = re.compile(r"^\d{4,}$")               # Table 4 values: "1071637"

# Table 8 state names that need mapping back to the 2011-census state.
T8_TO_CENSUS = {
    "TELANGANA": "ANDHRA PRADESH",
    "LADAKH": "JAMMU & KASHMIR",
    "DELHI": "NCT OF DELHI",
}


def key(name):
    """Normalize a district name for joining (drop case, spaces, punctuation)."""
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def rows_of(words, ytol=2.5):
    """Group PyMuPDF word tuples into visual rows, each sorted left-to-right."""
    rows, cur, cy = [], [], None
    for w in sorted(words, key=lambda w: w[1]):
        if cy is not None and abs(w[1] - cy) > ytol:
            rows.append(sorted(cur, key=lambda x: x[0]))
            cur = []
        cur.append(w)
        cy = w[1]
    if cur:
        rows.append(sorted(cur, key=lambda x: x[0]))
    return rows


def name_from(toks, n_values):
    """District name: tokens between the S.N. and the n_values trailing
    population numbers, with any (NN) census-code token dropped. The PDF
    occasionally omits the (NN) - e.g. Sikkim West District in Table 4."""
    span = toks[1:len(toks) - n_values]
    return " ".join(t for t in span if not NN_RE.match(t))


def parse_table4(doc):
    """Returns ([record], [(state_num, state_name)]) from Table 4."""
    assert "Table 4" in doc[TABLE4_PAGES[0]].get_text(), "Table 4 not at expected page"
    records, headers = [], []
    for pidx in TABLE4_PAGES:
        words = doc[pidx].get_text("words")
        # Three side-by-side panels; split on the x of each "S.N." header.
        sn_x = sorted(w[0] for w in words if w[4] == "S.N.")
        if len(sn_x) != 3:
            sys.exit(f"Table 4 page {pidx}: expected 3 panels, found {len(sn_x)}")
        bounds = [x - 3 for x in sn_x] + [1e9]
        for p in range(3):  # panels read top-to-bottom: left, middle, right
            panel = [w for w in words if bounds[p] <= w[0] < bounds[p + 1]]
            for row in rows_of(panel):
                toks = [w[4] for w in row]
                nums = [t for t in toks if PLAIN_NUM.match(t)]
                if len(nums) == 2 and toks[0].isdigit():
                    records.append({
                        "state_num": headers[-1][0],
                        "name": name_from(toks, 2),
                        "code": int(toks[0]),
                        "pop2011": int(nums[0]) + int(nums[1]),
                    })
                else:
                    m = STATE_RE.match(" ".join(toks))
                    if m and not nums:
                        headers.append((int(m.group(1)), m.group(2).strip().upper()))
    return records, headers


def parse_table8(doc):
    """Returns ([record], [(state_num, state_name)]) from Table 8."""
    assert "Table 8" in doc[TABLE8_PAGES[0]].get_text(), "Table 8 not at expected page"
    records, headers = [], []
    for pidx in TABLE8_PAGES:
        for row in rows_of(doc[pidx].get_text("words")):
            toks = [w[4] for w in row]
            nums = [t for t in toks if COMMA_NUM.match(t)]
            if len(nums) == 10 and toks[0].isdigit():
                vals = [int(n.replace(",", "")) for n in nums]
                records.append({
                    "state_name": headers[-1][1],
                    "name": name_from(toks, 10),
                    "pops": {YEARS[i]: vals[2 * i] + vals[2 * i + 1] for i in range(5)},
                })
            else:
                m = STATE_RE.match(" ".join(toks))
                if m and not nums:
                    headers.append((int(m.group(1)), m.group(2).strip().upper()))
    return records, headers


def shapefile_subdist_codes():
    """Set of (pc11_d_id, pc11_sd_id) that have a polygon in the SHRUG
    subdistrict shapefile - i.e. the subdistricts build_country.py will keep.
    Returns None if the shapefile is unavailable (then we don't filter)."""
    if not SUBDIST_SHP.exists():
        print(f"  (no subdistrict shapefile at {SUBDIST_SHP} - "
              "level-3 shares will NOT be renormalised to the geometry set)")
        return None
    import geopandas  # local import: only the geo step needs it
    gdf = geopandas.read_file(SUBDIST_SHP, columns=["pc11_d_id", "pc11_sd_id"])
    return {(str(d), str(sd)) for d, sd in zip(gdf["pc11_d_id"], gdf["pc11_sd_id"])}


def subdistrict_rows(district_proj):
    """Level-3 rows: each subdistrict's 2011 SHRUG population scaled by its
    district's projected growth. Shares within a district sum to 1, so the
    subdistricts always add up exactly to the district projection.

    Only subdistricts with a polygon are counted - both in the per-district
    share denominator and in the emitted rows. The census 99999 "unclassified"
    residuals (and a handful of other polygon-less subdistricts) carry real
    population in the weights file but have no geometry, so build_country.py
    drops them; if we left them in the denominator their population would
    vanish from the map and the subdistricts would undershoot their district
    (e.g. West Tripura summed to 1.30M against a 1.96M district). Excluding
    them here renormalises the surviving subdistricts to absorb that share.

    Code = pc11_district_id + pc11_subdistrict_id (pc11_district_id is global,
    so the pair is unique) - matches build_country.py's composite code_col.
    """
    if not SUBDIST_CSV.exists():
        print(f"  (no subdistrict CSV at {SUBDIST_CSV} - skipping level 3)")
        return []
    keep = shapefile_subdist_codes()
    subdists, dist_sum, dropped = [], {}, 0
    with SUBDIST_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d, sd = r["pc11_district_id"], r["pc11_subdistrict_id"]
            if keep is not None and (d, sd) not in keep:
                dropped += 1
                continue   # no polygon -> exclude from share denominator too
            raw = r["sd_pc11_pca_tot_p"]
            pop2011 = float(raw) if raw not in ("", "NA") else 0.0
            subdists.append((d, d + sd, pop2011))
            dist_sum[d] = dist_sum.get(d, 0.0) + pop2011
    rows, used = [], 0
    for d, full_code, pop2011 in subdists:
        proj = district_proj.get(d)
        if not proj or dist_sum.get(d, 0) <= 0:
            continue
        share = pop2011 / dist_sum[d]
        for year, dpop in proj.items():
            rows.append((full_code, 3, year, round(share * dpop)))
        used += 1
    print(f"level 3: {used}/{used + dropped} subdistricts downscaled "
          f"({dropped} polygon-less rows excluded from shares)")
    return rows


def main():
    if not PDF.exists():
        sys.exit(f"missing PDF: {PDF}")
    doc = fitz.open(PDF)
    t4, h4 = parse_table4(doc)
    t8, h8 = parse_table8(doc)
    print(f"Table 4: {len(t4)} districts, {len(h4)} states")
    print(f"Table 8: {len(t8)} districts, {len(h8)} states")

    name2num = {name: num for num, name in h4}          # census name -> census num
    t4_index = {(r["state_num"], key(r["name"])): r for r in t4}
    if len(t4_index) != len(t4):
        dups = [k for k, n in Counter((r["state_num"], key(r["name"])) for r in t4).items() if n > 1]
        print(f"  WARNING: {len(t4) - len(t4_index)} duplicate t4 keys: {dups}")

    rows, state_totals, unmatched, matched, district_proj = [], {}, [], set(), {}
    for d in t8:
        census_name = T8_TO_CENSUS.get(d["state_name"], d["state_name"])
        snum = name2num.get(census_name)
        rec = t4_index.get((snum, key(d["name"]))) if snum else None
        if rec is None:
            unmatched.append((d["state_name"], d["name"]))
            continue
        matched.add(rec["code"])
        code = f"{rec['code']:03d}"
        district_proj[code] = d["pops"]
        for year, pop in d["pops"].items():
            rows.append((code, 2, year, pop))
            state_totals.setdefault(snum, {})
            state_totals[snum][year] = state_totals[snum].get(year, 0) + pop

    # Table 4 districts with no Table 8 projection - keep the census-2011 point.
    t4_only = [r for r in t4 if r["code"] not in matched]
    for r in t4_only:
        rows.append((f"{r['code']:03d}", 2, 2011, r["pop2011"]))

    # Level 1 - state = sum of its districts.
    for snum, by_year in state_totals.items():
        for year, pop in by_year.items():
            rows.append((f"{snum:02d}", 1, year, pop))

    # Level 3 - subdistrict, proportional downscale of district projections.
    rows += subdistrict_rows(district_proj)

    print(f"matched {len(matched)} districts")
    if unmatched:
        print(f"UNMATCHED Table 8 ({len(unmatched)}): {unmatched}")
    if t4_only:
        print(f"census-2011-only ({len(t4_only)}): {[r['name'] for r in t4_only]}")
    for year in YEARS:
        print(f"  India {year}: {sum(by.get(year, 0) for by in state_totals.values()):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
