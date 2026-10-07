"""Arab Barometer first-language and ethnicity answers by region, for the countries built from
it without a census language question (sources/{tn,jo,lb,ps,ye}_surveys.py; Iraq's
sources/iq_surveys.py reads the same files its own way).

religiondots' .sav files, read-only. Counts are unweighted (one respondent, one answer), as Iraq.

  wave II   2010-11   q10191   "What is your first language?"
  wave III  2012-14   q1019_1  "First language"
  wave IV   2016-17   q1019a   "First language"
  wave VII  2021-22   Q1012B   "What is your ethnicity?" (ethnic group, not language)
Waves I, V, VI-1/2, VIII ask neither; VI-3's Q1012B is empty for these five countries.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

ARB = RD / "data" / "raw" / "arabbarometer"
WAVES = {
    "ABII": ("ABII_English.sav", "q10191"),
    "ABIII": ("ABIII_English.sav", "q1019_1"),
    "ABIV": ("ABIV_English_Updated.sav", "q1019a"),
    "ABVII": ("AB7_ENG_Release_Version6.sav", "Q1012B"),
}
NOT_DRAWN = {"Missing", "Refuse", "Refused to answer", "Don't know", "Don’t know", "nan",
             "0. missing", "No answer"}
_cache = {}


def _read(wave):
    if wave not in _cache:
        import pyreadstat
        fname, _q = WAVES[wave]
        df, _ = pyreadstat.read_sav(str(ARB / fname), apply_value_formats=True)
        _cache[wave] = df
    return _cache[wave]


def answers(country, wave, region_map, expect_n):
    """{unit: {answer: n}} for one country and wave. `country` is a substring of the country
    label; every region label must be in `region_map` (a value of None drops the respondent's
    region, never silently); the row count must equal `expect_n`."""
    df = _read(wave)
    q = WAVES[wave][1]
    ccol = next(c for c in df.columns if c.lower() == "country")
    gcol = next(c for c in df.columns if c.lower() == "q1")
    sub = df[df[ccol].astype(str).str.contains(country, case=False)]
    if len(sub) != expect_n:
        raise SystemExit(f"Arab Barometer {wave} {country}: {len(sub)} rows, expected {expect_n}")
    out = {}
    for g, a in zip(sub[gcol].astype(str), sub[q].astype(str)):
        if g not in region_map:
            raise SystemExit(f"Arab Barometer {wave} {country}: region {g!r} not mapped")
        u = region_map[g]
        if u is None or a in NOT_DRAWN:
            continue
        out.setdefault(u, {}).setdefault(a, 0)
        out[u][a] += 1
    return out


def shares_to_counts(shares, pop):
    """{label: weight} x an integer population -> {label: int}, largest remainder."""
    tot = sum(shares.values())
    raw = {k: v / tot * pop for k, v in shares.items() if v > 0}
    cnt = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda x: raw[x] - cnt[x], reverse=True)[:pop - sum(cnt.values())]:
        cnt[k] += 1
    assert sum(cnt.values()) == pop
    return cnt


def write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                          "count", "tier", "source_id", "year", "note"])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def report(rows, total):
    natl = {}
    for r in rows:
        natl[r["source_category"]] = natl.get(r["source_category"], 0) + r["count"]
    print(f"  {len(rows)} rows, {sum(natl.values()):,} people (expected {total:,})")
    for k, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<34} {n:>11,}  {n / total * 100:6.2f}%")
    if sum(natl.values()) != total:
        raise SystemExit("rows do not sum to the population")
    return natl
