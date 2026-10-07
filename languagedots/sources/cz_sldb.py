"""Czechia: ČSÚ, Sčítání lidu, domů a bytů 2021 (SLDB), mother tongue, down to obec and city district.

    python sources/cz_sldb.py --fetch    download the CSVs (14 MB each) and two code lists if missing
    python sources/cz_sldb.py            normalise from data/raw/cz/

-> data/normalized/cz.csv (levels `country`, `kraj`, `municipality`, `city_district`; alternatives,
   never summed across levels; the finest cover is every obec except the 8 that city districts
   subdivide, plus the 142 city districts, as in religiondots/countries/cz.py)

THE QUESTION. Mateřský jazyk (mother tongue), optional, one or two answers. ČSÚ publishes two open
data tables of it, in the layout of religiondots' religion file (religiondots/sources/cz.md):

  sldb2021_jazyk1.csv   "1 mateřský jazyk": persons who named ONE mother tongue, by language.
  sldb2021_jazyk.csv    "1 nebo 2 mateřské jazyky": "<X> celkem", every person who named X,
                        alone or as one of two.

THE DETAIL DEPENDS ON THE LEVEL. The country and its 14 kraje carry 55 language labels, "Osoby se
dvěma mateřskými jazyky" (persons who named two) and "Nezjištěno" (not stated), and add up exactly.
Okres, obec and city district carry only 13 languages (Czech, Slovak, Moravian, Silesian, Polish,
German, Romani, English, Russian, Ukrainian, Vietnamese, Hungarian, Chinese) and not stated; the
two-answer count and the other 43 labels are not printed there.

HOW A UNIT'S PEOPLE ARE DRAWN (spec §3.6, the exact split: half a person to each of two languages).
  * The 13 languages, per obec: single_l (measured) + (celkem_l - single_l) / 2 (derived). Exact.
  * The rest, per obec: pool = total - not stated - sum over the 13 of the line above. This is
    exactly the obec's people (and half-people) on the other 43 labels, because every two-answer
    person names exactly two languages. Its split among the 43 is published only per kraj, so the
    pool is shared out by the kraj's own split of those 43 (single + half of pairs), tier derived.
    Each language's obec shares then add up to the kraj's figure exactly (checked).
Not stated is the gap.

CHECKS (all asserted): both files' unit totals agree; at country and kraj, single + two + not
stated = total and two-answer mentions = 2 x two-answer persons; celkem >= single everywhere; obce
summed by kraj (ČSÚ's kraj-obec code list at the census date) equal the kraj table on all 13
languages, totals and not stated, and city districts likewise against the 8 obce they subdivide;
every pool >= 0 and the pools of a kraj's finest cover equal its 43-label total; the finest cover
summed per language equals the national figure.
"""
import os
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "cz"
OUT = ROOT / "data" / "normalized" / "cz.csv"
REPLACED = ROOT.parent / "religiondots" / "data" / "geo" / "cz" / "cz_replaced.csv"  # read-only

CSU = "https://csu.gov.cz/docs/107508/"
ISMS = ("https://apl2.czso.cz/iSMS/do_cis_export?kodcis=100&typdat=1&cisvaz={vaz}"
        "&datpohl=26.03.2021&cisjaz=203&format=2&separator=%2C")
FILES = {   # name: (url, minimum bytes)
    "sldb2021_jazyk1.csv": (CSU + "87997b45-9487-ccd0-980c-1b879f337058/sldb2021_jazyk1.csv",
                            12_000_000),
    "sldb2021_jazyk.csv": (CSU + "f970b567-ffeb-4ca6-8238-41e4c4824019/sldb2021_jazyk.csv",
                           12_000_000),
    # ČSÚ code lists: kraj -> obec and kraj -> city district, as of the census date
    "vazba_kraj_obec_20210326.csv": (ISMS.format(vaz="43_1250"), 500_000),
    "vazba_kraj_mc_20210326.csv": (ISMS.format(vaz="44_1252"), 10_000),
}
NATIONAL = 10_524_167            # usually resident population, 26 March 2021
LEVELS = {"97": "country", "100": "kraj", "43": "municipality", "44": "city_district"}
UNITS = {"country": 1, "kraj": 14, "municipality": 6254, "city_district": 142}
TWO = "Osoby se dvěma mateřskými jazyky"
NOT_STATED = "Nezjištěno"
FINE = ("municipality", "city_district")


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, min_bytes) in FILES.items():
        p = RAW / name
        if p.exists() and p.stat().st_size > min_bytes:
            continue
        r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if len(r.content) < min_bytes or not r.content.startswith(b'"'):
            raise SystemExit(f"{name}: {len(r.content):,} bytes, not the CSV (a 200 is not a download)")
        p.write_bytes(r.content)
        print(f"fetched {name}: {len(r.content):,} bytes")


def raw(name):
    p = RAW / name
    if not p.exists() or p.stat().st_size < FILES[name][1]:
        raise SystemExit(f"{p} missing or short; run with --fetch")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def read(name):
    df = raw(name)
    if not df["idhod"].iloc[-1]:
        raise SystemExit(f"{name}: last row has no id, truncated?")
    df = df[df["uzemi_cis"].isin(LEVELS)].copy()
    df["level"] = df["uzemi_cis"].map(LEVELS)
    df["hodnota"] = df["hodnota"].astype(int)
    n = df.groupby("level")["uzemi_kod"].nunique().to_dict()
    assert n == UNITS, (name, n)
    return df


def table(df, suffix, drop):
    d = df[(df["jazyk_txt"] != "") & ~df["jazyk_txt"].isin(drop)].copy()
    bad = sorted(set(t for t in d["jazyk_txt"] if not t.endswith(suffix)))
    assert not bad, bad
    d["lang"] = d["jazyk_txt"].str[: -len(suffix)]
    return d.pivot_table(index=["level", "uzemi_kod"], columns="lang", values="hodnota",
                         aggfunc="sum")


def series(df, label):
    return df[df["jazyk_txt"] == label].set_index(["level", "uzemi_kod"])["hodnota"]


def main():
    if "--fetch" in sys.argv:
        fetch()
    one, both = read("sldb2021_jazyk1.csv"), read("sldb2021_jazyk.csv")

    tot = series(one, "")
    assert tot.index.is_unique and tot.sort_index().equals(series(both, "").sort_index())
    assert tot[("country", tot.loc["country"].index[0])] == NATIONAL
    names = one[one["jazyk_txt"] == ""].set_index(["level", "uzemi_kod"])["uzemi_txt"]
    idx = tot.index
    ns = series(one, NOT_STATED).reindex(idx)
    assert ns.notna().all() and ns.equals(series(both, NOT_STATED).reindex(idx)), "not stated"
    two = series(one, TWO)                          # country and kraj only
    single = table(one, " jazyk", [TWO, NOT_STATED]).reindex(idx)
    total = table(both, " celkem", [NOT_STATED]).reindex(idx)
    assert set(single.columns) == set(total.columns)
    total = total[single.columns]
    langs = list(single.columns)

    # the 13 languages printed at every level, and the 43 printed only at country and kraj
    fine_rows = single.loc[list(FINE)]
    detail = [l for l in langs if fine_rows[l].notna().all()]
    rest = [l for l in langs if fine_rows[l].isna().all()]
    assert len(detail) + len(rest) == len(langs) and total.loc[list(FINE), rest].isna().all().all()
    print(f"{len(langs)} labels at country and kraj; {len(detail)} at obec and city district: "
          f"{', '.join(detail)}")

    # ---- country and kraj: complete ----
    for lvl in ("country", "kraj"):
        s, t = single.loc[lvl], total.loc[lvl]
        assert s.notna().all().all() and t.notna().all().all()
        tw = two.loc[lvl].reindex(s.index)
        assert (s.sum(axis=1) + tw + ns.loc[lvl] == tot.loc[lvl]).all(), lvl
        assert ((t - s) >= 0).all().all(), lvl
        assert ((t - s).sum(axis=1) == 2 * tw).all(), lvl
    print("country and 14 kraje: one tongue + two + not stated = total; two-answer mentions = "
          "2 x two-answer persons; celkem >= single")
    assert ((total.loc[list(FINE), detail] - single.loc[list(FINE), detail]) >= 0).all().all()

    # ---- obec and city district -> kraj ----
    vo, vm = raw("vazba_kraj_obec_20210326.csv"), raw("vazba_kraj_mc_20210326.csv")
    kraj_of = {("municipality", k): v for k, v in zip(vo["chodnota2"], vo["chodnota1"])}
    kraj_of.update({("city_district", k): v for k, v in zip(vm["chodnota2"], vm["chodnota1"])})
    fine_idx = [i for i in idx if i[0] in FINE]
    missing = [i for i in fine_idx if i not in kraj_of]
    assert not missing, missing[:5]
    kraje = set(tot.loc["kraj"].index)
    assert set(kraj_of[i] for i in fine_idx) == kraje
    rep = set(pd.read_csv(REPLACED, dtype=str)["kod"])
    assert len(rep) == 8
    cover = [i for i in fine_idx if not (i[0] == "municipality" and i[1] in rep)]
    print(f"finest cover: {len(cover):,} units ({sum(i[0] == 'municipality' for i in cover):,} obce "
          f"+ {sum(i[0] == 'city_district' for i in cover)} city districts)")

    for name, units in (("obce", [i for i in fine_idx if i[0] == "municipality"]),
                        ("finest cover", cover)):
        k = pd.Series([kraj_of[i] for i in units], index=pd.MultiIndex.from_tuples(units))
        for what, frame in (("single", single), ("celkem", total)):
            got = frame.loc[units, detail].groupby(k.values).sum()
            want = frame.loc["kraj", detail].loc[got.index]
            assert len(got) == 14 and (got - want).abs().max().max() == 0, (name, what)
        for what, s in (("total", tot), ("not stated", ns)):
            got = s.loc[units].groupby(k.values).sum()
            assert (got == s.loc["kraj"].loc[got.index]).all(), (name, what)
    print("obce, and the finest cover, summed by kraj equal the kraj table on all 13 languages "
          "(single and celkem), totals and not stated")

    # ---- the drawn figures ----
    drawn_detail = single[detail] + (total[detail] - single[detail]) / 2
    pool = tot - ns - drawn_detail.sum(axis=1)
    kraj_rest = (single.loc["kraj", rest] + (total.loc["kraj", rest] - single.loc["kraj", rest]) / 2)
    kraj_pool = kraj_rest.sum(axis=1)
    cov_pool = pool.loc[cover]
    assert (cov_pool >= 0).all(), cov_pool[cov_pool < 0].head()
    kk = pd.Series([kraj_of[i] for i in cover], index=cov_pool.index)
    got = cov_pool.groupby(kk.values).sum()
    assert (abs(got - kraj_pool.loc[got.index]) < 1e-6).all(), (got, kraj_pool)
    assert (abs(pool.loc["kraj"] - kraj_pool) < 1e-6).all()
    share = kraj_rest.div(kraj_pool, axis=0)
    print(f"pools: every one >= 0; per kraj they add up to the kraj's 43-label figure "
          f"({kraj_pool.sum():,.1f} people nationally)")

    n_two, n_ns = int(two.loc["country"].iloc[0]), int(ns.loc["country"].iloc[0])
    nat_s = single.loc["country"].iloc[0]
    print(f"national: {NATIONAL:,} people; {int(nat_s.sum()):,} named one tongue, {n_two:,} two, "
          f"{n_ns:,} not stated ({n_ns / NATIONAL:.1%})")

    # ---- rows ----
    rows = []

    def emit(i, lang, count, tier, part):
        if count > 0:
            rows.append((i[1], i[0], names[i], f"{lang} jazyk", count, tier, part))

    for i in idx:
        if i[0] in ("country", "kraj"):
            for l in langs:
                emit(i, l, int(single.at[i, l]), "measured", "one tongue")
                emit(i, l, (total.at[i, l] - single.at[i, l]) / 2, "derived", "half of two")
        else:
            for l in detail:
                emit(i, l, int(single.at[i, l]), "measured", "one tongue")
                emit(i, l, (total.at[i, l] - single.at[i, l]) / 2, "derived", "half of two")
            sh = share.loc[kraj_of[i]]
            for l in rest:
                emit(i, l, pool[i] * sh[l], "derived", "kraj split of the remainder")
        if ns[i] > 0:
            rows.append((i[1], i[0], names[i], NOT_STATED, int(ns[i]), "measured", "not stated"))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                      "count", "tier", "part"])
    out["year"] = 2021
    out["source_id"] = "cz_sldb2021_jazyk"

    # finest cover against the national figure, per language
    cov = out[[(lv, g) in set(cover) for lv, g in zip(out["geo_level"], out["geo_id"])]]
    nat = out[out["geo_level"] == "country"].groupby("source_category")["count"].sum()
    got = cov.groupby("source_category")["count"].sum()
    diff = (got.reindex(nat.index).fillna(0) - nat).abs()
    assert diff.max() < 1e-4, diff.sort_values().tail()
    print(f"finest cover summed per label equals the national figure ({len(nat)} labels; "
          f"largest difference {diff.max():.1e})")

    print("national, drawn (one tongue + half of two), largest first:")
    natd = nat.drop(f"{NOT_STATED}").sort_values(ascending=False)
    for lab, v in natd.items():
        print(f"  {lab:<24} {v:>12,.1f}")
    print(f"  {'(people drawn)':<24} {natd.sum():>12,.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
