"""Saudi Arabia, 2022 census: citizens and non-Saudis by nationality, read as languages.
-> data/normalized/sa.csv

    python sources/sa_census.py [--fetch]

Nobody in Saudi Arabia is asked a language (the 2022 census has no item). The census counts
Saudis and non-Saudis in each of the 13 regions, non-Saudis by nationality and sex for the whole
country, and each region's non-Saudi sex ratio. Built under Anita's 2026-10-05 ruling for
countries with no language question (AGENT_BRIEF.md §2): the national language for citizens,
immigrant languages proxied by citizenship. EVERY ROW IS `derived`.

  1. sources/sa_extract.py (its own process) copies religiondots' parsed and checked census
     tables into data/raw/sa/ (--fetch, or when they are missing). religiondots fetched them:
     GLMM's mirror of the census portal, checked against GASTAT's report.
  2. Saudis: all on Saudi Arabic, one node for Najdi, Hejazi, Gulf and the south's varieties,
     which nothing counts separately (as Iraq's and Sudan's Arabic).
  3. non-Saudis: each nationality on a language mix (NATIONALITY below, sources/sa.md §3):
       * most on the country's main language, fr_build.COUNTRY_LANG with Saudi overrides;
       * India by state of origin: Keralites from the Kerala Migration Survey 2023, everyone else
         by the Ministry of External Affairs' emigration clearances 2011-17 by state; each state at
         its own 2011 census mother-tongue mix (data/normalized/in.csv);
       * Pakistan by province of registration (Bureau of Emigration, 2019-20 and 2020-21), each
         province at its 2023 census mother-tongue mix (data/normalized/pk.csv);
       * a few multilingual origins already on this map (HOME_MIX) at their own drawn mix.
  4. by region: each region's non-Saudi men take the national language mix of non-Saudi men and
     its women that of non-Saudi women (religiondots' sex split, sources/sa_extract.py).
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "taxonomy"), str(ROOT)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW = ROOT / "data" / "raw" / "sa"
NORM = ROOT / "data" / "normalized"
OUT = NORM / "sa.csv"
CITIZENS, NON_SAUDIS = 18_792_262, 13_382_962

AR = "afroasiatic"
SAUDI_ARABIC = f"{AR}.saudi_arabic"
YEMENI_ARABIC = f"{AR}.yemeni_arabic"
EGYPTIAN_ARABIC = f"{AR}.egyptian_arabic"
LEVANTINE_ARABIC = f"{AR}.levantine_arabic"
ROHINGYA = "indoeuropean.indoaryan.eastern.rohingya"
PAHARI_POTHWARI = "indoeuropean.indoaryan.northwestern.pahari_pothwari"

# ---- India: state of origin --------------------------------------------------------------
# Kerala Migration Survey 2023 (IIMAD, draft report, 2024): 2,154,275 emigrants from Kerala
# (Table 3.1), 16.9% of them in Saudi Arabia (Table 3.7).
KMS_EMIGRANTS, KMS_SAUDI = 2_154_275, 0.169
# Everyone else: the ILO's India Labour Migration Update 2018, Figure 4, "State wise emigration
# clearance granted, 2011-17" (MEA data), the ten states named; Kerala's 10% left out because
# the survey above stands for it. Clearances are needed only by workers without ten years of
# school, to all 18 ECR countries, of which Saudi Arabia is the largest. Telangana is part of
# Andhra Pradesh in the 2011 census, so its 2% joins Andhra Pradesh's mix.
ECR_2011_17 = {"UTTAR PRADESH": 31, "BIHAR": 15, "TAMIL NADU": 11, "WEST BENGAL": 8,
               "RAJASTHAN": 7, "PUNJAB": 7, "ANDHRA PRADESH": 7 + 2, "ODISHA": 2}

# ---- Pakistan: province of registration ---------------------------------------------------
# Ministry of Overseas Pakistanis and HRD, Year Book 2020-21, Table 14: workers registered by
# the Bureau of Emigration and Overseas Employment, by province, FY 2019-20 and 2020-21 (all
# destinations; Saudi Arabia takes about half of all registrations since 1971).
BEOE = {"Federal": (4_008, 1_065), "Punjab": (276_294, 90_134), "Sindh": (40_059, 12_298),
        "KPK": (166_480, 48_628), "Baluchistan": (4_071, 1_333),
        "Azad Kashmir": (18_902, 5_683), "Northern area": (589, 441),
        "Tribal area": (20_688, 9_198)}
BEOE_TOTAL = (531_091, 168_780)
# province -> the 2023 census province whose mother-tongue mix it takes, or a node. The census
# does not cover Azad Kashmir or Gilgit-Baltistan; the former tribal areas are in Khyber
# Pakhtunkhwa since 2018 but are Pashto-speaking nearly throughout, so drawn as Pashto.
PK_PROVINCE = {"Federal": "PK23-islamabad", "Punjab": "PK23-punjab", "Sindh": "PK23-sindh",
               "KPK": "PK23-khyber-pakhtunkhwa", "Baluchistan": "PK23-balochistan",
               "Azad Kashmir": PAHARI_POTHWARI,
               "Northern area": "indoeuropean.indoaryan.dardic.shina",
               "Tribal area": "indoeuropean.iranian.pashto"}

# ---- origins at their own drawn mix (multilingual, 20,000+ in Saudi Arabia, on this map) ----
HOME_MIX = ["PH", "ID", "NP", "ET", "AF", "UG", "KE", "LK", "NG", "SD", "ML", "NE"]

# ---- single-language overrides of fr_build.COUNTRY_LANG (node ids) ------------------------
OVERRIDE = {
    "YE": YEMENI_ARABIC, "EG": EGYPTIAN_ARABIC,
    "SY": LEVANTINE_ARABIC, "JO": LEVANTINE_ARABIC, "PS": LEVANTINE_ARABIC,
    "LB": LEVANTINE_ARABIC,
    "IQ": "afroasiatic.iraqi_arabic",
    # GASTAT's Myanmar nationals are the Rohingya (religiondots sources/sa.py: Refugee Law
    # Initiative 2023; the 2017 special residency permits)
    "MM": ROHINGYA,
    "FR": "indoeuropean.romance.french",   # France is not in COUNTRY_LANG (France's own table)
}
# GASTAT's `Other` rows: the narrowest node holding everything in them
OTHER_ROW = {"ssa": "africa_other"}


def run_extract():
    need = [RAW / "sa_regions.csv", RAW / "sa_nationality.csv"]
    if "--fetch" in sys.argv or not all(p.exists() for p in need):
        r = subprocess.run([sys.executable, str(HERE / "sa_extract.py")])
        if r.returncode:
            raise SystemExit("sa_extract.py failed")


MIN_SHARE = 0.01   # a home mix keeps languages of 1%+ of its people, scaled back up to 100%


def mix_from(df, col="node"):
    """A state's, province's or country's language shares. Languages under MIN_SHARE are left
    out and the rest scaled up: drawing a migrant population at a whole country's long tail put
    830 languages on the map at a handful of people each, a precision the proxy does not have."""
    s = df.groupby(col)["count"].sum()
    s = s[s > 0]
    s = s / s.sum()
    s = s[s >= MIN_SHARE]
    return (s / s.sum()).to_dict()


def india_mix():
    import in2011
    d = pd.read_csv(NORM / "in.csv", dtype=str)
    d = d[d["geo_level"] == "state"].copy()
    d["count"] = d["count"].astype(int)
    d["node"] = [in2011.resolve(c, g[:2]) for c, g in zip(d["source_code"], d["geo_id"])]
    states = set(d["geo_name"])
    missing = [s for s in [*ECR_2011_17, "KERALA"] if s not in states]
    if missing:
        raise SystemExit(f"in.csv has no state rows for {missing}")
    kerala_n = KMS_EMIGRANTS * KMS_SAUDI
    return d, kerala_n


def combine(parts):
    """[(weight, {node: share})] -> {node: share}, weights normalised."""
    tot = sum(w for w, _ in parts)
    out = {}
    for w, m in parts:
        for n, s in m.items():
            out[n] = out.get(n, 0.0) + w / tot * s
    return out


def india(n_indians):
    d, kerala_n = india_mix()
    k_share = kerala_n / n_indians
    rest = sum(ECR_2011_17.values())
    parts = [(k_share, mix_from(d[d["geo_name"] == "KERALA"]))]
    parts += [((1 - k_share) * v / rest, mix_from(d[d["geo_name"] == s]))
              for s, v in ECR_2011_17.items()]
    print(f"  India: Keralites {kerala_n:,.0f} (KMS 2023), {k_share:.1%} of the census's "
          f"{n_indians:,} Indians; the rest by ECR clearances 2011-17 over {len(ECR_2011_17)} states")
    return combine(parts)


def pakistan():
    import pk2023
    d = pd.read_csv(NORM / "pk.csv")
    d["prov"] = d["geo_id"].str.split("/").str[0]
    d["node"] = d["source_category"].map(pk2023.resolve)
    tot = tuple(sum(v[i] for v in BEOE.values()) for i in (0, 1))
    if tot != BEOE_TOTAL:
        raise SystemExit(f"BEOE provinces sum to {tot}, Table 14 prints {BEOE_TOTAL}")
    parts = []
    for prov, (a, b) in BEOE.items():
        tgt = PK_PROVINCE[prov]
        m = mix_from(d[d["prov"] == tgt]) if tgt.startswith("PK23-") else {tgt: 1.0}
        if not m:
            raise SystemExit(f"pk.csv has no rows for {tgt}")
        parts.append((a + b, m))
    print(f"  Pakistan: {sum(BEOE_TOTAL):,} BEOE registrations 2019-21 over {len(BEOE)} provinces")
    return combine(parts)


def home_mix(cc):
    from countries import load_one
    df = load_one(cc.lower())["counts"]()
    return mix_from(df)


def nationality_mixes(nat):
    import fr2023
    import origin_mix
    from fr_build import COUNTRY_LANG

    def label_mix(v):
        items = [(v, 1.0)] if isinstance(v, str) else list(v.items())
        return {fr2023.NAMES[lab]: s for lab, s in items}

    mixes = {}
    for _, r in nat.iterrows():
        iso, n = r["iso"], int(r["men"] + r["women"])
        if r["table"] == "rest":
            # the Americas, mostly the US (GLMM's selected table; religiondots sources/sa.py)
            m = {fr2023.NAMES["English"]: 1.0}
        elif not iso:
            m = {OTHER_ROW.get(r["table"], "other"): 1.0}
        elif iso == "IN":
            m = india(n)
        elif iso == "PK":
            m = pakistan()
        elif iso in HOME_MIX:
            m = home_mix(iso)
        elif iso not in OVERRIDE and origin_mix.gulf_route(iso, "sa") is not None:
            m = origin_mix.gulf_route(iso, "sa")   # overrides, immigration countries
        elif iso in OVERRIDE:
            m = {OVERRIDE[iso]: 1.0}
        else:
            m = label_mix(COUNTRY_LANG[{"GB": "UK", "GR": "EL"}.get(iso, iso)])
        if abs(sum(m.values()) - 1) > 1e-9:
            raise SystemExit(f"{r['label']}: mix sums to {sum(m.values())}")
        mixes[r.name] = m
    return mixes


def round_within_rows(m):
    out = np.zeros(m.shape, dtype="int64")
    for i in range(m.shape[0]):
        row = m.iloc[i].to_numpy(dtype=float)
        target = int(round(row.sum()))
        base = np.floor(row).astype("int64")
        short = target - int(base.sum())
        if short:
            base[np.argsort(-(row - base))[:short]] += 1
        out[i] = base
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def main():
    run_extract()
    reg = pd.read_csv(RAW / "sa_regions.csv", dtype={"geo_id": str})
    nat = pd.read_csv(RAW / "sa_nationality.csv", keep_default_na=False)
    if int(reg["saudi"].sum()) != CITIZENS or int(reg["non_saudi"].sum()) != NON_SAUDIS:
        raise SystemExit("sa_regions.csv does not sum to the census")
    if int(nat["men"].sum() + nat["women"].sum()) != NON_SAUDIS:
        raise SystemExit("sa_nationality.csv does not sum to the census's non-Saudis")

    mixes = nationality_mixes(nat)
    nodes = sorted({n for m in mixes.values() for n in m})
    men, women = nat["men"].sum(), nat["women"].sum()
    mix_m = {n: sum(nat.at[i, "men"] * m.get(n, 0.0) for i, m in mixes.items()) / men for n in nodes}
    mix_f = {n: sum(nat.at[i, "women"] * m.get(n, 0.0) for i, m in mixes.items()) / women
             for n in nodes}

    fm = pd.DataFrame({r.geo_id: {n: r.non_saudi_men * mix_m[n] + r.non_saudi_women * mix_f[n]
                                  for n in nodes} for r in reg.itertuples()}).T[nodes]
    fc = round_within_rows(fm)
    for r in reg.itertuples():
        if int(fc.loc[r.geo_id].sum()) != r.non_saudi:
            raise SystemExit(f"{r.name}: rounded non-Saudis do not sum to {r.non_saudi:,}")

    rows = [dict(geo_id=r.geo_id, geo_level="region", geo_name=r.name, origin="Saudi",
                 source_category=SAUDI_ARABIC, count=int(r.saudi)) for r in reg.itertuples()]
    name = dict(zip(reg["geo_id"], reg["name"]))
    for gid, row in fc.iterrows():
        for n, c in row.items():
            if c > 0:
                rows.append(dict(geo_id=gid, geo_level="region", geo_name=name[gid],
                                 origin="non-Saudi", source_category=n, count=int(c)))
    out = pd.DataFrame(rows)
    out["tier"] = "derived"
    out["year"] = 2022
    if int(out["count"].sum()) != CITIZENS + NON_SAUDIS:
        raise SystemExit("output does not sum to the census")
    out.to_csv(OUT, index=False, encoding="utf-8")

    # ---- report ----
    tot = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    allpop = CITIZENS + NON_SAUDIS
    print(f"\nwrote {OUT}: {len(out)} rows, {out['geo_id'].nunique()} regions, {allpop:,} people, "
          f"{out['source_category'].nunique()} nodes")
    for n, c in tot.head(25).items():
        print(f"    {n:<50} {c:>11,}  {c / allpop:6.2%}")
    print("  non-Saudi men / women, top languages:")
    for lab, mx in (("men", mix_m), ("women", mix_f)):
        top = sorted(mx.items(), key=lambda kv: -kv[1])[:6]
        print(f"    {lab}: " + ", ".join(f"{n.split('.')[-1]} {s:.1%}" for n, s in top))
    for iso in ("IN", "PK"):
        i = nat.index[nat["iso"] == iso][0]
        top = sorted(mixes[i].items(), key=lambda kv: -kv[1])[:8]
        print(f"  {iso}: " + ", ".join(f"{n.split('.')[-1]} {s:.1%}" for n, s in top))


if __name__ == "__main__":
    main()
