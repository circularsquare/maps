"""Egypt: Arabic varieties and minority languages from cited speaker estimates placed on their
home governorates, refugees from UNHCR's governorate counts. Every count rests on CAPMAS's own
governorate population estimate for 2026-01-01 (religiondots' eg_lookup.csv).

    python sources/eg_build.py     -> data/normalized/eg.csv (governorate x node, counts)

No Egyptian census asks a language (the 2017 census asked religion and nationality; scout
2026-10-05). The surveys that ask one (Afrobarometer R5-R6, Arab Barometer II-IV: 6,003
answers, 2011-2016) put all but one respondent on Arabic, so they confirm the
big share and say nothing about the minorities (sources/eg.md section 2). So this is the
supervisor's estimate route: each minority's speaker figure from a cited source, put in the
governorates that are its home; Egypt's two big Arabic varieties split by governorate where
Glottolog and Ethnologue draw the line; everyone else on Egyptian Arabic. Node ids are written
straight into source_category (taxonomy/eg2026.py is the identity), and `source_label` keeps
the estimate each row came from. The record is sources/eg.md.

ORDER, per governorate: refugees (UNHCR, May 2026) and each minority are carved out of the
governorate's CAPMAS population; the rest is Sa'idi Arabic in the Sa'idi governorates and
Egyptian Arabic elsewhere.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "taxonomy"))
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

LOOKUP = RD_GEO / "eg" / "eg_lookup.csv"
HEXES = RD_GEO / "eg" / "eg_hexes.gpkg"
OUT = ROOT / "data" / "normalized" / "eg.csv"
CAPMAS_2026 = 108_528_518   # sum of eg_lookup.csv (CAPMAS api/GovernoratePopulation, 2026-01-01)
N_GOV = 27

AR = "afroasiatic"
EGYPTIAN = f"{AR}.egyptian_arabic"
SAIDI = f"{AR}.saidi_arabic"
BEDAWI = f"{AR}.bedawi_arabic"           # Eastern Egyptian Bedawi Arabic, east2690
LIBYAN = f"{AR}.libyan_arabic"           # Western Egyptian Bedawi is its dialect west2774
SIWI = f"{AR}.berber.siwi"
BEJA = f"{AR}.cushitic.beja"
NOBIIN = "nilosaharan.nobiin"
KENZI = "nilosaharan.kenzi"
DOMARI = "indoeuropean.indoaryan.domari"
LEVANTINE = f"{AR}.levantine_arabic"

# ---- Sa'idi Arabic (said1239): Ethnologue's region "Al Minya Governorate and south to Sudan
# border", 27 million speakers (2024; Ethnologue 27th ed. via Wikipedia, read 2026-10-05).
# Glottolog files Western Desert Egyptian Arabic (west2939, the oases) under Sa'idi, so New
# Valley joins. Beni Suef and Faiyum (Middle Egyptian) stay on Egyptian Arabic, as Ethnologue
# draws the line.
SAIDI_GOV = {"EG24", "EG25", "EG26", "EG27", "EG28", "EG29", "EG32"}
SAIDI_ETHNOLOGUE = 27_000_000

# ---- Nubian: Ethnologue (via Wikipedia, read 2026-10-05): Nobiin 502,000 in Egypt (2024),
# Kenzi 35,000 (2023). "In the far-Southern Upper Nile Valley, around Kom Ombo and Aswan,
# there are about 300,000 speakers of Nubian languages, mainly Nobiin, but also Kenuzi"
# (Wikipedia, Languages of Egypt): those go to Aswan; the remaining 237,000 to the cities the
# post-1964 migration went to, Cairo, Giza and Alexandria, by population.
NOBIIN_EG, KENZI_EG = 502_000, 35_000
NUBIAN_ASWAN = 300_000
NUBIAN_CITIES = ["EG01", "EG21", "EG02"]

SIWI_EG = 21_000           # Ethnologue 27th ed. (2013-2023), Siwa and Qara oases, Matrouh
BEJA_EG = 88_000           # Ethnologue via Wikipedia, "as of 2023 ... 88,000 Beja speakers in Egypt"
BEJA_RS_FRACTION = 0.5     # of the Red Sea's people south of 24N (Halaib-Shalateen-Abu Ramad)
BEJA_RS_LAT = 24.0

# ---- Domari: 10,000 speakers in Egypt (Ethnologue 2016 as Joshua Project carries it). The
# Ethnologue country page's 0.3% (~350,000) is three times the Dom population estimated at
# about 100,000 (Wikipedia, Doms in Egypt), so it cannot be speakers. A quarter each to
# Dakahlia (Joshua Project's location), Cairo, Alexandria, and Upper Egypt (the Sa'idi
# governorates, by population), the places Wikipedia names.
DOMARI_EG = 10_000
DOMARI_SPLIT = {"EG12": 0.25, "EG01": 0.25, "EG02": 0.25, "saidi": 0.25}

# ---- Bedouin Arabic.
# Matrouh: Awlad Ali "represent the majority (85 per cent) of its population" (Huesken 2017,
# International Affairs 93(4), p. 900); their Arabic is Western Egyptian Bedawi, a dialect of
# Libyan Arabic in Glottolog (west2774 under liby1240).
MATROUH_AWLAD_ALI = 0.85
# Sinai: "Seventy percent of Sinai residents are Bedouin" (Aziz, Brookings, 2017); North Sinai
# drawn at that share. South Sinai's 13 tribes total 38,000 (Senri Ethnological Studies 55,
# 2001, from the province's Tribal Affairs Department, late 1990s), 70% of the 1996 census's
# 54,495 people; grown since at Egypt's rate (59,312,914 in the 1996 census to CAPMAS 2026),
# since the governorate's own growth is mostly Nile-valley migrants to the resorts.
NORTH_SINAI_BEDOUIN = 0.70
SOUTH_SINAI_TRIBES_1996 = 38_000
EGYPT_1996 = 59_312_914

# ---- Refugees and asylum seekers registered with UNHCR, 31 May 2026, by governorate (UNHCR
# Egypt fact sheet, June 2026, "Main residential areas of refugees in Egypt"; the PDF and a
# render of its map are in data/raw/eg/).
UNHCR_GOV = {
    "EG01": 323_249, "EG02": 89_433, "EG03": 200, "EG04": 5_033, "EG11": 6_989,
    "EG12": 3_957, "EG13": 35_390, "EG14": 22_745, "EG15": 889, "EG16": 1_647,
    "EG17": 9_214, "EG18": 2_038, "EG19": 1_243, "EG21": 573_515, "EG22": 473,
    "EG23": 427, "EG24": 1_171, "EG25": 642, "EG26": 391, "EG27": 245, "EG28": 19_021,
    "EG29": 1_865, "EG31": 687, "EG32": 19, "EG33": 1_000, "EG34": 62, "EG35": 199,
}
UNHCR_TOTAL = 1_101_700    # the fact sheet's headline, rounded
# Same fact sheet: top four origins (rounded); the rest is "more than 57 other nationalities"
UNHCR_ORIGIN = {"SD": 852_000, "SY": 98_100, "SS": 56_000, "ER": 45_000}
# Each origin's language: Sudan and South Sudan at their drawn mixes on this map (Saudi
# Arabia's home-mix method, sources/sa.md); Syrians Levantine Arabic; Eritreans Tigrinya
# (fr_build.COUNTRY_LANG); the rest, mostly Ethiopians, Somalis, Yemenis and Iraqis by UNHCR's
# other releases, on `other`.
HOME_MIX = ["SD", "SS"]
SINGLE = {"SY": LEVANTINE, "ER": f"{AR}.ethiosemitic.tigrinya"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def kontur_share(unit, mask_fn):
    import geopandas as gpd
    g = gpd.read_file(HEXES)
    g = g[g["unit"] == unit]
    c = g.geometry.to_crs(3857).centroid.to_crs(4326)
    m = mask_fn(c.x.to_numpy(), c.y.to_numpy())
    return float(g.loc[m, "pop"].sum() / g["pop"].sum())


def refugee_mix():
    from sa_census import home_mix
    rest = UNHCR_TOTAL - sum(UNHCR_ORIGIN.values())
    parts = {**UNHCR_ORIGIN, "rest": rest}
    mix = {}
    for o, n in parts.items():
        if o in HOME_MIX:
            m = home_mix(o)
        elif o in SINGLE:
            m = {SINGLE[o]: 1.0}
        else:
            m = {"other": 1.0}
        for k, s in m.items():
            mix[k] = mix.get(k, 0.0) + s * n / UNHCR_TOTAL
    say(abs(sum(mix.values()) - 1) < 1e-9, f"refugee mix sums to 1 ({len(mix)} nodes; "
        f"rest {rest:,} on other)")
    return mix


def largest_remainder(vals, total):
    f = np.asarray(vals, dtype=float)
    base = np.floor(f)
    k = int(round(total - base.sum()))
    base[np.argsort(-(f - base))[:k]] += 1
    return base.astype(int)


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    lut["pop"] = lut["pop"].astype(int)
    say(len(lut) == N_GOV and int(lut["pop"].sum()) == CAPMAS_2026,
        f"eg_lookup.csv: {len(lut)} governorates, {int(lut['pop'].sum()):,} people (CAPMAS 2026)")
    say(set(UNHCR_GOV) == set(lut["geo_id"]), "UNHCR figures for all 27 governorates")
    pop = lut.set_index("geo_id")["pop"]
    name = lut.set_index("geo_id")["name"]
    ref_sum = sum(UNHCR_GOV.values())
    say(abs(ref_sum - UNHCR_TOTAL) / UNHCR_TOTAL < 0.005,
        f"UNHCR governorates sum to {ref_sum:,} against the headline {UNHCR_TOTAL:,}")

    rows = []   # (geo_id, node, label, count float, tier)

    def add(g, node, label, n, tier="modelled"):
        rows.append((g, node, label, float(n), tier))

    # refugees
    mix = refugee_mix()
    for g, n in UNHCR_GOV.items():
        for node, s in mix.items():
            add(g, node, "UNHCR registered refugees and asylum seekers, by nationality", n * s,
                "derived")

    # Nubian
    nub_share = {NOBIIN: NOBIIN_EG / (NOBIIN_EG + KENZI_EG), KENZI: KENZI_EG / (NOBIIN_EG + KENZI_EG)}
    rest = NOBIIN_EG + KENZI_EG - NUBIAN_ASWAN
    cities = pop[NUBIAN_CITIES]
    for g, n in [("EG28", NUBIAN_ASWAN)] + [(g, rest * p / cities.sum()) for g, p in cities.items()]:
        for node, s in nub_share.items():
            add(g, node, "Ethnologue: Nobiin 502,000, Kenzi 35,000; ~300,000 around Aswan", n * s)

    # Siwi
    add("EG33", SIWI, "Ethnologue: Siwi 21,000", SIWI_EG)

    # Beja
    rs_south = kontur_share("EG31", lambda x, y: y < BEJA_RS_LAT)
    beja_rs = BEJA_RS_FRACTION * rs_south * pop["EG31"]
    print(f"  Red Sea south of {BEJA_RS_LAT}N: {rs_south:.1%} of its Kontur population; "
          f"Beja there {beja_rs:,.0f}, Aswan {BEJA_EG - beja_rs:,.0f}")
    add("EG31", BEJA, "Ethnologue: Beja 88,000 in Egypt", beja_rs)
    add("EG28", BEJA, "Ethnologue: Beja 88,000 in Egypt", BEJA_EG - beja_rs)

    # Domari
    saidi_pop = pop[sorted(SAIDI_GOV)]
    for k, f in DOMARI_SPLIT.items():
        if k == "saidi":
            for g, p in saidi_pop.items():
                add(g, DOMARI, "Ethnologue 2016: Domari 10,000", DOMARI_EG * f * p / saidi_pop.sum())
        else:
            add(k, DOMARI, "Ethnologue 2016: Domari 10,000", DOMARI_EG * f)

    # Bedouin Arabic
    add("EG33", LIBYAN, "Awlad Ali 85% of Matrouh (Huesken 2017)", MATROUH_AWLAD_ALI * pop["EG33"])
    add("EG34", BEDAWI, "Bedouin 70% of Sinai (Brookings 2017)", NORTH_SINAI_BEDOUIN * pop["EG34"])
    ss = SOUTH_SINAI_TRIBES_1996 * CAPMAS_2026 / EGYPT_1996
    add("EG35", BEDAWI, "South Sinai's 13 tribes, 38,000 in the late 1990s (SES 55), grown at "
        "Egypt's rate", ss)
    print(f"  South Sinai Bedouin {ss:,.0f} = {ss / pop['EG35']:.1%} of the governorate")

    # the rest: Sa'idi or Egyptian Arabic
    df = pd.DataFrame(rows, columns=["geo_id", "node", "label", "count", "tier"])
    used = df.groupby("geo_id")["count"].sum().reindex(pop.index, fill_value=0.0)
    say(bool((used < pop).all()), "every governorate keeps a remainder after the carve-outs")
    for g in pop.index:
        main_node = SAIDI if g in SAIDI_GOV else EGYPTIAN
        lab = ("the rest, Sa'idi governorates (Ethnologue: Minya south to Sudan; Glottolog: "
               "the oases under Sa'idi)" if g in SAIDI_GOV else "the rest")
        add(g, main_node, lab, pop[g] - used[g])
    df = pd.DataFrame(rows, columns=["geo_id", "node", "label", "count", "tier"])
    df = df.groupby(["geo_id", "node", "tier"], as_index=False).agg(
        count=("count", "sum"), label=("label", "first"))
    out = []
    for g, d in df.groupby("geo_id"):
        d = d.copy()
        d["count"] = largest_remainder(d["count"], pop[g])
        out.append(d)
    df = pd.concat(out)
    df = df[df["count"] > 0]
    say(int(df["count"].sum()) == CAPMAS_2026, f"drawn total {int(df['count'].sum()):,}")
    bad = [g for g in pop.index if int(df.loc[df["geo_id"] == g, "count"].sum()) != pop[g]]
    say(not bad, f"every governorate sums to its CAPMAS population ({bad})")

    nat = df.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k, v in nat.head(20).items():
        print(f"    {k:45s} {v:>12,}  {v / CAPMAS_2026:7.3%}")
    sa = nat.get(SAIDI, 0)
    print(f"  Sa'idi drawn {sa:,} against Ethnologue's {SAIDI_ETHNOLOGUE:,} (2024) "
          f"({sa / SAIDI_ETHNOLOGUE - 1:+.1%})")

    res = pd.DataFrame({
        "geo_id": df["geo_id"], "geo_level": "governorate", "geo_name": df["geo_id"].map(name),
        "source_category": df["node"], "source_label": df["label"], "count": df["count"],
        "tier": df["tier"], "source_id": "eg_estimates_2026", "year": 2026,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False,
                                                                         encoding="utf-8")
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} nodes)")


if __name__ == "__main__":
    main()
