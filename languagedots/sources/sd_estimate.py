"""Sudan: minority languages from published speaker estimates placed on their home states, the
rest of each state on Sudanese Arabic. Ask 019's ruling (Anita, 2026-10-05). Every count rests
on the COD-PS 2022 state projections (religiondots' sd_lookup.csv). The record is sources/sd.md.

    python sources/sd_estimate.py     -> data/normalized/sd.csv (state x node, counts)

FIGURES, one rule for every language:
  * Ethnologue's speaker figure for Sudan, as Wikipedia's infobox carries it (read 2026-10-05),
    where the language is spoken only (or almost only) in Sudan and the figure is dated 2019 or
    later: ETHNOLOGUE below. These are speakers, counted near 2022, the projection's year.
  * Otherwise Joshua Project's people groups in Sudan (data/raw/pg/joshuaproject_pgic.csv,
    ROG3 = SU), summed by primary language (ROL3), scaled by COD-PS 2022 / JP's Sudan total.
    JP already files Arabic-speaking members of a group apart ("Zaghawa, Arabized", "Midob,
    Tidda Arabized", "Kadugli, Arabized": 6.6M people in 40 such groups, all on Sudanese
    Arabic), so its primary-language column is close to a speaker estimate. Ethnologue
    figures older than 2019 (Koalib 2009, Katcha-Kadugli-Miri 2004) lose to JP's.
  * Left out: groups JP lists that are foreign residents or refugees (Egyptian, Moroccan,
    Levantine, Algerian and Yemeni Arabic, Amharic, Oromo, Tigrinya, Kunama, Me'en, Swahili,
    Mandarin), whom COD-PS's projection does not separate either (the old gap); "Deaf"; Tigre
    (JP's Beni Amer, whom Ethnologue's 2.55M Beja figure includes as Beja speakers); Kanuri
    (JP's 461,000 "Kanuri, Yerwa" in East Darfur: no speaker source, and the surveys' Darfur
    verbatims name Hausa, Fula and Tama but no Kanuri).

PLACEMENT: each language on its home state(s). Languages JP locates at one point (most of the
Nuba Mountains and Blue Nile languages) go to the state holding that point (religiondots' hexes,
nearest centroid), with the overrides in HOME_POINT_FIX. Languages spread over several states
get a fixed split in SPLIT, each with its reason.

CAP: no state may be more than CAP non-Arabic. South Kordofan's projection (1.2M) cannot hold
the Nuba languages' 1.3M speakers; the excess goes to Khartoum at the state's own language mix
(the Nuba and Darfuri communities displaced to Khartoum since the 1980s). Recorded per state.
"""
import csv
import io
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
from rdlink import RD_GEO  # noqa: E402

LOOKUP = RD_GEO / "sd" / "sd_lookup.csv"
HEXES = RD_GEO / "sd" / "sd_hexes.gpkg"
JP = ROOT / "data" / "raw" / "pg" / "joshuaproject_pgic.csv"
OUT = ROOT / "data" / "normalized" / "sd.csv"
CODPS_2022 = 46_934_433
N_STATES = 18
CAP = 0.80
OVERFLOW_TO = "SD01"          # Khartoum

AA, NS, NC = "afroasiatic", "nilosaharan", "nigercongo"
ARABIC = f"{AA}.sudanese_arabic"
KO = f"{NC}.kordofanian"
KD = f"{NS}.kadu"

# ISO 639-3 (JP's ROL3) -> node. Glottolog family in the comment (data/raw/glottolog).
NODE = {
    "bej": f"{AA}.cushitic.beja",            # beja1238
    "fvr": f"{NS}.fur",                      # furr1244, Furan
    "fia": f"{NS}.nobiin",                   # nobi1240, Nubian
    "dgl": f"{NS}.dongolawi",                # dong1288 Andaandi, Nubian
    "mei": f"{NS}.midob",                    # mido1240, Nubian
    "ghl": f"{NS}.ghulfan",                  # ghul1238, Nubian (Hill Nubian)
    "kdu": f"{NS}.kadaru",                   # kada1282, Nubian (Hill Nubian)
    "kko": f"{NS}.karko",                    # kark1256, Nubian (Hill Nubian)
    "dil": f"{NS}.dilling",                  # dill1242, Nubian (Hill Nubian)
    "drb": f"{NS}.dair",                     # dair1239, Nubian (Hill Nubian)
    "wll": f"{NS}.wali",                     # wali1262, Nubian (Hill Nubian)
    "mls": f"{NS}.maban.masalit",            # nucl1440, Maban
    "mde": f"{NS}.maban.maba",               # maba1277, Maban
    "zag": f"{NS}.zaghawa",                  # zagh1240, Saharan
    "tuq": f"{NS}.tubu",                     # teda1241 Tedaga, Saharan
    "tma": f"{NS}.tama",                     # tama1331, Tamaic
    "mgb": f"{NS}.mararit",                  # mara1396, Tamaic
    "sjg": f"{NS}.assangori",                # assa1269, Tamaic
    "amj": f"{NS}.amdang",                   # amda1238, Furan
    "daj": f"{NS}.daju_darfur",              # darf1239 Dar Fur Daju, Dajuic
    "dau": f"{NS}.daju",                     # dars1235 Dar Sila Daju, td.txt's leaf
    "shj": f"{NS}.shatt",                    # shat1244, Dajuic
    "liu": f"{NS}.logorik",                  # logo1261, Dajuic
    "nyi": f"{NS}.nyimang",                  # amas1236 Ama, Nyimang family
    "aft": f"{NS}.afitti",                   # afit1238, Nyimang family
    "teq": f"{NS}.temein",                   # nucl1339, Temeinic
    "keg": f"{NS}.tese",                     # tese1238, Temeinic
    "tbi": f"{NS}.gaam",                     # gaam1241, Eastern Jebel
    "soh": f"{NS}.aka",                      # akaa1242, Eastern Jebel
    "xel": f"{NS}.kelo",                     # kelo1246, Eastern Jebel
    "wti": f"{NS}.berta.berta",              # bert1248, et.txt's leaf
    "guk": f"{NS}.gumuz",                    # gumu1244, et.txt's leaf
    "jum": f"{NS}.nilotic.jumjum",           # jumj1238, Nilotic
    "bdi": f"{NS}.nilotic.burun",            # buru1301, Nilotic
    "udu": f"{NS}.koman.uduk",               # uduk1239, Koman
    "xom": f"{NS}.koman.komo",               # komo1258, et.txt's leaf
    "gza": f"{AA}.omotic.mao",               # ganz1246 Ganza, Blue Nile Mao: et.txt's Mao leaf
    "krs": f"{NS}.centralsudanic.kresh",     # gbay1288 Kresh-Aja (Central Sudanic as usually grouped)
    "yul": f"{NS}.centralsudanic.yulu",      # yulu1243
    "kcm": f"{NS}.centralsudanic.gula",      # gula1266, cf.txt's leaf
    "sba": f"{NS}.centralsudanic.ngambay",   # ngam1268 Ngambay: td.txt's leaf since 2026-10-09
                                             # (was the plain Sara leaf, cf.txt's)
    "fgr": f"{NS}.centralsudanic.fongoro",   # fong1243
    "gya": f"{NC}.gbaya.gbaya",              # nort2775 Northwest Gbaya, cf.txt's leaf
    "hau": f"{AA}.chadic.hausa",
    "sok": f"{AA}.chadic.sokoro",            # soko1263
    "fub": f"{NC}.atlantic.fulah",
    # Kordofanian (Niger-Congo as most readers know it; Glottolog's Heibanic, Talodi, Rashad,
    # Katla-Tima and Tegem families)
    "kib": f"{KO}.koalib", "mor": f"{NC}.moro", "lro": f"{KO}.laro", "tic": f"{KO}.tira",
    "lof": f"{KO}.logol", "otr": f"{KO}.otoro", "fuj": f"{KO}.ko", "hbn": f"{KO}.heiban",
    "shw": f"{KO}.shwai", "wrn": f"{KO}.warnang",
    "dec": f"{KO}.dagik", "jle": f"{KO}.ngile", "acz": f"{KO}.acheron", "tlo": f"{KO}.talodi",
    "taz": f"{KO}.tocho", "eli": f"{KO}.nding", "lmd": f"{KO}.lumun",
    "ras": f"{KO}.tegali", "tag": f"{KO}.tagoi", "kcr": f"{KO}.katla", "tms": f"{KO}.tima",
    "laf": f"{KO}.lafofa",
    # Kadugli-Krongo (Kadu): Nilo-Saharan as it is usually placed; Glottolog has it apart
    "xtc": f"{KD}.katcha_kadugli_miri", "tey": f"{KD}.tulishi", "tbr": f"{KD}.tumtum",
    "kcp": f"{KD}.kanga", "kec": f"{KD}.keiga", "kgo": f"{KD}.krongo",
}
LEFT_OUT = {"apd": "Sudanese Arabic (the remainder)", "arz": "foreign", "ary": "foreign",
            "apc": "foreign", "arq": "foreign", "acq": "foreign", "amh": "foreign",
            "gaz": "foreign", "tir": "foreign", "kun": "foreign (Eritrean)", "mym": "foreign",
            "swh": "foreign", "cmn": "foreign", "xxx": "Deaf, not a language",
            "tig": "Beni Amer inside Ethnologue's Beja; Eritrean Tigre foreign",
            "knc": "no speaker source; surveys found none"}

# Ethnologue figures for Sudan, via Wikipedia's infoboxes (read 2026-10-05; edition in brackets)
ETHNOLOGUE = {
    "bej": (2_550_000, "Beja: 'In 2022 there were 2,550,000 Beja speakers in Sudan'"),
    "mls": (980_000, "Masalit 980,000 (2022-2024; Chad's Masalit have shifted, 10 speakers in 1991)"),
    "fvr": (790_000, "Fur 790,000 (2004-2023; Chad's share small)"),
    "nyi": (170_000, "Nyimang 170,000 (2022) [27th ed.]"),
    "tbi": (110_000, "Gaam 110,000 (2022)"),
    "mei": (93_000, "Midob 93,000 (2022)"),
    "mor": (79_000, "Moro 79,000 (2022) [27th ed.]"),
    "dgl": (35_000, "Dongolawi 35,000 (2023) [27th ed.]"),
}

# Split over several states: fractions of the language's Sudan total.
SPLIT = {
    # Beja: Red Sea (Amarar, Bisharin, Hadendowa hills and Port Sudan) and Kassala (Hadendowa
    # and Beni Amer heartland) as the big two; Bisharin on the Atbara and in River Nile's east,
    # a few in Gedaref.
    "bej": {"SD10": 0.45, "SD11": 0.45, "SD16": 0.05, "SD12": 0.05},
    # Nobiin: Halfa, Sukkot and Mahas in Northern; New Halfa in Kassala (the 1964 resettlement);
    # Khartoum, where "many have since migrated" (Wikipedia, Nobiin language).
    "fia": {"SD17": 0.50, "SD11": 0.20, "SD01": 0.30},
    # Fur: Jebel Marra and Zalingei (Central), Nyala's hinterland (South), Kebkabiya and Tawila
    # (North), a few in West Darfur.
    "fvr": {"SD06": 0.45, "SD03": 0.25, "SD02": 0.25, "SD04": 0.05},
    # Masalit: Dar Masalit (West Darfur), with the Masalit of Gereida and Nyala (South) and of
    # Mukjar and Bindisi (Central).
    "mls": {"SD04": 0.60, "SD03": 0.25, "SD06": 0.15},
    # Zaghawa: Dar Zaghawa (Kornoi, Tine, Um Baru, North Darfur); settlers in South Darfur.
    "zag": {"SD02": 0.80, "SD03": 0.20},
    # Dar Fur Daju: Nyala (South Darfur) and Lagowa (West Kordofan, JP's point).
    "daj": {"SD03": 0.50, "SD18": 0.50},
    # Fulfulde (Fellata): JP's point in South Darfur; the surveys' Fula answers are in Blue Nile
    # and White Nile; Sennar's Fellata villages.
    "fub": {"SD03": 0.40, "SD08": 0.30, "SD14": 0.15, "SD09": 0.15},
    # Hausa: JP's point in Khartoum; the Hausa settlements of Sennar (Maiurno), the Gezira
    # scheme, Gedaref and Blue Nile. Evenly.
    "hau": {"SD01": 0.20, "SD15": 0.20, "SD14": 0.20, "SD12": 0.20, "SD08": 0.20},
}
# JP points that land in the wrong state
HOME_POINT_FIX = {
    "mei": "SD02",   # Midob: Jebel Midob is North Darfur (JP's point is 1 degree east of it)
    "xtc": "SD07",   # Katcha-Kadugli-Miri: Kadugli, South Kordofan (JP's point is on the line)
}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def largest_remainder(vals, total):
    f = np.asarray(vals, dtype=float)
    base = np.floor(f)
    k = int(round(total - base.sum()))
    base[np.argsort(-(f - base))[:k]] += 1
    return base.astype(int)


def jp_rows():
    txt = JP.read_text(encoding="utf-8-sig")
    return [r for r in csv.DictReader(io.StringIO(txt[txt.index("ROG3,"):])) if r["ROG3"] == "SU"]


def state_of_points(pts):
    import geopandas as gpd
    g = gpd.read_file(HEXES)
    c = g.geometry.to_crs(3857).centroid.to_crs(4326)
    X, Y, U = c.x.to_numpy(), c.y.to_numpy(), g["unit"].astype(str).to_numpy()
    out = []
    for lon, lat in pts:
        out.append(U[((X - lon) ** 2 + (Y - lat) ** 2).argmin()])
    return out


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    lut["pop"] = lut["pop"].astype(int)
    say(len(lut) == N_STATES and int(lut["pop"].sum()) == CODPS_2022,
        f"sd_lookup.csv: {len(lut)} states, {int(lut['pop'].sum()):,} people (COD-PS 2022)")
    say(bool((lut["geo_id"] == lut["unit"]).all()), "sd_lookup.csv geo_id == unit")
    pop = lut.set_index("geo_id")["pop"]
    name = lut.set_index("geo_id")["name"]

    jp = jp_rows()
    jp_total = sum(int(r["Population"]) for r in jp)
    scale = CODPS_2022 / jp_total
    print(f"  Joshua Project: {len(jp)} groups in Sudan, {jp_total:,} people; scale {scale:.4f}")
    isos = {r["ROL3"] for r in jp}
    say(isos <= set(NODE) | set(LEFT_OUT), f"every JP language is mapped or left out "
        f"({sorted(isos - set(NODE) - set(LEFT_OUT))})")

    # per language: Sudan total and the JP points (population-weighted) for its home state
    by = {}
    states = state_of_points([(float(r["Longitude"]), float(r["Latitude"])) for r in jp])
    for r, st in zip(jp, states):
        iso = r["ROL3"]
        if iso not in NODE:
            continue
        d = by.setdefault(iso, {"jp": 0, "pts": {}})
        n = int(r["Population"])
        d["jp"] += n
        d["pts"][st] = d["pts"].get(st, 0) + n

    rows, check = [], []
    for iso, d in by.items():
        if iso in ETHNOLOGUE:
            total, src = ETHNOLOGUE[iso]
            label = "Ethnologue via Wikipedia: " + src
        else:
            total, label = d["jp"] * scale, f"Joshua Project {d['jp']:,} x {scale:.4f}"
        if iso in SPLIT:
            split = SPLIT[iso]
        elif iso in HOME_POINT_FIX:
            split = {HOME_POINT_FIX[iso]: 1.0}
        else:
            split = {k: v / d["jp"] for k, v in d["pts"].items()}
        say(abs(sum(split.values()) - 1) < 1e-9, f"{iso} split sums to 1")
        for st, f in split.items():
            rows.append((st, NODE[iso], label, total * f))
        check.append((NODE[iso], total, d["jp"] * scale, ", ".join(
            f"{name[s]} {f:.0%}" for s, f in sorted(split.items(), key=lambda t: -t[1]))))

    df = pd.DataFrame(rows, columns=["geo_id", "node", "label", "count"])
    df = df.groupby(["geo_id", "node"], as_index=False).agg(count=("count", "sum"),
                                                             label=("label", "first"))

    # cap, overflow to Khartoum at the state's own mix
    minor = df.groupby("geo_id")["count"].sum().reindex(pop.index, fill_value=0.0)
    over = []
    for st in pop.index:
        lim = CAP * pop[st]
        if minor[st] > lim:
            f = lim / minor[st]
            m = df["geo_id"] == st
            moved = df.loc[m].copy()
            moved["count"] *= (1 - f)
            moved["geo_id"] = OVERFLOW_TO
            moved["label"] = moved["label"] + f"; moved from {name[st]} over the {CAP:.0%} cap"
            df.loc[m, "count"] *= f
            over.append(moved)
            print(f"  cap: {name[st]} minority {minor[st]:,.0f} of {pop[st]:,} "
                  f"({minor[st] / pop[st]:.0%}); {minor[st] - lim:,.0f} moved to Khartoum")
    if over:
        df = pd.concat([df] + over).groupby(["geo_id", "node"], as_index=False).agg(
            count=("count", "sum"), label=("label", "first"))
    minor = df.groupby("geo_id")["count"].sum().reindex(pop.index, fill_value=0.0)
    say(bool((minor <= CAP * pop + 1).all()), f"every state at most {CAP:.0%} non-Arabic")

    rest = pd.DataFrame({"geo_id": pop.index, "node": ARABIC, "count": (pop - minor).values,
                         "label": "the rest of the state"})
    df = pd.concat([df, rest])
    out = []
    for g, d in df.groupby("geo_id"):
        d = d.copy()
        d["count"] = largest_remainder(d["count"], pop[g])
        out.append(d)
    df = pd.concat(out)
    df = df[df["count"] > 0]
    say(int(df["count"].sum()) == CODPS_2022, f"drawn total {int(df['count'].sum()):,}")
    bad = [g for g in pop.index if int(df.loc[df["geo_id"] == g, "count"].sum()) != pop[g]]
    say(not bad, f"every state sums to its COD-PS population ({bad})")

    print("\n  languages: figure used, JP scaled, homes")
    for node, t, j, s in sorted(check, key=lambda x: -x[1]):
        print(f"    {node:40s} {t:>10,.0f} {j:>10,.0f}  {s}")
    nat = df.groupby("node")["count"].sum().sort_values(ascending=False)
    print(f"\n  national: {len(nat)} nodes; Sudanese Arabic {nat[ARABIC]:,} "
          f"({nat[ARABIC] / CODPS_2022:.1%}), other languages {CODPS_2022 - nat[ARABIC]:,}")
    print("\n  by state, as drawn:")
    for st in pop.index:
        d = df[df["geo_id"] == st].set_index("node")["count"]
        top = d.drop(ARABIC, errors="ignore").sort_values(ascending=False).head(4)
        print(f"    {name[st]:15s} Arabic {d.get(ARABIC, 0) / pop[st]:6.1%}   " + ", ".join(
            f"{k.split('.')[-1]} {v / pop[st]:.1%}" for k, v in top.items()))

    res = pd.DataFrame({
        "geo_id": df["geo_id"], "geo_level": "state", "geo_name": df["geo_id"].map(name),
        "source_category": df["node"], "source_label": df["label"], "count": df["count"],
        "tier": "modelled", "source_id": "sd_estimates_2022", "year": 2022,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False,
                                                                         encoding="utf-8")
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} nodes)")


if __name__ == "__main__":
    main()
