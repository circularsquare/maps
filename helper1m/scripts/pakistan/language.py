"""Mother tongue for Pakistan's tehsils, districts and provinces -> countries/pakistan/composition.json.

Everything here is read from maps/languagedots (READ-ONLY), which drew the same data as dots:

  * Four provinces and Islamabad: 2023 census Table 11 (population by mother tongue), by
    tehsil / taluka / sub-division / sub-tehsil, read from the CRAN package PakPC2023 with
    languagedots' own reader (sources/pk_t11.py `read_table`). languagedots sums it to
    districts; here it stays at the tehsil-tier unit and goes through helper1m's own crosswalk
    (data/pakistan/crosswalk.csv, Table 1 unit -> adm3 code), so the join is by census unit,
    not by polygon name. Table 11 and Table 1 list the same 591 units; 577 match on folded
    district + unit name, the 14 left are in ALIAS (Table 1's PDF text drops "ll", and Quetta's
    units are written type-first).
  The two below are behind MODELLED_NORTH, off by default, so both areas draw no pie:
  * Gilgit-Baltistan: languagedots' model (data/normalized/pk_north.csv): the census's GB-wide
    mother-tongue shares, split by district with the GB MICS surveys. Per district, which is
    also helper1m's tehsil-level unit there.
  * Azad Kashmir: languagedots' model from the AJK Statistical Year Book 2025, Table 15.31
    (whole-percent district estimates) on the 2023 census population. District level only:
    nothing splits it by tehsil, so AJK tehsils draw no pie.

Labels go through languagedots' mapping (taxonomy/pk2023.py), colours through its palette;
scripts/language_common.py does the grouping, summing up and writing.

    C:\\Python39\\python.exe helper1m\\scripts\\pakistan\\language.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import language_common as lc  # noqa: E402

import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = lc.HELPER / "data" / "pakistan"
CROSSWALK = DATA / "crosswalk.csv"
CENSUS_UNITS = DATA / "census_units.csv"
HEAD_ONLY = 1_041_342   # Table 1's footnote: restricted areas, counted by head only
# Gilgit-Baltistan and Azad Kashmir from languagedots' models (north() below). Off: Anita
# preferred them blank, 2026-10-06, since Table 11 does not cover them.
MODELLED_NORTH = False

# Table 11 (district, unit, admin unit) -> Table 1 (district, unit) as crosswalk.csv writes them.
ALIAS = {
    ("BATAGRAM", "ALLAI", "TEHSIL"): ("BATAGRAM DISTRICT", "AI TEHSIL"),
    ("CHAKWAL", "KALLAR KAHAR", "TEHSIL"): ("CHAKWAL DISTRICT", "KAR KAHAR TEHSIL"),
    ("JHANG", "HAZARI", "TEHSIL"): ("JHANG DISTRICT", "18-HAZARI TEHSIL"),
    ("PANJGUR", "KALLAG", "SUB-TEHSIL"): ("PANJGUR DISTRICT", "KAG SUB-TEHSIL"),
    ("QUETTA", "KUCHLAK", "SUB-DIVISION"): ("QUETTA DISTRICT", "SUB-DIVISION KUCHLAK"),
    ("QUETTA", "PANJPAI", "SUB-TEHSIL"): ("QUETTA DISTRICT", "SUB-TEHSIL PANJPAI"),
    ("QUETTA", "QUETTA", "SUB-DIVISION_CITY"): ("QUETTA DISTRICT", "SUB-DIVISION CITY"),
    ("QUETTA", "SADDAR TEHSIL", "SUB-DIVISION"): ("QUETTA DISTRICT", "SUB-DIVISION SADDAR TEHSIL"),
    ("QUETTA", "SARIAB", "SUB-DIVISION"): ("QUETTA DISTRICT", "SUB-DIVISION SARIAB"),
    ("RAJANPUR", "RAJANPUR", "DE-EXCLUDED_AREA"): ("RAJANPUR DISTRICT", "DE-EXCLUDED AREA RAJANPUR"),
    ("RAWALPINDI", "KALLAR SAYADDAN", "TEHSIL"): ("RAWALPINDI DISTRICT", "KAR SAYADDAN TEHSIL"),
    ("TANDO ALLAHYAR", "CHAMBER", "TALUKA"): ("TANDO AHYAR DISTRICT", "CHAMBER TALUKA"),
    ("TANDO ALLAHYAR", "JHANDO MARI", "TALUKA"): ("TANDO AHYAR DISTRICT", "JHANDO MARI TALUKA"),
    ("TANDO ALLAHYAR", "TANDO ALLAHYAR", "TALUKA"): ("TANDO AHYAR DISTRICT", "TANDO AHYAR TALUKA"),
}

# Groups that exist only in the modelled north get a title saying so.
NORTH = ("Gilgit-Baltistan figure modelled by languagedots: the 2023 census's GB-wide share, "
         "split by district with the GB MICS surveys")
AJK = ("Azad Kashmir figure: the AJK government's whole-percent district estimate (Statistical "
       "Year Book 2025, Table 15.31) on the 2023 census population")
TITLES = {
    "isolate.burushaski": f"Burushaski. {NORTH}",
    "indoeuropean.indoaryan.dardic.khowar": f"Khowar. {NORTH}; in Chitral it is inside Table 11's "
                                            "'Others'",
    "indoeuropean.iranian.wakhi": f"Wakhi. {NORTH}",
    "indoeuropean.indoaryan.northwestern.pahari_pothwari":
        f"Pahari-Pothwari, every local Pahari variety the yearbook names. {AJK}",
    "indoeuropean.indoaryan.rajasthani.gujari": f"Gojri. {AJK}; elsewhere inside 'Others'",
    "indoeuropean.indoaryan.northwestern.dogri": f"Dogri. {AJK}",
    "other": "Languages too small or scattered for a colour of their own",
}


def fold(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def table11():
    """Table 11 as (DISTRICT, TEHSIL, ADMIN_UNIT, LANGUAGE, count), TOTAL rows checked and kept
    apart. Same reader and the same province fill as languagedots' sources/pk_t11.py."""
    pk_t11 = lc.ld_module("pk_t11", "sources")
    df = pk_t11.read_table()
    for c in ("PROVINCE", "DISTRICT", "TEHSIL", "ADMIN_UNIT", "LANGUAGE"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["count"] = df["ALL_SEXES_OVERALL"].fillna(0).astype("int64")
    known = df.dropna(subset=["PROVINCE"]).drop_duplicates("DISTRICT").set_index("DISTRICT")["PROVINCE"]
    known = {**{"TANDO ALLAHYAR": "SINDH"}, **known.to_dict()}
    lost = df["PROVINCE"].isna()
    df.loc[lost, "PROVINCE"] = df.loc[lost, "DISTRICT"].map(known)
    df = df[df["TEHSIL"].notna()].copy()
    key = ["PROVINCE", "DISTRICT", "TEHSIL", "ADMIN_UNIT"]
    tot = df[df["LANGUAGE"] == "TOTAL"].groupby(key)["count"].sum()
    lang = df[df["LANGUAGE"] != "TOTAL"]
    parts = lang.groupby(key)["count"].sum()
    if not tot.index.equals(parts.index) or (tot != parts).any():
        sys.exit("Table 11: a unit's languages do not add to its TOTAL row")
    lc.log(f"Table 11: {len(tot)} units, {int(tot.sum()):,} people, "
           f"{lang['LANGUAGE'].nunique()} language rows")
    return lang[lang["count"] > 0], tot, pk_t11


def check_against_languagedots(lang, pk_t11):
    """District sums must equal languagedots' data/normalized/pk.csv cell for cell, so this is
    the same data languagedots draws."""
    d = lang.groupby(["PROVINCE", "DISTRICT", "LANGUAGE"], as_index=False)["count"].sum()
    d["geo_id"] = [f"PK23-{pk_t11.PROVINCE[p]}/"
                   f"{pk_t11.ALIAS.get(pk_t11.slug(x), pk_t11.slug(x) + '-district')}"
                   for p, x in zip(d["PROVINCE"], d["DISTRICT"])]
    ours = d.set_index(["geo_id", "LANGUAGE"])["count"].sort_index()
    ld = pd.read_csv(lc.LD_NORM / "pk.csv").set_index(["geo_id", "source_category"])["count"]
    ld.index.names = ours.index.names
    ld = ld.sort_index()
    if not ours.index.equals(ld.index) or (ours != ld).any():
        sys.exit("Table 11 summed to districts differs from languagedots' pk.csv")
    lc.log(f"  district sums equal languagedots' pk.csv in all {len(ld):,} cells")


def join_units(tot):
    """Table 11 unit -> helper1m adm3 code, through crosswalk.csv. Also reports where Table 1
    (which includes the people counted by head only in restricted areas) is larger."""
    cw = pd.read_csv(CROSSWALK)
    cw = cw[cw["source"] == "pbs_table1"]
    cw_key = {(fold(r.district.replace(" PROTECTED AREA", "").replace(" DISTRICT", "")),
               fold(r.unit)): r for r in cw.itertuples()}
    if len(cw_key) != len(cw):
        sys.exit("crosswalk.csv: duplicate (district, unit) keys")
    cu = pd.read_csv(CENSUS_UNITS)
    cu = cu[cu["level"] == "tehsil"]
    pop23 = {(r.district, r.name): r.pop23 for r in cu.itertuples()}
    out, used = {}, set()
    for (prov, dist, teh, au), t in tot.items():
        if (dist, teh, au) in ALIAS:
            dd, uu = ALIAS[(dist, teh, au)]
            k = (fold(dd.replace(" DISTRICT", "")), fold(uu))
        else:
            k = (fold(dist), fold(f"{teh} {au.replace('_', ' ')}"))
        r = cw_key.get(k)
        if r is None:
            sys.exit(f"Table 11 unit {dist} / {teh} {au} has no crosswalk row")
        if k in used:
            sys.exit(f"two Table 11 units on crosswalk row {k}")
        used.add(k)
        p1 = pop23.get((r.district, r.unit))
        # Table 11 can only be smaller (restricted areas); the gaps must add to exactly the
        # 1,041,342 Table 1's footnote gives, checked below.
        # (Kandhkot taluka is one person larger in Table 11, a PBS slip; tolerated.)
        if p1 is None or not (t <= p1 + 5):
            sys.exit(f"{dist} / {teh}: Table 11 {t:,} against Table 1 {p1}")
        out[(dist, teh, au)] = (r.adm3, int(p1) - int(t))
    if len(used) != len(cw):
        sys.exit(f"{len(cw) - len(used)} crosswalk rows have no Table 11 unit")
    head = sorted(((gap, k) for k, (_, gap) in out.items() if gap), reverse=True)
    if sum(g for g, _ in head) != HEAD_ONLY:
        sys.exit(f"Table 1 - Table 11 = {sum(g for g, _ in head):,}, expected {HEAD_ONLY:,}")
    odd = [f"{k[1]} {k[2].lower()} {g:+,}" for g, k in head if g < 0]
    if odd:
        lc.log(f"  Table 11 larger than Table 1 in: {', '.join(odd)}")
    head = [(g, k) for g, k in head if g > 0]
    lc.log(f"  all {len(out)} units joined; Table 1 exceeds Table 11 (counted by head only) "
           f"in {len(head)} units, {sum(g for g, _ in head):,} people: "
           + ", ".join(f"{k[1]} {k[2].lower()} ({k[0].title()}) {g:,}" for g, k in head[:8]))
    return out


def north(comp, pk2023):
    n = pd.read_csv(lc.LD_NORM / "pk_north.csv")
    for lvl, area in ((3, "gilgit-baltistan"), (2, "azad-jammu-and-kashmir")):
        part = n[n["geo_id"].str.startswith(f"PK23-{area}/")]
        parent = "PK3" if area == "gilgit-baltistan" else "PK1"
        # GB's districts are helper1m's tehsil-level units too (same code, same polygon).
        feats = {fold(p["name"]): c for c, p in comp.feat[lvl].items()
                 if (p.get("group") == parent)}
        names = set(part["geo_name"])
        codes = {g: feats.get(fold(g)) for g in names}
        if None in codes.values() or len(set(codes.values())) != len(feats):
            sys.exit(f"{area}: districts {codes} against helper1m's {sorted(feats)}")
        for r in part.itertuples():
            comp.add(lvl, codes[r.geo_name], pk2023.resolve(r.source_category), r.count)
        lc.log(f"  {area}: {len(names)} districts, {int(part['count'].sum()):,} people, "
               f"at level {lvl}")


def main():
    pk2023 = lc.ld_module("pk2023")
    lang, tot, pk_t11 = table11()
    check_against_languagedots(lang, pk_t11)
    units = join_units(tot)

    comp = lc.Composition("pakistan")
    for r in lang.itertuples():
        code, _ = units[(r.DISTRICT, r.TEHSIL, r.ADMIN_UNIT)]
        comp.add(3, code, pk2023.resolve(r.LANGUAGE), r.count)
    if MODELLED_NORTH:
        north(comp, pk2023)

    comp.write(
        label="Mother tongue",
        year=2023,
        titles=TITLES,
        pop_year=2023,
        source=(
            "Pakistan 2023 census, Table 11, population by mother tongue, by tehsil / taluka / "
            "sub-division for the four provinces and Islamabad (via the PakPC2023 R package, as "
            "maps/languagedots reads it), joined to helper1m's tehsil-level units by census unit; "
            "districts and provinces summed from it. Table 11 leaves out the 1.04 million "
            "counted by head only."
            + (" Gilgit-Baltistan (per district) and Azad Kashmir (per district, no tehsil "
               "split) are languagedots' models: GB's census-wide shares split by the GB MICS "
               "surveys, and the AJK yearbook's whole-percent district estimates."
               if MODELLED_NORTH else
               " Gilgit-Baltistan and Azad Kashmir are not in Table 11 and draw no pie.")),
    )


if __name__ == "__main__":
    main()
