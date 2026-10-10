"""Mali placement inside régions: CLEAR Global's cercle shares (2026-10-07, fix-place).

    python sources/ml_place.py   -> data/geo/ml/ml_hexes.gpkg   (unit, pop, cercle, zone)
                                    data/normalized/ml_clear.csv (zone, clear_code, share)

The counts stay RGPH5 2022's by région (countries/ml.py). CLEAR Global's mali-languages (HDX,
CC BY-SA; "main language spoken in the household", from the IPUMS sample of the 2009 census)
gives shares for the 2009 geography: 45 named cercles, Kidal's four as one "level 2 unknown" row
and Bamako as another (`ML09XXX`). Its pcodes are the OLD COD ones (ML0102 = Diéma), not COD-AB
v03's (ML0102 = Bafoulabé), so they are joined by NAME to geoBoundaries' 50 pre-2016 cercles
(religiondots' data/raw/ml/geoBoundaries-MLI-ADM2.geojson, read-only).

Each hex of religiondots' ml_hexes.gpkg (the 20 régions of 2022, read-only) gets the old cercle
its centroid falls in (nearest for misses), and a `zone`: that cercle's CLEAR code, except that
a cercle holding under SLIVER of a région's Kontur population becomes "" there (the région's
mean, sources/clear_place.py): new région lines do not follow old cercle lines exactly, and a
border sliver should not pull a language into itself. Régions made of one old cercle (Bamako,
Dioïla, Douentza, Kita, Ménaka, Nara...) come out on population; those made of several (Bougouni
= Bougouni, Yanfolila, Kolondiéba; Bandiagara = Bandiagara, Bankass, Koro; Nioro = Nioro, Diéma;
San = San, Tominian; Koutiala = Koutiala, Yorosso; and the old régions' remainders) are split.

CHECKS: shares sum to 1 per location; all 45 named CLEAR cercles match a geoBoundaries name and
every geoBoundaries cercle has a zone; every hex gets a cercle; population equals religiondots'
layer; every CLEAR code the labels use is in the file.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD, RD_GEO  # noqa: E402
import clear_place  # noqa: E402

RAW = HERE / "data" / "raw" / "ml"
CLEAR_URL = ("https://data.humdata.org/dataset/4243e46c-3861-49ea-b30d-8e023cbb9b09/resource/"
             "6b47b5f7-b010-495b-b119-a8b377b58bcf/download/clearglobal_language_use_mli_admin2.csv")
CLEAR2 = RAW / "clearglobal_mli_admin2.csv"
OLD_CERCLES = RD / "data" / "raw" / "ml" / "geoBoundaries-MLI-ADM2.geojson"
HEX = RD_GEO / "ml" / "ml_hexes.gpkg"
OUT = HERE / "data" / "geo" / "ml" / "ml_hexes.gpkg"
OUT_SHARES = HERE / "data" / "normalized" / "ml_clear.csv"
SLIVER = 0.05

ALIAS = {"Baraoueli": "Baroueli"}                       # CLEAR name -> geoBoundaries name
UNKNOWN_ROWS = {"ML08XXX": ["Abeibara", "Kidal", "Tessalit", "Tin-Essako"],
                "ML09XXX": ["Bamako"]}

# census label (taxonomy/ml2022.py) -> CLEAR codes. Kunabere, Mossi, the foreign languages and
# the remainders are not in CLEAR's file and go on population.
CLEAR_CODES = {
    "Bambara/Bamanankan": ["bamb1269"],
    "Malinké/Maninkakan": ["west2500"],
    "Peulh/Fulfulde": ["fula1264"],
    "Sonrhai/Songhoy/Zarma": ["zarm1241"],
    "Sarakole/Sooninke": ["soni1259"],
    "Khassonké/Xhassonkakan": ["xaso1239"],
    "Sénoufo/Syenara": ["senu1239"],
    "Dogon/Dôgôsô": ["dogo1299"],
    "Maure/Hasaniya": ["hass1238"],
    "Tamasheq": ["tama1365"],
    "Bobo/Bomu": ["bobo1253"],
    "Dafing": ["mark1256"],
    "Minianka/Mamara": ["mama1271"],
    "Haoussa": ["haus1257"],
    "Samogo/Dungooma": ["duun1245"],
    "Bozo/Tyako": ["bozo1252"],
    "Arabe": ["arab1395"],
}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def node_codes():
    """{node as drawn (after taxonomy/regroup.txt): CLEAR codes}, for countries/ml.py."""
    sys.path.insert(0, str(HERE / "taxonomy"))
    import ml2022
    from regroup import move
    out = {}
    for lab, codes in CLEAR_CODES.items():
        n = ml2022.resolve(lab)
        if n is None:
            raise SystemExit(f"ml_place: {lab!r} is not a census label")
        out.setdefault(move(n), []).extend(codes)
    return out


def main():
    if not CLEAR2.exists():
        raise SystemExit(f"download {CLEAR_URL} to {CLEAR2}")
    c2 = clear_place.read_clear(CLEAR2)
    say(c2["location_code"].nunique() == 47, f"CLEAR admin2: {c2['location_code'].nunique()} "
        "locations, shares sum to 1")
    allc = {c for cs in CLEAR_CODES.values() for c in cs}
    say(allc <= set(c2["language_code"]), "every CLEAR code the labels use is in the file")

    old = gpd.read_file(OLD_CERCLES)[["shapeName", "geometry"]]
    say(len(old) == 50 and old["shapeName"].is_unique, "geoBoundaries: 50 old cercles")
    loc = c2.groupby("location_code")["location_name"].first()
    zone_of = {ALIAS.get(n, n): c for c, n in loc.items() if not c.endswith("XXX")}
    for c, names in UNKNOWN_ROWS.items():
        say(c in loc.index, f"CLEAR has {c} ({loc.get(c)})")
        zone_of.update({n: c for n in names})
    say(set(zone_of) == set(old["shapeName"]), "CLEAR's cercles = geoBoundaries' both ways "
        f"(unmatched {sorted(set(zone_of) ^ set(old['shapeName']))})")

    hexes = gpd.read_file(HEX)
    say(hexes["unit"].nunique() == 20, "religiondots' layer: 20 régions")
    cercle, moved = clear_place.nearest_join(hexes, old, "shapeName")
    print(f"  {moved:,} of {len(hexes):,} hexes on the nearest old cercle (centroid outside all)")
    hexes["cercle"] = cercle
    hexes["zone"] = hexes["cercle"].map(zone_of)
    say(hexes["zone"].notna().all(), "every hex has a CLEAR zone")
    # Bamako région (COD-AB v03, 733 km2) runs past the old district into Kati cercle's suburbs
    # (Kontur 1.48M people there); that is the city's sprawl, so the whole région is Bamako's row
    # rather than rural-weighted Kati cercle's
    b = hexes["unit"] == "Bamako"
    print(f"  Bamako: {hexes.loc[b & (hexes['zone'] != 'ML09XXX'), 'pop'].sum():,.0f} Kontur "
          "people outside the old district set to Bamako's row")
    hexes.loc[b, "zone"] = "ML09XXX"
    share = hexes.groupby(["unit", "zone"])["pop"].sum()
    share = share / share.groupby(level=0).transform("sum")
    small = share[share < SLIVER]
    key = pd.MultiIndex.from_arrays([hexes["unit"], hexes["zone"]])
    hexes.loc[key.isin(small.index), "zone"] = ""
    print(f"  {len(small)} (région, cercle) slivers under {SLIVER:.0%} of their région's "
          f"population set to the région's mean ({hexes.loc[hexes['zone'] == '', 'pop'].sum():,.0f} "
          "Kontur people)")
    z = hexes.groupby(["unit", "zone"])["pop"].sum()
    print("    " + z.round().astype(int).to_string().replace("\n", "\n    "))
    split = sorted(u for u, g in hexes.groupby("unit") if g.loc[g["zone"] != "", "zone"].nunique() > 1)
    print(f"  {len(split)} régions split among old cercles by CLEAR: {', '.join(split)}; the "
          "rest are one cercle and stay on population")

    rd = gpd.read_file(HEX, ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1 and len(hexes) == len(rd),
        f"population {hexes['pop'].sum():,.0f} and {len(hexes):,} hexes = religiondots'")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "cercle", "zone", "geometry"]].to_file(OUT, driver="GPKG")
    sh = c2.rename(columns={"location_code": "zone", "language_code": "clear_code",
                            "language_name": "clear_name", "proportion_value": "share"})
    sh[["zone", "clear_code", "clear_name", "share"]].to_csv(OUT_SHARES, index=False, encoding="utf-8")
    print(f"wrote {OUT} and {OUT_SHARES}")


if __name__ == "__main__":
    main()
