"""Tonga — boundaries for the 156 census villages.

Writes data/geo/to/to_villages.gpkg and data/geo/to/to_lookup.csv.

OCHA COD-AB Tonga (`cod-ab-ton`), the ADM3 **village** shapefile, on §12's Chile rule. It is
the statistics department's own layer: it carries **`TDOS_VID`**, TSD's village id, beside the
OCHA pcode, and its 23 ADM2 districts and 5 ADM1 divisions are the census's own tiers.

**THE JOIN IS ON THE DISTRICT AND THE VILLAGE TOGETHER, NEVER ON THE VILLAGE ALONE.**
Village names in Tonga are not unique and the duplicates are not a spelling problem. Niuafo'ou
was evacuated after the 1946 eruption and most of its people were resettled on 'Eua, where they
named the new villages after the ones they had left: **'Esia, Sapa'ata, Fata'ulua, Mata'aho,
Mu'a, Tongamama'o and Petani each exist twice**, once in 'Eua Fo'ou and once in Niuafo'ou,
900 km apart. Kolofo'ou, Hihifo, Pangai, Houma and Eueiki repeat for ordinary reasons. A name
join would pair some of them across the country and every total would still balance.
[[reference_name_join_wrong_neighbour]]

**151 OF 156 PAIR ON THE FOLDED NAME. THE OTHER FIVE ARE WITNESSED BY OPENSTREETMAP** rather
than assumed, because four of them are a rename rather than a respelling and one is an error in
COD. Each is a `place` node whose position falls inside the polygon claimed for it:

    census                          COD                     the witness
    Nukunukumotu (Kolofo'ou)        TO1103 `Nukumotu`       a contraction; Siesia falls in it
    Pangai (Pangai, Ha'apai)        TO3101 `Lifuka`         Pangai town is ON Lifuka island,
                                                            and OSM's Pangai node is inside
                                                            TO3101 while neighbouring Hihifo
                                                            is inside TO3102, so the pair is
                                                            not simply shifted by one
    Ha'atu'a / Kolomaile ('Eua)     TO4105 `Ha'atu'a`       the census prints the two villages
                                                            as ONE row; OSM's Kolomaile node
                                                            is inside COD's Ha'atu'a, so the
                                                            polygon already holds both
    Ta'anga ('Eua Motu'a)           TO4106 `Ohonua`         **COD labels two polygons Ohonua**
                                                            and has no Ta'anga at all. OSM's
                                                            Ta'anga node falls inside TO4106
                                                            and 'Ohonua town falls inside
                                                            TO4101, which settles which is
                                                            which
    Sapa'ata (Niuafo'ou)            TO5203 `SapaataNf`      the Niuas suffix, unspaced

COD disambiguates the resettled names with its own suffixes — `'Esia Nf`, `Hihifo Ntt` — so
those are stripped inside the two Niuas districts and nowhere else.

**TEN COD POLYGONS HAVE NO CENSUS ROW AND ARE NOT UNITS.** They are uninhabited islets: six in
the Vava'u lagoon (Foeata, Vaka'eitu, Mounu, Eueiki, Mala, Fofoa), plus Onevai, 'Ataa, Fukave
and Tapana. `sources/to_grid.py` snaps any population the grid puts on them to the nearest
village rather than dropping it.

Usage:
    python sources/to_geo.py --fetch    one ~120 KB zip from HDX
    python sources/to_geo.py            rebuild from data/raw/to/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "to")
OUT_DIR = os.path.join(ROOT, "data", "geo", "to")
OUT = os.path.join(OUT_DIR, "to_villages.gpkg")
LOOKUP = os.path.join(OUT_DIR, "to_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "to.csv")

ZIP_URL = ("https://data.humdata.org/dataset/723b5226-781a-4aee-9718-37e3fbddd886/resource/"
           "0642e284-9b71-428e-abba-6d4dc5949d02/download/ton_polbnda_adm3_village.zip")
ZIP_NAME = "ton_polbnda_adm3_village.zip"

EXPECTED_POLYGONS = 166
EXPECTED_UNITS = 156

# Tonga sits at 175 W and the antimeridian is 300 km away. It does not cross, and the shapefile
# is projected on a 150 E Mercator, so a bad unprojection would land the country on the far
# side of the Pacific rather than tearing it. Either failure shows up as a span.
# [[reference_antimeridian]]
MAX_SPAN_DEG = 12.0
EXPECTED_BBOX = (-176.5, -22.5, -173.0, -15.0)

# COD's ADM2 spelling against the census's district name. Only one differs, and it is a DBF
# field truncated at ten characters: `'Eua Proper` was written `'Eua Prope`.
DISTRICT_ALIAS = {"eua motu a": "eua prope"}

# The two Niuas districts, where COD suffixes a village name to separate it from its twin on
# 'Eua. Stripped only here, so a village elsewhere ending in these letters is unaffected.
NIUAS = {"niuatoputapu": "ntt", "niuafo ou": "nf"}

# (census district, census village) -> the COD pcode, for the five that do not pair on the
# name. Every one is witnessed by an OpenStreetMap `place` node inside the polygon; see the
# module docstring for which node and why it settles the pairing.
PCODE_OVERRIDE = {
    ("kolofo ou", "nukunukumotu"):        "TO1103",
    ("pangai", "pangai"):                 "TO3101",
    ("eua motu a", "ha atu a kolomaile"): "TO4105",
    ("eua motu a", "ta anga"):            "TO4106",
    ("eua motu a", "ohonua"):             "TO4101",   # the other half of COD's duplicate
    ("niuafo ou", "sapa ata"):            "TO5203",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("‐", "-").replace("`", "'").replace("‘", "'").replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def squash(s):
    """fold() with the spaces closed up too.

    COD writes `SapaataNf` for the census's `Sapa'ata`, so the apostrophe that fold() turns
    into a space has to go as well before the two can be compared.
    """
    return fold(s).replace(" ", "")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise SystemExit(f"HDX returned something that is not a zip ({len(r.content):,} bytes)")
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def build():
    import geopandas as gpd
    import pandas as pd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"{src} is missing — run `python sources/to_geo.py --fetch`")
    g = gpd.read_file("zip://" + src)
    if len(g) != EXPECTED_POLYGONS:
        raise SystemExit(f"{len(g)} ADM3 polygons, expected {EXPECTED_POLYGONS} — "
                         "COD reissued the layer")
    g = g.to_crs(4326)
    minx, miny, maxx, maxy = g.total_bounds
    if max(maxx - minx, maxy - miny) > MAX_SPAN_DEG:
        raise SystemExit(f"bbox spans {maxx - minx:.1f} x {maxy - miny:.1f} degrees — "
                         "the unprojection tore or wrapped (see MAX_SPAN_DEG)")
    if not (EXPECTED_BBOX[0] <= minx and maxx <= EXPECTED_BBOX[2]
            and EXPECTED_BBOX[1] <= miny and maxy <= EXPECTED_BBOX[3]):
        raise SystemExit(f"bbox {minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f} is not Tonga")

    # --- index COD by (district, village), with the Niuas suffix stripped in the Niuas only.
    by_key = {}
    for idx, r in g.iterrows():
        dk = squash(r["ADM2_NAME"])
        vk = squash(r["ADM3_NAME"])
        suffix = NIUAS.get(fold(r["ADM2_NAME"]))
        if suffix and vk.endswith(suffix):
            vk = vk[: -len(suffix)]
        by_key.setdefault((dk, vk), []).append((idx, r["ADM3_PCODE"]))
    dups = {k: v for k, v in by_key.items() if len(v) > 1}

    # --- the census villages, in print order, from the normalised CSV.
    if not os.path.exists(NORM):
        raise SystemExit(f"{NORM} is missing — run `python sources/to.py` first")
    df = pd.read_csv(NORM, dtype=str, keep_default_na=False)
    seen, villages = set(), []
    for _, r in df.iterrows():
        if r["geo_id"] in seen:
            continue
        seen.add(r["geo_id"])
        _div, dist, _code = r["note"].split("|")
        villages.append((r["geo_id"], dist, r["geo_name"]))
    if len(villages) != EXPECTED_UNITS:
        raise SystemExit(f"{len(villages)} villages in to.csv, expected {EXPECTED_UNITS}")

    # --- pair them.
    claimed, rows, by_name, overridden = {}, [], 0, 0
    for geo_id, dist, vname in villages:
        dk_census, vk_census = fold(dist), fold(vname)
        pcode = PCODE_OVERRIDE.get((dk_census, vk_census))
        if pcode is not None:
            overridden += 1
            hit = g.index[g["ADM3_PCODE"] == pcode]
            if len(hit) != 1:
                raise SystemExit(f"override {geo_id} -> {pcode}: {len(hit)} polygons carry "
                                 "that pcode")
            idx = hit[0]
        else:
            dk = DISTRICT_ALIAS.get(dk_census, dk_census).replace(" ", "")
            key = (dk, vk_census.replace(" ", ""))
            cands = by_key.get(key)
            if not cands:
                raise SystemExit(f"no COD polygon for {vname!r} in {dist!r} (key {key}) — "
                                 "add it to PCODE_OVERRIDE with a witness")
            if len(cands) > 1:
                raise SystemExit(f"{vname!r} in {dist!r} matches {len(cands)} polygons "
                                 f"{[c[1] for c in cands]} — resolve it in PCODE_OVERRIDE")
            idx, pcode = cands[0]
            by_name += 1
        if pcode in claimed:
            raise SystemExit(f"{pcode} claimed by both {claimed[pcode]} and {geo_id}")
        claimed[pcode] = geo_id
        rows.append((idx, geo_id, dist, vname, pcode))

    if len(claimed) != EXPECTED_UNITS:
        raise SystemExit(f"{len(claimed)} polygons claimed, expected {EXPECTED_UNITS}")
    for k, v in dups.items():
        unresolved = [p for _, p in v if p not in claimed]
        if len(unresolved) == len(v):
            raise SystemExit(f"COD has {len(v)} polygons keyed {k} and none is claimed")
    print(f"  joined {len(rows)}/{EXPECTED_UNITS}: {by_name} on the district-qualified name, "
          f"{overridden} by witnessed override")

    # --- witness: COD files each village under a division independently of the census, so the
    #     two organisations have to agree about which division every village is in.
    div_of = {}
    for _, r in df.iterrows():
        div_of[r["geo_id"]] = r["note"].split("|")[0]
    COD_DIV_ALIAS = {"niuas": "ongo niua"}
    bad = []
    for idx, geo_id, dist, vname, pcode in rows:
        cod_div = fold(g.at[idx, "ADM1_NAME"])
        cod_div = COD_DIV_ALIAS.get(cod_div, cod_div)
        if cod_div != fold(div_of[geo_id]):
            bad.append(f"{vname} ({dist}): census {div_of[geo_id]}, COD {g.at[idx, 'ADM1_NAME']}")
    if bad:
        raise SystemExit("division witness failed:\n   " + "\n   ".join(bad))
    print(f"  division witness: all {len(rows)} villages agree with COD's own ADM1")

    out = g.loc[[i for i, *_ in rows]].copy()
    out["unit"] = [geo_id for _, geo_id, *_ in rows]
    out["village"] = [vname for *_, vname, _ in rows]
    out["district"] = [dist for _, _, dist, _, _ in rows]
    out = out[["unit", "village", "district", "ADM3_PCODE", "ADM2_PCODE", "ADM1_NAME",
               "TDOS_VID", "geometry"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"          # GDAL wants the extension; [[reference_wb_truncates]]
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, driver="GPKG", layer="villages")
    os.replace(tmp, OUT)

    with open(LOOKUP + ".part", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "village", "district", "adm3_pcode", "tdos_vid"])
        for idx, geo_id, dist, vname, pcode in rows:
            w.writerow([geo_id, geo_id, vname, dist, pcode,
                        "" if g.at[idx, "TDOS_VID"] is None else int(g.at[idx, "TDOS_VID"])])
    os.replace(LOOKUP + ".part", LOOKUP)

    unclaimed = [f"{r['ADM3_PCODE']} {r['ADM2_NAME']}/{r['ADM3_NAME']}"
                 for _, r in g.iterrows() if r["ADM3_PCODE"] not in claimed]
    print(f"wrote {OUT} ({len(out)} units)")
    print(f"wrote {LOOKUP}")
    print(f"  {len(unclaimed)} COD polygons have no census row and are not units:")
    for u in unclaimed:
        print("     ", u)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        build()
