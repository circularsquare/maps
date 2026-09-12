"""Angola — boundaries for the 326 municipalities of Lei 14/24.

Writes data/geo/ao/ao_municipalities.gpkg and data/geo/ao/ao_lookup.csv.

**NO CANONICAL BOUNDARY SET EXISTS FOR THE TIER THE CENSUS IS PUBLISHED ON.** Lei 14/24 of
5 September 2024 replaced Angola's 18 provinces and 164 municipalities with 21 and 326, and
the 2024 census was retabulated onto the new division before publication. Everything the
usual sources hold is the old one: OCHA's COD-AB is the 2018 vintage (18 / 161 / 539),
geoBoundaries ADM2 represents 2006, and OpenStreetMap has admin_level 6 for Luanda's nine
old municipalities, a dozen in Huíla and Malanje, and nothing else in the country.

**COD-AB'S COMMUNE LAYER CANNOT BE DISSOLVED INTO THE NEW MUNICIPALITIES EITHER**, which
was the obvious way in and is worth writing down so nobody spends the afternoon on it
again. Its ADM3 attributes are wrong where it matters most: Luanda's `Belas` contains
`Viana`, `Viana Sede`, `Kilamba Kiaxi` and `Mussulo`; `Cazenga` contains `Kikolo`, which is
in Cacuaco; Bengo's `Dande` contains `Funda`, which is in Luanda; and Uíge municipality has
one commune where the census counts several. Its spellings are its own as well (`Kikabo`
for Quicabo, `Muxiluando` for Muxaluando), and only 282 of the law's 538 leaf units match
one of its 539 communes by name.

So the polygons come from **an ArcGIS Online feature service, `Nova Divisão Administrativa
de Angola`, item 4233a339ad9d482c83f617660dea2303, digitised from Diário da República I
série n.º 171 of 5 September 2024** -- which is Lei 14/24 itself, whose articles describe
every provincial, municipal and communal boundary as a named sequence of rivers, roads and
watersheds. It is one person's work rather than an official release and is treated as such:

  * its 326 polygons carry INE's `Cod_Prov` / `Cod_Munic` and its **municipality list
    matches the law's, province by province and name by name, on all 326** -- checked
    against `taxonomy`-independent parsing of the statute in `AO_LAW` below;
  * its per-province counts match Quadro 9 of the national census volume (10 for Cabinda,
    23 for Uíge, 16 for Luanda, 7 for Icolo e Bengo, and so on);
  * its outline is compared against **Natural Earth 10m**, which is nobody's derivative of
    it, on both area and bounding box.

Two names carry a parenthesised alias the census drops (`Boa Entrada (Cadá)`,
`Gangula (Kuvu)`); `fold()` removes the bracket, so no alias table is needed.

THE NAME IS TAKEN FROM THE CENSUS AND NOT FROM THE BOUNDARY FILE (§12, Chile).

Usage:
    python sources/ao_geo.py --fetch    326 + 21 polygons over the REST API, ~90 MB
    python sources/ao_geo.py            rebuild from data/raw/ao/
"""

import json
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ao_pdfs as P                                              # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "ao")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ao")
OUT = os.path.join(OUT_DIR, "ao_municipalities.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ao_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ao.csv")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")

SERVICE = ("https://services.arcgis.com/uHAHKfH1Z5ye1Oe0/arcgis/rest/services/"
           "Nova_Divisao_Administrativa_de_Angola_WFL1/FeatureServer")
LAYERS = {0: "ao_new_municipalities.geojson", 1: "ao_new_provinces.geojson"}
EXPECTED = 326

# CIA/INE land area. The check is an ORDER-OF-MAGNITUDE one on a third-party digitisation,
# not an assertion that it is right to the hectare.
AREA_KM2 = 1_246_700
AREA_TOLERANCE = 0.05


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)", " ", s)                # `Boa Entrada (Cadá)` -> `Boa Entrada`
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0"}
    for lyr, name in LAYERS.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"have {name}")
            continue
        url = f"{SERVICE}/{lyr}/query"
        r = requests.get(url, params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
                         headers=ua, timeout=180)
        r.raise_for_status()
        oids = r.json()["objectIds"]
        print(f"layer {lyr}: {len(oids)} features")
        feats = []
        # The service caps a response at 2000 records but the real limit here is SIZE:
        # these polygons carry the statute's river courses at full detail, so the whole
        # layer is 66 MB and asking for it in one request times out. 25 at a time.
        for i in range(0, len(oids), 25):
            batch = oids[i:i + 25]
            r = requests.get(url, headers=ua, timeout=600, params={
                "objectIds": ",".join(str(o) for o in batch),
                "outFields": "*", "outSR": 4326, "f": "geojson"})
            r.raise_for_status()
            feats.extend(r.json().get("features", []))
            print(f"  {min(i + 25, len(oids))}/{len(oids)}")
        if len(feats) != len(oids):
            raise SystemExit(f"layer {lyr}: asked for {len(oids)}, got {len(feats)}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh,
                      ensure_ascii=False)
        print(f"  wrote {dest} ({os.path.getsize(dest):,} bytes)")


# ------------------------------------------------------------------ the statute


AO_LAW = os.path.join(RAW, "lei_14_24.json")


def law():
    """Lei 14/24's own province -> municipalities list, if it has been saved beside the PDFs.

    Optional. When present it is the strongest check there is on the boundary file, because
    it is the text the boundary file was drawn from and it was parsed here independently.
    `sources/ao.md` says where the copy came from.
    """
    if not os.path.exists(AO_LAW):
        return None
    with open(AO_LAW, encoding="utf-8") as fh:
        return json.load(fh)["provinces"]


# ------------------------------------------------------------------ build


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    src = os.path.join(RAW, LAYERS[0])
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    g = gpd.read_file(src)
    if len(g) != EXPECTED:
        raise SystemExit(f"{src}: {len(g)} polygons, expected {EXPECTED}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    bad = int((~g.is_valid).sum())
    if bad:
        print(f"  repairing {bad} invalid geometries with buffer(0)")
        g["geometry"] = g.geometry.buffer(0)
        if (~g.is_valid).any():
            raise SystemExit("geometries still invalid after buffer(0)")

    # ---- 1. the layer against Quadro 9 of the census -------------------------
    per = g.groupby("Nome_Prov").size().to_dict()
    want = {fold(k): v for k, v in P.MUNICIPALITIES.items()}
    off = {p: (n, want.get(fold(p))) for p, n in per.items() if want.get(fold(p)) != n}
    if off or len(per) != 21:
        raise SystemExit(f"per-province municipality counts disagree with Quadro 9: {off} "
                         f"({len(per)} provinces)")
    print(f"  21 provinces, per-province counts match Quadro 9 of the census")

    # ---- 2. the layer against Lei 14/24, if the statute is on disk -----------
    lw = law()
    if lw:
        lay = {}
        for _, r in g.iterrows():
            lay.setdefault(fold(r["Nome_Prov"]), set()).add(fold(r["Nome_Munic"]))
        diffs = []
        for prov, munis in lw.items():
            a, b = {fold(m) for m in munis}, lay.get(fold(prov), set())
            if a != b:
                diffs.append((prov, sorted(a - b), sorted(b - a)))
        if diffs:
            for prov, only_law, only_layer in diffs:
                print(f"    {prov}: in the statute only {only_law}, "
                      f"in the layer only {only_layer}")
            raise SystemExit("the boundary file is not Lei 14/24's municipality list")
        print(f"  matches Lei 14/24's own list, all {sum(len(v) for v in lw.values())}")
    else:
        print(f"  (no {os.path.basename(AO_LAW)} on disk; the statute check is skipped)")

    # ---- 3. the outline against Natural Earth -------------------------------
    area = g.to_crs("+proj=cea").area.sum() / 1e6
    slip = area / AREA_KM2 - 1
    print(f"  area {area:,.0f} km2 against Angola's {AREA_KM2:,} ({slip:+.1%})")
    if abs(slip) > AREA_TOLERANCE:
        raise SystemExit("the digitised outline is not Angola's area")
    if os.path.exists(NE):
        ne = gpd.read_file(NE)
        col = "ISO_A3" if "ISO_A3" in ne.columns else "ADM0_A3"
        ao = ne[ne[col] == "AGO"]
        if len(ao):
            a, b = ao.total_bounds, g.total_bounds
            drift = max(abs(a[i] - b[i]) for i in range(4))
            print(f"  bounding box within {drift:.3f} degrees of Natural Earth 10m")
            if drift > 0.35:
                raise SystemExit(f"outline is {drift:.2f} degrees from Natural Earth's; "
                                 f"NE {list(a)} vs layer {list(b)}")
    else:
        print(f"  (no {os.path.basename(NE)}; the Natural Earth check is skipped)")

    # ---- 4. the join to the census ------------------------------------------
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/ao.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    cen = df.drop_duplicates("geo_id")[["geo_id", "geo_name", "note"]].copy()
    cen["prov"] = cen["note"].str.extract(r"province=([^;]+)")
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census municipalities, expected {EXPECTED}")

    poly = {}
    for i, r in g.iterrows():
        key = (fold(r["Nome_Prov"]), fold(r["Nome_Munic"]))
        if key in poly:
            raise SystemExit(f"two polygons named {r['Nome_Munic']!r} in {r['Nome_Prov']}")
        poly[key] = i

    pairs, missing = {}, []
    for _, r in cen.iterrows():
        key = (fold(r["prov"]), fold(r["geo_name"]))
        if key in poly:
            pairs[r["geo_id"]] = poly[key]
        else:
            missing.append((r["geo_id"], r["prov"], r["geo_name"]))
    spare = [i for i in poly.values() if i not in set(pairs.values())]

    print("\n  the join, both ways (§12):")
    print(f"    census municipalities      {len(cen):>4}")
    print(f"    polygons                   {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for gid, prov, nm in missing:
        print(f"      no polygon: {gid} {prov} {nm!r}")
    for i in spare:
        print(f"      no census : {g.at[i, 'Nome_Prov']} {g.at[i, 'Nome_Munic']!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # the layer's own province code against ao.py's positional one -- two different
    # derivations of the same order, so agreement is worth something
    import ao
    seen = {}
    for gid, i in pairs.items():
        seen.setdefault(int(gid[2:4]), set()).add(str(g.at[i, "Cod_Prov"]))
    multi = {k: v for k, v in seen.items() if len(v) != 1}
    if multi:
        raise SystemExit(f"a census province maps to several Cod_Prov: {multi}")
    print(f"    each census province maps to exactly one Cod_Prov")

    idx = {i: gid for gid, i in pairs.items()}
    out = g.loc[sorted(idx)].copy()
    out["unit"] = [idx[i] for i in out.index]
    names = dict(zip(cen["geo_id"], cen["geo_name"]))
    provs = dict(zip(cen["geo_id"], cen["prov"]))
    out["name"] = out["unit"].map(names)
    out["province"] = out["unit"].map(provs)
    out["cod_munic"] = (out["Cod_Prov"].astype(str) + "-" + out["Cod_Munic"].astype(str))

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "province", "cod_munic", "geometry"]].to_file(
        OUT, layer="municipalities", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
