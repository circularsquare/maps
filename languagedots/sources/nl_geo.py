"""Netherlands: the placement layer. Kontur 400 m hexes keyed to the 342 gemeenten of 2026 (the
counting unit) and to the postcode-4 area (PC4) each hex's centroid falls in (placement only).

    python sources/nl_geo.py --fetch     PDOK polygons + CBS 85640NED into data/raw/nl/
    python sources/nl_geo.py             writes data/geo/nl/nl_hexes.gpkg

Sources (open, no key):
  * PDOK gebiedsindelingen 2026, gemeente_gegeneraliseerd (CBS's own boundaries; the year is a
    path component of the WFS URL).
  * PDOK CBS postcode4 2024 (the newest PC4 polygons PDOK serves; 2025 and 2026 404).
  * CBS 85640NED, Bevolking; geslacht, herkomstland, geboorteland, PC4, 1 januari 2026: people
    born in and outside the Netherlands by 22 origin groups per PC4.

Each hex carries `pop` (Kontur) and one placement weight per origin group: the PC4's CBS count
spread over the PC4's hexes by Kontur population (w_<group>). `w_native` is people born in the
Netherlands of Dutch origin, `w_nlborn` everyone born in the Netherlands. Hexes in a PC4 that the
2026 table has and the 2024 polygons lack, or vice versa, fall back to Kontur population inside
the gemeente (countries/nl.py). AGENT_BRIEF §4.4: the counts stay CBS's per gemeente; PC4 only
decides where in the gemeente a language's dots go.
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402
import requests  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "nl")
OUT = os.path.join(ROOT, "data", "geo", "nl", "nl_hexes.gpkg")
GEM = os.path.join(RAW, "gemeente_2026.geojson")
PC4 = os.path.join(RAW, "postcode4_2024.geojson")
PC4_CSV = os.path.join(RAW, "cbs_85640_pc4_2026.csv")
GEM_CSV = os.path.join(RAW, "cbs_85458_gem_2026.csv")
H = {"User-Agent": "Mozilla/5.0"}
RD = 28992

# CBS 85640NED origin keys -> weight column. Born abroad, a partition of everyone born abroad
# (checked in build()): Europe = BE + DE + PL + other Europe; outside Europe = the eight below.
GROUPS = {"H008552": "BEL", "H008592": "DEU", "H008718": "POL", "H008800": "EUO",
          "H008632": "IDN", "H008673": "MAR", "H007119": "NCAR", "H008751": "SUR",
          "H008766": "TUR", "H008860": "AFR", "H008861": "AMO", "H008862": "ASI"}


def _wfs(url, typename, out):
    feats, start = [], 0
    while True:
        p = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
             "typeNames": typename, "outputFormat": "application/json", "count": 1000,
             "startIndex": start, "srsName": f"EPSG:{RD}"}
        r = requests.get(url, params=p, headers=H, timeout=300)
        r.raise_for_status()
        f = r.json()["features"]
        feats += f
        print(f"  {typename}: {len(feats)}")
        if len(f) < 1000:
            break
        start += 1000
    import json
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats,
                   "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{RD}"}}},
                  fh)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _wfs("https://service.pdok.nl/cbs/gebiedsindelingen/2026/wfs/v1_0",
         "gebiedsindelingen:gemeente_gegeneraliseerd", GEM)
    _wfs("https://service.pdok.nl/cbs/postcode4/2024/wfs/v1_0", "postcode4:postcode4", PC4)
    base = "https://opendata.cbs.nl/ODataApi/odata/85640NED/TypedDataSet"
    rows = []
    keys = ["T001040", "1012600"] + list(GROUPS)
    for k in keys:
        for gb in ("A051735", "A051736"):
            f = (f"Perioden eq '2026JJ00' and Geslacht eq 'T001038' and Herkomstland eq '{k}' "
                 f"and Geboorteland eq '{gb}'")
            for attempt in range(4):
                try:
                    r = requests.get(base, params={"$filter": f, "$format": "json"},
                                     headers=H, timeout=180)
                    r.raise_for_status()
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    print("  retry", e)
                    time.sleep(5)
            v = r.json()["value"]
            rows += v
            print(f"  85640 {k} {gb}: {len(v)}")
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    df.to_csv(PC4_CSV, index=False, encoding="utf-8")


def build():
    import geopandas as gpd
    sys.path.insert(0, HERE)
    from _grid import hex_layer

    gem = gpd.read_file(GEM)
    code_col = next(c for c in gem.columns if c.lower() in ("statcode", "code"))
    gem["unit"] = gem[code_col].astype(str).str.strip()
    gem = gem.set_crs(RD, allow_override=True)
    cbs = pd.read_csv(GEM_CSV)
    tot = cbs[(cbs.Herkomstland == "T001040") & (cbs.Geboorteland == "T001638")].dropna(
        subset=["Bevolking_1"]).set_index("RegioS")["Bevolking_1"]
    a, b = set(gem["unit"]), set(tot.index)
    print(f"gemeenten: PDOK 2026 {len(a)}, CBS 2026 {len(b)}; only PDOK {sorted(a - b)}, "
          f"only CBS {sorted(b - a)}")
    if a != b or len(a) != 342:
        raise SystemExit("gemeente join failed")
    layer = hex_layer("nl", gem[["unit", "geometry"]], census=tot.to_dict(), out=OUT)

    # PC4 of each hex, by centroid
    pc = gpd.read_file(PC4).set_crs(RD, allow_override=True)
    pc["pc4"] = "PC" + pc["postcode"].astype(int).astype(str).str.zfill(4)
    pts = gpd.GeoDataFrame(geometry=layer.to_crs(RD).geometry.centroid, crs=RD)
    j = gpd.sjoin(pts, pc[["pc4", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    layer["pc4"] = j["pc4"].fillna("").to_numpy()
    print(f"  hexes in no PC4: {(layer.pc4 == '').sum():,} "
          f"({layer.loc[layer.pc4 == '', 'pop'].sum():,.0f} people)")

    t = pd.read_csv(PC4_CSV)
    t = t[t.Postcode.str.match(r"^PC\d{4}$") & (t.Postcode != "PC0999")]
    t["Bevolking_1"] = t["Bevolking_1"].fillna(0)
    piv = t.pivot_table(index="Postcode", columns=["Herkomstland", "Geboorteland"],
                        values="Bevolking_1", aggfunc="sum").fillna(0)
    # the partition check, nationally
    abroad_all = piv[("T001040", "A051736")].sum()
    parts = sum(piv[(k, "A051736")].sum() for k in GROUPS)
    print(f"  PC4 born abroad {abroad_all:,.0f}; the 12 groups sum to {parts:,.0f}")
    cols = {"w_native": piv[("1012600", "A051735")],
            "w_nlborn": piv[("T001040", "A051735")]}
    for k, g in GROUPS.items():
        cols[f"w_{g}"] = piv[(k, "A051736")]
    cols = pd.DataFrame(cols)
    kpop = layer.groupby("pc4")["pop"].sum()
    have = layer["pc4"].isin(cols.index) & (layer["pc4"] != "")
    print(f"  hexes whose PC4 is in the 2026 table: {have.mean():.1%} "
          f"({layer.loc[have, 'pop'].sum() / layer['pop'].sum():.1%} of Kontur population)")
    share = layer["pop"] / layer["pc4"].map(kpop).replace(0, float("nan"))
    for c in cols.columns:
        layer[c] = (layer["pc4"].map(cols[c]) * share).fillna(-1.0)
    # -1 marks "no PC4 figure": countries/nl.py then falls back to population in the gemeente
    layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(layer):,} hexes)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
