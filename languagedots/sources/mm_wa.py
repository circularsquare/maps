"""Myanmar: the five townships GAD's tables skip, the Wa Self-Administered Division's Mongmao,
Pangwaun, Pangsang and Narphan and Mongla (Special Region 4).

    python sources/mm_wa.py --build    add their Kontur hexes to data/geo/mm/mm_hexes.gpkg
    python sources/mm_wa.py            print the people and languages drawn

GAD keeps no township office there (sources/mm.md, section 3), so the ethnic table is blank. The
2014 census table of the same townships exists (USCB's Age-Sex sheet, MM_AGE_SEX_2014census):
Mongmao 70,683, Pangwaun 97,097, Pangsang 105,972, Narphan 116,180, Mongla 43,068, together
433,000. Those are the people drawn, each township at its 2014 figure (not grown to 2017).

LANGUAGES. No count of language or ethnicity exists for these townships. Wa Self-Administered
Division: the World Food Programme's Wa profile (docs.wfp.org WFP-0000107118): 558,000 people,
16 minority groups, "ethnic Wa account for 70 percent of the total population, with Lahu and Shan
as the second and third largest". The split of the other 30% is a judgement (below). Mongla:
Burma Link's page on Special Region 4: "approximately 35% of the population are Wa and the rest
are of Shan, Lahu, Akha, Lisu and Loi ethnicities"; Radio Free Asia (17 Sep 2019): 13 minority
languages besides Burmese, and children mainly speak Chinese. Ethnicity is read as language, as
in the rest of this country (AGENT_BRIEF section 2), with the same `modelled` tier as AGENT_BRIEF's
"no measured source" rule allows.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import pandas as pd  # noqa: E402

GDB = ROOT / "data" / "raw" / "mm" / "burma.gdb.zip"
LAYER = "MM_GEOG_ADM3_2014_uscb_202003"
HEXES = ROOT / "data" / "geo" / "mm" / "mm_hexes.gpkg"

# unit (USCB GEO_MATCH) -> (township, 2014 census population)
TOWNSHIPS = {
    "MMR_13_12_02": ("MONGMAO", 70_683),
    "MMR_13_12_03": ("PANGWAUN", 97_097),
    "MMR_13_12_05": ("PANGSANG", 105_972),
    "MMR_13_12_06": ("NARPHAN", 116_180),
    "MMR_13_13_04": ("MONGLA", 43_068),
}
TOTAL = 433_000

ST, AA = "sinotibetan", "austroasiatic"
WA, LAHU, SHAN = f"{AA}.wa", f"{ST}.loloish.lahu", "kradai.shan"
AKHA, LISU = f"{ST}.loloish.akha", f"{ST}.loloish.lisu"
MANDARIN, PALAUNG = f"{ST}.sinitic.mandarin", f"{AA}.palaung"

# Wa SAD: Wa 70% (WFP), Lahu second and Shan third (WFP); the rest is judgement
WA_SAD_MIX = {WA: 70, LAHU: 12, SHAN: 8, AKHA: 4, MANDARIN: 3, LISU: 1, PALAUNG: 1, "other": 1}
# Mongla: Wa 35% (Burma Link), the rest Shan, Lahu, Akha, Lisu, Loi (Burma Link, order given there);
# shares judgement. Chinese is the children's common language (RFA) but no share exists, so the 5%
# for Mandarin is a floor-sized guess; "Loi" has no node here and goes on `other`.
MONGLA_MIX = {WA: 35, SHAN: 25, AKHA: 15, LAHU: 12, LISU: 5, MANDARIN: 5, "other": 3}


def counts():
    """DataFrame [unit, node, count, tier] for the five townships, 433,000 people."""
    rows = []
    for unit, (name, pop) in TOWNSHIPS.items():
        mix = MONGLA_MIX if name == "MONGLA" else WA_SAD_MIX
        t = float(sum(mix.values()))
        for node, v in mix.items():
            rows.append((unit, node, pop * v / t, "modelled"))
    df = pd.DataFrame(rows, columns=["unit", "node", "count", "tier"])
    assert abs(df["count"].sum() - TOTAL) < 1e-6, df["count"].sum()
    return df


def build_hexes():
    import geopandas as gpd
    from _grid import hex_layer
    g = gpd.read_file(f"zip://{GDB.as_posix()}!Burma.gdb", layer=LAYER)
    u = g[g["GEO_MATCH"].isin(TOWNSHIPS)].rename(columns={"GEO_MATCH": "unit"})
    assert sorted(u["ADM3_NAME"]) == sorted(v[0] for v in TOWNSHIPS.values()), sorted(u["ADM3_NAME"])
    scratch = ROOT / "data" / "geo" / "mm" / "mm_wa_hexes.tmp.gpkg"
    census = {k: v[1] for k, v in TOWNSHIPS.items()}
    new = hex_layer("mm", u[["unit", "geometry"]], census=None, out=scratch)
    per = new.groupby("unit")["pop"].sum()
    print("  Kontur people vs 2014 census:",
          ", ".join(f"{TOWNSHIPS[k][0]} {per.get(k, 0):,.0f}/{census[k]:,}" for k in TOWNSHIPS))
    missing = [TOWNSHIPS[k][0] for k in TOWNSHIPS if per.get(k, 0) <= 0]
    if missing:
        raise SystemExit(f"mm_wa: no populated hex in {missing}")
    old = gpd.read_file(HEXES, layer="hexes")
    old = old[~old["unit"].isin(TOWNSHIPS)]
    out = pd.concat([old, new[["unit", "pop", "geometry"]]], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=4326)
    tmp = HEXES.with_name("mm_hexes.tmp.gpkg")
    if tmp.exists():
        tmp.unlink()
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, HEXES)
    scratch.unlink()
    print(f"  {HEXES}: {len(old):,} + {len(new):,} = {len(out):,} hexes, {out['unit'].nunique()} units")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--build" in sys.argv:
        build_hexes()
    else:
        d = counts()
        print(f"{d['count'].sum():,.0f} people")
        print(d.groupby("node")["count"].sum().round(0).sort_values(ascending=False).to_string())
