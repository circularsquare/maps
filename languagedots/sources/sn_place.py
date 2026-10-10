"""Senegal placement inside régions: CLEAR Global's département shares (2026-10-07, fix-place).

    python sources/sn_place.py   -> data/geo/sn/sn_hexes_dept.gpkg  (unit, pop, dept, zone)
                                    data/normalized/sn_clear.csv    (zone, clear_code, share)

The counts stay RGPH-5 2023's by région (countries/sn.py). CLEAR Global's senegal-languages
(HDX, CC BY-SA; "main language spoken in the household", from the IPUMS sample of the 2013
census) gives shares for 18 of the 46 départements, a "level 2 unknown" row for six régions
(Dakar, Kédougou, Kolda, Saint-Louis, Sédhiou, Tambacounda: households IPUMS does not place in a
département), and nothing below région for Kaffrine and Matam, which IPUMS still files under
their 2008 parents. Each hex of sources/sn_geo.py's layer gets its département (COD-AB Senegal
admin2 pcode, kept inside its own région) and a `zone`:
  * the département's pcode, where CLEAR has it;
  * else its région's "level 2 unknown" code (Dakar's Guédiawaye, Pikine and Keur Massar take
    SN01XXX; Kolda's Kolda and Médina Yoro Foulah take SN07XXX);
  * else "" (Gossas, Guinguinéo, Linguère, and all of Kaffrine and Matam): the région's mean,
    i.e. population (sources/clear_place.py).
So placement changes inside Dakar, Diourbel, Fatick, Kaolack, Kolda, Louga, Thiès and
Ziguinchor; Kédougou, Saint-Louis, Sédhiou, Tambacounda, Kaffrine and Matam are one zone each
and stay on population.

CHECKS: the CLEAR file's shares sum to 1 per location; every CLEAR admin2 code is a COD-AB
pcode or a région's unknown row; every hex gets a département of its own région; the population
is sn_geo's to the person; every CLEAR code the census labels use is in the file; CLEAR's
région shares against the census's, printed (2013 household main language against 2023 language
spoken most often, so only a sanity check).
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
from rdlink import RD  # noqa: E402
import clear_place  # noqa: E402

RAW = HERE / "data" / "raw" / "sn"
CLEAR_URL = ("https://data.humdata.org/dataset/b110ae96-67b6-4e1e-bcc6-197f6852f1f3/resource/"
             "230d1048-92ff-4e80-8dc7-aa10982d09a2/download/clearglobal_language_use_sen_admin2.csv")
CLEAR2 = RAW / "clearglobal_sen_admin2.csv"
CLEAR1 = RAW / "clearglobal_sen_admin1.csv"
ADM2 = RD / "data" / "raw" / "sn" / "shp" / "sen_admin2.shp"
HEX = HERE / "data" / "geo" / "sn" / "sn_hexes.gpkg"
OUT = HERE / "data" / "geo" / "sn" / "sn_hexes_dept.gpkg"
OUT_SHARES = HERE / "data" / "normalized" / "sn_clear.csv"

ADM1 = {
    "SN01": "Dakar", "SN02": "Diourbel", "SN03": "Fatick", "SN04": "Kaffrine", "SN05": "Kaolack",
    "SN06": "Kédougou", "SN07": "Kolda", "SN08": "Louga", "SN09": "Matam", "SN10": "Saint-Louis",
    "SN11": "Sédhiou", "SN12": "Tambacounda", "SN13": "Thiès", "SN14": "Ziguinchor",
}

# census label (taxonomy/sn2023.py) -> CLEAR language codes. Labels not here (Bayot, Tourka,
# sign language, the remainders) go on population. Lebu Wolof is Wolof; Bilkire Fulani is
# Fula; West Manding, Western Maninkakan and Jahanka (Jakhanke, a Mandinka variety) are the
# Manding answers the census has only one label for.
CLEAR_CODES = {
    "Wolof": ["nucl1347", "lebu1234"],
    "Pulaar": ["pula1263", "bilk1238"],
    "Sereer": ["sere1260"],
    "Joola": ["jola1264"],
    "Màndienka": ["mand1436", "west2499", "west2500", "jaha1245"],
    "Sóninke": ["soni1259"],
    "Hasaniya (Maure)": ["hass1238"],
    "Balante": ["bala1302"],
    "Mànkaañ": ["mank1251"],
    "Mànjaku": ["mand1419"],
    "Mënik": ["bedi1235"],
    "Oniyan": ["bass1258"],
    "Guñuun": ["bain1261"],
    "Kanjad": ["bady1239"],
    "Jalunga": ["yalu1240"],
    "Womey": ["wame1240"],
    "Français": ["stan1290"],
}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def node_codes():
    """{node as drawn (after taxonomy/regroup.txt): CLEAR codes}, for countries/sn.py."""
    sys.path.insert(0, str(HERE / "taxonomy"))
    import sn2023
    from regroup import move
    out = {}
    for lab, codes in CLEAR_CODES.items():
        n = sn2023.resolve(lab)
        if n is None:
            raise SystemExit(f"sn_place: {lab!r} is not a census label")
        out.setdefault(move(n), []).extend(codes)
    return out


def main():
    if not CLEAR2.exists():
        raise SystemExit(f"download {CLEAR_URL} to {CLEAR2}")
    c2 = clear_place.read_clear(CLEAR2)
    say(True, f"CLEAR admin2: {c2['location_code'].nunique()} locations, shares sum to 1")
    adm2 = gpd.read_file(ADM2, engine="fiona")[["adm2_pcode", "adm1_pcode", "geometry"]]
    say(len(adm2) == 46 and set(adm2["adm1_pcode"]) == set(ADM1), "COD-AB: 46 départements, 14 régions")
    codes = set(c2["location_code"])
    named = codes & set(adm2["adm2_pcode"])
    unknown = {c for c in codes if c.endswith("XXX")}
    say(codes == named | unknown, f"CLEAR codes: {len(named)} départements + {len(unknown)} "
        f"région unknown rows ({sorted(codes - named - unknown)} unmatched)")
    allc = {c for cs in CLEAR_CODES.values() for c in cs}
    say(allc <= set(c2["language_code"]), "every CLEAR code the labels use is in the file")

    hexes = gpd.read_file(HEX)
    reg_pc = {v: k for k, v in ADM1.items()}
    say(set(hexes["unit"]) == set(reg_pc), "the layer's 14 régions")
    allowed = {u: set(adm2.loc[adm2["adm1_pcode"] == pc, "adm2_pcode"]) for u, pc in reg_pc.items()}
    dept, moved = clear_place.nearest_join(hexes, adm2, "adm2_pcode", allowed)
    print(f"  {moved:,} of {len(hexes):,} hexes put on the nearest département of their own "
          "région (centroid offshore or over a région line)")
    hexes["dept"] = dept
    say(hexes["dept"].str[:4].eq(hexes["unit"].map(reg_pc)).all(),
        "every hex's département lies in its région")
    say(hexes["dept"].nunique() == 46, f"all 46 départements hit ({hexes['dept'].nunique()})")

    def zone(d):
        if d in named:
            return d
        return d[:4] + "XXX" if d[:4] + "XXX" in unknown else ""
    hexes["zone"] = hexes["dept"].map(zone)
    z = hexes.groupby(["unit", "zone"])["pop"].sum()
    print("  hexes' population by région and CLEAR zone (\"\" = the région's mean):")
    print("    " + z.round().astype(int).to_string().replace("\n", "\n    "))
    split = sorted(u for u, g in hexes.groupby("unit") if g["zone"].nunique() > 1)
    print(f"  régions with more than one zone, so placed by CLEAR: {', '.join(split)}")
    say(len(split) == 8, "eight régions split")

    old = gpd.read_file(HEX, ignore_geometry=True)
    say(abs(hexes["pop"].sum() - old["pop"].sum()) < 1 and len(hexes) == len(old),
        f"population {hexes['pop'].sum():,.0f} and {len(hexes):,} hexes = sn_geo's layer")

    # sanity: CLEAR's région shares against the census's (different year and question)
    c1 = clear_place.read_clear(CLEAR1)
    name1 = {n.lower().replace("è", "e").replace("é", "e"): pc for pc, n in ADM1.items()}
    c1["unit"] = c1["location_name"].str.lower().map(name1).map(ADM1)
    cen = pd.read_csv(HERE / "data" / "normalized" / "sn.csv")
    cen = cen[cen["geo_level"] == "region"]
    cs = cen.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    cs = cs.div(cs.sum(axis=1), axis=0)
    rows = []
    for lab in ["Wolof", "Pulaar", "Sereer", "Joola", "Màndienka", "Sóninke"]:
        cc = c1[c1["language_code"].isin(CLEAR_CODES[lab])].groupby("unit")["proportion_value"].sum()
        for u, v in cc.items():
            if isinstance(u, str) and u in cs.index:
                rows.append((lab, u, round(100 * v, 1), round(100 * cs.at[u, lab], 1)))
    t = pd.DataFrame(rows, columns=["label", "région", "CLEAR 2013 %", "census 2023 %"])
    t["diff"] = t["CLEAR 2013 %"] - t["census 2023 %"]
    print("  CLEAR's région shares vs the census, largest differences:")
    print("    " + t.reindex(t["diff"].abs().sort_values(ascending=False).index).head(8)
          .to_string(index=False).replace("\n", "\n    "))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "dept", "zone", "geometry"]].to_file(OUT, driver="GPKG")
    sh = c2.rename(columns={"location_code": "zone", "language_code": "clear_code",
                            "language_name": "clear_name", "proportion_value": "share"})
    sh[["zone", "clear_code", "clear_name", "share"]].to_csv(OUT_SHARES, index=False, encoding="utf-8")
    print(f"wrote {OUT} and {OUT_SHARES}")


if __name__ == "__main__":
    main()
