"""Costa Rica — boundaries for the seven provinces, and the population they are drawn on.

Writes data/geo/cr/cr_provincias.gpkg and data/geo/cr/cr_lookup.csv.

OCHA COD-AB Costa Rica (`cod-ab-cri`, valid 2024-12), the **geodatabase** rather than the
145 MB shapefile bundle, read with `engine="fiona"`. ADM1 is 7 features and LAPOP measures
all seven, so unlike Panama and Ecuador nothing here is left undrawn.

## SPEC §12'S GEODATABASE TRAP REPRODUCES EXACTLY ON THIS FILE

`cri_admin_boundaries.gdb` returns **7 features with `engine="fiona"` and ZERO with
`engine="pyogrio"`** — same file, same call, one argument apart, no exception and no warning
from the failing one. pyogrio is geopandas' default wherever it is installed, so the silent
path is the one you get without asking. The engine is named explicitly below and the feature
count is asserted rather than the absence of an exception.

## THE JOIN IS ON NAME, AND THE CODE JOIN HAPPENS TO AGREE — WHICH IS A WITNESS, NOT A METHOD

LAPOP's `prov` is 600 plus Costa Rica's official province number, and COD's `CR` pcodes run in
that same official order, so `prov - 600` -> `CRn` pairs all seven correctly. **That is the
first country in this module's set where the code join is not a trap**, and it is exactly why
it must not be used: Guatemala's code join works, El Salvador's mispairs twelve of fourteen
and Panama's would have put 2.09 million people in a comarca of 32,016, and none of the three
looks different from the outside. So the join is on the NAME, read out of the Grand Merge's
own `prov` value-label set, and `check_code_join` asserts that the code join *agrees* — if it
ever stops agreeing, that is OCHA re-cutting the pcodes and it should fail here loudly rather
than quietly re-draw the country.

The seven names, verbatim from the value labels:

    601 San José   602 Alajuela   603 Cartago   604 Heredia
    605 Guanacaste 606 Puntarenas 607 Limón

They match COD's `adm1_name` one for one and **no alias is needed**, which is unusual here and
is asserted so that a future rename fails rather than silently dropping a province.

## THE POPULATION IS INEC'S OWN 2022 ESTIMATE, NOT COD-PS, AND THE MARGIN IS NOT CLOSE

This is Ecuador's call (§9bn) and it is easier here than it was there. COD-PS ships Costa Rica
as a **2021 UNFPA projection** built on INEC's 2013 projection revision. INEC has since
replaced that revision with its **Estimación de Población y Vivienda 2022**, and the two do
not agree:

    COD-PS 2021 total 5,163,021    INEC 2022 total 5,044,197    COD-PS is +2.36%

and the disagreement is **uneven and two-directional**, which is what matters for a map that
splits a country's dots between provinces:

    Heredia    +11.24%      San José  +4.53%     Puntarenas +0.91%    Alajuela  +0.70%
    Cartago     -0.10%      Limón     -1.15%     Guanacaste -3.25%

A 14.5-point spread across seven units, with COD-PS reading Heredia an eighth too large and
Guanacaste a thirtieth too small. Ecuador switched to its own office's count on a 3.4% national
error with per-province swings of -6.9% to +2.0%; this is worse per unit, and COD-PS is the
*older* of the two vintages besides. So Costa Rica is drawn on INEC's table.

**The typed table is checked against a column of the same page that this file does not use.**
Cuadro 4.4 prints a 2011 census column and an average annual growth rate beside the 2022
estimate. Recomputing the rate from the two population columns reproduces INEC's printed rate
to within 0.02 points on all seven provinces, so a digit typed wrong in either population
column would show up as a rate that does not match. That is the whole reason the 2011 column
and the rates are carried here at all.

Usage:
    python sources/cr_geo.py --fetch    one ~63 MB geodatabase from HDX
    python sources/cr_geo.py            rebuild from data/raw/cr/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cr")
GDB_DIR = os.path.join(RAW, "gdb")
OUT_DIR = os.path.join(ROOT, "data", "geo", "cr")
OUT = os.path.join(OUT_DIR, "cr_provincias.gpkg")
LOOKUP = os.path.join(OUT_DIR, "cr_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "cri_admin_boundaries.gdb.zip":
        "https://data.humdata.org/dataset/7cf2b2c2-ad83-4f65-aa69-dae9c574df13/resource/"
        "b40636ad-6db7-4836-a0c0-b35db98a0f3e/download/cri_admin_boundaries.gdb.zip",
    # kept for the comparison in `population()` and for nothing else
    "cri_admpop_adm1_unfpa2021.csv":
        "https://data.humdata.org/dataset/2ed8eb18-6df3-4e31-b907-4902334d0bbf/resource/"
        "2219dd9f-7af9-4ff6-930a-c2f4852433cd/download/cri_admpop_adm1_unfpa2021.csv",
}

# Verbatim from the Grand Merge's own `prov` value-label set, read with pyreadstat's
# `metadataonly=True`. Not inferred from a numbering, which is the whole point — see the
# docstring, and `sources/pa_geo.py` for what inferring it costs.
LAPOP_PROVINCES = {
    601: "San José", 602: "Alajuela", 603: "Cartago", 604: "Heredia",
    605: "Guanacaste", 606: "Puntarenas", 607: "Limón",
}

# LAPOP's names and COD's are the same seven words. Asserted empty so that a rename on either
# side fails here instead of dropping a province out of the join.
ALIASES = {}

# INEC, *Estimación de Población y Vivienda 2022. Resultados generales*, CUADRO 4.4,
# "Costa Rica. Población total y tasa de crecimiento, según provincia, 2011 - 2022", page 22
# of `rePoblacResultadosGenerales_Estimacion_poblacion_vivienda_2022.pdf`.
#   pcode: (census 2011, estimate 2022, INEC's own printed average annual growth rate)
# The rate is the cross-check on the two population columns and is used for nothing else.
INEC_2022 = {
    "CR1": (1_404_242, 1_601_167, 1.19),
    "CR2": (  848_146, 1_035_464, 1.81),
    "CR3": (  490_903,   545_092, 0.95),
    "CR4": (  433_677,   479_117, 0.91),
    "CR5": (  326_953,   412_808, 2.12),
    "CR6": (  410_929,   500_166, 1.79),
    "CR7": (  386_862,   470_383, 1.78),
}
INEC_TOTAL_2022 = 5_044_197
INEC_TOTAL_2011 = 4_301_712
RATE_TOLERANCE = 0.03          # points; the printed rates are to two decimals

N_ADM1 = 7


def fold(s):
    """Accent- and case-insensitive key for a province name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=1800) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def read_adm1():
    """COD-AB ADM1, from the geodatabase, with the engine named and the count asserted."""
    zpath = os.path.join(RAW, "cri_admin_boundaries.gdb.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    if not os.path.isdir(GDB_DIR):
        os.makedirs(GDB_DIR, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(GDB_DIR)
    gdb = next(os.path.join(GDB_DIR, d) for d in os.listdir(GDB_DIR) if d.endswith(".gdb"))

    # §12: `engine="fiona"` is load-bearing. pyogrio reads this same layer as zero features,
    # reports EPSG:4326, and raises nothing.
    g = gpd.read_file(gdb, layer="cri_admin1", engine="fiona")
    if len(g) != N_ADM1:
        raise SystemExit(f"{len(g)} ADM1 features, expected {N_ADM1}. If this is 0 the "
                         "geopandas engine has changed under the call above (§12); if it is "
                         "some other number, COD has re-cut Costa Rica.")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {gdb} layer cri_admin1: {len(g)} provinces, {g.crs}")
    return g


def check_code_join(by_name):
    """`prov - 600` -> `CRn` — assert that it AGREES, because here it happens to.

    Guatemala's code join is right, El Salvador's mispairs twelve of fourteen, Panama's would
    put 2.09 million people in a comarca of 32,016, and from the outside the three are
    indistinguishable. Costa Rica's agrees; the map still joins on the name, and this asserts
    the agreement so that OCHA re-cutting the pcodes fails here rather than re-drawing the
    country quietly.
    """
    wrong = []
    for code, lname in sorted(LAPOP_PROVINCES.items()):
        naive = f"CR{code - 600}"
        actual = by_name[fold(ALIASES.get(lname, lname))]
        if naive != actual:
            wrong.append((code, lname, naive, actual))
    print(f"\n  witness 2 — `prov - 600` -> CRn would pair "
          f"{len(LAPOP_PROVINCES) - len(wrong)} of {len(LAPOP_PROVINCES)} correctly")
    if wrong:
        for code, lname, naive, actual in wrong:
            print(f"      LAPOP {code} {lname:<12} -> {naive}, the real one is {actual}")
        raise SystemExit(
            "the code join and the name join now DISAGREE. They have agreed since this "
            "country was built, so either OCHA has re-cut Costa Rica's pcodes or LAPOP has "
            "renumbered `prov`. The name join is the one that draws the map — STOP and work "
            "out which side moved before trusting either.")
    print("      they agree, which is a witness on the name join and is NOT how it is done")


def population():
    """INEC's Estimación 2022 per province, checked against the growth column beside it."""
    print("\n  witness 3 — INEC's own 2022 estimate, checked against its own printed "
          "growth rates:")
    if sum(v[1] for v in INEC_2022.values()) != INEC_TOTAL_2022:
        raise SystemExit("the typed 2022 column does not sum to INEC's printed total")
    if sum(v[0] for v in INEC_2022.values()) != INEC_TOTAL_2011:
        raise SystemExit("the typed 2011 column does not sum to the 2011 census total")
    for pc, (c11, e22, printed) in INEC_2022.items():
        got = ((e22 / c11) ** (1 / 11) - 1) * 100
        flag = "" if abs(got - printed) <= RATE_TOLERANCE else "   <-- MISMATCH"
        print(f"      {pc}  {c11:>9,} -> {e22:>9,}   {got:5.2f}%/yr vs INEC's "
              f"{printed:4.2f}%{flag}")
        if abs(got - printed) > RATE_TOLERANCE:
            raise SystemExit(
                f"{pc}: the growth rate implied by the two typed population columns is "
                f"{got:.2f}%/yr against INEC's printed {printed:.2f}%. One of the three "
                "numbers is mistyped — that is what this check is for.")
    print(f"      all seven reproduce; total {INEC_TOTAL_2022:,} in 2022 against "
          f"{INEC_TOTAL_2011:,} counted in 2011")
    return {pc: v[1] for pc, v in INEC_2022.items()}


def compare_codps(pop):
    """Why this country is not drawn on COD-PS. Reported every run, decides nothing."""
    path = os.path.join(RAW, "cri_admpop_adm1_unfpa2021.csv")
    if not os.path.exists(path):
        print("\n  (COD-PS csv absent, skipping the comparison that motivates INEC)")
        return
    cod = pd.read_csv(path, encoding="utf-8-sig")
    cod = dict(zip(cod["ADM1_PCODE"].astype(str).str.strip(), cod["T_TL"].astype(int)))
    tot = sum(cod.values())
    print(f"\n  and why NOT COD-PS (§9bn, Ecuador's call): COD-PS 2021 sums to {tot:,} "
          f"against INEC's {sum(pop.values()):,}, {tot / sum(pop.values()) - 1:+.2%},")
    print("    and the error is uneven and runs in both directions:")
    errs = sorted(((c, cod[c] / pop[c] - 1) for c in pop), key=lambda t: -t[1])
    for pc, e in errs:
        print(f"      {pc}  COD-PS {cod[pc]:>9,}  vs INEC {pop[pc]:>9,}   {e:+7.2%}")
    print(f"    spread {(errs[0][1] - errs[-1][1]) * 100:.1f} points across seven units, "
          "against Ecuador's -6.9% to +2.0% which was\n    enough to switch that country. "
          "COD-PS is also the older vintage of the two.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    g = read_adm1()
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    by_name = dict(zip(g["adm1_name"].map(fold), g["pcode"]))
    if len(by_name) != N_ADM1:
        raise SystemExit("COD's province names are not unique — the name join is unsafe")

    # ---- witness 1: every LAPOP name is a COD name ----
    missing = [n for n in LAPOP_PROVINCES.values() if fold(ALIASES.get(n, n)) not in by_name]
    if missing:
        print(f"    LAPOP names with no polygon: {missing}")
        print(f"    COD names: {sorted(g['adm1_name'])}")
        raise SystemExit("the name join FAILED")
    spare = sorted(n for n in g["adm1_name"]
                   if fold(n) not in {fold(ALIASES.get(v, v))
                                      for v in LAPOP_PROVINCES.values()})
    if spare:
        raise SystemExit(f"COD units LAPOP never names: {spare} — LAPOP covers all seven "
                         "provinces and a spare means the geography has changed")
    if ALIASES:
        raise SystemExit(f"ALIASES is no longer empty: {ALIASES}. That is fine, but the "
                         "docstring says no alias is needed — update it deliberately.")
    print(f"  witness 1 — all {len(LAPOP_PROVINCES)} LAPOP names match a COD name with no "
          "alias, and COD has no eighth unit")

    check_code_join(by_name)
    pop = population()
    compare_codps(pop)

    g["pop"] = g["pcode"].map(pop).astype("int64")
    if g["pop"].isna().any() or int(g["pop"].sum()) != INEC_TOTAL_2022:
        raise SystemExit("the population join lost a province")
    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["adm1_name"]

    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "name"]
    hi = g.loc[g["density"].idxmax(), "name"]
    print(f"\n  sparsest {lo!r}, densest {hi!r}")
    if fold(hi) != fold("San José") or fold(lo) != fold("Guanacaste"):
        raise SystemExit(f"densest is {hi!r} and sparsest {lo!r}, expected San José and "
                         "Guanacaste — the population join is permuted")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="provincias", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons, {int(out['pop'].sum()):,} people)")

    prov_of = {by_name[fold(ALIASES.get(n, n))]: c for c, n in LAPOP_PROVINCES.items()}
    lut = pd.DataFrame({
        "geo_id": sorted(g["unit"]),
        "unit": sorted(g["unit"]),
        "name": [g.loc[g["unit"] == u, "name"].iloc[0] for u in sorted(g["unit"])],
        "pop": [int(g.loc[g["unit"] == u, "pop"].iloc[0]) for u in sorted(g["unit"])],
        "lapop_prov": [prov_of[u] for u in sorted(g["unit"])],
    })
    if len(lut) != N_ADM1 or lut["lapop_prov"].nunique() != N_ADM1:
        raise SystemExit("the lookup does not carry seven distinct LAPOP codes")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, every one with a LAPOP prov code)")
    print("  nothing in Costa Rica is undrawn: LAPOP measures all seven provinces, so this "
         "country has no `gap=` of the Galápagos kind (§9bn)")


if __name__ == "__main__":
    main()
