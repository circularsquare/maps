"""Spain — placement polygons: the 8,131 municipios, straight out of the GISCO LAU file.

Writes data/geo/es/es_municipios.gpkg with, per municipio:
    muni     5-digit INE municipal code, zero-padded
    unit     its province, which is the first two digits of that code
    pop      GISCO's POP_2021
    spanish  Spanish nationals resident there    ] INE Padron table 33571, and the
    foreign  foreign nationals resident there    ] reason this file has a --fetch
    name     LAU_NAME

Usage:
    python sources/es_geo.py --fetch   # INE 33571, municipio x nationality (streamed)
    python sources/es_geo.py

TWO PLACEMENT WEIGHTS, NOT ONE — §9as's finding, applied to the country it matters most in.
Italy's map put its foreign-half religions across the countryside in proportion to where
ITALIANS live, because both halves shared one population weight; ISTAT's comune-level
citizenship counts fixed it. **Spain has the same defect and a worse case of it**: CIS does
not sample foreigners at all, so the foreign half here is not a correction to the survey but
the other **15% of the country** — 7.4M people, against Italy's 8.5% — and all of it was
being scattered by where Spaniards live.

INE table 33571 is `Poblacion por sexo, municipios, nacionalidad (espanol/extranjero) y edad`,
which is exactly the split needed and at exactly this file's join key: INE's own five-digit
municipal code, the one GISCO carries verbatim as LAU_ID, so there is no crosswalk here
either. **Placement only, never a magnitude** (spec §8.2) — every province's totals stay
sources/es.py's.

THE FILE IS 354 MB AND IS NEVER STORED. It is one table crossed with three sexes, three
nationalities, four age groups and twenty years, and the one slice wanted is about 0.4% of
it. `fetch()` streams the HTTP response line by line and keeps only the matching rows, so
what lands on disk is a few hundred KB. The server also IGNORES Range requests — a
`-r 0-2000` comes back with the whole 50 MB gzip stream and a Content-Length that looks like
a small file — so probing its size before downloading it does not work.

WHY THIS FILE IS SHORT. §9e wrote that the GISCO LAU 2021 file is the boundary answer for
most of Europe and every country since has paid something for it; §9v found Portugal to be
the case where that is true without qualification. Spain is the second such case, and it is
better than Portugal's in one respect: **the counting geography is DERIVABLE from the
placement geography with no join at all.** INE's municipal code is province + municipality,
five digits, and GISCO carries it verbatim as LAU_ID. So `unit = muni[:2]` is the whole of
the province assignment — no spatial join, no name matching, no crosswalk, and therefore
none of §8.1's three failure modes can occur on this boundary.

THE ONE THING THAT WOULD GO WRONG QUIETLY is zero-padding. GISCO stores LAU_ID as text and
Spain's first nine provinces are 01-09, so a file read that drops a leading zero turns
Álava's 01059 into 1059 and its province into "10" — Cáceres, at the other end of the
country. Every read here forces a 5-character string and the province count is asserted at
52, which is the cheap test that would catch it.

POP_2021 IS A PLACEMENT WEIGHT AND NOT A POPULATION SOURCE. It sums to 47.39M against the
49.80M INE reports for 1 July 2026, because it is five years old — Spain grew by 2.4M in
between, almost all of it foreign immigration. That does not matter for what it is used for:
`place_weight` only needs the RELATIVE size of municipios inside a province, and the
magnitudes all come from sources/es.py. It would matter if it were ever summed into a count,
so it is deliberately not named `population` in the output.
"""

import argparse
import csv
import os
import ssl
import sys
import urllib.request

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")
RAW = os.path.join(ROOT, "data", "raw", "es")
NAT = os.path.join(RAW, "ine_33571_nationality.csv")
OUT_DIR = os.path.join(ROOT, "data", "geo", "es")
OUT = os.path.join(OUT_DIR, "es_municipios.gpkg")

# INE table 33571, "csv_bd" flavour. 354 MB; see the docstring for why it is streamed.
INE_URL = "https://www.ine.es/jaxiT3/files/t/es/csv_bd/33571.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}
KEEP_SEX = "Total"
KEEP_AGE = "Todas las edades"
PERIOD = "1 de enero de 2022"      # the most recent the table carries

# EVERY VALUE COMPARED AGAINST THIS FILE IS ASCII, DELIBERATELY. The obvious filter is
# `nacionalidad in ("Espanola", "Extranjera")` with the tilde, and it kept **zero of
# 5,857,920 rows** while looking correct: `Extranjera` matched and the accented one did not,
# so the failure was silent and half-selective rather than loud. Rather than chase whose
# encoding was wrong — the CSV is ISO-8859-15, the source file is UTF-8, and one of the two
# round-trips badly — take `Total` and `Extranjera`, which are pure ASCII, and get the
# Spanish count by subtraction. It is the same number and it cannot rot.
KEEP_NAT = ("Total", "Extranjera")

# The 52: 50 provinces plus Ceuta (51) and Melilla (52). Held here so the assert below is
# about a named expectation rather than about a number.
N_PROVINCES = 52


def fetch(src=None):
    """Stream INE 33571 and keep the ~0.4% of it that is one sex, one age band, one year.

    `src` reads an already-downloaded copy of the same CSV instead of the network. The
    download is 354 MB over a server that is slow and stalls — one run took twelve minutes
    and a second sat at zero bytes for twenty — and re-pulling it to re-run one filter is
    both wasteful and a way to lose an afternoon. The bytes are identical either way; the
    national-total assert below is what says so.
    """
    import certifi
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(NAT):
        print(f"  already on disk ({os.path.getsize(NAT):,} bytes)")
        return
    if src:
        print(f"  reading {src} instead of {INE_URL}")
        stream = open(src, "rb")
    else:
        ctx = ssl.create_default_context(cafile=certifi.where())
        stream = urllib.request.urlopen(
            urllib.request.Request(INE_URL, headers=UA), timeout=1800, context=ctx)
    kept = seen = 0
    with stream as r, open(NAT, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter=";")
        w.writerow(["muni", "nacionalidad", "total"])
        header = True
        for raw in r:
            if header:
                header = False
                continue
            seen += 1
            line = raw.decode("ISO-8859-15", "replace").rstrip("\r\n")
            p = line.split(";")
            if len(p) < 6:
                continue
            sexo, muni, nac, edad, per, tot = p[0], p[1], p[2], p[3], p[4], p[5]
            if sexo != KEEP_SEX or edad != KEEP_AGE or per != PERIOD:
                continue
            if nac not in KEEP_NAT:
                continue
            code = muni.split(" ", 1)[0].strip()
            if not (len(code) == 5 and code.isdigit()):
                continue        # "Total Nacional" and any other aggregate row
            w.writerow([code, nac, tot.replace(".", "").strip()])
            kept += 1
    print(f"  streamed {seen:,} rows, kept {kept:,} -> {os.path.basename(NAT)} "
          f"({os.path.getsize(NAT):,} bytes)")


def _nationality():
    """muni -> (Spanish nationals, foreign nationals)."""
    if not os.path.exists(NAT):
        sys.exit(f"missing {NAT} — run: python sources/es_geo.py --fetch")
    df = pd.read_csv(NAT, sep=";", dtype={"muni": str})
    df["muni"] = df["muni"].str.zfill(5)
    df["n"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
    tot = df[df["nacionalidad"] == "Total"].groupby("muni")["n"].sum()
    fo = df[df["nacionalidad"] == "Extranjera"].groupby("muni")["n"].sum()
    fo = fo.reindex(tot.index).fillna(0.0)
    sp = (tot - fo).clip(lower=0.0)     # see KEEP_NAT: Spanish nationals by subtraction
    total = sp.sum() + fo.sum()
    print(f"  {len(df):,} rows, {df['muni'].nunique():,} municipios")
    print(f"  {sp.sum():,.0f} Spanish + {fo.sum():,.0f} foreign = {total:,.0f} "
          f"({100 * fo.sum() / total:.2f}% foreign)")
    if not 44_000_000 < total < 50_000_000:
        sys.exit(f"!! INE nationality rows sum to {total:,.0f}, which is not Spain — the "
                 f"filter in fetch() is wrong")
    return sp, fo


def main():
    if not os.path.exists(LAU):
        sys.exit(f"missing {LAU} — the GISCO LAU 2021 shapefile is a shared asset, see "
                 f"sources/pl_geo.py for where it comes from")

    print(f"reading {os.path.basename(LAU)} (ES only)…")
    g = gpd.read_file(LAU, where="CNTR_CODE='ES'")
    print(f"  {len(g):,} Spanish LAUs, crs={g.crs}")

    g["muni"] = g["LAU_ID"].astype(str).str.strip().str.zfill(5)
    g["unit"] = g["muni"].str[:2]
    g["pop"] = g["POP_2021"].fillna(0).astype(float)
    g["name"] = g["LAU_NAME"].astype(str)

    bad = g[~g["muni"].str.fullmatch(r"\d{5}")]
    if len(bad):
        sys.exit(f"!! {len(bad)} LAU_IDs are not 5 digits: "
                 f"{bad['LAU_ID'].head().tolist()}")

    provs = sorted(g["unit"].unique())
    print(f"  {len(provs)} provinces: {provs[0]}…{provs[-1]}")
    if len(provs) != N_PROVINCES:
        sys.exit(f"!! expected {N_PROVINCES} provinces, got {len(provs)}: {provs}")
    if provs != [f"{i:02d}" for i in range(1, N_PROVINCES + 1)]:
        sys.exit(f"!! province codes are not 01..{N_PROVINCES:02d}: {provs}")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]

    dup = g["muni"].duplicated().sum()
    if dup:
        sys.exit(f"!! {dup} duplicate municipal codes")

    print(f"  POP_2021 total {g['pop'].sum():,.0f} "
          f"(a placement weight, not a count — see the docstring)")
    zero = int((g["pop"] <= 0).sum())
    if zero:
        print(f"  {zero} municipios have no POP_2021; they still get polygons and will "
              f"take dots only if a province has no populated municipio at all")

    print("reading INE nationality by municipio…")
    sp, fo = _nationality()
    # Both directions (§8.1). INE's 2022 municipal list against GISCO's 2021 one: Spain
    # merges and splits few municipios, so a large mismatch either way means the wrong file.
    only_lau = sorted(set(g["muni"]) - set(sp.index))
    only_ine = sorted(set(sp.index) - set(g["muni"]))
    print(f"  join: {len(set(g['muni']) & set(sp.index)):,} matched, "
          f"{len(only_lau)} LAU-only, {len(only_ine)} INE-only")
    g["spanish"] = g["muni"].map(sp).fillna(0.0)
    g["foreign"] = g["muni"].map(fo).fillna(0.0)
    if only_lau:
        print(f"  !! {len(only_lau)} municipios have no INE row and keep the population "
              f"weight: {only_lau[:6]}")
    share = 100 * g["foreign"].sum() / (g["spanish"].sum() + g["foreign"].sum())
    print(f"  foreign share of the joined municipios: {share:.2f}%")
    if not 8.0 < share < 20.0:
        sys.exit(f"!! foreign share {share:.2f}% is outside the plausible band")
    top = g.sort_values("foreign", ascending=False).head(5)
    print("  most foreign residents:")
    for _, r in top.iterrows():
        print(f"    {r['name'][:28]:<30} {r['foreign']:>9,.0f}  "
              f"{100 * r['foreign'] / max(r['spanish'] + r['foreign'], 1):5.1f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["muni", "unit", "pop", "spanish", "foreign", "name",
             "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="municipios")
    print(f"wrote {OUT}  ({len(out):,} municipios)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--from", dest="src", default=None,
                    help="an already-downloaded copy of INE 33571's CSV")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.src)
    else:
        main()
