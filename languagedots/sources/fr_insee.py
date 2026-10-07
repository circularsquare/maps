"""France: French, regional languages from regional surveys, immigrant languages by country of
birth. The record is sources/fr.md.

    python sources/fr_insee.py --fetch    download INSEE's tables into data/raw/fr/ if missing
    python sources/fr_insee.py            build data/normalized/fr.csv

France's census asks no language. Anita's rule (AGENT_BRIEF §2, 2026-10-05) for rich countries
with no language question: the national language, plus (a) regional languages from regional
language surveys, and (b) immigrant languages proxied by country of birth; every proxy row
`derived`.

TABLES (all INSEE, recensement de la population 2023 = the 2021-2025 survey cycle, exploitation
principale, through the open Melodi API, no key):
  DS_RP_TD_IMMI_AGESEX_PAYSNAISS_D_PRINC  immigrants by 46 detailed countries of birth plus five
        remainders ("other Africa"...), but only for areas of 500,000 people or more: 52
        départements, 13 régions, France.
  DS_RP_TD_IMMI_AGESEX_PAYSNAISS_R_PRINC  immigrants by 10 groups of country of birth, every
        département except Mayotte.
  DS_RP_TD_POPULATION_AGESEX_PRINC        population, the base the French remainder is cut from.
INSEE's "immigré" is a person born a foreigner abroad, so French citizens born in Algeria (the
pieds-noirs) are not in it, which is what this proxy wants.

The remainders (a group in a small département, "other Africa" in a large one) are split into
countries by Eurostat's census 2021 table cens_21ctz_r3 (citizenship by NUTS 3), read from
religiondots' raw folder (read-only): within each département, the foreign citizens of that
group's countries give the shares.

Regional languages: sources/fr_regional.py (one entry per survey, with its citation).
"""

import glob
import io
import json
import os
import sys
import urllib.request
import zipfile

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fr")
OUT = os.path.join(ROOT, "data", "normalized", "fr.csv")
RD_RAW = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "raw", "fr")
EU_CTZ = os.path.join(RD_RAW, "cens_21ctz_r3_fr.json")

YEAR = 2023
MELODI = "https://api.insee.fr/melodi/file/{ds}/{ds}_2023_CSV_FR"
DATASETS = {
    "immi_d": "DS_RP_TD_IMMI_AGESEX_PAYSNAISS_D_PRINC",
    "immi_r": "DS_RP_TD_IMMI_AGESEX_PAYSNAISS_R_PRINC",
    "pop": "DS_RP_TD_POPULATION_AGESEX_PRINC",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for key, ds in DATASETS.items():
        dest = os.path.join(RAW, f"{ds}_2023.zip")
        if os.path.exists(dest):
            print(f"  {ds}: on disk")
            continue
        req = urllib.request.Request(MELODI.format(ds=ds), headers=UA)
        blob = urllib.request.urlopen(req, timeout=600).read()
        zipfile.ZipFile(io.BytesIO(blob)).testzip()
        with open(dest + ".part", "wb") as f:
            f.write(blob)
        os.replace(dest + ".part", dest)
        print(f"  {ds}: {len(blob):,} bytes")


def read_melodi(key, geo_objects=("DEP", "REG", "FRANCE")):
    """Totals over age and sex, at the given geographic levels, with labels."""
    ds = DATASETS[key]
    with zipfile.ZipFile(os.path.join(RAW, f"{ds}_2023.zip")) as z:
        data = [n for n in z.namelist() if n.endswith("_data.csv")][0]
        meta = [n for n in z.namelist() if n.endswith("_metadata.csv")][0]
        df = pd.read_csv(z.open(data), sep=";", dtype=str)
        m = pd.read_csv(z.open(meta), sep=";", dtype=str)
    df = df[(df["AGE"] == "_T") & (df["SEX"] == "_T") & df["GEO_OBJECT"].isin(geo_objects)]
    df = df.assign(count=df["OBS_VALUE"].astype(float))
    lab = m[m["COD_VAR"] == ("AREA_COUNTRY" if "AREA_COUNTRY" in df.columns else "GEO")]
    return df, dict(zip(lab["COD_MOD"], lab["LIB_MOD"]))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        import fr_build  # noqa: F401  (the build lives beside the survey table)
        fr_build.main()
