"""Netherlands: CBS StatLine population by country of origin and birthplace, 1 Jan 2026.

    python sources/nl_cbs.py --fetch     tables into data/raw/nl/, then build
    python sources/nl_cbs.py             build data/normalized/nl.csv (sources/nl_build.py)

Tables (CBS open OData, no key, opendata.cbs.nl/ODataApi/odata/<id>):
  * 85458NED  Bevolking; herkomstland, geboorteland, leeftijd, regio, 1 januari. Every gemeente,
    64 origin categories (48 named countries, continents and CBS groups), split born in / born
    outside the Netherlands. CBS's herkomstland (2022 definition) is a person's own country of
    birth when born abroad, otherwise the mother's, else the father's.
  * 85384NED  the same population nationally with all 262 origin countries: splits the
    gemeente table's remainders ("Afrika (exclusief Marokko)" less its named countries, ...)
    into countries by the national mix of the countries the gemeente table does not name.
Both read for 2026 (1 January), totals over sex and age.
"""
import os
import sys
import time

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "nl")
API = "https://opendata.cbs.nl/ODataApi/odata/{tid}"
H = {"User-Agent": "Mozilla/5.0"}
PERIOD = "2026JJ00"
GEM_CSV = os.path.join(RAW, "cbs_85458_gem_2026.csv")
NAT_CSV = os.path.join(RAW, "cbs_85384_nat_2026.csv")
U15_CSV = os.path.join(RAW, "cbs_85458_gem_under15_2026.csv")
COROP = os.path.join(RAW, "corop_2026.geojson")
DIM_CSV =os.path.join(RAW, "cbs_{tid}_{dim}.csv")


def _get(url, params=None):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=H, timeout=120)
            r.raise_for_status()
            return r.json()["value"]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print("  retry", e)
            time.sleep(5)


def _dim(tid, dim):
    v = _get(API.format(tid=tid) + f"/{dim}?$format=json")
    df = pd.DataFrame(v)[["Key", "Title"]]
    df["Key"] = df["Key"].str.strip()
    df.to_csv(DIM_CSV.format(tid=tid, dim=dim), index=False, encoding="utf-8")
    return df


def fetch():
    os.makedirs(RAW, exist_ok=True)
    base = API.format(tid="85458NED")
    herk = _dim("85458NED", "Herkomstland")
    _dim("85458NED", "RegioS")
    rows = []
    for k in herk["Key"]:
        f = (f"Perioden eq '{PERIOD}' and Geslacht eq 'T001038' and Leeftijd eq '10000' "
             f"and Herkomstland eq '{k}' and substringof('GM',RegioS)")
        v = _get(base + "/TypedDataSet", {"$filter": f, "$format": "json",
                                          "$select": "Herkomstland,Geboorteland,RegioS,Bevolking_1"})
        rows += v
        print(f"  85458 {k}: {len(v)} rows")
    df = pd.DataFrame(rows)
    for c in ("Herkomstland", "Geboorteland", "RegioS"):
        df[c] = df[c].str.strip()
    df.to_csv(GEM_CSV, index=False, encoding="utf-8")
    print("wrote", GEM_CSV, len(df))

    base = API.format(tid="85384NED")
    _dim("85384NED", "Herkomstland")
    f = (f"Perioden eq '{PERIOD}' and Geslacht eq 'T001038' and Leeftijd eq '10000' "
         f"and BurgerlijkeStaat eq 'T001019' and GeboortelandOuders eq 'T001638'")
    v = _get(base + "/TypedDataSet", {"$filter": f, "$format": "json"})
    df = pd.DataFrame(v)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    df.to_csv(NAT_CSV, index=False, encoding="utf-8")
    print("wrote", NAT_CSV, len(df))
    fetch_extra()


def fetch_extra():
    """Under-15s per gemeente (85458NED, all origins), for the regional languages' child
    ratio, and PDOK's 2026 COROP regions, which place a gemeente in Achterhoek, Veluwe or a
    part of Limburg."""
    base = API.format(tid="85458NED")
    rows = []
    for age in ("70100", "70200", "70300"):
        f = (f"Perioden eq '{PERIOD}' and Geslacht eq 'T001038' and Leeftijd eq '{age}' "
             f"and Herkomstland eq 'T001040' and Geboorteland eq 'T001638' "
             f"and substringof('GM',RegioS)")
        rows += _get(base + "/TypedDataSet", {"$filter": f, "$format": "json",
                                              "$select": "Leeftijd,RegioS,Bevolking_1"})
    df = pd.DataFrame(rows)
    df["RegioS"] = df["RegioS"].str.strip()
    df.groupby("RegioS")["Bevolking_1"].sum(min_count=1).dropna().to_csv(U15_CSV)
    r = requests.get("https://service.pdok.nl/cbs/gebiedsindelingen/2026/wfs/v1_0",
                     params={"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                             "typeNames": "gebiedsindelingen:coropgebied_gegeneraliseerd",
                             "outputFormat": "application/json", "srsName": "EPSG:28992"},
                     headers=H, timeout=300)
    r.raise_for_status()
    with open(COROP, "w", encoding="utf-8") as fh:
        fh.write(r.text)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    sys.path.insert(0, HERE)
    import nl_build
    nl_build.main()
