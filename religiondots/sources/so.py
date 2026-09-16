"""Somalia: nobody is asked their religion. Everyone is drawn on Islam, by region, on the 2026
humanitarian planning estimate; foreign residents and refugees are sized in `gap`.

Reads data/geo/so/so_lookup.csv (`sources/so_geo.py`), data/raw/estimates/pew.zip and UN DESA's
International Migrant Stock 2024 (the copy `sources/mr.py` fetched into data/raw/mr/); writes
data/normalized/so.csv. `sources/so.md` is the record; `ask/RULINGS.md` 2026-09-15 and 2026-09-16
(Maghreb and Mauritania: draw a near-uniformly Muslim country on a compiler's figure) and ask 033
(size the foreigners who cannot be drawn) the rulings.

## NOBODY IS ASKED

No census since 1987. PESS 2014 (report and 145 variables), SHDS 2020 (five files, 1,495
variables), the Somali High Frequency Surveys 2016 and 2017 and the Somaliland Household Survey 2012
have no religion item (`sources.md` §11aq). Somalia is in no Afrobarometer or Arab Barometer round.

## EVERYONE IS DRAWN ON ISLAM

The compilers agree to the first decimal: Pew Research Center 2020, 99.833% Muslim for everyone
living in Somalia; the US State Department's 2023 report, "more than 99 percent" Sunni per the
federal Ministry of Endowments and Religious Affairs. Mauritania's construction is taken whole:
nationals on `islam`, and not Pew's residual, which here would be 27,741 people of whom 14,030 are
Pew's `other religions` and 6,660 Hindus, cells nothing explains and that Pew counts over foreigners
too. Unlike Mauritania there is no foreigner layer to draw non-Muslims from: no count of foreign
residents by region exists, so they go in `gap` (ask 033).

§14: SOMALI CHRISTIANS ARE NOT PLACED. Converts from Islam have been threatened and killed by
al-Shabaab (State Department 2023). No source gives where they live, and none should be invented;
the note names the national figure and nothing below it.

## THE GAP

UN DESA's *International Migrant Stock 2024*, Somalia as destination, mid-2024: foreign-born
residents with UNHCR's refugees added. On the planning estimate plus that figure it is an upper
bound on who is left out, by whatever part of them the estimate already holds, which nothing
measures (the construction `sources/sd.py` used). UNHCR's end-2024 count of refugees and asylum
seekers is printed beside it.

Usage:
    python sources/so.py            rebuild data/normalized/so.csv and print the gap figure
"""

import io
import json
import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

LOOKUP = os.path.join(ROOT, "data", "geo", "so", "so_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
DESA = os.path.join(ROOT, "data", "raw", "mr", "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
OUT = os.path.join(ROOT, "data", "normalized", "so.csv")
UNHCR_URL = ("https://api.unhcr.org/population/v1/population/?year=2024&coa=SOM&coo_all=true"
             "&cf_type=ISO&limit=100")

NATIONAL_2026 = 19_442_160
# Measured 2026-09-15 and asserted, so note_public and gap cannot drift from the data.
NOTE = dict(people=19_442_160, desa_2024=77_972, desa_ethiopia=28_964, desa_yemen=17_680,
            unhcr_2024=41_763, pew_muslim_pct="99.833", pew_christians=4_367)


def pew_somalia():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    r = t[(t["Country"] == "Somalia") & (t["Year"] == 2020)].iloc[0]
    pop = float(r["Population"])
    print(f"  Pew 2020, everyone living in Somalia: {int(pop):,}; Muslims {r['Muslims'] / pop:.3%}, "
          f"Christians {int(r['Christians']):,}, unaffiliated {int(r['Religiously_unaffiliated']):,}, "
          f"Hindus {int(r['Hindus']):,}, other religions {int(r['Other_religions']):,}")
    return f"{100 * r['Muslims'] / pop:.3f}", int(r["Christians"])


def desa_somalia():
    """(world, {origin: stock}, data type) for Somalia as destination, 2024."""
    with zipfile.ZipFile(DESA) as z:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for n in z.namelist():
                if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                    out.writestr(n, z.read(n))
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name="Table 1", header=None, engine="openpyxl")
    hdr = next(i for i in range(2, 20)
               if any("of destination" in str(x) for x in df.iloc[i])
               and any("of origin" in str(x) for x in df.iloc[i]))
    cols = [str(x).strip() for x in df.iloc[hdr]]
    dcol = next(i for i, x in enumerate(cols) if "of destination" in x)
    ocol = next(i for i, x in enumerate(cols) if "of origin" in x)
    ccol = next(i for i, x in enumerate(cols) if x == "Location code of origin")
    tcol = next(i for i, x in enumerate(cols) if x == "Data type")
    ycol = next(i for i, x in enumerate(cols) if x.replace(".0", "") == "2024")
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Somalia"]
    world = int(pd.to_numeric(m.loc[m[ocol].astype(str).str.strip() == "World", ycol]).iloc[0])
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[(code < 900) | (m[ocol].astype(str).str.strip() == "Others")]
    stock = dict(zip(ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip(),
                     pd.to_numeric(ctry[ycol]).astype(int)))
    if sum(stock.values()) != world:
        raise SystemExit(f"DESA's origins for Somalia sum to {sum(stock.values()):,}, World {world:,}")
    dtype = str(m[tcol].dropna().iloc[0]).strip() if m[tcol].notna().any() else "not given"
    print(f"  UN DESA 2024, Somalia as destination: {world:,} (data type {dtype}); "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(stock.items(), key=lambda kv: -kv[1])))
    return world, stock, dtype


def unhcr_2024():
    req = urllib.request.Request(UNHCR_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r)["items"]
    except Exception as e:                            # a witness only; the gap does not use it
        print(f"  UNHCR API not reached ({e}); witness skipped")
        return None
    ref = sum(int(i["refugees"]) + int(i["asylum_seekers"]) for i in items
              if i["coo_iso"] != "SOM")
    eth = [i for i in items if i["coo_iso"] == "ETH"][0]
    print(f"  UNHCR end 2024: {ref:,} refugees and asylum seekers in Somalia (Ethiopians "
          f"{int(eth['refugees']) + int(eth['asylum_seekers']):,})")
    return ref


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 18 or int(lut["pop"].sum()) != NATIONAL_2026:
        raise SystemExit(f"{LOOKUP}: {len(lut)} regions, {int(lut['pop'].sum()):,} people; re-run so_geo.py")
    out = pd.DataFrame({"geo_id": lut["geo_id"], "geo_level": "region", "geo_name": lut["name"],
                        "source_category": "Muslim", "count": lut["pop"].astype(int),
                        "basis": "estimate", "year": 2026, "source_id": "so_hrp2026_all_muslim",
                        "note": "no source asks religion; everyone is drawn on Islam (sources/so.py), "
                                "on the 2026 humanitarian planning estimate"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: 18 regions, {int(out['count'].sum()):,} people on Muslim")

    pew_pct, pew_chr = pew_somalia()
    world, stock, _dtype = desa_somalia()
    ref = unhcr_2024()
    share = world / (NATIONAL_2026 + world)
    print(f"\n  gap_share: DESA {world:,} / ({NATIONAL_2026:,} + {world:,}) = {share:.5f}")

    got = dict(people=int(out["count"].sum()), desa_2024=world, desa_ethiopia=stock.get("Ethiopia"),
               desa_yemen=stock.get("Yemen"), unhcr_2024=ref if ref is not None else NOTE["unhcr_2024"],
               pew_muslim_pct=pew_pct, pew_christians=pew_chr)
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
