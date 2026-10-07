"""DR Congo: ethnic group of household heads, Enquete 1-2-3 2005 + 2012, by territoire
-> data/normalized/cd.csv

    python sources/cd_e123.py [--fetch]

NO CENSUS SINCE 1984, AND NO OPEN LANGUAGE TABLE BELOW THE COUNTRY. What was looked at
(2026-10-05), in sources/cd.md: CLEAR Global / TWB's CAID data (languages *spoken*, several per
person, lingua francas at 80-100%: a check here, not the drawn source); MICS 2010 (asks the
language mainly spoken by the head and in the household, open frequencies national only; the
microdata and the 2017-18 round sit behind a UNICEF account); DHS (registration off); DR Congo is
not an Afrobarometer or WVS country.

SOURCE DRAWN. The U.S. Census Bureau's "Democratic Republic of the Congo Subnational Population
and Housing Data Tables" (HDX, CC BY), sheet "Tribe and Religion": the number of household heads
of each of 103 named ethnic groups ("tribe") plus "Other" and "No data", summed over the INS's
Enquete 1-2-3 rounds of 2005 and 2012, for the country, 26 provinces and 164 districts
(territoires and cities). 31,755 heads. religiondots reads its religion columns from the same
workbook (religiondots/data/raw/cd, read-only here; --fetch takes a copy from HDX instead).

LABELS. USCB renamed several columns to ISO names, some wrongly ("Twa" became Plains Bira,
"Makere" became Mangbetu, "Bale (Londu)" became Lendu). `source_category` is the survey's own
field name, which the Data Dictionary keeps ('Original field name: "Twa."').

UNITS. USCB's NSO_CODE is COD-AB's admin2 pcode for 163 of 164 districts; Kasongo-Lunda is
CD3107 in USCB and CD3106 in COD-AB (and religiondots' hexes), joined on the name, asserted.
16 districts were not sampled; they take their province's shares (sampled districts' shares
weighted by their COD-PS 2024 population, as religiondots/sources/cd.py weights them).

WHAT IS WRITTEN. One row per (territoire, ethnic label): heads sampled, the share among heads
with an ethnic group recorded (`share`), and `basis` (district, or province for the unsampled).
"No data" (731 heads, 2.3%) is carried as its own share of all heads (`nodata_share`) for `gap`.
The population each share is laid on (COD-PS 2024 by territoire) is religiondots'
cd_territoires.csv, read-only; countries/cd.py multiplies.

CHECKS (all must pass):
  1. 105 TRB_ columns besides the sample size, each with one original name, all distinct
  2. 1 country, 26 provinces, 164 districts; no negative cells
  3. every sampled row's columns sum to its sample size; districts sum to provinces and to the
     national row per column
  4. the 164 districts join one-to-one to religiondots' 164 territoires (one by name)
  5. a second source agrees on where named languages are: CLEAR/CAID's "can speak" shares per
     territoire against these ethnic shares, for languages both name (Spearman, printed)
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
RD_XLSX = RD / "data" / "raw" / "cd" / "democratic-republic-of-the-congo_uscb_202103.xlsx"
RAW = HERE / "data" / "raw" / "cd"
XLSX = RAW / "democratic-republic-of-the-congo_uscb_202103.xlsx"
URL = ("https://data.humdata.org/dataset/democratic-republic-of-the-congo-subnational-"
       "population-and-housing-data-tables")
TERR = RD / "data" / "geo" / "cd" / "cd_territoires.csv"
CAID = RAW / "clearglobal_language_use_cod_admin2.csv"
OUT = HERE / "data" / "normalized" / "cd.csv"
N_COLS = 105
NODATA = "TRB_NDTA"
RENAMED = {"CD3107": "CD3106"}     # Kasongo-Lunda; asserted on the name below

# CAID label -> this survey's label, for the second-source check only (same people, same name)
CAID_PAIRS = {"Nande": "Nande (Mundande)", "Mashi": "Shi (Bashi)", "Zande": "Azande (Zande)",
              "Alur": "Alur", "Lunda": "Lunda", "Tetela": "Tetela", "Yaka": "Yaka",
              "Hunde": "Hunde", "Songe": "Songye", "Budu": "Budu", "Hemba-Yazi": "Hemba",
              "Yombe": "Yombe", "Kanyok": "Kanioka", "Kaonde": "Kaonde",
              "Mbala": "Mbala", "Bemba (Zambia)": "Bemba", "Fuliiru": "Fulero", "Havu": "Havu",
              "Logo": "Logo", "Taabwa": "Tabwe", "Phende": "Pende", "Chokwe": "Tshokwo",
              "Lendu": "Bale (Londu)", "Bushoong": "Kuba (Bushoong)", "Mayogo": "Mayogo"}


def fetch():
    """The workbook's HDX resource; religiondots already holds it, so copy that if present."""
    RAW.mkdir(parents=True, exist_ok=True)
    if RD_XLSX.exists():
        XLSX.write_bytes(RD_XLSX.read_bytes())
        print(f"copied {RD_XLSX.name} from religiondots ({XLSX.stat().st_size:,} bytes)")
        return
    raise SystemExit(f"{RD_XLSX} missing; download the workbook from {URL} into {RAW}")


def names():
    d = pd.read_excel(XLSX, sheet_name="Data Dictionary", header=None)
    out = {}
    for r in d.itertuples(index=False):
        v = [str(x) for x in r if pd.notna(x)]
        if v and v[0].startswith("TRB_") and v[0] != "TRB_SSIZE":
            m = re.search(r'Original field name: "(.*?)\.?"', v[2])
            if not m:
                raise SystemExit(f"{v[0]}: no original field name")
            out[v[0]] = (m.group(1).strip(), v[1].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch or not XLSX.exists():
        fetch()
    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("DR Congo, Enquete 1-2-3 2005+2012, ethnic group of household heads (USCB)\n")
    nm = names()
    t = pd.read_excel(XLSX, sheet_name="Tribe and Religion", header=0, skiprows=[1])
    cols = [c for c in t.columns if c.startswith("TRB_") and c != "TRB_SSIZE"]
    fr = [nm[c][0] for c in cols]
    report(len(cols) == N_COLS and set(cols) == set(nm) and len(set(fr)) == len(fr),
           f"{len(cols)} ethnic columns, each with one original name, all distinct")
    for c in cols + ["TRB_SSIZE"]:
        t[c] = pd.to_numeric(t[c], errors="raise")
    lv = t["ADM_LEVEL"].value_counts().to_dict()
    report((lv.get(0), lv.get(1), lv.get(2)) == (1, 26, 164) and (t[cols] < 0).sum().sum() == 0,
           f"levels {dict(sorted(lv.items()))}, no negative cells")
    s = t[t["TRB_SSIZE"].notna()]
    resid = (s[cols].sum(axis=1) - s["TRB_SSIZE"]).abs().max()
    report(resid == 0, f"every sampled row's columns sum to its sample size (worst {resid})")
    nat = t[t.ADM_LEVEL == 0].iloc[0]
    prov = t[t.ADM_LEVEL == 1]
    dist = t[t.ADM_LEVEL == 2].copy()
    off_n = (dist[cols].sum() - nat[cols]).abs().max()
    off_p = (dist.groupby("ADM1_NAME")[cols].sum().sub(prov.set_index("ADM1_NAME")[cols])
             .abs().max().max())
    report(off_n == 0 and off_p == 0, f"districts sum to provinces (worst {off_p}) and to the "
                                      f"country (worst {off_n}) per column")
    print(f"     {int(nat['TRB_SSIZE']):,} heads; {int(nat[NODATA]):,} with no ethnic group "
          f"({nat[NODATA] / nat['TRB_SSIZE']:.1%}); {int(nat['TRB_OTHR']):,} 'Other' "
          f"({nat['TRB_OTHR'] / nat['TRB_SSIZE']:.1%})")

    # ---- join to the territoires
    ter = pd.read_csv(TERR, dtype={"territoire": str, "unit": str})
    dist["territoire"] = dist["NSO_CODE"].astype(str).replace(RENAMED)
    tn = ter.set_index("territoire")["name"].str.split(",").str[0].str.upper()
    for k, v in RENAMED.items():
        report(tn[v].replace("-", " ") == dist.loc[dist.territoire == v, "AREA_NAME"].iloc[0]
               .replace("-", " "), f"{k} -> {v}: USCB and COD-AB both name it "
                                   f"{tn[v]}")
    report(set(dist.territoire) == set(ter.territoire) and dist.territoire.is_unique,
           f"{dist.territoire.nunique()} districts = {ter.territoire.nunique()} territoires, "
           "one to one")
    pmap = ter.set_index("territoire")["unit"]
    report((dist["NSO_CODE"].astype(str).str[:4] == dist.territoire.map(pmap)).all(),
           "every district sits in the province religiondots' hexes put it in")
    dist["pop"] = dist.territoire.map(ter.set_index("territoire")["codps_2024"])
    dist["province"] = dist.territoire.map(pmap)

    named = [c for c in cols if c != NODATA]
    sampled = dist[dist.TRB_SSIZE.notna()].copy()
    sh = sampled[named].div(sampled[named].sum(axis=1), axis=0)
    sh.index = sampled.territoire
    rows = []
    for _, r in sampled.iterrows():
        for c in named:
            if r[c] > 0:
                rows.append(dict(geo_id=r.territoire, province=r.province, name=r.AREA_NAME,
                                 basis="district", source_category=nm[c][0], uscb_name=nm[c][1],
                                 heads=int(r[c]), share=sh.loc[r.territoire, c],
                                 nodata_share=r[NODATA] / r.TRB_SSIZE))
    unsampled = dist[dist.TRB_SSIZE.isna()]
    for _, r in unsampled.iterrows():
        sib = sampled[sampled.province == r.province]
        if sib.empty:
            raise SystemExit(f"{r.AREA_NAME}: no sampled district in its province")
        w = sib.set_index("territoire")["pop"]
        ps = sh.loc[w.index].mul(w, axis=0).sum() / w.sum()
        nd = (sib[NODATA] / sib.TRB_SSIZE * sib["pop"]).sum() / sib["pop"].sum()
        for c in named:
            if ps[c] > 0:
                rows.append(dict(geo_id=r.territoire, province=r.province, name=r.AREA_NAME,
                                 basis="province", source_category=nm[c][0], uscb_name=nm[c][1],
                                 heads=0, share=ps[c], nodata_share=nd))
    out = pd.DataFrame(rows)
    tot = out.groupby("geo_id")["share"].sum()
    report((tot - 1).abs().max() < 1e-9 and out.geo_id.nunique() == 164,
           f"{out.geo_id.nunique()} territoires, shares sum to 1 in each "
           f"({len(unsampled)} unsampled on their province's shares)")
    print(f"     heads per sampled district: min {int(sampled.TRB_SSIZE.min())}, median "
          f"{int(sampled.TRB_SSIZE.median())}, max {int(sampled.TRB_SSIZE.max())}")

    # ---- second source: CLEAR/CAID "can speak" shares (languages spoken, several each)
    if CAID.exists():
        from scipy.stats import spearmanr
        c = pd.read_csv(CAID, dtype=str)
        c["v"] = c["proportion_value"].astype(float)
        print("\n  CLEAR/CAID 2016 'can speak' share vs this ethnic share, by territoire "
              "(Spearman over territoires where either is non-zero)")
        rhos = []
        for cl, el in CAID_PAIRS.items():
            if el is None:
                continue
            a_ = c[c.language_name == cl].groupby("location_code")["v"].sum()
            b_ = out[out.source_category == el].set_index("geo_id")["share"]
            idx = sorted((set(a_.index) | set(b_.index)) & set(c.location_code))
            if a_.empty:
                raise SystemExit(f"CAID has no language named {cl!r}")
            if len(idx) < 5:
                print(f"     {cl:8s} too few territoires ({len(idx)})")
                continue
            rho = spearmanr(a_.reindex(idx).fillna(0), b_.reindex(idx).fillna(0)).statistic
            rhos.append(rho)
            print(f"     {cl:8s} rho {rho:+.2f} over {len(idx)} territoires")
        med = pd.Series(rhos).median()
        report(med > 0.3, f"median rho {med:+.2f} over {len(rhos)} languages: the two sources "
                          "put the same peoples in the same places")
        lf = c[c.language_name.isin(["Lingala-Bangala", "Congo Swahili"])].groupby("language_name")["v"]
        print(f"     CAID's lingua francas, median share where named: "
              + ", ".join(f"{k} {v:.0%}" for k, v in lf.median().items()))

    if not ok:
        raise SystemExit("\nchecks FAILED; nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(out):,} rows, {out.geo_id.nunique()} territoires, "
          f"{out.source_category.nunique()} labels")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
