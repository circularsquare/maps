"""Myanmar, GAD 2018 Township Profiles, ethnic nationalities living in each township
-> data/normalized/mm.csv (and mm_unrecorded.csv beside it).

    python sources/mm_gad.py [--fetch]

SOURCE. The U.S. Census Bureau's "Burma Subnational Population and Housing Data Tables with
Administrative Boundaries" on HDX (CC BY), sheet `Ethnicity`: the General Administration
Department's 2018 Township Profiles (published 2019), Table 14 "Ethnic Nationalities Living",
figures as of 1 April 2017, transcribed by USCB for 330 townships. These are administrative
records kept by township offices, not a census and not self-report. The 2014 census asked
ethnicity (135 codes) and never released it; no census or open survey asks language (sources/mm.md).
31 named groups plus "Foreign" and "Other". The same sheet carries GAD's religion table for the
same townships and date, which is how the unrecorded remainder below is measured.

READ AS LANGUAGE under AGENT_BRIEF section 2's ethnicity rule: every row `derived`.

THE FIVE WA SELF-ADMINISTERED DIVISION TOWNSHIPS AND MONGLA HAVE NO ROW (Mongmao, Pangwaun,
Pangsang, Narphan, Mongla): GAD has no office there. They are asserted by name and left undrawn.

RAKHINE'S ETHNICITY TOTAL IS 584,142 BELOW ITS RELIGION TOTAL. GAD's religion table counts
2,670,819 people in Rakhine and its ethnicity table 2,086,677; GAD's own Islam count there is
588,353. The difference is people the township offices recorded but gave no ethnic nationality,
which in Rakhine is the Rohingya (GAD does not recognise the name). The script writes the per
township difference (religion total minus ethnicity total, where positive) to mm_unrecorded.csv
for the record; it is NOT drawn (the brief for this country: say what the source leaves out, do
not invent a figure for a language).

CHECKS (all must pass):
  1. 330 townships under 15 states/regions; exactly the five named townships lack the table
  2. the national row equals the sum of the townships, per category, exactly
  3. every township's categories against its printed total: the rows that differ are printed and
     the largest difference is bounded (USCB transcribes GAD's own sums, which do not always add)
  4. national total 47,809,979 as USCB prints it
  5. the Rakhine remainder: per state, religion total minus ethnicity total, printed
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "mm"
XLSX = RAW / "burma_uscb_202003.xlsx"
GDB = RAW / "burma.gdb.zip"
BASE = ("https://data.humdata.org/dataset/2689fec5-6a1a-4773-8a0b-304445c52fe7/resource/")
URLS = {XLSX: BASE + "788f18ae-67a2-456b-8111-c4edf97baa34/download/burma_uscb_202003.xlsx",
        GDB: BASE + "20453862-f0e6-4d78-bb7e-15e639983ffb/download/burma.gdb.zip"}
OUT = HERE / "data" / "normalized" / "mm.csv"
OUT_GAP = HERE / "data" / "normalized" / "mm_unrecorded.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

NATIONAL = 47_809_979
NO_TABLE = {"MONGMAO", "PANGWAUN", "PANGSANG", "NARPHAN", "MONGLA"}
ROW_BOUND = 9_000          # largest |categories - printed total| per township, as measured (8,710)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url in URLS.items():
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600).read()
        if data[:4] != b"PK\x03\x04":
            raise SystemExit(f"{path.name}: not a zip/xlsx ({len(data):,} bytes)")
        path.write_bytes(data)
        print(f"wrote {path} ({len(data):,} bytes)")


def categories():
    """ETH_* column -> USCB's English label, from the data dictionary (2017 GAD fields only)."""
    d = pd.read_excel(XLSX, sheet_name="Data Dictionary", header=None)
    out = {}
    for r in d.itertuples(index=False):
        v = [str(x) for x in r if pd.notna(x)]
        if len(v) >= 3 and v[0].startswith("ETH_") and not v[0].endswith("13") \
                and v[0] != "ETH_TPOP" and "2018 Township Profiles" in " ".join(v):
            out[v[0]] = v[1].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not XLSX.exists() or not GDB.exists():
        fetch()

    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Myanmar, GAD 2018 Township Profiles Table 14 (USCB transcription)\n")
    cats = categories()
    df = pd.read_excel(XLSX, sheet_name="Ethnicity", header=0, skiprows=[1])
    cols = list(cats)
    report(len(cols) == 33 and all(c in df.columns for c in cols),
           f"{len(cols)} categories (31 named groups, Foreign, Other)")

    tw = df[df["ADM_LEVEL"] == 3].copy()
    states = df[df["ADM_LEVEL"] == 1]
    blank = set(tw.loc[tw["ETH_TPOP"].isna(), "ADM3_NAME"])
    report(len(tw) == 330 and len(states) == 15 and blank == NO_TABLE,
           f"{len(tw)} townships, {len(states)} states/regions; no table in {sorted(blank)}")
    tw = tw[tw["ETH_TPOP"].notna()].copy()
    for c in cols + ["ETH_TPOP", "RLG_TPOP"]:
        tw[c] = pd.to_numeric(tw[c], errors="raise").fillna(0)
    report(int((tw[cols] < 0).sum().sum()) == 0, "no negative cells")

    nat = df[df["ADM_LEVEL"] == 0].iloc[0]
    off = {c: int(tw[c].sum() - nat[c]) for c in cols + ["ETH_TPOP"]}
    report(all(v == 0 for v in off.values()), "townships sum to the national row in every category")
    report(int(nat["ETH_TPOP"]) == NATIONAL, f"national total {int(nat['ETH_TPOP']):,}")

    tw["resid"] = (tw[cols].sum(axis=1) - tw["ETH_TPOP"]).astype(int)
    bad = tw[tw["resid"] != 0]
    report(tw["resid"].abs().max() <= ROW_BOUND,
           f"{len(bad)} townships whose categories differ from their printed total; largest "
           f"{tw['resid'].abs().max():,} (bound {ROW_BOUND:,}), net {tw['resid'].sum():+,}")
    for _, r in bad.iterrows():
        print(f"       {r['ADM1_NAME']:<14} {r['ADM3_NAME']:<16} total {int(r['ETH_TPOP']):>9,} "
              f"categories {int(r['ETH_TPOP'] + r['resid']):>9,} ({r['resid']:+,})")

    print("\n  religion total minus ethnicity total, per state (GAD, same townships, same date):")
    tw["unrec"] = (tw["RLG_TPOP"] - tw["ETH_TPOP"]).clip(lower=0).astype(int)
    by_state = tw.groupby("ADM1_NAME")[["ETH_TPOP", "RLG_TPOP", "unrec", "RLG_ISL"]].sum()
    for s, r in by_state.sort_values("unrec", ascending=False).iterrows():
        print(f"       {s:<14} ethnicity {int(r.ETH_TPOP):>10,} religion {int(r.RLG_TPOP):>10,} "
              f"unrecorded {int(r.unrec):>8,}  (Islam {int(r.RLG_ISL):,})")
    rk = by_state.loc["RAKHINE STATE"]
    report(550_000 < rk["unrec"] < 620_000,
           f"Rakhine unrecorded {int(rk['unrec']):,} against GAD's Islam {int(rk['RLG_ISL']):,}")
    print(f"       national unrecorded {int(tw['unrec'].sum()):,}")
    tw.loc[tw["unrec"] > 0, ["GEO_MATCH", "ADM1_NAME", "ADM3_NAME", "ETH_TPOP", "RLG_TPOP",
                             "unrec", "RLG_ISL"]].to_csv(OUT_GAP, index=False)

    rows = []
    for _, r in tw.iterrows():
        for c in cols:
            n = int(r[c])
            if n > 0:
                rows.append((r["GEO_MATCH"], "township", f"{r['ADM1_NAME']} / {r['ADM3_NAME']}",
                             cats[c], n, "derived"))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                      "count", "tier"])
    report(out["geo_id"].nunique() == 325, f"{out['geo_id'].nunique()} townships written")
    print(f"\n  {len(out):,} rows, {out['count'].sum():,} people, "
          f"{out['source_category'].nunique()} categories")
    if not ok:
        raise SystemExit("checks failed; nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}\nwrote {OUT_GAP}")


if __name__ == "__main__":
    main()
