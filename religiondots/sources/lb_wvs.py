"""Lebanon: WVS wave 7 (2018) checked for a sect quota, and failed. Nothing is built from it.

Reads data/raw/lb/WVS_Wave_7_Lebanon_Stata_v5.1.dta (Anita's download, 2026-09-15; the zip
beside it is the original). Writes nothing. `sources/lb.md` §9 is the record; sources.md
§lb-2026-09-15 the short version.

## WHAT IT CHECKS

The Arab Barometer's Lebanese composition by governorate is a fieldwork quota (`sources/lb.md`
§2), found by two waves agreeing to the interview. One WVS wave cannot be tested that way, so this
reads the file against the survey's own design instead:

  1. `I_PSU`: 120 clusters of exactly 10 interviews, and every one of them holds a single
     community (Sunni, Shia, Druze or Christian). The four `Other; nfd` answers sit one each in
     four Christian clusters in El Meten and are left out of the purity test.
  2. The WVS7 Sample Design for Lebanon (IHSN catalogue 12281, related material 104802, pp.3-4)
     prints a table of Mohafaza, Kadaa, Sample, Number of PSUs and **Sect**. `DESIGN` below is
     that table summed to kadaa and sect; the file's clusters reproduce it in all 23 kadaa.
     Its methodology report (104800) ticks Yes on quota controls (Q15), says "the daily work
     sheet mentioned also the profile required of the interviewee" (Q14), and gives the
     stratification factors as "by governorates / districts / religions" (Q20).
  3. Printed only: within Christian clusters the denominations mix freely (Maronite, Orthodox,
     Catholic, Armenian), which is the one part of the religion answer the design did not set.

The fieldwork firm is Statistics Lebanon Ltd., which is also the Arab Barometer's Lebanese
partner; Arab Barometer wave V's technical report (p.7) gives its strata as "Governorates and
sect". So both surveys' sect mix by governorate is the firm's allocation, not a measurement.

Exits 0 when the file still reproduces the design (the record's claim holds), 1 otherwise.

Usage:
    python sources/lb_wvs.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DTA = os.path.join(ROOT, "data", "raw", "lb", "WVS_Wave_7_Lebanon_Stata_v5.1.dta")

N_RESP = 1200
N_PSU = 120
PER_PSU = 10

SECT = {10100000: "Catholic", 10205010: "Maronite", 30112000: "Orthodox",
        30202001: "Armenian", 50101000: "Sunni", 50202000: "Shia", 50500010: "Druze",
        90000000: "Other"}
COMMUNITY = {"Catholic": "Christian", "Maronite": "Christian", "Orthodox": "Christian",
             "Armenian": "Christian", "Sunni": "Sunni", "Shia": "Shia", "Druze": "Druze",
             "Other": None}

# Sample Design pp.3-4, summed to kadaa and sect: number of PSUs. Transcribed 2026-09-15 from the
# rendered pages; the table's own subtotals (Beirut 13, El Nabatieh 7, Bekaa 15, Mount Lebanon
# 48, North 24, South 13, total 120) are asserted below.
DESIGN = {
    ("Beirut", "Christian"): 5, ("Beirut", "Shia"): 1, ("Beirut", "Sunni"): 7,
    ("Bent Jbeil", "Shia"): 3, ("El Nabatieh", "Shia"): 4,
    ("West Bekaa", "Sunni"): 2, ("West Bekaa", "Shia"): 2,
    ("Zahle", "Sunni"): 1, ("Zahle", "Christian"): 2, ("Zahle", "Shia"): 1,
    ("Baalbek", "Shia"): 5, ("Rachaya", "Druze"): 2,
    ("Aley", "Druze"): 3,
    ("Baabda", "Christian"): 4, ("Baabda", "Druze"): 3, ("Baabda", "Shia"): 5,
    ("Ech Chouf", "Christian"): 2, ("Ech Chouf", "Druze"): 2, ("Ech Chouf", "Sunni"): 3,
    ("El Meten", "Christian"): 15,
    ("Jbeil", "Christian"): 4, ("Jbeil", "Shia"): 2,
    ("Kesseroune", "Christian"): 5,
    ("Akkar", "Sunni"): 5, ("Bcharre", "Christian"): 2, ("El Batroun", "Christian"): 2,
    ("El Koura", "Christian"): 2, ("El Koura", "Sunni"): 1, ("El Minieh-Dennie", "Sunni"): 2,
    ("Tripoli", "Sunni"): 8, ("Zgharta", "Christian"): 2,
    ("Jezzine", "Christian"): 2, ("Saida", "Shia"): 3, ("Saida", "Sunni"): 3,
    ("Sour", "Shia"): 5,
}
DESIGN_MOHAFAZA = {
    "Beirut": ["Beirut"], "El Nabatieh": ["Bent Jbeil", "El Nabatieh"],
    "Bekaa": ["West Bekaa", "Zahle", "Baalbek", "Rachaya"],
    "Mount Lebanon": ["Aley", "Baabda", "Ech Chouf", "El Meten", "Jbeil", "Kesseroune"],
    "North": ["Akkar", "Bcharre", "El Batroun", "El Koura", "El Minieh-Dennie", "Tripoli",
              "Zgharta"],
    "South": ["Jezzine", "Saida", "Sour"],
}
DESIGN_SUBTOTAL = {"Beirut": 13, "El Nabatieh": 7, "Bekaa": 15, "Mount Lebanon": 48,
                   "North": 24, "South": 13}
# `N_TOWN` label -> the design's kadaa. The file splits Saida into "Saida" and "Saida Villages"
# (the design's three Shia PSUs are the file's Saida Villages).
TOWN_TO_KADAA = {"LB: BEIRUT": "Beirut", "LB: Kessroune": "Kesseroune",
                 "LB: Saida Villages": "Saida"}


def load():
    import pyreadstat

    if not os.path.exists(DTA):
        raise SystemExit(f"{DTA} missing; unzip F00013081-WVS_Wave_7_Lebanon_Stata_v5.1.zip there")
    df, meta = pyreadstat.read_dta(DTA)
    if len(df) != N_RESP:
        raise SystemExit(f"{len(df)} respondents, expected {N_RESP}")
    if not ((df["A_YEAR"] == 2018).all() and (df["B_COUNTRY"] == 422).all()):
        raise SystemExit("A_YEAR / B_COUNTRY do not hold 2018 / 422")
    if not (df["W_WEIGHT"] == 1).all():
        raise SystemExit("W_WEIGHT is no longer 1 for everyone")
    if not (df["Q269"] == 1).all():
        raise SystemExit("Q269 (citizen) is no longer Yes for everyone")
    town = meta.value_labels[meta.variable_to_label["N_TOWN"]]
    df["sect"] = df["Q289CS9"].astype(int).map(SECT)
    if df["sect"].isna().any():
        raise SystemExit(f"unexpected Q289CS9 codes {sorted(set(df['Q289CS9']) - set(SECT))}")
    df["community"] = df["sect"].map(COMMUNITY)
    df["kadaa"] = [TOWN_TO_KADAA.get(town[int(t)], town[int(t)].removeprefix("LB: "))
                   for t in df["N_TOWN"]]
    df["psu"] = df["I_PSU"].astype(int)
    return df


def psu_purity(df, psu_col="psu", group_col="community"):
    """How many clusters hold one value of `group_col` only (missing values ignored).

    A general form: in a single-round survey with cluster ids, a religion stratum shows as every
    cluster pure on the stratified answer while a finer answer inside it mixes.
    """
    t = pd.crosstab(df[psu_col], df[group_col])
    pure = t.gt(0).sum(axis=1) == 1
    return int(pure.sum()), len(t), t[~pure]


def main():
    df = load()
    sizes = df.groupby("psu").size()
    if len(sizes) != N_PSU or not (sizes == PER_PSU).all():
        raise SystemExit(f"{len(sizes)} PSUs with sizes {sizes.value_counts().to_dict()}")
    print(f"WVS-7 Lebanon 2018: {len(df):,} citizens, {len(sizes)} PSUs of {PER_PSU}, no weights")

    ok = True
    n_pure, n_psu, mixed = psu_purity(df)
    print(f"  PSUs holding one community only (Other; nfd left out): {n_pure} of {n_psu}")
    if n_pure != n_psu:
        ok = False
        print(mixed)

    for m, kadaas in DESIGN_MOHAFAZA.items():
        got = sum(v for (k, _s), v in DESIGN.items() if k in kadaas)
        if got != DESIGN_SUBTOTAL[m]:
            raise SystemExit(f"DESIGN sums to {got} PSUs in {m}, the table prints "
                             f"{DESIGN_SUBTOTAL[m]}; transcription slip")
    psu_comm = (df.dropna(subset=["community"]).groupby("psu")
                .agg(kadaa=("kadaa", "first"), community=("community", "first")))
    file_cells = psu_comm.groupby(["kadaa", "community"]).size().to_dict()
    if file_cells != DESIGN:
        ok = False
        keys = sorted(set(file_cells) | set(DESIGN))
        for k in keys:
            if file_cells.get(k, 0) != DESIGN.get(k, 0):
                print(f"    {k}: file {file_cells.get(k, 0)} PSUs, design {DESIGN.get(k, 0)}")
    else:
        print(f"  PSUs per kadaa and sect equal the Sample Design's table in all "
              f"{len({k for k, _ in DESIGN})} kadaa ({sum(DESIGN.values())} PSUs)")

    chr_psus = psu_comm.index[psu_comm["community"] == "Christian"]
    ch = df[df["psu"].isin(chr_psus) & (df["community"] == "Christian")]
    n_mix = int((pd.crosstab(ch["psu"], ch["sect"]).gt(0).sum(axis=1) > 1).sum())
    print(f"  printed only: Christian PSUs mixing denominations {n_mix} of {len(chr_psus)} "
          "(the part of the answer the design did not set)")

    gov = df.pivot_table(index="N_REGION_ISO", columns="community", values="psu",
                         aggfunc="size", fill_value=0)
    print("\n  interviews by governorate (N_REGION_ISO) and community, as the design allocated them:")
    print(gov.to_string())

    if ok:
        print("\nCONFIRMED: the sect mix by governorate is the design's allocation, not a "
              "measurement. Lebanon is not drawn from this file (sources/lb.md §9).")
        return 0
    print("\nTHE FILE NO LONGER REPRODUCES THE DESIGN. Re-read sources/lb.md §9 before "
          "believing anything here.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
