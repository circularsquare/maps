"""Mauritania: Afrobarometer Round 10 (2024) home language by wilaya, on the 2023 census's
Mauritanians per wilaya; foreign residents by the census's nationality groups.

    python sources/mr_afro.py          -> data/normalized/mr.csv

The 2013 census asked mother tongue and never published it; the 2023 census's sixteen thematic
reports have no language table (searched 2026-10-05); the DHS 2019-21 has no item; MICS 2015
prints the household head's language nationally only (Arabic 82.0, Pulaar 13.0, Soninke 2.6,
Wolof 1.7, other 0.7; Tableau HH.3, `data/raw/mr/mics5_2015_rapport.pdf`, kept as the check).
Afrobarometer's first Mauritanian round, R10 (country file released with the round, CC BY,
`data/raw/mr/MTA_R10...sav`), asks Q2 "langue parlée dans le ménage" of 1,200 adult citizens,
with REGION = the 15 wilayas.

French at home (1.5%, 18 respondents) is a learned language here (AGENT_BRIEF §2; ask 018): those
answers move to the language of the respondent's own ethnic group (Q83A); the few who gave no
group are dropped before the shares.

Population: religiondots' normalized mr.csv and mr_foreign.csv (read-only), whose per-wilaya sums
are the RGPH 2023's Mauritanians (Thème 1 minus Tableau 16.6) and foreigners (Tableau 16.6).
Foreigners (125,933): 46,800 refugees in Hodh Chargui (Tableau 16.5; the Mbera camp) on northern
Mali's census mix (Tombouctou, Mopti, Ségou weighted by UNHCR's 2018 origin shares, as
religiondots does); the other 79,133 in one national mix of Tableau 16.2's groups, each wilaya's
foreigners (Hodh Chargui's net of the refugees) at that mix.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "taxonomy"))
from rdlink import RD, RD_GEO  # noqa: E402
from mono_afro import gkey  # noqa: E402
from origin_mix import mix  # noqa: E402

SAV = ROOT / "data" / "raw" / "mr" / "MTA_R10.data_.final_.wtd_release.18Mar25_Updated.10Oct25.sav"
SAV_URL = ("https://www.afrobarometer.org/wp-content/uploads/2026/02/"
           "MTA_R10.data_.final_.wtd_release.18Mar25_Updated.10Oct25.sav")
RD_NAT = RD / "data" / "normalized" / "mr.csv"
RD_FOR = RD / "data" / "normalized" / "mr_foreign.csv"
LOOKUP = RD_GEO / "mr" / "mr_lookup.csv"
ML_CSV = ROOT / "data" / "normalized" / "ml.csv"
OUT = ROOT / "data" / "normalized" / "mr.csv"
NATIONALS, FOREIGNERS, REFUGEES = 4_801_598, 125_933, 46_800
REFUGEE_UNIT = "MR01"
# Tableau 16.2 (Thème 16, RGPH 2023), as religiondots reads and asserts it
GROUPS = {"ML": 83_681, "SN": 22_906, "MA": 1_602, "DZ": 284, "ARAB_OTHER": 2_280,
          "AFRICA_OTHER": 11_660, "EUROPE": 600, "REST": 2_920}
GROUP_NODE = {"ARAB_OTHER": "afroasiatic.arabic", "AFRICA_OTHER": "africa_other",
              "EUROPE": "other", "REST": "other"}
# UNHCR 2018 origin map of the Mbera refugees (religiondots sources/mr.py ORIGIN_2018)
REFUGEE_ORIGIN = {"Tombouctou": 0.897, "Mopti": 0.065, "Ségou": 0.036}

Q2 = {"Arabe / Hassaniya": "Hassaniya", "Poular": "Pulaar", "Soninké": "Soninke",
      "Wolof": "Wolof", "Français": None}
Q83A = {"Maure (Hassanya)": "Hassaniya", "Pular": "Pulaar", "Soninké": "Soninke",
        "Wolof": "Wolof"}


def fetch():
    import requests
    SAV.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(SAV_URL, timeout=120, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124"})
    r.raise_for_status()
    SAV.write_bytes(r.content)


def survey():
    import pyreadstat
    if not SAV.exists():
        fetch()
    df, m = pyreadstat.read_sav(str(SAV), usecols=["REGION", "Q2", "Q83A", "withinwt_hh"])
    lab = {c: df[c].map(m.variable_value_labels[c]).astype(str).str.strip()
           for c in ("REGION", "Q2", "Q83A")}
    w = pd.to_numeric(df["withinwt_hh"])
    if len(df) != 1200 or abs(w.sum() - 1200) > 0.5:
        raise SystemExit(f"R10 file moved: {len(df)} rows, weights {w.sum():.1f}")
    miss = sorted(set(lab["Q2"]) - set(Q2))
    if miss:
        raise SystemExit(f"Q2 answers with no mapping: {miss}")
    lang = lab["Q2"].map(Q2)
    fr = lang.isna()
    lang[fr] = lab["Q83A"][fr].map(Q83A)
    print(f"R10: 1,200 respondents; {int(fr.sum())} answered French at home, "
          f"{int(lang[fr].notna().sum())} moved to their ethnic group's language "
          f"({lang[fr].value_counts().to_dict()}), {int(lang.isna().sum())} with no group dropped")
    a = pd.DataFrame({"region": lab["REGION"], "lang": lang, "w": w}).dropna()
    nat = a.groupby("lang")["w"].sum() / a["w"].sum()
    print("R10 national, weighted: " + ", ".join(f"{k} {v:.1%}" for k, v in
                                                 nat.sort_values(ascending=False).items()))
    return a


def refugee_mix():
    import ml2022
    ml = pd.read_csv(ML_CSV)
    ml = ml[(ml["geo_level"] == "region") & ml["geo_name"].isin(REFUGEE_ORIGIN)]
    ml["node"] = ml["source_category"].map(ml2022.resolve)
    ml = ml[ml["node"].notna()]
    out = {}
    for reg, wt in REFUGEE_ORIGIN.items():
        s = ml[ml["geo_name"] == reg].groupby("node")["count"].sum()
        if s.empty:
            raise SystemExit(f"ml.csv has no {reg} rows")
        for k, v in (s / s.sum()).items():
            out[k] = out.get(k, 0) + wt * v
    t = sum(out.values())
    return {k: v / t for k, v in out.items()}


def round_rows(m, totals):
    out = {}
    for g, row in m.iterrows():
        fl = row.apply(int)
        short = int(totals[g]) - int(fl.sum())
        fl[(row - fl).sort_values(ascending=False).index[:short]] += 1
        out[g] = fl
    return pd.DataFrame(out).T.fillna(0).astype(int)


def main():
    lut = pd.read_csv(LOOKUP, dtype=str).set_index("geo_id")
    nat_pop = pd.read_csv(RD_NAT).groupby("geo_id")["count"].sum().reindex(lut.index)
    for_pop = pd.read_csv(RD_FOR).groupby("geo_id")["count"].sum().reindex(lut.index)
    if int(nat_pop.sum()) != NATIONALS or int(for_pop.sum()) != FOREIGNERS:
        raise SystemExit("religiondots' Mauritanian or foreign totals moved")
    if sum(GROUPS.values()) != FOREIGNERS:
        raise SystemExit("Tableau 16.2 groups do not sum to the foreigners")

    a = survey()
    norm = {gkey(n): g for g, n in lut["name"].items()}
    a["unit"] = a["region"].map(gkey).map(norm)
    bad = sorted(a.loc[a["unit"].isna(), "region"].unique())
    if bad:
        raise SystemExit(f"REGION labels with no wilaya: {bad}")
    sh = a.groupby(["unit", "lang"])["w"].sum().unstack(fill_value=0)
    sh = sh.div(sh.sum(axis=1), axis=0).reindex(lut.index)
    if sh.isna().any().any():
        raise SystemExit("a wilaya has no respondents")
    nat = round_rows(sh.mul(nat_pop, axis=0), nat_pop)
    n = a.groupby("unit").size().reindex(lut.index)
    print("respondents per wilaya: " + ", ".join(f"{lut.loc[u, 'name']} {k}" for u, k in n.items()))

    # foreigners: refugees in Hodh Chargui, the rest one national mix
    rest = dict(GROUPS)
    rest["ML"] -= REFUGEES
    tot = float(sum(rest.values()))
    comp = {}
    for g, c in rest.items():
        m = {GROUP_NODE[g]: 1.0} if g in GROUP_NODE else mix(g, "mr")
        for node, s in m.items():
            comp[node] = comp.get(node, 0) + (c / tot) * s
    other_for = for_pop.copy()
    other_for[REFUGEE_UNIT] -= REFUGEES
    ext = round_rows(pd.DataFrame({u: comp for u in lut.index}).T.mul(other_for, axis=0), other_for)
    rmix = refugee_mix()
    ref = round_rows(pd.DataFrame({REFUGEE_UNIT: rmix}).T * REFUGEES,
                     pd.Series({REFUGEE_UNIT: REFUGEES}))
    print("refugees' mix: " + ", ".join(f"{k.split('.')[-1]} {v:.1%}" for k, v in
                                       sorted(rmix.items(), key=lambda kv: -kv[1])[:5]))

    rows = []
    for df, tier, src, note in (
            (nat, "modelled", "afrobarometer_r10_x_rgph2023", "Mauritanians"),
            (ext, "derived", "rgph2023_t16_2_groups", "foreign residents"),
            (ref, "derived", "rgph2023_t16_5_refugees_x_mali_rgph2022", "refugees, Mbera")):
        s = df.stack().rename("count").reset_index()
        s.columns = ["geo_id", "source_category", "count"]
        s = s[s["count"] > 0].copy()
        s["tier"], s["source_id"], s["note"] = tier, src, note
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out["geo_level"] = "wilaya"
    out["geo_name"] = out["geo_id"].map(lut["name"])
    out["year"] = 2023
    chk = out.groupby("geo_id")["count"].sum().reindex(lut.index)
    if (chk != nat_pop + for_pop).any():
        raise SystemExit("a wilaya does not sum to the census")
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"every wilaya sums to RGPH 2023; total {int(t.sum()):,}")
    for k, v in t.head(14).items():
        print(f"   {v:>9,}  {v / t.sum():6.2%}  {k}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "source_id",
         "year", "note"]].to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")


if __name__ == "__main__":
    main()
