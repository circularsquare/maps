"""Chad: mother tongue of the household head by province, from MICS6 2019 microdata, as shares
applied to the 2009 census région populations -> data/normalized/td.csv.

    python sources/td_mics.py

SOURCE. Enquête par grappes à indicateurs multiples (MICS6) Tchad 2019 (INSEED with UNICEF), SPSS
files from mics.unicef.org (Anita's UNICEF account, 2026-10-09), unzipped to
data/raw/td/mics_2019/ (gitignored; research use, no redistribution, and the readme asks that
copies of publications go to INSEED and UNICEF Chad). 19,217 households sampled, 18,967
interviewed, 112,604 household members; 769 clusters, 27-36 per province (N'Djaména 48, Ennedi
Est and Ouest together 57). The file has no households in Tibesti.

ITEM. HC1B, "Langue maternelle du chef de ménage" (French, Chadian Arabic, Sar, Gorane, Kanembou,
Maba/Ouaddaï, Moundang, Massa, Peul, Lélé, Toupouri, Ngambaye, Zaghawa, other), MICS's standard
household mother-tongue item, read as every member's: hl.sav members x hhweight. HH16 (the
respondent's mother tongue, same list) agrees with it in 91.7% of households and does NOT slide
towards the interview language the way Iraq's and Afghanistan's did: Chadian Arabic is 13.82% of
persons by HC1B and 13.79% by HH16, although 62% of interviews (HH15) were in Chadian Arabic.
HC1B is drawn for consistency with the other MICS countries; WM14 (women 15-49) is printed too.

OTHER (37.5% of persons). Split by HC2, "Ethnie du chef de ménage" (19 groups and other, the
census's grands groupes of Tableau 5.02, Annexe 2), towards the languages of that group that the
MICS list does not name, as the census record supports them (sources/td.md §0):
  * a single language: Arabs -> Chadian Arabic (below); Gorane (Téda) -> Gorane's Tubu node;
    Zaghawa (Bideyat, Kobé) -> Zaghawa; Peul (Bodoré) -> Fula; Boulala/Médégo -> Boulala
    (Glottolog's Naba is the one Bilala-Kuka-Medogo language); Toupouri/Kéra -> Kéra.
  * several census rows: Ouaddaï/Mimi -> Massalit, Mimi; Marba/Lélé -> Marba, Mesmé; Karo/Zimé ->
    Karo, Pévé; Sara -> the census's Sara row less Ngambay and Sar (Mbay, Gulay, Gor, Laka...),
    Sara Kaba, Daye, Mboum.
  * census rows plus the group's languages the census filed in "Autres" (Annexes 2-3): Massa ->
    Mousseye and Mousgoum (its only unprinted member); Tama -> Tama and Assongori/Mararit (both
    Tamaic: on `nilosaharan`); Baguirmi, Dadjo, Bidio (Hadjaraï), Gabri, other ethnic groups ->
    their printed rows and an unnamed remainder on africa_other.
  * no language MICS does not already name: Kanembou/Bornou (Kanouri, Boudouma), Mesmédjé, and
    Moundang -> africa_other.
Each candidate's weight in a province is its census count (Tableau 5.10 after the Arabic move of
the previous build; a group's "Autres" share is the part of its ethnic count that neither its
printed rows nor the move account for) times its nearness to that province (Glottolog points,
exp(-d/50 km), the previous build's seed). So the census decides between a group's languages and
MICS decides how many of the group live in each province.

ARABS WHO ANSWERED "OTHER". 676 Arab-headed households (23%) gave "other": an interviewer
convention, not another language. Interviewers in the same cluster disagree (p = 2e-63); in 29
clusters one interviewer coded all their Arab households "other" and another none (246 of the
676). In Salamat interviewer 63 coded 41 of 44 Arab households "other" and interviewer 84 none of
17; in Sila four interviewers of one team coded all 43, another 0 of 16; in Batha 30 of 41
against 1 of 59 and 1 of 57. 95% of these interviews were held in Chadian Arabic. They are drawn
as Chadian Arabic (check 4).

TIBESTI has no MICS households: drawn on Borkou's shares (BORROW), 21,303 people.

SHARES. Weighted persons per (province, answer); MICS's Ennedi Est and Ennedi Ouest pooled into
the 2009 région of Ennedi; x the 2009 census population of each région (Tableau 5.07, the units
and totals of religiondots' td.csv), largest remainder: 10,941,682 people. The previous build
drew the 8,088,816 aged 6+ of Tableau 5.10; MICS's shares are of everyone, so the 6+ universe
no longer applies.

CHECKS. Row counts; every member matches a household; 22 MICS provinces, Tibesti absent, the
crosswalk covering all 22 régions; HC1B vs HH16 household agreement >= 90% and Arabic within
1 point nationally, 10 points per province (only Chari Baguirmi, 34 vs 29, and Wadi Fira, 8.5
vs 16.6, are 3+ apart); the Arab-headed "other" answers split inside clusters by interviewer
and held in Arabic; national Chadian Arabic between MICS's Arab ethnic share and the census's
21.7%; the Sara group within 3 points of MICS's Sara ethnic share; région totals = Tableau
5.07 = religiondots' td.csv.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD, RD_GEO  # noqa: E402
import td_rgph  # noqa: E402  (Tableau 5.10, 5.02, 5.07 as transcribed and checked there)

RAW = HERE / "data" / "raw" / "td" / "mics_2019"
OUT = HERE / "data" / "normalized" / "td.csv"
HEXES = RD_GEO / "td" / "td_hexes.gpkg"           # religiondots, read-only
GLOTTOLOG = HERE / "data" / "raw" / "glottolog" / "languages.csv"
SOURCE_ID = "mics6_2019_hc1b"
N_HH, N_IV, N_HL = 19_217, 18_967, 112_604
KERNEL_KM, LAMBDA = 50.0, 0.95      # the previous build's nearness seed (countries/td.py, 2026-10-05)

# MICS HH7 code -> 2009 région (Tableau 5.07's spelling, religiondots' unit)
PROV = {1: "Batha", 2: "Borkou", 3: "Chari Baguirmi", 4: "Guéra", 5: "Hadjer Lamis", 6: "Kanem",
        7: "Lac", 8: "Logone Occidental", 9: "Logone Oriental", 10: "Mandoul",
        11: "Mayo Kebbi Est", 12: "Mayo Kebbi Ouest", 13: "Moyen Chari", 14: "Ouaddaï",
        15: "Salamat", 16: "Tandjilé", 17: "Wadi Fira", 18: "N'Djaména", 19: "Barh El Gazal",
        20: "Ennedi", 21: "Sila", 22: "Tibesti", 23: "Ennedi"}   # 20 Ennedi Ouest, 23 Ennedi Est
BORROW = {"Tibesti": "Borkou"}     # no MICS households

# HC1B code -> label written to td.csv (taxonomy/td2019.py maps these)
NAMED = {1: "FRANCAIS", 2: "ARABE TCHADIEN", 3: "SAR", 4: "GORANE", 5: "KANEMBOU",
         6: "MABA/OUADDAI", 7: "MOUNDANG", 8: "MASSA", 9: "PEUL", 10: "LELE", 11: "TOUPOURI",
         12: "NGAMBAYE", 13: "ZAGHAWA"}
OTHER, NO_ANSWER = 96, 99

AR = "Arabe local"
SARA_REST = "Sara (autres langues sara)"
REM = "Autres langues nationales"
MUSGUM = "Mouloui/Mousgoum"
TAMAIC = "Assongori/Mararit"

# HC2 (head's ethnic group) -> (Tableau 5.02 group, candidate labels for a head who answered
# "other"). A candidate is a Tableau 5.10 row, SARA_REST, or the group's own remainder (REM,
# MUSGUM, TAMAIC), whose census weight is the group's "Autres" estimate.
SPLIT = {
    "ARABE": (None, [AR]),
    "GORANE": (None, ["Gorane"]),
    "ZAGHAWA": (None, ["Zaghawa/Béri/Bideyat"]),
    "PEUL/FOULBE": (None, ["Peul/Foulfouldé/Bodoré"]),
    "BOULALA/MEDEGO": (None, ["Boulala"]),
    "TOUPOURI/KERA": (None, ["Kéra"]),
    "OUADDAI/MIMI": (None, ["Massalit", "Mimi"]),
    "MARBA/LELE": (None, ["Marba", "Mesmé"]),
    "KARO/ZIME": (None, ["Karo/Kado", "Lamé/Pévé"]),
    "SARA": (None, [SARA_REST, "Sara Kaba", "Daye", "Mboum"]),
    "MASSA/MOUSSEYE": ("Massa/Mousseye/Mousgoume", ["Mousseye", MUSGUM]),
    "TAMA/ASSONGORI": ("Tama/Assongori/Mararit", ["Tama", TAMAIC]),
    "BAGUIRMI/BARMA": ("Baguirmi/Barma et autres", ["Barma/Baguirmi", "Toumak/Ndom", REM]),
    "DADJO/MOURO": ("Dadjo/Kibet/Mouro et autres", ["Dadjo", REM]),
    "BIDIO/KENGA/DANGLEAT": ("Bidio/Migami/Kinga/dangléat et autres", ["Moubi", REM]),
    "GABRI/NANGTCHERE": ("Gabri/Kabalaye/Nangtchéré/Soumraye et autres",
                         ["Gabri", "Kabalaye", "Nangtchéré", REM]),
    "AUTRE ETHNIE": ("Autres ethnies tchadiennes (Achit/Banda/Kim et autres)",
                     ["Rounga", "Kim", REM]),
    "KANEMBOU/BORNOU": (None, [REM]),
    "MESMEDJE/MASSALAT": (None, [REM]),
    "MOUNDANG": (None, [REM]),
    "NON REPONSE": (None, [REM]),
}

# Glottolog points per candidate (the previous build's GLOTTO, plus the remainders' languages
# from Annexes 2-3). A candidate without points (REM of the other ethnic groups) is even.
GLOTTO = {
    SARA_REST: ["mbay1241", "gula1268", "gorr1238", "laka1254", "mang1398", "bedj1245",
                "ngam1269", "kaba1281"],
    "Sara Kaba": ["sara1321", "sara1322"], "Daye": ["dayy1236"], "Mboum": ["kara1478", "nzak1246"],
    "Massalit": ["nucl1440"], "Mimi": ["mimi1241", "mimi1240"],
    "Marba": ["marb1239"], "Mesmé": ["mesm1239"],
    "Karo/Kado": ["herd1236", "nget1241"], "Lamé/Pévé": ["peve1243"],
    "Mousseye": ["muse1242"], MUSGUM: ["musg1254"],
    "Tama": ["tama1331"], TAMAIC: ["assa1269", "mara1396"],
    "Barma/Baguirmi": ["bagi1246"], "Toumak/Ndom": ["tuma1260"],
    "Dadjo": ["dard1243", "dars1235"], "Moubi": ["mubi1246"],
    "Gabri": ["gabr1253"], "Kabalaye": ["kaba1292"], "Nangtchéré": ["nanc1253"],
    "Rounga": ["rung1258"], "Kim": ["kimm1246"],
}
REM_GLOTTO = {   # a group's remainder: the languages Annexes 2-3 put in it
    "Baguirmi/Barma et autres": ["buaa1245", "niel1243", "tuni1251", "ndam1251"],
    "Dadjo/Kibet/Mouro et autres": ["kibe1241", "tora1267"],
    "Bidio/Migami/Kinga/dangléat et autres": ["bidi1241", "dang1274", "miga1249", "keng1240",
                                              "soko1263", "mogu1251", "jonk1238", "muku1242",
                                              "saba1276"],
    "Gabri/Kabalaye/Nangtchéré/Soumraye et autres": ["somr1248", "besm1235"],
}

# The previous build's Arabic move (countries/td.py, 2026-10-05): Tableau 5.02 groups -> their
# printed Tableau 5.10 rows, and whether some of the group's languages are in "Autres"
GROUPS = {
    "Gorane": (["Gorane"], True),
    "Baguirmi/Barma et autres": (["Barma/Baguirmi", "Toumak/Ndom"], True),
    "Kanembou/Bornou/Boudouma": (["Kanembou"], True),
    "Boulala/Médégo/Kouka": (["Boulala"], True),
    "Ouaddaï/Maba/Massalit/Mimi": (["Maba/Ouaddaï", "Massalit", "Mimi"], False),
    "Zaghawa (Bideyat/Kobé)": (["Zaghawa/Béri/Bideyat"], False),
    "Dadjo/Kibet/Mouro et autres": (["Dadjo"], True),
    "Bidio/Migami/Kinga/dangléat et autres": (["Moubi"], True),
    "Moundang": (["Moundang"], False),
    "Massa/Mousseye/Mousgoume": (["Massa", "Mousseye"], True),
    "Toupouri/Kéra": (["Toupouri", "Kéra"], False),
    "Sara (Ngambaye/Sara Madjingaye/Mbaye et autres)": (["Sara", "Sara Kaba", "Daye", "Mboum"], False),
    "Peul/Foulbé/Bodoré": (["Peul/Foulfouldé/Bodoré"], False),
    "Tama/Assongori/Mararit": (["Tama"], True),
    "Gabri/Kabalaye/Nangtchéré/Soumraye et autres": (["Gabri", "Kabalaye", "Nangtchéré"], True),
    "Marba/Lélé/Mesmé": (["Marba", "Lélé", "Mesmé"], False),
    "Mesmedjé/Massalat/Kadjaksé": ([], True),
    "Karo/Zimé/Pévé": (["Karo/Kado", "Lamé/Pévé"], False),
    "Autres ethnies tchadiennes (Achit/Banda/Kim et autres)": (["Rounga", "Kim"], True),
}


def census_weights():
    """Tableau 5.10 counts after the Arabic move, and each mixed group's "Autres" estimate."""
    T = td_rgph.T510
    raw = {k: v[0] / 100 * td_rgph.URBAN + v[1] / 100 * td_rgph.RURAL for k, v in T.items()}
    s = sum(raw.values())
    cnt = {k: x * td_rgph.TOTAL / s for k, x in raw.items()}
    etot, total = td_rgph.T502_TOTAL, td_rgph.TOTAL
    E = {g: td_rgph.T502[g] / etot * total for g in GROUPS}
    X = cnt[AR] - td_rgph.T502["Arabe"] / etot * total
    D = {g: max(0.0, E[g] - sum(cnt[n] for n in names)) for g, (names, _) in GROUPS.items()}
    clean = sum(D[g] for g, (_, m) in GROUPS.items() if not m)
    mixed = sum(D[g] for g, (_, m) in GROUPS.items() if m)
    sc, rho = (X / clean, 0.0) if clean >= X else (1.0, min(1.0, (X - clean) / mixed))
    N, autres = dict(cnt), {}
    for g, (names, m) in GROUPS.items():
        R = D[g] * (rho if m else sc)
        L = sum(cnt[n] for n in names)
        to_named = (R * (L / (L + (1 - rho) * D[g]) if m else 1.0)) if names else 0.0
        for n in names:
            N[n] += to_named * cnt[n] / L
        N[AR] -= R
        if m:
            autres[g] = (1 - rho) * D[g] + R - to_named
    return N, autres, rho


def nearness(regs):
    """{candidate: array over regs}: each Glottolog point's population-weighted pull on each
    région, normalised on its own, summed, mixed LAMBDA : (1 - LAMBDA) with even."""
    import numpy as np
    import pandas as pd
    import pyogrio
    place = pyogrio.read_dataframe(HEXES)
    pts = place.geometry.representative_point()
    hx, hy = pts.x.to_numpy(), pts.y.to_numpy()
    pop = place["pop"].to_numpy(dtype=float)
    ri = {r: i for i, r in enumerate(regs)}
    if set(place["unit"].astype(str)) != set(regs):
        raise SystemExit("td_hexes.gpkg units differ from the 22 régions")
    ridx = place["unit"].astype(str).map(ri).to_numpy()
    nR = len(regs)
    kpop = np.bincount(ridx, weights=pop, minlength=nR)
    g = pd.read_csv(GLOTTOLOG, usecols=["ID", "Latitude", "Longitude"]).set_index("ID")

    def near(codes):
        k = np.zeros(nR)
        for gc in codes:
            lat, lon = g.loc[gc, ["Latitude", "Longitude"]].astype(float)
            d = np.hypot((hx - lon) * 111.32 * np.cos(np.radians(lat)), (hy - lat) * 110.57)
            kk = np.bincount(ridx, weights=pop * np.exp(-d / KERNEL_KM), minlength=nR) / kpop
            k += kk / kk.sum()
        return (1 - LAMBDA) / nR + LAMBDA * k / k.sum()
    out = {c: near(v) for c, v in GLOTTO.items()}
    out.update({("rem", grp): near(v) for grp, v in REM_GLOTTO.items()})
    return out


def lr_round(v, total):
    s = sum(v.values())
    raw = {k: x / s * total for k, x in v.items()}
    cnt = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:total - sum(cnt.values())]:
        cnt[k] += 1
    assert sum(cnt.values()) == total
    return cnt


def main():
    import numpy as np
    import pandas as pd
    import pyreadstat

    hh, meta = pyreadstat.read_sav(str(RAW / "hh.sav"), usecols=[
        "HH1", "HH2", "HH3", "HH7", "HH15", "HH16", "HC1B", "HC2", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2", "HL1"])
    wm, _ = pyreadstat.read_sav(str(RAW / "wm.sav"), usecols=["HH7", "WM14", "wmweight"])
    lab = meta.variable_value_labels
    hc2 = {int(k): v for k, v in lab["HC2"].items()}
    if len(hh) != N_HH or len(hl) != N_HL or int((hh.hhweight > 0).sum()) != N_IV:
        raise SystemExit(f"hh {len(hh)} / hl {len(hl)} / interviewed {(hh.hhweight > 0).sum()}")
    if {int(k): v for k, v in lab["HC1B"].items() if int(k) in NAMED} != NAMED:
        raise SystemExit("HC1B's value labels are not the expected list")
    if set(hc2.values()) - set(SPLIT):
        raise SystemExit(f"HC2 groups without a SPLIT entry: {set(hc2.values()) - set(SPLIT)}")
    iv = hh[hh.hhweight > 0].copy()
    if iv[["HH7", "HC1B", "HC2"]].isna().any().any():
        raise SystemExit("an interviewed household lacks HH7, HC1B or HC2")
    iv["reg"] = iv["HH7"].astype(int).map(PROV)
    persons = hl.merge(iv, on=["HH1", "HH2"], how="left", indicator=True)
    if (persons["_merge"] == "left_only").any() and not persons.loc[
            persons["_merge"] == "left_only", "HH1"].isin(hh.loc[hh.hhweight <= 0, "HH1"]).all():
        raise SystemExit("hl.sav members with no household in hh.sav")
    persons = persons[persons["_merge"] == "both"]
    cl = iv.groupby("reg")["HH1"].nunique()
    regs = sorted(td_rgph.T507)
    if set(cl.index) | set(BORROW) != set(regs) or set(BORROW) & set(cl.index):
        raise SystemExit(f"crosswalk: MICS régions {sorted(cl.index)}")
    print(f"1. {len(hh):,} households, {N_IV:,} interviewed, {len(hl):,} members, all matched; "
          f"{iv.HH1.nunique()} clusters, {cl.min()}-{cl.max()} per région (Ennedi pooled), "
          f"none in {', '.join(BORROW)}")

    # 2. HC1B against HH16, HH15 and WM14
    agree = (iv.HC1B == iv.HH16).mean()
    w = persons.groupby("HC1B")["hhweight"].sum()
    w16 = persons.groupby("HH16")["hhweight"].sum()
    a1, a16 = w[2] / w.sum() * 100, w16[2] / w16.sum() * 100
    i15 = (iv.HH15 == 2).mean() * 100
    print(f"2. HC1B = HH16 in {agree * 100:.1f}% of households; Chadian Arabic {a1:.2f}% of persons "
          f"by HC1B, {a16:.2f}% by HH16; {i15:.0f}% of interviews (HH15) in Chadian Arabic")
    assert agree >= 0.90 and abs(a1 - a16) < 1.0
    diffs = {}
    for r, d in persons.groupby("reg"):
        x = d.groupby("HC1B")["hhweight"].sum()
        y = d.groupby("HH16")["hhweight"].sum()
        diffs[r] = (x.get(2, 0) / x.sum() * 100, y.get(2, 0) / y.sum() * 100)
    big = {r: v for r, v in diffs.items() if abs(v[0] - v[1]) >= 3}
    print("   Arabic HC1B vs HH16 per région, 3+ points apart: " + ", ".join(
        f"{r} {a:.1f} / {b:.1f}" for r, (a, b) in big.items()))
    assert all(abs(a - b) < 10 for a, b in diffs.values())
    ww = wm[wm.wmweight > 0].groupby("WM14")["wmweight"].sum()
    print("   women 15-49, own mother tongue (WM14): " + ", ".join(
        f"{lab['HC1B'].get(k, k)} {v / ww.sum() * 100:.1f}" for k, v in ww.sort_values(ascending=False).items()
        if v / ww.sum() >= 0.02))

    # 3. who answered Arabic, who answered "other"
    ar_hc2 = iv[iv.HC1B == 2].HC2.map(hc2).value_counts()
    print(f"3. heads naming Chadian Arabic: {int(ar_hc2.sum()):,} households, "
          f"{int(ar_hc2.get('ARABE', 0)):,} of them Arab; non-Arab {int(ar_hc2.sum() - ar_hc2.get('ARABE', 0)):,} "
          f"(" + ", ".join(f"{k} {v}" for k, v in ar_hc2.drop('ARABE').head(6).items()) + ")")

    # 4. Arab heads who answered "other": an interviewer convention
    arab = iv[iv.HC2 == 2].copy()
    arab["oth"] = arab.HC1B == OTHER
    # Inside one cluster (one village, one team) interviewers should agree; a coding habit shows as
    # disagreement there, which geography cannot explain
    from scipy.stats import chi2, chi2_contingency
    stat = dof = 0.0
    split_cl, split_n = 0, 0
    for _, d in arab.groupby("HH1"):
        if d.HH3.nunique() > 1 and d.oth.nunique() > 1:
            s_, _, df_, _ = chi2_contingency(pd.crosstab(d.HH3, d.oth), correction=False)
            stat, dof = stat + s_, dof + df_
        r_ = d.groupby("HH3")["oth"].agg(["mean", "size"])
        r_ = r_[r_["size"] >= 2]
        if (r_["mean"] == 1).any() and (r_["mean"] == 0).any():
            split_cl, split_n = split_cl + 1, split_n + int(d.oth.sum())
    pv = chi2.sf(stat, dof)
    n_oth = int(arab.oth.sum())
    in_ar = (arab[arab.oth].HH15 == 2).mean()
    print(f"4. Arab-headed households answering 'other': {n_oth} of {len(arab):,}. Interviewers in "
          f"the same cluster disagree (within-cluster interviewer x answer, p = {pv:.0e}); in "
          f"{split_cl} clusters one interviewer coded all their Arab households 'other' and another "
          f"none ({split_n} of the {n_oth}); {in_ar * 100:.0f}% interviewed in Chadian Arabic -> drawn "
          f"as Chadian Arabic")
    assert pv < 1e-20 and split_n / n_oth > 0.25 and in_ar >= 0.9

    # 5. persons per (région, written label)
    N, autres, rho = census_weights()
    near = nearness(regs)
    ri = {r: i for i, r in enumerate(regs)}
    named_nat = persons[persons.HC1B.isin([3, 12])]["hhweight"].sum() / persons[
        persons.HC1B != NO_ANSWER]["hhweight"].sum()
    r_sara = 1 - named_nat / (N["Sara"] / td_rgph.TOTAL)
    assert 0.05 < r_sara < 0.5, r_sara
    N[SARA_REST] = r_sara * N["Sara"]
    print(f"5. census weights: Arabic move rho {rho:.3f}; Sara row less MICS's Ngambay + Sar "
          f"({named_nat * 100:.1f}% of persons): {r_sara:.2f} of the census's Sara")
    for grp, a in sorted(autres.items(), key=lambda kv: -kv[1]):
        print(f"     'Autres' estimate {grp[:44]:<44} {a:>9,.0f}")

    def weight(c, grp, i):
        if c in (REM, MUSGUM, TAMAIC):
            base = autres[grp]
            k = near[("rem", grp)][i] if ("rem", grp) in near else (
                near[c][i] if c in near else 1 / len(regs))
        else:
            base = N[c]
            k = near[c][i] if c in near else 1 / len(regs)
        return base * k

    p = persons[persons.HC1B != NO_ANSWER]
    table = {}   # reg -> {label: weighted persons}
    route = {}   # (reg, label) -> {source: persons}
    for (r, code, eth), s in p.groupby(["reg", "HC1B", "HC2"])["hhweight"].sum().items():
        if not s:
            continue
        code = int(code)
        if code in NAMED:
            parts = {NAMED[code]: s}
            src = "HC1B"
        else:
            g = hc2.get(int(eth), "NON REPONSE")
            grp, cands = SPLIT[g]
            if len(cands) == 1:
                parts = {cands[0]: s}
            else:
                wt = {c: weight(c, grp, ri[r]) for c in cands}
                tw = sum(wt.values())
                parts = {c: s * x / tw for c, x in wt.items()}
            src = f"other x {g}"
        for k, x in parts.items():
            table.setdefault(r, {})[k] = table.setdefault(r, {}).get(k, 0) + x
            rt = route.setdefault((r, k), {})
            rt[src] = rt.get(src, 0) + x
    for r, b in BORROW.items():
        table[r] = dict(table[b])
        for (rr, k), v in list(route.items()):
            if rr == b:
                route[(r, k)] = {f"{src} (Borkou's shares)": x for src, x in v.items()}

    # 6. counts
    rd = pd.read_csv(RD / "data" / "normalized" / "td.csv")
    rdpop = rd.groupby("geo_id")["count"].sum()
    assert all(rdpop.get(k) == v for k, v in td_rgph.T507.items())
    n_hh = iv.groupby("reg").size()
    out, natl = [], {}
    for r in regs:
        v = table[r]
        tot = sum(v.values())
        cnt = lr_round(v, td_rgph.T507[r])
        for k in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[k]:
                continue
            natl[k] = natl.get(k, 0) + cnt[k]
            src = route[(r, k)]
            how = "; ".join(f"{s_} {x / tot * 100:.2f}%" for s_, x in
                            sorted(src.items(), key=lambda kv: -kv[1]))
            hh_n = n_hh.get(BORROW.get(r, r))
            out.append(dict(geo_id=r, geo_level="region", geo_name=r, source_category=k,
                            count=cnt[k], tier="modelled", source_id=SOURCE_ID, year=2019,
                            note=f"MICS6 2019, % of persons in {hh_n} interviewed households "
                                 f"(weighted): {how}; 2009 census population {td_rgph.T507[r]}"))
    total = sum(natl.values())
    assert total == td_rgph.T507_TOTAL, total

    pct = {k: v / total * 100 for k, v in natl.items()}
    arabic = pct.get("ARABE TCHADIEN", 0) + pct.get(AR, 0)
    sara = sum(pct.get(k, 0) for k in ("NGAMBAYE", "SAR", SARA_REST, "Sara Kaba", "Daye", "Mboum"))
    eth = persons.groupby("HC2")["hhweight"].sum()
    sara_e = eth[13] / eth.sum() * 100
    arab_e = eth[2] / eth.sum() * 100
    print(f"6. Chadian Arabic {arabic:.1f}% (MICS Arab ethnic share {arab_e:.1f}%; census 2009 first "
          f"language named 21.7%, Arab ethnic 12.9%); Sara group {sara:.1f}% (MICS Sara ethnic "
          f"{sara_e:.1f}%)")
    assert abs(sara - sara_e) < 3
    assert arab_e - 1 < arabic < 21.7

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                           "count", "tier", "source_id", "year", "note"])
        wr.writeheader()
        wr.writerows(out)
    tmp.replace(OUT)
    print(f"   wrote {OUT.relative_to(HERE)}: {len(out)} rows, {total:,} people in 22 régions")
    for k, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<30} {n:>11,}  {n / total * 100:5.2f}%")
    print("   per région, % (top five):")
    for r in regs:
        v = table[r]
        t = sum(v.values())
        top = sorted(v.items(), key=lambda kv: -kv[1])[:5]
        print(f"     {r:<18} " + ", ".join(f"{k} {x / t * 100:.0f}" for k, x in top))


if __name__ == "__main__":
    main()
