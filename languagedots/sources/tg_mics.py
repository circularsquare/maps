"""Togo: mother tongue of the household head by region from MICS6 2017 microdata, split inside
each MICS language group by Afrobarometer's finer answers, as shares applied to the 2022 census
region populations -> data/normalized/tg.csv.

    python sources/wafr_afro.py tg     Afrobarometer respondents (once; read-only .sav)
    python sources/tg_mics.py

SOURCE. Togo Multiple Indicator Cluster Survey 2017 (MICS6; INSEED, UNICEF), SPSS files from
mics.unicef.org (Anita's UNICEF account, 2026-10-09), unzipped to data/raw/tg/mics_2017/
(gitignored; research use, no redistribution, and the readme asks that copies of reports and
publications go to INSEED and UNICEF Togo). 8,404 households sampled, 7,916 interviewed, 34,988
members, 420 clusters: 60 in each of seven strata (HH7).

ITEM. HC1B, "Langue maternelle du chef de ménage", read as every member's (hl.sav members x
hhweight). Eleven answers: EWE/MINA, KABYE, MOBA-GOURMA, KOTOKOLI/TEM, BASSAR/KONKOMBA,
AKPOSSO/AKEBOU, IFE/ANA, TCHOKOSSI, AUTRES LANGUES NATIONALES, LANGUES ETRANGERES, FRANCAIS.
HH16 (the respondent's mother tongue) agrees household by household 96% of the time and does
not slide towards the interview language as Iraq's did (2,203 interviews were in French, 29
respondents named French); WM14 and MWM14 (women and men 15-49) are printed as further readings.
HC1B is used because it is MICS's standard household item and the one Iraq used.

HOW IT ENTERS: MICS SETS EACH GROUP'S SHARE, AFROBAROMETER SPLITS IT. MICS's groups are coarse,
so it cannot replace Afrobarometer R5-R7's 24 answers (sources/tg_afro.py). Per region, each MICS
group gets MICS's share; the Afrobarometer languages filed under it share that out in their
own proportions in that region (French answers moved to the ethnic language first, as before).
MICS rather than Afrobarometer for the group shares, and not a pool of the two: MICS samples
every household member, children and foreign residents included (Afrobarometer samples adult
citizens, which is why it finds almost no foreign languages; MICS finds 5%), and has 60
clusters in every stratum and about 4,400 persons per region against Afrobarometer's
proportional 370 to 1,070 adults; a pool would need MICS's groups anyway, where MICS dominates.

GROUPS (GROUP below). EWE/MINA holds every Gbe answer (Ewe, Mina, Ouatchi, Aja, Fon): 2,854 of
the 2,935 heads of Adja-Ewe ethnic group answer EWE/MINA and 26 "other national", and Maritime,
where Afrobarometer has 9% Ouatchi, has 5 of 1,096 households on "other national". Every
Afrobarometer language MICS does not name goes under AUTRES LANGUES NATIONALES: Nawdm, Lama,
Akaselem (Tchamba), Fulfulde, Hausa, Yoruba (Afrobarometer's respondents are citizens), the
card's own "Other", and Ngangam, a Gurma language that MOBA-GOURMA does not name; Savanes bears
this out (Moba + Gourma 79% and the unnamed rest 13% in Afrobarometer, MOBA-GOURMA 67% and other
national 12% in MICS).

SPLIT RULE. A group's split in a region is that region's Afrobarometer split when at least
MIN_RESP respondents there gave one of its languages. Otherwise a named group (two languages)
takes the national split (Afrobarometer shares x region populations), and AUTRES LANGUES
NATIONALES goes on "Other" (africa_other) unsplit.

FRENCH (FRENCH = "home"; Anita's standing rule, 2026-10-09: the map leans towards the language
spoken at home). French is the one answer where mother tongue and home language part ways in
Togo: MICS's mother-tongue item gives 0.32% and Afrobarometer R7's mother-tongue question
(ask 018's figure) 1.5%, but Afrobarometer R8-R9 (2021-22, 2024), which ask the language
spoken at home, give about 4.2%. French is drawn at R8-R9's share per unit (french_home()),
shrunk to the national share by 50 respondents where a unit is thin; each unit's other rows
are scaled pro rata to what is left. FRENCH = "r7" or "mics" draws the other two readings. The
check that drawn groups equal MICS's runs before this step.

LANGUES ETRANGERES (4.7% of persons): one label, "Foreign language", on africa_other. 236 of the
308 heads give "other nationalities" as ethnic group, 179 are Muslim, 103 live in rural
households in every region (Togo's rural foreign residents come from the neighbouring
countries), so it is nearly all African languages; Afrobarometer, which samples citizens,
cannot split it.

UNITS. religiondots' six (tg_lookup.csv, 2022 census): MICS's Lomé Commune is "Lomé (Golfe 1 to
5)" (religiondots draws it on COD-AB's Lome Commune); MICS's Golfe Urbain (Lomé's suburbs in
Golfe prefecture) and Maritime together are the rest of Maritime, pooled with their weights.

CHECKS (asserted). hh/hl/wm row counts; every member matches a household; the crosswalk covers
all seven strata and all six units; HC1B and HH16 agree for >= 95% of households and within
3 points per group per unit (person-weighted); drawn group sums equal MICS's group shares;
unit totals equal the 2022 populations.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD_GEO  # noqa: E402
import tg_afro  # noqa: E402

RAW = HERE / "data" / "raw" / "tg" / "mics_2017"
OUT = HERE / "data" / "normalized" / "tg.csv"
LOOKUP = RD_GEO / "tg" / "tg_lookup.csv"          # religiondots, read-only: 2022 census
SOURCE_ID = "mics6_2017_hc1b_afro_split"
CENSUS_2022 = 8_095_498
N_HH, N_INT, N_HL = 8_404, 7_916, 34_988
MIN_RESP = 5
# French's drawn share: "home" = Afrobarometer R8-R9 home language (Anita's rule, 2026-10-09);
# "r7" = R7 mother tongue (ask 018's figure, 1.5%); "mics" = MICS HC1B (0.32%)
FRENCH = "home"
K_SHRINK = 50             # respondents; as wafr_afro.K_SHRINK

# MICS stratum (HH7, any case) -> religiondots unit
UNIT = {"lomé commune": "TG0305", "lome commune": "TG0305", "golfe urbain": "TG03",
        "maritime": "TG03", "plateaux": "TG04", "centrale": "TG01", "kara": "TG02",
        "savanes": "TG05"}


def key(s):
    """MICS answer -> one spelling (HC1B, HH16, WM14 and MWM14 differ in spaces and dashes)."""
    s = str(s).upper().replace(" ", "").replace("-", "/")
    return {"AUTRESLANGUESETRANGERES": "LANGUESETRANGERES"}.get(s, s)


EWE, KAB, MOBA, TEM = "EWE/MINA", "KABYE", "MOBA/GOURMA", "KOTOKOLI/TEM"
BAS, AKP, IFE, TCH = "BASSAR/KONKOMBA", "AKPOSSO/AKEBOU", "IFE/ANA", "TCHOKOSSI"
AUT, ETR, FRA = "AUTRESLANGUESNATIONALES", "LANGUESETRANGERES", "FRANCAIS"
GROUPS = [EWE, KAB, MOBA, TEM, BAS, AKP, IFE, TCH, AUT, ETR, FRA]
NAMED = {EWE, MOBA, BAS, AKP}         # groups naming more than one language
# Afrobarometer answer (tg_afro.LANG's spellings) -> MICS group
GROUP = {
    "Ewe": EWE, "Mina (Gen)": EWE, "Ouatchi (Waci)": EWE, "Aja": EWE, "Fon": EWE,
    "Kabiyè": KAB, "Moba": MOBA, "Gourmanchéma": MOBA, "Tem (Kotokoli)": TEM,
    "Ntcham (Bassar)": BAS, "Konkomba": BAS, "Ikposo (Akposso)": AKP, "Akebu": AKP,
    "Ifè (Ana)": IFE, "Anufo (Tchokossi)": TCH, "French": FRA,
    "Nawdm (Losso)": AUT, "Lama (Lamba)": AUT, "Tchamba (Akaselem)": AUT,
    "Ngangam (Gangam)": AUT, "Fulfulde": AUT, "Hausa": AUT, "Yoruba": AUT, "Other": AUT,
}
SINGLE = {KAB: "Kabiyè", TEM: "Tem (Kotokoli)", IFE: "Ifè (Ana)", TCH: "Anufo (Tchokossi)",
          FRA: "French", ETR: "Foreign language"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def mics_shares(d, col, w):
    """{unit: Series(group -> share)} for one item, person- or respondent-weighted."""
    d = d[(d[w] > 0) & d[col].notna()].copy()
    d["u"] = d["HH7"].astype(str).str.lower().map(UNIT)
    say(d["u"].notna().all(), f"{col}: every stratum in the crosswalk "
        f"{sorted(d.loc[d['u'].isna(), 'HH7'].astype(str).unique())}")
    d["g"] = d[col].map(key)
    miss = sorted(set(d["g"]) - set(GROUPS))
    say(not miss, f"{col}: every answer is a MICS group {miss}")
    t = d.groupby(["u", "g"], observed=True)[w].sum()
    return {u: (s.droplevel(0) / s.sum()).reindex(GROUPS, fill_value=0.0)
            for u, s in t.groupby(level=0)}


def afro():
    """Afrobarometer R5-R7 weighted shares and respondent counts per unit, French moved as in
    tg_afro.py (French's final share is set later, in main(), by FRENCH_AT_AFRO)."""
    a = tg_afro.load()
    sh, _ = tg_afro.shares(a, (5, 6, 7), True)
    s = a[a["round"].isin((5, 6, 7))].copy()
    fr = (s["l"] == "French") & s["e"].notna() & ~s["e"].isin(["French", "Other"])
    s.loc[fr, "l"] = s.loc[fr, "e"]
    n = s.groupby(["unit", "l"]).size()
    miss = sorted(set(sh.index.get_level_values(1)) - set(GROUP))
    say(not miss, f"every Afrobarometer answer has a MICS group {miss}")
    return sh, n


def french_home():
    """Afrobarometer R8-R9 (2021-22, 2024; "language spoken in home") French share per unit,
    weighted, shrunk to the national share by K_SHRINK respondents:
    (French_r + K * national) / (n_r + K). French answers are taken as given (no move)."""
    a = tg_afro.load()
    s = a[a["round"].isin((8, 9))]
    w = s.groupby("unit")["w"].sum()
    f = s[s["l"] == "French"].groupby("unit")["w"].sum().reindex(w.index, fill_value=0.0)
    n = s.groupby("unit").size()
    nat = f.sum() / w.sum()
    # weights rescaled to respondents per unit, so K is in respondents
    fw, nw = f / w * n, n
    out = (fw + K_SHRINK * nat) / (nw + K_SHRINK)
    say(len(out) == 6, f"R8-R9 home-language answers in all six units ({len(out)})")
    print(f"  R8-R9 home-language French: {nat:.2%} of {int(n.sum())} respondents nationally; "
          + ", ".join(f"{u} {f[u] / w[u]:.1%} of {int(n[u])} -> {out[u]:.1%}" for u in out.index))
    return out


def main():
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HH7", "HH15", "HH16", "HC1A", "HC1B",
                                         "HC2", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2", "HL1"])
    wm, _ = pyreadstat.read_sav(str(RAW / "wm.sav"), apply_value_formats=True,
                                usecols=["HH7", "WM14", "wmweight"])
    mn, _ = pyreadstat.read_sav(str(RAW / "mn.sav"), apply_value_formats=True,
                                usecols=["HH7", "MWM14", "mnweight"])
    say(len(hh) == N_HH and int((hh.hhweight > 0).sum()) == N_INT and len(hl) == N_HL,
        f"{len(hh):,} households, {int((hh.hhweight > 0).sum()):,} interviewed, "
        f"{len(hl):,} members")
    p = hl.merge(hh, on=["HH1", "HH2"], how="left", indicator=True)
    say((p["_merge"] == "both").all(), "every hl.sav member has a household in hh.sav")
    say(set(hh["HH7"].astype(str).str.lower()) <= set(UNIT), "seven strata, all crosswalked")

    lut = pd.read_csv(LOOKUP, dtype={"unit": str})
    pop = lut.set_index("unit")["pop"].astype(int)
    names = lut.set_index("unit")["name"]
    say(set(pop.index) == set(UNIT.values()) and int(pop.sum()) == CENSUS_2022,
        f"crosswalk covers religiondots' six units, {int(pop.sum()):,} people")

    # ---- which item: HC1B against HH16, household by household, and the interview language
    h = hh[hh.hhweight > 0]
    same = (h["HC1B"].map(key) == h["HH16"].map(key))
    say(same.mean() >= 0.95, f"HC1B = HH16 for {same.mean():.1%} of interviewed households")
    fr_int = h["HH15"].astype(str) == "FRANCAIS"
    print(f"  interviews in French {int(fr_int.sum()):,}; HH16 French among them "
          f"{int((h.loc[fr_int, 'HH16'].map(key) == FRA).sum())}, HC1B French "
          f"{int((h.loc[fr_int, 'HC1B'].map(key) == FRA).sum())}")
    ew_int = h["HH15"].map(key) == EWE
    nonew = ew_int & (h["HC1B"].map(key) != EWE)
    print(f"  heads not EWE/MINA in Ewe/Mina interviews {int(nonew.sum())}: respondent EWE/MINA "
          f"{int((h.loc[nonew, 'HH16'].map(key) == EWE).sum())}")
    fe = h[h["HC1B"].map(key) == ETR]
    print(f"  LANGUES ETRANGERES heads {len(fe)}: ethnic group "
          f"{fe['HC2'].astype(str).value_counts().head(3).to_dict()}; Muslim "
          f"{int((fe['HC1A'].astype(str) == 'MUSULMANE').sum())}")

    head = mics_shares(p, "HC1B", "hhweight")
    resp = mics_shares(p, "HH16", "hhweight")
    women = mics_shares(wm, "WM14", "wmweight")
    men = mics_shares(mn, "MWM14", "mnweight")
    say(set(head) == set(pop.index), "HC1B answers in all six units")
    worst = max((abs(head[u][g] - resp[u][g]), u, g) for u in head for g in GROUPS)
    say(worst[0] <= 0.03, f"HC1B vs HH16 within 3 points everywhere (largest "
        f"{worst[0] * 100:.1f}, {names[worst[1]]} {worst[2]})")
    clusters = h.assign(u=h["HH7"].astype(str).str.lower().map(UNIT)).groupby("u")["HH1"].nunique()
    print("  % of persons: head HC1B | respondent HH16 | women WM14 | men MWM14  (clusters)")
    for u in pop.index:
        print(f"    {names[u]:<20} ({clusters[u]:3d}) " + ", ".join(
            f"{g.split('/')[0][:8]} {head[u][g] * 100:.1f}|{resp[u][g] * 100:.1f}|"
            f"{women[u][g] * 100:.1f}|{men[u][g] * 100:.1f}"
            for g in GROUPS if head[u][g] >= 0.02))

    # ---- Afrobarometer inside each group
    sh, n = afro()
    natl = (sh * pop.reindex(sh.index.get_level_values(0)).values).groupby(level=1).sum()
    drawn, how = {}, []
    for u in pop.index:
        su = sh.loc[u]
        nu = n.loc[u] if u in n.index.get_level_values(0) else pd.Series(dtype=float)
        d = {}
        for g in GROUPS:
            m = head[u][g]
            if m <= 0:
                continue
            if g in SINGLE:
                d[SINGLE[g]] = d.get(SINGLE[g], 0) + m
                continue
            mem = [l for l in su.index if GROUP[l] == g]
            k = int(nu.reindex(mem).fillna(0).sum())
            if k >= MIN_RESP:
                part = su[mem]
            elif g in NAMED:
                part = natl[[l for l in natl.index if GROUP[l] == g]]
                how.append(f"{names[u]} {g}: {k} respondents, national split")
            else:
                part = pd.Series({"Other": 1.0})
                how.append(f"{names[u]} {g}: {k} respondents, on Other")
            for l, v in (part / part.sum() * m).items():
                d[l] = d.get(l, 0) + v
        drawn[u] = pd.Series(d)
        # drawn group sums equal MICS's group shares
        back = drawn[u].groupby(lambda l: GROUP.get(l, ETR)).sum()
        say(all(abs(back.get(g, 0) - head[u][g]) < 1e-9 for g in GROUPS),
            f"{names[u]}: drawn groups equal MICS's")
    for line in how:
        print("  fallback: " + line)

    # ---- French at Afrobarometer's home-language share (FRENCH; Anita, 2026-10-09), the rest
    # of each unit pro rata
    mics_fr = {u: float(drawn[u].get("French", 0)) for u in pop.index}
    if FRENCH == "r7":
        ms = tg_afro.french_at_r7(pd.concat({u: drawn[u] for u in pop.index}))
        drawn = {u: ms.loc[u] for u in pop.index}
    elif FRENCH == "home":
        fr = french_home()
        for u in pop.index:
            s = drawn[u].drop("French", errors="ignore")
            s = s / s.sum() * (1 - fr[u])
            s["French"] = fr[u]
            drawn[u] = s
    else:
        say(FRENCH == "mics", f"FRENCH is home, r7 or mics, not {FRENCH!r}")
    print("  French, % of unit: MICS HC1B -> drawn")
    for u in pop.index:
        print(f"    {names[u]:<20} {mics_fr[u] * 100:.2f} -> "
              f"{drawn[u].get('French', 0) / drawn[u].sum() * 100:.2f}")

    # ---- counts, largest remainder
    rows, nat = [], {}
    for u in pop.index:
        s = drawn[u] / drawn[u].sum()
        raw = s * pop[u]
        cnt = raw.astype(int)
        for l in (raw - cnt).sort_values(ascending=False).index[:pop[u] - int(cnt.sum())]:
            cnt[l] += 1
        say(int(cnt.sum()) == pop[u], f"{names[u]} {int(cnt.sum()):,} = 2022 population")
        n_hh = int(((h["HH7"].astype(str).str.lower().map(UNIT)) == u).sum())
        for l in cnt.sort_values(ascending=False).index:
            if cnt[l] <= 0:
                continue
            nat[l] = nat.get(l, 0) + int(cnt[l])
            g = GROUP.get(l, ETR)
            rows.append(dict(geo_id=u, geo_level="region", geo_name=names[u],
                             source_category=l, count=int(cnt[l]), tier="modelled",
                             source_id=SOURCE_ID, year="2017 (shares), 2022 (population)",
                             note=f"share {s[l]:.5f}; MICS6 HC1B group {g} "
                                  f"{head[u][g] * 100:.2f}% of persons in {n_hh} interviewed "
                                  f"households" + ("" if g in SINGLE else
                                                   ", split by Afrobarometer R5-R7")
                                  + ({"home": "; French at Afrobarometer R8-R9's home-language "
                                              "share, the unit's other rows scaled to the rest",
                                      "r7": "; French at Afrobarometer R7's mother-tongue "
                                            "share, the unit's other rows scaled to the rest"}
                                     .get(FRENCH, ""))))

    # ---- before (sources/tg_afro.py's build) and after
    before = pd.read_csv(HERE / "data" / "normalized" / "tg_afro.csv")
    b = before.groupby(["geo_id", "source_category"])["count"].sum()
    a = pd.DataFrame(rows).groupby(["geo_id", "source_category"])["count"].sum()
    print("  % per unit, before (Afrobarometer, tg_afro.csv) -> after:")
    for u in pop.index:
        bu, au = b.loc[u] / pop[u] * 100, a.loc[u] / pop[u] * 100
        top = au.add(bu, fill_value=0).sort_values(ascending=False).index[:7]
        print(f"    {names[u]:<20} " + ", ".join(
            f"{l} {bu.get(l, 0):.1f}->{au.get(l, 0):.1f}" for l in top))
    bn = b.groupby(level=1).sum()
    print("  national:")
    for l in sorted(set(nat) | set(bn.index), key=lambda x: -nat.get(x, 0)):
        print(f"      {l:<22} {int(bn.get(l, 0)):>10,} -> {nat.get(l, 0):>10,}  "
              f"{bn.get(l, 0) / CENSUS_2022 * 100:5.2f}% -> {nat.get(l, 0) / CENSUS_2022 * 100:5.2f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(rows)} rows, {sum(nat.values()):,} people, "
          f"{len(nat)} answers")


if __name__ == "__main__":
    main()
