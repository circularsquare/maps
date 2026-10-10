"""DR Congo: the household head's mother tongue by province from MICS-Palu 2017-18 microdata (the
four national languages, French and "another language"), with "another language" split among
the local languages by the Enquete 1-2-3 ethnic model, per territoire -> data/normalized/cd_mics.csv

    python sources/cd_mics.py           (needs data/normalized/cd.csv from sources/cd_e123.py)

SOURCE. Enquete par grappes a indicateurs multiples et de paludisme (MICS-Palu) 2017-18, MICS6
(Institut National de la Statistique, UNICEF), SPSS files from mics.unicef.org (Anita's UNICEF
account, 2026-10-09), unzipped to data/raw/cd/mics_2017/ (gitignored; research use, no
redistribution, copies of publications to the INS and UNICEF Kinshasa). 20,810 households in 26
provinces (HH7), 20,792 interviewed, 103,422 members.

ITEM. HC1B, "Langue maternelle du Chef de menage" (Francais / Kikongo / Lingala / Swahili /
Tshiluba / Autre langue), read as every member's: hl.sav members x hhweight, so the shares are of
persons. HH16 (the household respondent's own mother tongue, same codes) and the women's and
men's own answers (WM14, MWM14) are printed as checks, not drawn:
  * HC1B and HH16 agree in 93% of households, and on each national language within 10 points per
    province (asserted; the widest, Ituri, Swahili 29% against 38%). Unlike
    Iraq, HH16 does not slide to the interview language (HH15): of the 10,834 heads with "another
    language", 9,956 respondents also said another language, though 4,076 interviews were in
    Lingala and 2,994 in Swahili.
  * The individual interviews do slide. Women who head their household answered both HC1B (as the
    household respondent) and WM14 (in their own interview): where HC1B said "another language",
    13% named a national language in WM14. For male heads the same test gives 6%. So WM14 and
    MWM14, though asked of each person, add a few points of lingua franca that the same person did
    not give the household questionnaire; they are not used.
  * Sons and daughters 15-49 of heads with "another language" name a national language more
    often (14% of sons, 19% of daughters), a few points above that drift: the head's answer leans
    to the older generation, most in Kinshasa (heads 46.5% Lingala; HH16 54%; women 57%, men 62%).

HOW IT IS LAID ON THE TERRITOIRES. MICS is representative by province and by urban/rural (HH6).
  * Kinshasa is one unit and takes the province's shares.
  * The 18 cities that are units of their own (VILLES) take their province's urban-stratum shares
    as measured.
  * The province's other territoires share the remainder (the province's HC1B shares x its COD-PS
    2024 population, less the cities), by an iterative proportional fit over territoires x the six
    categories, row totals each territoire's COD-PS population (religiondots' cd_territoires.csv).
    The seed is flat except where the Enquete 1-2-3's groups account for a category (their
    province share at least half of MICS's): "another language" by the territoire's share of
    non-national-language groups over the province's, Tshiluba by Luba-Kasai and its clans,
    Kikongo by the Kongo varieties (Kongo Central only). Lingala, Swahili and French are spread by
    population. Seed floor 1/1000.
  * MICS's urban weights do not match COD-PS's cities (Kasai-Oriental is 41% urban in MICS, while
    Mbuji-Mayi alone is 63% of COD-PS): where the cities alone exceed a province's total of a
    category, the remainder is clipped at 0 (printed; under 1.5% of the rest of the province).
Then within each territoire: "another language" is split among the territoire's non-national
ethnic groups in the Enquete 1-2-3's proportions (the province's where the territoire has none);
Tshiluba is Luba-Kasai; Kikongo is the Kongo varieties in the territoire's ethnic proportions in
the two provinces where Kikongo is over 5% and ethnic Kongo heads are at least half of it
(Kinshasa, Kongo Central; KONGO_PROVINCES, asserted). Elsewhere Kikongo goes to the Kongo
varieties only up to the territoire's ethnic Kongo share, and the rest to Kikongo ya leta
(Kituba): Kwilu and Kwango's 11-15%, where hardly any heads are ethnic Kongo.

COUNTS. Every territoire's whole COD-PS 2024 population, largest remainder (the old build left
out the 2.3% of heads with no ethnic group; MICS answered for all but 16 households, so nobody
is left out now). Tier `modelled`.

CHECKS (asserted). hh/hl row counts; every member matches a household; 26 provinces, each
mapped to one religiondots province with every territoire covered; HC1B vs HH16 per province;
the fit reproduces every row total; the national totals drawn equal the provinces' HC1B shares x
COD-PS (within 0.2% of the total, the clipping), and are within 3 points of MICS's own national figures (which weight the
provinces differently: Autre 43.1% drawn, 41.0% in MICS); every territoire sums to its
population.
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
sys.path.insert(0, str(HERE / "taxonomy"))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "cd" / "mics_2017"
E123 = HERE / "data" / "normalized" / "cd.csv"
OUT = HERE / "data" / "normalized" / "cd_mics.csv"
TERR = RD_GEO / "cd" / "cd_territoires.csv"        # religiondots, read-only
SOURCE_ID = "mics_2017_hc1b"
N_HH, N_HL, N_IV = 20_810, 103_422, 20_792

PROV = {"Kinshasa": "CD10", "Kongo Central": "CD20", "Kwango": "CD31", "Kwilu": "CD32",
        "Maindombe": "CD33", "Equateur": "CD41", "Sud Ubangi": "CD42", "Nord Ubangi": "CD43",
        "Mongala": "CD44", "Tshuapa": "CD45", "Tshopo": "CD51", "Bas Uele": "CD52",
        "Haut Uele": "CD53", "Ituri": "CD54", "Nord Kivu": "CD61", "Sud Kivu": "CD62",
        "Maniema": "CD63", "Haut Katanga": "CD71", "Lualaba": "CD72", "Haut Lomami": "CD73",
        "Tanganyika": "CD74", "Lomami": "CD81", "Kasai Oriental": "CD82", "Sankuru": "CD83",
        "Kasai Central": "CD91", "Kasai": "CD92"}
CAT = {"FRANÇAIS": "French", "KIKONGO": "Kikongo", "LINGALA": "Lingala", "SWAHILI": "Swahili",
       "TSHILUBA": "Tshiluba", "AUTRE LANGUE": "Autre"}
CATS = ["French", "Kikongo", "Lingala", "Swahili", "Tshiluba", "Autre"]
NATIONAL = ["Kikongo", "Lingala", "Swahili", "Tshiluba"]
# Cities that are units of their own in COD-AB (small populated area, checked on religiondots'
# hexes): they take the urban stratum's shares as their seed. Kolwezi, Tshikapa, Uvira, Bunia,
# Kalemie, Gemena, Isiro and Kabinda are inside territoires.
VILLES = {"CD1000": "Kinshasa", "CD2001": "Matadi", "CD2002": "Boma", "CD3201": "Bandundu",
          "CD3203": "Kikwit", "CD4101": "Mbandaka", "CD4206": "Zongo", "CD4301": "Gbadolite",
          "CD5101": "Kisangani", "CD6101": "Goma", "CD6109": "Beni", "CD6110": "Butembo",
          "CD6201": "Bukavu", "CD6301": "Kindu", "CD7101": "Lubumbashi", "CD7106": "Likasi",
          "CD8201": "Mbuji-Mayi", "CD9101": "Kananga"}
KONGO_PROVINCES = {"CD10", "CD20"}
FLOOR = 1e-3
LABEL = {"French": "French", "Lingala": "Lingala", "Swahili": "Swahili",
         "Tshiluba": "Tshiluba", "Kituba": "Kikongo (Kikongo ya leta)"}


def weighted(df, col, w):
    """{province: {category: weighted persons}}, NON REPONSE and blanks dropped."""
    out = {}
    d = df[(df[w] > 0) & df[col].notna()]
    for (g, a), s in d.groupby(["HH7", col], observed=True)[w].sum().items():
        if a not in CAT or not s:
            continue
        out.setdefault(PROV[g], {}).setdefault(CAT[a], 0.0)
        out[PROV[g]][CAT[a]] += s
    return out


def norm(v):
    t = sum(v.values())
    return {k: v.get(k, 0) / t for k in CATS}


def ipf(seed, rows, cols, tol=1e-10, it=5000):
    import numpy as np
    x = seed * (rows / seed.sum(axis=1))[:, None]
    for _ in range(it):
        c = x.sum(axis=0)
        x = x * np.divide(cols, c, out=np.zeros_like(cols), where=c > 0)[None, :]
        r = x.sum(axis=1)
        x = x * (rows / r)[:, None]
        if np.abs(x.sum(axis=0) - cols).max() <= tol * rows.sum():
            return x
    raise SystemExit("fit did not converge")


def main():
    import numpy as np
    import pandas as pd
    import pyreadstat
    import cd2012

    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HH6", "HH7", "HH15", "HH16", "HH47",
                                         "HC1B", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HL1", "HL3"])
    mn, _ = pyreadstat.read_sav(str(RAW / "mn.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "LN", "MWM14", "mnweight"])
    wm, _ = pyreadstat.read_sav(str(RAW / "wm.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "LN", "WM14", "wmweight"])
    if len(hh) != N_HH or len(hl) != N_HL or int((hh.hhweight > 0).sum()) != N_IV:
        raise SystemExit(f"hh {len(hh)} / hl {len(hl)} / interviewed "
                         f"{int((hh.hhweight > 0).sum())}, expected {N_HH} / {N_HL} / {N_IV}")
    for c in ("HH7", "HH6", "HH15", "HH16", "HC1B"):
        hh[c] = hh[c].astype(str)
    mn["MWM14"] = mn["MWM14"].astype(str).str.replace("?", "Ç", regex=False)
    wm["WM14"] = wm["WM14"].astype(str).str.replace("?", "Ç", regex=False)
    if set(hh["HH7"]) != set(PROV):
        raise SystemExit(f"HH7 provinces {sorted(set(hh['HH7']) ^ set(PROV))} not in PROV")
    persons = hl.merge(hh, on=["HH1", "HH2"], how="left", indicator=True)
    if (persons["_merge"] != "both").any():
        raise SystemExit("hl.sav members with no household in hh.sav")
    iv = hh[hh.hhweight > 0]
    nr = int((~iv["HC1B"].isin(CAT)).sum())
    print(f"  {len(hh):,} households, {len(iv):,} interviewed, {len(hl):,} members; "
          f"{nr} interviewed households with no HC1B answer (not drawn)")
    persons = persons[persons.hhweight > 0]

    # ---- the items against each other
    agree = (iv["HC1B"] == iv["HH16"]).mean()
    print(f"  HC1B = HH16 in {agree:.1%} of interviewed households")
    if agree < 0.90:
        raise SystemExit("HC1B and HH16 disagree in over 10% of households")
    aut = iv[iv["HC1B"] == "AUTRE LANGUE"]
    print(f"  heads with another language: {len(aut):,}; respondent also another language "
          f"{int((aut.HH16 == 'AUTRE LANGUE').sum()):,}; interviews in Lingala "
          f"{int((aut.HH15 == 'LINGALA').sum()):,}, Swahili {int((aut.HH15 == 'SWAHILI').sum()):,}")
    heads = hl[hl["HL3"].astype(str) == "CHEF"][["HH1", "HH2", "HL1"]]
    for who, d, col, w in (("men", mn, "MWM14", "mnweight"), ("women", wm, "WM14", "wmweight")):
        d = d[d[w] > 0].merge(heads, left_on=["HH1", "HH2", "LN"], right_on=["HH1", "HH2", "HL1"])
        d = d.merge(iv[["HH1", "HH2", "HC1B", "HH47"]], on=["HH1", "HH2"])
        d = d[d["HH47"] == d["LN"]]            # the head was also the household respondent
        a = d[d["HC1B"] == "AUTRE LANGUE"]
        lf = a[col].isin([k for k, v in CAT.items() if v in NATIONAL]).mean()
        print(f"  same person, {who} heads who answered both HC1B and their own interview: "
              f"{len(d):,}, agreement {(d['HC1B'] == d[col]).mean():.1%}; HC1B another language "
              f"-> own interview a national language {lf:.1%}")

    head = weighted(persons, "HC1B", "hhweight")
    resp = weighted(persons, "HH16", "hhweight")
    urb = weighted(persons[persons.HH6 == "URBAIN"], "HC1B", "hhweight")
    clusters = iv.groupby("HH7")["HH1"].nunique()
    uclusters = iv[iv.HH6 == "URBAIN"].groupby("HH7")["HH1"].nunique()
    bad = []
    print("  per province, % of persons, HC1B (HH16 where 3+ points apart); clusters (urban)")
    for g, p in sorted(PROV.items(), key=lambda kv: kv[1]):
        h, r = norm(head[p]), norm(resp[p])
        for c in NATIONAL:
            if abs(h[c] - r[c]) > 0.10:
                bad.append(f"{g} {c} HC1B {h[c]:.1%} HH16 {r[c]:.1%}")
        print(f"    {p} {g:<15}" + " ".join(
            f"{c[:3]} {h[c] * 100:4.1f}" + (f" ({r[c] * 100:.0f})" if abs(h[c] - r[c]) >= .03 else "")
            for c in CATS) + f"   {clusters[g]} ({uclusters.get(g, 0)})")
    if bad:
        raise SystemExit("HC1B and HH16 more than 10 points apart: " + "; ".join(bad))
    tot = {c: sum(head[p].get(c, 0) for p in head) for c in CATS}
    natl_mics = {c: tot[c] / sum(tot.values()) for c in CATS}
    print("  national, MICS weights: " + ", ".join(f"{c} {v:.1%}" for c, v in natl_mics.items()))

    # ---- the ethnic model and the territoires
    ter = pd.read_csv(TERR, dtype={"territoire": str, "unit": str})
    pop = ter.set_index("territoire")["codps_2024"].astype(int)
    tprov = ter.set_index("territoire")["unit"]
    names = ter.set_index("territoire")["name"].str.split(",").str[0]
    if set(tprov) != set(PROV.values()):
        raise SystemExit("religiondots' provinces differ from MICS's")
    if not set(VILLES) <= set(pop.index):
        raise SystemExit(f"VILLES not territoires: {sorted(set(VILLES) - set(pop.index))}")
    eth = pd.read_csv(E123, dtype={"geo_id": str, "province": str})
    if set(eth.geo_id) != set(pop.index):
        raise SystemExit("cd.csv territoires differ from cd_territoires.csv; re-run cd_e123.py")
    eth["node"] = eth["source_category"].map(cd2012.resolve)
    if eth["node"].isna().any():
        raise SystemExit("cd.csv has labels cd2012 does not map")
    eth["grp"] = np.where(eth["node"] == cd2012.LUBA_KASAI, "Tshiluba",
                          np.where(eth["node"].str.startswith(cd2012.KG + "."), "Kikongo", "Autre"))
    eg = eth.pivot_table(index="geo_id", columns="grp", values="share", aggfunc="sum",
                         fill_value=0).reindex(columns=["Autre", "Tshiluba", "Kikongo"],
                                               fill_value=0)

    rows_out, natl = [], {}
    kongo_check, clipped, seeded = {}, [], []
    for g, p in PROV.items():
        ts = sorted(t for t in pop.index if tprov[t] == p)
        P = pop[ts].to_numpy(dtype=float)
        M, U = norm(head[p]), norm(urb.get(p, head[p]))
        kongo_check[p] = (float((eg.loc[ts, "Kikongo"] * P).sum() / P.sum()), M["Kikongo"])
        x = np.zeros((len(ts), len(CATS)))
        if len(ts) == 1:                                  # Kinshasa: the province is the unit
            x[0] = [M[c] * P[0] for c in CATS]
            basis = {ts[0]: "province"}
        else:
            # cities: the urban stratum as measured; the territoires: the province's remainder
            v = np.array([t in VILLES for t in ts])
            x[v] = np.outer(P[v], [U[c] for c in CATS])
            rest = np.array([M[c] for c in CATS]) * P.sum() - x[v].sum(axis=0)
            if (rest < 0).any():
                clipped += [f"{g} {c} {rest[j] / P[~v].sum():+.2%}" for j, c in enumerate(CATS)
                            if rest[j] < 0]
                rest = np.clip(rest, 0, None)
            rest *= P[~v].sum() / rest.sum()
            tv = [t for t in ts if t not in VILLES]
            Pv = P[~v]
            ebar = eg.loc[tv].mul(Pv, axis=0).sum() / Pv.sum()
            seed = np.ones((len(tv), len(CATS)))
            for j, c in enumerate(CATS):
                # an ethnic seed only where the ethnic groups account for the category
                if c in ebar.index and M[c] >= 0.005 and ebar[c] >= 0.5 * M[c] and (
                        c != "Kikongo" or p in KONGO_PROVINCES):
                    seed[:, j] = eg.loc[tv, c].to_numpy() / ebar[c]
                    seeded.append(f"{p} {c}")
            seed = np.maximum(seed, FLOOR)
            x[~v] = ipf(seed, Pv, rest)
            if np.abs(x.sum(axis=1) - P).max() > 1e-6 * P.max():
                raise SystemExit(f"{g}: fit misses a row total")
            basis = {t: "city, urban stratum" if t in VILLES else "province remainder"
                     for t in ts}

        # province mixes, for territoires whose own ethnic split is empty
        e_p = eth[eth.province == p].assign(w=lambda d: d.share * d.geo_id.map(pop))
        for i, t in enumerate(ts):
            cnt = {}
            for j, c in enumerate(CATS):
                v = x[i, j]
                if v <= 0:
                    continue
                if c in LABEL:
                    cnt[LABEL[c]] = cnt.get(LABEL[c], 0) + v
                    continue
                grp = eth[(eth.geo_id == t) & (eth.grp == c)]
                wts = grp.groupby("source_category")["share"].sum()
                if c == "Kikongo" and p not in KONGO_PROVINCES:
                    # Kongo varieties up to the territoire's ethnic Kongo, the rest Kikongo ya leta
                    kv = min(v, wts.sum() * P[i])
                    cnt[LABEL["Kituba"]] = cnt.get(LABEL["Kituba"], 0) + v - kv
                    for lab, wv in (wts / wts.sum() if kv > 0 else wts).items():
                        cnt[lab] = cnt.get(lab, 0) + kv * wv
                    continue
                if wts.sum() <= 0:
                    wts = e_p[e_p.grp == c].groupby("source_category")["w"].sum()
                if wts.sum() <= 0:
                    wts = pd.Series({"Other": 1.0})
                for lab, wv in (wts / wts.sum()).items():
                    cnt[lab] = cnt.get(lab, 0) + v * wv
            n = int(P[i])
            base = {k: int(v) for k, v in cnt.items()}
            for k in sorted(cnt, key=lambda k: cnt[k] - base[k], reverse=True)[:n - sum(base.values())]:
                base[k] += 1
            if sum(base.values()) != n:
                raise SystemExit(f"{t}: rounding gives {sum(base.values())} of {n}")
            share = {c: x[i, j] / P[i] for j, c in enumerate(CATS)}
            for k in sorted(base, key=base.get, reverse=True):
                if not base[k]:
                    continue
                natl[k] = natl.get(k, 0) + base[k]
                rows_out.append(dict(
                    geo_id=t, province=p, geo_name=names[t], source_category=k, count=base[k],
                    tier="modelled", source_id=SOURCE_ID, year=2018,
                    note=(f"MICS-Palu 2017-18 HC1B ({basis[t]}): Lingala {share['Lingala']:.3f}, Swahili {share['Swahili']:.3f}, "
                          f"Tshiluba {share['Tshiluba']:.3f}, Kikongo {share['Kikongo']:.3f}, French "
                          f"{share['French']:.3f}, another language {share['Autre']:.3f}; "
                          f"COD-PS 2024 population {n}")))

    print("  ethnic seeds used: " + ", ".join(seeded))
    if clipped:
        print("  cities took more than the province total of (remainder clipped to 0): "
              + "; ".join(clipped))
    for p, (e, m) in sorted(kongo_check.items()):
        if (p in KONGO_PROVINCES) != (m >= 0.05 and e >= 0.5 * m):
            raise SystemExit(f"{p}: ethnic Kongo {e:.1%} against MICS Kikongo {m:.1%}; "
                             "revisit KONGO_PROVINCES")
    total = int(pop.sum())
    if sum(natl.values()) != total:
        raise SystemExit("territoires do not sum to the COD-PS total")
    drawn = {c: 0 for c in CATS}
    for r in rows_out:
        k = r["source_category"]
        c = next((c for c, lab in LABEL.items() if lab == k), None)
        c = "Kikongo" if c == "Kituba" else c
        if c is None:
            c = "Kikongo" if cd2012.resolve(k, None).startswith(cd2012.KG + ".") else (
                "Tshiluba" if cd2012.resolve(k) == cd2012.LUBA_KASAI else "Autre")
        drawn[c] += r["count"]
    # drawn = each province's HC1B shares x its COD-PS population, exactly (the fit's column
    # totals); MICS's own weights give the provinces other sizes (Kinshasa 12.5% of COD-PS)
    expect = {c: sum(norm(head[p])[c] * pop[tprov == p].sum() for p in PROV.values())
              for c in CATS}
    print("  national, drawn on COD-PS 2024 (on MICS's own province weights):")
    off = []
    for c in CATS:
        print(f"      {c:<9} {drawn[c]:>12,}  {drawn[c] / total:6.1%}  ({natl_mics[c]:.1%})")
        if abs(drawn[c] - expect[c]) > 2e-3 * total or abs(drawn[c] / total - natl_mics[c]) > .03:
            off.append(c)
    if off:
        raise SystemExit(f"drawn national shares off the provinces' sum or MICS's: {off}")
    for t in ("CD1000", "CD7101", "CD6201", "CD6101", "CD5101", "CD4101", "CD8201", "CD9101",
              "CD3203", "CD6110"):
        r = [x for x in rows_out if x["geo_id"] == t]
        n = sum(x["count"] for x in r)
        top = sorted(r, key=lambda x: -x["count"])[:5]
        print(f"    {names[t]:<11} " + ", ".join(f"{x['source_category']} {x['count'] / n:.1%}"
                                                 for x in top))

    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(rows_out):,} rows, {total:,} people, "
          f"{len(natl)} labels")


if __name__ == "__main__":
    main()
