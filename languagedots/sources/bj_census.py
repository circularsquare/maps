"""Benin: RGPH-4 2013 "primary language spoken in the household", commune shares from the IPUMS
10% sample (via CLEAR Global), on the census's commune populations; ethnicity as the check.

    python sources/bj_census.py      -> data/normalized/bj.csv (commune x language, counts)

THE LANGUAGE TABLE. RGPH-4 (2013) asked every resident aged 3+ (Q17) "Quelle est la langue
principale parlée par [la personne] dans le ménage ?" with a write-in answer (IPUMS
BJ2013A_LANG / LANGBJ, about 75 codes). INStaD publishes no table of it. CLEAR Global's HDX
dataset "Benin - Languages" (CC BY-SA) tabulates IPUMS's 10% sample of it by commune as
proportions, with glottocodes: data/raw/bj/clearglobal_language_use_ben_admin2.csv (77
communes, 57 answers, each commune summing to 1). Two of CLEAR's glottocodes are wrong and
are relabelled here (see LABEL): "Central Malay" is IPUMS's Lekpa (Lokpa: Ouaké 54%, Djougou
15%, Copargo, Bassila), and the bare "Gbe" is IPUMS's Toligbe/Setogbe/Kogbe, which have no
glottocode (Avrankou 55%, Akpro-Missérété 46%, Tori-Bossito 36%).

THE POPULATION. Each commune's RGPH-4 population, Tableau 2 of INStaD's twelve departmental
Principaux indicateurs booklets, read with religiondots' parser (sources/bj.py, imported
read-only; it sums to 10,008,749). The language question excludes children under 3; their
language is taken to be their commune's mix. "Unknown" answers are not drawn (gap).

THE CHECK, AND THE FIRST-LANGUAGE READING (ask 018). Tableau 8 of the same booklets prints
every commune's ethnic group as nine clusters ("Fon et apparentés", "Adja et apparentés"...)
from the full count. Every language answer is filed under its cluster (CLUSTER) and the two
are compared per commune: the household-language question pulls towards the lingua francas
(Fon in Cotonou and Abomey-Calavi, French in the towns, Dendi and Baatonum in the north).
FIRST_LANGUAGE = True (the default) draws each commune's ethnic clusters at Tableau 8's
full-count shares, split into languages by the census's own language answers inside that
cluster in the commune (department, then nation, where the commune has none); French and
other answers with no Beninese cluster are spread with them. False draws the language answers
as given. One line; sources/bj.md has both.
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
CLEAR = HERE / "data" / "raw" / "bj" / "clearglobal_language_use_ben_admin2.csv"
LOOKUP = RD / "data" / "geo" / "bj" / "bj_lookup.csv"
OUT = HERE / "data" / "normalized" / "bj.csv"
SOURCE_ID = "bj_rgph4_2013_lang_ipums_clearglobal"

FIRST_LANGUAGE = True      # False = the household-language answers as given

NATIONAL = 10_008_749

# CLEAR's language_name -> the label drawn (only where CLEAR's glottocode is wrong or bare)
LABEL = {"Central Malay": "Lokpa (Lukpa)", "Gbe": "Toli, Seto and Kogbe (Gbe)",
         # IPUMS's Agouna (129, a Gbe language of Djidja) came out as Tagwana Senoufo, an Ivorian
         # language: 8% of Djidja, a few in Zogbodomey and Savalou, all Agouna country
         "Tagwana Senoufo": "Agouna (Gbe)"}
# IPUMS has two "Defi" codes: 116 among the Gbe languages, and 193 after the northern ones.
# CLEAR files both on Defi Gbe. In the south (Sèmè-Kpodji 4%) it is Defi; in the north
# (Ouaké 21%, where Tableau 8 has no Adja at all) it is a northern language IPUMS mislabelled,
# drawn unnamed under Gur, in the Yoa-Lokpa cluster.
NORTH = {"Alibori", "Atacora", "Borgou", "Donga"}
DEFI_NORTH = "Defi (northern code, language not identified)"

# Tableau 8's ethnic block, in print order. Two rows print their label on two lines and come
# off the page with no label; they are told apart by position and asserted below.
ETH_ROWS = ["Adja", "Fon", "Bariba", "Dendi", "Yoa-Lokpa", "Peulh", "Ottamari", "Yoruba",
            "Autres ethnies du Bénin", "Ethnies étrangères"]
ETH_PRINTED = {"Adja": "adjaetapparentes", "Fon": "fonetapparentes",
               "Bariba": "baribaetapparentes", "Dendi": "dendietapparentes",
               "Peulh": "peulhoupeul", "Yoruba": "yorubaetapparentes",
               "Autres ethnies du Bénin": "autresethniesdubenin",
               "Ethnies étrangères": "ethniesetrangeres"}

# language answer (after LABEL) -> Tableau 8 cluster. None: no Beninese cluster (French,
# English), spread over the commune's clusters under FIRST_LANGUAGE.
CLUSTER = {
    "Aja (Benin)": "Adja", "Gen": "Adja", "Saxwe Gbe": "Adja", "Xwela Gbe": "Adja",
    "Defi Gbe": "Adja", "Ci Gbe": "Adja", "Agu (Ewe)": "Adja",
    # Kotafon is filed under Fon by INStaD: Lokossa and Athiémé are 61-66% "Fon et
    # apparentés" in Tableau 8 and 55-57% Kotafon in the language answers
    "Kotafon Gbe": "Fon",
    "Fon": "Fon", "Gun": "Fon", "Maxi Gbe": "Fon", "Weme Gbe": "Fon", "Ayizo Gbe": "Fon",
    "Tofin Gbe": "Fon", "Toli, Seto and Kogbe (Gbe)": "Fon", "Agouna (Gbe)": "Fon",
    DEFI_NORTH: "Yoa-Lokpa",
    "Baatonum": "Bariba", "Boko (Benin)": "Bariba", "Boo": "Bariba",
    "Dendi (Benin)": "Dendi", "Zarma": "Dendi", "Hausa": "Dendi", "Songhay": "Dendi",
    "Yom": "Yoa-Lokpa", "Lokpa (Lukpa)": "Yoa-Lokpa", "Kabiyé": "Yoa-Lokpa",
    "Anii": "Yoa-Lokpa", "Miyobe": "Yoa-Lokpa",
    "Borgu Fulfulde": "Peulh",
    "Ditammari": "Ottamari", "Waama": "Ottamari", "Biali": "Ottamari", "Nateni": "Ottamari",
    "Mbelime": "Ottamari", "Gourmanchéma": "Ottamari", "Moba": "Ottamari",
    "Yoruba": "Yoruba", "Ede Nago": "Yoruba", "Ede Cabe": "Yoruba", "Ede Idaca": "Yoruba",
    "Ifè": "Yoruba", "Ede Ije": "Yoruba", "Manigri-Kambolé Ede Nago": "Yoruba",
    "Igbo": "Ethnies étrangères", "Akan": "Ethnies étrangères",
    "Bambara": "Ethnies étrangères", "Bozo": "Ethnies étrangères",
    "Soninke": "Ethnies étrangères", "Syenara Senoufo": "Ethnies étrangères",
    "Dagaari Dioula": "Ethnies étrangères", "Wolof": "Ethnies étrangères",
    "Lingala-Bangala": "Ethnies étrangères", "Mossi": "Ethnies étrangères",
    "Standard Arabic": "Ethnies étrangères",
    "French": None, "English": None,
}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def rd_module():
    spec = importlib.util.spec_from_file_location("rd_bj", RD / "sources" / "bj.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tableau8_ethnic(m, doc, path):
    """The ten ethnic rows of Tableau 8, positionally, with the eight printed labels asserted."""
    page = doc[m._find_page(doc, m.T8_RE, "Tableau 8", path)]
    words = m._words(page)
    rows = m._data_rows(words, lambda t: bool(m.PCT.match(t)) or t == m.STAR, 4)
    n, cols = m._columns(rows, path, "Tableau 8")
    names = m._header(words, rows, cols, path)
    full = []
    for ws, vals in rows:
        if len(vals) != n:
            continue
        label = " ".join(w[4] for w in ws if w[2] <= vals[0][0] - 1)
        full.append((m.fold(label.replace("(%)", "")), vals))
    vod = [i for i, (k, _) in enumerate(full) if k == "vodoun"]
    say(len(vod) == 1 and vod[0] == len(ETH_ROWS), f"{Path(path).name}: ten ethnic rows "
        "above Vodoun")
    out = {}
    for g, (key, vals) in zip(ETH_ROWS, full[:len(ETH_ROWS)]):
        want = ETH_PRINTED.get(g, "")
        # a label wrapped onto two lines comes off empty (which rows do varies by booklet);
        # a printed label must be its row's
        want = {"Yoa-Lokpa": "yoa", "Ottamari": "otamari"}.get(g, want)
        if key and want not in key:
            raise SystemExit(f"{path}: ethnic row {g!r} reads {key!r}")
        out[g] = [m._cell(v[4], f"{path} {g}") or 0.0 for v in vals]
    return names, out


def census():
    """commune unit -> (name, department, population, {cluster: share})"""
    import fitz
    m = rd_module()
    lut = pd.read_csv(LOOKUP, dtype=str).set_index("geo_id")["unit"]
    res = {}
    for dep, (code, communes) in m.DEPARTMENTS.items():
        path = os.path.join(m.RAW, m._pdf_name(dep))
        doc = fitz.open(path)
        h2, pops = m._read_populations(doc, path)
        h8, eth = tableau8_ethnic(m, doc, path)
        say([m.fold(x) for x in h2] == [m.fold(x) for x in h8], f"{dep}: T2 and T8 columns agree")
        lit = dep == "Littoral"
        for j, name in enumerate(communes):
            col = j if lit else j + 1
            geo_id = f"BJ{code}-{j + 1:02d}"
            sh = {g: eth[g][col] for g in ETH_ROWS}
            res[lut[geo_id]] = (name, dep, pops[col], sh)
    tot = sum(v[2] for v in res.values())
    say(len(res) == 77 and tot == NATIONAL, f"77 communes, {tot:,} people (RGPH-4)")
    worst = max(abs(sum(v[3].values()) - 1) for v in res.values())
    say(worst < 0.03, f"ethnic shares sum to 1 within {worst:.3f} (rest is non déclaré)")
    return res


def language(cen):
    d = pd.read_csv(CLEAR)
    d["lang"] = d["language_name"].map(lambda s: LABEL.get(s, s))
    north = d["location_code"].map(lambda u: cen[u][1] in NORTH)
    d.loc[north & (d["lang"] == "Defi Gbe"), "lang"] = DEFI_NORTH
    d = d[d["lang"] != "Unknown"]
    miss = sorted(set(d["lang"]) - set(CLUSTER))
    say(not miss, f"every answer has a cluster {miss}")
    gdf = __import__("geopandas").read_file(RD / "data" / "geo" / "bj" / "bj_communes.gpkg")
    m = rd_module()
    nm = dict(zip(gdf["unit"], gdf["name"]))
    names = d.drop_duplicates("location_code").set_index("location_code")["location_name"]
    say(set(names.index) == set(cen), "CLEAR's 77 commune codes are religiondots' units")
    # COD pcodes on both sides; names checked loosely (spellings differ: Akpo/Akpro, Sakete)
    same = {"BJ0203", "BJ1004", "BJ0307"}   # Kobli/Cobly, Akpo/Akpro-Missérété, Tori/Torri
    bad = [(c, n, nm[c]) for c, n in names.items()
           if m.fold(n)[:4] != m.fold(nm[c])[:4] and c not in same]
    say(not bad, f"CLEAR names agree with the commune layer {bad}")
    p = d.pivot_table(index="location_code", columns="lang", values="proportion_value",
                      aggfunc="sum").fillna(0.0)
    return p.div(p.sum(axis=1), axis=0)


def compare(cen, p):
    eth = pd.DataFrame({u: v[3] for u, v in cen.items()}).T
    lc = p.T.groupby(p.columns.map(lambda c: CLUSTER[c] or "French/English")).sum().T
    print("\n  cluster share, % : language answers / Tableau 8 ethnicity (big gaps only)")
    for u in p.index:
        diffs = [(g, lc.loc[u].get(g, 0) * 100, eth.loc[u, g] * 100) for g in ETH_ROWS[:8]]
        big = [x for x in diffs if abs(x[1] - x[2]) >= 8]
        fr = lc.loc[u].get("French/English", 0) * 100
        if big or fr >= 3:
            print(f"    {cen[u][0]:16s} " + "  ".join(f"{g} {a:.0f}/{b:.0f}" for g, a, b in big)
                  + (f"  French/English {fr:.1f}" if fr >= 3 else ""))
    w = pd.Series({u: cen[u][2] for u in cen})
    nl = (lc.mul(w, axis=0).sum() / w.sum() * 100).round(2)
    ne = (eth.mul(w, axis=0).sum() / w.sum() * 100).round(2)
    print("\n  national %: language answers vs ethnicity")
    print(pd.DataFrame({"language": nl, "ethnicity": ne}).fillna(0).to_string())


def first_language(cen, p):
    """Tableau 8's cluster shares, split by the language answers inside each cluster."""
    lang_cl = {l: CLUSTER[l] for l in p.columns}
    dep = pd.Series({u: cen[u][1] for u in cen})
    w = pd.Series({u: cen[u][2] for u in cen})
    pw = p.mul(w, axis=0)
    dep_p = pw.groupby(dep).sum()
    nat_p = pw.sum()
    out = {}
    for u in p.index:
        shares = pd.Series(0.0, index=p.columns)
        eth = dict(cen[u][3])
        # "Autres ethnies du Bénin" (0.9%; 15% of Malanville) has no language among the
        # answers to give it, so it is left out and the commune's other clusters scaled up
        # (spec: no named answer on a group node). "Ethnies étrangères" (1.9%) mostly answer
        # a Beninese language (Nigerian Yoruba, Togolese Gen): the foreign-language answers
        # keep their as-given share (up to the cluster's size) and the rest of the cluster
        # is shared over the eight named clusters like "Autres". French and English, a
        # first language for almost nobody here, get nothing under this reading.
        eth.pop("Autres ethnies du Bénin")
        fe = eth.pop("Ethnies étrangères")
        es = sum(eth.values()) + fe
        fcols = [l for l, c in lang_cl.items() if c == "Ethnies étrangères"]
        t = min(p.loc[u, fcols].sum(), fe / es)
        if t > 0:
            shares[fcols] += t * p.loc[u, fcols] / p.loc[u, fcols].sum()
        named = sum(eth.values())
        for g, share in eth.items():
            mass = share / named * (1 - t)
            if mass == 0:
                continue
            cols = [l for l, c in lang_cl.items() if c == g]
            for src in (p.loc[u, cols], dep_p.loc[cen[u][1], cols], nat_p[cols]):
                if src.sum() > 0:
                    shares[cols] += mass * src / src.sum()
                    break
        out[u] = shares
    return pd.DataFrame(out).T


def main():
    global FIRST_LANGUAGE
    if "--as-given" in sys.argv:
        FIRST_LANGUAGE = False
    cen = census()
    p = language(cen)
    compare(cen, p)
    sh = first_language(cen, p) if FIRST_LANGUAGE else p
    say(np.allclose(sh.sum(axis=1), 1), "every commune's shares sum to 1")
    rows = []
    for u in sh.index:
        s = sh.loc[u]
        s = s[s > 0]
        n = cen[u][2]
        f = (s * n).to_numpy()
        base = np.floor(f)
        k = int(round(n - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        for lang, share, c in zip(s.index, s.values, base.astype(int)):
            if c > 0:
                rows.append((u, cen[u][0], lang, c, share))
    df = pd.DataFrame(rows, columns=["unit", "name", "lang", "count", "share"])
    say(int(df["count"].sum()) == NATIONAL, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("lang")["count"].sum().sort_values(ascending=False)
    print(f"\n  national, as drawn ({'first-language reading' if FIRST_LANGUAGE else 'as given'}):")
    for k_, v in nat.head(30).items():
        print(f"    {k_:30s} {v:>10,}  {v / NATIONAL:6.2%}")
    res = pd.DataFrame({
        "geo_id": df["unit"], "geo_level": "commune", "geo_name": df["name"],
        "source_category": df["lang"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": 2013,
        "note": [f"share {s:.5f}" for s in df["share"]]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")


if __name__ == "__main__":
    main()
