"""Zimbabwe: Afrobarometer home language by district, for placing the census's province
counts inside each province (never for the counts themselves).

    python sources/zw_afro.py --fetch   extract Zimbabwe's rows from religiondots' merged .sav
                                        files (read-only) -> data/raw/zw/ab_zw_language.csv
    python sources/zw_afro.py           checks against the census, then
                                        -> data/normalized/zw_district.csv (district x answer
                                        shares, placement only)

The counts are the 2022 census's (sources/zw_census.py, Table 2.17). Inside a province, a
language's dots lean towards the COD-AB districts where the survey's respondents named it,
shrunk to the census's own province share with K respondents' weight, so a language the
survey barely sees falls back to plain population. The record is sources/zw.md.
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
AB_DIR = RD / "data" / "raw" / "afrobarometer"
RAW = HERE / "data" / "raw" / "zw"
EXTRACT = RAW / "ab_zw_language.csv"
CENSUS = HERE / "data" / "normalized" / "zw.csv"
OUT_DIST = HERE / "data" / "normalized" / "zw_district.csv"
ADM_ZIP = RD / "data" / "raw" / "zw" / "zwe_admin_boundaries.shp.zip"

# (round, file, language, verbatim, weight, ethnic group, its verbatim, interview language)
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3OTHER", "Withinwt", "Q79", "Q79OTHER", "Q103"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q2OTHER", "withinwt", "Q84", "Q84OTHER", "Q103"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt", "Q87", "Q87OTHER",
     "Q103"),
    (7, "r7_merged_data_34ctry.release.sav", "Q2B", "Q2BOTHER", "withinwt", "Q84", "Q84OTHER",
     "Q103"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "Q81", "Q81OTHER", "Q103"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "Q84A", "Q84AOTHER", "Q102"),
]
LANG_LABEL = ("language of respondent", "language spoken in home", "home language")
ETH_LABEL = ("tribe or ethnic group", "ethnic community", "ethnic")


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def fetch():
    import pyreadstat
    out = []
    for rnd, name, q, qo, wt, eth, etho, il in ROUNDS:
        p = AB_DIR / name
        try:
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
            enc = {}
        except Exception:  # noqa: BLE001  R6 is not valid UTF-8
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True, encoding="LATIN1")
            enc = {"encoding": "LATIN1"}
        up = {c.upper(): c for c in meta.column_names}
        locs = [c for c in ("DISTRICT", "LOCATION.LEVEL.1") if c in up]
        want = list(dict.fromkeys(["COUNTRY", "REGION", "RESPNO", "URBRUR", q, qo, wt, eth, etho,
                                   il] + locs))
        want = [c for c in want if c.upper() in up]
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question "
            f"({lab!r})")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group ({elab!r})")
        df, meta = pyreadstat.read_sav(str(p), usecols=[up[c.upper()] for c in want], **enc)
        c = {k: up[k.upper()] for k in want}
        vl = meta.variable_value_labels
        cl = df[c["COUNTRY"]].map(vl.get(c["COUNTRY"], {})).astype(str).str.strip().str.casefold()
        sub = df[cl == "zimbabwe"]

        def lab_of(k):
            if k not in c:
                return ""
            m = vl.get(c[k], {})
            return sub[c[k]].map(m) if m else sub[c[k]]
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(len(sub) > 0 and 0.98 <= w.sum() / len(sub) <= 1.02,
            f"R{rnd}: {len(sub):,} Zimbabwean respondents; {wt} averages "
            f"{w.sum() / max(1, len(sub)):.3f}")
        o = pd.DataFrame({
            "round": rnd, "respno": sub[c["RESPNO"]].astype(str),
            "region_code": pd.to_numeric(sub[c["REGION"]], errors="coerce").astype("Int64"),
            "region": lab_of("REGION"),
            "district": lab_of(locs[0]) if locs else "",
            "urb": lab_of("URBRUR"),
            "lang": lab_of(q), "verbatim": sub[c[qo]].astype(str).str.strip() if qo in c else "",
            "eth": lab_of(eth),
            "eth_verbatim": sub[c[etho]].astype(str).str.strip() if etho in c else "",
            "intlang": lab_of(il), "w": w,
        })
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- answers -> census categories

# Every survey answer (card label, or free text upper-cased) -> the Table 2.17 category it is
# placed as. The Shona dialect answers (Karanga, Zezuru, Manyika, Korekore, and the free-text
# Buja, Bocha, Maungwe, Hwesa, Jindwi, Budya, Shangwe, Toko, Garwe, Vhitori = Victoria, i.e.
# Masvingo Karanga) are Shona here: the census has one Shona.
SHONA = {"Shona", "Karanga", "Zezuru", "Manyika", "Korekore", "Buja", "Bocha", "Maungwe",
         "Vhitori"}
CARD = {"Ndebele": "Ndebele", "Ndau": "Ndau", "Tonga": "Tonga", "Kalanga": "Kalanga",
        "English": "English", "Venda": "Venda", "Shangani": "Shangani", "Shangaan": "Shangani",
        "Suthu": "Sotho", "Nambya": "Nambya", "Nyanja": "Chewa", "Portuguese": "Other"}
VERB = {
    "CHEWA": "Chewa", "NYAHO (CHAWA)": "Chewa", "NYANJA": "Chewa", "NYASALAND": "Chewa",
    "MALAWIAN": "Chewa", "TUMBUKA": "Other", "YAO": "Other", "CHIKUNDA": "Other",
    "CHANGANI": "Shangani", "SHANGANI": "Shangani", "TSHANGANI": "Shangani",
    "CHNGANI AND CHIKARANGA": "Shangani",
    "NAMBIA": "Nambya", "NAMBYA": "Nambya",
    # Dombe, Hwange's other Kalanga-cluster speech: nearest census category Nambya
    "DOMBE": "Nambya", "DUMBE": "Nambya", "MDOMBE": "Nambya", "SIDOMBE": "Nambya",
    "CHIDHOMBE": "Nambya",
    "LOZWI": "Kalanga",                 # Rozvi/Lozwi of the south-west, Kalanga-speaking
    "SOTHO": "Sotho", "XHOSA": "Xhosa", "NGUNI": "Ndebele", "VENDA": "Venda",
    "MANICA (MOZAMBICAN)": "Shona", "TAURA (MOZAMBICAN)": "Other", "MOZAMBICAN": "Other",
    "HWESA": "Shona", "JINDWI": "Shona", "JINDWE": "Shona", "BUDYA": "Shona",
    "CHIBUDYA": "Shona", "SHANGWE": "Shona", "MUSHANGWE": "Shona", "TOKO": "Shona",
    "CHITOKO": "Shona", "GARWE": "Shona", "BOCHA": "Shona", "MAUNGWE": "Shona", "BUJA": "Shona",
    "MUHERA": "Shona", "TSENGA": "Shona", "MLEMBA": "Shona",
    "NYONGWE": "Other", "CHITAWALA": "Other", "HINDI": "Other", "PFUMBI": "Other",
    "DUMA": "Shona", "NLEYA ZAMBIA": "Other",
}
NON_ANSWERS = {"Don't know", "Missing", "Refused"}


def answer(row):
    lab = str(row["lang"]).strip()
    if lab in ("Other", "Others"):
        v = " ".join(str(row["verbatim"]).split()).upper()
        if not v or v == "NAN":
            return "Other"
        if v not in VERB:
            raise SystemExit(f"verbatim {v!r} (R{row['round']}) not in VERB: decide it there")
        return VERB[v]
    if lab in NON_ANSWERS:
        return None
    if lab in SHONA:
        return "Shona"
    if lab not in CARD:
        raise SystemExit(f"card label {lab!r} (R{row['round']}) not in CARD: decide it there")
    return CARD[lab]


# ---------------------------------------------------------------- provinces and districts

PROV = {"bulawayo": "ZW10", "harare": "ZW19", "manicaland": "ZW11",
        "mashonaland central": "ZW12", "mashonaland east": "ZW13", "mashonaland west": "ZW14",
        "masvingo": "ZW18", "matabeleland north": "ZW15", "matebeland north": "ZW15",
        "matebeleland north": "ZW15", "matabeleland south": "ZW16", "matebeland south": "ZW16",
        "matebeleland south": "ZW16", "midlands": "ZW17"}
# census geo_id (religiondots' zw_lookup.csv) -> hex unit (COD-AB adm1 pcode)
CENSUS_UNIT = {"ZW01": "ZW10", "ZW02": "ZW11", "ZW03": "ZW12", "ZW04": "ZW13", "ZW05": "ZW14",
               "ZW06": "ZW15", "ZW07": "ZW16", "ZW08": "ZW17", "ZW09": "ZW18", "ZW10": "ZW19"}

# Placement districts are COD-AB's 91 ADM2 with each urban council folded into its rural
# district of the same name (60 bases): the survey often writes "MUTARE" for either.
SUFFIXES = (" URBAN", " RURAL", " TOWN", " LOCAL BOARD", " CENTRE")
ALIAS = {
    "CHEGETU": "CHEGUTU", "MUREWA": "MUREHWA", "MT DARWIN": "MOUNT DARWIN",
    "CHIRUMANZU": "CHIRUMHANZU", "MHONDORO- NGEZI": "MHONDORO-NGEZI",
    "MHONDORO NGEZI": "MHONDORO-NGEZI", "UMP": "UZUMBA MARAMBA PFUNGWE",
    "MUZARABANI": "CENTENARY/ MUZARABANI", "DZIVARASEKWA": "HARARE",
    "BULILIMA-MANGWE NORT": "BULILIMA", "PLUMTREE": "BULILIMA", "VICTORIA FALLS": "HWANGE",
    "RENCO MINE": "MASVINGO", "CHINHOYI": "MAKONDE", "KAROI": "HURUNGWE",
    "RUSAPE": "MAKONI", "MVURWI": "MAZOWE", "RUWA": "GOROMONZI", "NORTON": "ZVIMBA",
    "REDCLIFF": "KWEKWE", "EPWORTH": "HARARE RURAL",
}
# The fold: an urban council that has no rural district of its own name goes on the district
# around it (Chinhoyi in Makonde, Karoi in Hurungwe, Rusape in Makoni, Mvurwi in Mazowe, Ruwa
# in Goromonzi, Norton in Zvimba, Redcliff in Kwekwe, Plumtree in Bulilima, Victoria Falls in
# Hwange, Kadoma Urban in Sanyati is NOT done: Kadoma stands alone, as COD-AB has it).
ADM2_FOLD = {"CHINHOYI": "MAKONDE", "KAROI": "HURUNGWE", "RUSAPE": "MAKONI",
             "MVURWI": "MAZOWE", "RUWA": "GOROMONZI", "NORTON": "ZVIMBA", "REDCLIFF": "KWEKWE",
             "PLUMTREE": "BULILIMA", "VICTORIA FALLS": "HWANGE", "KADOMA": "KADOMA"}


def base(name):
    s = " ".join(str(name).upper().split())
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    if s == "HARARE RURAL" or s == "HARARE":
        pass
    s = ALIAS.get(s, s)
    return ADM2_FOLD.get(s, s)


def adm2_key(adm1, name):
    n = " ".join(str(name).upper().split())
    if n == "HARARE RURAL":
        return f"{adm1}:HARARE RURAL"
    if n == "GOKWE SOUTH URBAN":
        return f"{adm1}:GOKWE SOUTH"
    return f"{adm1}:{base(n)}"


def survey_key(adm1, label):
    s = " ".join(str(label).upper().split())
    if not s:
        return None
    if s in ("HARARE RURAL", "EPWORTH"):
        return f"{adm1}:HARARE RURAL"
    if s in ("HARARE URBAN", "DZIVARASEKWA"):
        return f"{adm1}:HARARE"
    if s == "GOKWE CENTRE":
        return f"{adm1}:GOKWE SOUTH"
    return f"{adm1}:{base(s)}"


def adm2():
    import geopandas as gpd
    g = gpd.read_file(f"zip://{ADM_ZIP}!zwe_admin2.shp", engine="pyogrio")
    say(len(g) == 91, f"COD-AB admin2: {len(g)} districts")
    g["key"] = [adm2_key(p, n) for p, n in zip(g["adm1_pcode"], g["adm2_name"])]
    return g


# ---------------------------------------------------------------- main

K = 8.0          # prior weight, in respondents (Tanzania's and Nigeria's choice)
NEAREST = 3


def main():
    if "--fetch" in sys.argv:
        fetch()
    a = pd.read_csv(EXTRACT, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    a["cat"] = a.apply(answer, axis=1)
    a = a[a["cat"].notna()].copy()
    a["adm1"] = a["region"].map(lambda s: PROV.get(" ".join(s.split()).casefold()))
    say(a["adm1"].notna().all(), "every REGION label is a province")

    c = pd.read_csv(CENSUS, dtype={"geo_id": str})
    c["adm1"] = c["geo_id"].map(CENSUS_UNIT)
    cs = c.pivot_table(index="adm1", columns="source_category", values="count", fill_value=0)
    cs = cs.div(cs.sum(axis=1), axis=0)

    # survey against census, nationally and per round
    nat_c = c.groupby("source_category")["count"].sum() / c["count"].sum()
    t = a.pivot_table(index="cat", columns="round", values="w", aggfunc="sum", fill_value=0)
    t = t / t.sum()
    t["all"] = a.groupby("cat")["w"].sum() / a["w"].sum()
    t["census"] = nat_c
    print("\n  survey (all rounds) against census 2022, national %:")
    print((t.fillna(0) * 100).round(1).sort_values("census", ascending=False).to_string())
    # per province, the census's larger minorities
    sv = a.pivot_table(index="adm1", columns="cat", values="w", aggfunc="sum", fill_value=0)
    sv = sv.div(sv.sum(axis=1), axis=0)
    print("\n  province shares, survey / census (%):")
    for lang in ("Ndebele", "Ndau", "Tonga", "Shangani", "Venda", "Kalanga", "Nambya", "Sotho"):
        p = cs[lang].idxmax()
        print(f"    {lang:9s} {p}: {sv.loc[p].get(lang, 0) * 100:5.1f} / {cs.loc[p, lang] * 100:5.1f}")
    # interview language: minorities interviewed in Shona / Ndebele / English only
    mi = a[a["cat"].isin(["Tonga", "Venda", "Shangani", "Kalanga", "Nambya", "Ndau"])]
    print("\n  interview language of minority-language answers: "
          + str(mi.groupby("intlang").size().to_dict()))
    tonga_p = a[(a["adm1"] == "ZW15")]
    print("  Matabeleland North, answers by interview language: "
          + str(pd.crosstab(tonga_p["intlang"], tonga_p["cat"]).to_dict("index")))

    # districts
    g = adm2()
    keys = set(g["key"])
    print(f"  {len(keys)} placement districts (urban councils folded)")
    a["dkey"] = [survey_key(p, d) for p, d in zip(a["adm1"], a["district"])]
    lab = a[a["dkey"].notna()]
    miss = lab[~lab["dkey"].isin(keys)]
    print(f"  {len(lab) - len(miss):,} of {len(lab):,} district-labelled respondents join "
          f"(rounds {sorted(lab['round'].unique())}); unjoined: "
          f"{miss.groupby('dkey').size().to_dict()}")
    say(len(miss) <= 0.02 * len(lab), "98%+ of district labels join a district of their province")
    lab = lab[lab["dkey"].isin(keys)]

    c3 = g.to_crs(32735).representative_point()
    g["x"], g["y"] = c3.x, c3.y
    cen = g.groupby("key")[["x", "y"]].mean()
    rows = []
    for u, gu in g.groupby("adm1_pcode"):
        prov = cs.loc[u]
        prov = prov[prov > 0]
        b = lab[lab["adm1"] == u]
        own = {}
        for d, bd in b.groupby("dkey"):
            n = bd.groupby("cat")["w"].sum().reindex(prov.index).fillna(0.0)
            own[d] = (n + K * prov) / (bd["w"].sum() + K)
            own[d] = own[d] / own[d].sum()
        samp = list(own)
        for d in sorted(set(gu["key"])):
            if d in own:
                v, how = own[d], "sampled"
            elif samp:
                sx = cen.loc[samp]
                d2 = ((sx["x"] - cen.loc[d, "x"]) ** 2 + (sx["y"] - cen.loc[d, "y"]) ** 2)
                d2 = d2.to_numpy()
                near = np.argsort(d2)[:NEAREST]
                w = 1.0 / np.maximum(d2[near], 1e6)
                v = sum(own[samp[i]] * w[j] for j, i in enumerate(near)) / w.sum()
                how = "borrowed"
            else:
                v, how = prov, "province"
            for k, x in v.items():
                rows.append((u, d, k, float(x), how))
    t = pd.DataFrame(rows, columns=["unit", "district", "source_category", "share", "basis"])
    tot = t.groupby("district")["share"].sum()
    say((tot - 1).abs().max() < 1e-6, "every district's shares sum to 1")
    print(f"  district basis: {t.drop_duplicates('district')['basis'].value_counts().to_dict()}")
    t.to_csv(OUT_DIST, index=False)
    print(f"wrote {OUT_DIST} ({len(t)} rows)")
    for lang, u in (("Ndau", "ZW11"), ("Tonga", "ZW15"), ("Nambya", "ZW15"), ("Venda", "ZW16"),
                    ("Kalanga", "ZW16"), ("Sotho", "ZW16"), ("Shangani", "ZW18")):
        x = t[(t["unit"] == u) & (t["source_category"] == lang)].sort_values("share",
                                                                            ascending=False)
        print(f"    {lang:9s} {u}: " + ", ".join(f"{r.district.split(':')[1].title()} "
                                                 f"{r.share:.0%}" for r in x.head(4).itertuples()))


if __name__ == "__main__":
    main()
