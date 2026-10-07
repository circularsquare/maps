"""Republic of the Congo: Afrobarometer round 9 (2022-23) home language, read as FIRST language
through the respondent's ethnic group, by département, on the RGPH-5 2023 preliminary
populations -> data/normalized/cg.csv.

    python sources/cg_afro.py

NO CENSUS ASKS: RGPH 2007 and RGPH-5 2023 have no language question (coverage sweep; the 2023
results are preliminary totals only). The only open survey with a language item and a
département code is Afrobarometer R9, 1,200 respondents in all 12 départements (religiondots'
merged .sav, read-only). Q2 "Language spoken in home": French 609, Kituba 275, Lingala 248, Teke
28, Lari 15, Other 25 (verbatims below).

THE LINGUA FRANCA TRAP (ask 018). R9's wording captures the language USED at home: half the
sample names French, and Kituba and Lingala, the two national lingua francas, take most of the
rest. The map draws first languages (DR Congo, Kenya, Nigeria), so:
  * a named local language (Teke, Lari, a verbatim) is drawn as answered;
  * French, Kituba or Lingala from a respondent who names an ethnic group with a language of its
    own (Q84A: Kongo, Teke, Mbosi, Mbede, Echira, Kota, Makaa, Fang, Autochtones) is drawn as that
    group's language;
  * French, Kituba or Lingala from a respondent with no such group (national identity only,
    Oubanguiens, Sangha, refused, don't know) is kept as answered: these are the people for whom
    the lingua franca is most plausibly the first language.
Every row `modelled` (survey shares x population). Weighted by withinwt_hh, as sources/ng_afro.py.

POPULATION: RGPH-5 (17 May 2023) preliminary results by département, 6,142,180, as printed in
the press release of 29 December 2023 (INS-Congo's preliminary report; Les Echos du Congo
Brazzaville and ADIAC carry the table). Religiondots draws on RGPH 2007 (3,697,490) because its
religion table is 2007; language has no census table either year, so the newer count is used.
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
from rdlink import RD  # noqa: E402

SAV = (RD / "data" / "raw" / "afrobarometer" /
       "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav")
OUT = HERE / "data" / "normalized" / "cg.csv"
SOURCE_ID = "afrobarometer_r9_congo"
N_RESP = 1200

POP_2023 = {
    "Brazzaville": 2_145_783, "Pointe-Noire": 1_420_612, "Pool": 394_532, "Bouenza": 363_850,
    "Likouala": 355_570, "Niari": 334_863, "Cuvette": 316_599, "Plateaux": 283_421,
    "Sangha": 209_701, "Cuvette-Ouest": 119_328, "Lékoumou": 100_559, "Kouilou": 97_362,
}
TOTAL = 6_142_180
REGION = {"Cuvette Ouest": "Cuvette-Ouest", "Lekoumou": "Lékoumou"}

LINGUA_FRANCA = {"French", "Kituba", "Lingala"}
# Q2 answers and Q2OTHER verbatims -> label written to cg.csv
HOME = {"Teke": "Teke", "Lari": "Laari", "French": "French", "Kituba": "Kituba",
        "Lingala": "Lingala"}
VERBATIM = {
    "LAALI": "Teke-Laali",   # all 7 in Lékoumou, where Teke-Laali (lli) is spoken; not Laari
    "MBOSI": "Mbosi", "BAYAKA": "Aka", "LIKOUBA": "Likuba", "MAKOUA": "Akwa (Makoua)",
    "BEMBE": "Beembe", "SOUNDI": "Suundi", "KOYO": "Koyo", "BOMITABA": "Bomitaba",
}
# Q84A ethnic group -> the language drawn for a lingua-franca answer (None: keep the answer)
ETHNIC = {
    "Kongo": "Kongo", "Teke": "Teke", "Mbosi": "Mbosi", "Mbede": "Mbere (Mbede)",
    "Echira": "Sira", "Kota": "Kota", "Mekee/ makaa": "Makaa", "Fang": "Fang",
    "Autochtones": "Aka",
    "Oubanguiens": None, "Sangha": None,
    "(National identity) only, or “doesn’t think of self in those terms”": None,
    "Refused to answer": None, "Don’t know": None, "Other": None,
}


def load():
    import pyreadstat
    cols = ["COUNTRY", "REGION", "Q2", "Q2OTHER", "Q84A", "Q84AOTHER", "withinwt_hh"]
    df, _ = pyreadstat.read_sav(str(SAV), usecols=cols, apply_value_formats=True)
    cg = df[df["COUNTRY"].astype(str) == "Congo-Brazzaville"].copy()
    if len(cg) != N_RESP:
        raise SystemExit(f"R9: {len(cg)} Congo-Brazzaville rows, expected {N_RESP}")
    return cg


def label(row):
    q2 = str(row["Q2"])
    if q2 == "Other":
        v = str(row["Q2OTHER"]).strip().upper()
        if v not in VERBATIM:
            raise SystemExit(f"Q2OTHER verbatim {v!r} not in VERBATIM")
        return VERBATIM[v], "verbatim"
    if q2 not in HOME:
        raise SystemExit(f"Q2 answer {q2!r} not in HOME")
    if q2 not in LINGUA_FRANCA:
        return HOME[q2], "as answered"
    eth = str(row["Q84A"])
    if eth not in ETHNIC:
        raise SystemExit(f"Q84A {eth!r} not in ETHNIC")
    if ETHNIC[eth]:
        return ETHNIC[eth], "ethnic"
    return HOME[q2], "lingua franca kept"


def main():
    import pandas as pd
    cg = load()
    cg["dept"] = cg["REGION"].astype(str).map(lambda r: REGION.get(r, r))
    bad = sorted(set(cg["dept"]) - set(POP_2023))
    if bad:
        raise SystemExit(f"départements not in POP_2023: {bad}")
    assert sum(POP_2023.values()) == TOTAL
    lab = cg.apply(label, axis=1)
    cg["lang"], cg["how"] = [x[0] for x in lab], [x[1] for x in lab]
    w = cg["withinwt_hh"].astype(float)
    print("  answers by route (unweighted):", cg["how"].value_counts().to_dict())
    print("  lingua franca answers by ethnic group:")
    lf = cg[cg["Q2"].astype(str).isin(LINGUA_FRANCA)]
    print(pd.crosstab(lf["Q84A"].astype(str).str[:40], lf["Q2"].astype(str)).to_string())
    print("  ethnic verbatims:", cg["Q84AOTHER"].astype(str).value_counts().head(8).to_dict())

    out, natl = [], {}
    for d in sorted(POP_2023):
        sub = cg[cg["dept"] == d]
        sw = w[sub.index]
        shares = sw.groupby(sub["lang"]).sum() / sw.sum()
        raw = {k: v * POP_2023[d] for k, v in shares.items()}
        cnt = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda x: raw[x] - cnt[x], reverse=True)[:POP_2023[d] - sum(cnt.values())]:
            cnt[k] += 1
        assert sum(cnt.values()) == POP_2023[d]
        for k in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[k]:
                continue
            natl[k] = natl.get(k, 0) + cnt[k]
            n_resp = int((sub["lang"] == k).sum())
            out.append(dict(geo_id=d, geo_level="departement", geo_name=d, source_category=k,
                            count=cnt[k], tier="modelled", source_id=SOURCE_ID, year=2023,
                            note=f"{n_resp} of {len(sub)} R9 respondents (weighted share "
                                 f"{shares[k]:.3f}); RGPH-5 2023 population {POP_2023[d]}"))
        print(f"  {d:<14} n {len(sub):>4}  " + ", ".join(
            f"{k} {v:.0%}" for k, v in shares.sort_values(ascending=False).items() if v >= 0.02))

    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                           "count", "tier", "source_id", "year", "note"])
        wr.writeheader()
        wr.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {sum(natl.values()):,} people")
    for k, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<24} {n:>10,}  {n / TOTAL * 100:5.2f}%")


if __name__ == "__main__":
    main()
