"""Panama, MICS 2013 (UNICEF / MINSA, Encuesta de Indicadores Multiples por Conglomerados):
mother tongue by indigenous people and by Afro-descendant group -> retention shares that
taxonomy/pa2023.py lays on the 2023 census's groups.

    python sources/pa_mics.py

Reads religiondots' copy of the household-members file (data/raw/pa/mics2013/..., hl.sav,
read-only). Every member of every household is asked (by proxy) HC1B "Lengua materna / idioma
nativo" (Espanol, Kuna, Ngabere, Buglere/Bokota, Embera, Wounmeu, Naso, other indigenous,
English, other), HC1D/HC1E indigenous people and HC1F/HC1G Afro-descendant group. Weighted by
hhweight; 42,568 people.

Writes data/normalized/pa_mics_shares.csv: (group, scope, lang, share, n). Scope `inside` is a
household in one of the three comarcas the survey separates (Kuna Yala, Embera Wounaan, Ngobe
Bugle), `outside` everywhere else, `all` both. Missing answers are dropped before the shares.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

HL = (RD / "data" / "raw" / "pa" / "mics2013" / "Panama_MICS5_Datasets"
      / "Panama MICS 2013 SPSS Datasets" / "hl.sav")
OUT = HERE / "data" / "normalized" / "pa_mics_shares.csv"
LANG = {"Español": "es", "Kuna": "kuna", "Ngäbere": "ngab", "Buglere/Bokota": "bugl",
        "Emberá": "emb", "Wounmeu (Wounaan)": "woun", "Naso": "naso",
        "Otra lengua indígena": "oind", "Inglés": "en", "Otra": "oth"}
COMARCA = {"Kuna Yala", "Emberá Wounaan", "Ngöbe Buglé"}


def main():
    import pyreadstat

    hl, _ = pyreadstat.read_sav(str(HL), apply_value_formats=True,
                                usecols=["HH7", "HC1B", "HC1D", "HC1E", "HC1F", "HC1G", "hhweight"])
    if len(hl) != 42_568:
        raise SystemExit(f"hl.sav has {len(hl):,} people, expected 42,568")
    hl["lang"] = hl["HC1B"].map(LANG)
    unk = sorted(set(hl.loc[hl["lang"].isna(), "HC1B"].dropna()) - {"Missing"})
    if unk:
        raise SystemExit(f"HC1B labels not in LANG: {unk}")
    hl = hl[hl["lang"].notna()].copy()
    hl["scope"] = hl["HH7"].map(lambda p: "inside" if p in COMARCA else "outside")
    ind = hl["HC1D"] == "Sí"
    hl["group"] = None
    hl.loc[ind, "group"] = "ind:" + hl.loc[ind, "HC1E"].astype(str)
    afro = ~ind & (hl["HC1F"] == "Si")
    hl.loc[afro, "group"] = "afro:" + hl.loc[afro, "HC1G"].astype(str)
    hl.loc[~ind & ~afro, "group"] = "neither"
    rows = []
    for (grp, scope), sub in list(hl.groupby(["group", "scope"])) + [((g, "all"), s) for g, s in hl.groupby("group")]:
        w = sub.groupby("lang")["hhweight"].sum()
        for lang, x in (w / w.sum()).items():
            rows.append(dict(group=grp, scope=scope, lang=lang, share=round(x, 6), n=len(sub)))
    df = pd.DataFrame(rows).sort_values(["group", "scope", "share"], ascending=[True, True, False])
    df.to_csv(OUT, index=False)
    sys.stdout.reconfigure(encoding="utf-8")
    show = df[df["share"] >= 0.01]
    print(show.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
