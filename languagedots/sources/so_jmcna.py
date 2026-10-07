"""Somalia: household home language from REACH's Joint Multi-Cluster Needs Assessment (JMCNA)
2021, by region, on the COD-PS 2026 region populations -> data/normalized/so.csv.

    python sources/so_jmcna.py            (downloads the dataset if it is missing)

NO CENSUS LANGUAGE QUESTION (no census since 1975; PESS 2014 asks none). The JMCNA 2021
(REACH for the Somalia IMAWG; phone survey, 30 May - 18 Aug 2021, 11,349 households kept, 74
districts in 17 regions) asks "What is the main language your household speaks at home?" with
one answer from: Standard / Northern Somali (Maxaa tiri), Benaadir Somali, Maay Somali, Arabic,
English, Italian, Bravanese (Chimwiini / Chimbalazi), Kibajuni, Mushunguli, Somali Sign
Language, other (specify), don't know, prefer not to answer. So AGENT_BRIEF §2's survey route:
weighted shares per region x population, every row `modelled`. The file's own `weights` column
(mean 1.0) is used. CLEAR Global's HDX dataset `somalia-languages` is built from this same file.

ANSWERS NOT DRAWN, shares renormalised over the rest:
- "Somali Sign Language", 256 households, up to 15% of Galgaduud. No deaf population is that
  large; the Somali wording ("calaamadaha luuqada soomaaliga", "signs of the Somali language")
  is easy to take for "Somali", and the answer is concentrated where Standard Somali dominates.
  Read as a response error, not drawn, and not guessed onto Somali either (gap).
- don't know (50), prefer not to answer (4): not stated (gap).
OTHER (specify): "biyo maal" (11) is a clan (Biyomaal, Dir, Lower Shabelle), not a language,
drawn on Standard Somali; "oromo" (1) on Oromo; the 9 blank "other" on `other`.

POPULATION. religiondots' so_lookup.csv (read-only): COD-PS 2026 HRP region figures,
19,442,160 people on 18 regions, Somaliland's five included.

MIDDLE JUBA (SO27, 447,217 people) has no respondent: its only sampled district, Jilib, was
dropped in cleaning. It is drawn on Lower Juba's shares, its downstream neighbour on the Juba,
and its note says so.
"""
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "so" / "REACH_SOM2101_Final-Dataset_JMCNA_Somalia_01112021.xlsx"
URL = ("https://repository.impact-initiatives.org/document/impact/a261cf1e/"
       "REACH_SOM2101_Final-Dataset_JMCNA_Somalia_01112021.xlsx")
CACHE = HERE / "data" / "raw" / "so" / "jmcna2021_language.csv"
OUT = HERE / "data" / "normalized" / "so.csv"
POP2026 = 19_442_160

REGION = {"awdal": "SO11", "woqooyi_galbeed": "SO12", "togdheer": "SO13", "sool": "SO14",
          "sanaag": "SO15", "bari": "SO16", "nugaal": "SO17", "mudug": "SO18",
          "galgaduud": "SO19", "hiraan": "SO20", "middle_shabelle": "SO21", "banadir": "SO22",
          "lower_shabelle": "SO23", "bay": "SO24", "bakool": "SO25", "gedo": "SO26",
          "lower_juba": "SO28"}
BORROW = {"SO27": "SO28"}          # Middle Juba <- Lower Juba
LABEL = {"standard": "Standard / Northern Somali", "banaadir": "Benaadir Somali",
         "maay": "Maay Somali", "arabic": "Arabic", "english": "English", "italian": "Italian",
         "bravanese": "Bravanese (Chimwiini / Chimbalazi)", "kibajuni": "Kibajuni",
         "mushunguli": "Mushunguli", "other": "Other"}
NOT_DRAWN = {"somali_sign", "dnk", "prefer_not_answer"}
OTHER_SPEC = {"biyo maal": "standard", "oromo": "Oromo"}


def fetch():
    if not RAW.exists():
        import urllib.request
        RAW.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        RAW.write_bytes(urllib.request.urlopen(req, timeout=600).read())
    import openpyxl
    z = zipfile.ZipFile(RAW)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as o:   # styles.xml makes it crawl
        for n in z.namelist():
            if n != "xl/styles.xml":
                o.writestr(n, z.read(n))
    buf.seek(0)
    ws = openpyxl.load_workbook(buf, read_only=True)["Clean_Data"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    cols = ["region", "district", "idp_settlement", "main_language", "main_language_other",
            "weights"]
    ix = [hdr.index(c) for c in cols]
    df = pd.DataFrame([[r[i] for i in ix] for r in it], columns=cols)
    df.to_csv(CACHE, index=False, encoding="utf-8")
    return df


def main():
    d = pd.read_csv(CACHE) if CACHE.exists() else fetch()
    assert len(d) == 11349, len(d)
    d["weights"] = pd.to_numeric(d["weights"])     # stored as text in the workbook
    assert abs(d["weights"].mean() - 1) < 1e-6
    d["unit"] = d["region"].map(REGION)
    assert d["unit"].notna().all(), sorted(d.loc[d["unit"].isna(), "region"].unique())
    known = set(LABEL) | NOT_DRAWN
    assert set(d["main_language"]) <= known, set(d["main_language"]) - known
    oth = d["main_language"] == "other"
    spec = d["main_language_other"].astype(str).str.strip().str.lower()
    cat = d["main_language"].map(LABEL)
    for k, v in OTHER_SPEC.items():
        m = oth & (spec == k)
        cat[m] = LABEL.get(v, v)
    left = oth & ~spec.isin(OTHER_SPEC) & d["main_language_other"].notna()
    assert not left.any(), d.loc[left, "main_language_other"].unique()
    d["cat"] = cat
    drop = d["main_language"].isin(NOT_DRAWN)
    print("not drawn (weighted share of households):",
          (d.loc[drop].groupby("main_language")["weights"].sum() / d["weights"].sum()).round(4)
          .to_dict())
    k = d[~drop]
    sh = k.groupby(["unit", "cat"])["weights"].sum()
    sh = sh / sh.groupby(level=0).transform("sum")
    n_resp = k.groupby("unit").size()

    lut = pd.read_csv(RD_GEO / "so" / "so_lookup.csv", dtype={"unit": str, "geo_id": str})
    assert len(lut) == 18 and int(lut["pop"].sum()) == POP2026
    rows = []
    for r in lut.itertuples():
        src = BORROW.get(r.unit, r.unit)
        s = sh.loc[src]
        x = s * int(r.pop)
        n = x.astype(int)
        n[(x - n).sort_values(ascending=False).index[:int(r.pop) - int(n.sum())]] += 1
        assert int(n.sum()) == int(r.pop)
        note = (f"{n_resp[src]} households" if src == r.unit else
                f"no respondent; Lower Juba's shares ({n_resp[src]} households)")
        for c, v in n.items():
            if v > 0:
                rows.append(dict(geo_id=r.unit, geo_level="region", geo_name=r.name,
                                 source_category=c, count=int(v), tier="modelled",
                                 source_id="reach_jmcna_2021_somalia", year=2021,
                                 note=f"share {s[c]:.4f}; {note}"))
    df = pd.DataFrame(rows)
    assert int(df["count"].sum()) == POP2026 and df["geo_id"].nunique() == 18
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {df['geo_id'].nunique()} regions, {df['count'].sum():,} people")
    print((nat / nat.sum() * 100).round(2).to_string())
    print(df.pivot_table(index="geo_name", columns="source_category", values="count",
                         aggfunc="sum").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
