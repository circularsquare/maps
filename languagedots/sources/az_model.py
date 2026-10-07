"""Azerbaijan: a NATIONALITY MODEL of mother tongue by rayon -> data/normalized/az_model.csv.

    python sources/az_model.py

NOT DRAWN unless countries/az.py's MODEL is True (ask 006). The census prints mother tongue for the
country only (sources/az_census.py), so the drawn map is national grain. This is the alternative:
the 2019 census's mother tongue by nationality, placed where each nationality lives.

THE MODEL. For each of the 74 rayons and cities (religiondots' COD-AB units, read-only:
`religiondots/data/geo/az/az_lookup.csv`, 2019 EXISTING population, 66 populated):
  1. every nationality except Azerbaijanis: its 2019 national count (Table 30) spread over the
     populated units in proportion to its 2009 count there (the 2009 census's nationality by rayon,
     *XIX cild*, in Tim Bespyatov's transcription, the file religiondots already holds and checked
     against the committee's tables 1.11 and 1.17; read-only). The four units the 2019 census
     found partly held keep their 2009 Armenians out, as in religiondots (they were the Karabakh
     Armenians the 2009 census estimated). 2019 nationalities with no 2009 column take one:
     Ingiloys the Georgians', Grysz and Haputs the Kryts', Budukhs "other".
  2. Azerbaijanis are the rest of the unit's 2019 existing population.
  3. each nationality's people in a unit are split over mother tongues by THAT NATIONALITY'S
     NATIONAL SPLIT in Table 30: Lezgins 75.1% Lezgian and 24.6% Azerbaijani everywhere, Talysh
     49.4% Talysh and 50.5% Azerbaijani everywhere.

So the part that does not match (people whose mother tongue is not their nationality's language)
is a published number per nationality, applied flat across places. What the model cannot see:
whether Baku's Lezgins name Azerbaijani more often than Gusar's (almost certainly), and the change
in where each nationality lives between 2009 and 2019. Every row is `modelled`.
"""
import html
import os
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402

LOOKUP = RD_GEO / "az" / "az_lookup.csv"
ETH2009 = RD / "data" / "raw" / "az" / "mashke_azerbaijan_ethnic2009.htm"
NORM = ROOT / "data" / "normalized" / "az.csv"
OUT = ROOT / "data" / "normalized" / "az_model.csv"
EXISTING = 9_943_958

# 2019 nationality (Table 30's label) -> the 2009 column that places it
PLACE_BY = {"Lezgi": "Lezgins", "Talish": "Talyshs", "Russian": "Russians",
            "Ukrainian": "Ukrainians", "Avar": "Avars", "Turkish": "Turks", "Tat": "Tats",
            "Sakhur": "Tsakhurs", "Georgian": "Georgians", "Ingiloy": "Georgians", "Kurd": "Kurds",
            "Tatarian": "Tatars", "Griz": "Kryts", "Jews": "Jews", "Udin": "Udins",
            "Khinalig": "Khinalugs", "Budug": "Other", "Armenian": "Armenians", "Khaput": "Kryts",
            "Other": "Other"}
PARTLY_HELD = {"Aghdam", "Fuzuli", "Tartar", "Jabrayil"}


def fold(s):
    s = str(s).replace("ə", "e").replace("Ə", "E")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("ı", "i")
    return re.sub(r"[^a-z]", "", s)


def read_2009():
    """As religiondots' sources/az.py: the unit rows (no Baku districts, no urban/rural rows)."""
    h = ETH2009.read_text(encoding="utf-8-sig")
    rows = []
    for r in re.split(r"(?i)<tr[^>]*>", h)[1:]:
        rows.append([html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                     for c in re.split(r"(?i)<t[dh][^>]*>", r)[1:]])
    eng = next(r for r in rows if len(r) > 2 and r[1] == "Total")
    cols = ["unit"] + eng[1:]
    body = [r for r in rows if len(r) == len(cols) and r[0] and not r[0].startswith("-")
            and r[1] not in ("Cəmi", "Total")]
    df = pd.DataFrame(body, columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c].str.replace("-", "0"), errors="raise").astype(int)
    df["unit"] = df["unit"].str.replace(r"\s+ş\.$", "", regex=True).str.strip()
    df = df.set_index("unit")
    if not (df.drop(columns=["Total"]).sum(axis=1) == df["Total"]).all():
        raise SystemExit("2009 rows whose groups do not sum to Total")
    return df


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 74 or int(lut["pop"].sum()) != EXISTING:
        raise SystemExit(f"{LOOKUP}: {len(lut)} units, {int(lut['pop'].sum()):,} people")
    pop = dict(zip(lut["geo_id"], lut["pop"]))
    name = dict(zip(lut["geo_id"], lut["name"]))

    e09 = read_2009()
    key = {fold(n): u for n, u in zip(lut["name_az"], lut["geo_id"])}
    units09 = [n for n in e09.index if n not in ("Azərbaycan", "Naxçıvan MR")]
    j = {n: key.get(fold(n)) for n in units09}
    if None in j.values() or len(set(j.values())) != 74 or len(j) != 74:
        raise SystemExit(f"2009 units do not join one-to-one onto the 74 COD-AB units: "
                         f"{[n for n, u in j.items() if u is None]}")
    e = e09.loc[units09].copy()
    e.index = [j[n] for n in e.index]
    nat_row = e09.loc["Azərbaycan"]
    bad = [c for c in e.columns if int(e[c].sum()) != int(nat_row[c])]
    if bad:
        raise SystemExit(f"2009: the 74 units do not sum to the national row in {bad}")
    print("2009 nationality by unit: 74 units joined one-to-one on the Azerbaijani name; every "
          "column sums to the national row")

    t30 = pd.read_csv(NORM)
    t30 = t30[t30["area"] == "total"]
    natpop = t30.groupby("nationality")["count"].sum()
    split = {n: g.set_index("source_category")["count"] / natpop[n]
             for n, g in t30.groupby("nationality")}

    rows, placed = [], {u: 0.0 for u in pop}
    for nat, col in PLACE_BY.items():
        w = e[col].astype(float).copy()
        w[[u for u in w.index if pop[u] == 0]] = 0.0
        if nat == "Armenian":
            w[[u for u in w.index if name[u] in PARTLY_HELD]] = 0.0
        n_u = natpop[nat] * w / w.sum()
        for u, n in n_u.items():
            if n <= 0:
                continue
            placed[u] += n
            for cat, s in split[nat].items():
                rows.append((u, nat, cat, n * s))
    over = [name[u] for u in pop if placed[u] > pop[u]]
    if over:
        raise SystemExit(f"units where the placed nationalities exceed the population: {over}")
    for u in pop:
        if pop[u] > 0:
            for cat, s in split["Azerbaijani"].items():
                rows.append((u, "Azerbaijani", cat, (pop[u] - placed[u]) * s))
    df = pd.DataFrame(rows, columns=["geo_id", "nationality", "source_category", "count"])
    df = df.groupby(["geo_id", "source_category"], as_index=False)["count"].sum()
    df = df[df["count"] > 0]
    df["geo_level"] = "unit"
    df["geo_name"] = df["geo_id"].map(name)
    df["tier"] = "modelled"
    if abs(df["count"].sum() - EXISTING) > 1:
        raise SystemExit(f"model sums to {df['count'].sum():,.0f}, not {EXISTING:,}")
    df[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier"]].to_csv(
        OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {df['geo_id'].nunique()} units, {df['count'].sum():,.0f} people")

    sys.path.insert(0, str(ROOT / "taxonomy"))
    import az2019
    df["node"] = df["source_category"].map(az2019.resolve)
    piv = df.groupby(["geo_name", "node"])["count"].sum().unstack(fill_value=0)
    piv = piv.div(piv.sum(axis=1), axis=0)
    print("\n  units where Azerbaijani is under 90% of mother tongues:")
    az = "turkic.azerbaijani"
    for u in piv.index[piv[az] < 0.9]:
        top = piv.loc[u].sort_values(ascending=False).head(4)
        print(f"    {u:<12} " + ", ".join(f"{k.split('.')[-1]} {v:.1%}" for k, v in top.items()))


if __name__ == "__main__":
    main()
