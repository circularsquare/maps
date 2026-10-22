"""Libya: non-Libyans, from IOM's Displacement Tracking Matrix migrant baseline assessment.

    python sources/ly_migrants.py     check the sheets and write data/normalized/ly_migrants.csv
                                      (district, nationality, people) and ly_migrant_sites.csv
                                      (muhalla points with their migrants)

The base of this map is the Libyan population (BSC 2020 estimate); the UN counts about 827,000
non-Libyans (migrant stock). IOM DTM Libya's Round 32 (July-August 2020; HDX
`libya-migration-data-migrants-baseline-assessment-iom-dtm`, resource 5b23eecf-8550-4c50-ada6-
40d02b94a419, data/raw/ly/dtm_ly_migrants_r32.xlsx) counted 584,509 migrants in 575 muhallas
(neighbourhoods) of 100 baladiyas (municipalities) in the 22 mantikas, with nationality per muhalla
(sheet "Nationalities": 24 named nationalities and "Other"). Round 32 is the one with a
nationality sheet nearest the census year; earlier rounds (R28-R31) are totals only, R33 has no
nationality sheet. Later rounds count more (R49, 2023: about 700,000).

Used as counted, not scaled to the UN's 827,000: the DTM is a key-informant count of the migrants
present, the UN figure is a stock estimate for a different date, and nothing says where the extra
people would be. Both numbers are in sources/ly.md.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import pandas as pd  # noqa: E402

RAW = ROOT / "data" / "raw" / "ly" / "dtm_ly_migrants_r32.xlsx"
OUT = ROOT / "data" / "normalized" / "ly_migrants.csv"
SITES = ROOT / "data" / "normalized" / "ly_migrant_sites.csv"
TOTAL = 584_509

# DTM mantika -> the 22 districts of religiondots' ly layer (BSC's shabiyat)
MANTIKA = {
    "Derna": "LY0101", "Almarj": "LY0102", "Benghazi": "LY0103", "Tobruk": "LY0104",
    "Ejdabia": "LY0105", "Al Jabal Al Akhdar": "LY0106", "Alkufra": "LY0107", "Sirt": "LY0208",
    "Nalut": "LY0209", "Almargeb": "LY0210", "Tripoli": "LY0211", "Aljfara": "LY0212",
    "Azzawya": "LY0213", "Misrata": "LY0214", "Zwara": "LY0215", "Al Jabal Al Gharbi": "LY0216",
    "Aljufra": "LY0317", "Wadi Ashshati": "LY0318", "Sebha": "LY0319", "Ubari": "LY0320",
    "Ghat": "LY0321", "Murzuq": "LY0322",
}
# the sheet's nationality labels -> ISO 3166 alpha-2 ("Other" -> None, drawn as `other`)
ISO = {"Algeria": "DZ", "Bangladesh": "BD", "BurkinaFaso": "BF", "CotedIvoire": "CI",
       "Cameroon": "CM", "Chad": "TD", "Egypt": "EG", "Eritrea": "ER", "Ethiopia": "ET",
       "Ghana": "GH", "Guinea": "GN", "Mali": "ML", "Mauritania": "MR", "Morocco": "MA",
       "Niger": "NE", "Nigeria": "NG", "Other": None, "Pakistan": "PK", "Palestine": "PS",
       "Senegal": "SN", "Somalia": "SO", "Sudan": "SD", "Syria": "SY", "Tunisia": "TN",
       "Zambia": "ZM"}


def _clean(s):
    return "".join(ch for ch in str(s) if ch.isascii()).strip()


def read():
    n = pd.read_excel(RAW, sheet_name="Nationalities")
    c = list(n.columns)
    n = n.rename(columns={c[2]: "mantika", c[4]: "baladiya", c[6]: "muhalla", c[8]: "pcode",
                          c[9]: "nat", c[10]: "people"})
    n["nat"] = n["nat"].map(lambda s: "CotedIvoire" if "tedIvoire" in str(s) else _clean(s))
    n = n[["mantika", "baladiya", "muhalla", "pcode", "nat", "people"]].copy()
    n["people"] = n["people"].astype(int)
    m = pd.read_excel(RAW, sheet_name="Muhalla Main", header=1)
    m = m.rename(columns={"Muhalla_pcode": "pcode", "Latitude": "lat", "Longitude": "lon",
                          "Migrant IND": "ind"})[["pcode", "lat", "lon", "ind"]]
    m = m.drop_duplicates("pcode")
    return n, m


def build():
    n, m = read()
    if n["people"].sum() != TOTAL:
        raise SystemExit(f"ly_migrants: {n['people'].sum():,} migrants, expected {TOTAL:,}")
    bad = sorted(set(n["mantika"]) - set(MANTIKA))
    if bad or set(MANTIKA) - set(n["mantika"]):
        raise SystemExit(f"ly_migrants: mantikas {bad} / {sorted(set(MANTIKA) - set(n['mantika']))}")
    bad = sorted(set(n["nat"]) - set(ISO))
    if bad:
        raise SystemExit(f"ly_migrants: nationalities with no ISO code: {bad}")
    # the muhalla counts of the main sheet equal the nationality sheet's (checked 2026-10-19)
    s = n.groupby("pcode")["people"].sum()
    j = m.set_index("pcode")["ind"].reindex(s.index)
    if (s != j).any():
        raise SystemExit("ly_migrants: Muhalla Main and Nationalities disagree")
    n["unit"] = n["mantika"].map(MANTIKA)
    n.groupby(["unit", "nat"], as_index=False)["people"].sum().to_csv(OUT, index=False)
    sites = n.groupby(["unit", "baladiya", "pcode"], as_index=False)["people"].sum().merge(m[["pcode", "lat", "lon"]],
                                                                                          on="pcode", how="left")
    sites.to_csv(SITES, index=False)
    return n, sites


_INV = {}


def _written(node):
    """origin_mix reads other countries' drawn (regrouped) ids; counts() uses written ones, which
    countries.py regroups (taxonomy/regroup.py). Same as sources/sg_nonres.py."""
    if not _INV:
        from regroup import move
        ids = set()
        for f in [ROOT / "taxonomy" / "tree.txt", *sorted((ROOT / "taxonomy" / "tree.d").glob("*.txt"))]:
            for ln in f.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "|" in ln:
                    ids.add(ln.split("|")[0].strip())
        for w in sorted(ids):
            _INV.setdefault(move(w), w)
    return _INV.get(node, node)


def counts():
    """DataFrame [unit, node, count]: each district's migrants of each nationality on the languages
    of that nationality's drawn home mix (origin_mix.mix(iso, "ly")), 1% cut as everywhere."""
    import origin_mix as om
    d = pd.read_csv(OUT)
    out = []
    for r in d.itertuples():
        iso = ISO[r.nat]
        if iso is None:
            out.append((r.unit, "other", float(r.people)))
            continue
        for node, share in om.mix(iso, "ly").items():
            out.append((r.unit, _written(node), r.people * share))
    df = pd.DataFrame(out, columns=["unit", "node", "count"])
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    assert abs(df["count"].sum() - TOTAL) < 1e-3, df["count"].sum()
    return df


if __name__ == "__main__":
    n, sites = build()
    print(f"{OUT}: {n['people'].sum():,} migrants, {n['nat'].nunique()} nationalities, {n['unit'].nunique()} districts")
    print(n.groupby("unit")["people"].sum().sort_values(ascending=False).head(8).to_string())
    print(f"{len(sites)} muhallas, {int(sites['lat'].isna().sum())} without coordinates "
          f"({int(sites.loc[sites['lat'].isna(), 'people'].sum()):,} people)")
