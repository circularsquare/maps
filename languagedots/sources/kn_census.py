"""St Kitts and Nevis: first language from the 2011 census's country of birth by island
-> data/normalized/kn.csv and data/geo/kn/kn_hexes.gpkg.

    python sources/kn_census.py [--fetch]

NO CENSUS LANGUAGE QUESTION (2011). Built as Barbados and Antigua (sources/bb.md, ag.md): the
native-born on the Leeward creole (Glottolog's Antigua and Barbuda Creole English, anti1245,
which covers St Kitts and Nevis; node `antiguan` from tree.d/bb.txt), the foreign-born on their
birth country's languages through sources/origin_mix.py (dest "kn"), with the English-Caribbean
creoles named where the origin is not drawn on this map. Every row `derived`.

TABLES (Department of Statistics, stats.gov.kn, HTML tables, saved to data/raw/kn/):
- "Foreign Born Population by Country of Birth (2011)", columns St Kitts, Nevis, total.
- "Number of Households and Population by Parish and Island 2001 to 2011": St Kitts 34,918,
  Nevis 12,277, total 47,195 (the figure religiondots draws).
Native-born per island = island population - foreign-born (including the 212 whose birthplace
was not stated, who are left out: `gap`).

PLACEMENT: religiondots' kn_hexes.gpkg is one unit (read-only); re-keyed by island at 17.217 N
(the Narrows). Check: Kontur's island split within a factor 1.5 of the census's.
"""
import os
import sys
import urllib.request
from io import StringIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "kn"
OUT = HERE / "data" / "normalized" / "kn.csv"
GEO = HERE / "data" / "geo" / "kn" / "kn_hexes.gpkg"
BASE = "https://www.stats.gov.kn/topics/demographic-social-statistics/population/"
PAGES = {"kn_foreign_born_2011.html": "foreign-born-population-by-country-of-birth-2011/",
         "kn_pop_by_parish_2011.html":
             "number-of-households-and-population-by-parish-and-island-2001-to-2011/"}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
ISLANDS = {"St Kitts": ("KN-K", 34_918), "Nevis": ("KN-N", 12_277)}
SPLIT_LAT = 17.217   # Nevis's northernmost hex centroid 17.209, St Kitts's southernmost 17.225

EN = "indoeuropean.germanic.english"
CR = "creole.english_based."
LEEWARD = CR + "antiguan"
# rows that are region subtotals, not countries
SUBTOTALS = {"O.E.C.S. Countries", "Caricom Countries (Non-O.E.C.S)", "Rest Of The Americas",
             "Rest Of Europe", "Rest Of The World", "Total", "Not Stated"}
# birthplace -> node, where origin_mix is not used (bb.md / bs.md conventions)
NODE = {
    "Antigua And Barbuda": LEEWARD, "Monsterrat": LEEWARD, "Anguilla": LEEWARD,
    "British Virgin Islands": CR + "virgin_islands", "St. Croix": CR + "virgin_islands",
    "St. Thomas": CR + "virgin_islands",
    "Usvi United States Virgin Islands (Not Stated)": CR + "virgin_islands",
    "Grenada et al": CR + "grenadian", "Guyana": CR + "guyanese",
    "Trinidad And Tobago": CR + "trinidadian", "St. Lucia": "creole.french_based.antillean",
    "Turks And Caicos": CR + "turks_caicos",
    # as Barbados, Antigua and Bermuda: many are Kittitians' children; the US and Canadian home
    # mixes' Spanish and French would be wrong here
    "Usa": EN, "Canada": EN, "United Kingdom": EN, "Bermuda": EN, "Cayman Islands": EN,
    "Belize": EN, "Puerto Rico": "indoeuropean.romance.spanish",
}
ISO = {
    "Dominica": "DM", "St. Vincent And The Grenadines": "VC", "Bahamas": "BS", "Barbados": "BB",
    "Haiti": "HT", "Jamaica": "JM", "Suriname": "SR", "Argentina": "AR", "Aruba": "AW",
    "Brazil": "BR", "Colombia": "CO", "Cuba": "CU", "Curacao": "CW", "Dominican Republic": "DO",
    "Ecuador": "EC", "El Salvador": "SV", "French Guyana": "GF", "Guadeloupe": "GP",
    "Guatemala": "GT", "Honduras": "HN", "Martinique": "MQ", "Mexico": "MX",
    "Netherlands Antilles": "CW", "Panama": "PA", "Peru": "PE", "St Eustatius": "BQ",
    "St. Martin/St. Maarteen": "SX", "Venezuela": "VE", "Austria": "AT", "Belgium": "BE",
    "Cyprus": "CY", "Denmark": "DK", "France": "FR", "Germany": "DE", "Greece": "GR",
    "Hungary": "HU", "Ireland": "IE", "Italy": "IT", "Malta": "MT", "Netherlands": "NL",
    "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Russia": "RU", "Spain": "ES",
    "Sweden": "SE", "Switzerland": "CH", "Afghanistan": "AF", "Australia": "AU",
    "Bangladesh": "BD", "Botswanna": "BW", "Cameroon": "CM", "China": "CN", "Congo": "CG",
    "Czech Republic": "CZ", "Egypt": "EG", "Equatorial Guinea": "GQ", "Ethiopia": "ET",
    "Gambia": "GM", "Ghana": "GH", "Greenland": "GL", "Hong Kong": "HK", "India": "IN",
    "Indonesia": "ID", "Iran (Islamic Republic Of)": "IR", "Iraq": "IQ", "Israel": "IL",
    "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE",
    "Korea, Democratic People'S Rep": "KP", "Korea, Republic Of": "KR", "Lebanon": "LB",
    "Malaysia": "MY", "Malawi": "MW", "Mauritius": "MU", "Mongolia": "MN", "Nepal": "NP",
    "Niger": "NE", "Nigeria": "NG", "Norway": "NO", "Pakistan": "PK", "Papua New Guinea": "PG",
    "Paraguay": "PY", "Philippines": "PH", "Saudi Arabia": "SA", "Sierra Leone": "SL",
    "Singapore": "SG", "Somalia": "SO", "South Africa": "ZA", "Sri Lanka": "LK", "Sudan": "SD",
    "Swaziland": "SZ", "Syria": "SY", "Taiwan, Republic Of China": "TW",
    "Tanzania, United Republic Of": "TZ", "Thailand": "TH", "Togo": "TG", "Tunisia": "TN",
    "Uganda": "UG", "Ukraine": "UA", "United Arab Emirates": "AE", "Viet Nam": "VN",
    "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW",
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for fn, path in PAGES.items():
        req = urllib.request.Request(BASE + path, headers={"User-Agent": UA})
        (RAW / fn).write_bytes(urllib.request.urlopen(req, timeout=60).read())


def table(fn):
    return pd.read_html(StringIO((RAW / fn).read_text(encoding="utf-8", errors="replace")))[0]


def main():
    if "--fetch" in sys.argv:
        fetch()
    from origin_mix import mix

    p = table("kn_pop_by_parish_2011.html")
    p.columns = ["parish", "h01", "p01", "h11", "p11"]
    pop = dict(zip(p["parish"], p["p11"]))
    for isl, (_, n) in ISLANDS.items():
        assert pop[isl] == n, (isl, pop[isl])
    assert pop["St Kitts & Nevis"] == sum(n for _, n in ISLANDS.values())

    f = table("kn_foreign_born_2011.html")
    f.columns = ["country", "St Kitts", "Nevis", "both"]
    f = f.dropna(subset=["both"])
    f = f[~f["country"].str.startswith(("Date", "Source", "Note"))]
    assert (f["St Kitts"] + f["Nevis"] == f["both"]).all()
    tot = f[f["country"] == "Total"].iloc[0]
    rows = f[~f["country"].isin(SUBTOTALS)]
    for isl in ISLANDS:   # countries + not stated = the printed total
        ns = f.loc[f["country"] == "Not Stated", isl].iloc[0]
        assert rows[isl].sum() + ns == tot[isl], (isl, rows[isl].sum(), ns, tot[isl])
    missing = sorted(set(rows["country"]) - set(NODE) - set(ISO))
    assert not missing, missing

    out = []
    for isl, (unit, n) in ISLANDS.items():
        native = n - int(tot[isl])
        out.append(dict(geo_id=unit, geo_name=isl, birthplace="St Kitts and Nevis",
                        source_category=LEEWARD, count=native))
        for _, r in rows.iterrows():
            if not r[isl]:
                continue
            m = {NODE[r["country"]]: 1.0} if r["country"] in NODE else mix(ISO[r["country"]], "kn")
            for node, s in m.items():
                out.append(dict(geo_id=unit, geo_name=isl, birthplace=r["country"],
                                source_category=node, count=r[isl] * s))
    df = pd.DataFrame(out)
    df["geo_level"] = "island"
    df["tier"] = "derived"
    df["year"] = 2011
    ns = int(f.loc[f["country"] == "Not Stated", "both"].iloc[0])
    assert abs(df["count"].sum() - (47_195 - ns)) < 1e-6, df["count"].sum()
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,.0f} people ({ns} not stated)")

    import geopandas as gpd
    g = gpd.read_file(RD_GEO / "kn" / "kn_hexes.gpkg")
    assert set(g["unit"]) == {"KN"}
    cy = g.to_crs(32620).geometry.centroid.to_crs(4326).y
    g["unit"] = ["KN-N" if y < SPLIT_LAT else "KN-K" for y in cy]
    k = g.groupby("unit")["pop"].sum()
    for isl, (unit, n) in ISLANDS.items():
        r = (k[unit] / k.sum()) / (n / 47_195)
        print(f"  {isl}: {(g['unit'] == unit).sum()} hexes, Kontur share / census share {r:.2f}")
        assert 1 / 1.5 <= r <= 1.5
    GEO.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(GEO, driver="GPKG")


if __name__ == "__main__":
    main()
