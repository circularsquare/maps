"""Greenland: first language from the population register's birthplace by locality and
citizenship -> data/normalized/gl.csv and data/geo/gl/gl_places.gpkg.

    python sources/gl_census.py [--fetch]

NO LANGUAGE QUESTION. Greenland has had no census since the register replaced it, and the
register has no language item. Built under AGENT_BRIEF §2 (2026-10-05, no language question):
the national language for the native-born, immigrant languages proxied by citizenship.

- Born in Greenland -> Greenlandic, by where they live: Tunumiisut (East Greenlandic, Glottolog
  tunu1234, a language of its own) in the Tasiilaq and Ittoqqortoormiit districts; Inuktun
  (Polar Inuit, Glottolog's Polar Eskimo) in the Qaanaaq district; Kalaallisut everywhere else.
- Born outside Greenland -> foreign citizens by citizenship (origin_mix, dest "gl"); the rest of
  the born-outside, Danish citizens born in Denmark and elsewhere, on Danish. Nuuk town takes
  Nuuk's own citizenship table (BEXST6NUK), every other locality the national table less Nuuk.

TABLES (Statistics Greenland Statbank, 1 January 2026):
- BEXSTD, population by place of birth (Total / Greenland / Born outside Greenland) and locality:
  religiondots' download data/raw/gl/bexstd_2026_born.csv, read only; locality codes from its
  gl_localities_born_2026.csv.
- BEXST6 and BEXST6NUK, population by citizenship, Greenland and Nuuk: fetched here to
  data/raw/gl/.

PLACEMENT: religiondots' gl_hexes.gpkg (one 1.5 km disc per town or settlement, read only),
re-keyed so each disc is its own unit (the locality code). The register's "Uoplyst i <district>"
rows (no known locality) go on the district's main town.

CHECKS: per locality Greenland + outside = Total; localities sum to 56,740; the citizenship
tables sum to their totals; Nuuk's foreign citizens are no more than the nation's per country.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD, RD_GEO  # noqa: E402

RDRAW = RD / "data" / "raw" / "gl"
RAW = HERE / "data" / "raw" / "gl"
OUT = HERE / "data" / "normalized" / "gl.csv"
GEO = HERE / "data" / "geo" / "gl" / "gl_places.gpkg"
API = "https://bank.stat.gl/api/v1/en/Greenland/BE/BE01/BE0125/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
TOTAL = 56_740
NUUK = "9560600NUK"
EAST = {"18", "19"}       # Tasiilaq, Ittoqqortoormiit districts of 956 (Sermersooq)
QAANAAQ = {"17"}          # Qaanaaq district of 960 (Avannaata)
# citizenship label -> ISO for origin_mix; pooled rows -> a node directly
ISO = {"Finland": "FI", "Iceland": "IS", "Norway": "NO", "Sweden": "SE", "Estonia": "EE",
       "Latvia": "LV", "Lithuania": "LT", "Belgium": "BE", "Cyprus": "CY", "France": "FR",
       "Greece": "GR", "Netherlands": "NL", "Ireland": "IE", "Italy": "IT", "Luxembourg": "LU",
       "Poland": "PL", "Portugal": "PT", "Slovakia": "SK", "Croatia": "HR", "Spain": "ES",
       "Great Britain": "GB", "Czechia": "CZ", "Czechoslovakia (former)": "QT", "Germany": "DE",
       "Hungary": "HU", "Austria": "AT", "Switzerland": "CH", "Bulgaria": "BG", "Romania": "RO",
       "Turkey": "TR", "Russia": "RU", "Soviet Union": "SU", "Belarus": "BY", "Ukraine": "UA",
       "USA": "US", "Canada": "CA", "Morocco": "MA", "Somalia": "SO", "Afghanistan": "AF",
       "Bangladesh": "BD", "Philippines": "PH", "India": "IN", "Iraq": "IQ", "Iran": "IR",
       "Japan": "JP", "China": "CN", "Lebanon": "LB", "Pakistan": "PK", "Sri Lanka": "LK",
       "Syria": "SY", "Thailand": "TH", "Vietnam": "VN"}
POOLED = {"Other Europe": "other", "Other America": "other", "Other Africa": "africa_other",
          "Other Asia": "other", "Oceania": "other"}
# Stateless, Unknown and "???" stay with the Danish remainder (6 people nationally)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    q = {"query": [{"code": "age", "selection": {"filter": "item", "values": ["-1"]}},
                   {"code": "citizenship", "selection": {"filter": "all", "values": ["*"]}},
                   {"code": "time", "selection": {"filter": "item", "values": ["2026"]}}],
         "response": {"format": "csv"}}
    for t in ("BEXST6", "BEXST6NUK"):
        req = urllib.request.Request(API + t + ".px", data=json.dumps(q).encode(),
                                     headers={"User-Agent": UA,
                                              "Content-Type": "application/json"})
        body = urllib.request.urlopen(req, timeout=120).read()
        (RAW / f"{t.lower()}_2026.csv").write_bytes(body)
        print(f"  {t}: {len(body):,} bytes")


def citizenship(t):
    d = pd.read_csv(RAW / f"{t}_2026.csv", encoding="latin-1")
    d.columns = ["age", "cit", "n"]
    d["cit"] = d["cit"].str.replace(r"^Chinese People.s Republic$", "China", regex=True)
    s = d.set_index("cit")["n"]
    assert s.drop("Total").sum() == s["Total"], t
    return s.drop(["Total", "Denmark"])


def mixes(foreign, outside):
    """{node: count} for `outside` born-outside people whose foreign citizens are `foreign`."""
    from origin_mix import mix
    out = {}
    named = 0
    for lab, n in foreign.items():
        if n == 0 or lab in ("Stateless", "Unknown", "???"):
            continue
        named += n
        m = {POOLED[lab]: 1.0} if lab in POOLED else mix(ISO[lab], "gl")
        for node, s in m.items():
            out[node] = out.get(node, 0) + n * s
    assert named <= outside, (named, outside)
    dan = "indoeuropean.germanic.north.danish"
    out[dan] = out.get(dan, 0) + outside - named
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    b = pd.read_csv(RDRAW / "bexstd_2026_born.csv")
    b.columns = ["pob", "locality", "n"]
    w = b.pivot(index="locality", columns="pob", values="n")
    assert w.loc["Total", "Total"] == TOTAL
    w = w.drop("Total")
    assert (w["Greenland"] + w["Born outside Greenland"] == w["Total"]).all()
    assert w["Total"].sum() == TOTAL
    lut = pd.read_csv(RDRAW / "gl_localities_born_2026.csv", dtype=str)
    code = dict(zip(lut["locality"], lut["code"]))
    missing = sorted(set(w.index) - set(code))
    assert not missing, missing

    import geopandas as gpd
    g = gpd.read_file(RD_GEO / "gl" / "gl_hexes.gpkg")
    assert g["code"].is_unique
    have = set(g["code"])

    def unit(c):
        # code = municipality (3) + district (2) + locality (2) + letters; "00" is the town
        if c in have:
            return c
        same = sorted(x for x in have if x[:5] == c[:5] and x[5:7] == "00")
        if same:
            return same[0]
        same = sorted(x for x in have if x[:3] == c[:3])
        assert same, c
        return same[0]
    w["code"] = [code[l] for l in w.index]
    w["unit"] = w["code"].map(unit)
    moved = w[(w["code"] != w["unit"]) & (w["Total"] > 0)]
    print(f"  {len(moved)} localities without a disc folded onto a town: "
          f"{moved['Total'].sum():,} people")

    nat = citizenship("bexst6")
    nuk = citizenship("bexst6nuk")
    assert (nuk.reindex(nat.index, fill_value=0) <= nat).all()
    rest = nat - nuk.reindex(nat.index, fill_value=0)
    out_nuuk = w.loc[w["code"] == NUUK, "Born outside Greenland"].sum()
    out_rest = w.loc[w["code"] != NUUK, "Born outside Greenland"].sum()
    m_nuuk = {k: v / out_nuuk for k, v in mixes(nuk, out_nuuk).items()}
    m_rest = {k: v / out_rest for k, v in mixes(rest, out_rest).items()}

    rows = []
    for loc, r in w.iterrows():
        muni, dist = r["code"][:3], r["code"][3:5]
        if muni == "956" and dist in EAST:
            gnode = "Tunumiisut"
        elif muni == "960" and dist in QAANAAQ:
            gnode = "Inuktun"
        else:
            gnode = "Kalaallisut"
        if r["Greenland"]:
            rows.append(dict(geo_id=r["unit"], geo_name=loc, source_category=gnode,
                             count=float(r["Greenland"])))
        o = r["Born outside Greenland"]
        if o:
            m = m_nuuk if r["code"] == NUUK else m_rest
            for node, s in m.items():
                rows.append(dict(geo_id=r["unit"], geo_name=loc,
                                 source_category="born outside: " + node, count=o * s))
    df = pd.DataFrame(rows)
    df["geo_level"] = "locality"
    df["tier"] = "derived"
    df["year"] = 2026
    assert abs(df["count"].sum() - TOTAL) < 1e-6, df["count"].sum()
    df.to_csv(OUT, index=False, encoding="utf-8")
    s = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(s.round(0).head(15).to_string())
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,.0f} people, "
          f"{df['geo_id'].nunique()} units")

    g2 = g.copy()
    g2["unit"] = g2["code"]
    GEO.parent.mkdir(parents=True, exist_ok=True)
    g2.to_file(GEO, driver="GPKG")
    print(f"  wrote {GEO.relative_to(HERE)}: {len(g2)} discs")


if __name__ == "__main__":
    main()
