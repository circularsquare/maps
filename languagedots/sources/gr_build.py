"""Greece: build data/normalized/gr.csv and data/geo/gr/gr_weights.csv.

    python sources/gr_build.py

Greece has asked no language question since the 1951 census. Anita's 2026-10-05 ruling for rich
countries with no language question (AGENT_BRIEF §2): the national language, plus regional and
minority languages from the most recent cited figures, plus immigrant languages proxied by
citizenship. Per NUTS 3 regional unit (52, plus Mount Athos):

  1. population and citizenship: the 2021 census, Eurostat `cens_21ctz_r3` (ELSTAT's figures at
     NUTS 3; religiondots' raw copy, read-only). TOTAL = NAT + FOR + STLS + UNK.
  2. immigrant languages: foreign citizens, each on its country's main language (France's table,
     fr_build.COUNTRY_LANG, with GREECE_OVERRIDES), times RETENTION (Italy's 61.5%, ISTAT 2024
     tav. 11: no Greek survey gives a share); the rest of them Greek.
  3. minority languages, all taken out of the Greek citizens (NAT):
       Turkish (Thrace)   60,000  = 120,000 Muslim minority x 50% (1991 composition), split
                                    over Evros, Xanthi, Rodopi by the 1951 Muslim Turkish speakers
       Turkish (Dodecanese) 4,500 = Muslim associations of Rhodes (2,500) and Kos (2,000)
       Pomak              36,000  = 2001-census-based estimate by regional unit (23k / 11k / 2k)
       Romani            117,495  = 2021 Roma mapping, split by the 2017 mapping's regions,
                                    then by Greek citizens within a region
       Aromanian          50,000  = 2018 estimate, by the 1951 Vlach mother tongue
       Arvanitika         50,000  = Sasse 1991, by the 1951 Albanian mother tongue outside Epirus
       Macedonian / Bulgarian 50,000 = low end of the 50,000-250,000 range, by the 1951
                                    Slavic mother tongue in Macedonia; Drama, Kavala and Serres
                                    as Bulgarian (Trudgill 2000), the rest as Macedonian
  4. Greek: everyone else.
Every row is `derived`. The record is sources/gr.md.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

RAW = os.path.join(ROOT, "data", "raw", "gr")
T1951 = os.path.join(RAW, "census1951_table7b_mothertongue.csv")
OUT = os.path.join(ROOT, "data", "normalized", "gr.csv")
WEIGHTS = os.path.join(ROOT, "data", "geo", "gr", "gr_weights.csv")
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
EU_CTZ = os.path.join(RD, "data", "raw", "gr", "cens_21ctz_r3_el.json")
RD_LAU = os.path.join(RD, "data", "geo", "gr", "gr_lau.gpkg")
YEAR = 2021
SOURCE_ID = "gr_census2021_ctz_x_minority_estimates"

RETENTION = 0.615     # ISTAT 2024 tav. 11 (Italy): non-Italian mother tongues speaking it at home
GREECE_OVERRIDES = {
    "CY": "Greek",
    "IN": "Punjabi",   # Greece's Indians are mostly Punjabi Sikhs (farm labour, Marathon, Kiato)
    "CH": "German", "BE": "Dutch", "CA": "English", "FR": "French",
}
ATHOS = "ELZZZ"       # monks keep their language: every foreign monk drawn on it

# ---- minority figures --------------------------------------------------------------------------
MUSLIM_MINORITY = 120_000          # Greek government figure, ELIAMEP 2006 p. 3 (Alexandris)
TURKISH_SHARE = 0.50               # 1991: 50% Turkish origin, 35% Pomak, 15% Roma
POMAK = {"EL512": 23_000, "EL513": 11_000, "EL511": 2_000}    # 2001-census-based estimate
DODECANESE_TURKISH = {"690101": 2_500, "640101": 2_000}       # Rhodes town, Kos town (LAU mu)
ROMA_2021 = 117_495                # GSSSFAP 2021 mapping (NRIS 2021-2030)
ROMA_2017 = {                      # Operational Action Plan 2017-2021, table 3, by NUTS 2
    "EL30": 30_363, "EL51": 16_435, "EL52": 15_374, "EL63": 15_898, "EL61": 14_534,
    "EL64": 4_958, "EL65": 1_661, "EL54": 1_500, "EL43": 1_211, "EL41": 755, "EL62": 639,
    "EL42": 554, "EL53": 328,
}
AROMANIAN = 50_000                 # 2018 estimate of native speakers in Greece
ARVANITIKA = 50_000                # Sasse 1991
# Euromosaic's Arvanite areas: East and West Attica, Piraeus and the Saronic islands (Salamis,
# Hydra, Poros, Aegina, Spetses), Boeotia, Euboea, Argolida, Corinthia
ARVANITIKA_UNITS = ["EL305", "EL306", "EL307", "EL641", "EL642", "EL651", "EL652"]
SLAVIC = 50_000                    # low end of the usual 50,000-250,000
SLAVIC_BULGARIAN = {"EL514", "EL515", "EL526"}   # Drama, Kavala, Serres: Trudgill 2000
# Pomak villages, by LAU code prefix (municipality / municipal unit): Myki municipality in
# Xanthi; Kechros and Organi in Rodopi; Mikro Derio in Evros (Mega Derio sits in that LAU).
POMAK_LAU = {"EL512": ["0603"], "EL513": ["010203", "010204"], "EL511": ["03050206"]}
POMAK_FILL = 0.85                  # at most this share of a Pomak LAU's people drawn Pomak


def eurostat():
    d = json.load(open(EU_CTZ, encoding="utf-8"))
    dims, sizes = d["id"], d["size"]
    cats = {x: list(d["dimension"][x]["category"]["index"]) for x in dims}
    rows = []
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rec = {x: cats[x][idx[j]] for j, x in enumerate(dims)}
        g = rec["geo"]
        if g.startswith("EL") and len(g) == 5:
            rows.append((g, rec["citizen"], v))
    df = pd.DataFrame(rows, columns=["unit", "ctz", "n"])
    return df.pivot_table(index="unit", columns="ctz", values="n", aggfunc="sum").fillna(0)


def lang_of(iso):
    """The shared origin table (sources/origin_mix.py, 2026-10-05), Greek as a label.
    GREECE_OVERRIDES above is kept for the record: none had a figure behind it (India on
    Punjabi included), so all went back to the home mixes; Cyprus's home mix is Greek and
    Turkish."""
    from origin_mix import mix
    return {("Greek" if n == "indoeuropean.hellenic.greek" else n): s
            for n, s in mix(iso, "gr").items()}


def spread(total, weights):
    w = pd.Series(weights, dtype=float)
    w = w[w > 0]
    return total * w / w.sum()


def build():
    import geopandas as gpd
    import gr2021

    e = eurostat()
    units = sorted(e.index)
    if len(units) != 53:
        sys.exit(f"!! {len(units)} NUTS 3 units, expected 53")
    named = [c for c in e.columns if len(c) == 2 and c != "EL"]
    agg = ["EU_FOR", "NEU", "EUR_NEU", "AFR", "AME_N", "AME_X_N", "ASI", "OCE", "FOR", "NAT",
           "TOTAL", "RNC"]
    rest = [c for c in e.columns if c.endswith("_OTH")]
    chk = (e["NAT"] + e["FOR"] + e["STLS"] + e["UNK"] - e["TOTAL"]).abs().max()
    chk2 = (e[named].sum(axis=1) + e[rest].sum(axis=1) - e["FOR"]).abs().max()
    print(f"  census 2021: {e['TOTAL'].sum():,.0f} people, NAT {e['NAT'].sum():,.0f}, FOR "
          f"{e['FOR'].sum():,.0f}, STLS {e['STLS'].sum():,.0f}, UNK {e['UNK'].sum():,.0f}; "
          f"NAT+FOR+STLS+UNK vs TOTAL off by {chk:,.0f}; named+_OTH vs FOR off by {chk2:,.0f}")
    # Eurostat's census cells are not perfectly additive (rounding of small cells): 9 and 33
    # people in the worst unit. Greek takes TOTAL less UNK less everything drawn elsewhere.
    if chk > 50 or chk2 > 50:
        sys.exit("!! citizenship table does not add up")
    _ = agg

    t = pd.read_csv(T1951, comment="#")
    # Florina's printed row puts 4,303 under "Russian"; the national table's Slavic (41,017)
    # and Russian (3,815) only add up with them as Slavic (sources/gr.md §3).
    fl = t["nomos"] == "Florina"
    t.loc[fl, "slavic"] += t.loc[fl, "russian"]
    t.loc[fl, "russian"] = 0
    for col, nat in (("total", 7_632_801), ("turkish", 179_895), ("slavic", 41_017),
                     ("aromanian", 39_855), ("albanian", 22_736), ("romani", 7_429)):
        if abs(t[col].sum() - nat) > 1:   # Slavic comes to 41,018 with Florina's 4,303
            sys.exit(f"!! 1951 {col}: {t[col].sum():,} against the national {nat:,}")
    print(f"  1951 table 7b: 52 nomoi reconcile to the national row (Pomak "
          f"{t['pomak'].sum():,} against 18,671)")

    # 1951 nomos groups -> NUTS 3: Attica's two rows go over EL301-EL307 by Greek citizens
    nat = e["NAT"]
    attica = [u for u in units if u.startswith("EL30")]

    def by_1951(col, keep=lambda u: True):
        """Weights: the 1951 share speaking `col` in each unit's nomoi, times the unit's Greek
        citizens in 2021. A rate rather than a count, so that a unit that has emptied since
        1951 (Florina, Evrytania) or filled (Attica, Thessaloniki) is not drawn at its 1951
        size."""
        g = t.groupby("nuts3")[[col, "total"]].sum()
        out = {}
        for k, (v, tot51) in g.iterrows():
            for u in (attica if k == "EL30" else [k]):
                out[u] = v / tot51 * nat[u]
        return {u: v for u, v in out.items() if keep(u) and v > 0}

    rows, wrows = [], []
    used = {u: 0.0 for u in units}

    def add(u, lab, n, part):
        rows.append((u, lab, n, part))
        used[u] += n

    # ---- immigrant languages ------------------------------------------------------------------
    for u in units:
        r = 1.0 if u == ATHOS else RETENTION
        for c in named:
            n = e.at[u, c]
            if n <= 0:
                continue
            lg = lang_of(c)
            for lab, sh in (lg.items() if isinstance(lg, dict) else [(lg, 1.0)]):
                add(u, lab, n * sh * r, "immigrants by citizenship")
        oth = e.loc[u, rest].sum() + e.at[u, "STLS"]
        if oth > 0:
            add(u, "Other", oth * r, "immigrants, unnamed citizenship")
    imm = sum(n for _, _, n, p in rows if p.startswith("immigrants"))
    print(f"  immigrant languages: {imm:,.0f} of {e['FOR'].sum() + e['STLS'].sum():,.0f} foreign "
          f"or stateless residents ({RETENTION:.1%} retention)")

    # ---- minorities ---------------------------------------------------------------------------
    minority = {}

    def put(lab, d, part):
        for u, n in d.items():
            minority[(u, lab)] = minority.get((u, lab), 0.0) + n
            add(u, lab, n, part)

    thrace = by_1951("turkish_muslim", lambda u: u in POMAK)
    put("Turkish", spread(MUSLIM_MINORITY * TURKISH_SHARE, thrace).to_dict(),
        "Muslim minority of Thrace, Turkish speakers")
    put("Turkish", {"EL421": sum(DODECANESE_TURKISH.values())}, "Turks of Rhodes and Kos")
    put("Pomak", POMAK, "Muslim minority of Thrace, Pomaks")
    roma = {}
    for reg, n in ROMA_2017.items():
        us = [u for u in units if u.startswith(reg)]
        for u, v in spread(ROMA_2021 * n / sum(ROMA_2017.values()), nat[us]).items():
            roma[u] = v
    put("Romani", roma, "Roma mapping 2021")
    put("Aromanian", spread(AROMANIAN, by_1951("aromanian", lambda u: u != ATHOS)).to_dict(),
        "estimate, placed by 1951 mother tongue")
    # not by 1951: its Albanian answers run the wrong way (Attica 0.09%, Evros 2.9%; sources/
    # gr.md §3), so the units Euromosaic names, by Greek citizens
    put("Arvanitika", spread(ARVANITIKA, nat[ARVANITIKA_UNITS]).to_dict(),
        "estimate, placed on the Arvanite areas")
    sl = spread(SLAVIC, by_1951("slavic", lambda u: u[:4] in ("EL52", "EL53") or
                                u in ("EL514", "EL515")))
    put("Macedonian", sl[[u for u in sl.index if u not in SLAVIC_BULGARIAN]].to_dict(),
        "estimate, placed by 1951 mother tongue")
    put("Bulgarian", sl[[u for u in sl.index if u in SLAVIC_BULGARIAN]].to_dict(),
        "estimate, placed by 1951 mother tongue")
    mino = pd.Series(minority).groupby(level=0).sum()
    over = (mino / nat.reindex(mino.index)).sort_values(ascending=False)
    print(f"  minorities: {mino.sum():,.0f} Greek citizens; largest share of a unit's citizens "
          f"{over.index[0]} {over.iloc[0]:.1%}")
    if over.iloc[0] > 0.8:
        sys.exit("!! a unit's minorities exceed 80% of its citizens")

    # ---- Greek --------------------------------------------------------------------------------
    for u in units:
        left = e.at[u, "TOTAL"] - e.at[u, "UNK"] - used[u]
        add(u, "Greek", left, "everyone else")

    # ---- placement weights (Pomak villages, Rhodes and Kos) -----------------------------------
    lay = gpd.read_file(RD_LAU, ignore_geometry=True)
    lay["lau"] = lay["lau"].astype(str)
    for u, prefixes in POMAK_LAU.items():
        z = lay[lay["nuts3"] == u]
        inn = z["lau"].str.startswith(tuple(prefixes))
        n = POMAK[u]
        in_pop = z.loc[inn, "pop"].sum()
        a = min(n, POMAK_FILL * in_pop)
        out_pop = z.loc[~inn, "pop"].sum()
        for c, p in zip(z["lau"], z["pop"]):
            w = a * p / in_pop if c.startswith(tuple(prefixes)) else (n - a) * p / out_pop
            wrows.append((c, "Pomak", w))
        # Turkish in Thrace: every LAU but the Pomak villages
        for c, p in zip(z.loc[~inn, "lau"], z.loc[~inn, "pop"]):
            wrows.append((c, "Turkish", p))
        print(f"  Pomak {u}: {a:,.0f} of {n:,} on {inn.sum()} village LAUs ({in_pop:,.0f} people)")
    for mu, n in DODECANESE_TURKISH.items():
        z = lay[lay["lau"].str.startswith(mu)]
        if z.empty:
            sys.exit(f"!! no LAU under {mu}")
        for c, p in zip(z["lau"], z["pop"]):
            wrows.append((c, "Turkish", n * p / z["pop"].sum()))

    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "part"])
    df = df.groupby(["geo_id", "source_category", "part"], as_index=False)["count"].sum()
    if (df["count"] < -0.5).any():
        sys.exit(f"!! negative rows:\n{df[df['count'] < -0.5]}")
    tot = df.groupby("geo_id")["count"].sum()
    off = (tot - (e["TOTAL"] - e["UNK"])).abs().max()
    if off > 1:
        sys.exit(f"!! units do not sum to the census total less UNK: {off:,.0f}")
    # rows under half a person (the origin mixes' long tails) go onto Greek, so totals hold
    small = df["count"] <= 0.5
    tiny = df[small].groupby("geo_id")["count"].sum()
    df = df[~small].copy()
    rest = (df["source_category"] == "Greek") & (df["part"] == "everyone else")
    df.loc[rest, "count"] += df.loc[rest, "geo_id"].map(tiny).fillna(0.0)
    df["geo_level"] = "nuts3"
    df["tier"] = "derived"
    df["year"] = YEAR
    df["source_id"] = SOURCE_ID
    for lab in df["source_category"].unique():
        gr2021.resolve(lab)
    df = df[["geo_id", "geo_level", "source_category", "count", "tier", "part", "year",
             "source_id"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)

    w = pd.DataFrame(wrows, columns=["lau", "key", "weight"])
    w = w.groupby(["lau", "key"], as_index=False)["weight"].sum()
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    w.to_csv(WEIGHTS + ".tmp", index=False)
    os.replace(WEIGHTS + ".tmp", WEIGHTS)

    natl = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {len(df):,} rows, {df['geo_id'].nunique()} units, "
          f"{df['count'].sum():,.0f} people, {len(natl)} languages; {WEIGHTS}: {len(w):,} rows")
    for lab, n in natl.head(30).items():
        print(f"  {lab:22} {n:12,.0f}  {n / natl.sum():6.2%}")
    print("Greek share by unit, lowest:")
    g = df[df["source_category"] == "Greek"].groupby("geo_id")["count"].sum() / tot
    for u, s in g.sort_values().head(8).items():
        print(f"  {u}  {s:6.1%}")


if __name__ == "__main__":
    build()
