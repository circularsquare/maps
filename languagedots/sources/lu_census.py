"""Luxembourg: census of 8 November 2021, main language, by commune (STATEC).

Reads (or fetches) data/raw/lu/ and writes data/normalized/lu.csv. The record is sources/lu.md.

## What the census published, and where

The question: "Quelle est la langue dans laquelle vous pensez et que vous connaissez le mieux ?",
one answer, called the main language (langue principale). The form offered six boxes
(Luxembourgish, Portuguese, French, English, Italian, German) and a write-in.

- **National table**: STATEC RP2021 "1ers résultats" n°8, *Une diversité linguistique en forte
  hausse* (Fehlen, Gilles et al., 2023), Tableau 1 (the seven categories, 563,092 answers of
  643,941 residents) and Tableau 3 (52 write-in languages with over 100 speakers, 57,146 of the
  60,582 "other"). LUSTAT (the .Stat Suite) has no language dataflow at all: every DSD_CENSUS
  flow was listed on 2026-10-05 and none carries language.
- **By commune**: only as geoportail.lu WMS layers 2735-2740, each "la part de la population
  communale dont la langue principale est le <x> lors du recensement de la population du 8
  novembre 2021": Luxembourgish, French, German, Portuguese, English (one decimal), and
  "aucune des trois langues officielles" (allophones, unrounded). There is no 2021 Italian or
  "other" layer. GetFeatureInfo at one point inside each commune returns the share.

Shares are of the people who answered, not of all residents (publication p. 13: "par rapport au
nombre total d'habitants par commune ayant répondu à la question"), and the commune's number of
respondents is not published.

## How the counts are built

1. Per commune and language, share x the commune's census population (LUSTAT DF_B1625, 643,941).
   Summed over communes this overshoots each national count by the non-response (12.6%), so each
   language is scaled to its Tableau 1 figure. The commune distribution of each language is the
   census's; the per-language factor (printed) is the only adjustment. measured.
2. Italian + other = allophones - Portuguese - English, per commune (clipped at 0 for rounding).
   It is split into Italian and other by iterative proportional fitting: rows are the communes'
   Italian+other counts, columns the national Italian (20,021) and other (60,582), the seed for
   Italian is the commune's Italian citizens (geoportail layer 2610, same census) and for other
   the commune's Italian+other itself. The commune totals and the national totals are the
   census's; only the Italian/other split inside each commune is modelled.
3. The 60,582 "other" are split into Tableau 3's 52 languages and a 3,436 unnamed remainder at
   the national proportions, in every commune. modelled.

Witness for step 2: the 2011 census layers (1609-1615) carry Italian and other by commune with
counts, so the 2021 Italian placement can be compared with 2011's Italian speakers.

## Checks

- DF_B1625's communes sum to 643,941 and number 102, and the geoportail commune at each point
  carries the census's name for the code (two spellings differ, GP_SPELLING);
- Tableau 1 sums to 563,092 and Tableau 3 to 57,146, as printed;
- per commune, Lb + Fr + De + allophones = 100% to rounding;
- after scaling, every category's commune sum equals the national table.

Usage:
    python sources/lu_census.py --fetch     DF_B1625, the PDF, 102 GetFeatureInfo calls
    python sources/lu_census.py             normalise from data/raw/lu/
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lu")
OUT = os.path.join(ROOT, "data", "normalized", "lu.csv")
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
RD_UNITS = os.path.join(RD, "data", "geo", "lu", "lu_units.gpkg")   # read-only: 102 LAU 2021 communes

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

STATEC_B1625 = ("https://lustat.statec.lu/rest/data/LU1,DSD_CENSUS_GROUP7_10@DF_B1625,/all"
                "?dimensionAtObservation=AllDimensions")
STATEC_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0;labels=both"
PDF_URL = ("https://statistiques.public.lu/dam-assets/catalogue-publications/rp-2021/"
           "rp08-diversit-linguistique/rp08-03-02-fr.pdf")
CENSUS_TOTAL = 643_941
N_COMMUNES = 102

GEOPORTAIL = "https://wms.geoportail.lu/public_map_layers/service"
LUREF = 2169
# 2021 language layers, the 2021 Italian-citizens layer, and the 2011 language layers (witness)
LAYERS_2021 = ["2735", "2736", "2737", "2738", "2739", "2740", "2610"]
LAYERS_2011 = ["1609", "1610", "1611", "1612", "1613", "1614", "1615"]
GP_SPELLING = {"0702": "Préizerdaul", "1006": "Rosport-Mompach"}
# Two names that some geoportail layers spell another way than the others (same commune)
GP_ALIAS = {"Redange/Attert": "Redange-sur-Attert", "Lac de la Haute Sûre": "Lac de la Haute-Sûre"}

# Tableau 1, RP2021 n°8 p. 4 (answers; 643,941 residents, 80,849 no answer or "not of an age to speak")
TABLE1 = {"Luxembourgeois": 275_361, "Portugais": 86_598, "Français": 83_802, "Anglais": 20_316,
          "Italien": 20_021, "Allemand": 16_412, "Autre langue": 60_582}
TABLE1_TOTAL = 563_092
# Tableau 3, p. 6: the write-in languages with over 100 speakers, as printed
TABLE3 = {
    "Espagnol": 6473, "Arabe": 3904, "Néerlandais": 3661, "Russe": 3325, "Polonais": 3251,
    "Roumain": 3092, "Chinois": 2855, "Serbe": 2736, "Bosniaque": 2601, "Grec": 2485,
    "Monténégrin": 1721, "Créole du Cap-Vert": 1510, "Albanais": 1357, "Hongrois": 1283,
    "Créole": 1148, "Serbo-Croate": 1086, "Danois": 1059, "Turc": 1053, "Bulgare": 982,
    "Suédois": 978, "Tigrigna": 961, "Croate": 934, "Lithuanien": 793, "Slovaque": 780,
    "Tchèque": 718, "Finnois": 650, "Persan": 592, "Hindi": 343, "Ukrainien": 337,
    "Yougoslave": 331, "Thaïlandais": 326, "Estonien": 310, "Kurde": 307, "Japonais": 294,
    "Slovène": 262, "Macédonien": 261, "Islandais": 208, "Catalan": 207, "Vietnamien": 201,
    "Philippin": 188, "Afrikaans": 166, "Flamand": 158, "Tamoul": 151, "Farsi": 147,
    "Népalais": 145, "Arménien": 140, "Letton": 128, "Tagalog": 121, "Pular": 113,
    "Norvégien": 109, "Bengali": 104, "Coréen": 101,
}
TABLE3_TOTAL = 57_146
OTHER_REST = "Autre langue (moins de 100 locuteurs)"
SOURCE_ID = "lu_rp2021_langue_principale"


def _p(name):
    return os.path.join(RAW, name)


def _get(url, dest, accept=None, magic=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  already on disk:", os.path.basename(dest))
        return
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
        body = r.read()
    if magic and not body.startswith(magic):
        sys.exit(f"!! {url}: starts {body[:40]!r}, expected {magic!r}")
    with open(dest + ".tmp", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".tmp", dest)
    print(f"  {os.path.basename(dest)}: {len(body):,} bytes")


def _points():
    import geopandas as gpd
    g = gpd.read_file(RD_UNITS)
    if len(g) != N_COMMUNES:
        sys.exit(f"!! {RD_UNITS} has {len(g)} communes, expected {N_COMMUNES}")
    pts = g.geometry.representative_point().to_crs(LUREF)
    return [(u, n, p.x, p.y) for u, n, p in zip(g["unit"], g["name"], pts)]


def _geoportail(dest):
    """One GetFeatureInfo per commune at a point inside it (WMS 1.3.0 reads EPSG:2169 northing
    first; text/plain, not JSON, which would send every polygon)."""
    if os.path.exists(dest):
        print("  already on disk:", os.path.basename(dest))
        return
    layers = ",".join(LAYERS_2021 + LAYERS_2011)
    out = {}
    for unit, name, x, y in _points():
        q = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
             "LAYERS": layers, "QUERY_LAYERS": layers, "CRS": f"EPSG:{LUREF}",
             "BBOX": f"{y - 50},{x - 50},{y + 50},{x + 50}", "WIDTH": 101, "HEIGHT": 101,
             "I": 50, "J": 50, "INFO_FORMAT": "text/plain", "FEATURE_COUNT": 30}
        url = GEOPORTAIL + "?" + urllib.parse.urlencode(q)
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                     timeout=120).read().decode("utf-8")
        out[unit] = {"name": name, "x": x, "y": y, "text": txt}
        time.sleep(0.3)
    with open(dest + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    os.replace(dest + ".tmp", dest)
    print(f"  geoportail: {len(out)} communes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    print("STATEC census 2021 by commune (DF_B1625)…")
    _get(STATEC_B1625, _p("statec_b1625.csv"), accept=STATEC_ACCEPT, magic=b"STRUCTURE")
    print("RP2021 n°8 (the national tables)…")
    _get(PDF_URL, _p("rp08-03-02-fr.pdf"), magic=b"%PDF")
    print("geoportail.lu language shares per commune…")
    _geoportail(_p("geoportail_languages.json"))


# ---------------------------------------------------------------------------------------------

def _census():
    d = pd.read_csv(_p("statec_b1625.csv"))
    d = d[(d["SEX: Sex"] == "_T: Total") & (d["CITIZEN: Citizenship"].str.startswith("_T:"))].copy()
    geo = d["GEO: Geographic level"].str.split(": ", n=1, expand=True)
    d["code"], d["name"] = geo[0], geo[1]
    nat = d[d["code"] == "_T"]["OBS_VALUE"].sum()
    if int(nat) != CENSUS_TOTAL:
        sys.exit(f"!! census total {nat:,} is not {CENSUS_TOTAL:,}")
    com = d[d["code"].str.len() == 9].copy()
    com["unit"] = com["code"].str[-4:]
    com = com.groupby(["unit", "name"], as_index=False)["OBS_VALUE"].sum().set_index("unit")
    if len(com) != N_COMMUNES or int(com["OBS_VALUE"].sum()) != CENSUS_TOTAL:
        sys.exit(f"!! DF_B1625: {len(com)} communes summing to {com['OBS_VALUE'].sum():,}")
    print(f"  DF_B1625: {len(com)} communes, {CENSUS_TOTAL:,} people")
    return com.rename(columns={"OBS_VALUE": "pop"})


PAT_2021 = re.compile(r"Commune : (.+?)\s*\n\s*(Luxembourgeois|Français|Allemand|Portugais|Anglais)"
                      r" est la langue principale \(%\) : ([0-9.]+)\s*\n\s*\n")
PAT_ALLO = re.compile(r"Commune : (.+?)\s*\n\s*Aucune des trois langues officielles est la langue "
                      r"principale \(%\) : ([0-9.eE+-]+)")
PAT_IT = re.compile(r"Part des Italiens \(en %\) : ([0-9.]+)\s*\n\s*Commune : (.+?)\s*\n")
PAT_2011 = re.compile(r"Commune : (.+?)\s*\n\s*(\S+(?: langue)?) est la langue principale \(%\) : "
                      r"[0-9.]+\s*\n\s*\S+(?: langue)? est la langue principale \(personnes\) : "
                      r"([0-9]+)")


def _shares(com):
    d = json.load(open(_p("geoportail_languages.json"), encoding="utf-8"))
    if set(d) != set(com.index):
        sys.exit(f"!! geoportail communes vs census: {sorted(set(d) ^ set(com.index))}")
    rows, w11 = {}, {}
    wrong = []
    for unit, rec in d.items():
        t = rec["text"].replace("\r", "")
        names = set()
        got = {}
        for cname, lang, val in PAT_2021.findall(t):
            got[lang] = float(val)
            names.add(GP_ALIAS.get(cname.strip(), cname.strip()))
        m = PAT_ALLO.findall(t)
        if len(m) != 1:
            sys.exit(f"!! {unit}: {len(m)} allophone values")
        names.add(GP_ALIAS.get(m[0][0].strip(), m[0][0].strip()))
        got["allo"] = float(m[0][1])
        m = PAT_IT.findall(t)
        if len(m) != 1:
            sys.exit(f"!! {unit}: {len(m)} Italian-citizen values")
        got["it_ctz"] = float(m[0][0])
        names.add(GP_ALIAS.get(m[0][1].strip(), m[0][1].strip()))
        if len(got) != 7 or len(names) != 1:
            sys.exit(f"!! {unit} {rec['name']}: geoportail returned {sorted(got)} for {names}")
        gname = names.pop()
        if gname != com.loc[unit, "name"] and GP_SPELLING.get(unit) != gname:
            wrong.append((unit, com.loc[unit, "name"], gname))
        rows[unit] = got
        w11[unit] = {lang: int(n) for _c, lang, n in PAT_2011.findall(t)}
    if wrong:
        sys.exit(f"!! geoportail's commune is not the census's at {len(wrong)} codes: {wrong[:8]}")
    sh = pd.DataFrame(rows).T
    off = (sh["Luxembourgeois"] + sh["Français"] + sh["Allemand"] + sh["allo"] - 100).abs()
    print(f"  geoportail: {len(sh)} communes, names agree with the census; Lb+Fr+De+allophones "
          f"is 100% to within {off.max():.2f} points")
    if off.max() > 0.25:
        sys.exit("!! the shares do not add up")
    return sh, pd.DataFrame(w11).T.fillna(0)


def _ipf(rows, cols, seed, n=200):
    """Scale seed (communes x 2) so its row sums are `rows` and its column sums `cols`."""
    x = seed.copy()
    for _ in range(n):
        r = x.sum(axis=1)
        x = x.mul((rows / r.where(r > 0, 1)).where(r > 0, 0), axis=0)
        c = x.sum(axis=0)
        x = x.mul(cols / c, axis=1)
    return x


def normalise():
    if sum(TABLE1.values()) != TABLE1_TOTAL:
        sys.exit(f"!! Tableau 1 sums to {sum(TABLE1.values()):,}")
    if sum(TABLE3.values()) != TABLE3_TOTAL:
        sys.exit(f"!! Tableau 3 sums to {sum(TABLE3.values()):,}, printed {TABLE3_TOTAL:,}")
    com = _census()
    sh, w11 = _shares(com)
    pop = com["pop"].astype(float)

    out = {}
    print("  per-language factor, national table / sum(commune share x population):")
    for lang in ["Luxembourgeois", "Français", "Allemand", "Portugais", "Anglais"]:
        raw = sh[lang] / 100 * pop
        k = TABLE1[lang] / raw.sum()
        out[lang] = raw * k
        print(f"    {lang:<15} {k:.3f}")
    allo = sh["allo"] / 100 * pop
    print(f"    {'allophones':<15} {(TABLE1_TOTAL - sum(TABLE1[x] for x in ['Luxembourgeois', 'Français', 'Allemand'])) / allo.sum():.3f}")
    resid = ((sh["allo"] - sh["Portugais"] - sh["Anglais"]).clip(lower=0)) / 100 * pop
    n_it_oth = TABLE1["Italien"] + TABLE1["Autre langue"]
    k = n_it_oth / resid.sum()
    resid = resid * k
    print(f"    {'Italian+other':<15} {k:.3f}  (clipped in {int((sh['allo'] - sh['Portugais'] - sh['Anglais'] < 0).sum())} communes)")

    seed = pd.DataFrame({"Italien": sh["it_ctz"] / 100 * pop, "Autre langue": resid})
    fit = _ipf(resid, pd.Series({"Italien": TABLE1["Italien"], "Autre langue": TABLE1["Autre langue"]}), seed)
    err = (fit.sum(axis=1) - resid).abs().max()
    if err > 0.01:
        sys.exit(f"!! IPF did not converge: row error {err}")
    out["Italien"] = fit["Italien"]
    # witness: 2021 Italian placement against the 2011 census's Italian speakers
    it11 = w11.get("Italien")
    if it11 is not None:
        a = out["Italien"] / out["Italien"].sum()
        b = it11.reindex(a.index).fillna(0) / it11.sum()
        print(f"  Italian placement vs 2011 Italian speakers: r = {a.corr(b):.3f}, "
              f"share-weighted gap {0.5 * (a - b).abs().sum():.3f} (half the L1 distance)")
        print(f"  2011 layers: Italian {int(it11.sum()):,} speakers in {len(it11)} communes")

    oth = fit["Autre langue"]
    for lang, n in TABLE3.items():
        out[lang] = oth * n / TABLE1["Autre langue"]
    out[OTHER_REST] = oth * (TABLE1["Autre langue"] - TABLE3_TOTAL) / TABLE1["Autre langue"]

    measured = {"Luxembourgeois", "Français", "Allemand", "Portugais", "Anglais"}
    rows = []
    for lang, s in out.items():
        for unit, v in s.items():
            rows.append({"geo_id": unit, "geo_level": "commune", "geo_name": com.loc[unit, "name"],
                         "source_category": lang, "count": round(float(v), 3),
                         "tier": "measured" if lang in measured else "modelled",
                         "year": 2021, "source_id": SOURCE_ID})
    df = pd.DataFrame(rows)
    nat = df.groupby("source_category")["count"].sum()
    want = dict(TABLE1)
    want.pop("Autre langue")
    want.update(TABLE3)
    want[OTHER_REST] = TABLE1["Autre langue"] - TABLE3_TOTAL
    bad = {k: (nat[k], v) for k, v in want.items() if abs(nat[k] - v) > 0.5}
    if bad:
        sys.exit(f"!! national sums off: {bad}")
    if abs(df["count"].sum() - TABLE1_TOTAL) > 2:
        sys.exit(f"!! total {df['count'].sum():,.1f} is not {TABLE1_TOTAL:,}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".tmp", index=False, encoding="utf-8")
    os.replace(OUT + ".tmp", OUT)
    print(f"  wrote {OUT}: {len(df):,} rows, {df['geo_id'].nunique()} communes, "
          f"{df['source_category'].nunique()} categories, {df['count'].sum():,.0f} people; "
          f"every category's commune sum equals the national table")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    normalise()
