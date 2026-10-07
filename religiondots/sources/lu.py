"""Luxembourg: one national mix from two surveys of 2020-22, on the 2021 census's 102 communes.

Usage:
    python sources/lu.py --fetch   # ESS 1-2, STATEC census, geoportail shares, Eurostat, Pew, the PDFs
    python sources/lu.py           # build data/normalized/lu.csv

NO OFFICIAL SOURCE HAS COUNTED RELIGION IN LUXEMBOURG SINCE THE 1970 CENSUS. AHA's release says
the census has been barred from asking since 1979, and STATEC says it "n'a actuellement pas de
mandat" for it (Regards 03/23). What exists, all of it national, none of it below the country:

    ESS rounds 1-2 (2002-04)   3,187 adults, citizens and foreigners. `regionlu` has ONE value
                               (`Luxembourg`) in both rounds, so no split-half can run.
    EVS 1999, 2008             GESIS login. 2008's national shares are printed in CEPS/INSTEAD's
                               cahier 2011-02 (Borsenberger and Dickes), Tableau 1.
    EVS 2020/21                fielded for the University of Luxembourg (STUDIALUX) by TNS Ilres, late
                               2020 to early 2021; NOT in the EVS 2017 integrated release or the
                               joint EVS/WVS v5.0 (Luxembourg is not among its 92 countries). Its
                               figures are printed in STATEC's Regards 03/23 (Allegrezza, 2023).
    TNS Ilres for AHA, 2022    515 residents 16+, 9-18 March 2022, online panel and telephone; the
                               national table is in AHA's press release of 21 June 2022.

WHAT IS DRAWN: everyone in the census (643,941) at the mean of EVS 2020/21 and TNS Ilres 2022,
each normalised to 100 over the five categories both print (Catholic, Protestant, Muslim, other
religion, none). Every commune takes the same shares; the communes only place the dots. Both
surveys sample residents of every nationality, so no foreign half is needed for coverage.

THE FOREIGN-HALF METHOD WAS BUILT AND REJECTED (`_foreign_half_test`, printed and pinned). Census
2021 by commune (STATEC DF_B1625) with the Portuguese, French, Italian, Belgian and German shares
per commune (geoportail.lu layers 2608-2612, the same census), each nationality on Pew 2020's
origin composition, other EU and non-EU citizens on Eurostat cens_21ctz_r3's national mix. On
those compositions the foreigners alone are 9.4% Muslim, 4.43% of the country, against 1.30%
(EVS) and 2.90% (TNS) of everyone and Pew's own 1.83% for Luxembourg; a citizen mix solved as the
remainder would need negative Muslims, Protestants and other religions. ESS 2002-04 says why:
Luxembourg's French and Belgian residents answered 2.5% and 2.1% Muslim against Pew's 9.1% and
6.8% for France and Belgium, while on Christian and none Pew matches all five nationalities.

ESS 2002-04 IS A WITNESS ONLY, twenty years and about 20 points of belonging older (72% then,
48% and 59% in the late surveys). Its `Other Christian denomination` is 16% of Luxembourgers,
Portuguese and Italians alike, against 1.9% `autre religion chrétienne` in EVS 2008: an artefact
of the Luxembourg card, not a church. sources/lu.md.
"""

import io
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

import no as _no  # noqa: E402  ESS_API, _TAB, _save, _eurostat, PEW_ZIP

RAW = os.path.join(ROOT, "data", "raw", "lu")
OUT = os.path.join(ROOT, "data", "normalized", "lu.csv")
# The two-half design wrote this; the build stops if a stale copy is left on disk.
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "lu_foreign.csv")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326", "LAU_RG_01M_2021_4326.shp")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# --- ESS rounds 1-2, the witness ----------------------------------------------------------
# From search.seriesMetadata (spec §12 "ESS's `searchDatafiles` IS GONE"); Luxembourg answers in
# rounds 1 and 2 only (probed 2026-10-03: no LU response in the main file of rounds 3-11).
ESS_FILES = {
    1: ("bcc624a3-edbb-4df3-ab18-9a96702c92ae", 67),   # ESS1e06_7
    2: ("edee45f2-976b-4c8b-902d-b65dc003c92e", 59),   # ESS2e03_6
}
CTZSHIP_VAR = {1: "ctzship", 2: "ctzshipa"}
ESS_MAIN = ["ctzcntr", "rlgblg", "rlgdnm"]

# --- STATEC, the census by commune ----------------------------------------------------------
STATEC_B1625 = ("https://lustat.statec.lu/rest/data/LU1,DSD_CENSUS_GROUP7_10@DF_B1625,/all"
                "?dimensionAtObservation=AllDimensions")
STATEC_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0;labels=both"
CENSUS_TOTAL = 643_941          # DF_B1625 _T, 8 November 2021; asserted
N_COMMUNES = 102                # the census's communes, and GISCO LAU 2021's

# --- geoportail.lu: five nationalities' shares per commune, census 2021 -----------------------
# WMS layers whose abstract reads "la part des <nationalité> parmi la population par commune lors
# du recensement de la population du 8 novembre 2021"; GetFeatureInfo returns the commune, its
# canton and the share to one decimal. Read at one point inside each LAU 2021 commune.
#
# TWO THINGS ABOUT THIS SERVICE THAT COST AN HOUR. WMS 1.3.0 reads an EPSG:2169 BBOX NORTHING FIRST,
# and an easting-first box anywhere north of the capital returns an empty FeatureCollection with
# HTTP 200, not an error (the capital itself answers either way, because its easting and northing
# are both near 75-77 km). And INFO_FORMAT=application/json sends each commune's polygon for every
# layer, 1.2 MB a call; text/plain is 200 bytes and carries the same three fields.
GEOPORTAIL = "https://wms.geoportail.lu/public_map_layers/service"
GP_LAYERS = {"2608": "PT", "2609": "FR", "2610": "IT", "2611": "BE", "2612": "DE"}
GP_LABEL = {"Portugais": "PT", "Français": "FR", "Italiens": "IT", "Belges": "BE",
            "Allemands": "DE"}
LUREF = 2169
# The two communes geoportail spells differently from STATEC's table; every other name is identical.
GP_SPELLING = {"0702": "Préizerdaul", "1006": "Rosport-Mompach"}

# --- Eurostat and Pew -----------------------------------------------------------------------
EU_CTZ = "cens_21ctz_r3"
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU",
        "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
NAMED5 = ["PT", "FR", "IT", "BE", "DE"]

# --- the published figures, kept as files beside the numbers typed from them ------------------
PDFS = {
    "statec_regards_03_23.pdf":
        "https://statistiques.public.lu/dam-assets/catalogue-publications/regards/2023/regards-03-23.pdf",
    "aha_umfrage_2022.pdf":
        "https://www.aha.lu/images/Pressemitteilungen/2022-06-21-AHA-Umfrage.pdf",
    "ceps_cahier_2011_02.pdf":
        "https://liser.elsevierpure.com/ws/portalfiles/portal/19767299/cahier_n_2011_02.pdf",
}

CATHOLIC = "Catholic"
PROTESTANT = "Protestant"
MUSLIM = "Muslim"
OTHER = "Other religion"
NONE = "No religion"
CATS = [CATHOLIC, PROTESTANT, MUSLIM, OTHER, NONE]

# EVS 2020/21, STATEC Regards 03/23 p. 2: 48% belong (Graphique 4); of those, Catholics 85.3%,
# Christians "si on compte les réformés de tout obédience" 92%, Muslims 2.7%; the text then gives
# Catholics as 41% and Christians 44% of everyone, which these reproduce. Graphique 5's other bars
# (Juifs, Bouddhistes, Autres) are not printed as numbers and are taken together as the remainder.
EVS_2021_BELONG = 48.0
EVS_2021_OF_BELONGERS = {CATHOLIC: 85.3, PROTESTANT: 92.0 - 85.3, MUSLIM: 2.7}
# TNS Ilres for AHA, March 2022, press release p. 4, "Verteilung der Religionszugehörigkeit":
# belong 59 (katholisch 53, Muslime 3, Protestanten 2, andere Religion 3), none 41. The four parts
# sum to 61 by rounding and are scaled to 59.
TNS_2022 = {CATHOLIC: 53.0, MUSLIM: 3.0, PROTESTANT: 2.0, OTHER: 3.0}
TNS_2022_BELONG = 59.0

# Pew's nodes -> the five late-survey categories. Orthodox and every other Christian body that is
# neither Catholic nor Protestant is `other religion`: neither late survey names Orthodoxy, and
# EVS's `Autres` and AHA's `andere Religion` are where an Orthodox respondent can have answered.
def _cat_of(node):
    if node.startswith("christianity.catholic"):
        return CATHOLIC
    if node.startswith("christianity.protestant") or node.startswith("christianity.lutheran") \
            or node.startswith("christianity.reformed") or node.startswith("christianity.anglican"):
        return PROTESTANT
    if node == "islam" or node.startswith("islam.") or node == "alevism":
        return MUSLIM
    if node == "unaffiliated":
        return NONE
    return OTHER


def _p(name):
    return os.path.join(RAW, name)


# =======================================================================================
# fetch
# =======================================================================================

def _ess_fetch(rnd, bv, weight, dest):
    if os.path.exists(dest):
        return
    fid, ver = ESS_FILES[rnd]
    q = _no._TAB % (f' weightVariable:"{weight}",' if weight else "")
    d = _no._ess(q, {"id": fid, "v": ver, "bv": bv})
    hit = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
           if x["by"][0]["value"] == "LU"]
    if not hit:
        sys.exit(f"!! ESS round {rnd} has no LU response")
    _no._save(hit[0]["response"], dest)


def _get(url, dest, accept=None, magic=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  already on disk:", os.path.basename(dest))
        return
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600,
                                context=ctx) as r:
        body = r.read()
    if magic and not body.startswith(magic):
        sys.exit(f"!! {url}: starts {body[:40]!r}, expected {magic!r}")
    with open(dest + ".tmp", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".tmp", dest)
    print(f"  {os.path.basename(dest)}: {len(body):,} bytes")


def _geoportail(dest):
    """One GetFeatureInfo per LAU 2021 commune, at a point inside it, all five layers at once."""
    import time

    import geopandas as gpd

    if os.path.exists(dest):
        print("  already on disk:", os.path.basename(dest))
        return
    g = gpd.read_file(LAU, where="CNTR_CODE='LU'")
    if len(g) != N_COMMUNES:
        sys.exit(f"!! LAU 2021 has {len(g)} Luxembourg communes, expected {N_COMMUNES}")
    pts = g.geometry.representative_point().to_crs(LUREF)
    layers = ",".join(GP_LAYERS)
    out = {}
    for lau, name, p in zip(g["LAU_ID"], g["LAU_NAME"], pts):
        x, y = p.x, p.y
        q = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
             "LAYERS": layers, "QUERY_LAYERS": layers, "CRS": f"EPSG:{LUREF}",
             "BBOX": f"{y - 50},{x - 50},{y + 50},{x + 50}", "WIDTH": 101, "HEIGHT": 101,
             "I": 50, "J": 50, "INFO_FORMAT": "text/plain", "FEATURE_COUNT": 10}
        url = GEOPORTAIL + "?" + urllib.parse.urlencode(q)
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                     timeout=120).read().decode("utf-8")
        out[lau] = {"lau_name": name, "x": x, "y": y, "text": txt}
        time.sleep(0.3)
    _no._save(out, dest)
    print(f"  geoportail: {len(out)} communes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    print("ESS rounds 1-2 (the witness)…")
    for rnd in sorted(ESS_FILES):
        _ess_fetch(rnd, ESS_MAIN, None, _p(f"ess_r{rnd}_n.json"))
        _ess_fetch(rnd, ESS_MAIN, "pspwght", _p(f"ess_r{rnd}_w.json"))
        _ess_fetch(rnd, [CTZSHIP_VAR[rnd], "rlgblg", "rlgdnm"], None, _p(f"ess_r{rnd}_ctz.json"))
        _ess_fetch(rnd, ["regionlu"], None, _p(f"ess_r{rnd}_region.json"))
    print("STATEC census 2021 by commune and citizenship (DF_B1625)…")
    _get(STATEC_B1625, _p("statec_b1625.csv"), accept=STATEC_ACCEPT, magic=b"STRUCTURE")
    print("geoportail.lu nationality shares per commune…")
    _geoportail(_p("geoportail_nationalities_2021.json"))
    print("Eurostat census…")
    dest = _p("cens_21ctz_r3_lu.json")
    if not os.path.exists(dest):
        d = _no._eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T", geo=["LU000", "LU"])
        _no._save(d, dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")
    print("Pew…")
    _get(_no.PEW_ZIP, _p("pew.zip"), magic=b"PK")
    print("the published figures…")
    for name, url in PDFS.items():
        _get(url, _p(name), magic=b"%PDF")


# =======================================================================================
# read
# =======================================================================================

def _ess_table(name):
    d = json.load(open(_p(name), encoding="utf-8"))
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rows = []
    for cell in d["table"]:
        rec = {"count": float(cell["count"])}
        for i, n in enumerate(order):
            rec[n] = codes[n][cell["path"][i]]["label"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _census():
    """Per commune: NAT, EU_FOR, NEU, STLS, UNK, _T, with the code and name."""
    d = pd.read_csv(_p("statec_b1625.csv"))
    d = d[d["SEX: Sex"] == "_T: Total"].copy()
    geo = d["GEO: Geographic level"].str.split(": ", n=1, expand=True)
    d["code"], d["name"] = geo[0], geo[1]
    d["cit"] = d["CITIZEN: Citizenship"].str.split(":").str[0]
    p = d.pivot_table(index=["code", "name"], columns="cit", values="OBS_VALUE",
                      aggfunc="sum", fill_value=0).reset_index()
    nat = p[p["code"] == "_T"].iloc[0]
    if int(nat["_T"]) != CENSUS_TOTAL:
        sys.exit(f"!! census total {nat['_T']:,} is not {CENSUS_TOTAL:,}")
    com = p[p["code"].str.len() == 9].copy()
    com["lau"] = com["code"].str[-4:]
    if len(com) != N_COMMUNES:
        sys.exit(f"!! {len(com)} communes in DF_B1625")
    bad = com[com["FOR"] != com["EU_FOR"] + com["NEU"]]
    if len(bad):
        sys.exit(f"!! FOR is not EU_FOR + NEU in {list(bad['name'])}")
    bad = com[com["_T"] != com["NAT"] + com["FOR"] + com["STLS"] + com["UNK"]]
    if len(bad):
        sys.exit(f"!! _T is not NAT + FOR + STLS + UNK in {list(bad['name'])}")
    if int(com["_T"].sum()) != CENSUS_TOTAL:
        sys.exit("!! communes do not sum to the national total")
    print(f"  DF_B1625: {len(com)} communes, {CENSUS_TOTAL:,} people, Luxembourgers "
          f"{com['NAT'].sum():,} ({100 * com['NAT'].sum() / CENSUS_TOTAL:.2f}%), EU foreigners "
          f"{com['EU_FOR'].sum():,}, non-EU {com['NEU'].sum():,}, stateless {com['STLS'].sum():,}, "
          f"not stated {com['UNK'].sum():,}")
    return com.set_index("lau")


def _named_shares(com):
    """Per commune, the five nationalities' counts from geoportail's shares x the census total.

    Joined by LAU code, witnessed by name: the commune the point fell in (geoportail's own
    polygon) must carry the census's name for the code (STATEC's table), which neither key decides.
    """
    d = json.load(open(_p("geoportail_nationalities_2021.json"), encoding="utf-8"))
    if set(d) != set(com.index):
        sys.exit(f"!! geoportail communes vs census: {sorted(set(d) ^ set(com.index))}")
    rows = {}
    wrong = []
    import re
    pat = re.compile(r"Part des (\S+) \(en %\) : ([0-9.]+)\s+Commune : (.+?)\s*\n\s*Canton : (.+?)\s*\n")
    for lau, rec in d.items():
        got, gnames = {}, set()
        for label, val, cname, _canton in pat.findall(rec["text"] + "\n"):
            if label not in GP_LABEL:
                sys.exit(f"!! {lau}: unknown layer label {label!r}")
            got[GP_LABEL[label]] = float(val)
            gnames.add(cname.strip())
        if set(got) != set(NAMED5) or len(gnames) != 1:
            sys.exit(f"!! {lau} {rec['lau_name']}: geoportail returned {sorted(got)} for {gnames}")
        gname = gnames.pop()
        if gname != com.loc[lau, "name"] and GP_SPELLING.get(lau) != gname:
            wrong.append((lau, com.loc[lau, "name"], gname))
        rows[lau] = got
    if wrong:
        sys.exit(f"!! geoportail's commune is not the census's at {len(wrong)} codes: {wrong[:8]}")
    sh = pd.DataFrame(rows).T[NAMED5] / 100.0
    cnt = sh.mul(com["_T"], axis=0)
    over = cnt.sum(axis=1) - com["EU_FOR"]
    print(f"  geoportail: 102 communes, names agree with the census at every code; the five "
          f"nationalities exceed the census's EU foreigners in {int((over > 0).sum())} communes "
          f"(at most {over.max():.1f} people, the one-decimal rounding)")
    if over.max() > 0.0005 * com["_T"].max():
        sys.exit("!! five nationalities exceed EU foreigners by more than rounding")
    return cnt


def _eurostat_named():
    d = json.load(open(_p("cens_21ctz_r3_lu.json"), encoding="utf-8"))
    dims = d["id"]
    cats = [list(d["dimension"][x]["category"]["index"]) for x in dims]
    sizes = d["size"]
    rows = []
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rows.append([cats[j][idx[j]] for j in range(len(dims))] + [v])
    df = pd.DataFrame(rows, columns=dims + ["value"])
    df = df[df["geo"] == "LU000"]
    return df.groupby("citizen")["value"].sum()


def _pew():
    with zipfile.ZipFile(_p("pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    return pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")


def _composition(iso, pew):
    import origin_religion as origin
    pn = origin.PEW_BY_ISO.get(iso, "MISSING")
    if pn == "MISSING":
        return None
    row = origin.REGIONAL.get(iso) if pn is None else (
        {f: float(pew.loc[pn, f]) for f in origin.FAMILIES} if pn in pew.index else None)
    if row is None:
        return None
    return origin.composition(iso, row, "other.lu")


# =======================================================================================
# build
# =======================================================================================

def _late_target():
    """The five categories, % of residents: each survey normalised to 100, then the mean."""
    evs = {k: v * EVS_2021_BELONG / 100 for k, v in EVS_2021_OF_BELONGERS.items()}
    evs[OTHER] = EVS_2021_BELONG - sum(evs.values())
    evs[NONE] = 100 - EVS_2021_BELONG
    s = sum(TNS_2022.values())
    tns = {k: v * TNS_2022_BELONG / s for k, v in TNS_2022.items()}
    tns[NONE] = 100 - TNS_2022_BELONG
    mean = {k: (evs[k] + tns[k]) / 2 for k in CATS}
    print(f"\n  the national level, % of residents:")
    print(f"    {'':<16}{'EVS 2020/21':>12}{'TNS 2022':>10}{'mean':>8}")
    for k in CATS:
        print(f"    {k:<16}{evs[k]:12.2f}{tns[k]:10.2f}{mean[k]:8.2f}")
    if abs(evs[CATHOLIC] - 41) > 0.5 or abs(evs[CATHOLIC] + evs[PROTESTANT] - 44) > 0.5:
        sys.exit("!! EVS figures no longer reproduce STATEC's 41% Catholic and 44% Christian")
    for d in (evs, tns, mean):
        if abs(sum(d.values()) - 100) > 1e-9:
            sys.exit("!! a late survey does not sum to 100")
    return mean


def _ess_witness(pew):
    """Printed, not drawn: what ESS 2002-04 says about citizens, foreigners and Pew."""
    print("\nESS rounds 1-2, the witness (unweighted respondents)…")
    for rnd in sorted(ESS_FILES):
        reg = _ess_table(f"ess_r{rnd}_region.json")
        vals = sorted(set(reg.loc[reg["count"] > 0, "regionlu"]))
        print(f"  round {rnd}: regionlu takes {vals}")
        if vals != ["Luxembourg"]:
            sys.exit("!! regionlu now has more than one value; a split-half becomes possible")
    w = pd.concat([_ess_table(f"ess_r{r}_w.json") for r in ESS_FILES])
    n = pd.concat([_ess_table(f"ess_r{r}_n.json") for r in ESS_FILES])
    for tag, df in (("unweighted", n), ("pspwght", w)):
        tot = df["count"].sum()
        nonc = df.loc[df["ctzcntr"] == "No", "count"].sum()
        print(f"  {tag}: {tot:,.0f}, non-citizens {100 * nonc / tot:.1f}% (census 2001, STATEC DF_B1753: 36.9%)")
    for who in ("Yes", "No"):
        d = w[w["ctzcntr"] == who]
        ans = d[d["rlgblg"].isin(["Yes", "No"]) & ~d["rlgdnm"].isin(["Refusal", "No answer", "Don't know"])]
        t = ans["count"].sum()
        blg = ans.loc[ans["rlgblg"] == "Yes"].groupby("rlgdnm")["count"].sum()
        print(f"  {'citizens' if who == 'Yes' else 'non-citizens'} (pspwght): belong "
              f"{100 * blg.sum() / t:.1f}%; " + ", ".join(
                  f"{k} {100 * v / t:.1f}%" for k, v in blg.sort_values(ascending=False).items()))

    print("\n  Pew 2020's origin composition against ESS's residents of that nationality, %:")
    print(f"    {'':<10}{'n':>5}  {'Christian':>16}  {'none':>14}  {'Muslim':>13}")
    rows = []
    for rnd in sorted(ESS_FILES):
        t = _ess_table(f"ess_r{rnd}_ctz.json")
        t = t.rename(columns={CTZSHIP_VAR[rnd]: "ctz"})
        rows.append(t)
    t = pd.concat(rows)
    t = t[t["rlgblg"].isin(["Yes", "No"]) & ~t["rlgdnm"].isin(["Refusal", "No answer", "Don't know"])]
    names = {"PT": "Portugal", "FR": "France", "IT": "Italy", "BE": "Belgium", "DE": "Germany"}
    christian = {"Roman Catholic", "Protestant", "Eastern Orthodox", "Other Christian denomination"}
    for iso, nm in names.items():
        g = t[t["ctz"] == nm]
        tot = g["count"].sum()
        blg = g[g["rlgblg"] == "Yes"]
        c = blg.loc[blg["rlgdnm"].isin(christian), "count"].sum() / tot
        m = blg.loc[blg["rlgdnm"] == "Islam", "count"].sum() / tot
        u = g.loc[g["rlgblg"] == "No", "count"].sum() / tot
        p = pew.loc[nm]
        ps = sum(float(p[f]) for f in ("Christians", "Muslims", "Religiously_unaffiliated",
                                        "Buddhists", "Hindus", "Jews", "Other_religions"))
        print(f"    {nm:<10}{tot:5.0f}  {100 * c:6.1f} vs {100 * p['Christians'] / ps:5.1f}  "
              f"{100 * u:5.1f} vs {100 * p['Religiously_unaffiliated'] / ps:5.1f}  "
              f"{100 * m:4.1f} vs {100 * p['Muslims'] / ps:4.1f}")
    oc = t[(t["rlgdnm"] == "Other Christian denomination")]
    print(f"  `Other Christian denomination`: {oc['count'].sum():,.0f} of {t['count'].sum():,.0f} "
          f"respondents ({100 * oc['count'].sum() / t['count'].sum():.1f}%); EVS 2008 printed 1.9%")


def _foreign_half_test(com):
    """PRINTED AND ASSERTED, NOT DRAWN: the foreign-half method against Luxembourg's own totals.

    The construction es, gr, be and the Nordic countries use: each foreign citizen on Pew 2020's
    composition for their country (origin_religion.py), at the commune, from the census. Here it
    cannot be combined with the late surveys, because the foreign half ALONE holds more Muslims,
    Protestants and other religions than both surveys find in the whole population, so no citizen
    mix can make up the total (it would need negative Muslims). Pew's own 2020 estimate for
    Luxembourg, 1.8% Muslim, says the same thing: the origin compositions put 4.4% of the country
    in the foreign half's Muslims alone. ESS 2002-04 shows why: Luxembourg's French and Belgian
    residents answered 2.4% and 2.1% Muslim against Pew's 9.1% and 6.8% for France and Belgium.
    Returns the per-category shares of the foreign half, for the record.
    """
    named = _named_shares(com)
    eu = _eurostat_named()
    pew = _pew()

    print("\n  five nationalities, commune shares x census against cens_21ctz_r3 LU000:")
    for iso in NAMED5:
        a, b = named[iso].sum(), float(eu[iso])
        print(f"    {iso}  {a:>9,.0f}  {b:>9,.0f}  {a / b:.4f}")
        if abs(a / b - 1) > 0.005:
            sys.exit(f"!! {iso}: commune shares do not reproduce Eurostat's count")
    for k in ("NAT", "FOR"):
        if abs(float(eu[k]) - com[k].sum()) > 0.5:
            sys.exit(f"!! Eurostat {k} {eu[k]:,.0f} is not STATEC's {com[k].sum():,}")

    leaf = eu[eu.index.str.fullmatch(r"[A-Z]{2}") & (eu.index != "LU")]
    leaf = leaf[leaf > 0]
    comp, missing = {}, []
    for iso in leaf.index:
        c = _composition(iso, pew)
        if c is None:
            missing.append(iso)
        else:
            comp[iso] = c
    if missing:
        sys.exit(f"!! no composition for {missing}")
    print(f"  {len(leaf)} named citizenships cover {leaf.sum():,.0f} of {eu['FOR']:,.0f} foreigners")

    def mix(isos):
        tot = sum(float(leaf[i]) for i in isos)
        out = {}
        for i in isos:
            for node, s in comp[i].items():
                out[node] = out.get(node, 0.0) + float(leaf[i]) * s / tot
        return out
    group_mix = {iso: comp[iso] for iso in NAMED5}
    group_mix["OEU"] = mix([i for i in leaf.index if i in EU27 and i not in NAMED5])
    group_mix["NEU"] = mix([i for i in leaf.index if i not in EU27])
    grp = named.copy()
    grp["OEU"] = (com["EU_FOR"] - named.sum(axis=1)).clip(lower=0)
    grp["NEU"] = com["NEU"] + com["STLS"]
    by_cat = {}
    for g, n in grp.sum().items():
        for node, s in group_mix[g].items():
            k = _cat_of(node)
            by_cat[k] = by_cat.get(k, 0.0) + n * s
    foreign_n = float(grp.to_numpy().sum())
    people = float(com["_T"].sum())
    return {k: v / foreign_n for k, v in by_cat.items()}, foreign_n, people, pew


def build():
    import lu2021 as tax

    com = _census()
    target = _late_target()

    print("\nthe foreign-half method, tested against the national level (not drawn)…")
    fshare, foreign_n, people, pew = _foreign_half_test(com)
    n_cit = float(com["NAT"].sum())
    print(f"\n    {'':<16}{'all, target':>12}{'foreigners':>12}{'citizens would be':>19}")
    impossible = []
    for k in CATS:
        need = (target[k] / 100 * people - fshare.get(k, 0.0) * foreign_n) / n_cit
        print(f"    {k:<16}{target[k]:11.2f}%{100 * fshare.get(k, 0.0):11.2f}%{100 * need:18.2f}%")
        if need < 0:
            impossible.append(k)
    pew_lu = pew.loc["Luxembourg"]
    m_alone = fshare.get(MUSLIM, 0.0) * foreign_n / people
    print(f"    the foreign half's Muslims alone are {100 * m_alone:.2f}% of the country; Pew 2020's "
          f"own Luxembourg estimate is {pew_lu['Muslims']:.2f}%")
    # Pinned, so a change in Pew, the census or the late figures that makes the method workable
    # stops the build and gets looked at, rather than leaving this file arguing a stale case.
    if impossible != [PROTESTANT, MUSLIM, OTHER]:
        sys.exit(f"!! the foreign-half test changed: negative citizen shares now {impossible}")

    _ess_witness(pew)

    # ---- the drawn mix: everyone, every commune, the late surveys' mean
    names = com["name"]
    rows = []
    for lau in com.index:
        for k in CATS:
            rows.append((lau, names[lau], k, target[k] / 100 * float(com.loc[lau, "_T"]),
                         "mean of EVS 2020/21 (STATEC Regards 03/23) and TNS Ilres 2022 (AHA), "
                         "one national mix"))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count", "note"])
    out["geo_level"] = "commune"
    out["basis"] = "self_id"
    out["year"] = 2021
    out["source_id"] = "evs2021_tns2022_national"
    out = out[COLUMNS]
    unknown = sorted(set(out["source_category"]) - set(tax.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")
    if abs(out["count"].sum() - CENSUS_TOTAL) > 0.5:
        sys.exit("!! the drawn total is not the census")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    if os.path.exists(OUT_FOREIGN):
        sys.exit(f"!! {OUT_FOREIGN} exists from an earlier design; nothing reads it, remove it")
    print(f"\nwrote {OUT}  ({len(out):,} rows, {out['geo_id'].nunique()} communes, "
          f"{out['count'].sum():,.0f} people)")
    for k in CATS:
        print(f"  {target[k]:6.2f}%  {k} -> {tax.resolve(k)}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    build()


if __name__ == "__main__":
    main()
