"""T-511: residents without a car, per cell, for the local mode choice (docs/demand.md 4.3).

    python pipeline/nocar.py <id> [<id> ...] | --all      patch these packs
    python pipeline/nocar.py <id> --dry                    the shares only, nothing written
    python pipeline/nocar.py --plan                        each planned country's share and source

Adds the per-cell array `nocar` (f32, people living in a household with no car) and a `nocar`
header block to a finished pack: first to its res-9 copy in data/work/res9, then the same array
summed into the merged pack in data/packs by merge_cells.py's own rule, after checking that rule
still reproduces every other per-cell array of the live pack. Both writes only append to the .bin
(.bin replaced before .json, each through a temporary file), so a reader never sees a half-written
pack. No other array changes; nothing needs a re-solve.

A cell's share of people without a car comes from the finest open source:

- **US**: ACS 2019-23 5-year B08201 (households by size by vehicles available) by tract: people in
  households with no vehicle over people in households (sizes 1, 2, 3 and 4+ at 4.6 people). A
  cell takes the tract of its nearest populated 2020 block (the pack's blocks CSV).
- **England**: Census 2021 TS045 (car or van availability, households) by LSOA 2021, households
  made people as below.
- **Wales, Scotland, Northern Ireland** (T-562): Census 2021 TS045 by LSOA 2021, Scotland's
  Census 2022 UV405 by data zone 2022, NISRA Census 2021 car or van availability by data zone
  2021; households made people as below.
- **Everywhere else**: cars per household `c` gives the households with no car by
  `h0 = exp(-1.30 c^1.33)`, fitted on English LSOAs and US tracts of 50 households or more
  together (the US alone gives 1.34 and 1.29, England 1.24 and 1.29; rmse 0.04 against 0.11 for
  the Poisson exp(-c)), households weighted. Japan: AIRIA
  private cars per household by prefecture (March 2025). Other countries (`national`), first
  found: passenger cars per 1,000 people from national registers times the mean household size
  (COUNTRIES); a DHS survey's households with a car from 2015 on; Eurostat's cars per 1,000 and
  household size; a DHS survey from 2010 on; else households with a car from GDP per head
  (FALLBACK), or the median of the country's income group.
- **Households to people**: a household with no car is smaller, 0.693 times the size of one with a
  car (ACS 2019-23 national: 1.74 against 2.51 people), so the people's share is
  `h0 k / (h0 k + 1 - h0)` with k = 0.693 (England 23.5% of households, 17.6% of people).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_pack import PACKS, ROOT, load_pack  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "t511")
WORK = os.path.join(ROOT, "data", "work")
RES9 = os.path.join(WORK, "res9")
CURVE = (1.30, 1.33)  # h0 = exp(-a c^b)
SIZE_RATIO = 0.693  # people per no-car household over people per car household (ACS 2019-23, US)
ACS_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b08201.dat"
TS045 = ("Census 2021 TS045 car or van availability by LSOA 2021 (Nomis NM_2063_1), "
         "households made people at a no-car household size ratio of 0.693")
# passenger cars per 1,000 people (national register, year) and people per household
COUNTRIES = {
    "CA": (587, 2.40, "Statistics Canada vehicle registrations 2023, light-duty 23.55M / 40.1M; census 2021 household size"),
    "FR": (577, 2.17, "Eurostat road_eqs_carhab 2024 (provisional); INSEE household size"),
    "DE": (590, 2.03, "Eurostat road_eqs_carhab 2024; Destatis household size 2023"),
    "KR": (421, 2.21, "MOLIT registrations end 2024, passenger cars 21.77M / 51.7M (index.go.kr 1257); census 2024 household size"),
    "CN": (251, 2.62, "Ministry of Public Security 2024, 353M cars (汽车) / 1,408M; census 2020 household size"),
    "IN": (35, 4.44, "MoRTH 2022 cars, jeeps and taxis 49.05M / 1,417M; NFHS-5 household size"),
    "BR": (298, 2.79, "Senatran end 2024, 63.3M automóveis / 212.6M; census 2022 household size"),
    "NG": (24, 4.70, "NBS road transport data 2018, 4.74M private vehicles / 198M; NDHS 2018 household size"),
    # T-562: places the GDP rule misjudges (city states) or cannot reach (no World Bank figures)
    "HK": (77, 2.7, "Transport Department licensed private cars June 2025, 580,797 / 7.53M; census 2021 household size"),
    "SG": (116, 3.09, "LTA private cars end 2024, 524,418 over 1,463,400 resident households of 3.09 (SingStat 2024)"),
    "TW": (353, 2.49, "Highway Bureau private cars May 2025, 8.255M / 23.4M; household registration 2024 household size"),
}
T562 = os.path.join(ROOT, "data", "raw", "t562")
DHS_URL = ("https://api.dhsprogram.com/rest/dhs/data?indicatorIds=HC_TRNS_H_CAR&breakdown=national&f=json&perpage=5000"
           "&returnFields=CountryName,DHS_CountryCode,SurveyYear,SurveyId,SurveyType,Value,IndicatorId")
DHS_COUNTRIES_URL = "https://api.dhsprogram.com/rest/dhs/countries?f=json&returnFields=DHS_CountryCode,ISO2_CountryCode,CountryName"
EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{}?format=SDMX-CSV&startPeriod=2015"
WB_GDP_URL = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?format=json&per_page=400&mrnev=1"
WB_COUNTRIES_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
# logit(households with a car) = a + b ln(GDP per head, PPP 2021 $); fitted on every country with a
# figure above (notes/t562_fit.py); a country with no GDP figure takes its income group's median
FALLBACK = (-15.10, 1.502)
INCOME_H0 = {"High income": 0.172, "Upper middle income": 0.63, "Lower middle income": 0.916, "Low income": 0.971}
WORLD_H0 = 0.803
NATIONS = {"TLL": "GB-WLS", "TLM": "GB-SCT", "TLN": "GB-NIR"}  # ITL1 of a planned row's units
TS045_WALES_URL = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_2063_1.data.csv?date=latest&geography=2092957700TYPE151"
                   "&measures=20100&select=geography_code,c2021_cars_5,obs_value")
LSOA_WALES_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
                  "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/FeatureServer/0/query"
                  "?where=LSOA21CD+LIKE+%27W%25%27&outFields=LSOA21CD&outSR=4326&f=geojson&resultOffset={}&resultRecordCount=2000")
UV405_URL = "https://www.scotlandscensus.gov.uk/media/zz8620251217_10_04_asdfsfgbsdtbbjutiui465in/Datazone2022.zip"
DZ_SCOTLAND_URL = ("https://maps.gov.scot/server/rest/services/ScotGov/StatisticalUnits/MapServer/10/query?where=1%3D1"
                   "&outFields=dzcode&outSR=4326&maxAllowableOffset=0.0001&f=geojson&resultOffset={}&resultRecordCount=1000")
DZ_NI_URL = ("https://services3.arcgis.com/HRuPlEcokYlz4mdz/ArcGIS/rest/services/NISRA_Other/FeatureServer/1/query?where=1%3D1"
             "&outFields=DZ2021_cd,No_cars_or_vans_available,F1_or_more_cars_or_vans_availab&outSR=4326"
             "&maxAllowableOffset=0.0001&f=geojson&resultOffset={}&resultRecordCount=2000")
# AIRIA, 令和7年3月末: private passenger cars (registered and kei) per household, prefectures 01-47
AIRIA_JP = [0.986, 1.209, 1.369, 1.234, 1.360, 1.617, 1.510, 1.508, 1.537, 1.558, 0.915, 0.921, 0.405, 0.664,
            1.497, 1.616, 1.450, 1.670, 1.494, 1.532, 1.506, 1.344, 1.209, 1.414, 1.318, 0.784, 0.611, 0.880,
            1.060, 1.221, 1.431, 1.384, 1.331, 1.086, 1.233, 1.337, 1.311, 1.129, 1.129, 1.032, 1.470, 1.099,
            1.283, 1.261, 1.271, 1.183, 1.286]
AIRIA = "AIRIA private passenger cars per household by prefecture, March 2025 (kenbetsu2025.pdf)"
PACK_COUNTRY = {"lon": "GB-ENG", "par": "FR", "ber": "DE", "seo": "KR", "sha": "CN", "mum": "IN", "sao": "BR",
                "lag": "NG", "tor": "CA"}


def households_no_car(c):
    a, b = CURVE
    return np.exp(-a * np.power(np.maximum(c, 0.0), b))


def people_share(h0):
    k = SIZE_RATIO
    return h0 * k / (h0 * k + 1.0 - h0)


def country_of(cid, header):
    if header.get("country") == "JP":
        return "JP"
    if cid in PACK_COUNTRY:
        return PACK_COUNTRY[cid]
    if os.path.exists(os.path.join(WORK, f"{cid}.blocks.csv.gz")):
        return "US"
    import plan_config
    row = next((r for r in plan_config.load()[0] if r["id"] == cid and r.get("status") == "planned"), None)
    if row:  # a planned pack (T-555): England's module covers English rows only
        return plan_country(row)
    sys.exit(f"{cid}: no country for the no-car share")


def plan_country(row):
    if row.get("module") == "england":
        return "GB-ENG"
    if row["iso2"] == "GB":
        return NATIONS.get(row["units"]["GBR"][0][:3], "GB")
    if len(row["iso2"]) != 2 or not row["iso2"].isalpha():  # Natural Earth's -99 (Somalia's two rows)
        return worldbank()[2][row["countries"][0]]
    return row["iso2"]


def _cached(name, url):
    path = os.path.join(T562, name)
    if not os.path.exists(path):
        import requests
        import build_city as bc
        os.makedirs(T562, exist_ok=True)
        r = requests.get(url, headers=bc.UA, timeout=600)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


def _cached_pages(name, url, page):
    """Every page of an ArcGIS geojson query, as one cached feature list."""
    path = os.path.join(T562, name)
    if not os.path.exists(path):
        import requests
        import build_city as bc
        os.makedirs(T562, exist_ok=True)
        feats = []
        while True:
            r = requests.get(url.format(len(feats)), headers=bc.UA, timeout=600)
            r.raise_for_status()
            got = r.json()["features"]
            feats += got
            if len(got) < page:
                break
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump(feats, f)
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_memo = {}


def _once(fn):
    def wrap():
        if fn.__name__ not in _memo:
            _memo[fn.__name__] = fn()
        return _memo[fn.__name__]
    return wrap


@_once
def dhs():
    """{ISO2: (share of households possessing a private car, survey year, survey id)}, latest survey."""
    with open(_cached("dhs_countries.json", DHS_COUNTRIES_URL), encoding="utf-8") as f:
        iso = {c["DHS_CountryCode"]: c["ISO2_CountryCode"] for c in json.load(f)["Data"]}
    with open(_cached("dhs_car_national.json", DHS_URL), encoding="utf-8") as f:
        rows = sorted(json.load(f)["Data"], key=lambda r: (r["SurveyYear"], r["SurveyId"]))
    return {iso[r["DHS_CountryCode"]]: (r["Value"] / 100.0, r["SurveyYear"], r["SurveyId"]) for r in rows
            if r["DHS_CountryCode"] in iso}


@_once
def eurostat():
    """({ISO2: (passenger cars per 1,000, year)}, {ISO2: (people per household, year)}), latest years."""
    out = []
    for code in ("road_eqs_carhab", "ilc_lvph01"):
        t = pd.read_csv(_cached(f"eurostat_{code}.csv", EUROSTAT_URL.format(code)), usecols=["geo", "TIME_PERIOD", "OBS_VALUE"])
        t = t.dropna().sort_values("TIME_PERIOD").groupby("geo").last()
        t = t[t.index.str.len() == 2].rename(index={"EL": "GR", "UK": "GB"})
        out.append({g: (float(v.OBS_VALUE), int(v.TIME_PERIOD)) for g, v in t.iterrows()})
    return tuple(out)


@_once
def worldbank():
    """({ISO2: (GDP per head PPP, year)}, {ISO2: income group}, {ISO3: ISO2})."""
    with open(_cached("worldbank_NY.GDP.PCAP.PP.KD.json", WB_GDP_URL), encoding="utf-8") as f:
        gdp = {r["country"]["id"]: (r["value"], int(r["date"])) for r in json.load(f)[1] if r["value"]}
    with open(_cached("worldbank_countries.json", WB_COUNTRIES_URL), encoding="utf-8") as f:
        cs = json.load(f)[1]
    return gdp, {c["iso2Code"]: c["incomeLevel"]["value"] for c in cs}, {c["id"]: c["iso2Code"] for c in cs}


def national(country):
    """(households with no car, source) for a country without small areas, by the first rule that has it."""
    curve = "households with no car by exp(-1.30 c^1.33), made people at 0.693"
    if country in COUNTRIES:
        v, hh, ref = COUNTRIES[country]
        c = v / 1000.0 * hh
        return households_no_car(c), (f"{country}: {v} passenger cars per 1,000 people x {hh} people per household = "
                                      f"{c:.2f} cars per household, {curve} ({ref})")
    survey = dhs().get(country)
    if survey and survey[1] >= 2015:
        return 1.0 - survey[0], f"{country}: {survey[0]:.1%} of households possess a private car (DHS {survey[2]}), made people at 0.693"
    cars, hh = eurostat()
    if country in cars and country in hh:
        (v, vy), (h, hy) = cars[country], hh[country]
        c = v / 1000.0 * h
        return households_no_car(c), (f"{country}: {v:.0f} passenger cars per 1,000 people (Eurostat road_eqs_carhab {vy}) x "
                                      f"{h} people per household (Eurostat ilc_lvph01 {hy}) = {c:.2f} cars per household, {curve}")
    if survey and survey[1] >= 2010:
        return 1.0 - survey[0], f"{country}: {survey[0]:.1%} of households possess a private car (DHS {survey[2]}), made people at 0.693"
    gdp, income, _ = worldbank()
    if country in gdp:
        (g, gy), (a, b) = gdp[country], FALLBACK
        h0 = 1.0 / (1.0 + np.exp(a + b * np.log(g)))
        return h0, (f"{country}: no register or survey; households with a car from GDP per head ({g:,.0f} PPP $, World Bank "
                    f"{gy}) by logit = {a} + {b} ln(GDP), fitted on the countries with a figure; made people at 0.693")
    group = income.get(country)
    if group in INCOME_H0:
        return INCOME_H0[group], f"{country}: no register, survey or GDP; the median of {group} countries with a figure, made people at 0.693"
    return WORLD_H0, f"{country}: no register, survey, GDP or income group; the median of every country with a figure, made people at 0.693"


def cell_lonlat(a):
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    return ll[:, 1], ll[:, 0]


def units_with_nearest(lat, lon, units):
    """Index of the unit holding each point; a point in none takes the nearest unit."""
    import shapely
    from shapely import STRtree
    import england
    idx, _ = england.cell_units(lat, lon, units)
    miss = np.nonzero(idx < 0)[0]
    if len(miss):
        tree = STRtree([g for _, g in units])
        idx[miss] = tree.nearest(shapely.points(lon[miss], lat[miss]))
    return idx, len(miss)


def us_tracts():
    """{tract GEOID: (people in no-vehicle households, people in households)}."""
    path = os.path.join(RAW, "acsdt5y2023-b08201.dat")
    if not os.path.exists(path):
        sys.exit(f"download {ACS_URL} to {path}")
    cols = ["GEO_ID"] + [f"B08201_E{i:03d}" for i in (7, 8, 13, 14, 19, 20, 25, 26)]
    t = pd.read_csv(path, sep="|", usecols=cols, dtype={"GEO_ID": str})
    t = t[t.GEO_ID.str.startswith("1400000US")]
    e = lambda i: t[f"B08201_E{i:03d}"].to_numpy(float)  # noqa: E731
    sizes = ((7, 1.0), (13, 2.0), (19, 3.0), (25, 4.6))
    no = sum(e(b + 1) * s for b, s in sizes)
    al = sum(e(b) * s for b, s in sizes)
    return pd.DataFrame({"geoid": t.GEO_ID.str[9:].to_numpy(), "no": no, "all": al})


def share_us(cid, lat, lon):
    from scipy.spatial import cKDTree
    b = pd.read_csv(os.path.join(WORK, f"{cid}.blocks.csv.gz"), dtype={"geoid": str}, usecols=["geoid", "pop", "lat", "lon"])
    b = b[b["pop"] > 0]
    t = us_tracts()
    tract = b.geoid.str[:11]
    by = t.set_index("geoid")
    # CT's 2020 tracts sit in 2022 planning regions in ACS 2023: state + tract code (T-042)
    ct = t[t.geoid.str[:2] == "09"].assign(k=lambda d: d.geoid.str[:2] + d.geoid.str[5:]).drop_duplicates("k").set_index("k")
    no = tract.map(by["no"])
    al = tract.map(by["all"])
    m = no.isna() & (tract.str[:2] == "09")
    k = tract[m].str[:2] + tract[m].str[5:]
    no[m], al[m] = k.map(ct["no"]), k.map(ct["all"])
    # a tract with no figure (renumbered, or no households): its county's
    county = t.assign(c=t.geoid.str[:5]).groupby("c")[["no", "all"]].sum()
    m = no.isna() | ~(al > 0)
    unmatched = float(b["pop"][m].sum())
    no[m], al[m] = tract[m].str[:5].map(county["no"]), tract[m].str[:5].map(county["all"])
    s = (no / al).to_numpy(float)
    s = np.where(np.isfinite(s), s, np.nan)
    mean = np.nansum(s * b["pop"]) / b["pop"][np.isfinite(s)].sum()
    s = np.where(np.isfinite(s), s, mean)
    c = np.cos(np.radians(np.mean(lat)))
    _, i = cKDTree(np.c_[b.lon.to_numpy() * c, b.lat.to_numpy()]).query(np.c_[lon * c, lat])
    src = (f"ACS 2019-23 5-year B08201, people in households with no vehicle by tract (2020 tracts; Connecticut by "
           f"state and tract code), a cell taking its nearest populated 2020 block's tract; {unmatched:,.0f} block "
           f"people in tracts with no figure take their county's")
    return s[i], src


def share_england(cid, lat, lon):
    path = os.path.join(ROOT, "data", "raw", "tier0", f"lsoa2021_bgc_{cid}.json")  # the build's (england.lsoas)
    with open(path, encoding="utf-8") as f:
        feats = json.load(f)
    from shapely import make_valid
    from shapely.geometry import shape
    units = [(ft["properties"]["LSOA21CD"], make_valid(shape(ft["geometry"]))) for ft in feats]
    t = pd.read_csv(os.path.join(RAW, "ts045_lsoa2021.csv")).set_index("GEOGRAPHY_CODE")
    h0 = (t.c0 / t["all"]).reindex([u for u, _ in units]).to_numpy(float)
    mean = float(t.c0.sum() / t["all"].sum())
    h0 = np.where(np.isfinite(h0), h0, mean)
    idx, miss = units_with_nearest(lat, lon, units)
    return people_share(h0[idx]), f"{TS045}; {miss} cells in no LSOA take the nearest"


def _units(feats, key):
    from shapely import make_valid
    from shapely.geometry import shape
    return [(ft["properties"][key], make_valid(shape(ft["geometry"]))) for ft in feats if ft.get("geometry")]


@_once
def wales():
    """([(LSOA 2021, polygon)], households with no car per unit, the nation's share, source), all of Wales."""
    t = pd.read_csv(_cached("ts045_lsoa2021_wales.csv", TS045_WALES_URL))
    t = t.pivot(index="GEOGRAPHY_CODE", columns="C2021_CARS_5", values="OBS_VALUE")
    units = _units(_cached_pages("lsoa2021_bgc_wales.json", LSOA_WALES_URL, 2000), "LSOA21CD")
    h0 = (t[1] / t[0]).reindex([u for u, _ in units]).to_numpy(float)
    total = t[1].sum() / t[0].sum()
    return units, np.where(np.isfinite(h0), h0, total), total, (
        "Census 2021 TS045 car or van availability by LSOA 2021 (Nomis NM_2063_1, Wales)")


@_once
def scotland():
    """The same for Scotland's data zones 2022."""
    path = os.path.join(T562, "scotland_uv405_dz2022.csv")
    if not os.path.exists(path):  # the zip (88 MB, every topic table) goes once UV405 is out
        import zipfile
        zpath = _cached("scotland_datazone2022.zip", UV405_URL)
        with zipfile.ZipFile(zpath) as z, open(path + ".part", "wb") as f:
            f.write(z.read("Table UV405 - Car or van availability.csv"))
        os.replace(path + ".part", path)
        os.remove(zpath)
    t = pd.read_csv(path, skiprows=4, header=None, names=["dz", "what", "n"], encoding="utf-8-sig")
    t = t[t.dz.str.startswith("S01", na=False)]
    t["n"] = pd.to_numeric(t.n, errors="coerce")
    alls = t[t.what == "All occupied households"].set_index("dz").n
    none = t[t.what.str.endswith("No cars or vans")].set_index("dz").n
    units = _units(_cached_pages("dz2022_scotland.json", DZ_SCOTLAND_URL, 1000), "dzcode")
    h0 = (none / alls).reindex([u for u, _ in units]).to_numpy(float)
    total = none.sum() / alls.sum()
    return units, np.where(np.isfinite(h0), h0, total), total, (
        "Scotland's Census 2022 UV405 car or van availability by data zone 2022 (NRS; boundaries maps.gov.scot)")


@_once
def northern_ireland():
    """The same for Northern Ireland's data zones 2021, but people, not households: the layer counts
    usual residents in households (1.88M in all, against NI's 0.77M households)."""
    feats = _cached_pages("dz2021_ni_cars.json", DZ_NI_URL, 2000)
    units = _units(feats, "DZ2021_cd")
    p = {ft["properties"]["DZ2021_cd"]: ft["properties"] for ft in feats}
    none = np.array([p[u]["No_cars_or_vans_available"] for u, _ in units], float)
    some = np.array([p[u]["F1_or_more_cars_or_vans_availab"] for u, _ in units], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = none / (none + some)
    total = none.sum() / (none.sum() + some.sum())
    return units, np.where(np.isfinite(share), share, total), total, (
        "NISRA Census 2021 car or van availability, people in households, by data zone 2021 (NISRA_Other CarAvailability)")


UK_NATIONS = {"GB-WLS": wales, "GB-SCT": scotland, "GB-NIR": northern_ireland}


def uk_people(nation, share):
    return share if nation == "GB-NIR" else people_share(share)


def share_uk_nation(nation, lat, lon):
    units, share, _, ref = UK_NATIONS[nation]()
    idx, miss = units_with_nearest(lat, lon, units)
    made = "" if nation == "GB-NIR" else ", households made people at 0.693"
    return uk_people(nation, share[idx]), f"{ref}{made}; {miss} cells in no unit take the nearest"


def share_japan(cid, lat, lon):
    import build_city as bc
    import japan
    prefs = bc.CITIES[cid]["prefectures"]
    ms = japan.municipalities(prefs)
    idx, miss = units_with_nearest(lat, lon, [(m[0], m[3]) for m in ms])
    pref = np.array([int(ms[i][0][:2]) for i in idx])
    c = np.array(AIRIA_JP)[pref - 1]
    return people_share(households_no_car(c)), f"{AIRIA}, households with no car by exp(-1.30 c^1.33), made people at 0.693"


def shares(cid, header, a):
    country = country_of(cid, header)
    lon, lat = cell_lonlat(a)
    if country == "US":
        s, src = share_us(cid, lat, lon)
    elif country == "GB-ENG":
        s, src = share_england(cid, lat, lon)
    elif country == "JP":
        s, src = share_japan(cid, lat, lon)
    elif country in UK_NATIONS:
        s, src = share_uk_nation(country, lat, lon)
    else:
        h0, src = national(country)
        s = np.full(len(lat), people_share(h0))
    return country, np.clip(s, 0.0, 1.0), src


def plan_report():
    """The share and source for every country of the plan's planned rows; nothing written."""
    import collections
    import plan_config
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [r for r in plan_config.load()[0] if r.get("status") == "planned"]
    by = collections.Counter(plan_country(r) for r in rows)
    kinds = collections.Counter()
    for country in sorted(by):
        if country == "GB-ENG":
            t = pd.read_csv(os.path.join(RAW, "ts045_lsoa2021.csv"))
            share, src = people_share(t.c0.sum() / t["all"].sum()), TS045
        elif country in UK_NATIONS:
            _, _, total, src = UK_NATIONS[country]()
            share = uk_people(country, total)
        else:
            h0, src = national(country)
            share = people_share(h0)
        rule = "register" if country in COUNTRIES else next(k for k in ("Census", "DHS", "Eurostat", "GDP", "median") if k in src)
        kinds[rule] += by[country]
        where = " (the nation; each pack by small area)" if rule == "Census" else ""
        print(f"{country} {by[country]:3d} rows: {share:.1%} of people without a car{where}; {src}", flush=True)
    print(f"{len(by)} countries, {sum(by.values())} planned rows; rows by rule: {dict(kinds)}")


def append_array(stem, header, name, values, block):
    """Append (or replace) a per-cell f32 array and set header[name] = block, .bin first."""
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    header = json.loads(json.dumps(header))
    old = [e for e in header["arrays"] if e["name"] == name]
    v = np.ascontiguousarray(values, dtype="<f4").tobytes()
    if old and old[0]["count"] * 4 == len(v):
        buf[old[0]["offset"]:old[0]["offset"] + len(v)] = v
    else:
        header["arrays"] = [e for e in header["arrays"] if e["name"] != name]
        buf += b"\0" * (-len(buf) % 8)
        header["arrays"].append({"name": name, "dtype": "f32", "per": "cell", "offset": len(buf), "count": len(values)})
        buf += v
    header[name] = block
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    return header


def run(cid, dry=False):
    import merge_cells as mc
    h9, a9 = load_pack(cid, RES9)
    live, la = load_pack(cid, PACKS)
    assert "merged" in live and h9["h3_res"] == 9
    if mc.written(cid) != mc.stamp(PACKS, cid):
        sys.exit(f"{cid}: the merged pack changed after merging; merge it again first")
    country, s, src = shares(cid, h9, a9)
    pop = a9["pop"].astype(np.float64)
    nocar = (pop * s).astype(np.float32)
    line = f"{cid}: {country}, {nocar.sum():,.0f} of {pop.sum():,.0f} people without a car ({nocar.sum() / pop.sum():.1%})"
    if dry:
        print(line + f"; cells {s.min():.2f}-{s.max():.2f}", flush=True)
        return
    block = {"task": "T-511", "what": "people living in a household with no car, per cell", "source": src,
             "share": round(float(nocar.sum() / pop.sum()), 4)}
    # the merged values by merge_cells' rule, checked against the live pack first
    cells, _ = mc.merge(dict(h9, arrays=h9["arrays"] + [{"name": "nocar", "dtype": "f32", "per": "cell", "count": h9["cells"]}]),
                        dict(a9, nocar=nocar))
    for spec in live["arrays"]:
        k = spec["name"]
        if spec.get("per") == "cell" and k != "nocar":
            assert np.array_equal(cells[k], la[k]), f"{cid}: merging again changes {k}; merge_cells.py {cid} first"
    append_array(os.path.join(RES9, cid), h9, "nocar", nocar, block)
    live = json.loads(json.dumps(live))
    live["merged"]["source"] = mc.stamp(RES9, cid)
    append_array(os.path.join(PACKS, cid), live, "nocar", cells["nocar"], block)
    with open(os.path.join(RES9, f"{cid}.merged.json"), "w") as f:
        json.dump(mc.stamp(PACKS, cid), f)
    print(line, flush=True)


if __name__ == "__main__":
    import merge_cells as mc
    args = sys.argv[1:]
    if "--plan" in args:
        plan_report()
        sys.exit()
    ids =mc.registered() if "--all" in args else [x for x in args if not x.startswith("--")]
    if not ids:
        sys.exit(__doc__)
    for cid in ids:
        run(cid, dry="--dry" in args)
