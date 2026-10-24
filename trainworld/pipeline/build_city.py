"""Build a city pack (format 1, notes/T-004.md) and its boundary (notes/T-003.md).

    python pipeline/build_city.py nyc

Downloads are cached in data/raw/; outputs go to data/packs/<city>.json, .bin and
.boundary.geojson. Safe to re-run: cached files are reused, outputs are replaced whole.
The zone gravity is solved by the sim crate (cargo, `pack_gravity`), so Rust must be installed.

    python pipeline/build_city.py --boundary-only [ids]     (T-352: only the boundary files)
    python pipeline/build_city.py nyc --coarse-next         (T-1083: coarse_zones follows; one gravity pass)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # before numpy; Anita is using the machine

import datetime
import gzip
import io
import json
import shutil
import struct
import subprocess
import sys
import time
import zipfile

import h3
import numpy as np
import pandas as pd
import requests
import shapefile  # pyshp
from shapely.geometry import mapping, shape
from shapely import make_valid, union_all

import group_quarters
import regions
import spread
import water
import wfh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work")
PACKS = os.path.join(ROOT, "data", "packs")
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
R_EARTH = 6371008.8
FORMAT = 1

TIGER_BLOCKS = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{fips}_tabblock20.zip"
LODES_WAC = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/wac/{st}_wac_S000_JT00_{year}.csv.gz"
LODES_RAC = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/rac/{st}_rac_S000_JT00_{year}.csv.gz"
# 2020 county polygons: they match the 2020 block GEOIDs (Connecticut's old counties, which
# LODES 8 also uses; the 2022+ files have Connecticut's planning regions instead).
COUNTIES = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
# T-352: the DRAWN boundary (data/packs/<id>.boundary.geojson: outline, shade, reach test) is TIGER/Line
# counties, which include the water out to the state and county lines at sea (pipeline/tiger_counties.py).
# The cartographic file above, clipped to the shoreline, stays the land footprint the build itself uses
# (the water mask's extent, the points kept for schools, airports, parks and the rest).
BOUNDARY_NAME = "Census TIGER/Line 2020 counties (tl_2020_us_county, water included, simplified), counties in "
BOUNDARY_URL = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
# Checks only, not inputs.
POPEST = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv"
QCEW_TOTAL = "https://data.bls.gov/cew/data/api/{year}/a/industry/10.csv"

STATES = {"01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
          "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia",
          "20": "ks", "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn",
          "28": "ms", "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm",
          "36": "ny", "37": "nc", "38": "nd", "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri",
          "45": "sc", "46": "sd", "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa",
          "54": "wv", "55": "wi", "56": "wy", "72": "pr"}

# T-162: the work gravity's decay between zones in different states is multiplied by this
# (notes/T-162.md: fitted jointly with the decay on LODES OD for New York and Philadelphia). It is
# a per-pack value, shipped in the pack's gravity block: a pack from another country's pipeline
# sets its own from its own commuting data, and a pack that sets none gets 1 (no effect). A city
# config's own "border_factor" overrides it: Washington's 0.45 (T-181, Anita 2026-10-12; refitted
# at the T-198 curve), its own fit, because DC is a city counted as a state. T-198 rechecked 0.23 at
# the flatter curve: New York's own fit 0.228, Philadelphia's 0.207.
BORDER_FACTOR_US = 0.23
# The work gravity's distance curve, f(d) = d^-DECAY_POW_US exp(-d / DECAY_KM_US km), shipped in the
# pack's gravity block (T-198, from T-185's diagnosis: the flatter near end sends job-rich homes'
# commuters out as far as LODES does). The sim crate's `Params` default is the same pair; since T-277
# the game takes each pack's shipped curve at opening (`DemandApi::new`), so a pack with its own fitted
# curve (Toronto's "decay_pow"/"decay_km") opens from its shipped factors too.
DECAY_POW_US = 0.8
DECAY_KM_US = 30.0
# Japan's telework share of the workers not at home, outside the three big metros: MLIT 2024, 16.8%
# of employed, at 2 days in 5 (research/japan.md)
TELEWORK_JP = 0.168 * 0.4

CITIES = {
    "bal": {
        "origin": (-76.6122, 39.2904),
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-137.md",
        "counties": {'24003': 'Anne Arundel', '24005': 'Baltimore County', '24013': 'Carroll', '24025': 'Harford', '24027': 'Howard', '24035': "Queen Anne's", '24510': 'Baltimore city'},
    },
    "pvd": {
        "origin": (-71.4128, 41.824),
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-137.md",
        "counties": {'44001': 'Bristol RI', '44003': 'Kent', '44005': 'Newport', '44007': 'Providence', '44009': 'Washington RI', '25005': 'Bristol MA'},
    },
    "nhv": {
        "origin": (-72.9279, 41.3083),
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-137.md",
        "counties": {'09009': 'New Haven'},
    },
    # hfd (Hartford) is a world plan row since T-1113, built with the usa module
    "spf": {
        "origin": (-72.5898, 42.1015),
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-137.md",
        "counties": {'25011': 'Franklin MA', '25013': 'Hampden', '25015': 'Hampshire'},
    },
    "bos": {
        "origin": (-71.0589, 42.3601),  # central Boston
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-134-packs.md",
        # Broad Boston-Cambridge-Newton county metro extent, including southern NH.
        "counties": {
            "25009": "Essex MA", "25017": "Middlesex MA", "25021": "Norfolk",
            "25023": "Plymouth", "25025": "Suffolk MA",
            "33015": "Rockingham NH", "33017": "Strafford",
        },
    },
    "wdc": {
        "origin": (-77.0365, 38.8977),  # central Washington
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-134-packs.md",
        # Broad Washington-Arlington-Alexandria metro extent. Keep Calvert's
        # established DC commuter catchment rather than chasing recent MSA revisions.
        "border_factor": 0.45,  # its own LODES fit (0.454 at T-198's curve; 0.486 at T-162's), not the US 0.23
        "counties": {
            "11001": "District of Columbia",
            "24009": "Calvert", "24017": "Charles", "24021": "Frederick MD",
            "24031": "Montgomery MD", "24033": "Prince George's",
            "51013": "Arlington", "51043": "Clarke", "51047": "Culpeper",
            "51059": "Fairfax County", "51061": "Fauquier", "51107": "Loudoun",
            "51153": "Prince William", "51157": "Rappahannock", "51177": "Spotsylvania",
            "51179": "Stafford", "51187": "Warren", "51510": "Alexandria",
            "51600": "Fairfax city", "51610": "Falls Church", "51630": "Fredericksburg",
            "51683": "Manassas", "51685": "Manassas Park", "54037": "Jefferson WV",
        },
    },
    "phl": {
        "origin": (-75.1636, 39.9526),  # Philadelphia City Hall
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-115.md",
        # Philadelphia-Camden-Wilmington MSA, OMB July 2023: 11 counties.
        # Disjoint from nyc; Mercer/Trenton is in New York's pack since T-276.
        "counties": {
            "42017": "Bucks", "42029": "Chester", "42045": "Delaware PA",
            "42091": "Montgomery", "42101": "Philadelphia",
            "34005": "Burlington", "34007": "Camden", "34015": "Gloucester",
            "34033": "Salem", "10003": "New Castle", "24015": "Cecil",
        },
    },
    "nyc": {
        "origin": (-73.985, 40.758),  # Times Square / Midtown
        "lodes_years": (2023, 2022),
        # T-030: 25 m pixels; water under 2 pixels (~50 m) wide is dropped (creeks, ditches,
        # ponds a railway crosses on a culvert or a short bridge at grade)
        "water": {"cell_m": 25.0, "min_width_px": 2},
        # T-014, every case argued in notes/T-014.md
        "job_adjustments": {
            "moves": [
                # North Shore University Hospital campus, Manhasset: 41,521 jobs, about Northwell's
                # whole non-campus staff; keep 10,000 (NSUH ~7,000 staff plus the Feinstein
                # Institutes beside it), the rest over the population of the counties where
                # Northwell ran hospitals in 2023
                {"name": "Northwell at NSUH", "block": "360593018002016", "keep": 10000, "sector": "CNS16",
                 "to_counties": ["36059", "36103", "36081", "36061", "36085", "36119"]},
                # JetBlue's head office, 27-01 Queens Plaza North: ~1,300 staff there, ~10,300
                # transport jobs on the block; the rest are crew based at JFK (Terminal 5's block)
                {"name": "JetBlue crew at the LIC head office", "block": "360810033011005",
                 "sector": "CNS08", "keep_sector": 1300, "to_block": "360810716001015"},
            ],
            # home care agencies: aides work in clients' homes, the agency reports them at its
            # office. Health care jobs of such blocks go over their county's population.
            "home_care": {"min_jobs": 1000, "min_health": 0.8, "min_low_pay": 0.0, "max_high_pay": 0.35},
        },
        # notes/T-003.md: the 2023 New York-Newark-Jersey City MSA (22 counties) plus
        # Fairfield CT, Dutchess NY and Orange NY.
        "counties": {
            # MSA, New York
            "36005": "Bronx", "36047": "Kings", "36061": "New York", "36081": "Queens",
            "36085": "Richmond", "36059": "Nassau", "36103": "Suffolk", "36119": "Westchester",
            "36087": "Rockland", "36079": "Putnam",
            # MSA, New Jersey
            "34003": "Bergen", "34013": "Essex", "34017": "Hudson", "34019": "Hunterdon",
            "34023": "Middlesex", "34025": "Monmouth", "34027": "Morris", "34029": "Ocean",
            "34031": "Passaic", "34035": "Somerset", "34037": "Sussex", "34039": "Union",
            # added
            "09001": "Fairfield CT", "36027": "Dutchess", "36071": "Orange NY",
            # T-276 (Anita): Mercer (Trenton, Princeton), in the New York CSA, with twice the
            # LODES commuting to New York's counties (175k jobs both ways) as to Philadelphia's
            "34021": "Mercer",
        },
    },
    # T-277: the first pack outside the US, built by pipeline/canada.py from Statistics Canada's
    # 2021 census. "counties" are census divisions (4 digits): the Greater Toronto and Hamilton
    # Area. One province, so one region and no border factor (SPEC 4, "Borders outside the US").
    "tor": {
        "country": "ca",
        "origin": (-79.3832, 43.6534),  # Toronto City Hall
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-277.md",
        "border_factor": 1.0,
        # fitted on the 2021 census commuting flows between its 28 municipalities (notes/t277_fit.py:
        # 5.6% of commutes misplaced between municipality pairs, against 15.1% at the US curve)
        "decay_pow": 1.2, "decay_km": 20.0,
        "counties": {"3518": "Durham", "3519": "York", "3520": "Toronto", "3521": "Peel",
                     "3524": "Halton", "3525": "Hamilton"},
        "everyday": lambda city, lat, lon, jobs: __import__("canada").ct_sectors(city, lat, lon, jobs),
    },
    # T-288 stage a: London from tier 0 (global sources only, pipeline/tier0.py). The outline is
    # whole ITL3 units (ONS, January 2025): Greater London's 21, every unit touching it, West
    # Sussex (North East) for Gatwick and Medway (else a hole between Gravesham and Swale).
    # Population scaled to Census 2021 district totals (pipeline/england.py).
    "lon": {
        "tier0": True,
        "origin": (-0.1246, 51.5073),  # Charing Cross
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-288.md",
        "itl3": ["TLI33", "TLI34", "TLI35", "TLI36", "TLI41", "TLI42", "TLI43", "TLI44",
                 "TLI45", "TLI51", "TLI52", "TLI53", "TLI54", "TLI61", "TLI62", "TLI63", "TLI71", "TLI72",
                 "TLI73", "TLI74", "TLI75",
                 "TLJ26", "TLH27", "TLH32", "TLJ15", "TLH26", "TLJ25", "TLJ46", "TLH35", "TLH36", "TLJ13",
                 "TLJ43", "TLJ28", "TLJ41"],
        "boundary_file": lambda: __import__("england").itl3_boundary("lon", CITIES["lon"]["itl3"]),
        "admin_totals": lambda boundary: __import__("england").lad_totals(boundary),
        # T-360: England's small-area counts in place of the GHSL population and the jobs proxy
        "population": lambda city, boundary, sig: __import__("england").lsoa_population(city, boundary, sig),
        "jobs": lambda city, boundary, sig, w: __import__("england").bres_jobs(city, boundary, sig, w),
        "everyday": lambda city, lat, lon, jobs: __import__("england").bres_sectors(city, lat, lon, jobs),
        "jobs_url": "https://www.nomisweb.co.uk/datasets/newbres6pub",
        "schools": lambda boundary: __import__("england").gias_schools(boundary),
        "water_source": lambda bbox: __import__("osm_water").source(["europe/united-kingdom/england"])(bbox),
        "extract": "europe/united-kingdom/england",
        "iso2": "GB",
        # ONS Opinions and Lifestyle Survey (GB, 2022-23, "Who are the hybrid workers?", Nov 2024):
        # 13% work only from home, 28% hybrid; about half a hybrid worker's days at home
        "wfh": 0.27,
        "od_share": {"EGLL": 0.7},  # Heathrow's transfer passengers, judgement as Pearson's 0.75
        # fitted on the 2011 census flows (WU01EW) with this pack's own homes and jobs (T-360)
        "decay_pow": 1.85, "decay_km": 30.0,
        "region_name": "England",
    },
    "par": {
        "tier0": True,
        "origin": (2.3488, 48.8534),  # point zéro, parvis de Notre-Dame
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-289.md",
        "departements": ["75", "77", "78", "91", "92", "93", "94", "95"],  # Ile-de-France
        "boundary_file": lambda: __import__("france").departements_boundary("par", CITIES["par"]["departements"]),
        "population": lambda city, boundary, sig: __import__("france").commune_population(city, boundary, sig, CITIES["par"]["departements"]),
        "jobs": lambda city, boundary, sig, w: __import__("france").insee_jobs(city, boundary, sig, w, CITIES["par"]["departements"]),
        "jobs_url": "https://www.insee.fr/fr/statistiques/8582949",
        "water_source": lambda bbox: __import__("osm_water").source(["europe/france/ile-de-france"])(bbox),
        "extract": "europe/france/ile-de-france",
        "iso2": "FR",
        # Eurostat LFS lfsa_ehomp, France 2025, employed 15-64: 11.1% usually work from home, 24.4%
        # sometimes; judged at about 0.6 and 0.2 of their days (T-289)
        "wfh": 0.12,
        "od_share": {"LFPG": 0.75},  # Charles de Gaulle's transfer passengers, judgement as Pearson's
        # fitted on INSEE's 2022 commune flows with this pack's own homes and jobs (notes/t289_curve.py)
        "decay_pow": 1.3, "decay_km": 20.0,
        "region_name": "Ile-de-France",
    },
    "sha": {
        "tier0": True,
        "origin": (121.4737, 31.2304),  # People's Square
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-291.md",
        # the 16 districts, whole, Chongming included; their 2020 census totals in china.SHANGHAI
        "boundary_file": lambda: __import__("china").boundary_file("sha"),
        "population": lambda city, boundary, sig: __import__("china").census_population(city, boundary, sig),
        "jobs": lambda city, boundary, sig, w: __import__("china").census_jobs(city, boundary, sig, w),
        "jobs_url": "https://tjj.sh.gov.cn/cmsres/b3/b36d6d16841e4ac0a64a2e6664c69833/85fdf28606c6847035976a586d8d028d.pdf",
        # the Shanghai extract only: Jiangsu and Zhejiang lie outside the outline, dead space (T-291)
        "water_source": lambda bbox: __import__("osm_water").source(["asia/china/shanghai"])(bbox),
        "extract": "asia/china/shanghai",
        "iso2": "CN",
        # mean commute 10.2 km a trip, Shanghai's 7th comprehensive transport survey (2024-2025),
        # taken as straight-line: CAUPD-Baidu's 2024 home-work figure is 9.8 km (notes/T-291.md)
        "curve": {"mean_km": 10.2},
        "od_share": {"ZSPD": 0.85},  # Pudong's transfer passengers, judgement (notes/T-291.md)
        # CAAC 2024 airport bulletin via the press (notes/T-291.md): passengers / 2
        "boardings": {"ZSPD": (76789000 / 2, "2024, CAAC via press"), "ZSSS": (47972000 / 2, "2024, CAAC via press")},
        "region_name": "Shanghai",
        "latin_names": True,  # places.py: name:en, then name:zh-Latn-pinyin, then name (T-291)
    },
    "mum": {
        "tier0": True,
        "origin": (72.8428, 19.0192),  # Dadar station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-292.md",
        # Mumbai City and Suburban districts and 15 talukas of Thane, Palghar and Raigad, whole (india.MUMBAI)
        "boundary_file": lambda: __import__("india").boundary_file("mum"),
        "population": lambda city, boundary, sig: __import__("india").census_population(city, boundary, sig),
        # informal and home-based work: half the proxy by population (T-292)
        "job_blend": {"nres": 0.2, "osm": 0.3, "pop": 0.5},
        # B-28's 'no travel' share of other workers in the four districts (T-292)
        "wfh": 0.201,
        "water_source": lambda bbox: __import__("osm_water").source(["asia/india/western-zone"])(bbox),
        "extract": "asia/india/western-zone",
        "iso2": "IN",
        # both parameters fitted to Census 2011 B-28 distance bands, taken as road distance (T-292)
        "curve": {"bands": lambda: __import__("india").b28_bands("mum"), "road_factor": 1.3},
        # MIAL's calendar 2025 total via the press (notes/T-292.md): passengers / 2
        "boardings": {"VABB": (55500000 / 2, "2025, MIAL via press")},
        "region_name": "Mumbai",
        "latin_names": True,
    },
    "sao": {
        "tier0": True,
        "origin": (-46.6339, -23.5503),  # Praca da Se
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-458.md",
        # the 39 municipalities of the Regiao Metropolitana de Sao Paulo, whole (brazil.RMSP)
        "boundary_file": lambda: __import__("brazil").boundary_file("sao"),
        "population": lambda city, boundary, sig: __import__("brazil").census_population(city, boundary, sig),
        "jobs": lambda city, boundary, sig, w: __import__("brazil").survey_jobs(city, boundary, sig, w),
        "jobs_url": "https://transparencia.metrosp.com.br/dataset/pesquisa-origem-e-destino",
        # OD 2017: workers whose first job is at home (T-458)
        "wfh": 0.091,
        "water_source": lambda bbox: __import__("osm_water").source(["south-america/brazil/sudeste"])(bbox),
        "extract": "south-america/brazil/sudeste",
        "iso2": "BR",
        # both parameters fitted to OD 2017 workers' home-to-job straight-line distances (T-458)
        "curve": {"bands": lambda: __import__("brazil").survey_bands("sao"), "road_factor": 1.0},
        # GRU Airport's 2025 total via CAPA (notes/T-458.md): passengers / 2; a hub, so O&D by judgement
        "boardings": {"SBGR": (47188000 / 2, "2025, GRU Airport via CAPA")},
        "od_share": {"SBGR": 0.75},
        "region_name": "São Paulo",  # place names stay OSM's Portuguese `name`
    },
    # T-457: the Seoul Capital Area, Seoul, Incheon (less Ongjin's islands) and Gyeonggi whole;
    # South Korea's inputs from pipeline/korea.py
    "seo": {
        "tier0": True,
        "origin": (126.9780, 37.5665),  # Seoul City Hall
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-457.md",
        "boundary_file": lambda: __import__("korea").boundary_file("seo"),
        "population": lambda city, boundary, sig: __import__("korea").registered_population(city, boundary, sig),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/south-korea"])(bbox),
        "extract": "asia/south-korea",
        "iso2": "KR",
        # Statistics Korea with SK Telecom (2023-12): capital-area commuters travel 20.4 km on average,
        # taken as one-way route distance over a road factor of 1.3 (notes/T-457.md)
        "curve": {"mean_km": 15.7},
        "od_share": {"RKSI": 0.85},  # Incheon's transfer passengers, judgement
        "region_name": "Seoul Capital Area",
        "latin_names": True,
    },
    # T-459: Lagos, a data-poor tier 0 test from global sources only (notes/T-459.md)
    "lag": {
        "tier0": True,
        "origin": (3.3786, 6.5158),  # Yaba
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-459.md",
        # Lagos State's 20 LGAs and Ado-Odo/Ota, Ifo and Obafemi Owode (Ogun), whole (nigeria.UNITS)
        "boundary_file": lambda: __import__("nigeria").boundary_file("lag"),
        # mostly informal work: half the proxy by population, as Mumbai's (T-292)
        "job_blend": {"nres": 0.2, "osm": 0.3, "pop": 0.5},
        "water_source": lambda bbox: __import__("osm_water").source(["africa/nigeria"])(bbox),
        "extract": "africa/nigeria",
        "iso2": "NG",
        # no commute flows or distance table: tier0.borrowed_curve (T-314)
        "curve": "borrowed",
        # FAAN's 2024 harmonised report via Daily Trust: 3,006,102 international and 4,134,211
        # domestic passengers at Murtala Muhammed; passengers / 2
        "boardings": {"DNMM": (7140313 / 2, "2024, FAAN via Daily Trust")},
        "region_name": "Lagos",
    },
    "ber": {
        "tier0": True,
        "origin": (13.3777, 52.5163),  # Brandenburger Tor
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-290.md",
        # Berlin, Potsdam, the eight Landkreise touching Berlin, Brandenburg an der Havel (else a hole
        # between Havelland and Potsdam-Mittelmark) and Frankfurt (Oder), whole (T-290)
        "kreise": ["11000", "12054", "12069", "12063", "12065", "12060", "12064", "12067", "12061", "12072",
                   "12051", "12053"],
        "boundary_file": lambda: __import__("germany").kreise_boundary("ber", CITIES["ber"]["kreise"]),
        "population": lambda city, boundary, sig: __import__("germany").zensus_population(city, boundary, sig, CITIES["ber"]["kreise"]),
        "jobs": lambda city, boundary, sig, w: __import__("germany").pendler_jobs(city, boundary, sig, w, CITIES["ber"]["kreise"]),
        "jobs_url": "https://pendleratlas.statistikportal.de/",
        # Geofabrik's Brandenburg extract includes Berlin; the Berlin extract would duplicate it (T-290)
        "water_source": lambda bbox: __import__("osm_water").source(["europe/germany/brandenburg"])(bbox),
        "extract": "europe/germany/brandenburg",
        "iso2": "DE",
        # Eurostat LFS lfsa_ehomp, Germany 2025, employed 15-64: 13.0% usually work from home, 11.7%
        # sometimes; judged at about 0.6 and 0.2 of their days, as Paris's (T-290)
        "wfh": 0.10,
        # fitted on the Pendlerrechnung's 2024 published flows between Gemeinden with this pack's own
        # homes and jobs (notes/t290_curve.py); Paris's d^-1.3 exp(-d/20) scores the same
        "decay_pow": 1.4, "decay_km": 28.0,
        "region_name": "Berlin-Brandenburg",
    },
    # T-420 stage 1 and T-426: Greater Tokyo, five prefectures whole, Tokyo's Izu and Ogasawara islands
    # left out (research/japan.md); Japan's census inputs from pipeline/japan.py
    "tyo": {
        "tier0": True,
        "origin": (139.7671, 35.6812),  # Tokyo Station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-420.md",
        "prefectures": ["08", "11", "12", "13", "14"],  # Ibaraki, Saitama, Chiba, Tokyo, Kanagawa
        "neighbours": ["07", "09", "10", "19", "22"],  # land beyond the outline, for meshes it cuts
        "boundary_file": lambda: __import__("japan").boundary_file("tyo", CITIES["tyo"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["tyo"]["prefectures"], CITIES["tyo"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["tyo"]["prefectures"], CITIES["tyo"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["tyo"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["tyo"]["prefectures"], CITIES["tyo"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/kanto"])(bbox),
        "extract": "asia/japan/kanto",
        "iso2": "JP",
        # at home by municipality of work (2020 census table 4, T-465); of the rest, MLIT's 2024
        # telework survey has 37.5% of Greater Tokyo's employed teleworking, judged at 2 days in 5
        # (research/japan.md)
        "telework": 0.15,
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["tyo"]["prefectures"]),
        # fitted on the 2020 census municipality flows with this pack's own homes and jobs
        # (notes/t420_curve.py); flat from power 1.85 (London's) to 2.3, and only the steep end keeps
        # the census's 33.5% working in their own municipality
        "decay_pow": 2.2, "decay_km": 80.0,
        "od_share": {"RJAA": 0.8},  # Narita's transfer passengers, judgement
        "region_name": "Greater Tokyo",
        "latin_names": True,
        "kana_romaji": True,  # places.py: romaji from the kana reading, else the kanji's dictionary reading, where OSM has no Latin name
    },
    # T-431: Keihanshin, four prefectures whole (notes/T-431.md)
    "osa": {
        "tier0": True,
        "origin": (135.4959, 34.7025),  # Osaka Station, Umeda
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-431.md",
        "prefectures": ["26", "27", "28", "29"],  # Kyoto, Osaka, Hyogo, Nara
        "neighbours": ["18", "24", "25", "30", "31", "33", "36"],  # land beyond the outline, for meshes it cuts
        "boundary_file": lambda: __import__("japan").boundary_file("osa", CITIES["osa"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["osa"]["prefectures"], CITIES["osa"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["osa"]["prefectures"], CITIES["osa"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["osa"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["osa"]["prefectures"], CITIES["osa"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/kansai", "asia/japan/chubu"])(bbox),
        "extract": "asia/japan/kansai",
        "iso2": "JP",
        # at home by municipality of work (table 4, T-465); MLIT 2024 telework, Kinki 24.4% of
        # employed, 2 days in 5 as Tokyo
        "telework": 0.098,
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["osa"]["prefectures"]),
        # Tokyo's curve: ties the best fit on the 2020 census flows with this pack's margins (17.7%
        # misplaced, notes/t420_curve.py) and keeps the census's 41% in their own municipality
        "decay_pow": 2.2, "decay_km": 80.0,
        "region_name": "Keihanshin",
        "latin_names": True,
        "kana_romaji": True,
    },
    # T-431: Chukyo, three prefectures whole; Mie is in Geofabrik's Kansai, so features from both
    # extracts (osm_features.read keeps an object in both once)
    "ngo": {
        "tier0": True,
        "origin": (136.8815, 35.1709),  # Nagoya Station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-431.md",
        "prefectures": ["21", "23", "24"],  # Gifu, Aichi, Mie
        "neighbours": ["16", "17", "18", "20", "22", "25", "26", "29", "30"],  # land beyond the outline, for meshes it cuts
        "boundary_file": lambda: __import__("japan").boundary_file("ngo", CITIES["ngo"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["ngo"]["prefectures"], CITIES["ngo"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["ngo"]["prefectures"], CITIES["ngo"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["ngo"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["ngo"]["prefectures"], CITIES["ngo"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/chubu", "asia/japan/kansai"])(bbox),
        "extract": ["asia/japan/chubu", "asia/japan/kansai"],
        "iso2": "JP",
        # at home by municipality of work (table 4, T-465); MLIT 2024 telework, Chukyo 19.6% of
        # employed, 2 days in 5 as Tokyo
        "telework": 0.078,
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["ngo"]["prefectures"]),
        # Tokyo's curve: ties the best fit on the 2020 census flows with this pack's margins (12.6%
        # misplaced, notes/t420_curve.py) and keeps the census's 51% in their own municipality
        "decay_pow": 2.2, "decay_km": 80.0,
        "region_name": "Chukyo",
        "latin_names": True,
        "kana_romaji": True,
    },
    # T-432: Japan's regional metros, whole prefectures (Sapporo: Ishikari subprefecture), Tokyo's
    # inputs; telework is MLIT's 2024 share outside the three big metros (TELEWORK_JP); curves in
    # notes/T-432.md
    "fuk": {
        "tier0": True,
        "origin": (130.4207, 33.5897),  # Hakata Station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-432.md",
        "prefectures": ["40"],  # Fukuoka
        "neighbours": ["35", "41", "43", "44"],  # Yamaguchi over the Kanmon strait, Saga, Kumamoto, Oita
        "boundary_file": lambda: __import__("japan").boundary_file("fuk", CITIES["fuk"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["fuk"]["prefectures"], CITIES["fuk"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["fuk"]["prefectures"], CITIES["fuk"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["fuk"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["fuk"]["prefectures"], CITIES["fuk"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/kyushu", "asia/japan/chugoku"])(bbox),
        "extract": "asia/japan/kyushu",
        "iso2": "JP",
        "telework": TELEWORK_JP,  # census at home 8.6%
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["fuk"]["prefectures"]),
        "decay_pow": 2.0, "decay_km": 32.0,  # fitted, notes/t420_curve.py
        "region_name": "Fukuoka",
        "latin_names": True,
        "kana_romaji": True,
    },
    "spk": {
        "tier0": True,
        "origin": (141.3508, 43.0687),  # Sapporo Station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-432.md",
        # japan.SUBPREFECTURES; Otaru (01203) sends 15.2% of its resident workers into Ishikari (2020
        # census table 4), so it joins
        "prefectures": ["01-ishikari", "01-01203"],
        "neighbours": ["01-shiribeshi", "01-sorachi", "01-iburi", "01-rumoi"],
        "boundary_file": lambda: __import__("japan").boundary_file("spk", CITIES["spk"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["spk"]["prefectures"], CITIES["spk"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["spk"]["prefectures"], CITIES["spk"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["spk"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["spk"]["prefectures"], CITIES["spk"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/hokkaido"])(bbox),
        "extract": "asia/japan/hokkaido",
        "iso2": "JP",
        "telework": TELEWORK_JP,  # census at home 6.8%
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["spk"]["prefectures"]),
        "decay_pow": 2.2, "decay_km": 80.0,  # Tokyo's: ties the best fit (notes/t420_curve.py) and keeps the own-municipality share
        "region_name": "Sapporo",
        "latin_names": True,
        "kana_romaji": True,
    },
    "sdj": {
        "tier0": True,
        "origin": (140.8824, 38.2601),  # Sendai Station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-432.md",
        "prefectures": ["04"],  # Miyagi
        "neighbours": ["03", "05", "06", "07"],  # Iwate, Akita, Yamagata, Fukushima
        "boundary_file": lambda: __import__("japan").boundary_file("sdj", CITIES["sdj"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["sdj"]["prefectures"], CITIES["sdj"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["sdj"]["prefectures"], CITIES["sdj"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["sdj"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["sdj"]["prefectures"], CITIES["sdj"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/tohoku"])(bbox),
        "extract": "asia/japan/tohoku",
        "iso2": "JP",
        "telework": TELEWORK_JP,  # census at home 9.5%
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["sdj"]["prefectures"]),
        "decay_pow": 1.8, "decay_km": 50.0,  # fitted, notes/t420_curve.py
        "region_name": "Sendai",
        "latin_names": True,
        "kana_romaji": True,
    },
    "hij": {
        "tier0": True,
        "origin": (132.4757, 34.3975),  # Hiroshima Station
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "notes/T-432.md",
        "prefectures": ["34"],  # Hiroshima
        "neighbours": ["31", "32", "33", "35", "38"],  # Tottori, Shimane, Okayama, Yamaguchi, Ehime over the straits
        "boundary_file": lambda: __import__("japan").boundary_file("hij", CITIES["hij"]["prefectures"]),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, CITIES["hij"]["prefectures"], CITIES["hij"]["neighbours"]),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, CITIES["hij"]["prefectures"], CITIES["hij"]["neighbours"]),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, CITIES["hij"]["prefectures"]),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, CITIES["hij"]["prefectures"], CITIES["hij"]["neighbours"]),
        "water_source": lambda bbox: __import__("osm_water").source(["asia/japan/chugoku", "asia/japan/shikoku"])(bbox),
        "extract": "asia/japan/chugoku",
        "iso2": "JP",
        "telework": TELEWORK_JP,  # census at home 8.2%
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, CITIES["hij"]["prefectures"]),
        "decay_pow": 1.9, "decay_km": 32.0,  # fitted, notes/t420_curve.py
        "region_name": "Hiroshima",
        "latin_names": True,
        "kana_romaji": True,
    },
    # T-276: a non-city zone (docs/zones.md 2): one county between two city zones. Same sources and
    # cells as a city; `kind` "zone" in the registry. Windham, New London and Litchfield went into
    # Hartford's plan row (T-1113).
    "wor": {
        "kind": "zone", "name": "Worcester County",
        "origin": (-71.8023, 42.2626),  # Worcester city hall
        "lodes_years": (2023, 2022),
        "water": {"cell_m": 25.0, "min_width_px": 2},
        "boundary_notes": "docs/zones.md",
        "counties": {"25027": "Worcester MA"},
    },
}


def _japan_zone(zid, z):
    """T-420 stage 3: a Japanese prefecture zone (or merged Hokkaido subprefectures) from
    pipeline/japan_zones.json, Japan's inputs as the metro packs'; the curve is the zone's own fit
    (notes/t420s3_curve.py) on the census flows."""
    units, nb = z["units"], z["neighbours"]
    return {
        "tier0": True, "kind": "zone", "name": z["name"],
        "origin": tuple(z["origin"]),
        "water": {"cell_m": z["water_cell_m"], "min_width_px": 2},
        "boundary_notes": "notes/T-420.md",
        "prefectures": units, "neighbours": nb,
        "boundary_file": lambda: __import__("japan").boundary_file(zid, units, far_islets=False),
        "population": lambda city, boundary, sig: __import__("japan").mesh_population(city, boundary, sig, units, nb),
        "jobs": lambda city, boundary, sig, w: __import__("japan").census_jobs(city, boundary, sig, w, units, nb),
        "everyday": lambda city, lat, lon, jobs: __import__("japan").mesh_sectors(city, lat, lon, jobs, units),
        "jobs_url": "https://www.e-stat.go.jp/gis/statmap-search?type=1",
        "schools": lambda boundary: __import__("japan").p29_schools(boundary, units, nb),
        "water_source": lambda bbox: __import__("osm_water").source(z["water_extracts"])(bbox),
        "extract": z["extract"],
        "iso2": "JP",
        "telework": TELEWORK_JP,
        "at_home": lambda lat, lon: __import__("japan").at_home_share(lat, lon, units),
        "decay_pow": z.get("decay_pow", 2.2), "decay_km": z.get("decay_km", 80.0),
        "region_name": z["name"],
        "utc_offset": 9,
        "latin_names": True,
        "kana_romaji": True,
    }


_JZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "japan_zones.json")
if os.path.exists(_JZ):
    with open(_JZ, encoding="utf-8") as _f:
        for _zid, _z in json.load(_f)["zones"].items():
            CITIES[_zid] = _japan_zone(_zid, _z)

# T-537: the world plan's planned cities, configs generated from their rows (pipeline/plan_config.py)
import plan_config  # noqa: E402
for _cid, _cfg in plan_config.planned().items():
    CITIES.setdefault(_cid, _cfg)


class Cities(dict):
    """CITIES; a planned city's country keys join its config on its first lookup (T-555)."""

    def __getitem__(self, cid):
        return plan_config.with_country_keys(dict.__getitem__(self, cid))

    def get(self, cid, default=None):
        return self[cid] if cid in self else default


CITIES = Cities(CITIES)


# ---------------------------------------------------------------- downloads

def fetch(url, name=None):
    """Download url into data/raw once; return the local path."""
    path = os.path.join(RAW, name or url.rsplit("/", 1)[1])
    if os.path.exists(path):
        return path
    print(f"  downloading {url}", flush=True)
    tmp = f"{path}.{os.getpid()}.part"  # builds at once may fetch the same file (T-1065)
    with requests.get(url, headers=UA, timeout=300, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return settle(tmp, path)


def settle(tmp, path):
    """tmp in place as path; if another process got there first and Windows will not replace a file
    it has open, theirs stays."""
    try:
        os.replace(tmp, path)
    except OSError:
        if not os.path.exists(path):
            raise
        os.remove(tmp)
    return path


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP Range requests, so zipfile can pull one member."""

    def __init__(self, url):
        self.url = url
        h = requests.head(url, headers=UA, timeout=60, allow_redirects=True)
        h.raise_for_status()
        if h.headers.get("Accept-Ranges") != "bytes":
            raise OSError("no range support")
        self.size = int(h.headers["Content-Length"])
        self.pos = 0
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        hdr = dict(UA, Range=f"bytes={self.pos}-{self.pos + n - 1}")
        r = requests.get(self.url, headers=hdr, timeout=300)
        if r.status_code != 206:
            raise OSError(f"range request got {r.status_code}")
        self.pos += len(r.content)
        self.fetched += len(r.content)
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_zip_member(url, suffix):
    """Fetch only the member ending in `suffix` from a remote zip (cached in data/raw)."""
    base = url.rsplit("/", 1)[1][: -len(".zip")]
    path = os.path.join(RAW, base + suffix)
    if os.path.exists(path):
        return path
    print(f"  fetching {base}{suffix} out of {url}", flush=True)
    tmp = f"{path}.{os.getpid()}.part"
    try:
        remote = HttpRangeFile(url)
        src = io.BufferedReader(remote, buffer_size=1 << 22)
        with zipfile.ZipFile(src) as z:
            member = next(m for m in z.namelist() if m.endswith(suffix))
            with z.open(member) as zin, open(tmp, "wb") as out:
                while True:
                    chunk = zin.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
        print(f"    {remote.fetched / 1e6:.1f} MB transferred of a {remote.size / 1e6:.0f} MB zip")
    except OSError as e:  # no range support: take the whole zip
        print(f"    range read failed ({e}); downloading the whole zip")
        with zipfile.ZipFile(fetch(url)) as z:
            member = next(m for m in z.namelist() if m.endswith(suffix))
            with z.open(member) as zin, open(tmp, "wb") as out:
                out.write(zin.read())
    return settle(tmp, path)


# ---------------------------------------------------------------- readers

def read_dbf(path, fields):
    """Read named fields of a dBASE III file into numpy arrays of bytes (fixed width)."""
    with open(path, "rb") as f:
        head = f.read(32)
        nrec, hlen, rlen = struct.unpack("<IHH", head[4:12])
        desc = f.read(hlen - 32)
        f.seek(hlen)
        data = f.read(nrec * rlen)
    names, formats, offsets = [], [], []
    off = 1  # deletion flag
    for i in range(0, len(desc) - 1, 32):
        d = desc[i : i + 32]
        if d[0] == 0x0D:
            break
        name = d[:11].split(b"\0")[0].decode()
        length = d[16]
        names.append(name)
        formats.append(f"S{length}")
        offsets.append(off)
        off += length
    assert off == rlen, (off, rlen)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": rlen})
    rec = np.frombuffer(data, dtype=dt, count=nrec)
    return {k: rec[k] for k in fields}


def blocks_for_state(fips, county_set):
    """The state's 2020 blocks in the county set. `pop` is the census population less the people
    in institutional group quarters (prisons, nursing homes; T-234, pipeline/group_quarters.py),
    with `pop_census` and `gq_inst` beside it."""
    dbf = fetch_zip_member(TIGER_BLOCKS.format(fips=fips), ".dbf")
    d = read_dbf(dbf, ["GEOID20", "POP20", "INTPTLAT20", "INTPTLON20"])
    geoid = np.char.strip(d["GEOID20"]).astype("U15")
    keep = np.isin(geoid.astype("U5"), list(county_set))  # astype to U5 truncates: state+county
    df = pd.DataFrame({
        "geoid": geoid[keep],
        "pop": np.char.strip(d["POP20"][keep]).astype(np.int64),
        "lat": np.char.strip(d["INTPTLAT20"][keep]).astype(np.float64),
        "lon": np.char.strip(d["INTPTLON20"][keep]).astype(np.float64),
    })
    return group_quarters.remove(df, {c for c in county_set if c[:2] == fips}, fetch), len(geoid)


def jobs_for_state(st, years, county_set):
    for y in years:
        url = LODES_WAC.format(st=st, year=y)
        cached = os.path.exists(os.path.join(RAW, url.rsplit("/", 1)[1]))
        if cached or requests.head(url, headers=UA, timeout=60).status_code == 200:
            # sector and earnings columns are for the T-014 adjustments
            w = pd.read_csv(fetch(url), usecols=["w_geocode", "C000", "CE01", "CE03"] + wfh.CNS,
                            dtype={"w_geocode": str})
            w = w[w["w_geocode"].str[:5].isin(county_set)]
            return w.rename(columns={"w_geocode": "geoid"}).assign(jobs=w["C000"].to_numpy()), y, url
    raise RuntimeError(f"no LODES WAC for {st} in {years}")


def residents_for_state(st, year, county_set):
    """T-054: LODES RAC S000 JT00, jobs by the block their holder lives in (C000): the home end
    of the same jobs WAC counts at work."""
    url = LODES_RAC.format(st=st, year=year)
    r = pd.read_csv(fetch(url), usecols=["h_geocode", "C000"], dtype={"h_geocode": str})
    r = r[r["h_geocode"].str[:5].isin(county_set)]
    return r.rename(columns={"h_geocode": "geoid", "C000": "rac"}), url


# ---------------------------------------------------------------- build

def build_boundary(city, cfg):
    """Write data/packs/<city>.boundary.geojson, the TIGER/Line counties dissolved (T-352), and
    return the shoreline footprint (cartographic counties, dissolved) that the rest of the build
    filters by: so a pack's cells, water mask and points do not depend on the drawn water."""
    if cfg.get("tier0"):  # T-287: global sources only, pipeline/tier0.py
        import tier0
        return tier0.build_boundary(city, cfg)
    if cfg.get("country") == "ca":  # T-277: census subdivisions, pipeline/canada.py
        import canada
        return canada.build_boundary(city, cfg)
    import tiger_counties
    tiger = tiger_counties.counties(fetch)
    absent = [c for c in cfg["counties"] if c not in tiger]
    assert not absent, f"TIGER counties missing {absent}"
    drawn = union_all([tiger[c] for c in cfg["counties"]])
    path = fetch(COUNTIES)
    r = shapefile.Reader(path)
    fields = [f[0] for f in r.fields[1:]]
    gi = fields.index("GEOID")
    geoms = [make_valid(shape(sr.shape.__geo_interface__)) for sr in r.iterShapeRecords()
             if sr.record[gi] in cfg["counties"]]
    found = len(geoms)
    assert found == len(cfg["counties"]), f"found {found} of {len(cfg['counties'])} counties"
    merged = union_all(geoms)

    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(c[0], 6), round(c[1], 6)]
        return [rnd(x) for x in c]

    geom = mapping(drawn)
    geom = {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"city": city, "counties": sorted(cfg["counties"]),
                       "source": "Census TIGER/Line 2020 counties tl_2020_us_county, dissolved, "
                                 f"simplified {tiger_counties.TOLERANCE_DEG} degrees (T-352)"},
        "geometry": geom,
    }]}
    out = os.path.join(PACKS, f"{city}.boundary.geojson")
    with open(out + ".part", "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    os.replace(out + ".part", out)
    print(f"boundary: {geom['type']}, {len(geom['coordinates'])} part(s), "
          f"{os.path.getsize(out) / 1e6:.2f} MB -> {out}")
    return merged


def adjust_jobs(df, cfg, rates=None):
    """T-014: move LODES jobs reported at an office that are worked elsewhere (notes/T-014.md).
    Works on the block table (jobs as float) before spreading; totals are unchanged. With
    `rates` (T-042: work-from-home share per CNS sector), the `work_c` column (commuting jobs)
    moves with them: n jobs of sector s carry n x (1 - rate_s) commuters."""
    adj = cfg.get("job_adjustments", {})
    jobs = df["jobs"].to_numpy(np.float64).copy()
    work = df["work_c"].to_numpy(np.float64).copy() if rates else None
    pop = df["pop"].to_numpy(np.float64)
    county = df["geoid"].str[:5].to_numpy()
    row = {g: i for i, g in enumerate(df["geoid"])}
    log = []

    def move(i, n, sector, to_block=None, to_counties=None):
        nc = n * (1.0 - rates[sector]) if rates else 0.0
        jobs[i] -= n
        if rates:
            work[i] -= nc
        if to_block is not None:
            jobs[to_block] += n
            if rates:
                work[to_block] += nc
        else:
            w = np.where(np.isin(county, list(to_counties)), pop, 0.0)
            jobs[:] += n * w / w.sum()
            if rates:
                work[:] += nc * w / w.sum()

    for a in adj.get("moves", []):
        i = row[a["block"]]
        if "keep" in a:
            n = jobs[i] - a["keep"]
        else:  # one sector's jobs, less what stays at the office
            n = float(df[a["sector"]].iat[i]) - a["keep_sector"]
        assert 0 < n <= jobs[i], (a["name"], n, jobs[i])
        move(i, n, a["sector"], row.get(a.get("to_block")), a.get("to_counties"))
        log.append((a["name"], a["block"], n))
    hc = adj.get("home_care")
    if hc:
        c0 = df["C000"].to_numpy(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            sel = ((c0 >= hc["min_jobs"]) & (df["CNS16"] / c0 >= hc["min_health"])
                   & (df["CE01"] / c0 >= hc["min_low_pay"]) & (df["CE03"] / c0 <= hc["max_high_pay"])).to_numpy()
        for i in np.nonzero(sel)[0]:
            n = float(df["CNS16"].iat[i])
            move(i, n, "CNS16", to_counties=[county[i]])
            log.append(("home care", df["geoid"].iat[i], n))
    moved = sum(n for *_, n in log)
    print(f"job adjustments (T-014): {len(log)} blocks, {moved:,.0f} jobs moved")
    for name, g, n in log:
        if name != "home care":
            print(f"    {name}: {n:,.0f} jobs off block {g}")
    nh = [n for name, _, n in log if name == "home care"]
    if nh:
        print(f"    home care: {len(nh)} agency blocks, {sum(nh):,.0f} jobs spread over their county's population")
    assert abs(jobs.sum() - df["jobs"].sum()) < 1e-3 * df["jobs"].sum()
    out = df.copy()
    out["jobs"] = jobs
    if rates:
        assert abs(work.sum() - df["work_c"].sum()) < 1e-3 * df["work_c"].sum()
        out["work_c"] = work
    return out, log


def build_water(cfg, boundary):
    """The water mask over the boundary's bounding box (T-030): TIGER water at `cell_m`, then
    water narrower than `min_width_px` pixels dropped (pipeline/water.py)."""
    if cfg.get("country") == "ca":  # T-277: StatCan lakes and the land outside its shorelines
        import canada
        return canada.build_water(cfg, boundary)
    t = time.time()
    lon0, lat0 = cfg["origin"]
    wc = cfg["water"]
    mask, info = water.build_mask(lon0, lat0, boundary.bounds, wc["cell_m"], fetch, fetch(COUNTIES))
    raw_runs = len(mask.xs) // 2
    if wc["min_width_px"] > 1:
        mask = mask.without_narrow(wc["min_width_px"])
    info["min_width_m"] = wc["min_width_px"] * wc["cell_m"]
    print(f"water: {mask.W} x {mask.H} pixels of {mask.cell:.0f} m, {info['counties']} counties' AREAWATER "
          f"({info['areawater_polygons']:,} polygons); {raw_runs:,} runs, {len(mask.xs) // 2:,} after dropping "
          f"water under {info['min_width_m']:.0f} m wide; {(mask.row_off.nbytes + mask.xs.nbytes) / 1e6:.2f} MB; "
          f"water {mask.water_pixels() / (mask.W * mask.H):.1%} of the extent; {time.time() - t:.0f} s")
    return mask, info


def commute_by_county(city, cfg, df):
    """T-042: the gravity's two margins by county, before (workers = population scaled to all
    jobs; all LODES jobs) and after (commuters only). Printed and kept in data/work."""
    c = df["geoid"].str[:5]
    g = pd.DataFrame({"pop": df["pop"], "jobs": df["jobs"], "home_c": df["home_c"], "work_c": df["work_c"]}).groupby(c).sum()
    J, P, W, H = g["jobs"].sum(), g["pop"].sum(), g["work_c"].sum(), g["home_c"].sum()
    t = pd.DataFrame({"county": [cfg["counties"][k] for k in g.index],
                      "from_home_before": g["pop"] * J / P, "from_home_after": g["home_c"] * W / H,
                      "to_work_before": g["jobs"], "to_work_after": g["work_c"]}, index=g.index)
    t.round(0).to_csv(os.path.join(WORK, f"{city}.commute_by_county.csv"))
    print(f"commutes per day (gravity margins), before -> after work from home: {J:,.0f} -> {W:,.0f} ({W / J - 1:+.1%})")
    for k, r in t.sort_values("to_work_before", ascending=False).iterrows():
        print(f"    {r['county']:<14} from home {r['from_home_before']:>10,.0f} -> {r['from_home_after']:>10,.0f} "
              f"({r['from_home_after'] / r['from_home_before'] - 1:+6.1%})   to work {r['to_work_before']:>10,.0f} -> "
              f"{r['to_work_after']:>10,.0f} ({r['to_work_after'] / r['to_work_before'] - 1:+6.1%})")


def write_cells(city, cfg, cells, totals, sources, mask=None, mask_info=None, commute=None, regions=None):
    """The cell arrays (and the water mask) as a format-1 pack without the gravity, in
    data/work/<city>.cells.*; pack_gravity keeps every array it does not know."""
    order = np.argsort(cells["h3"])
    arrays = [
        ("h3", "u64", "cell", cells["h3"][order].astype("<u8")),
        ("x_m", "f32", "cell", cells["x_m"][order].astype("<f4")),
        ("y_m", "f32", "cell", cells["y_m"][order].astype("<f4")),
        ("pop", "f32", "cell", cells["pop"][order].astype("<f4")),
        ("jobs", "f32", "cell", cells["jobs"][order].astype("<f4")),
    ]
    if "commute_home" in cells:  # T-042: what the gravity balances on (notes/T-004.md)
        arrays += [("commute_home", "f32", "cell", cells["commute_home"][order].astype("<f4")),
                   ("commute_work", "f32", "cell", cells["commute_work"][order].astype("<f4"))]
    for name in ("errand_jobs", "leisure_jobs"):
        if name in cells:
            arrays.append((name, "f32", "cell", cells[name][order].astype("<f4")))
    if "region" in cells:  # T-162: what the gravity's zones are split by
        arrays.append(("region", "u8", "cell", cells["region"][order].astype("u1")))
    if mask is not None:
        arrays += [("water_row", "u32", "water", mask.row_off.astype("<u4")),
                   ("water_x", "u16", "water", mask.xs.astype("<u2"))]
    buf = bytearray()
    entries = []
    for name, dtype, per, a in arrays:
        buf += b"\0" * (-len(buf) % 8)
        entries.append({"name": name, "dtype": dtype, "per": per, "offset": len(buf), "count": int(len(a))})
        buf += a.tobytes()
    lon0, lat0 = cfg["origin"]
    header = {
        "format": FORMAT,
        "city": city,
        "built": datetime.date.today().isoformat(),
        "origin": {"lon": lon0, "lat": lat0},
        "h3_res": 9,
        "cells": int(len(order)),
        "arrays": entries,
        "totals": totals,
        "sources": sources,
    }
    if mask is not None:
        header["water"] = mask.header(mask_info)
    if commute is not None:
        header["commute"] = commute
    if regions is not None:
        header["regions"] = regions
    if cfg.get("iso2"):  # a tier 0 pack's country, for the joint commute gravity's border factors
        header["country"] = cfg["iso2"]
    stem = os.path.join(WORK, f"{city}.cells")
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    os.replace(stem + ".bin.part", stem + ".bin")
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".json.part", stem + ".json")
    return stem + ".json"


# T-1083: `--coarse-next` (world_run --coarse): coarse_zones re-solves the gravity straight after the
# build, and nothing before it reads the factors (only the zones), so the build's solve stops at once
GRAVITY_ONCE = False


def add_gravity(cells_json, city, extra=None):
    """Solve the 2 km zone gravity with the sim crate's own code (sim/src/bin/pack_gravity.rs)
    and write the finished pack to data/packs/. One implementation of the gravity: the game
    rebuilds zone trips from these factors with the same decay table (notes/T-019.md). Zones
    split by the cells' regions, the decay x the city's border factor across them (its config's
    "border_factor", else BORDER_FACTOR_US; T-162, T-181); `extra` passes other pack_gravity
    arguments (a later --border-factor wins)."""
    factor = CITIES[city].get("border_factor", BORDER_FACTOR_US)
    # a pack's own fitted curve (T-277: Toronto's, from the census commuting flows), else the US one
    if CITIES[city].get("curve") in ("borrowed", "borrowed_zone") and "decay_km" not in CITIES[city]:
        CITIES[city]["decay_pow"], CITIES[city]["decay_km"] = __import__("tier0").borrowed_curve(
            zones=CITIES[city]["curve"] == "borrowed_zone")
    decay_pow = CITIES[city].get("decay_pow", DECAY_POW_US)
    decay_km = CITIES[city].get("decay_km", DECAY_KM_US)
    env = dict(os.environ)
    cargo_bin = os.path.join(os.path.expanduser("~"), ".cargo", "bin")
    env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    # its own target dir, so the pipeline never waits on (or blocks) the app's wasm builds
    # (from ROOT, not WORK: a scratch build that points WORK elsewhere still shares the compiled crate)
    # (TW_CARGO_TARGET_DIR: a target of its own for a TW_SIM_DIR copy, as intercity.py)
    env["CARGO_TARGET_DIR"] = os.environ.get("TW_CARGO_TARGET_DIR", os.path.join(ROOT, "data", "work", "target"))
    out_json = os.path.join(PACKS, f"{city}.json")
    if GRAVITY_ONCE:
        extra = list(extra or []) + ["--tol", "1e9"]
    cargo = shutil.which("cargo", path=env["PATH"])  # Windows does not search env's PATH itself
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    cmd = [cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", "pack_gravity",
           # TW_SIM_DIR: a copy of sim/ to build from while the tree's own is mid-edit (as routes.py)
           "--manifest-path", os.path.join(os.environ.get("TW_SIM_DIR", os.path.join(ROOT, "sim")), "Cargo.toml"), "--", cells_json, out_json,
           "--border-factor", str(factor), "--decay-pow", str(decay_pow), "--decay-km", str(decay_km)] + list(extra or [])
    subprocess.run(cmd, env=env, check=True)
    with open(os.path.join(PACKS, f"{city}.bin"), "rb") as f:
        buf = f.read()
    gz = len(gzip.compress(buf, 6))
    print(f"pack: bin {len(buf) / 1e6:.2f} MB raw, {gz / 1e6:.2f} MB gzipped -> {out_json}")
    return len(buf), gz


def checks(cfg, county_pop, county_jobs, lodes_year):
    """Compare against published county figures; prints, returns nothing."""
    counties = cfg["counties"]
    try:
        pe = pd.read_csv(fetch(POPEST), encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
        pe["fips"] = pe["STATE"] + pe["COUNTY"]
        base = pe.set_index("fips")["ESTIMATESBASE2020"]
        pub = sum(int(base[c]) for c in counties)
        ours = sum(county_pop.get(c, 0) for c in counties)
        print(f"check pop: blocks {ours:,} vs Census estimates base 2020 {pub:,} "
              f"({(ours - pub) / pub:+.3%}); worst counties:")
        diffs = sorted(((county_pop.get(c, 0) - int(base[c]), c) for c in counties), key=lambda t: -abs(t[0]))
        for d, c in diffs[:4]:
            print(f"    {counties[c]:<14} blocks {county_pop.get(c, 0):>10,}  base {int(base[c]):>10,}  {d:+,}")
    except Exception as e:  # a check, never a build blocker
        print(f"check pop: skipped ({e})")
    try:
        q = pd.read_csv(fetch(QCEW_TOTAL.format(year=lodes_year), f"qcew_{lodes_year}_a_industry10.csv"),
                        dtype={"area_fips": str})
        q = q[(q["own_code"] == 0) & q["area_fips"].isin(counties)]
        pub = int(q["annual_avg_emplvl"].sum())
        ours = sum(county_jobs.get(c, 0) for c in counties)
        print(f"check jobs: LODES {lodes_year} {ours:,} vs BLS QCEW {lodes_year} annual average "
              f"{pub:,} ({(ours - pub) / pub:+.1%}; {len(q)} of {len(counties)} counties in QCEW)")
    except Exception as e:
        print(f"check jobs: skipped ({e})")


def main(city):
    t0 = time.time()
    cfg = CITIES[city]
    if cfg.get("tier0"):  # T-287: global sources only, pipeline/tier0.py
        import tier0
        return tier0.main(city)
    if cfg.get("country") == "ca":  # T-277: Canada's census sources, pipeline/canada.py
        import canada
        return canada.main(city)
    for d in (RAW, WORK, PACKS):
        os.makedirs(d, exist_ok=True)
    county_set = set(cfg["counties"])
    state_fips = sorted({c[:2] for c in county_set})

    boundary = build_boundary(city, cfg)
    mask, mask_info = build_water(cfg, boundary)

    blocks, jobs, years, sources = [], [], set(), []
    for fips in state_fips:
        b, n_all = blocks_for_state(fips, county_set)
        print(f"blocks {STATES[fips]}: {len(b):,} of {n_all:,} in the county set, pop {b['pop_census'].sum():,}, "
              f"{b['gq_inst'].sum():,} of them in institutions (T-234) -> {b['pop'].sum():,}")
        blocks.append(b)
        w, y, url = jobs_for_state(STATES[fips], cfg["lodes_years"], county_set)
        print(f"jobs {STATES[fips]}: LODES {y}, {len(w):,} blocks, {w['jobs'].sum():,} jobs")
        jobs.append(w)
        years.add(y)
    blocks = pd.concat(blocks, ignore_index=True)
    jobs = pd.concat(jobs, ignore_index=True)
    lodes_year = max(years)
    if len(years) > 1:
        print(f"WARNING: mixed LODES years {sorted(years)}")

    m = jobs.merge(blocks[["geoid"]], on="geoid", how="left", indicator=True)
    lost = m[m["_merge"] == "left_only"]
    print(f"LODES blocks with no 2020 block centroid: {len(lost):,} blocks, {lost['jobs'].sum():,} jobs lost")
    df = blocks.merge(jobs, on="geoid", how="left")
    # T-054: where the job holders live (LODES RAC), the home end of the commute
    rac = pd.concat([residents_for_state(STATES[f], lodes_year, county_set)[0] for f in state_fips], ignore_index=True)
    df = df.merge(rac, on="geoid", how="left")
    print(f"residents (LODES RAC {lodes_year}): {rac['rac'].sum():,} jobs by home block, "
          f"{df['rac'].sum():,.0f} on blocks with a centroid")
    for k in ["jobs", "C000", "CE01", "CE03", "rac"] + wfh.CNS:
        df[k] = df[k].fillna(0).astype(np.int64)
    # T-234: no more job holders on a block with institutions than it has residents left
    df = group_quarters.cap_job_holders(df)
    print(f"job holders on blocks with institutions capped at their residents (T-234): "
          f"{df['rac_lodes'].sum():,} -> {df['rac'].sum():,}")
    df = df[(df["pop"] > 0) | (df["jobs"] > 0) | (df["rac"] > 0)]

    county_pop = blocks.groupby(blocks["geoid"].str[:5])["pop_census"].sum().to_dict()
    county_jobs = df.groupby(df["geoid"].str[:5])["jobs"].sum().to_dict()

    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(WORK, f"{city}.blocks.csv.gz"), index=False)  # for T-014/T-006 analysis
    lon0, lat0 = cfg["origin"]
    # before T-013: each block whole in the cell of its internal point (kept for the comparison)
    cell_str = [h3.latlng_to_cell(la, lo, 9) for la, lo in zip(df["lat"].to_numpy(), df["lon"].to_numpy())]
    ids = np.array([h3.str_to_int(c) for c in cell_str], dtype=np.uint64)
    pt_uniq, inv = np.unique(ids, return_inverse=True)
    pt_pop = np.bincount(inv, weights=df["pop"].to_numpy(), minlength=len(pt_uniq))
    pt_job = np.bincount(inv, weights=df["jobs"].to_numpy(), minlength=len(pt_uniq))
    print(f"internal points: {len(pt_uniq):,} cells, {int((pt_pop > 0).sum()):,} with people, "
          f"{int((pt_job > 0).sum()):,} with jobs")
    # T-013: spread over the block polygon's land
    # T-042: commuters only (work from home makes no commute), ACS 2019-2023
    home_c, work_c, rates, wstats = wfh.commuters(df, RAW, fetch)
    df["home_c"], df["work_c"] = home_c, work_c
    ts = wstats["tracts"]
    print(f"work from home: {ts['tracts']:,} tracts, {ts['worked_from_home']:,.0f} of {ts['workers']:,.0f} "
          f"resident workers ({ts['region_share']:.1%}); blocks without a tract {ts['blocks_without_tract']} "
          f"({ts['pop_without_tract']:,.0f} people, given their county's share)")
    for r in wstats["industries"]:
        print(f"    {r['industry']:<42} {r['share']:6.1%} of {r['workers']:>9,}")
    adj_df, adj_log = adjust_jobs(df, cfg, rates)
    # Customer attractors retain their observed sector locations, independently of the
    # corrections to office/headquarters commuting employment.
    adj_df["errand_jobs"] = df["CNS07"] + df["CNS19"]
    adj_df["leisure_jobs"] = df["CNS17"] + df["CNS18"]
    # T-162: each cell's state, for the gravity's zones and border factor (pipeline/regions.py)
    adj_df, region_codes, region_cols = regions.add_columns(adj_df)
    uniq, vals, st = spread.spread(adj_df, county_set, fetch, mask, lon0, lat0,
                                   cols=("pop", "jobs", "home_c", "work_c", "errand_jobs", "leisure_jobs") + tuple(region_cols))
    pop, job, hc, wc = vals["pop"], vals["jobs"], vals["home_c"], vals["work_c"]
    print(f"spread: {st}")
    keep = (pop > 0) | (job > 0)
    uniq, pop, job, hc, wc = uniq[keep], pop[keep], job[keep], hc[keep], wc[keep]
    region = regions.of_cells({c: vals[c][keep] for c in region_cols}, region_cols)
    commute_by_county(city, cfg, adj_df)
    print(f"spread: {len(uniq):,} cells, {int((pop > 0).sum()):,} with people, {int((job > 0).sum()):,} with jobs")
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in uniq])
    x = R_EARTH * np.radians(ll[:, 1] - lon0) * np.cos(np.radians(lat0))
    y = R_EARTH * np.radians(ll[:, 0] - lat0)

    totals = {"pop": int(round(pop.sum())), "jobs": int(round(job.sum())),
              "commute_home": int(round(hc.sum())), "commute_work": int(round(wc.sum()))}
    sources = [
        {"what": "population", "name": group_quarters.POP_SOURCE,
         "url": "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"},
        {"what": "jobs", "name": f"LEHD LODES 8 WAC S000 JT00 {lodes_year}, C000 (all jobs)",
         "url": "https://lehd.ces.census.gov/data/lodes/LODES8/"},
        {"what": "boundary", "name": BOUNDARY_NAME + cfg.get("boundary_notes", "notes/T-003.md"),
         "url": BOUNDARY_URL},
    ]
    sources.append({"what": "water", "name": "TIGER/Line 2020 AREAWATER by county, and the sea outside "
                    "the TIGER 2020 state polygons", "url": "https://www2.census.gov/geo/tiger/TIGER2020/AREAWATER/"})
    sources.append({"what": "work from home", "name": "ACS 2019-2023 5-year, B08301 by tract (home end) and "
                    "B08126 by industry over the city's tracts (work end), table-based summary files",
                    "url": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/"})
    sources.append({"what": "home end of commutes", "name": f"LEHD LODES 8 RAC S000 JT00 {lodes_year}, C000 (jobs by home block)",
                    "url": "https://lehd.ces.census.gov/data/lodes/LODES8/"})
    sources.append({"what": "group quarters", "name": group_quarters.SOURCE, "url": group_quarters.SOURCE_URL})
    commute = {"home": group_quarters.HOME_RULE,
               "work": "sum over LODES sectors of jobs x (1 - industry work-from-home share)",
               "region_share": round(ts["region_share"], 4),
               "industries": wstats["industries"]}
    cells_json = write_cells(city, cfg, {"h3": uniq, "x_m": x, "y_m": y, "pop": pop, "jobs": job,
                                         "errand_jobs": vals["errand_jobs"][keep], "leisure_jobs": vals["leisure_jobs"][keep],
                                         "commute_home": hc, "commute_work": wc, "region": region},
                             totals, sources, mask, mask_info, commute, regions.header_block(region_codes))
    nbytes, gz = add_gravity(cells_json, city)
    # Attendance inputs are separate from commuting sectors; rebuild them with the pack. Every
    # US city, New York included: NCES public schools and IPEDS (T-142, T-181).
    import enrich_schools
    enrich_schools.main(city)
    import enrich_universities
    enrich_universities.main(city)
    # T-142: airport ground trips (OurAirports, FAA, DB1B) and medical visits (OSM)
    import enrich_airports
    enrich_airports.main(city)
    import enrich_medical
    enrich_medical.main(city)
    # T-196: parks as social/leisure destinations (OSM)
    import enrich_parks
    enrich_parks.main(city)

    print(f"totals: pop {totals['pop']:,}, jobs {totals['jobs']:,}")
    checks(cfg, county_pop, county_jobs, lodes_year)
    n = len(uniq)
    both = int(((pop > 0) & (job > 0)).sum())
    print(f"cells: {n:,}; pop only {int(((pop > 0) & (job == 0)).sum()) / n:.1%}, "
          f"jobs only {int(((pop == 0) & (job > 0)).sum()) / n:.1%}, both {both / n:.1%}")
    for label, v in (("pop", pop), ("jobs", job)):
        print(f"top cells by {label}:")
        for i in np.argsort(-v)[:8]:
            print(f"    {h3.int_to_str(int(uniq[i]))}  {ll[i, 0]:.4f},{ll[i, 1]:.4f}  "
                  f"pop {pop[i]:>8,.0f}  jobs {job[i]:>8,.0f}")
    print(f"done in {time.time() - t0:.1f} s")


def boundary_only(ids):
    """T-352: rewrite only data/packs/<id>.boundary.geojson (and the pack header's boundary source
    line) for US packs, with no other build step: the pack, its gravity and its routes stay as they
    are. No ids: every US pack."""
    import re
    for city in ids or [c for c, v in CITIES.items() if not v.get("tier0") and v.get("country") != "ca"]:
        cfg = CITIES[city]
        assert not cfg.get("tier0") and cfg.get("country") != "ca", f"{city} is not a US pack"
        build_boundary(city, cfg)
        path = os.path.join(PACKS, f"{city}.json")
        with open(path, "rb") as f:
            text = f.read().decode("utf-8")
        notes = cfg.get("boundary_notes", "notes/T-003.md")
        pattern = re.compile(r'("name": )"Census cb_2020_us_county_500k, counties in ' + re.escape(notes) + r'"(,\s*"url": )"'
                             + re.escape(COUNTIES) + '"')
        new, n = pattern.subn(lambda m: f'{m.group(1)}"{BOUNDARY_NAME}{notes}"{m.group(2)}"{BOUNDARY_URL}"', text)
        if n:
            with open(path + ".part", "wb") as f:
                f.write(new.encode("utf-8"))
            os.replace(path + ".part", path)
        print(f"  {city}: pack header boundary source {'updated' if n else 'already current'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # place names outside English; a redirected log is cp1252 on Windows (T-289)
    GRAVITY_ONCE = "--coarse-next" in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--boundary-only" in sys.argv:
        boundary_only(ids)
    else:
        main(ids[0] if ids else "nyc")
