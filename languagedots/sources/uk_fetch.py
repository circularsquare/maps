"""United Kingdom: download every raw file sources/uk_census.py reads, into data/raw/uk/.

    python sources/uk_fetch.py            fetch what is missing; resumes the OA download

Three censuses, three agencies (sources/uk.md):

  England and Wales, ONS Census 2021
    census2021-ts024.zip              TS024 Main language (detailed), 95 categories, LTLA and up
                                      (Nomis bulk; no OA, LSOA or MSOA file exists for TS024)
    ew_oa_main_language_26a.csv       main_language_detailed_26a, 26 categories, all 188,880
                                      Output Areas, from ONS's "create a custom dataset" API
                                      (census-observations), 300 OAs a request because the API
                                      refuses a whole-country OA query ("Too many rows")
    ew_ltla_main_language_26a.csv     the same variable at LTLA, to check the OA download
    ew_oa_welsh_speak.csv             welsh_skills_speak (can / cannot speak Welsh, aged 3+;
                                      TS033's question) at Wales's 10,275 Output Areas, which
                                      splits Wales's "English or Welsh" box (sources/uk.md §1)
    ew_ltla_main_language_x_welsh_speak.csv
                                      main_language_11a by welsh_skills_speak at LTLA: the
                                      cross-tab itself, published for 15 of Wales's 22 districts
                                      (the API blocks the rest and nearly every MSOA, LSOA and
                                      OA); the check on the split
    ew_ltla_welsh_skills_6a.csv       welsh_skills_all_6a at LTLA, for the stricter
                                      "speak, read and write" comparison in sources/uk.md
    oa21_lsoa_msoa_lad21_ew_lu.csv    ONS OA (2021) -> LAD (Dec 2021) exact-fit lookup, the
                                      331 districts TS024 uses (ArcGIS item b9ca90c1...)
  Scotland, NRS Census 2022
    sc_Census-2022-Output-Area-v1.zip all OA topic tables; UV212 Main language is read
  Northern Ireland, NISRA Census 2021
    ni_main_language_DZ21.csv         MAIN_LANGUAGE_1000 (20 categories) by Data Zone, Flexible
                                      Table Builder
    ni_main_language_LGD14.csv        the same by district, a check
    ni_census-2021-ms-b13.xlsx        MS-B13 Main language, full detail, Northern Ireland only

Open Government Licence v3.0 throughout. Generic browser User-Agent; nothing needs a login.
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "uk"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
ONS = "https://api.beta.ons.gov.uk/v1/population-types/UR/census-observations"
VAR = "main_language_detailed_26a"
OA_FILE = RAW / "ew_oa_main_language_26a.csv"
LTLA_FILE = RAW / "ew_ltla_main_language_26a.csv"
LOOKUP = RAW / "oa21_lsoa_msoa_lad21_ew_lu.csv"
WELSH_OA = RAW / "ew_oa_welsh_speak.csv"
WELSH_X = RAW / "ew_ltla_main_language_x_welsh_speak.csv"
WELSH_6A = RAW / "ew_ltla_welsh_skills_6a.csv"
CHUNK = 300

FILES = {
    "census2021-ts024.zip": "https://www.nomisweb.co.uk/output/census/2021/census2021-ts024.zip",
    "sc_Census-2022-Output-Area-v1.zip":
        "https://www.scotlandscensus.gov.uk/media/zz85kfinmf97whklasd98gfkadft5hj4f_Topic2H_"
        "20241120_1747/Census-2022-Output-Area-v1.zip",
    "ni_main_language_DZ21.csv":
        "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=DZ21&v=MAIN_LANGUAGE_1000",
    "ni_main_language_LGD14.csv":
        "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=LGD14&v=MAIN_LANGUAGE_1000",
    # home and daily use (sources/uk_home_use.py, 2026-10-06); the Welsh APS table
    # wales_aps_welsh_frequency_la.csv is paged from its StatsWales HTML view (uk.md §10)
    "ni_irish_speak_frequency_DZ21.csv":
        "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=DZ21&v=IRISH_SKILLS_SPEAK_FREQUENCY",
    "ni_irish_speak_frequency_x_student_LGD14.csv":
        "https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v=LGD14&v=IN_FULL_TIME_EDUCATION"
        "&v=IRISH_SKILLS_SPEAK_FREQUENCY",
    # school use taken out of NI's daily Irish (uk.md §10, 2026-10-06); DZ and SDZ cells are
    # partly blanked by NISRA, DEA is complete. The DZ -> SDZ -> DEA lookup
    # ni_dz2021_lookup.csv is the attribute table of religiondots' NISRA DZ2021 shapefile
    **{f"ni_irish_speak_frequency_x_student_{g}.csv":
       f"https://build.nisra.gov.uk/en/custom/table.csv?d=PEOPLE&v={g}&v=IN_FULL_TIME_EDUCATION"
       "&v=IRISH_SKILLS_SPEAK_FREQUENCY" for g in ("DZ21", "SDZ21", "DEA14")},
    "ni_census-2021-ms-b13.xlsx":
        "https://www.nisra.gov.uk/system/files/statistics/census-2021-ms-b13.xlsx",
    # Scotland's "Other language" split (sources/uk_scot_other.py, 2026-10-06)
    "sc2011_AT_002_2011.xls": "http://web.archive.org/web/2016id_/http://www.scotlandscensus.gov.uk"
                              "/documents/additional_tables/AT_002_2011.xls",
    "sc2011_AT_003_2011.xls": "http://web.archive.org/web/2016id_/http://www.scotlandscensus.gov.uk"
                              "/documents/additional_tables/AT_003_2011.xls",
    "sc2022_UV204_Electoral_Ward_2022.csv": "https://ukds-ckan.s3.eu-west-1.amazonaws.com/2022/NRS"
                                            "/UV204/census_2022_UV204_Country_of_birth_Electoral_Ward_2022.csv",
    "sc2022_UV204_ctry.csv": "https://ukds-ckan.s3.eu-west-1.amazonaws.com/2022/NRS/UV204"
                             "/census_2022_UV204_Country_of_birth_ctry.csv",
    "sc_census_2022_index.zip": "https://www.nrscotland.gov.uk/media/utrbt5ze/census_2022_index.zip",
    "ew_ctry_country_of_birth_190a.json": f"{ONS}?area-type=ctry&dimensions=country_of_birth_190a",
}


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=180).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            if code in (400, 403, 404) or i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def fetch_lookup():
    """ONS's Hub export is asynchronous: poll until Completed, then fetch resultUrl."""
    url = ("https://hub.arcgis.com/api/download/v1/items/b9ca90c10aaa4b8d9791e9859a38ca67/csv"
           "?redirect=false&layers=0")
    for _ in range(60):
        d = json.loads(get(url))
        if d.get("status") == "Completed":
            LOOKUP.write_bytes(get(d["resultUrl"]))
            return
        time.sleep(5)
    raise SystemExit("ONS Hub export of the OA lookup never completed")


def observations(area_arg, var=VAR, allow_blocked=False):
    d = json.loads(get(f"{ONS}?area-type={area_arg}&dimensions={var}"))
    if d.get("blocked_areas") and not allow_blocked:
        raise SystemExit(f"ONS blocked {d['blocked_areas']} areas in {area_arg[:40]}...")
    return [(o["dimensions"][0]["option_id"],
             *[x for dim in o["dimensions"][1:] for x in (dim["option_id"], dim["option"])],
             o["observation"]) for o in d["observations"]]


def fetch_wales():
    """Welsh speaking ability for Wales's OAs, plus the two district tables that check it."""
    with open(LOOKUP, encoding="utf-8-sig", newline="") as fh:
        oas = sorted({r["OA21CD"] for r in csv.DictReader(fh) if r["OA21CD"].startswith("W")})
    assert len(oas) == 10_275, len(oas)
    if not WELSH_OA.exists():
        chunks = [oas[i:i + CHUNK] for i in range(0, len(oas), CHUNK)]
        with ThreadPoolExecutor(3) as ex:
            parts = list(ex.map(lambda c: observations("oa," + ",".join(c), "welsh_skills_speak"),
                                chunks))
        tmp = WELSH_OA.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["oa", "cat_id", "category", "count"])
            for rows in parts:
                w.writerows(rows)
        tmp.replace(WELSH_OA)
    if not WELSH_X.exists():
        rows = observations("ltla", "main_language_11a,welsh_skills_speak", allow_blocked=True)
        with open(WELSH_X, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ltla", "lang_id", "language", "speak_id", "speak", "count"])
            w.writerows(rows)
    if not WELSH_6A.exists():
        rows = observations("ltla", "welsh_skills_all_6a", allow_blocked=True)
        with open(WELSH_6A, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ltla", "cat_id", "category", "count"])
            w.writerows(r for r in rows if r[0].startswith("W"))


def fetch_oa():
    with open(LOOKUP, encoding="utf-8-sig", newline="") as fh:
        oas = sorted({r["OA21CD"] for r in csv.DictReader(fh)})
    done = set()
    if OA_FILE.exists():
        with open(OA_FILE, encoding="utf-8", newline="") as fh:
            done = {r["oa"] for r in csv.DictReader(fh)}
    todo = [o for o in oas if o not in done]
    print(f"OA: {len(oas):,} in the lookup, {len(done):,} already fetched, {len(todo):,} to go")
    if not todo:
        return
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    new = not OA_FILE.exists()
    with open(OA_FILE, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["oa", "cat_id", "category", "count"])
        t0 = time.time()
        with ThreadPoolExecutor(3) as ex:
            for i, rows in enumerate(ex.map(lambda c: observations("oa," + ",".join(c)), chunks)):
                w.writerows(rows)          # a chunk is written whole or not at all
                fh.flush()
                if i % 25 == 0:
                    print(f"  {i + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        if not (RAW / name).exists():
            print("fetching", name)
            (RAW / name).write_bytes(get(url))
    if not LOOKUP.exists():
        print("fetching the OA -> LAD lookup")
        fetch_lookup()
    if not LTLA_FILE.exists():
        rows = observations("ltla")
        with open(LTLA_FILE, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ltla", "cat_id", "category", "count"])
            w.writerows(rows)
    fetch_oa()
    fetch_wales()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
