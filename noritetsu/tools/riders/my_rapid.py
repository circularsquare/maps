"""Malaysia, Klang Valley Rapid Rail (Prasarana): data.gov.my ridership_od_rapidrail_daily, 2025.

rapidrail_2025_daily.parquet (data.gov.my, CC BY 4.0, 5.5 MB): trips per origin station,
destination station and day for every LRT (Kelana Jaya, Ampang, Sri Petaling), MRT (Kajang,
Putrajaya) and KL Monorail station, 1 January to 31 December 2025 (365 days). A trip leaves
its origin and arrives at its destination, so a station's figure is the trips with it as
origin plus the trips with it as destination (a trip from a station back to itself counts at
it twice: an entry and an exit), summed over the year and divided by 365. This is entries +
exits at the station's own gates: a passenger changing lines inside the paid area (Masjid
Jamek, Hang Tuah) is not counted there. The file's "A0: All Stations" rows are totals of the
rest and are left out.

The Shah Alam line (LRT3, "SA" codes) is not in the 2025 file; its stations come from
rapidrail_2026_daily.parquet, 1 January to 8 October 2026 (281 days, every SA station there
from the first day), as their own operator ("op") with year 2026. Bandar Utama's figure is
therefore KG09 (2025) + SA01 (2026), written with y0 2025.

The source names each line's platforms separately ("KJ15: KL Sentral", "MR01: KL Sentral"),
and noritetsu has one id per interchange: those rows are added (COMBINE sum), since each is
its own gate line's count. The file has no positions; names are matched within the Klang
Valley among stations a metro-class line stops at, and FORCE takes the spellings that differ.
"""
from collections import defaultdict

KEY = "my_rapid"
CC = "my"
FOLDER = "my_rapid"
COMBINE = "sum"
MODES = {"metro"}
META = {
    "label": "Rapid KL ridership from data.gov.my",
    "name": "Rapid Rail origin-destination ridership, daily, 2025 (Prasarana, via data.gov.my)",
    "url": "https://data.gov.my/data-catalogue/ridership_od_rapidrail_daily",
    "licence": "CC BY 4.0",
    "counts": "trips entering + trips leaving the station's gates (origin + destination in the "
              "OD file), 2025 total / 365; LRT, MRT and KL Monorail",
    "note": "interchanges with one gate line per line are added up; transfers inside the "
            "paid area are not counted; the Shah Alam line is 1 January to 8 October 2026 "
            "(it is not in the 2025 file)",
}
FILE = "rapidrail_2025_daily.parquet"
YEAR = 2025
FILE_SA = "rapidrail_2026_daily.parquet"      # Shah Alam line only
YEAR_SA = 2026
BOX = (101.3, 2.7, 102.0, 3.5)          # Klang Valley
FORCE = {
    "TTDI": "ya457857876",                      # Taman Tun Dr Ismail
    "Bandar Tun Hussien Onn": "ya505251364",    # the file's spelling of Hussein
    "Kentomen": "ya769009409",                  # MRT Kentonmen (PYL14)
    "Kinrara": "ya379709358",                   # Kinrara BK 5 (SP22)
    "Cgc Glenmarie": "ya424884270",             # Glenmarie (KJ27)
    "Bank Rakyat Bangsar": "ya24375064",        # Bangsar (KJ16)
    "Bandar Utama 11": "ya1119899987",           # BU 11 (Bandar Utama 11), SA03
    "Dato' Menteri - SA Sentral": "y10605496603",   # Dato' Menteri, SA12
    "Seksyen 7": "y10605496597",                # Seksyen 7 Shah Alam, SA15
}


def parquet_rows(path, cols):
    import pyarrow.parquet as pq      # ParquetFile, not read_table: no pandas import
    t = pq.ParquetFile(str(path)).read(columns=cols)
    return [t.column(c).to_pylist() for c in cols]


def station_means(path, keep):
    """{"KG18: Bukit Bintang": trips in + out per day} for the codes keep() accepts."""
    org, dst, date, n = parquet_rows(path, ["origin", "destination", "date", "ridership"])
    tot = defaultdict(float)
    for a, b, k in zip(org, dst, n):
        if not k or a.startswith("A0") or b.startswith("A0"):
            continue        # "A0: All Stations" rows are totals of the rest
        for s in (a, b):
            if keep(s):
                tot[s] += k
    days = len({str(d)[:10] for d in date})
    return {s: t / days for s, t in tot.items()}


def records(raw):
    out = []
    for path, year, op, keep in (
            (raw / FILE, YEAR, "rapid", lambda s: not s.startswith("SA")),
            (raw / FILE_SA, YEAR_SA, "shah_alam", lambda s: s.startswith("SA"))):
        for code_name, n in station_means(path, keep).items():
            code, _, name = code_name.partition(": ")
            out.append({"name": name.strip(), "code": code, "op": op, "box": BOX,
                        "n": n, "year": year})
    return out
