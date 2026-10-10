"""Malaysia, KTMB: data.gov.my origin-destination ridership for KTM Komuter (Klang Valley),
KTM Komuter Utara, ETS and KTM Intercity, 2025.

ridership_od_komuter, ridership_od_komuter_utara, ridership_od_ets, ridership_od_intercity
(data.gov.my, CC BY 4.0; built from KTMB's KITS ticketing transactions): trips per origin,
destination, day and hour. The 2025 files are 1 January to 31 December (365 days); their
yearly totals equal the data.gov.my headline daily ridership for each service. A station's
figure is trips with it as origin plus trips with it as destination (entries + exits), summed
over the year and divided by 365. "Unknown" and "Penalty" rows are left out. Shuttle Tebrau
has no OD file (headline only), so JB Sentral counts only ETS and Intercity.

Each service is its own record ("op"): where several services stop at one station (KL Sentral:
Komuter and ETS), their trips are separate tickets and are added (COMBINE sum). Intercity's
Hat Yai (Thailand) is not in my and gets nothing. No positions in the files: names are matched
country-wide among stations a rail-class line stops at, FORCE taking the spellings that differ.
"""
from collections import defaultdict

KEY = "my_ktm"
CC = "my"
FOLDER = "my_ktm"
COMBINE = "sum"
MODES = {"rail"}
META = {
    "label": "KTM ridership from data.gov.my",
    "name": "KTMB origin-destination ridership 2025: KTM Komuter, Komuter Utara, ETS, "
            "Intercity (via data.gov.my)",
    "url": "https://data.gov.my/data-catalogue/ridership_od_komuter",
    "licence": "CC BY 4.0",
    "counts": "trips starting + trips ending at the station (origin + destination in the OD "
              "files), 2025 total / 365, all four KTMB services added",
    "note": "also ridership_od_komuter_utara, ridership_od_ets, ridership_od_intercity; "
            "Shuttle Tebrau has no OD file",
}
FILES = {"komuter": "komuter_2025.parquet", "komuter_utara": "komuter_utara_2025.parquet",
         "ets": "ets_2025.parquet", "intercity": "intercity_2025.parquet"}
YEAR = 2025
SKIP = {"Unknown", "Penalty", "Hat Yai"}
FORCE = {
    # KTMB's ticketing spellings
    "Bandar Tasek Selatan": "ya24375080",
    "Tanjong Malim": "y4524260193",
    "Telok Gadong": "y9901332520",
    "Telok Pulai": "ya257615487",
    "JB Sentral": "y1639480363",              # Johor Bahru Sentral
    "Krai": "y2630742717",                    # Kuala Krai, the only Krai on the East Coast line
    "Sg Mengkuang Baru": "y2630745481",       # Kampung Baru Sungai Mengkuang
    "Krambit": "y13711708353",                # Kerambit
    "Padang Tungku": "y2630937596",           # Padang Tengku
    "Sungai Sirian": "y2630769255",           # Sungai Serian
    "Kampung Sirian": "y2630769256",          # Kampung Sungai Serian
    # the Port Klang line's terminus (in South Port); the file has no plain "Pelabuhan Klang"
    "Pelabuhan Klang Selatan": "ya198136704",
}


def parquet_rows(path, cols):
    import pyarrow.parquet as pq      # ParquetFile, not read_table: no pandas import
    t = pq.ParquetFile(str(path)).read(columns=cols)
    return [t.column(c).to_pylist() for c in cols]


def records(raw):
    out = []
    for op, fn in FILES.items():
        org, dst, date, n = parquet_rows(raw / fn, ["origin", "destination", "date",
                                                    "ridership"])
        tot = defaultdict(float)
        for a, b, k in zip(org, dst, n):
            if not k:
                continue
            tot[a] += k
            tot[b] += k
        days = len({str(d)[:10] for d in date})
        for name, t in tot.items():
            if name in SKIP or t <= 0:
                continue
            out.append({"name": name, "op": op, "n": t / days, "year": YEAR})
    return out
