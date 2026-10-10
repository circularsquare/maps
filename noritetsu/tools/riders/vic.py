"""Victoria: annual train station entries, metropolitan (Metro Trains) and regional (V/Line).

"Annual metropolitan train station patronage (station entries)" and "Annual regional train
station patronage (station entries)", Department of Transport and Planning, Victoria (CC BY
4.0), financial year July 2024 - June 2025 ("year" 2024), the latest published. Both give
Pax_annual, the year's station entries (rounded to 50); the metropolitan file also has
averages per day type (weekday, Saturday, Sunday), not used, since the annual total gives
the average of all days. n = Pax_annual x 2 / 365: entries only, doubled for entries + exits.

Positions are in the files. COMBINE "sum": the one station in both files (Southern Cross)
gets Metro's entries and V/Line's entries added, each being its own operator's count.
"""
import csv

KEY = "vic"
CC = "au"
FOLDER = "vic"
MODES = {"rail"}
COMBINE = "sum"
META = {
    "label": "Victorian Department of Transport station counts",
    "name": "Annual metropolitan and regional train station patronage (station entries) "
            "2024-25, Department of Transport and Planning Victoria",
    "url": "https://discover.data.vic.gov.au/dataset/"
           "annual-metropolitan-train-station-patronage-station-entries",
    "licence": "Creative Commons Attribution 4.0 International",
    "counts": "station entries per year x 2 / 365 (entries only, doubled for entries + "
              "exits), Metro Trains and V/Line",
    "note": "financial year 2024-25; regional file: discover.data.vic.gov.au/dataset/"
            "annual-regional-train-station-patronage-station-entries",
}
FILES = {"metro": "annual_metropolitan_train_station_entries_fy_2024_2025.csv",
         "vline": "annual_regional_train_station_entries_fy_2024_2025.csv"}
YEAR = 2024
FORCE = {
    # au's node at Chiltern (0.1 km from the source's position, V/Line North East) has no
    # name
    "Chiltern": "n2342994891",
}


def records(raw):
    out = []
    for op, fn in FILES.items():
        with open(raw / fn, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    n = float(r["Pax_annual"])
                    x, y = float(r["Stop_long"]), float(r["Stop_lat"])
                except (TypeError, ValueError):
                    continue
                if n <= 0:
                    continue
                out.append({"name": r["Stop_name"].strip(), "x": x, "y": y,
                            "n": n * 2 / 365, "year": YEAR, "op": op})
    return out
