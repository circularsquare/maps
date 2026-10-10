"""Switzerland: SBB Passagierfrequenz (passenger frequency per station).

DTV ("durchschnittlicher täglicher Verkehr"): passengers getting on and off per day, averaged
over every day of the year, for the trains of the railway companies SBB counts at that
station (the "evu" column; the remarks name what is left out, e.g. "Ohne AVA." at Aarau).
The latest year per station. The BPUIC code is what ch's "c<uic>" ids are made of (trusted);
the file's position serves the rest.
"""
import csv

KEY = "sbb"
CC = "ch"
FOLDER = "sbb"
COMBINE = "max"
MODES = {"rail"}
TRUST_CODE = True
META = {
    "label": "SBB passenger counts",
    "name": "Passagierfrequenz, SBB",
    "url": "https://data.sbb.ch/explore/dataset/passagierfrequenz/",
    "licence": "SBB open data: free use, source must be cited",
    "counts": "passengers getting on + off per day, average over all days of the year "
              "(DTV), trains of the companies SBB counts there",
    "note": "",
}
FILE = "passagierfrequenz.csv"


def records(raw):
    best = {}
    with open(raw / FILE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                uic = str(int(float(r["uic"])))
                year = int(r["jahr_annee_anno"])
                n = float(r["dtv_tjm_tgm"])
            except (ValueError, TypeError):
                continue
            if n <= 0:
                continue
            if uic in best and best[uic]["year"] >= year:
                continue
            try:
                lat, lon = (float(t) for t in r["geopos"].split(","))
            except (ValueError, AttributeError):
                lat = lon = None
            best[uic] = {"name": r["bahnhof_gare_stazione"].strip(), "sid": f"c{uic}",
                         "x": lon, "y": lat, "n": n, "year": year,
                         "remark": r.get("remarks") or ""}
    return list(best.values())
