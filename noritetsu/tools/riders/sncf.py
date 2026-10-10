"""France: SNCF Gares & Connexions "Fréquentation en gares", 2015-2024.

"total_voyageurs_<year>": travellers getting on and off at the station in the year (montants +
descendants; the column with non_voyageurs adds people passing through without a train, left
out). SNCF's figure at a station shared with RATP (RER A and B in Paris) counts the whole
station, RATP's passengers included. Annual / 365 for an average day, the latest year with a
figure. The UIC code (code_uic_complet) is what fr's "fr<uic>" ids are made of; the name is
trusted, and stations without such an id are matched by name and position (positions from
SNCF's "Gares de voyageurs" list by UIC; a name unique in France where it has none).
"""
import csv

KEY = "sncf"
CC = "fr"
FOLDER = "sncf"
COMBINE = "max"
MODES = {"rail", "tram"}   # tram-trains (T11, T13) are SNCF stations
TRUST_CODE = True   # fr's "fr<uic>" ids are SNCF's own codes ("Paris Est" is fr87113001)
META = {
    "label": "SNCF Gares & Connexions station counts",
    "name": "Fréquentation en gares, SNCF Gares & Connexions",
    "url": "https://ressources.data.sncf.com/explore/dataset/frequentation-gares/",
    "licence": "Open Licence Etalab 2.0 (licence ouverte)",
    "counts": "travellers getting on + off per day (annual total / 365); RATP passengers "
              "included at shared stations",
    "note": "",
}
FILE = "frequentation-gares.csv"


def positions(raw):
    """UIC -> (lon, lat) from SNCF's "Gares de voyageurs" list (codes_uic may hold several)."""
    pos = {}
    with open(raw / "gares-de-voyageurs.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                lat, lon = (float(t) for t in r["position_geographique"].split(","))
            except ValueError:
                continue
            for c in (r["codes_uic"] or "").replace(",", ";").split(";"):
                c = c.strip()
                if c:
                    pos[c] = (lon, lat)
    return pos


def records(raw):
    pos = positions(raw)
    out = []
    with open(raw / FILE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            year, n = None, None
            for y in range(2030, 2014, -1):
                v = r.get(f"total_voyageurs_{y}") or r.get(f"totalvoyageurs{y}")
                if v not in (None, ""):
                    try:
                        val = float(v)
                    except ValueError:
                        continue
                    if val > 0:
                        year, n = y, val
                        break
            if not n:
                continue
            uic = (r["code_uic_complet"] or "").strip()
            xy = pos.get(uic) or (None, None)
            out.append({"name": r["nom_gare"].strip(), "sid": f"fr{uic}" if uic else None,
                        "uic": uic, "x": xy[0], "y": xy[1], "n": n / 365, "year": year})
    return out
