"""T-509 stage 3: Paris inputs for realnet.py (`net par`) and calib.py (`targets par`).

    python pipeline/realnet_par.py gtfs      # IDFM GTFS -> its rail routes only, data/raw/t509/idfm/rail/

IDFM's open GTFS (data.iledefrance-mobilites.fr "offre-horaires-tc-gtfs-idfm",
https://eu.ftp.opendatasoft.com/stif/GTFS/IDFM-gtfs.zip, about 160 MB, 1.1 GB of stop times with
the buses) is cut to the Metro, RER, Transilien, tram and the two airport VALs (route_type 0, 1,
2 less TER); realnet's `gtfs` adapter reads the cut folder.
"""
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDFM = os.path.join(ROOT, "data", "raw", "t509", "idfm")
ZIP = os.path.join(IDFM, "IDFM-gtfs.zip")
RAIL = os.path.join(IDFM, "rail")
GROUPS = os.path.join(RAIL, "zdc_groups.csv")


def cmd_gtfs():
    z = zipfile.ZipFile(ZIP)
    rd = lambda n, **kw: pd.read_csv(z.open(n), dtype=str, keep_default_na=False, **kw)
    os.makedirs(RAIL, exist_ok=True)
    routes = rd("routes.txt")
    agency = rd("agency.txt")
    tn = dict(zip(agency["agency_id"], agency["agency_name"]))
    keep = routes[routes["route_type"].isin(["0", "1", "2"]) & (routes["agency_id"].map(tn) != "TER")]
    keep.to_csv(os.path.join(RAIL, "routes.txt"), index=False)
    agency.to_csv(os.path.join(RAIL, "agency.txt"), index=False)
    trips = rd("trips.txt")
    trips = trips[trips["route_id"].isin(keep["route_id"])]
    trips.to_csv(os.path.join(RAIL, "trips.txt"), index=False)
    tids = set(trips["trip_id"])
    sids = set(trips["service_id"])
    for n in ("calendar.txt", "calendar_dates.txt"):
        c = rd(n)
        c[c["service_id"].isin(sids)].to_csv(os.path.join(RAIL, n), index=False)
    first = True
    used = set()
    out = os.path.join(RAIL, "stop_times.txt")
    for ch in pd.read_csv(z.open("stop_times.txt"), dtype=str, keep_default_na=False, chunksize=2_000_000,
                          usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]):
        ch = ch[ch["trip_id"].isin(tids)]
        used |= set(ch["stop_id"])
        ch.to_csv(out, index=False, mode="w" if first else "a", header=first)
        first = False
        print(".", end="", flush=True)
    stops = rd("stops.txt")
    par = set(stops.loc[stops["stop_id"].isin(used), "parent_station"]) - {""}
    stops[stops["stop_id"].isin(used | par)].to_csv(os.path.join(RAIL, "stops.txt"), index=False)
    print(f"\n{len(keep)} routes, {len(trips)} trips, {len(used)} stops -> {RAIL}")
    cmd_groups()
    cmd_shapes()


def cmd_shapes():
    """The rail trips' shapes (realnet.py geom draws the save's track on them)."""
    z = zipfile.ZipFile(ZIP)
    ids = set(pd.read_csv(os.path.join(RAIL, "trips.txt"), dtype=str, keep_default_na=False)["shape_id"]) - {""}
    sh = pd.read_csv(z.open("shapes.txt"), dtype=str, keep_default_na=False)
    sh[sh["shape_id"].isin(ids)].to_csv(os.path.join(RAIL, "shapes.txt"), index=False)
    print(f"{len(ids)} shapes -> {RAIL}")


def cmd_groups():
    """Each GTFS parent station is an IDFM zone de correspondance (ZdC, IDFM:<id_zdc>), the unit
    the validation counts use -> rail/zdc_groups.csv, realnet's group table."""
    s = pd.read_csv(os.path.join(RAIL, "stops.txt"), dtype=str, keep_default_na=False)
    p = s[s["location_type"] == "1"]
    g = pd.DataFrame({"stop_id": p["stop_id"], "zdc": p["stop_id"].str.replace("IDFM:", "", regex=False),
                      "name": p["stop_name"], "lon": p["stop_lon"], "lat": p["stop_lat"]})
    g.to_csv(GROUPS, index=False)
    print(f"{len(g)} stations -> {GROUPS}")


VALID = os.path.join(IDFM, "validations_oct2025.csv")
VALID_URL = ("https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/validations-reseau-ferre-nombre-validations-par-jour-4eme-trimestre/"
             "exports/csv?select=jour,id_zdc,libelle_arret,code_stif_trns,sum(nb_vald)%20as%20v&where=jour%3E%3Ddate'2025-10-06'%20and%20jour%3C%3Ddate'2025-10-17'"
             "&group_by=jour,id_zdc,libelle_arret,code_stif_trns&delimiter=,")
EMP = os.path.join(IDFM, "base-cc-caract_emp-2022_csv.zip")  # https://www.insee.fr/fr/statistiques/8581424
FR = os.path.join(ROOT, "data", "raw", "tier0", "france")
CALIB = os.path.join(ROOT, "data", "work", "calib", "par")
DAYS = ["2025-10-07", "2025-10-08", "2025-10-09", "2025-10-14", "2025-10-15", "2025-10-16"]  # Tue-Thu before the school holidays


def cmd_targets():
    """Station entries: IDFM validations on the rail network per zone de correspondance, mean of
    six Tue-Thu days of October 2025 (VALID_URL). Areas: INSEE census 2022 (base "caractéristiques
    de l'emploi", P22_ACTOCC15P_COMMUN over employed residents less those not travelling), Paris
    by arrondissement and the seven other departments; each pack cell's area -> cell_area.u8.
    `--pack <dir>` reads the pack from a copy."""
    import json
    import numpy as np
    from shapely import contains_xy, make_valid
    from shapely.geometry import shape
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import read_pack
    import realnet as rn
    os.makedirs(CALIB, exist_ok=True)
    v = pd.read_csv(VALID, dtype={"id_zdc": float}, encoding="utf-8-sig")
    v = v[v["jour"].str[:10].isin(DAYS) & (v["id_zdc"] > 0)]
    days = v["jour"].str[:10].nunique()
    g = v.groupby("id_zdc").agg(v=("v", "sum"), name=("libelle_arret", "first"))
    st = [{"target": str(int(k)), "name": str(r["name"]).title(), "entries": float(r["v"]) / days} for k, r in g.iterrows()]
    e = pd.read_csv(zipfile.ZipFile(EMP).open("base-cc-caract_emp-2022.CSV"), sep=";", dtype={"CODGEO": str},
                    usecols=["CODGEO", "P22_ACTOCC15P", "P22_ACTOCC15P_PASTRANS", "P22_ACTOCC15P_COMMUN"])
    e = e.set_index("CODGEO")
    pk = sys.argv[sys.argv.index("--pack") + 1] if "--pack" in sys.argv else read_pack.PACKS
    header, a = read_pack.load_pack("par", pk)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    clon = lon0 + np.degrees(x / (rn.R_EARTH * np.cos(np.radians(lat0))))
    clat = lat0 + np.degrees(y / rn.R_EARTH)
    units = []  # (name, polygon, commune codes)
    for f in json.load(open(os.path.join(FR, "arm_75.geojson"), encoding="utf-8"))["features"]:
        p = f["properties"]
        units.append((p["nom"], make_valid(shape(f["geometry"])), [p["code"]]))
    for f in json.load(open(os.path.join(FR, "par_departements.geojson"), encoding="utf-8"))["features"]:
        p = f["properties"]
        code = p.get("code") or p.get("codeDepartement")
        if code == "75":
            continue
        units.append((p.get("nom") or f"Dept {code}", make_valid(shape(f["geometry"])), [c for c in e.index if c[:2] == code]))
    areas, cell = [], np.full(len(x), 255, np.uint8)
    for name, poly, codes in units:
        s = e.loc[[c for c in codes if c in e.index]].sum()
        comm = float(s["P22_ACTOCC15P"] - s["P22_ACTOCC15P_PASTRANS"])
        cm = contains_xy(poly, clon, clat) & (cell == 255)
        if comm <= 0 or not cm.any():
            continue
        cell[cm] = len(areas)
        areas.append({"name": name, "rail_share": float(s["P22_ACTOCC15P_COMMUN"]) / comm, "commuters": comm})
    cell.tofile(os.path.join(CALIB, "cell_area.u8"))
    out = {"stations": st, "areas": areas, "pack": pk,
           "sources": {"stations": f"IDFM validations sur le réseau ferré, nombre de validations par jour, Q4 2025, by zone de correspondance, mean of {days} Tue-Thu days 7-16 Oct 2025 (every ticket type; entries without a validation, at ungated SNCF stations especially, are missing)",
                       "areas": "INSEE RP 2022 base caractéristiques de l'emploi: employed residents by main mode to work, public transport (bus included: the census has no rail split) over those who travel; Paris by arrondissement, the other seven departments whole"}}
    json.dump(out, open(os.path.join(CALIB, "inputs.json"), "w"), indent=0)
    print(f"{len(st)} stations ({days} days), {len(areas)} areas ({(cell != 255).sum()} of {len(cell)} cells) -> {CALIB}")


def targets():
    """calib.py's targets for Paris from inputs.json. No operator or line counts (none open by
    line); hubs: the main line termini, where riders arrive on TER, Intercites and TGV."""
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import realnet as rn
    d = json.load(open(os.path.join(CALIB, "inputs.json")))
    termini = ("Gare De Lyon", "Gare Du Nord", "Gare De L'Est", "Montparnasse", "Saint-Lazare", "Gare D'Austerlitz", "Austerlitz", "Haussmann")
    hubs = [s["target"] for s in d["stations"] if any(t.lower() in s["name"].lower() for t in termini)]
    return {"city": "par", "net": rn.CITIES["par"]["out"], "areas_file": os.path.join(CALIB, "cell_area.u8"),
            "pack": os.path.join(d["pack"], "par.json"), "station_min": 2000.0, "hub_m": 300.0,
            "stations": d["stations"], "hubs": hubs, "operators": {}, "operators_info": {},
            "areas": d["areas"], "sources": d["sources"],
            # the census share includes the bus: the model's rail should come out at or below it,
            # so only an overshoot is scored (T-509 stage 6)
            "areas_upper": True}


if __name__ == "__main__":
    {"gtfs": cmd_gtfs, "groups": cmd_groups, "shapes": cmd_shapes, "targets": cmd_targets}[sys.argv[1]]()
