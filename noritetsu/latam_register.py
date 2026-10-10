"""Latin America (cr, pa, cu, pe, bo, ec, do, uy, ve, co, pr): register lines are the track OSM's
passenger route relations run over, grouped into lines by hand (LINES below), read by
ar_register's code (kr_register's named-track recipe), as cl_register.py reads Chile.

    python latam_register.py --clip <cc>      # after every extract: neighbours out, stale routes out
    python latam_register.py --report <cc>    # each LINE's ways, km and pieces, before any build
    python build_model.py --region <cc> --register latam_register:data/raw/<cc>
    python build_model.py --region do         # do, pr: metros only, no register (Qatar's way)

One module for eleven countries, since most have one to four register lines: the country is
the last part of the `path` argument (data/raw/<cc>). Metros, trams and light rail stay OSM
lines (their route relations are clean), as the US, UK and Brazilian builds leave them.

CLIP. ar_register.clip (the neighbours' ground out, a way kept whole if any node is at home,
route relations whose name says "en construcción" out), then, per country:
  - `drop_route_ids`: route relations that run nothing in 2026 (Ecuador's Tren Ecuador routes,
    closed since 2020; Bolivia's El Alto - Guaqui; Colombia's Bogotá Metro, under construction).
    A dropped route builds neither a register line nor an OSM line.
  - `train_routes_only`: route=train relations other than these are dropped (Cuba: only the
    four national trains are known to run; ~110 local, commuter and sugar-estate routes are a
    pre-crisis inventory).
  - `rename`: tags set on a relation ({id: {tag: value}}): Costa Rica's route_masters take
    INCOFER's line names, so build_model finds each the twin of its register line.
  - `drop_way_names`: ways whose name matches are taken out of the extract (the Santo Domingo
    monorail's ways, tagged railway=monorail although named "(en construcción)").
Each `<cc>_sources.md` has what runs, what does not and each call.
"""
import os
import pickle
import re
import sys
from pathlib import Path

import ar_register as core

ROOT = Path(__file__).resolve().parent
L = core.L

# ---------------------------------------------------------------- Costa Rica
# INCOFER's three lines (its GTFS, data/raw/cr/survey/incofer_gtfs_20260817.zip). OSM's
# routes are refs 2 (Heredia / Alajuela), 4 (Cartago) and 1 (Pacífico - Belén); each way goes
# to the first line listing a route over it, so Atlántico - CFIA, which every L2 train runs,
# is L2's, and L3 starts at Atlántico (its trains that run on to CFIA do so over L2).
#
# OSM's routes are stale in places (no UCR on the Cartago routes, Tres Ríos as a platform,
# untagged stop nodes at Heredia and Pedregal, nothing past Los Ángeles), so every line also
# takes INCOFER's feed's stop list (`lists`), and the station records are read under the
# feed's names (`name_alias`: OSM's "Fátima" is the feed's Bulevar Aeropuerto, 10 m apart;
# "Hospital Alajuela" is Alajuela, "Flores" San Joaquín). The clip renames OSM's three
# route_masters to INCOFER's line names (`rename`), so each is its register line's twin.
TI = "Tren Interurbano"
SJ_CARTAGO = r""          # any track: the shortest path between two stations 3-4 km apart
ATL = (-84.06885, 9.93481)
CR = [
    L("San José - Cartago",
      [6559776, 6559533, 7799200, 7358193],
      {"name_en": "Line 2: San José – Cartago", "operator": "INCOFER",
       "network": TI, "ref": "L2", "colour": "#002B7F"},
      # Cartago - Plaza Paraíso: 10 trips a weekday in the feed; OSM's routes stop at Los
      # Ángeles.
      extent=[(r"Cartago - Para[ií]so|" + SJ_CARTAGO,
               [(-83.92207, 9.86660), (-83.89002, 9.85054)])],
      lists=["Atlántico", "UCR", "U Latina", "CFIA", "UACA", "Tres Ríos", "Cartago",
             "Los Ángeles", "Oreamuno", "Plaza Paraíso"]),
    # L1: 7 of its 46 weekday trips run on from Atlántico to UCR and U Latina, over L2's
    # track; the line is built Atlántico - Alajuela (a shared extent found no station there,
    # and the stretch is L2's and L3's already). OSM's two Heredia - U. Latina routes are
    # clipped so OSM's L1 is the register line's twin.
    L("San José - Heredia - Alajuela",
      [6908011, 6553390, 6908041, 6908042],
      {"name_en": "Line 1: San José – Heredia – Alajuela", "operator": "INCOFER",
       "network": TI, "ref": "L1", "colour": "#CE1126"},
      lists=["Atlántico", "Calle Blancos", "Colima", "Santa Rosa",
             "Miraflores", "Heredia", "San Francisco", "San Joaquín", "Río Segundo",
             "Bulevar Aeropuerto", "Alajuela"]),
    # L3: 2 of its 20 trips run Metrópoli - CFIA through Atlántico, over L2's track.
    L("Curridabat - San José - Pavas - Belén",
      [7357210, 6562389, 6562498, 7357207, 7357188, 7357187, 6562922, 7799195],
      {"name_en": "Line 3: Curridabat – San José – Pavas – Belén", "operator": "INCOFER",
       "network": TI, "ref": "L3", "colour": "#245C02"},
      extent=[(SJ_CARTAGO, [ATL, (-84.03704, 9.92530)])],
      lists=["CFIA", "U Latina", "UCR", "Atlántico", "La Corte", "Plaza Víquez", "Pacífico",
             "Barrio Cuba", "Contraloría", "La Salle", "AyA", "Jack's", "Pavas Centro",
             "Pecosa", "Demasa", "Metrópoli", "Pedregal", "Belén"]),
]
CR_ALIAS = {
    "Fátima": "Bulevar Aeropuerto", "Hospital Alajuela": "Alajuela", "Flores": "San Joaquín",
    "ULatina (Lourdes)": "U Latina", "UCR (San Pedro)": "UCR", "CFIA (Curridabat)": "CFIA",
    "UACA (Cipreses)": "UACA", "Plaza Cleto González Víquez": "Plaza Víquez",
    "Tubo Tico (AyA)": "AyA", "Jacks": "Jack's", "Pavas": "Pavas Centro",
    "Pecosa (María Reina)": "Pecosa", "Demasa (Pueblo Nuevo)": "Demasa", "Cuba": "Barrio Cuba",
    "Plaza Paraíso (Llanos de Santa Lucía)": "Plaza Paraíso",
}
CR_RENAME = {
    6562749: {"name": "San José - Heredia - Alajuela", "ref": "L1", "operator": "INCOFER"},
    6559772: {"name": "San José - Cartago", "ref": "L2", "operator": "INCOFER"},
    6563467: {"name": "Curridabat - San José - Pavas - Belén", "ref": "L3",
              "operator": "INCOFER"},
}

# ---------------------------------------------------------------- Panama
# OSM's one route has two unnamed stop nodes, at its two terminals: named here
# (`extra_stations`, at those nodes).
PA = [
    L("Ferrocarril de Panamá", [2020587],
      {"name_en": "Panama Canal Railway (Panamá – Colón)",
       "operator": "Panama Canal Railway Company", "network": "Panama Canal Railway"},
      lists=["Panamá (Corozal)", "Colón (Atlantic Passenger Station)"]),
]
PA_STATIONS = [("Panamá (Corozal)", -79.56743, 8.97607),
               ("Colón (Atlantic Passenger Station)", -79.89972, 9.35017)]

# ---------------------------------------------------------------- Cuba
# The four national trains (UFC, every eight days or so in 2026, cu_sources.md). The Línea
# Central first, so each branch line is the stretch its train adds past the trunk.
CU_TRAINS = (6520980, 6520977, 6520978, 6520976)
CU = [
    L("Línea Central: La Habana – Santiago de Cuba", [6520980],
      {"name_en": "Central Line: Havana – Santiago de Cuba", "operator": "UFC",
       "network": "Ferrocarriles de Cuba"}),
    L("Ramal a Holguín", [6520978],
      {"name_en": "Holguín Branch", "operator": "UFC", "network": "Ferrocarriles de Cuba"}),
    # It leaves the Línea Central east of San Luis, with no station at the junction: a shared
    # extent from San Luis takes in the trunk's few km to it, so the branch's first section
    # starts at a station and is not dropped as track leading to no station.
    L("Ramal a Guantánamo", [6520977],
      {"name_en": "Guantánamo Branch", "operator": "UFC", "network": "Ferrocarriles de Cuba"},
      extent=[(r"", [(-75.80931, 20.18884), (-75.64393, 20.18501)])],
      lists=["San Luis - Combinado"]),
    L("Línea a Bayamo – Manzanillo", [6520976],
      {"name_en": "Bayamo – Manzanillo Line", "operator": "UFC",
       "network": "Ferrocarriles de Cuba"}),
]

# ---------------------------------------------------------------- Peru
# OSM's routes list few stops (Cusco - Machu Picchu only Ollantaytambo and an unnamed node),
# so each line lists the stations its trains call at (pe_sources.md has whose timetable).
CHILCA = (-75.20659, -12.08037)
CUENCA = (-75.03627, -12.42603)      # an unnamed station record; the Tren Macho's terminus
PE = [
    # PeruRail's and Inca Rail's trains; the Urubamba - Pachar branch (the Sacred Valley
    # train) and Machu Picchu - Hidroeléctrica (PeruRail sells it; OSM's routes stop at
    # Machu Picchu: an extent over the Ferrocarril Santa Ana's track).
    L("Ferrocarril Sur Oriente: Cusco – Hidroeléctrica", [5646859, 8275132],
      {"name_en": "Southern Eastern Railway: Cusco – Machu Picchu – Hidroeléctrica",
       "operator": "PeruRail", "network": "Ferrocarril Transandino"},
      # San Pedro - Poroy too: the 2026 Cusco - Ollantaytambo train starts at San Pedro.
      extent=[(r"", [(-72.52397, -13.15585), (-72.55958, -13.17471)]),
              (r"", [(-71.98370, -13.52161), (-72.04234, -13.49463)])],
      lists=["San Pedro", "Poroy", "Huarocondo", "Pachar", "Urubamba", "Ollantaytambo",
             "Piscacucho", "Km. 104.000", "Machu Picchu Pueblo", "Hidroeléctrica"]),
    # PeruRail Titicaca, three a week each way.
    L("Ferrocarril del Sur: Cusco – Puno", [8266111],
      {"name_en": "Southern Railway: Cusco – Puno", "operator": "PeruRail",
       "network": "Ferrocarril Transandino"},
      lists=["Wanchaq", "Juliaca", "Puno"]),
    # The Tren Macho, Mondays and Fridays since Dec 2024, Chilca - Cuenca only.
    # The running line takes the route's ways and lists stations to Cuenca only, so its
    # track past Cuenca leads to no station of its own and is dropped from it; the greyed
    # line takes that stretch as a shared extent from Cuenca.
    L("Ferrocarril Huancayo – Huancavelica (Chilca – Cuenca)", [3986378],
      {"name_en": "Huancayo – Huancavelica Railway (Tren Macho): Chilca – Cuenca",
       "operator": "MTC", "network": "Tren Macho"},
      lists=["Chilca", "Viques", "Chanca", "Retama", "Ingahuasi", "Huarisca",
             "Manuel Tellería", "Cuenca"]),
    # Cuenca - Huancavelica: closed for rebuilding (concession of Aug 2024): greyed.
    L("Ferrocarril Huancayo – Huancavelica (Cuenca – Huancavelica)", [],
      {"name_en": "Huancayo – Huancavelica Railway: Cuenca – Huancavelica",
       "operator": "MTC", "network": "Tren Macho", "suspended": True},
      extent=[(r"", [CUENCA, (-74.96784, -12.78700)])],
      lists=["Cuenca", "Izcuchaca", "Mariscal Cáceres", "Acoria", "Yauli", "Huancavelica"]),
    # Tacna - Arica: no passenger train in 2026 (being rebuilt). Not a line here: Peru's side
    # has one station (Tacna), and a register line needs two; its route is clipped and the
    # track drawn as track. When trains return: a line Tacna - border point, with the point
    # in borders.EXTRA (Chile has the train as a named train meanwhile, rules/cl.py).
]
PE_ALIAS = {"Nueva Estacion por Huancavelica": "Chilca",
            "Estación de Ingahuasi": "Ingahuasi", "Estación de Huarisca": "Huarisca",
            "Machu Picchu Pueblo (Avenida Imperio de los Incas)": "Machu Picchu Pueblo"}

# ---------------------------------------------------------------- Bolivia
BO = [
    # The Expreso del Sur's route has ways only to Uyuni; the old "Uyuni-Villazón" route
    # r3397889 has the rest (kept, a named train like the others). Stations: FCA's 2026
    # ferrobús timetable.
    L("Ferrocarril Oruro – Villazón", [2082775, 3397889],
      {"name_en": "Oruro – Villazón Railway", "operator": "Ferroviaria Andina",
       "network": "Ferroviaria Andina"},
      lists=["Oruro", "Uyuni", "Atocha", "Tupiza", "Villazón"]),
    L("Ferrocarril Viacha – Charaña", [3397885],
      {"name_en": "Viacha – Charaña Railway", "operator": "Ferroviaria Andina",
       "network": "Ferroviaria Andina"},
      lists=["Viacha", "Charaña"]),
    # Stations: the Expreso Oriental's calls (IRJ) and OSM's route.
    L("Ferrocarril Santa Cruz – Puerto Quijarro", [8262149],
      {"name_en": "Santa Cruz – Puerto Quijarro Railway", "operator": "Ferroviaria Oriental",
       "network": "Ferroviaria Oriental"},
      # (Pailón, Aguas Calientes and Puerto Suárez, also calls, have no OSM record.)
      lists=["Santa Cruz de la Sierra", "Cotoca", "San José de Chiquitos", "Roboré",
             "El Carmen Rivero Torrez", "Puerto Quijarro"]),
    # Its first km out of Santa Cruz are the Quijarro line's: a shared extent from Santa Cruz
    # to Charagua, so the line starts at Santa Cruz.
    L("Ferrocarril Santa Cruz – Yacuiba", [8262212, 2084750],
      {"name_en": "Santa Cruz – Yacuiba Railway", "operator": "Ferroviaria Oriental",
       "network": "Ferroviaria Oriental"},
      extent=[(r"", [(-63.16077, -17.78905), (-63.14562, -19.78258)])],
      lists=["Santa Cruz de la Sierra", "Charagua", "Boyuibe", "Villa Montes", "Yacuiba"]),
]

# ---------------------------------------------------------------- Ecuador
EC = [
    L("Tren Nariz del Diablo", [2016146, 2016147],
      {"name_en": "Devil's Nose Train (Alausí – Sibambe)", "operator": "Tren Ecuador",
       "network": "Tren Ecuador"}),
]

# ---------------------------------------------------------------- Uruguay
UY = [
    # Stations: AFE's eight, from OSM's routes, and AFE's 14 request halts named by km post
    # (OSM has each as a railway=halt "km 457" ...), which also check the chainage.
    L("Línea Rivera (Tacuarembó – Rivera)", [9205334, 9209781],
      {"name_en": "Rivera Line (Tacuarembó – Rivera)", "operator": "AFE", "network": "AFE"},
      lists=[f"km {k}" for k in (457, 469, 475, 484, 487, 496, 500, 504, 508, 526, 531, 539,
                                 548, 552)]),
]

# ---------------------------------------------------------------- Venezuela
VE = [
    L("Sistema Ferroviario Ezequiel Zamora: Caracas – Cúa", [4651199, 4651227],
      {"name_en": "Ezequiel Zamora Railway: Caracas – Cúa",
       "operator": "Instituto de Ferrocarriles del Estado", "network": "IFE"}),
]

# ---------------------------------------------------------------- Colombia
# OSM has no passenger route for the Sabana train, and its "Ferrocarril de La Sabana"
# route=railway relation is not in the extract: the line is an extent, the shortest path over
# any track through its stations. It starts at Usaquén: OSM's track stops short of Bogotá's
# Estación de la Sabana (nothing south of Calle 63 joins it), so the first ~15 km cannot be
# traced (co_sources.md).
CO = [
    L("Tren Turístico de la Sabana", [],
      {"name_en": "Sabana Tourist Train (Bogotá – Zipaquirá)", "operator": "Turistren",
       "network": "Turistren"},
      extent=[(r"", [(-74.03803, 4.69063), (-74.02754, 4.85947),
                     (-74.02309, 4.91671), (-74.00115, 5.02164)])],
      lists=["Usaquén", "La Caro", "Cajicá", "Zipaquirá"]),
]
# Medellín's Tranvía de Ayacucho: OSM's two routes list one stop each (the rest are stop_area
# relations of platform ways, which extract.py does not read), so build_model's OSM half drops
# them; built here as a register line on the routes' ways, its stations at the stop_areas'
# platform centroids (read from the 2026-10-08 extract). Miraflores has no stop_area: left out.
CO_TRAM = [("San Antonio", -75.56915, 6.24704), ("San José", -75.56540, 6.24732),
           ("Pabellón del Agua", -75.56200, 6.24558), ("Bicentenario", -75.55877, 6.24394),
           ("Buenos Aires", -75.55389, 6.24140), ("Loyola", -75.54518, 6.23902),
           ("Alejandro Echavarría", -75.54175, 6.23549), ("Oriente", -75.54034, 6.23329)]
CO.append(
    L("Tranvía de Ayacucho", [6491411, 6491412],
      {"name_en": "Ayacucho Tram", "operator": "Metro de Medellín",
       "network": "Metro de Medellín", "ref": "T-A", "colour": "#009933", "kind": "tram"},
      lists=[n for n, _x, _y in CO_TRAM]))
CO_ALIAS = {"Estación Usaquén": "Usaquén", "Estación La Caro": "La Caro",
            "Estación de Tren de Cajicá": "Cajicá", "Estación Zipaquirá": "Zipaquirá"}

CFGS = {
    "cr": {"region": "cr", "prefix": "cr", "label": "CR", "lines": CR,
           "foreign": ("ni", "pa"), "name_alias": CR_ALIAS, "rename": CR_RENAME,
           "drop_route_ids": (6562632, 6562672)},
    "pa": {"region": "pa", "prefix": "pa", "label": "PA", "lines": PA,
           "foreign": ("cr", "co"), "extra_stations": PA_STATIONS,
           "rename": {2020587: {"name": "Ferrocarril de Panamá",
                                "operator": "Panama Canal Railway Company"}}},
    "cu": {"region": "cu", "prefix": "cu", "label": "CU", "lines": CU, "foreign": (),
           "train_routes_only": CU_TRAINS,
           # route=tram "Tren Urbano de Las Tunas": no source shows it running
           "drop_route_ids": (4657180,)},
    "pe": {"region": "pe", "prefix": "pe", "label": "PE", "lines": PE,
           "foreign": ("ec", "co", "br", "bo", "cl"),
           "drop_route_ids": (8277101, 15537427, 8279775),
           "name_alias": PE_ALIAS, "extra_stations": [("Cuenca", *CUENCA)]},
    "bo": {"region": "bo", "prefix": "bo", "label": "BO", "lines": BO,
           "foreign": ("pe", "br", "py", "ar", "cl"),
           "drop_route_ids": (2084702, 7854527, 2084694, 2084708, 9998216, 9931946, 9993924),
           "name_alias": {"Estación Uyuni": "Uyuni", "Estación Atocha": "Atocha",
                          "Estación Cotoca": "Cotoca"}},
    "ec": {"region": "ec", "prefix": "ec", "label": "EC", "lines": EC,
           "foreign": ("co", "pe"),
           "drop_route_ids": (16002309, 2016090, 8277374, 2016178, 9340444, 2016179, 3154014,
                              2016155, 2016158, 2016129, 2016101, 8277400, 16002308, 2016134),
           "rename": {2016150: {"name": "Tren Nariz del Diablo", "operator": "Tren Ecuador"}}},
    "do": {"region": "do", "prefix": "do", "label": "DO", "lines": [], "foreign": ("ht",),
           "drop_way_names": re.compile(r"en construcci[oó]n", re.IGNORECASE)},
    "uy": {"region": "uy", "prefix": "uy", "label": "UY", "lines": UY,
           "foreign": ("ar", "br"),
           "rename": {9209782: {"name": "Línea Rivera (Tacuarembó – Rivera)"}}},
    "ve": {"region": "ve", "prefix": "ve", "label": "VE", "lines": VE,
           "foreign": ("co", "br", "gy"),
           # OSM's two routes (no route_master) take the register line's name, so they
           # group into one OSM line that drops as its twin.
           "rename": {r: {"name": "Sistema Ferroviario Ezequiel Zamora: Caracas – Cúa",
                          "operator": "Instituto de Ferrocarriles del Estado"}
                      for r in (4651199, 4651227)}},
    "co": {"region": "co", "prefix": "co", "label": "CO", "lines": CO,
           "foreign": ("pa", "ve", "br", "pe", "ec"),
           # Bogotá Metro (under construction), the Barrancabermeja ferrobús (not running),
           # an unnamed route=train of 293 ways with no stops (freight corridor)
           "drop_route_ids": (19566688, 13708000, 12478927),
           "name_alias": CO_ALIAS,
           "extra_stations": CO_TRAM},
    "pr": {"region": "pr", "prefix": "pr", "label": "PR", "lines": [], "foreign": ()},
}
for _c in CFGS.values():
    _c.setdefault("keep_routes", ())
    _c.setdefault("drop_routes", re.compile(r"en construcci[oó]n|proyectad", re.IGNORECASE))


def setup(cc):
    core.configure(CFGS[cc])


def build(path, log):
    """ar_register's build. A LINE may list route=railway relations (extract.py keeps them
    in infra.pkl, not rels.pkl) where OSM has no passenger route (Colombia's Sabana train):
    while the register reads, build_model.load adds the CFG's `infra_routes` to the
    relations it returns. Their route=railway is no ROUTE_KIND, so nothing else reads them."""
    cc = Path(path).name
    return with_infra(cc, lambda: core.build(path, log))


def with_infra(cc, fn):
    import build_model as bm
    setup(cc)
    want = set(CFGS[cc].get("infra_routes", ()))
    if not want:
        return fn()
    orig = bm.load

    def load(region, lg):
        ways, rels, stops, cid, cx, cy = orig(region, lg)
        p = ROOT / "data" / "proc" / region / "infra.pkl"
        infra = pickle.load(open(p, "rb")) if p.exists() else {}
        got = {k: v for k, v in infra.items() if k in want}
        lg(f"{cc.upper()}: {len(got)} of {len(want)} route=railway relations read as routes")
        rels = dict(rels)
        rels.update(got)
        return ways, rels, stops, cid, cx, cy
    bm.load = load
    try:
        return fn()
    finally:
        bm.load = orig


def clip(cc, log=print):
    setup(cc)
    cfg = CFGS[cc]
    core.clip(log)
    d = ROOT / "data" / "proc" / cc
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    drop = set(cfg.get("drop_route_ids", ()))
    only = cfg.get("train_routes_only")
    if only is not None:
        drop |= {k for k, (t, _m) in rels.items()
                 if t.get("type") == "route" and t.get("route") == "train" and k not in only}
    gone = sorted(f"{rels[k][0].get('name') or ''} r{k}" for k in drop if k in rels)
    rels = {k: v for k, v in rels.items() if k not in drop}
    # a route_master left with no route goes too
    rels = {k: v for k, v in rels.items()
            if v[0].get("type") != "route_master"
            or any(t == "r" and r in rels for t, r, _ in v[1])}
    log(f"{cc}: {len(gone)} route relations dropped by id: {', '.join(gone) or 'none'}")
    for rid, tags in cfg.get("rename", {}).items():
        if rid in rels:
            t, m = rels[rid]
            rels[rid] = (dict(t, **tags), m)
            log(f"{cc}: r{rid} {t.get('name')!r} retagged {tags}")
    rx = cfg.get("drop_way_names")
    n_w = 0
    if rx is not None:
        before = len(ways)
        ways = {k: v for k, v in ways.items() if not rx.search(v[0].get("name") or "")}
        n_w = before - len(ways)
        log(f"{cc}: {n_w} ways dropped by name ({rx.pattern})")
    for name, obj in (("rels", rels), ("ways", ways)):
        tmp = d / f"{name}.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / f"{name}.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--report", metavar="CC")
    a = ap.parse_args()
    if a.clip:
        clip(a.clip)
    elif a.report:
        with_infra(a.report, core.report)
    else:
        print(__doc__)
