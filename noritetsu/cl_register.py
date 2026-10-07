"""Chile: register lines are the track OSM's passenger route relations run over, grouped by
the legal line the track belongs to, read by ar_register's code (kr_register's recipe).

    python cl_register.py --clip        # drop Argentina, Bolivia, Peru and unopened lines from data/proc/cl
    python cl_register.py --report      # each LINE's ways, km and pieces, before any build
    python build_model.py --region cl --register cl_register:data/raw/cl

THE LINES. OSM names Chile's track for its legal line (88% of main-line km: "Línea Central
Sur", "Ramal San Rosendo - Talcahuano", "Ramal Talca-Constitución"), as Korea's is, but almost
all of it is freight or closed: Línea Central Sur runs 1,245 km from Santiago to Puerto Montt
and EFE's passenger trains use three stretches of it. So, as in Argentina, a register line is
the track of the passenger routes listed for it (LINES), named for the legal line and, where
trains use only part of it, the stretch: "Línea Central Sur (Alameda – Chillán)". EFE's
services (Tren Nos, Rancagua, San Fernando, Curicó - Linares, Chillán; Biotren L1, L2 and
Corto Laja) stay OSM lines over those register lines, as Korea's 1호선 does over 경부선. The
Merval's track is named for its service ("Tren Limache – Puerto"), which is the register line's
name too, so OSM's line is its twin. Santiago's Metro stays OSM lines (its route relations
are clean). cl_sources.md has what runs, what does not, and each call.
"""
import re
import sys

import ar_register as core

EFE = "EFE Trenes de Chile"
L = core.L

LINES = [
    L("Línea Central Sur (Alameda – Chillán)",
      [7106219, 7106218, 14219794, 14217248, 17627011, 17627012, 8264359, 14240669,
       17794413, 17794412, 17727736, 17727737],
      {"name_en": "Southern Main Line (Alameda – Chillán)", "operator": "EFE Central",
       "network": EFE}),
    L("Línea Central Sur (Victoria – Pitrufquén)", [8272464, 15951637, 16064569, 12840625],
      {"name_en": "Southern Main Line (Victoria – Temuco – Pitrufquén)", "operator": "EFE Sur",
       "network": EFE}),
    L("Línea Central Sur (Llanquihue – Puerto Montt)", [18446339],
      {"name_en": "Southern Main Line (Llanquihue – Puerto Montt)", "operator": "EFE Sur",
       "network": EFE}),
    # Where one service runs the whole of a stretch, the register line takes the service's
    # name, so OSM's line for it is the register line's twin and is dropped (a second copy
    # of the same line otherwise, the buscarril's measuring 131 km for 88 from its request
    # stops); the legal line is in name_en.
    L("Tren Laja – Talcahuano", [6757915, 6757914, 2170415, 6738372],
      {"name_en": "Corto Laja (Ramal San Rosendo – Talcahuano: Laja – Talcahuano)",
       "operator": "EFE Sur", "network": EFE}),
    L("Biotren Línea 2", [6852288, 2170438],
      {"name_en": "Biotren Line 2 (Ramal Concepción – Curanilahue: Concepción – Coronel)",
       "operator": "Ferrocarriles del Sur S.A.", "network": "Biotren", "ref": "L2"}),
    L("Tren Talca – Constitución", [6855721, 6855722],
      {"name_en": "Talca – Constitución Buscarril (Ramal Talca – Constitución)",
       "operator": "EFE Central", "network": EFE, "colour": "#abc300"}),
    L("Tren Limache – Puerto", [7114437, 1408080],
      {"name_en": "Valparaíso Metro (Limache – Puerto)", "operator": "EFE Valparaíso",
       "network": "Merval", "colour": "#dd3018"}),
]

CFG = {
    "region": "cl",
    "prefix": "l",
    "label": "CL",
    "lines": LINES,
    "foreign": ("ar", "bo", "pe"),
    "keep_routes": (),
    # Lines OSM maps before they open; clip() drops their route relations so build_model
    # does not build them as running: Metro Line 7 (2027), the Alameda - Melipilla train
    # (under construction), Santiago - Batuco (projected), Line 6's and 7's extensions.
    "drop_routes": re.compile(r"en construcci[oó]n|proyectad|Propuesta|Extensi[oó]n Proyectada",
                              re.IGNORECASE),
}


def setup():
    core.configure(CFG)


def build(path, log):
    setup()
    return core.build(path, log)


if __name__ == "__main__":
    setup()
    if "--clip" in sys.argv:
        core.clip()
    elif "--report" in sys.argv:
        core.report()
    else:
        print(__doc__)
