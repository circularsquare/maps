"""New South Wales: Opal entries and exits per light rail stop (Sydney, Parramatta, Newcastle).

The Station_Type "Light rail" rows of the same Transport for NSW file as the source nsw (its
download folder is nsw's): financial year July 2025 - June 2026, n = (entries + exits) / days
of the months the stop has rows for, the same "less than 50" bands. See nsw.py.

Where a light rail stop and a train station are one id in au (Town Hall, Wynyard, Circular
Quay, Parramatta), the two figures are added, since each is its own mode's taps.
"""
from . import nsw

KEY = "nsw_lr"
CC = "au"
FOLDER = "nsw"
MODES = {"metro", "tram"}       # lines.json "light_rail" is the metro class
COMBINE = "sum"     # records share an id only through FORCE: separate stops on one node
META = {
    "label": "Transport for NSW Opal counts",
    "name": "Train, Metro and Light Rail Station Monthly Usage (Opal), Transport for NSW: "
            "light rail",
    "url": nsw.META["url"],
    "licence": nsw.META["licence"],
    "counts": "Opal entries + exits per day at light rail stops, July 2025 - June 2026 (days "
              "of the months the stop was open), Sydney, Parramatta and Newcastle light rail",
    "note": nsw.META["note"],
}
FORCE = {
    # au's name for the Newcastle stop
    "Honeysuckle": "n5094852701",
    # au has no node for the L2/L3 George Street stops Chinatown and Central Chalmers
    # Street; the nodes beside them carry the L2/L3 lines (Capitol Square, 100 m from
    # Chinatown; Central Grand Concourse, au's light rail stop for Central, 300 m from
    # Chalmers Street), so the two stops' counts are added there
    "Chinatown": "n5692089432",
    "Central Chalmers Street": "n5692089433",
    # Parramatta Square is the Parramatta Light Rail stop beside Parramatta station; au's
    # Parramatta node carries the light rail line
    "Parramatta Square": "n20964687",
    # Haymarket and QVB (L2/L3) have no node near enough. Fish Market's rows end in
    # January 2026 as Bank Street's begin; au has only Bank Street, 0.2 km from where Fish
    # Market stood. Bank Street keeps its own months; Fish Market's earlier ones are not
    # added (another period, perhaps another stop)
    "Fish Market": None,
}


def records(raw):
    return nsw.parse(raw, {"Light rail"})
