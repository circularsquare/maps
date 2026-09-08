"""
Route colours and draw order for the nycriders poster — a port of the constants
in riders/nycriders/index.html so the poster and the web map agree.

The colours are the MTA's own palette jittered within each trunk group so that
lines running side by side on a shared trunk can be told apart (A/C/E, B/D/F/M,
N/Q/R/W...). Median dE76 off the official colours is about 11, but that jitter
was spent on within-group separation, so the map still reads by group: red is
the 1/2/3, orange is the 6 Av lines, and so on. Anything printed from this needs
the MTA credit line — see posters/NOTES.md.
"""

ROUTE_COLORS = {
    'A': '#007dc5', 'C': '#0064d4', 'E': '#084bdf',
    'B': '#f87700', 'D': '#f06a00', 'F': '#f55700', 'FX': '#f55700',
    'M': '#e15100',
    '4': '#00b33c', '5': '#00b360', '6': '#008f4d', '6X': '#008f4d',
    '1': '#de2b39', '2': '#df4134', '3': '#e0383a',
    '7': '#bb42bd', '7X': '#ba4fc2',
    'N': '#f7d92b', 'Q': '#f7be2b', 'W': '#f3d417', 'R': '#f6b717',
    'J': '#7f582e', 'Z': '#734a29',
    'G': '#6cbe45',
    'L': '#818a91',
    'GS': '#787981', 'FS': '#787981', 'H': '#787981',
    'SI': '#000000',
}

# Higher draws later, i.e. on top. Grouped so a trunk's lines stay together
# rather than each crossing being decided route by route.
LINE_PRIORITY = {
    'SI': 0, 'H': 1, 'FS': 1, 'GS': 1,
    'G': 2, 'J': 3, 'Z': 3, 'L': 4, 'M': 5,
    '5': 6, '6': 6, '6X': 6, '4': 6,
    'A': 7, 'C': 7, 'E': 7,
    'R': 8, 'W': 8,
    'B': 9, 'D': 9,
    'F': 10, 'FX': 10,
    'N': 11, 'Q': 11,
    '7': 13, '7X': 13,
    '1': 14, '2': 14, '3': 14,
}

# Express variants shadow their base route's track and run a handful of trains a
# day; given their own stripe they would overstate their visual weight, so their
# riders are summed into the base route instead.
STATIC_FOLD = {'FX': 'F', '6X': '6', '7X': '7'}

# Ordering within a fanned trunk. Default is the route name, so '4' < '5' < ...
# < 'A' < 'B'; the overrides place a route next to a chosen neighbour — C sorts
# as "A1" so the 8 Av pair A,C stays together and B,D follow, rather than
# A,B,C,D interleaving two different trunks.
ROUTE_SORT_OVERRIDE = {'C': 'A1', 'Z': 'J1'}


def sort_key(route):
    return ROUTE_SORT_OVERRIDE.get(route, route)


def color(route):
    return ROUTE_COLORS.get(route, '#888888')


def priority(route):
    return LINE_PRIORITY.get(route, 0)
