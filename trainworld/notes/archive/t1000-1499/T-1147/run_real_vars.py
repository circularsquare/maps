B = dict(road_open_kmh=110.0, road_city_kmh=16.5, road_town_kmh=44.0, road_far_kmh=88.0)


def scaled(f, keys=None):
    return {k: v * f for k, v in B.items() if keys is None or k in keys}


VARS = {
    "base": {},
    "x1.1": scaled(1.1),
    "x1.2": scaled(1.2),
    "far+open x1.2": scaled(1.2, ["road_far_kmh", "road_open_kmh"]),
    "town+far x1.2": scaled(1.2, ["road_town_kmh", "road_far_kmh"]),
    "city x1.2": scaled(1.2, ["road_city_kmh"]),
}
