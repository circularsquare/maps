"""Build style.json: OpenFreeMap's dark style, recoloured to match Stadia alidade_smooth_dark,
with every label dropped except street names.  usage: make_style.py dark.json style.json"""
import json, sys

d = json.load(open(sys.argv[1]))

LAND = "rgb(51,51,51)"
WATER = "rgb(34,34,34)"
PARK = "rgb(55,59,55)"
BLDG = "rgb(56,59,53)"
BLDG_LINE = "rgb(72,73,70)"
ROAD_MINOR, CASE_MINOR = "rgb(51,51,51)", "rgb(72,72,72)"
ROAD_MAJOR, CASE_MAJOR = "rgb(45,45,45)", "rgb(72,72,72)"
ROAD_MOTOR = "rgb(45,45,45)"
RAIL = "rgb(70,70,70)"

line = ["LineString", "MultiLineString"]
is_line = ["match", ["geometry-type"], line, True, False]
MAJOR = ["primary", "secondary", "tertiary", "trunk"]


def road_width(a, b, base):
    return ["interpolate", ["exponential", base], ["zoom"], 13, a, 20, b]


layers = []
for l in d["layers"]:
    i = l["id"]
    if l["type"] == "symbol" or i.startswith("boundary") or i.startswith("aeroway") and i != "aeroway-runway":
        continue
    if i in ("landcover_ice_shelf", "landcover_glacier", "landuse_residential", "highway_major_subtle",
             "highway_motorway_subtle", "aeroway-runway", "aeroway-area", "aeroway-taxiway"):
        continue
    layers.append(l)

by = {l["id"]: l for l in layers}
by["background"]["paint"] = {"background-color": LAND}
by["water"]["paint"] = {"fill-antialias": True, "fill-color": WATER}
by["waterway"]["paint"] = {"line-color": WATER}
by["landuse_park"]["paint"] = {"fill-color": PARK}
by["landcover_wood"]["paint"] = {"fill-color": PARK, "fill-opacity": 1}
by["building"]["paint"] = {"fill-antialias": True, "fill-color": BLDG, "fill-outline-color": BLDG_LINE}
by["building"]["minzoom"] = 12
by["road_area_pier"]["paint"] = {"fill-antialias": True, "fill-color": ROAD_MINOR}
by["road_pier"]["paint"]["line-color"] = ROAD_MINOR
by["highway_path"]["paint"] = {"line-color": "rgb(66,66,66)",
                               "line-width": ["interpolate", ["exponential", 1.2], ["zoom"], 13, 0.5, 20, 4]}
# side streets: land-coloured street with a light edge, like Stadia
minor_case = json.loads(json.dumps(by["highway_minor"]))
minor_case["id"] = "highway_minor_casing"
minor_case["paint"] = {"line-color": ["interpolate", ["linear"], ["zoom"], 13, "rgb(62,62,62)", 15.5, CASE_MINOR],
                        "line-width": road_width(4.2, 22, 1.55)}
layers.insert(layers.index(by["highway_minor"]), minor_case)
by["highway_minor"]["paint"] = {"line-color": ROAD_MINOR, "line-width": road_width(1.6, 18, 1.55)}
by["highway_major_casing"]["paint"] = {"line-color": CASE_MAJOR,
                                       "line-width": ["interpolate", ["exponential", 1.3], ["zoom"], 10, 2.4, 20, 23]}
by["highway_major_inner"]["paint"] = {"line-color": ROAD_MAJOR,
                                      "line-width": ["interpolate", ["exponential", 1.3], ["zoom"], 10, 1.6, 20, 20]}
by["highway_motorway_casing"]["paint"]["line-color"] = CASE_MAJOR
by["highway_motorway_inner"]["paint"]["line-color"] = ROAD_MOTOR
for k in ("railway_transit", "railway_minor", "railway"):
    by[k]["paint"]["line-color"] = "rgb(58,58,57)"
for k in ("railway_transit_dashline", "railway_minor_dashline", "railway_dashline"):
    by[k]["paint"]["line-color"] = RAIL

# street names only: arterials larger and lighter, side streets smaller and dimmer (Stadia: about 190 and 160 grey)
src = {"source": "openmaptiles", "source-layer": "transportation_name", "minzoom": 12.5}
txt = ["case", ["has", "name:nonlatin"], ["concat", ["get", "name:latin"], " ", ["get", "name:nonlatin"]],
       ["coalesce", ["get", "name_en"], ["get", "name"]]]
common = {"symbol-placement": "line", "symbol-spacing": 300, "text-field": txt, "text-max-angle": 30,
          "text-pitch-alignment": "viewport", "text-rotation-alignment": "map", "text-transform": "uppercase",
          "text-letter-spacing": 0.02}
names = [
    {"id": "name_minor", "type": "symbol", **src,
     "filter": ["all", is_line, ["!", ["in", ["get", "class"], ["literal", MAJOR + ["motorway"]]]],
                ["!", ["in", ["get", "class"], ["literal", ["rail", "transit", "ferry", "aerialway", "pier"]]]]],
     "layout": {**common, "text-font": ["Noto Sans Regular"],
                "text-size": ["interpolate", ["linear"], ["zoom"], 13, 8, 16, 10.5]},
     "paint": {"text-color": "rgb(160,160,160)", "text-halo-color": "rgb(41,41,41)", "text-halo-width": 1.1}},
    {"id": "name_major", "type": "symbol", **src,
     "filter": ["all", is_line, ["in", ["get", "class"], ["literal", MAJOR + ["motorway"]]]],
     "layout": {**common, "text-font": ["Noto Sans Bold"],
                "text-size": ["interpolate", ["linear"], ["zoom"], 13, 9, 16, 12]},
     "paint": {"text-color": "rgb(190,190,190)", "text-halo-color": "rgb(41,41,41)", "text-halo-width": 1.3}},
]
d["layers"] = layers + names
d.pop("name", None)
d["name"] = "trackdepth basemap"
d["glyphs"] = "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf"
d.pop("sprite", None)
json.dump(d, open(sys.argv[2], "w"), indent=1)
print(len(d["layers"]), "layers:", [l["id"] for l in d["layers"]])
