"""Turn published depth figures that build.py could not place into rail heights (m, NAVD88).

Writes evidence/placed_rows.csv   (a figure -> a rail height at a point, on named OSM ways)
       evidence/placed_bounds.csv (a figure that is only a limit)
Reads data/tracks.json (the v1 model: way centrelines every 25 m, ground, z), evidence/riverbed.csv
and evidence/depths.csv. Run after build.py and make_riverbed.py. Every number below that is a
judgement (a datum offset, a structure height) is written beside the row it is used in.

Columns of placed_rows.csv:
  id            depths.csv id(s) joined by "+", the figures that make one anchor
  lon, lat      where the figure applies
  osm_way_ids   space-separated OSM way ids of the tracks it applies to
  rail_navd88_m top of rail, metres NAVD88
  sd_m          one sd: the figure's own spread, what it measures vs the rail, datum, position along the tube
  how           the arithmetic, with every offset
  notes         what the figure measures, checks, contradictions
  radius_m      points of those ways within this distance of (lon, lat) are the ones it pins
  model_z_m     the v1 model's mean rail height at those points (for review, not input)
  model_sd_m    the v1 model's sd there
  confidence    high / medium / low

Datums. NOAA CO-OPS datums API (metric, epoch 1983-2001), MHW and MSL above NAVD88 (station datum minus
the station's NAVD88 value):
  The Battery 8518750      MHW +0.597  MSL -0.063  MLW -0.783
  Queensboro Br 8518687    MHW +0.589  MSL -0.061 (rel. its NAVD88 value -0.002)
  Williamsburg Br 8518699  MHW +0.554
  Dyckman St 8518902       MHW +0.533  MSL -0.029
  Harlem R. Randalls Is. 8518643 has no NAVD88 tie: MHW-MSL 0.685, MSL-MLW 0.704; with MSL -0.05
                           (Queensboro / Dyckman) MHW +0.64, MLW -0.75
  Willets Point 8516990 MHW +1.02 and Kings Point 8516945 MHW +0.99 are 15-20 km east of every tunnel
                           here, where the range is larger; not used.
Sea level rises 2.95 mm/yr at the Battery (NOAA trend 1856-2025, 1.16 in/decade). A figure "below mean
high water" written in year Y refers to the water of that time, so MHW(Y) = MHW(1992) - 0.00295 * (1992 - Y).
Sources written today (Wikipedia, press) use today's MHW and are not shifted.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
FT = 0.3048
SLR = 0.00295           # m/yr
MHW_BATTERY, MSL_BATTERY, MLW_BATTERY = 0.597, -0.063, -0.783
MHW_EAST, MHW_HARLEM, MLW_HARLEM = 0.59, 0.64, -0.75
MSL_EAST = -0.06


def hist(level, year):
    """A tidal level (m NAVD88, 1992 epoch) as it stood in `year`."""
    return level - SLR * (1992 - year)


TR = {t["id"]: t for t in json.loads((ROOT / "data" / "tracks.json").read_text(encoding="utf-8"))["tracks"]}
BED = pd.read_csv(ROOT / "evidence" / "riverbed.csv")


def mpd(lat):
    return 111320 * math.cos(math.radians(lat)), 110574.0


def points(ways, lon, lat, radius):
    kx, ky = mpd(lat)
    out = []
    for w in ways:
        t = TR[w]
        for i in range(len(t["lon"])):
            if math.hypot((t["lon"][i] - lon) * kx, (t["lat"][i] - lat) * ky) <= radius:
                out.append((w, i))
    return out


def stats(ways, lon, lat, radius):
    pts = points(ways, lon, lat, radius)
    if not pts:
        raise SystemExit(f"no points: {ways} {lon} {lat} {radius}")
    g = [TR[w]["ground"][i] for w, i in pts if TR[w]["ground"][i] is not None]
    z = [TR[w]["z"][i] for w, i in pts]
    s = [TR[w]["sd"][i] for w, i in pts]
    return (float(np.mean(g)) if g else float("nan")), float(np.mean(z)), float(np.mean(s)), len(pts)


def wet_centre(ways):
    """Centroid of the points the build treats as wet (ground < 1 m) on these ways."""
    xs, ys = [], []
    for w in ways:
        t = TR[w]
        for i in range(len(t["lon"])):
            g = t["ground"][i]
            if g is None or g < 1.0:
                xs.append(t["lon"][i]); ys.append(t["lat"][i])
    return float(np.mean(xs)), float(np.mean(ys))


def bed_info(ways, lon, lat, radius):
    kx, ky = mpd(lat)
    b = BED[BED["way_id"].isin(ways) & (BED["notes"] == "water")]
    near = b[np.hypot((b["lon"] - lon) * kx, (b["lat"] - lat) * ky) <= radius]
    return (float(near["bed_navd88_m"].median()) if len(near) else float("nan"),
            float(b["bed_navd88_m"].min()) if len(b) else float("nan"))


ROWS = []


def add(id, ways, rail, sd, how, notes, conf, lon=None, lat=None, radius=100, river=False):
    if lon is None:
        lon, lat = wet_centre(ways)
    g, z, s, n = stats(ways, lon, lat, radius)
    if river:
        bmed, bmin = bed_info(ways, lon, lat, radius)
        notes += (f" CUDEM bed here {bmed:.1f} m (deepest on these ways {bmin:.1f} m), so the rail is "
                  f"{bmed - rail:.1f} m below the bed.")
    ROWS.append({"id": id, "lon": round(lon, 6), "lat": round(lat, 6), "osm_way_ids": " ".join(str(w) for w in ways),
                 "rail_navd88_m": round(rail, 1), "sd_m": sd, "how": how, "notes": notes.strip(), "radius_m": radius,
                 "model_z_m": round(z, 1), "model_sd_m": round(s, 1), "confidence": conf})


# ---------------------------------------------------------------- river tunnels
# Joralemon St tunnel (IRT Lexington, Bowling Green - Borough Hall), 1904. 94 ft below MHW at mid-river
# (1904 book; 1912 copy; Wikipedia 91-95). Not said whether this is grade line, invert or tube bottom:
# rail = reference + 1.0 m, sd 1.8 on that alone (grade line = +0, outside bottom of the tube = +2.6).
mhw = hist(MHW_BATTERY, 1904)
ref = mhw - 94 * FT
add("TD0008+TD0045+TD0098", [797157484, 797157483], ref + 1.0, 2.0,
    f"MHW(1904)={mhw:+.2f} (Battery +0.597 - 0.00295*88); 94 ft = {94*FT:.2f} m below it -> {ref:.1f}; +1.0 m to rail; "
    f"sd = sqrt(1.8^2 reference + 0.5^2 position + 0.45^2 range 91-95 ft + 0.2^2 datum)",
    "Deepest point of the river tubes, centre of the river, about 94 ft below MHW (1904, 1912; Wikipedia 91-95 ft). "
    "What point of the tube is not stated. v1 model has this crossing at its -12 m flat rule plus approaches. "
    "Check: 1906 chapter gives river depth 47 ft at high water here = -14.0 m; CUDEM bed -13.8 median, -15.4 deepest. "
    "Agrees.", "medium", radius=120, river=True)

# Clark St tunnel (IRT Broadway-Seventh, Old Slip - Clark St). Bottom of the tunnel 88 ft (1916) / 85 ft (1919)
# below the river surface (tide not given, so mean level with sd 0.4 for the tide).
msl = hist(MSL_EAST, 1917)
ref = msl - 86.5 * FT
add("TD0027+TD0028", [797157480, 797157481], ref + 1.6, 1.3,
    f"MSL(1917)={msl:+.2f}; 86.5 ft (mean of 88 and 85) = {86.5*FT:.2f} m below it -> tube bottom {ref:.1f}; "
    f"+1.6 m to rail (17.5 ft tube, concrete invert); sd = sqrt(0.7^2 conversion + 0.46^2 88 vs 85 ft + 0.4^2 tide + "
    f"0.4^2 position + 0.2^2 datum)",
    "Greatest depth of the Clark St tubes, bottom of the tunnel. 1916 construction news says 88 ft, the 1919 opening "
    "article 85 ft; the later figure is likelier as-built. v1 model has the tunnel at about -12 m, 13 m too shallow.",
    "medium", radius=120, river=True)

# Lexington Av Harlem River tubes (4/5/6), 1915 thesis and 1914 Engineering News: lowest point of the structure
# (subgrade, bottom of the concrete) 57-57.2 ft below MHW. Top of the structure at least 30.7 ft below MHW, built
# as near to that as possible, so the structure is about 26.5 ft (8.1 m) high; rail is about 2.5 m above its bottom.
mhw = hist(MHW_HARLEM, 1915)
ref = mhw - 57.1 * FT
add("TD0020+TD0021+TD0024", [816976503, 816976502, 360153048, 816976501], ref + 2.5, 1.4,
    f"MHW(1915)={mhw:+.2f} (Harlem +0.64 - 0.00295*77); 57.1 ft = {57.1*FT:.2f} m below -> structure bottom {ref:.1f}; "
    f"+2.5 m to rail (bottom of concrete to top of rail in a twin-tube section); sd = sqrt(1.0^2 conversion + "
    f"0.4^2 position + 0.3^2 datum + 0.3^2 figure)",
    "Lowest point of the structure, bottom of the concrete-encased tubes, so below the rail. Cross-check from the top: "
    f"top of structure {mhw - 30.7*FT:.1f} m (30.7 ft below MHW, 'as near this limit as possible'), 5.5 m above the rail "
    "if the structure is 8.1 m high. CONTRADICTION: the same sources put the original river bed at 20-26 ft (avg 23 ft) "
    f"below MHW = {mhw - 23*FT:.1f} m and the structure top 7 ft under it; CUDEM bed is about 2 m shallower than that "
    "(silted or filled since 1915, or CUDEM is shallow here).", "medium", radius=120, river=True)

# Lenox Av / West Farms branch Harlem River tunnel (2/3), 1904: lowest point of the invert 48 ft below MHW.
mhw = hist(MHW_HARLEM, 1904)
ref = mhw - 48 * FT
add("TD0023", [816596212, 816596213, 816596210, 816596211], ref + 1.2, 1.1,
    f"MHW(1904)={mhw:+.2f} (Harlem +0.64 - 0.00295*88); 48 ft = {48*FT:.2f} m below -> invert {ref:.1f}; +1.2 m to rail "
    f"(lowest inside point of a circular tube to top of rail); sd = sqrt(0.6^2 conversion + 0.4^2 position + "
    f"0.3^2 datum + 0.5^2 as-designed)",
    "Lowest point of the invert, first (1904) Harlem crossing of the Lenox Av branch. Corroborated by the same book: "
    "dredged trench 39 ft below low water, about 10 ft above subgrade (TD0005), i.e. subgrade about "
    f"{hist(MLW_HARLEM, 1904) - 49*FT:.1f} m, about 1.6 m under the invert. v1 model is 3-5 m too shallow in the middle.",
    "medium", radius=130, river=True)

# Steinway tunnel (Flushing line, 7), 1907. Top (crown) of the tunnel at the low point 88 ft below MHW and 25 ft
# beneath the river silt (1960 article). Crown to rail in a 15 ft cast-iron tube is about 4.0 m.
mhw = hist(MHW_EAST, 1907)
ref = mhw - 88 * FT
add("TD0040", [797157473, 815757679], ref - 4.0, 1.4,
    f"MHW(1907)={mhw:+.2f} (Queensboro +0.59 - 0.00295*85); 88 ft = {88*FT:.2f} m below -> crown {ref:.1f}; "
    f"-4.0 m to rail (tube crown to top of rail, 15 ft cast-iron tube); sd = sqrt(0.7^2 conversion + 0.6^2 position "
    f"of the low point + 0.3^2 figure + 0.3^2 datum)",
    "Top of tunnel at the low point of the profile. The same sentence puts it 25 ft under the silt, so the silt surface "
    f"there was {ref + 25*FT:.1f} m; CUDEM bed is 1.5-2.5 m shallower than that. A 1960 secondary source. "
    "v1 model has this crossing at about -12 m, 18 m too shallow.", "medium", radius=150, river=True)

# Cranberry St tunnel (IND Eighth Av, 8th Av line A/C). Wikipedia: 90 ft below "the surface" at its lowest point,
# stated beside a 70 ft figure below the street at High St. Water surface (MSL) or street at the Manhattan portal
# (about +2.5 m) are the two readings; take the midpoint of the tube bottom and widen.
msl = hist(MSL_EAST, 1933)
bot_w = msl - 90 * FT
bot_s = 2.5 - 90 * FT
ref = (bot_w + bot_s) / 2
add("TD0095", [157005691, 797157478], ref + 2.0, 2.6,
    f"90 ft = {90*FT:.2f} m: below water surface MSL(1933)={msl:+.2f} -> {bot_w:.1f}; below Manhattan street (+2.5) -> "
    f"{bot_s:.1f}; midpoint {ref:.1f}; +2.0 m to rail (about 23 ft tube); sd = sqrt(1.4^2 surface reading + "
    f"1.0^2 conversion + 0.6^2 position + 0.5^2 figure + 0.3^2 datum)",
    "'The tunnel is 90 feet below the surface at its lowest point' (low confidence Wikipedia sentence, no citation). "
    "Either reading puts the rail at -24 to -27 m. v1 model has this crossing at about -12 m.",
    "low", radius=120, river=True)

# ---------------------------------------------------------------- bridges over the East River
# Williamsburg and Manhattan Bridge: "maximum height 135 / 134 ft above MHW at the middle of the river". This is the
# navigation clearance (underside of the span), NYC DOT's figure. The subway tracks are on the lower deck, about 2 m
# above the underside; widened for the case that the figure is the deck top and the tracks sit lower.
ref = 135 * FT + MHW_EAST
add("TD0126", [180330673, 206583332], ref + 2.0, 4.5,
    f"135 ft = {135*FT:.2f} m above MHW (today's, Williamsburg Br +0.55) -> {ref:.1f}; +2.0 m to rail; sd 4.5 = "
    f"deck vs underside vs track level",
    "Midspan height of the Williamsburg Bridge, 'maximum height above MHW at the middle of the river' (deck about 122 ft "
    "above MHW at either shore). CONTRADICTION: v1 model has the J/M/Z at about 15 m over the water, 28 m lower; the bridge "
    "ends are pinned to low approaches.", "medium", radius=80)
ref = 134 * FT + MHW_EAST
add("TD0125", [46340237, 46346751, 616217282, 817756457, 823986481, 823986482], ref + 2.0, 4.5,
    f"134 ft = {134*FT:.2f} m above MHW (+0.55) -> {ref:.1f}; +2.0 m to rail; sd 4.5",
    "Midspan height of the Manhattan Bridge (tracks on the lower deck; OSM names these ways BMT Brighton Line). "
    "CONTRADICTION: v1 model has these tracks at 9-13 m over the water.", "medium", radius=80,
    lon=wet_centre([46340237, 46346751, 616217282, 817756457, 823986481, 823986482])[0],
    lat=wet_centre([46340237, 46346751, 616217282, 817756457, 823986481, 823986482])[1])

# ---------------------------------------------------------------- absolute datum, land
# 34 St-Hudson Yards: station 125 ft below street and 108 ft below sea level in total.
ref = -108 * FT
lon, lat = -74.00196, 40.75584
g, z, s, n = stats([552272274, 815757675], lon, lat, 100)
add("TD0066", [552272274, 815757675], ref - 1.1, 2.0,
    f"-108 ft = {ref:.2f} m (sea-level datum not named; NGVD29 vs NAVD88 differ about 0.1 m at New York); platform to rail -1.1 m",
    f"'Station' level read as the platform. CONTRADICTION with its own sentence: 125 ft below street and 108 ft below "
    f"sea level put the street at +17 ft = +5.2 m, but the DEM here is {g:.1f} m (the station is under the Hudson Yards deck "
    f"and rail yard). Street-relative gives rail {g - 125*FT - 1.1:.1f} m (the elevator anchor, {z:.1f}); the sea-level "
    "figure gives 4.6 m lower. If the street at the entrance is lower than the DEM, both fit.",
    "medium", lon=lon, lat=lat, radius=100)

# South Ferry old inner loop (1905): station 23 ft below Battery Park, up to 17 ft below MHW.
mhw = hist(MHW_BATTERY, 1905)
ref = mhw - 17 * FT
add("TD0110", [547384540], ref - 1.1, 2.3,
    f"MHW(1905)={mhw:+.2f}; 17 ft = {17*FT:.2f} m below -> station level {ref:.1f}; platform to rail -1.1 m; sd = sqrt(1.5^2 "
    f"'as much as' + 1.0^2 platform vs rail + 0.9^2 position + 0.3^2 datum + 0.3^2 figure)",
    "Station floor of the old loop, 'as much as 17 ft beneath MHW' with Battery Park ground 5-7 ft above MHW. "
    f"CONTRADICTION: v1 model has the loop at about -11 m, 5-6 m too deep (model depth below the DEM about 14 m, "
    f"the source 7 m).", "low", lon=-74.0140, lat=40.7022, radius=150)

# Canal St, BMT Broadway double-deck structure: subgrade of the lower level 40 ft below MHW (1914), excavation 55-60 ft
# deep (1915). The lower level is taken to be the tracks OSM puts at level -3 (the Manhattan Bridge branch).
mhw = hist(MHW_BATTERY, 1914)
ref = mhw - 40 * FT
add("TD0025+TD0155", [817986805, 817986806], ref + 1.2, 2.0,
    f"MHW(1914)={mhw:+.2f}; 40 ft = {40*FT:.2f} m below -> subgrade {ref:.1f}; +1.2 m to top of rail (floor slab, track fixing, rail); "
    f"sd = sqrt(0.6^2 + 1.5^2 which level + 0.4^2 datum + 0.5^2 position)",
    "Subgrade (bottom of structure) of the lower level at Canal St, 'normal ground-water level'. Which level is the lower one "
    "is my reading: the 1914 text builds the lower level for a future cross-town line; today it would be the Manhattan "
    "Bridge branch (55 ft platform, TD0086) or the Nassau St line (40 ft platform, TD0085, -10.3 m). The excavation figure "
    "(55-60 ft below street, datum unknown) gives a subgrade about 2 m lower, which an excavation overshoot would explain.",
    "low", lon=-74.0000, lat=40.71806, radius=100)

# Vesey St reversed curve, BMT Broadway Sec 1-A: bottom of the iron-lined structure about 4 ft below MHW.
mhw = hist(MHW_BATTERY, 1914)
ref = mhw - 4 * FT
add("TD0026", [817415207, 1547538614], ref + 1.6, 1.8,
    f"MHW(1914)={mhw:+.2f}; 4 ft = {4*FT:.2f} m below -> structure bottom {ref:.1f}; +1.6 m to rail (iron tube); sd = sqrt(1.0^2 + "
    f"1.0^2 which route + 0.7^2 + 0.3^2)",
    "'Bottom of the subway structure about 4 ft below MHW' where the material is sand and 'comparatively dry'. Line named only "
    "as Broadway Sec. 1-A; my reading is the BMT Broadway Line near Vesey St. Consistent with v1.",
    "low", lon=-74.0092, lat=40.7117, radius=130)

# ---------------------------------------------------------------- land: rail below / above street
def street_row(id, ways, depth_ft, sd, off, off_text, how_extra, notes, conf, lon, lat, radius=100, up=False):
    g, z, s, n = stats(ways, lon, lat, radius)
    d = depth_ft * FT
    rail = g + d - off if up else g - d - off
    sign = "+" if up else "-"
    add(id, ways, rail, sd,
        f"street {g:.1f} m (DEM along these ways) {sign} {d:.2f} m ({depth_ft:g} ft) {off_text}; {how_extra}", notes, conf,
        lon=lon, lat=lat, radius=radius)


# Jay St / Concord St: excavation 65 ft deep (maximum, where the Rutgers St tubes' rise meets high street levels).
# Excavation bottom to top of rail: invert slab 0.6 + track 0.5 + over-excavation 0.4 = 1.5 m.
ws = [818168015, 818608013]
lon, lat = np.mean([TR[w]["lon"][i] for w, i in points(ws, -73.9870, 40.6975, 100)]), np.mean([TR[w]["lat"][i] for w, i in points(ws, -73.9870, 40.6975, 100)])
street_row("TD0033", ws, 65, 3.0, -1.5, "+ 1.5 m excavation bottom to rail", "sd = sqrt(2^2 conversion/over-dig + 1^2 'below street' assumed + 1.5^2 position of the maximum)",
           "Maximum excavation depth for the Jay St-Smith St subway at Concord and Jay Sts (1934, high). 'Below street' is not "
           "stated. v1 has this stretch 1-2 m higher than this.", "medium", lon, lat, 110)

# 1907 Steinway tunnel terminal at Lexington Av, 156 East 42nd St: platform 65 ft below street.
street_row("TD0041", [46340255, 815757677, 1546629371, 1546629372], 65, 3.5, 1.1, "- 1.1 m platform to rail", "sd 3.5 = 1960 secondary source, address-level location, platform vs rail",
           "Single platform of the 1907 temporary terminal, 65 ft below street (1960 article). The present Flushing line tunnel "
           "there is the same bore; location is the address (156 E 42nd St, between Lexington and Third Avs) put on the 7 line.",
           "low", -73.9744, 40.7516, 130)

# Woodside: 48 ft above street. Rail = platform - 1.1.
street_row("TD0072", [813466487, 813466488, 813466489], 48, 3.5, 1.1, "- 1.1 m platform to rail",
           "sd 3.5 = platform vs rail, street reference in a natural depression",
           "'The station is about 48 feet above street level, above a natural depression along Roosevelt Av' (Wikipedia, cited). "
           "The DEM along the viaduct is the street or the LIRR cut; the figure is a station peak height. v1 puts the rail 4 m lower.",
           "low", -73.9029, 40.7456, 120, up=True)

# Smith-9 Sts: highest station, 88 ft (Untapped, citing the MTA) / 87.5 ft above ground level (Wikipedia).
street_row("TD0052+TD0109", [46341911, 426568714, 426568715, 426568716], 87.75, 3.0, 1.1, "- 1.1 m platform to rail",
           "sd 3.0 = platform vs rail 1.0, ground reference (canal, street, expressway) 2.5, figure 0.3",
           "Station height above ground, 'the highest station in the system'. The track here crosses the Gowanus Canal at the "
           "height of the ship-mast clearance. CONTRADICTION: v1 model has the Culver viaduct at 10.6 m, 18 m lower, because the "
           "ground here is only 3.5 m and the elevated prior is ground + 8 m.", "medium", -73.99722, 40.67444, 120, up=True)

# Van Cortlandt Park-242 St: station 29 ft above the west side of the street.
street_row("TD0116", [326107998, 813505704, 813505715, 824260896, 824260897], 29, 3.0, 1.1, "- 1.1 m platform to rail",
           "sd 3.0 = platform vs rail, which street level",
           "'Station is 29 feet above the west side of the street' (Wikipedia, no citation). v1 has the rail about 6 m lower, at "
           "ground + 1 m.", "low", -73.89833, 40.88889, 100, up=True)

# IRT Broadway-Seventh Clark St branch, 'passing under the Broadway and Park Row subway lines': 60 ft below surface level.
street_row("TD0121", [483762087, 797157463], 60, 4.0, 0.0, "(rail assumed)", "sd 4.0 = 'located' is rail, roof or invert; which stretch",
           "The branch dives under Broadway and Park Row on grades up to 3% south of Park Place; I put the figure on the two OSM "
           "ways that leave Park Place for Fulton St (level -3), the nearest bored-under crossing; the way's own grade is only 0.6% "
           "there, so the 3% stretch is probably further on.", "low", -74.0074, 40.7121, 120)

# Lexington Av/59 St BMT platform (47 ft below street): OSM names the track BMT Astoria Line.
street_row("TD0103", [817383474, 1547538607], 47, 3.5, 1.1, "- 1.1 m platform to rail", "sd 3.5 = Wikipedia no citation",
           "BMT platform on the second basement level, 47 ft below street. Not placed before only because OSM names these ways "
           "BMT Astoria Line.", "low", -73.96790, 40.76247, 100)

# Lexington Av-53 St IND (E/M) platform 70 ft below street. Not at the coordinates of the 51 St IRT station.
ws = [1546629369, 1546629370, 759735515, 819186603]
cx = np.mean([TR[w]["lon"][i] for w, i in points(ws, -73.9693, 40.7575, 150)]); cy = np.mean([TR[w]["lat"][i] for w, i in points(ws, -73.9693, 40.7575, 150)])
street_row("TD0101", ws, 70, 3.5, 1.1, "- 1.1 m platform to rail", "sd 3.5",
           "'It was built 70 feet below street level' (IND Lexington Av/53 St, Wikipedia no citation). Row's coordinates were the "
           "IRT 51 St station, 250 m from the IND platform.", "low", cx, cy, 150)

# Lexington Av/51 St IRT local platform 25 ft below street (the shallow of two OSM levels).
street_row("TD0100", [813712302, 816976506], 25, 3.0, 1.1, "- 1.1 m platform to rail", "sd 3.0",
           "Local platforms 25 ft below street; the express tracks (OSM level -2) run deeper. Consistent with v1.", "low",
           -73.97198, 40.75708, 100)

# High St: station site 70 ft below street (platform or mezzanine not stated).
street_row("TD0094", [46341536, 797157477], 70, 3.5, 1.1, "- 1.1 m platform to rail", "sd 3.5",
           "'Station site constructed 70 ft below the street'; another sentence of the article gives a 70 ft drop from upper fare "
           "control to the lower mezzanine, so this may be the mezzanine and the platform deeper. Consistent with v1.", "low",
           -73.98972, 40.69889, 100)

# Atlantic Av-Barclays Ctr: Brighton platform about 60 ft deep, the deepest of the three levels.
street_row("TD0080", [817912149, 817912150], 60, 3.5, 1.1, "- 1.1 m platform to rail", "sd 3.5 = 'deep' read as below street",
           "Brighton Line platform, 'approximately 60 ft deep, with two mezzanines above', the deepest level. CONTRADICTION: v1 has "
           "the Brighton tracks here at +3.5 m, 3 m ABOVE the Fourth Av tracks, though the Brighton platform is the lowest. "
           "Not placed before only because the ways lie 120-150 m from the complex centre.", "medium", -73.9770, 40.6846, 100)

# ---------------------------------------------------------------- limits
BOUNDS = []


def bound(id, ways, kind, value, sd, how, notes, lon=None, lat=None, radius=120):
    if lon is None:
        lon, lat = wet_centre(ways)
    g, z, s, n = stats(ways, lon, lat, radius)
    BOUNDS.append({"id": id, "lon": round(lon, 6), "lat": round(lat, 6), "osm_way_ids": " ".join(str(w) for w in ways),
                   "bound": kind, "rail_navd88_m": round(value, 1), "sd_m": sd, "how": how, "notes": notes, "radius_m": radius,
                   "model_z_m": round(z, 1)})


mlw = hist(MLW_HARLEM, 1904)
bound("TD0004", [816596212, 816596213, 816596210, 816596211], "max", mlw - 20 * FT - 4.1, 1.5,
      f"MLW(1904)={mlw:+.2f}; roof at or below 20 ft under it = {mlw - 20*FT:.1f}; rail at least 4.1 m under the roof",
      "War Department minimum: 20 ft of water over the roof at low tide (Harlem, Lenox branch). The roof was set at that limit, "
      "so the rail is at or below this.")
mlw = hist(MLW_BATTERY, 1906)
bound("TD0017", [797157484, 797157483], "max", mlw - 45 * FT - 4.0, 1.5,
      f"MLW(1906)={mlw:+.2f}; crown at or below 45 ft under it = {mlw - 45*FT:.1f}; rail 4.0 m under the crown",
      "Joralemon tubes: at least 45 ft of water over the top of the tunnel at low tide.")
bound("TD0071", [351855720, 817383473], "max", -100 * FT - 0.2 + 2.0, 3.0,
      f"'over 100 ft deep in sections' read as 100 ft below the water surface: bottom {-100*FT-0.2:.1f}, rail 2.0 m above",
      "60th St tunnel, East River, 'over 100 feet deep in sections', no reference level (low confidence Wikipedia sentence). "
      "If it means below the water surface the v1 model (about -12 m) is far too shallow; if it means below the street it "
      "says little under the river.")

out = pd.DataFrame(ROWS)
out.to_csv(ROOT / "evidence" / "placed_rows.csv", index=False)
pd.DataFrame(BOUNDS).to_csv(ROOT / "evidence" / "placed_bounds.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(out[["id", "rail_navd88_m", "sd_m", "model_z_m", "model_sd_m", "radius_m", "confidence"]].to_string())
print(pd.DataFrame(BOUNDS)[["id", "bound", "rail_navd88_m", "model_z_m"]].to_string())
