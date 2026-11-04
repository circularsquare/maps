"""Published depths for the rail lines added after the subway (PATH, Amtrak, LIRR, Metro-North, AirTrain),
turned into rail heights (m, NAVD88).

Writes evidence/placed_rows_rail.csv   same columns as placed_rows.csv (ids RL0001...)
       evidence/placed_bounds_rail.csv same columns as placed_bounds.csv     (ids RB0001...)
build.py reads both (it globs placed_rows*.csv and placed_bounds*.csv).
Reads data/tracks.json, raw/osm_tracks.json (operator, level, name), evidence/riverbed.csv.
Run after build.py and make_riverbed.py. Datums and every offset are written beside the row they are used in
(see the header of make_placed_rows.py for the NOAA values; MHW(Y) = MHW(1992) - 0.00295 * (1992 - Y)).

A way list is chosen by rule (operator, tunnel, level, distance from the point), so it follows the OSM ids in
raw/osm_tracks.json rather than a hand-typed list.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
FT = 0.3048
SLR = 0.00295
MHW_BATTERY, MSL_BATTERY, MLW_BATTERY = 0.597, -0.063, -0.783
# Hell Gate has no NOAA NAVD88 tie here: between Queensboro Br (+0.59) and Willets Point (+1.02): +0.75
MHW_HELLGATE = 0.75


def hist(level, year):
    return level - SLR * (1992 - year)


TR = {t["id"]: t for t in json.loads((ROOT / "data" / "tracks.json").read_text(encoding="utf-8"))["tracks"]}
OSM = {e["id"]: e["tags"] for e in json.loads((ROOT / "raw" / "osm_tracks.json").read_text(encoding="utf-8"))["elements"]}
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


def sel(lon, lat, radius, ops, cls="tunnel", level=None, name=None, wet=None):
    """OSM way ids in tracks.json of class `cls` with a point within radius of (lon, lat), operator containing one of
    `ops`, optionally one OSM level, a name fragment, and (wet True/False) a point the build treats as wet."""
    kx, ky = mpd(lat)
    out = []
    for wid, t in TR.items():
        if t["cls"] != cls:
            continue
        tg = OSM.get(wid, {})
        if ops and not any(o in tg.get("operator", "") for o in ops):
            continue
        if level is not None and t["level"] != level:
            continue
        if name and name not in t["name"]:
            continue
        near = [i for i in range(len(t["lon"]))
                if math.hypot((t["lon"][i] - lon) * kx, (t["lat"][i] - lat) * ky) <= radius]
        if not near:
            continue
        if wet is not None:
            isw = any(t["ground"][i] is None or t["ground"][i] < 1.0 for i in near)
            if isw != wet:
                continue
        out.append(wid)
    return sorted(out)


def stats(ways, lon, lat, radius):
    pts = points(ways, lon, lat, radius)
    if not pts:
        raise SystemExit(f"no points: {ways} {lon} {lat} {radius}")
    g = [TR[w]["ground"][i] for w, i in pts if TR[w]["ground"][i] is not None]
    z = [TR[w]["z"][i] for w, i in pts]
    s = [TR[w]["sd"][i] for w, i in pts]
    return (float(np.mean(g)) if g else float("nan")), float(np.mean(z)), float(np.mean(s)), len(pts)


def bed_info(ways, lon, lat, radius):
    kx, ky = mpd(lat)
    b = BED[BED["way_id"].isin(ways) & (BED["notes"] == "water")]
    near = b[np.hypot((b["lon"] - lon) * kx, (b["lat"] - lat) * ky) <= radius]
    return (float(near["bed_navd88_m"].median()) if len(near) else float("nan"),
            float(b["bed_navd88_m"].min()) if len(b) else float("nan"))


ROWS, BOUNDS = [], []


def add(id, ways, lon, lat, radius, rail, sd, how, notes, conf, river=False):
    g, z, s, n = stats(ways, lon, lat, radius)
    if river:
        bmed, bmin = bed_info(ways, lon, lat, radius)
        notes += (f" CUDEM bed here {bmed:.1f} m (deepest on these ways {bmin:.1f} m), so the rail is "
                  f"{bmed - rail:.1f} m below the bed.")
    ROWS.append({"id": id, "lon": round(lon, 6), "lat": round(lat, 6), "osm_way_ids": " ".join(str(w) for w in ways),
                 "rail_navd88_m": round(rail, 1), "sd_m": sd, "how": how, "notes": notes.strip(), "radius_m": radius,
                 "model_z_m": round(z, 1), "model_sd_m": round(s, 1), "confidence": conf})


def bound(id, ways, lon, lat, radius, value, sd, how, notes):
    g, z, s, n = stats(ways, lon, lat, radius)
    BOUNDS.append({"id": id, "lon": round(lon, 6), "lat": round(lat, 6), "osm_way_ids": " ".join(str(w) for w in ways),
                   "bound": "max", "rail_navd88_m": round(value, 1), "sd_m": sd, "how": how, "notes": notes, "radius_m": radius,
                   "model_z_m": round(z, 1)})


# ---------------------------------------------------------------- Pennsylvania Station
# Raymond 1910 (ASCE Trans. LXVIII, paper 1150, Gutenberg #18229): "The level of the track system below the street
# surface varies from 39 to 58 ft., and is from 7 to 10 ft. below mean high water in the harbor".
# Midpoint 8.5 ft; "track system" taken as top of rail (a platform would be 1.1 m higher, a track slab 0.5 m lower).
mhw = hist(MHW_BATTERY, 1910)
ref = mhw - 8.5 * FT
lon, lat, rad = -73.9950, 40.7505, 300
ws = sel(lon, lat, rad, ["Amtrak", "Long Island"], "tunnel")
add("RL0001", ws, lon, lat, rad, ref, 1.4,
    f"MHW(1910)={mhw:+.2f} (Battery +0.597 - 0.00295*82); 8.5 ft (mid of 7-10) = {8.5*FT:.2f} m below it -> {ref:.1f}; "
    f"'track system' = top of rail (+0); sd = sqrt(0.46^2 range 7-10 ft + 1.0^2 rail vs platform vs slab + 0.8^2 along the "
    f"station (39 to 58 ft below the street) + 0.3^2 datum)",
    "Raymond 1910, 'The New York Tunnel Extension of the Pennsylvania Railroad: general description', ASCE Transactions LXVIII "
    "(gutenberg.org/ebooks/18229): 'The level of the track system below the street surface varies from 39 to 58 ft., and is "
    "from 7 to 10 ft. below mean high water in the harbor, thereby necessitating the establishment of an elaborate system of "
    "drainage'. Station box 7th to 9th Av, 31st to 33rd St. Not the figure the elevator chain gave for the subway station "
    "next door. The same ways carry Amtrak, NJ Transit and LIRR trains.", "medium")

# ---------------------------------------------------------------- Weehawken shaft (North River tunnels, west bank)
# Lavis 1910 (ASCE Trans. LXVIII, Bergen Hill tunnels, Gutenberg #21083): "The elevation of the top of the rail at the
# Weehawken Shaft ..., on the west bank of the Hudson River, is about 64 ft. below mean high water". Wikipedia (North River
# Tunnels): the shaft was "a few hundred yards west of the river's western shore". Shore (bulkhead) at -74.0205.
ref = mhw - 64 * FT
lon, lat, rad = -74.0235, 40.7633, 150
ws = [316621098, 32965981]
add("RL0002", ws, lon, lat, rad, ref, 3.0,
    f"MHW(1910)={mhw:+.2f}; 64 ft = {64*FT:.2f} m below it -> {ref:.1f} (top of rail, as stated); sd = sqrt(0.5^2 'about' + "
    f"2.5^2 position of the shaft along a 1.3-2% grade (a few hundred yards from the shore) + 0.3^2 datum)",
    "Lavis 1910, ASCE Transactions LXVIII (gutenberg.org/ebooks/21083): 'The elevation of the top of the rail at the Weehawken "
    "Shaft ... on the west bank of the Hudson River, is about 64 ft. below mean high water; and at the Western Portal, or "
    "Hackensack end, the rail is about 17 ft. above; the grade throughout is 1.3%, ascending from east to west.' These are the "
    "Bergen Hill tubes; the North River tubes continue east from this shaft. Position is read off 'a few hundred yards west "
    "of the shore', so it is soft.", "medium")

# ---------------------------------------------------------------- Hell Gate Bridge
# Wikipedia (Hell Gate Bridge), citing Railway Age Gazette 1914 p. 1368: "the deck reaches its maximum height, 145 feet above
# mean high water"; clearance below 135 ft (RAG 1914 p. 891; Ammann). Two readings: deck 145 ft (taken as the track grade),
# and 135 ft + 2.5 m underside to rail (the figure used for the Manhattan and Williamsburg Bridges).
mhw_hg = hist(MHW_HELLGATE, 1914)
r1 = 145 * FT + mhw_hg
r2 = 135 * FT + mhw_hg + 2.5
lon, lat, rad = -73.9219, 40.78235, 130
ws = sel(lon, lat, rad, ["Amtrak"], "elevated")
add("RL0003", ws, lon, lat, rad, (r1 + r2) / 2, 1.5,
    f"MHW(1914)={mhw_hg:+.2f} (Hell Gate +0.75 interpolated between Queensboro +0.59 and Willets Point +1.02, less 0.00295*78); "
    f"deck 145 ft = {145*FT:.1f} m -> {r1:.1f}; clearance 135 ft = {135*FT:.1f} m + 2.5 m underside to rail -> {r2:.1f}; mean; "
    f"sd = sqrt(1.0^2 deck vs rail + 0.5^2 two readings + 0.5^2 datum + 0.8^2 position along the span)",
    "Main span of the Hell Gate Bridge. Wikipedia: 'the deck reaches its maximum height, 145 feet (44 m) above mean high water' "
    "(footnote Railway Age Gazette 1914 p. 1368) and 'with a clearance below of 135 feet' (RAG 1914 p. 891; Ammann p. 1661). "
    "The arch crown is 305 ft above MHW. Check against the lidar rail on the same ways (structure_heights.csv).", "medium")

# ---------------------------------------------------------------- PATH (Hudson & Manhattan) river tubes
# Fitzherbert 1964 (Electric Railroaders' Association; nycsubway.org 'The Public Be Pleased'): the uptown "river tubes were
# 5,650 feet long between shafts and reached a maximum depth of 97 feet below mean sea level"; the downtown tubes "were built
# through mud and rock at a depth of 60 to 90 feet below mean low water level ... lying 15 to 40 feet below the bottom of the
# river". The reference point (invert, grade line, tube bottom) is not said. Tube outside diameter 15 ft 3 in: rail about 1.0 m
# above the tube bottom's inside, 1.6 m above the outside bottom; take +1.0 +- 1.0.
msl = hist(MSL_BATTERY, 1960)
ref = msl - 97 * FT
lon, lat, rad = -74.0200, 40.7313, 250
ws = sel(lon, lat, rad, ["Trans-Hudson", "Port Authority"], "tunnel", wet=True)
ws = [w for w in ws if TR[w]["level"] in (-2, -3)]
add("RL0004", ws, lon, lat, rad, ref + 1.0, 2.5,
    f"MSL(1960)={msl:+.2f} (Battery -0.063 - 0.00295*32); 97 ft = {97*FT:.2f} m below it -> {ref:.1f}; +1.0 m to rail "
    f"(reference taken as grade line to tube bottom); sd = sqrt(1.0^2 which point of the tube + 2.0^2 where the maximum lies "
    f"along 1.7 km of river tube + 1.0^2 mean sea level of the 1960s article + 0.5^2 as-built)",
    "Uptown Hudson tubes (Hoboken to Christopher St), maximum depth. Fitzherbert 1964: 'The river tubes were 5,650 feet long "
    "between shafts and reached a maximum depth of 97 feet below mean sea level.' Wikipedia gives 97 ft (29.6 m) 'below river "
    "level'. Secondary source, point not stated. Cross-check: river bottom about -16 m here and the tubes 'lie 15 to 40 feet "
    "below the bottom of the river' gives -21 to -28 m.", "low", river=True)

mlw = hist(MLW_BATTERY, 1960)
ref = mlw - 90 * FT
lon, lat, rad = -74.0190, 40.7128, 250
ws = sel(lon, lat, rad, ["Trans-Hudson", "Port Authority"], "tunnel", wet=True)
ws = [w for w in ws if TR[w]["level"] in (-2, -3)]
add("RL0005", ws, lon, lat, rad, ref + 1.0, 3.0,
    f"MLW(1960)={mlw:+.2f} (Battery -0.783 - 0.00295*32); the deepest of '60 to 90 ft' = 90 ft = {90*FT:.2f} m below it -> {ref:.1f}; "
    f"+1.0 m to rail; sd = sqrt(1.0^2 which point + 2.5^2 the 90 ft is a range end, location along the tubes unknown + "
    f"1.0^2 datum + 0.5^2 as-built)",
    "Downtown Hudson tubes (Exchange Place to World Trade Center), deepest stretch. Fitzherbert 1964: 'They were built through mud "
    "and rock at a depth of 60 to 90 feet below mean low water level' and 'lying 15 to 40 feet below the bottom of the river'. "
    "Taken as the deepest 90 ft at mid-river. Secondary source.", "low", river=True)

# Exchange Place: "platform level, located 75 feet beneath street level" (Jersey Journal 12 Sept 1989 via Wikipedia).
lon, lat, rad = -74.0336, 40.7169, 110
ws = sel(lon, lat, rad, ["Trans-Hudson", "Port Authority"], "tunnel", wet=False)
g, z, s, n = stats(ws, lon, lat, rad)
rail = g - 75 * FT - 1.1
add("RL0006", ws, lon, lat, rad, rail, 3.0,
    f"street {g:.1f} m (DEM along these ways) - {75*FT:.2f} m (75 ft) - 1.1 m platform to rail; sd 3.0 = platform vs rail 1.0, "
    f"street reference 1.5, 'beneath street level' of a 1989 press line 2.5",
    "Exchange Place PATH platform, Jersey City: 'provide access to the platform level, located 75 feet (23 m) beneath street "
    "level' (Stapinski, The Jersey Journal, 12 Sep 1989, as cited by Wikipedia). A 1994 Pathways piece has a lower passage 63 ft "
    "down with an elevator 'the short distance' to the platform, which fits. The way list is every PATH tunnel track within "
    "110 m, so it also catches tunnels that bend away from the platform.", "low")

# ---------------------------------------------------------------- East Side Access / Grand Central Madison
# amNY: "the lowest tracks run about 150 feet below street level" (caves-grand-central-east-side-access-project-track).
# Wikipedia (East Side Access) lead: "a two-level station 100 feet (30.5 m) below street level", and "tunnels with an average
# depth of 140 feet beneath street level". Other reports run 140 to 180 ft. OSM levels: tracks 301-304 are -5, 201-204 are -3.
# Revised 2026-11-03 from the mezzanine. Wikipedia (Grand Central Madison): "A mezzanine sits on a center level between
# the LIRR's two track levels", "more than 140 ft (43 m) deep below Park Avenue" (infobox depth 140 ft), reached by
# escalators that "descend more than 90 ft (27 m)" from the concourse; each cavern has "two tracks on one island platform
# per level". So the upper platforms are one flight over the mezzanine floor and the lower one flight under it. A level
# here is about 6.5 m floor to floor (mezzanine headroom and slab over the upper platform; the lower level's train
# clearance and slab under the mezzanine): upper rail about 43 - 5.5 = 37.5 m down (platform 1.1 m over the rail),
# lower rail about 43 + 6.5 = 49.5 m; the amNY "about 150 feet" (45.7 m) for the lowest tracks sits between.
# The old upper-level row read Wikipedia's "a two-level station 100 feet below street level" (30.5 m), which can't hold
# with a mezzanine at 43 m between the levels.
MEZZ = 140 * FT
lon, lat, rad = -73.9765, 40.7547, 150
ws = sel(lon, lat, rad, ["Long Island"], "tunnel", level=-5, name="Grand Central")
g, z, s, n = stats(ws, lon, lat, rad)
lower = (MEZZ + 6.5 + 150 * FT) / 2
rail = g - lower
add("RL0007", ws, lon, lat, rad, rail, 3.5,
    f"street {g:.1f} m (DEM along these ways, Park Av between 44th and 48th St) - {lower:.1f} m: the mean of the mezzanine "
    f"{MEZZ:.1f} m (140 ft) + 6.5 m (one level down to the rail) and amNY's 150 ft ({150*FT:.1f} m); sd 3.5 = the two "
    f"readings 2 m apart, 'more than 140 ft', level height 1.5 m, street reference 1.5 m",
    "Lower (tracks 301-304) level of Grand Central Madison. Wikipedia: 'A mezzanine sits on a center level between the LIRR's "
    "two track levels', 'more than 140 ft (43 m) deep below Park Avenue'. amNY: 'the lowest tracks run about 150 feet below "
    "street level'. Other reports: 160, 170, 180 ft.", "medium")

ws = sel(lon, lat, rad, ["Long Island"], "tunnel", level=-3, name="Grand Central")
g, z, s, n = stats(ws, lon, lat, rad)
upper = MEZZ - 5.5
rail = g - upper
add("RL0008", ws, lon, lat, rad, rail, 3.0,
    f"street {g:.1f} m (DEM) - {upper:.1f} m: the mezzanine {MEZZ:.1f} m (140 ft) less one level up to the upper platforms "
    f"(6.5 m floor to floor, platform 1.1 m over the rail); sd 3.0 = 'more than 140 ft', level height 1.5 m, street 1.5 m",
    "Upper (tracks 201-204) level of Grand Central Madison, from the mezzanine between the two track levels, 'more than 140 ft "
    "(43 m) deep below Park Avenue' (Wikipedia, Grand Central Madison). Replaces the 2026-11-02 reading of 'a two-level "
    "station 100 feet below street level' (30.5 m), which does not fit a mezzanine at 43 m between the levels.", "medium")

# ---------------------------------------------------------------- PATH World Trade Center (2026-11-03)
# Wikipedia (World Trade Center station (PATH)): "The station has five tracks served by three island platforms and one side
# platform in a basement four stories underground", built "four levels underground, along the base of the World Trade Center
# 'bathtub'". The bathtub (slurry wall to bedrock) is about 70 ft deep. Platform about 21 m down (four stories of 5.3 m, or the
# bathtub floor), rail 1.1 m under it.
lon, lat, rad = -74.0118, 40.7117, 120
ws = sel(lon, lat, rad, ["Trans-Hudson", "Port Authority"], "tunnel")
g, z, s, n = stats(ws, lon, lat, rad)
rail = g - 21.3 - 1.1
add("RL0013", ws, lon, lat, rad, rail, 2.5,
    f"street {g:.1f} m (DEM along these ways) - 21.3 m (platform at the base of the 70 ft bathtub, four stories down) - 1.1 m "
    f"platform to rail; sd 2.5 = stories 4.5-6 m, the bathtub floor's depth, street reference",
    "PATH World Trade Center: 'five tracks served by three island platforms and one side platform in a basement four stories "
    "underground', 'four levels underground, along the base of the World Trade Center bathtub' (Wikipedia, World Trade Center "
    "station (PATH)). The model had the loop about 32 m down (rail near -29 m), deeper than the bathtub.", "medium")

# ---------------------------------------------------------------- East River tunnels (PRR), two pinned places
# Noble 1910 (ASCE LXVIII, paper 1152, Gutenberg #18065): "Eastward from the sumps the tunnels had a rising grade of 0.7% to the
# established bulkhead line on the Long Island side, giving a cover at the points where the tunnels enter rock, a short distance
# westward, of about 10 ft. (if the dredging plane should be fixed at some future time at 40 ft. below mean low tide ...)".
# Crown = 40 + 10 = 50 ft below mean low water; rail 5.5 m under the outside crown (Hewett and Brown, see RB0001 note).
ref = mlw10_ = hist(MLW_BATTERY, 1910) - 50 * FT
lon, lat, rad = -73.9625, 40.74208, 120
ws = sel(lon, lat, rad, ["Amtrak", "Long Island"], "tunnel", wet=True)
add("RL0009", ws, lon, lat, rad, ref - 5.5, 2.5,
    f"MLW(1910)={hist(MLW_BATTERY, 1910):+.2f}; crown 40 ft (future dredging plane) + 10 ft cover = 50 ft = {50*FT:.1f} m below it -> {ref:.1f}; "
    f"rail 5.5 m under the crown; sd = sqrt(1.5^2 'about 10 ft' and 'a short distance westward' along a 0.7% grade + 1.5^2 the "
    f"plane is a plan not a survey + 1.0^2 crown to rail + 0.4^2 datum)",
    "Long Island City end of the East River tunnels, where they come out of the river bed into rock west of the bulkhead (Noble "
    "1910, 'The East River Division', gutenberg.org/ebooks/18065): 'giving a cover at the points where the tunnels enter rock, a "
    "short distance westward, of about 10 ft. (if the dredging plane should be fixed at some future time at 40 ft. below mean low "
    "tide, as may be reasonably anticipated)'. Position read as 100 m west of the bulkhead shore.", "low", river=True)

# Noble 1910: at the Manhattan shafts (just east of First Av, 34th St) "the rock surface was several feet above the level of the top
# of the tunnels"; Brace et al. 1910 (paper 1159, Gutenberg #18722): the cutting edges stand on rock "31 ft. below mean high water"
# (south shaft) and "38 ft." (north shaft). Crown = rock - several (5 +- 3) ft.
mhw05 = hist(MHW_BATTERY, 1905)
ref = mhw05 - 34.5 * FT - 5 * FT
lon, lat, rad = -73.9735, 40.7434, 100
ws = sel(lon, lat, rad, ["Amtrak", "Long Island"], "tunnel")
add("RL0010", ws, lon, lat, rad, ref - 5.5, 3.0,
    f"MHW(1905)={mhw05:+.2f}; rock under the caisson edges 31 and 38 ft (mean 34.5) below MHW; crown 5 ft (several) under the rock "
    f"surface = {34.5*FT + 5*FT:.1f} m below MHW -> {ref:.1f}; rail 5.5 m under the crown; sd = sqrt(2.0^2 'several feet' + "
    f"1.5^2 rock surface varies across the 74 ft x 40 ft shaft + 1.0^2 crown to rail + 1.0^2 position along the grade)",
    "Manhattan (First Av) shafts of the East River tunnels. Noble 1910: 'the rock surface was several feet above the level of the "
    "top of the tunnels'. Brace, Mason and Woodard 1910: the south caisson 'finally came to rest 31 ft. below mean high water', "
    "the north one found sound rock 'at a depth of 38 ft. below mean high water'. Rock surface read as the crown plus about 5 ft.",
    "low")

# Noble 1910 (paper 1152, Gutenberg #18065), the Manhattan section east of the station: "a grade of 0.8% in 33d Street and 0.9%
# in 32d Street was substituted for the original 0.4% grade", descending eastward from the station, and "From Fifth Avenue
# eastward to the lowest point under the river, the grade was to be 1.5% on all lines" (with 0.3% links near Madison Av). The
# station tracks (RL0001, -2.2) are level to the east end of the station box at Seventh Av; the grade taken as 0.85% from there.
# Added 2026-11-02: the model had these tunnels about 1 m above NAVD88 at Sixth Av, carrying Penn Station's shallowness east along
# the 1.5 km prior-error chain, which put them over the Sixth Av subway that they pass under (crossing rule, build.py).
for rid, lon, lat, dist in (("RL0011", -73.98855, 40.74880, 280.0), ("RL0012", -73.98595, 40.74810, 560.0)):
    ws = sel(lon, lat, 70, ["Amtrak", "Long Island"], "tunnel", name="Terminal")
    rail = -2.2 - 0.0085 * dist
    add(rid, ws, lon, lat, 70, rail, 1.8,
        f"Penn Station track level -2.2 (RL0001) at the east end of the station box (Seventh Av), then 0.85% (0.8% under 33rd St, "
        f"0.9% under 32nd St) for {dist:.0f} m east -> {rail:.1f}; sd = sqrt(1.4^2 RL0001 + 0.9^2 where the descent starts (+-100 m) "
        f"+ 0.6^2 which grade, which street)",
        "East River tunnels under 32nd and 33rd St, at " + ("Sixth Av" if dist < 400 else "Fifth Av") + ". Noble 1910, 'The East "
        "River Division' (gutenberg.org/ebooks/18065): 'a grade of 0.8% in 33d Street and 0.9% in 32d Street was substituted for "
        "the original 0.4% grade' east of the station, and 'From Fifth Avenue eastward to the lowest point under the river, the "
        "grade was to be 1.5% on all lines'. The profile itself (Plate XIII) is not in the text.", "medium")

# ---------------------------------------------------------------- limits (river tunnels)
# Raymond 1910: tunnels placed "at sufficient depth below the dredging plane of the War Department (which in the North and
# East Rivers is 40 and 26 ft. below mean low water, respectively) to insure them against possible injury from heavy anchors".
# Crown at or below the dredging plane; 23 ft outside diameter tube: top of rail is 17 ft 1 in below the inner edge of the
# iron flange and 5 ft above the outside of the iron at the bottom, flange 11 in deep (Hewett and Brown 1910, ASCE LXVIII,
# gutenberg.org/ebooks/42149), so the outside crown is 18.0 ft = 5.5 m above the rail.
mlw10 = hist(MLW_BATTERY, 1910)
lon, lat, rad = -74.0137, 40.7578, 520
ws = [w for w in sel(lon, lat, rad, ["Amtrak"], "tunnel", wet=True)]
bound("RB0001", ws, lon, lat, rad, mlw10 - 40 * FT - 5.5, 1.5,
      f"MLW(1910)={mlw10:+.2f}; dredging plane 40 ft below = {mlw10 - 40*FT:.1f}; crown at or below it; rail 5.5 m under the crown",
      "North River (PRR) tunnels: placed below the War Department's 40 ft dredging plane (Raymond 1910, gutenberg.org/ebooks/"
      "18229). The tube is 23 ft outside diameter.")
lon, lat, rad = -73.9655, 40.7455, 380
ws = [w for w in sel(lon, lat, rad, ["Amtrak", "Long Island"], "tunnel", wet=True)]
if ws:
    bound("RB0002", ws, lon, lat, rad, mlw10 - 26 * FT - 5.5, 1.5,
          f"MLW(1910)={mlw10:+.2f}; dredging plane 26 ft below = {mlw10 - 26*FT:.1f}; crown at or below it; rail 5.5 m under the crown",
          "East River (PRR) tunnels: placed below the War Department's 26 ft dredging plane (Raymond 1910). 23 ft outside "
          "diameter tubes. (Noble 1910 adds that the river bottom is about 60 ft below mean high tide at its deepest, west of "
          "the reef, and that a later 40 ft plane 'may be reasonably anticipated' with about 10 ft of cover.)")

out = pd.DataFrame(ROWS)
out.to_csv(ROOT / "evidence" / "placed_rows_rail.csv", index=False)
pd.DataFrame(BOUNDS).to_csv(ROOT / "evidence" / "placed_bounds_rail.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(out[["id", "rail_navd88_m", "sd_m", "model_z_m", "model_sd_m", "radius_m", "confidence"]].to_string())
print(pd.DataFrame(BOUNDS)[["id", "bound", "rail_navd88_m", "model_z_m"]].to_string())
for r in ROWS:
    print(r["id"], r["osm_way_ids"][:150])
