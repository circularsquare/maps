"""evidence/placed_unused.csv: every depths.csv row that is not a document or elevator anchor in
build.py (data/doc_rows.csv) and not turned into a rail height by make_placed_rows.py, with the
reason. Run after make_placed_rows.py."""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent
ev = pd.read_csv(ROOT / "evidence" / "depths.csv", dtype=str).fillna("")
doc = pd.read_csv(ROOT / "data" / "doc_rows.csv", dtype=str).fillna("")
status = dict(zip(doc["id"], doc["status"]))
used = set()
for f in ("placed_rows.csv", "placed_bounds.csv"):
    for s in pd.read_csv(ROOT / "evidence" / f, dtype=str)["id"]:
        used.update(s.split("+"))

R = {
 "TD0003": "1904 gives 90-100 ft for the Mott Av platform, the 1906 book a 46 ft shaft; OSM names the track IRT White Plains Road Line (level -4, v1 depth 57 ft) and it is not certain the 1904 'Mott Avenue' is today's 149 St-Grand Concourse",
 "TD0005": "dredged trench bottom (excavation), not a rail; used only to cross-check TD0023 (subgrade about 1.6 m under the invert)",
 "TD0006": "maximum excavation, reference level not stated",
 "TD0007": "a bound (over 40 ft) on an excavation, reference not stated",
 "TD0010": "roof 5 ft below street 'south of the Post Office', the shallowest stretch of the IRT Lexington line under Broadway; v1 has the rail 14-21 m below the street from Fulton St to Bowling Green (the Joralemon ramp prior leaking up the line), so this is a contradiction to look at, but the stretch is not defined enough to pin (a roof 1.5 m down cannot hold where the line crosses other lines or starts down to the river)",
 "TD0011": "viaduct height at a station, rail vs platform vs structure not stated; Manhattan Valley viaduct over a valley, ground reference unclear (TD0018 says platform 55 ft)",
 "TD0012": "an upper limit (29 ft) for the two-bent viaducts", "TD0127": "an upper limit (29 ft)", "TD0128": "an upper limit (29 ft)",
 "TD0129": "an upper limit (29 ft)", "TD0130": "an upper limit (29 ft)", "TD0131": "an upper limit (29 ft)", "TD0132": "an upper limit (29 ft)",
 "TD0013": "access shaft depth, not platform", "TD0014": "access shaft depth, not platform", "TD0015": "access shaft depth, not platform",
 "TD0016": "river depth (bed), not a rail; checks CUDEM: 47 ft at high water = -14.0 m, CUDEM -13.8 median at the Joralemon crossing",
 "TD0017": "design limit; in placed_bounds.csv",
 "TD0018": "platform height of a viaduct station, 55 ft vs 68 ft (TD0011) disagree, and the street reference under a valley viaduct is unclear",
 "TD0019": "river depth (bed), not a rail; checks CUDEM: avg 23 ft below MHW = -6.6 m in 1915, CUDEM bed -4.6 m median, -5.6 m deepest (about 2 m shallower now)",
 "TD0022": "a grade (0.3%), not a height", "TD0009": "a grade, not a height",
 "TD0029": "depth of four station and ventilation shafts, not tracks",
 "TD0030": "excavation depth of a cut, line not named, reference not stated",
 "TD0031": "datum definition: Brooklyn IND datum 100 ft below MHW. Checked with TD0032 and the DEM: Fulton St at Utica Av, Elevation 150 = 50 ft above MHW = +15.7 m (1934 MHW +0.43), DEM 16.2 m; at Rockaway Av Elevation 200 = +30.9 m, DEM 29.4 m (coordinates approximate)",
 "TD0032": "street (ground) elevation, not a rail; used only to check the DEM and the datum, see TD0031",
 "TD0034": "double-deck station 'depth 45 ft' (structure, of which 25 ft in ground water), reference not stated",
 "TD0036": "a vertical distance between mezzanine and platform, not a depth",
 "TD0037": "a grade", "TD0038": "a grade", "TD0039": "a grade",
 "TD0043": "superseded 1890s plan, not as built; shaft location not given",
 "TD0044": "revised plan, not necessarily as built; shaft location not given",
 "TD0046": "relative to the 2004 design", "TD0047": "relative to the 2004 design", "TD0048": "relative to the 2004 design",
 "TD0053": "height of a steel arch, not the rail",
 "TD0057": "an elevator rise between levels, not a depth",
 "TD0058": "depth from a hilltop entrance to a mezzanine",
 "TD0060": "kiosk ground elevation, not a rail; checks the DEM: 188 ft = 57.3 m above sea level, DEM at the 181 St tracks 56.1 m",
 "TD0063": "pedestrian tunnel, reference unclear",
 "TD0067": "mezzanine depth, not a platform",
 "TD0068": "excavation of the original trench, datum unclear, section unclear",
 "TD0069": "an upper bound ('ranges up to 50 ft'), level not stated",
 "TD0081": "vertical separation between two tracks at Borough Hall (18 ft); a relative figure, needs a two-way input, not the single-height format here",
 "TD0082": "vertical separation between two lines (20 ft); relative, needs a two-way input",
 "TD0091": "lobby depth, not a platform",
 "TD0093": "open-cut excavation depth during construction, not the final rail",
 "TD0096": "likely 60 ft below the LIRR, not below ground: TD0097 says 45 ft below street and 60 ft below the LIRR tracks for the same crossing",
 "TD0097": "45 ft below street under the LIRR at Jamaica, crossing location not pinned (Main Line vs Atlantic Branch, 450 m apart); v1 already has the BMT Archer lower level 47 ft down there",
 "TD0099": "relative to a road underpass",
 "TD0104": "mezzanine depth, not a platform",
 "TD0111": "planned 50 ft for the 2009 South Ferry station; v1 already has it at street -15.2 m (rail), the same",
 "TD0113": "reference not named ('runs 40 ft under Seventh Av')", "TD0114": "reference not named ('50 ft under Broadway')",
 "TD0118": "reference of 'underground' not stated, and the old tunnel is gone",
 "TD0119": "relative to street car conduits, planning statement",
 "TD0122": "a grade", "TD0123": "mined stations as built, 'deep', conflicts with TD0074/TD0076 which are anchors",
 "TD0124": "planning statement, reference not stated", "TD0133": "platform height of an at-grade stretch (2 m), not rail; TD0134 is the rail",
 "TD0134": "a bound (under 1 m above street)",
 "TD0151": "two figures (105/135 ft) with no level named, 2004 design; Wikipedia says 140/155 ft for the built station",
 "TD0153": "range over the whole 2004 design",
 "TD0154": "alignment depth of the unbuilt 2004 design, reference not stated",
 "TD0156": "mezzanine floor, not a platform", "TD0157": "mezzanine floor elevation 128.4 ft in a datum the article does not name. If it is MHW + 100 ft as in Brooklyn it is +9.0 m (street 14.3 m, DEM there about 16); a mezzanine, not a rail",
 "TD0158": "an escalator rise between levels, not a depth",
 "TD0071": "limit only; in placed_bounds.csv",
 "TD0084": "Canal St Broadway mainline (20 ft): OSM names the only level -1 tracks there BMT Brighton Line/Nassau and the line identity is unclear; the Canal St levels (Nassau 40 ft, Manhattan Bridge 55 ft) already anchor",
 "TD0135": "2004 design of an unbuilt station",
 "TD0004": "design limit; in placed_bounds.csv",
 "TD0088": "passageway level, not a platform",
}
for i in list(range(136, 151)) + [152]:
    R.setdefault(f"TD{i:04d}", "2004 Second Avenue Subway design of an unbuilt station (south of 72 St or north of 96 St): no OSM track there")
R["TD0135"] = R["TD0136"]
rows = []
for _, r in ev.iterrows():
    i = r["id"]
    if i in used or status.get(i) in ("anchor", "elevator", "held out"):
        continue
    why = R.get(i)
    if why is None:
        if r["measure"] == "grade_percent":
            why = "a grade, not a height"
        elif r["measure"] == "mezzanine_below_street":
            why = "mezzanine depth, not a platform"
        else:
            why = "UNSORTED"
    rows.append({"id": i, "measure": r["measure"], "line": r["line"][:50], "place": r["place"][:60], "value": r["value"],
                 "unit": r["unit"], "reason": why})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "evidence" / "placed_unused.csv", index=False)
print(len(out), "unused;", (out["reason"] == "UNSORTED").sum(), "unsorted")
print(out[out["reason"] == "UNSORTED"][["id", "measure", "place", "value"]].to_string())
