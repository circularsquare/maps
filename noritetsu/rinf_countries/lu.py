"""Luxembourg: CFL is the one infrastructure manager (0082_IM), 98 sections, 263.7 km (pulled
2026-10-01). RINF's line id is CFL's line NAME, "Luxembourg - Troisvierges-frontière", not its
number; the public numbers (1, 1a, 6f...) are on Wikidata's P1671 of each line's item and on
some OSM route=tracks relations, and the 20 ids map onto them one to one, so they are fixed
here (`LU_FIXED`) rather than guessed from OSM, which numbers only 9 of them and filed all
three Pétange - Rodange ids under 6j.

The points carry no coordinate where rinf.py first looks: LU's geometry hangs off each
point's netReference (`netref_wkt`). Without it 16 junction and border points were unplaced
and the eight border points ("Kleinbettingen frontière") were placed by name onto their
station, so every section to the border traced 0 km and was rejected.

Also set here, each explained where it is set: `skip_line` (6d, the Langengrund freight
branch) and `fix` (RINF's Athus border point, 870 m inside Belgium).

Names: "Ligne 1 Luxembourg – Troisvierges-frontière", CFL's own number and line name (the
name is the RINF id, dashes typeset). English "Line 1 (Luxembourg – Troisvierges-frontière)".
Wikidata is fetched without English labels: they are "CFL Line 1", "Ettelbréck - Dikrech
railway line", which rinf.py would put in place of, or after, the English template.
"""

# RINF id (CFL's line name) -> public line number. Numbers from Wikidata P1671 (each item's
# label and length agree with the RINF id's ends and length) and CFL's network statement.
LU_FIXED = {
    "Luxembourg - Troisvierges-frontière": "1",
    "Ettelbruck - Diekirch": "1a",
    "Kautenbach - Wiltz": "1b",
    "Ettelbruck - Bissen": "2b",
    "Luxembourg - Wasserbillig-frontière via Sandweiler-Contern": "3",
    "Luxembourg - Berchem - Oetrange": "4",
    "Luxembourg - Kleinbettingen-frontière": "5",
    "Luxembourg - Bettembourg-frontière": "6",
    "Bettembourg - Esch/Alzette": "6a",
    "Bettembourg - Dudelange-Usines (Volmerange)": "6b",
    "Noertzange - Rumelange": "6c",
    "Tétange - Langengrund": "6d",
    "Esch/Alzette - Audun-le-Tiche": "6e",
    "Esch/Alzette - Pétange": "6f",
    "Pétange - Rodange-frontière (Aubange)": "6g",
    "Pétange - Rodange-frontière (Mont St. Martin)": "6h",
    "Pétange - Rodange-frontière (Athus)": "6j",
    "Brucherberg - Scheuerbusch": "6k",
    "Luxembourg - Pétange": "7",
}
ROUTES = {ref: lid.replace(" - ", " – ") for lid, ref in LU_FIXED.items()}
ROUTES["3"] = "Luxembourg – Wasserbillig-frontière"


def lu_skip(lid):
    """6d, Tétange - Langengrund, is a freight branch to the Langengrund industrial sidings
    (no OSM passenger route ends or stops there). Its first 1.6 km run beside 6c into
    Rumelange, where CFL's 60b runs, and that alone kept the whole 3.3 km section as ridden."""
    return lid == "Tétange - Langengrund"


def lu_fix(secs, points):
    """RINF places "Rodange frontière B A" (EU00098, the 6j border with Belgium towards Athus)
    at 5.8106 49.5518: 870 m inside Belgium, on the longitude of the Mont-Saint-Martin border
    point and the latitude of the Aubange one, so a copy slip. 6j's OSM relation (20658090)
    and Infrabel's L167 meet the border at 5.82501 49.55180, which also fits RINF's 1.478 km
    from Rodange. Unfixed, the 6j border section traced 2.53 km through Athus' approach."""
    out = []
    for p in points.values():
        if p.get("uopid") == "EU00098" and abs(p.get("lon", 0) - 5.8106) < 1e-3:
            p["lon"], p["lat"] = 5.82501, 49.55180
            out.append("EU00098 Rodange frontière B A moved to 6j's border crossing (5.82501 49.55180)")
    return out


class LuName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        route = ROUTES.get(ref.lower())
        if route:
            return str.format(self, ref=ref.lower(), route=route)
        return str.format(self.split(" {route}")[0].split(" ({route})")[0], ref=ref.lower())


COUNTRY = {
    "iso3": "LUX", "wikidata": "Q32", "langs": ["fr", "lb", "de"],
    "fixed": LU_FIXED, "ref": lambda _lid: None,
    # rinf.py upper-cases refs as keys ("6F"); CFL writes 6f.
    "ref_display": lambda k: k.lower(),
    "name": LuName("Ligne {ref} {route}"), "name_en": LuName("Line {ref} ({route})"),
    "im": {"0082_IM": "CFL"},
    "netref_wkt": True,
    "skip_line": lu_skip, "fix": lu_fix,
}
