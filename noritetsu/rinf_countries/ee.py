"""Estonia: Eesti Raudtee (IM 0001) and Edelaraudtee Infrastruktuur (0002: Tallinn - Lelle -
Pärnu, Lelle - Viljandi, Liiva - Ülemiste), plus four industrial managers in Ida-Virumaa
(0007-0010: the oil-shale mines' and power stations' railways around Ahtme, Kohtla-Järve,
Sõtke, Musta and Narva Elektrijaamad), whose lines carry no passengers and are left out.

RINF's id is the line's name, "Tapa-Tartu", "Tallinn-Lelle-Pärnu", with place names run
together ("NarvaElektrijaamad") and "piir" (boundary) for the points where one manager's
track ends. Estonia has no public line numbers (Wikidata's one P1671, R14 on Tallinn -
Paldiski, is a timetable route), so lines have no ref and their names come from the id
(`id_name`, needs rinf.py to read it for lines without a number), written out as Eesti Raudtee
and Elron write them, "Tapa–Tartu".

Every point's name is "Rakenduspunkt Tapa" (operational point Tapa); `fix` strips the word,
or nothing matched an OSM station by name and Ülemiste landed on the halt Vesse by distance.
"""
import re

NAMES = {
    "Tallinn-Tapa": "Tallinn–Tapa",
    "Tapa-Narva": "Tapa–Narva",
    "Tapa-Tartu": "Tapa–Tartu",
    "Tartu-Valga": "Tartu–Valga",
    "Tartu-Koidula": "Tartu–Koidula",
    "Valga-Koidula": "Piusa–Koidula",       # the ridden end of Valga - Koidula (DROP_AT)
    "Keila-Tallinn": "Tallinn–Keila",
    "Keila-Paldiski": "Keila–Paldiski",
    "Riisipere-Keila": "Keila–Riisipere",
    "Klooga-Kloogarand": "Klooga–Kloogaranna",
    "Lagedi-Muuga": "Lagedi–Muuga",
    "Ülemiste-Blokkpost": "Ülemiste–Blokkpost 4 km",
    "Tallinn-TallinnVäikepiir": "Tallinn–Tallinn-Väike",
    "Tallinn-Lelle-Pärnu": "Tallinn–Rapla–Lelle",   # Lelle - Pärnu closed (DROP_AT)
    "Lelle-Viljandi": "Lelle–Viljandi",
    "Liiva-Ülemistepiir": "Liiva–Ülemiste",
    "Ülemiste-Ülemiste-piir": "Ülemiste",
    "Narva-Ivangorodpiir": "Narva–Ivangorod",
    "Koidula-Petšorõpiir": "Koidula–Petseri",
    "Valga-Lugažipiir": "Valga–riigipiir",   # to the Latvian border, towards Lugaži
    "Valga-Valkapiir": "Valga–Valka",
    "Jõhvi-Jõhvipiir": "Jõhvi",
    "Vaivarapiir-Vaivara": "Vaivara",
    "Kohtla-Kohtlapiir": "Kohtla",
    "Kohtlapiir2-Kohtla": "Kohtla",
    "Soldina-NarvaElektrijaamad": "Soldina–Narva Elektrijaamad",
}


# Lines ridden over part of their length. Elron runs no train Lelle - Pärnu (ended December
# 2018) and none on Valga - Võru - Piusa (its Tartu - Koidula trains turn at Piusa); OSM has no
# route on either. Their RINF points between are OSM stations or merge away (Võru, Antsla and
# Karula have no OSM station node), so the sections were kept unquestioned. Sections touching
# these points are left out.
DROP_AT = {
    "Tallinn-Lelle-Pärnu": {"Tootsi", "Parnu", "Papiniidu"},
    "Valga-Koidula": {"Valga", "Karula", "Antsla", "Sõmerpalu", "Võru", "Lepassaare"},
}


def ee_fix(secs, points):
    """rinf.py's `fix`: "Rakenduspunkt Tapa" -> "Tapa", and DROP_AT."""
    n = 0
    for p in points.values():
        if p.get("name", "").startswith("Rakenduspunkt "):
            p["name"] = re.sub(r"^Rakenduspunkt\s+", "", p["name"])
            n += 1
    before = len(secs)
    secs[:] = [s for s in secs
               if not ({points.get(s["a"], {}).get("name"), points.get(s["b"], {}).get("name")}
                       & DROP_AT.get(s["base"], set()))]
    return [f"'Rakenduspunkt ' taken off {n} point names",
            f"DROP_AT left out {before - len(secs)} sections with no passenger train"]


def ee_id_name(lid, _uop):
    return NAMES.get(lid) or (lid or "").replace("-", "–") or None


# The industrial managers' lines (0007-0010), and Eesti Raudtee's stubs to their boundary
# points, which no passenger train uses. Also Ülemiste - Blokkpost 4 km, the freight bypass
# towards Muuga port: it runs beside the Tapa line past the halt Vesse, which `osm_stops` then
# put on it, and Ülemiste - Vesse became a section between two stops that nothing questions.
INDUSTRIAL = {
    "Ahtme-Raudi", "Sõtke-Musta", "Jõhvipiir-Ahtme-Vaivarapiir", "Kohtla-Järve-Ahtme",
    "Kohtlapiir-Tehase-Kohtla-Järve-Vaheküla-Kohtlapiir2", "Soldina-NarvaElektrijaamad",
    "Jõhvi-Jõhvipiir", "Vaivarapiir-Vaivara", "Kohtla-Kohtlapiir", "Kohtlapiir2-Kohtla",
    "Ülemiste-Blokkpost",
    # The same 1.9 km from Valga to the Latvian border as Valga-Lugažipiir: the two border
    # points (Valkapiir, and LDz's EU00205 Lugaži/Valga) are 5 m apart.
    "Valga-Valkapiir",
}

# Ids that are one line to a rider. Balti - Tallinn-Väike (Eesti Raudtee's) is the first 3 km
# of every Rapla and Viljandi train, the rest of which is Edelaraudtee's Tallinn-Lelle-Pärnu.
# The key only groups them: `ref_display` writes no number, so the line has no ref and takes
# its name from `id_name` of the first id ("Tallinn-Lelle-Pärnu").
FOLD = {"Tallinn-TallinnVäikepiir": "TLP", "Tallinn-Lelle-Pärnu": "TLP"}


COUNTRY = {
    "iso3": "EST", "wikidata": "Q191", "langs": ["et"],
    "fix": ee_fix,
    # Some points lie well off their station: Ülemiste 1.46 km, Kärkna 1.1 km.
    "name_m": 1600,
    # RINF lists about one Elron stop in three (none between Tallinn and Keila but Pääsküla
    # and Valingu); OSM's train-route stops fill in the rest.
    "osm_stops": True,
    # RINF's points lie at the middle of the operational point, which at Ülemiste is 1.46 km
    # past the platforms: Balti - Ülemiste is 7.97 km in RINF for 6.41 of track, and the next
    # section long by as much. At rinf.py's 0.3 it was rejected, losing Tallinn - Ülemiste.
    "tol_abs": 1.0,
    "fixed": FOLD, "ref_display": lambda _k: "",
    "osm_ref": lambda _r: None,
    "id_name": ee_id_name,
    "skip_line": lambda lid: lid in INDUSTRIAL,
    "im": {"0001_IM": "Eesti Raudtee", "0002_IM": "Edelaraudtee Infrastruktuur"},
}
