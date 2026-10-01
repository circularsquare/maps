"""Bulgaria: RINF carries NRIC alone (НКЖИ, Национална компания "Железопътна инфраструктура",
IM code 0052), 54 line ids, 3,766 km. The public numbers are NRIC's ("Железопътна линия 2",
bg.wikipedia's and OSM's route=railway relations' too), but RINF splits several lines into
lettered parts and writes a few dotted numbers without the dot, so FIXED maps them back:

- 1A Sofia - Plovdiv - Svilengrad, 1B Sofia - Kalotina: line 1. 1A1 (Svilengrad - Greek
  border, 3.9 km) is left to OSM, whose line 1 relation covers it.
- 3A Iliyantsi - Karlovo - Zimnitsa, 3B Karnobat - Varna feribotna: line 3, which runs over
  line 8 between Zimnitsa and Karnobat, so it is two pieces.
- 4A Ruse - Gorna Oryahovitsa - Stara Zagora (with the Ruse Razpredelitelna - Giurgiu border
  stub), 4B Dimitrovgrad - Mihaylovo, 4C Dimitrovgrad - Podkova, and 4A2-4A4, NRIC's Ruse
  junction connections (Ruse Zapad, Ruse Sever): line 4. Stara Zagora - Mihaylovo is line 8.
- 6A Voluyak - Pernik Razpredelitelna, 6B Radomir - Gyueshevo: line 6 (Pernik - Radomir is
  line 5).
- 8A, the second Burgas approach via Burgas Razpredelitelna: line 8.
- 51A Dupnitsa - Bobov dol is 51; 51B General Todorov - Petrich is NRIC's line 52.
- 821 Dolna Mahala - Hisarya is 82.1 (OSM writes "82.1"); 291 and 331 are read the same way.
Other ids that are one or two digits are the number. 701 (Vidin feribotna - Danube Bridge 2)
and 7A2 (Vidin - Kapitanovtsi) never take a number: OSM's line 7 relation lies on them, and
joined to line 7 they made Vidin Tovarna a branch point, so build_model cut Vidbol - Vidin,
the line's own passenger end, as an unridden junction section.

STOPS. RINF types some of Bulgaria's busiest stations as junctions (op type 80: Mezdra,
Shumen, Karnobat, Kaspichan, Levski, Radomir, Voluyak, Sofia Sever...), Ruse Razpredelitelna as
a marshalling yard (100) and three halts as switches (120). STOP_NAMES lists every one that has
an OSM railway=station or halt within 100 m and passenger trains (checked one by one,
2026-10-01). Without them Mezdra - Vratsa, all Sofia - Vidin traffic, was dropped as an
unridden junction section. Freight terminals (40: Vidin Tovarna, Stanyantsi, Bobov dol) are
not listed, so freight branches still end at junctions and answer to OSM's routes.

RINF lists stations (гари) but no halts (спирки): 265 passenger points for about 650 places
trains call at. `osm_stops: "all"` makes every OSM rail station lying on a traced stop-to-stop
section a stop of it (rinf.split_at_osm_stops); "all" rather than True because Bulgaria has
16 OSM train routes, so "a route stops there" would keep almost none.

RINF's point names are an old Latin transliteration in capitals (TRJAVNA, KJUSTENDIL); OSM's
are Cyrillic. rinf.norm folds Cyrillic to Latin (CYRILLIC_FOLD), so 236 of 288 stops match by
name; the other 52 by distance (all within 100 m, listed in the build log). Names shown are
OSM's Cyrillic ones.

NAMES are NRIC's form, "Железопътна линия 2"; English "Line 2", with Wikidata's English label
added where its item passes rinf.wikidata_item's length check ("Line 2 (Sofia–Varna railway)").
"""
import re

FIXED = {
    "1A": "1", "1B": "1",
    "3A": "3", "3B": "3",
    "4A": "4", "4B": "4", "4C": "4", "4A2": "4", "4A3": "4", "4A4": "4",
    "6A": "6", "6B": "6",
    "8A": "8",
    "51A": "51", "51B": "52",
    "821": "82.1", "291": "29.1", "331": "33.1",
}

# RINF points typed junction (80), marshalling yard (100) or switch (120) that are passenger
# stations or halts, by RINF's own name (STOPS in the docstring).
STOP_NAMES = {
    "BATANOVTSI", "ILIYANTSI", "KARNOBAT", "KASPICHAN", "KAZICHENE", "LEVSKI", "MEZDRA",
    "MEZDRA YUG", "PERNIK RAZPREDELITELNA", "POLIKRAJSHTE", "POVELJANOVO", "RADOMIR",
    "RAZDELNA", "RAZMENNA", "RESEN", "RP KURTOVO KONARE", "RP LOZOVO", "RUSKA BIALA",
    "SAMOVODENE", "SAMUIL", "SHUMEN", "SOFIA SEVER", "TSAREVA LIVADA", "TULOVO", "VOLUYAK",
    "RUSE RAZPREDELITELNA", "CHUMERNA", "RP ALEKSANDAR DIMITROV", "RP KOPILOVCI",
}

NO_NUMBER = {"701", "7A2"}


def bg_ref(lid):
    if lid in FIXED:
        return FIXED[lid]
    return lid if re.fullmatch(r"\d{1,2}", lid or "") else None


COUNTRY = {
    "iso3": "BGR", "wikidata": "Q219", "langs": ["bg", "en"],
    "fixed": FIXED, "ref": bg_ref, "rule_certain": True,
    "no_ref": lambda lid: lid in NO_NUMBER,
    "stop_names": STOP_NAMES,
    "osm_stops": "all",
    "name": "Железопътна линия {ref}", "name_en": "Line {ref}",
    "im": {"0052_IM": "НКЖИ"},
}
