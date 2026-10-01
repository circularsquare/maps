"""Slovenia: RINF's ids are SŽ-Infrastruktura's own line numbers, "10", "50", "62" (all 319
sections under one manager code, 0079_IM; pulled 2026-10-01). Wikidata's P1671 carries the
same numbers on each line's item, and OSM's route=railway relations the same `ref` (25 of
the 26 ids confirmed that way). So the id IS the number (`rule_certain`, as in Austria and
Poland).

Slovenian riders and sl.wikipedia know a line by its two ends, not its number: the articles
are "Železniška proga Ljubljana–Sežana–d. m." ("d. m." is državna meja, the state border).
The name here is number then route, "Proga 50 Ljubljana–Sežana", with the route from
`ROUTES` below: sl.wikipedia's article title for that number (matched through Wikidata's
P1671), less its "Železniška proga" and less a trailing "–d. m." where two places are left
without it. The English name is "Line 50 (Ljubljana–Sežana)".

Also set here, each explained where it is set: `skip_line` (the Ljubljana Zalog freight
tracks), `name_m` (three halts RINF places over 1 km from their station) and `stop_names`
(four passenger stations RINF types as depots or yards).

Wikidata is fetched with Slovenian labels only, as Hungary does: rinf.py appends an English
label that does not contain the number to the English template, which would give "Line 50
(Ljubljana–Sežana) (Ljubljana–Sežana railway line)". Several numbers have a second, historic
item (50 is also the whole Südbahn Spielfeld-Straß - Trieste, 70 the Bohinj Railway, 64
Pivka - Rijeka), so labels from Wikidata were never going to be safe names anyway.
"""
import re

# Number -> route, from sl.wikipedia's line articles (titles, retrieved 2026-10-01) and,
# where sl.wikipedia has no article of its own for the number, Wikidata's Slovenian label for
# its P1671. 11-13 are the three tracks between Ljubljana Zalog yard and Ljubljana
# (sl.wikipedia: "Ljubljana Zalog–Ljubljana P3/P4/P5"; RINF carries 12 and 13, 3.8 and 3.5
# km, so 12 is P4 and 13 is P5).
ROUTES = {
    "10": "Ljubljana–Dobova",
    "11": "Ljubljana Zalog–Ljubljana P3",
    "12": "Ljubljana Zalog–Ljubljana P4",
    "13": "Ljubljana Zalog–Ljubljana P5",
    "20": "Ljubljana–Jesenice",
    "21": "Ljubljana Šiška–Kamnik Graben",
    "22": "Kranj–Naklo",
    "30": "Zidani Most–Šentilj",
    "31": "Celje–Velenje",
    "32": "Grobelno–Rogatec",
    "33": "Stranje–Imeno",
    "34": "Maribor–Prevalje",
    "35": "Maribor Tezno–Maribor Studenci",
    "36": "Dravograd–Otiški Vrh",
    "40": "Pragersko–Ormož",
    "41": "Ormož–Hodoš",
    "42": "Ljutomer–Gornja Radgona",
    "43": "Lendava–d. m.",
    "44": "Ormož–Središče",
    "50": "Ljubljana–Sežana",
    "60": "Divača–Prešnica",
    "61": "Prešnica–Podgorje",
    "62": "Prešnica–Koper",
    "64": "Pivka–Ilirska Bistrica",
    "70": "Jesenice–Sežana",
    "71": "Šempeter pri Gorici–Vrtojba",
    "72": "Prvačina–Ajdovščina",
    "73": "Kreplje–Repentabor",
    "80": "Ljubljana–Metlika",
    "81": "Sevnica–Trebnje",
    "82": "Grosuplje–Kočevje",
    "83": "Novo mesto–Straža",
}


def si_skip(lid):
    """11-13 are the freight tracks between Ljubljana Zalog yard and Ljubljana (P3-P5);
    passenger trains run on line 10 beside them. Once Ljubljana Zalog is a stop
    (`stop_names`), 12 and 13 run stop to stop, which build_model never questions, and both
    traced 8 km for RINF's 3.8 and 3.5 over line 10's own track."""
    return lid in {"11", "12", "13"}


def si_ref(lid):
    return lid if re.fullmatch(r"\d{2}", lid or "") else None


class SiName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""
    def __new__(cls, form):
        return super().__new__(cls, form)

    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        route = ROUTES.get(ref)
        return str.format(self, ref=ref, route=route) if route else \
            str.format(self.split(" {route}")[0].split(" ({route})")[0], ref=ref)


COUNTRY = {
    "iso3": "SVN", "wikidata": "Q215", "langs": ["sl"],
    "ref": si_ref, "rule_certain": True, "skip_line": si_skip,
    "name": SiName("Proga {ref} {route}"), "name_en": SiName("Line {ref} ({route})"),
    "im": {"0079_IM": "SŽ-Infrastruktura"},
    # Three SŽ halts have their RINF coordinate 1.1-1.3 km from the OSM station of the same
    # name (Frankovci 1127 m, Radeče 1129 m, Vidina 1298 m). At rinf.py's 1000 m they became
    # junctions and dropped out of lines 41, 30 and 32 as stops. Only a name match uses this
    # radius; matching by distance alone stays at BLIND_M.
    "name_m": 1500,
    # Passenger stations RINF types as something else, each with an OSM rail station of its
    # name within 50 m and OSM routes calling: Kamnik Graben (type 50, depot; the terminus of
    # line 21, whose last 0.49 km was otherwise dropped as unridden), Ajdovščina (50, the
    # terminus of line 72), Ljubljana Vižmarje (60, technical services; mid-line 20) and
    # Ljubljana Zalog (100, shunting yard; also a stop on line 10).
    "stop_names": {"Kamnik Graben", "Ajdovščina", "Ljubljana Vižmarje", "Ljubljana Zalog"},
}
