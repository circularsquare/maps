"""Latvia: Latvijas dzelzceļš (LDz, IM 0025) alone. RINF's id is not LDz's line number but the
codes of the section's two end points: "LV2060100301" is LV106010 (Rīga Pasažieru) to
LV103010 (Jelgava), "EU203010B145" is Jelgava to border point EU00145 (Meitene). LDz numbers
its lines 01-36 (its network statement, lv.wikipedia, Wikidata P1671: 14 is Rīga—Jelgava), so
FIXED maps each of the 20 ids to its number by reading the id's end points against those
lines. OSM's route=railway relations in Latvia carry no LDz numbers (one is "ref=Rīga -
Daugavpils"), so they never number anything here (`osm_ref` returns None).

RINF has only 36 points here, the junction stations and border points: there is no
intermediate stop at all (Rīga Pasažieru - Aizkraukle is one 82 km section). See lv_sources.md
for what that does to the strip diagrams.

Names are the line's two ends as LDz and lv.wikipedia write them ("Dzelzceļa līnija
Rīga—Jelgava" is the article), shown as "Rīga–Jelgava"; Wikidata's Latvian labels put the words
in either order and some lines have none, so `LV_NAMES` fixes one form. The number is the ref.
"""

# id -> LDz line number. LV2120300503 is two curves around Daugavpils, 26 (Postenis 191. km -
# 524. km) and 13 (524. km - 401. km), under one id; it takes 26's number and is freight only.
FIXED = {
    "LV2010100201": "01",   # Ventspils I - Tukums II
    "LV2020100301": "02",   # Tukums II - Jelgava
    "LV2030100401": "03",   # Jelgava - Krustpils
    "LV2040100501": "04",   # Krustpils - Daugavpils
    "EU205010B204": "05",   # Daugavpils - Indra - border
    "LV2060100401": "06",   # Rīga - Aizkraukle - Krustpils
    "LV2040100801": "07",   # Krustpils - Rēzekne II
    "EU208010B143": "08",   # Rēzekne II - Zilupe - border
    "EU209060B144": "09",   # Kārsava border - Rēzekne I, with the Rēzekne I - II link
    "LV2090601009": "10",   # Rēzekne I - Daugavpils Šķirotava
    "EU210090B147": "11",   # Daugavpils Šķirotava - Kurcums - border
    "EU205010B146": "12",   # Daugavpils - Eglaine - border
    "LV2060100301": "14",   # Rīga - Jelgava
    "LV2030101511": "15",   # Jelgava - Glūda - Liepāja
    "EU203010B145": "16",   # Jelgava - Meitene - border
    "EU206010B205": "17",   # Rīga - Sigulda - Cēsis - Lugaži - border
    "LV2060100201": "18",   # Rīga - Zasulauks - Sloka - Tukums II
    "EU215010B148": "21",   # Glūda - Reņģe - border
    "LV2180102202": "22",   # Zasulauks - Bolderāja
    "LV2120300503": "26",   # Daugavpils bypass curves (26 and 13)
}

# Lines with no passenger train whose ends are both stations, which build_model never
# questions. Vivi's timetable page (vivi.lv/lv/informacija-pasazieriem/, read 2026-10-01) lists
# its routes as Rīga-Tukums/Dubulti, Rīga-Skulte, Rīga-Sigulda-Valga, Rīga-Jelgava-Liepāja and
# Rīga-Aizkraukle/Gulbene/Zilupe/Indra; LTG Link runs Vilnius - Rīga over 16. None of these use
# 01 Ventspils - Tukums II, 02 Tukums II - Jelgava, 03 Jelgava - Krustpils, 10 Rēzekne I -
# Daugavpils (Krāce and Aglona are RINF stations, so it was kept) or 09's Rēzekne I - II link,
# and OSM has no route on any of them. Junction-ended freight lines (11, 12, 21, 22, 26) go
# without help, as unridden.
FREIGHT = {"LV2010100201", "LV2020100301", "LV2030100401", "LV2090601009", "EU209060B144"}

LV_NAMES = {
    "01": ("Ventspils I—Tukums II", "Ventspils–Tukums II"),
    "02": ("Tukums II—Jelgava", "Tukums II–Jelgava"),
    "03": ("Jelgava—Krustpils", "Jelgava–Krustpils"),
    "04": ("Krustpils—Daugavpils", "Krustpils–Daugavpils"),
    "05": ("Daugavpils—Indra", "Daugavpils–Indra"),
    "06": ("Rīga—Krustpils", "Riga–Krustpils"),
    "07": ("Krustpils—Rēzekne II", "Krustpils–Rēzekne II"),
    "08": ("Rēzekne II—Zilupe", "Rēzekne II–Zilupe"),
    "09": ("Kārsava—Rēzekne I", "Kārsava–Rēzekne I"),
    "10": ("Rēzekne I—Daugavpils", "Rēzekne I–Daugavpils"),
    "11": ("Daugavpils—Kurcums", "Daugavpils–Kurcums"),
    "12": ("Eglaine—Daugavpils", "Eglaine–Daugavpils"),
    "14": ("Rīga—Jelgava", "Riga–Jelgava"),
    "15": ("Jelgava—Liepāja", "Jelgava–Liepāja"),
    "16": ("Jelgava—Meitene", "Jelgava–Meitene"),
    "17": ("Rīga—Lugaži", "Riga–Lugaži"),
    "18": ("Torņakalns—Tukums II", "Torņakalns–Tukums II"),
    "21": ("Glūda—Reņģe", "Glūda–Reņģe"),
    "22": ("Zasulauks—Bolderāja", "Zasulauks–Bolderāja"),
    "26": ("Postenis 191. km—Postenis 524. km", "Daugavpils bypass"),
}


class LvName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""
    def __new__(cls, which):
        s = super().__new__(cls, "{ref}")
        s.which = which
        return s

    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        got = LV_NAMES.get(ref)
        if not got:
            return f"Līnija {ref}" if self.which == 0 else f"Line {ref}"
        return got[0].replace("—", "–") if self.which == 0 else f"{got[1]} line"


COUNTRY = {
    # Latvian labels only: with an English label rinf.py appends it to name_en.
    "iso3": "LVA", "wikidata": "Q211", "langs": ["lv"],
    "fixed": FIXED, "rule_certain": True, "skip_line": lambda lid: lid in FREIGHT,
    # RINF has no stop between junction stations; OSM's train-route stops fill them in.
    "osm_stops": True,
    "osm_ref": lambda _r: None,
    "name": LvName(0), "name_en": LvName(1),
    "im": {"0025_IM": "Latvijas dzelzceļš"},
}
