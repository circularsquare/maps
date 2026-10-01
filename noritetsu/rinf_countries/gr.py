"""Greece: RINF carries OSE alone (Οργανισμός Σιδηροδρόμων Ελλάδος, IM code 0073), the standard-
gauge network only: 25 line ids "01.00.00" to "28.00.00" (02, 14 and 15 unused), 193 sections,
1,830 km once line 04's metre slip is put right. No metre-gauge or 750/600 mm line is in it
(Patras suburban, Katakolo - Olympia, Diakopto - Kalavryta, Pelion), and no metro or tram; those
are OSM lines or nothing (gr_sources.md).

NUMBERS. OSE's ids are not public line numbers: nobody writes "line 25", and Greek lines are
known by their ends ("Σιδηροδρομική γραμμή Θεσσαλονίκης - Αλεξανδρούπολης"). So lines carry no
ref. `fixed` groups the ids that are one line to a rider under a key and `ref_display` writes no
number (Estonia's way); `id_name` gives the Greek name and the English one (rinf.py reads a
(name, name_en) tuple). OSM's route=railway refs ("Ε 85", "Θ-Α") are ignored (`osm_ref`).

- 01 Piraeus - Platy - Thessaloniki TX1 junction, with 22 (TX1 - Thessaloniki station) so the
  line ends at the station every train from Athens uses.
- 03 Airport - SKA, 04 SKA - Ano Liosia, 05 Ano Liosia - Kiato and 06 (a 1.5 km link at Ano
  Liosia): the new Airport - Kiato line Proastiakos A1, A2 and A4 run over, one line.
- 20 Thessaloniki TX1 - Polykastro - Idomeni border and 21 Polykastro - Idomeni: one line.
- 27 Alexandroupoli - Ormenio - Bulgarian border and 28 (Pythio - Turkish border, 1 km).
Every other id is a line by itself.

FIXES TO RINF (`fix`):
- Line 04's SWITCH 635A - ANO LIOSIA N. is 1192 for 1.2 km between its points: metres.
- RINF's line 01 has a gap: Lianokladi - Domokos is only there as 11, the old mountain line
  through Karya (65.4 km, closed when the new double-track line through the Othrys tunnel
  opened; most of its track is gone from OSM). The new line is added to 01 as one section of
  53.23 km: the new Tithorea - Domokos alignment is 107.33 km (en.wikipedia "Piraeus–Platy
  railway") less RINF's Tithorea - Molos - Lianokladi, 54.10. Traced: 51.3 km via Angeies.
- Points renamed to the Greek name of the station they are (RENAME): where RINF's old
  transliteration matches no OSM name (LARISSA is "Λάρισα", and its RINF point lies 1.5 km off
  the station, hence `name_m`; VRISI is the halt Μεγάλη Βρύση), and RINF's ALEXANDROUPOLIS,
  1.1 km east of the port station that every train uses, is that station. Border points and
  named switches get Greek names too, since they show on the strip diagram.

LEFT OUT (`skip_line`): 07 (Thriasio - Ikonio port freight line), 23 and 24 (Thessaloniki freight
station links), 11 (the old Lianokladi - Domokos line, above). 17 (Amyntaio - Ptolemaida -
Kozani) and 18 (Mesonisi - Neos Kafkasos, to North Macedonia) have no track in OSM and come out
empty on their own; 19 (Axios - Gefyra link) is unridden and dropped by build_model.

NOT RUNNING (`suspended`, greyed by not_running.py): 09, the old Tithorea - Lianokladi line over
Bralos and the Gorgopotamos viaduct, bypassed by the new line through the Kallidromo tunnel since
2018 (OSM tags it usage=tourism; no train in Hellenic Train's timetable), and 20, Thessaloniki -
Idomeni (passenger service suspended since 2020, en.wikipedia; none in the timetable).
Lines with trains on only part of their length, or suspended for works, are left to the
timetable check (gr_sources.md lists them): Serres - Alexandroupoli on 25, 10, 12, 13, 26.

STOPS. RINF lists stations and halts (172 passenger points), but not the Proastiakos halts added
since (Lefka, Tavros, Pyrgos Vasilissis...); `osm_stops: True` adds every OSM station an OSM train
route stops at. With rinf.py's GREEK_FOLD, 97 RINF points match their OSM station by name and 50
more by distance (all within 200 m, listed in the build log); 25 have no OSM station, nearly all
closed halts of the old lines (09, 11, 17) or halts no train calls at.

CREDIT (`highspeed_sections: False`). The main line is one line over old and new alignment,
with no conventional line beside the new one. With speed known per section, OSM's ICE (three
stops, so one 316 km section Athens - Larissa, under half on highspeed=yes track) credited none
of the new Tithorea - Molos - Lianokladi alignment; with it left unknown the ICE credits 481 of
481 km of 01 and nothing it should not.
"""

# RINF id -> group key; ids sharing a key are one line (see NUMBERS).
GROUP = {
    "01.00.00": "01", "22.00.00": "01",
    "03.00.00": "03", "04.00.00": "03", "05.00.00": "03", "06.00.00": "03",
    "20.00.00": "20", "21.00.00": "20",
    "27.00.00": "27", "28.00.00": "27",
}

NAMES = {
    "01": ("Πειραιάς – Θεσσαλονίκη", "Piraeus – Thessaloniki"),
    "03": ("Αεροδρόμιο – Κιάτο", "Athens Airport – Kiato"),
    "08": ("Οινόη – Χαλκίδα", "Oinoi – Chalkida"),
    "09": ("Τιθορέα – Λειανοκλάδι (παλαιά γραμμή)", "Tithorea – Lianokladi (old line)"),
    "10": ("Λειανοκλάδι – Στυλίδα", "Lianokladi – Stylida"),
    "11": ("Λειανοκλάδι – Δομοκός (παλαιά γραμμή)", "Lianokladi – Domokos (old line)"),
    "12": ("Παλαιοφάρσαλος – Καλαμπάκα", "Palaiofarsalos – Kalambaka"),
    "13": ("Λάρισα – Βόλος", "Larissa – Volos"),
    "16": ("Πλατύ – Φλώρινα", "Platy – Florina"),
    "17": ("Αμύνταιο – Πτολεμαΐδα", "Amyntaio – Ptolemaida"),
    "18": ("Μεσονήσι – Νέος Καύκασος", "Mesonisi – Neos Kafkasos"),
    "19": ("Ενωτική Αξιού – Γέφυρας", "Axios – Gefyra link"),
    "20": ("Θεσσαλονίκη – Ειδομένη", "Thessaloniki – Idomeni"),
    "25": ("Θεσσαλονίκη – Αλεξανδρούπολη", "Thessaloniki – Alexandroupoli"),
    "26": ("Στρυμόνας – Προμαχώνας", "Strymonas – Promachonas"),
    "27": ("Αλεξανδρούπολη – Ορμένιο", "Alexandroupoli – Ormenio"),
}

# RINF point name -> the Greek name it is matched and shown by (FIXES).
RENAME = {
    "LARISSA": "Λάρισα",
    "VRISI": "Μεγάλη Βρύση",
    "POLYSITON": "Πολύσιτος",
    "ALEXANDROUPOLIS": "Αλεξανδρούπολη Λιμάνι",
    "PYTHION-UZUNKOPRU": "Πύθιο – Ουζούνκιοπρου (σύνορα)",
    "SVILENGRAD-DIKEA": "Δίκαια – Σβίλενγκραντ (σύνορα)",
    "KULATA-PROMACHON": "Προμαχώνας – Κουλάτα (σύνορα)",
    "IDOMENI-GEVGELIJA": "Ειδομένη – Γευγελή (σύνορα)",
    "NEOS KAFKASOS-KREMENITSA": "Νέος Καύκασος – Κρεμένιτσα (σύνορα)",
    "IKONIO (SWITCH)": "Ικόνιο (αλλαγή)",
    "THRIASIO (SWITCH)": "Θριάσιο (αλλαγή)",
    "MESSONISSION (SWITCH)": "Μεσονήσι (αλλαγή)",
    "THESSALONIKI TRIAGE": "Θεσσαλονίκη Διαλογή",
    "AXIOS": "Αξιός",
}

FREIGHT = {"07.00.00", "23.00.00", "24.00.00"}
SKIP = FREIGHT | {"11.00.00"}
SUSPENDED = {"09.00.00", "20.00.00", "21.00.00"}

# Tithorea - Domokos new alignment, 107.33 km, less RINF's Tithorea - Molos - Lianokladi.
NEW_LIANOKLADI_DOMOKOS_KM = 107.33 - (26.04 + 28.06)


def gr_fix(secs, points):
    out = []
    for s in secs:
        if s["km"] and s["km"] > 100:
            out.append(f"{s['line']} {s['label']}: {s['km']} read as metres")
            s["km"] /= 1000
    n = 0
    for p in points.values():
        if p.get("name") in RENAME:
            p["name"] = RENAME[p["name"]]
            n += 1
    out.append(f"{n} points renamed to their Greek names")
    by_uop = {p.get("uopid"): op for op, p in points.items()}
    a, b = by_uop["EL01410"], by_uop["EL01490"]           # LIANOKLADION, DOMOKOS
    secs.append({"sol": "gr:new-lianokladi-domokos", "line": "01.00.00#1",
                 "base": "01.00.00", "a": a, "b": b,
                 "km": round(NEW_LIANOKLADI_DOMOKOS_KM, 2), "im": "0073_IM",
                 "label": "LIANOKLADION - DOMOKOS (new line, added in gr.py)"})
    out.append(f"added the new Lianokladi - Domokos line to 01, "
               f"{NEW_LIANOKLADI_DOMOKOS_KM:.2f} km")
    return out


def gr_key(lid):
    return GROUP.get(lid, (lid or "")[:2])


def gr_id_name(lid, _uop):
    return NAMES.get(gr_key(lid))


COUNTRY = {
    "iso3": "GRC", "wikidata": "Q41", "langs": ["el", "en"],
    "fix": gr_fix,
    "skip_line": lambda lid: lid in SKIP,
    "suspended": lambda _ref, ids: any(i.split("#")[0] in SUSPENDED for i in ids),
    # LARISSA's point is 1.49 km from the station.
    "name_m": 1600,
    "osm_stops": True,
    # Speed unknown, so credit is not filtered by it (CREDIT in the docstring).
    "highspeed_sections": False,
    "fixed": {f"{n:02d}.00.00": gr_key(f"{n:02d}.00.00") for n in range(1, 29)},
    "ref_display": lambda _k: "",
    "osm_ref": lambda _r: None,
    "id_name": gr_id_name,
    "im": {"0073_IM": "ΟΣΕ"},
}
