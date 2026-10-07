"""Montenegro, Popis 2023, mother tongue (maternji jezik) -> node. sources/me_census.py.

Keyed by MONSTAT's own labels (Montenegrin, Latin script) in Tabela 3 of the open-data portal,
exactly as data/normalized/me.csv carries them. 25 labels and "Ne zeli da se izjasni" (does not
wish to declare); one answer per person.

CALLS (sources/me.md says more):
  Crnogorski (Montenegrin), Srpski (Serbian), Bosanski (Bosnian), Hrvatski (Croatian): four
    leaves, as rs2022, hr2021 and ba2013 keep them. Glottolog files all four standards as
    dialects of Serbian-Croatian-Bosnian (sout1528); the answer largely follows nationality
    (Tabela 1 of the same census), and the map says so.
  Srpsko-Hrvatski (12,999) and Hrvatsko-Srpski (233): the existing `serbocroatian` and
    `croatoserbian` leaves, kept apart as in Croatia and Bosnia.
  Bosnjacki (Bosniak, 2,030): the existing `bosniak` leaf (ba2013), apart from Bosanski.
  Jugoslovenski (Yugoslav, 178): the existing `yugoslav` leaf (lu2021).
  Crnogorski-Srpski (1,336), Srpski-Crnogorski (1,210), Crnogorski-Srpski-Bosanski-Hrvatski
    (1,721): three new leaves. Compound answers the census prints in columns of their own; the
    two orders of Montenegrin-Serbian are kept apart, as ba.txt keeps the two orders of
    Bosnian-Croatian-Serbian.
  Bokeljski (Bokelj, 186): a new leaf. The regional name for the speech of the Bay of Kotor
    (Herceg Novi, Kotor, Tivat); a census label, no glottocode.
  Goranski (Gorani, 226): a new leaf. The Slavic speech of the Gora region (Glottolog gora1268,
    "Gora (Serbian-Macedonian)", a dialect filed under Macedonian); here Gorani who moved to
    Podgorica, Bar, Berane and Rozaje. A sibling of Macedonian rather than a child, which would
    turn Macedonian into a group drawn washed out.
  Maternji (1,408): literally "mother tongue", an answer that names no language. It sits on South
    Slavic (`indoeuropean.slavic.south`), drawn as "language not named": every municipality
    where it is published is Serbian- and Montenegrin-speaking (Podgorica 534, Niksic 195, Herceg
    Novi 118, Kotor 103), and it is absent from all six Albanian- and Bosniak-majority ones.
    Not guessed into Serbian or Montenegrin, which is the choice the answer declines to make.
  Romski (4,658): `romani.romani`, variety not stated, as mk2021 and rs2022.
  Ruski, Ukrajinski, Bjeloruski, Makedonski, Turski, Engleski, Njemacki, Albanski: as printed.
  Ostali jezici (other languages, 3,109) and Ostalo (other, 200): both `other`. MONSTAT prints no
    breakdown of either; no indigenous remainder to keep apart.
  NOT_STATED: "Ne zeli da se izjasni" (10,691, 1.7%), the gap; "Ukupno", the universe row; and
    "z (suppressed)", sources/me_census.py's row for each municipality's cells MONSTAT printed
    as `z` (716 people).
"""
IE = "indoeuropean"
SL = f"{IE}.slavic.south"

NAMES = {
    "Crnogorski": f"{SL}.montenegrin",
    "Srpski": f"{SL}.serbian",
    "Bosanski": f"{SL}.bosnian",
    "Hrvatski": f"{SL}.croatian",
    "Srpsko-Hrvatski": f"{SL}.serbocroatian",
    "Hrvatsko-Srpski": f"{SL}.croatoserbian",
    "Bošnjački": f"{SL}.bosniak",
    "Jugoslovenski": f"{SL}.yugoslav",
    "Crnogorski-Srpski": f"{SL}.montenegrinserbian",
    "Srpski-Crnogorski": f"{SL}.serbianmontenegrin",
    "Crnogorski-Srpski-Bosanski-Hrvatski": f"{SL}.montenegrinserbianbosniancroatian",
    "Bokeljski": f"{SL}.bokelj",
    "Goranski": f"{SL}.gorani",
    "Makedonski": f"{SL}.macedonian",
    "Maternji": SL,
    "Albanski": f"{IE}.albanian.albanian",
    "Romski": f"{IE}.indoaryan.romani.romani",
    "Ruski": f"{IE}.slavic.east.russian",
    "Ukrajinski": f"{IE}.slavic.east.ukrainian",
    "Bjeloruski": f"{IE}.slavic.east.belarusian",
    "Turski": "turkic.turkish",
    "Engleski": f"{IE}.germanic.english",
    "Njemački": f"{IE}.germanic.continental.german",
    "Ostali jezici": "other",
    "Ostalo": "other",
}
NOT_STATED = {"Ne želi da se izjasni", "Ukupno", "z (suppressed)"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"me2023: unmapped label {label!r}")
    return NAMES[label]
