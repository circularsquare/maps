"""Albania, Census 2011, mother tongue (gjuha amtare) -> node. sources/al_census.py.

Keyed by the Albanian labels of INSTAT's PxWeb table 1.1.14 (prefecture branch), exactly as
data/normalized/al.csv carries them. The census prints Albanian, eight named languages, Other and
Invalid/undetermined; nothing finer exists by prefecture.

CALLS (sources/al.md says more):
  Shqip (Albanian, 2,765,610): the leaf `albanian.albanian`. Gheg and Tosk are not asked apart.
  Rumanisht / Arumanisht (3,848): Aromanian. The prefecture table writes `Rumanisht`, the national
    table `Arumanisht`, and the prefecture figures sum to the national Arumanisht figure exactly;
    the census has no Romanian answer. `romance.aromanian` (au.txt).
  Maqedonisht (Macedonian, 4,443): as printed. 3,183 are in Liqenas (Pustec, Prespa); 105 in
    Shishtavec (Kukes) are Gora people, whose Slavic speech Glottolog does not separate from
    Macedonian/Bulgarian at the language level; kept on Macedonian as the census printed it.
  Rome (Romani, 4,025): `romani.romani`, variety not stated.
  Serbokroatisht (66): `serbocroatian`, as printed.
  Tjetër (Other, 1,870): `other`. The census does not let an indigenous remainder be told apart
    from foreign languages, and Albania's own minority languages are printed by name.
  NOT_STATED: E pavlefshme / e papercaktuar (invalid or undetermined, 3,843, 0.14%): not drawn.
"""
IE = "indoeuropean"

NAMES = {
    "Shqip": f"{IE}.albanian.albanian",
    "Greqisht": f"{IE}.hellenic.greek",
    "Maqedonisht": f"{IE}.slavic.south.macedonian",
    "Rome": f"{IE}.indoaryan.romani.romani",
    "Rumanisht": f"{IE}.romance.aromanian",
    "Turqisht": "turkic.turkish",
    "Italisht": f"{IE}.romance.italian",
    "Serbokroatisht": f"{IE}.slavic.south.serbocroatian",
    "Tjetër": "other",
}
NOT_STATED = {"E pavlefshme /e papërcaktuar", "Gjithsej"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"al2011: unmapped label {label!r}")
    return NAMES[label]
