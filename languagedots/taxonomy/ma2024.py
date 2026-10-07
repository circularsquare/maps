"""Morocco RGPH 2024, local languages used ("langues locales utilisées", several allowed) -> node.

Keyed by the five column labels of HCP's indicator workbook (sources/ma_rgph.py). The census
tabulates only these five; nobody's answer is filed under an "other".

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Darija: Moroccan Arabic (moro1292), a node of its own beside `arabic` (a leaf many countries
    draw on), as `hassaniya` and `judeo_arabic` sit beside it. Its colour is Arabic's, a shade
    apart (tree.d/ma.txt).
  * Tachelhit: Tachelhit (tach1250), Berber.
  * Tamazight: in Morocco's census this is Central Atlas Tamazight (cent2194, "Central Moroccan
    Berber"), the Middle Atlas and eastern High Atlas language; HCP prints it beside Tachelhit and
    Tarifit, so it is not the pan-Berber "Tamazight" of the constitution. It goes on ca.txt's
    existing `berber.tamazight` leaf (label "Tamazight"), which Canada's census uses for the
    same name.
  * Tarifit: Tarifit (tari1263, Glottolog's "Tarifiyt-Beni-Iznasen-Eastern Middle Atlas Berber";
    Riffian riff1234 is its dialect), Berber.
  * Hassania: Hassaniyya Arabic (hass1238), ml.txt's `hassaniya`.
"""
AA = "afroasiatic"

NAMES = {
    "Darija": f"{AA}.darija",
    "Tachelhit": f"{AA}.berber.tachelhit",
    "Tamazight": f"{AA}.berber.tamazight",
    "Tarifit": f"{AA}.berber.tarifit",
    "Hassania": f"{AA}.hassaniya",
}


def resolve(name):
    return NAMES.get(name)
