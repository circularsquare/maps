"""Afghanistan, MICS6 2022-23, HC1B "Language of household head" -> node.
sources/af_mics.py; keyed by the labels that script writes (MICS's answers, respelled:
UZBAKI -> Uzbek, TURKMANI -> Turkmen, NOORISTANI -> Nuristani, PASHAIE -> Pashai).

CALLS (sources/af.md §0 says more):
  Dari: `indoeuropean.iranian.dari`, as before. MICS does not separate Hazaragi or Aimaq, so
    Hazara households (Bamyan, Daykundi, parts of Ghazni and Ghor) are Dari and drawn so.
  Pashai, Nuristani: the same single leaves the previous build made (tree.d/af.txt); MICS names
    each as one answer.
  Other language: MICS's "other" -> `other`. It is not spread: the languages MICS does not list
    that have speaker estimates are carved out of it first (sources/af_mics.py MINORITIES), and
    what is left (Jawzjan 3.5%, rural Herat 3.1%, small elsewhere) stays unnamed.
  Speaker estimates: the same nodes as taxonomy/af2007.py (sources/af.md §3b).
"""
IR = "indoeuropean.iranian"
NAMES = {
    "Dari": f"{IR}.dari",
    "Pashto": f"{IR}.pashto",
    "Uzbek": "turkic.uzbek",
    "Turkmen": "turkic.turkmen",
    "Nuristani": "indoeuropean.nuristani",
    "Balochi": f"{IR}.balochi",
    "Pashai": "indoeuropean.indoaryan.dardic.pashai",
    "Other language": "other",
    # languages MICS does not list, from cited speaker estimates (sources/af_mics.py)
    "Shughni (speaker estimate)": f"{IR}.shughni",
    "Wakhi (speaker estimate)": f"{IR}.wakhi",
    "Munji (speaker estimate)": f"{IR}.munji",
    "Sanglechi (speaker estimate)": f"{IR}.sanglechi",
    "Ishkashimi (speaker estimate)": f"{IR}.ishkashimi",
    "Parachi (speaker estimate)": f"{IR}.parachi",
    "Kyrgyz (speaker estimate)": "turkic.kyrgyz",
    "Gawar-Bati (speaker estimate)": "indoeuropean.indoaryan.dardic.gawarbati",
    "Brahui (speaker estimate)": "dravidian.northern.brahui",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"af2022: unmapped label {label!r}")
