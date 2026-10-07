"""Afghanistan, MRRD provincial profiles (c. 2006-07), language of the village majority -> node.
sources/af_mrrd.py; keyed by the labels that script writes, which are the profiles' own
spellings ("Pashtu", "Uzbeki", "Turkmeni", "Pashaie").

CALLS (sources/af.md says more):
  Spelling variants merged in sources/af_mrrd.py: Pashto/Pashtu, Turkmani/Turkmen/Turkmeni,
    Baluchi/Balochi, Pashayi/Pashaie, Nooristani/Nuristani, Uzbek/Uzbeki.
  Dari: `indoeuropean.iranian.dari`, us.txt's node. The profiles never name Hazaragi; Hazara
    villages (Bamyan, Daykundi, parts of Ghazni and Ghor) are filed as Dari and drawn so.
  Pashaie: a new leaf under Dardic. Glottolog has Pashayi as four languages (pash1270); the
    profiles name one, and Dardic is where the conventional classification puts it.
  Nuristani: a new leaf directly under Indo-European. Glottolog's Nuristani (nuri1243) is a
    branch of five or so languages (Kati, Waigali, Ashkun, Prasun, Tregami); the profiles name
    one "Nuristani", so it is one leaf labelled "Nuristani languages".
  "X and Y, one figure": the profile gives one percentage for two languages. Kandahar's and
    Helmand's "Balochi and Dari" go on Iranian; Herat's "Turkmeni and Uzbeki" on Turkic. Group
    nodes draw as "language not named" (0.2 million people in all).
  Kabul city and Herat's "Dari and Pashtu": until 2026-10-06 on Iranian, 7.7 million people,
    90% of the city dots washed out. Now split into Dari and Pashtu in sources/af_mrrd.py by the
    Asia Foundation 2006 national first-language residual (sources/af.md §3a).
  Other language: Ghazni, Paktika, Paktia's "some other language" -> `other`.
  Not described: the part of a province the profile's figures leave out; not drawn, in gap.
"""
IR = "indoeuropean.iranian"
NAMES = {
    "Dari": f"{IR}.dari",
    "Dari (Tajiks, the profile's ethnic group)": f"{IR}.dari",
    "Pashtu": f"{IR}.pashto",
    "Balochi": f"{IR}.balochi",
    "Pashaie": "indoeuropean.indoaryan.dardic.pashai",
    "Nuristani": "indoeuropean.nuristani",
    "Uzbeki": "turkic.uzbek",
    "Turkmeni": "turkic.turkmen",
    # Kabul city and Herat's joint figure, split in sources/af_mrrd.py (split_unsplit) at the
    # ratio that brings the map's Dari:Pashto to the Asia Foundation 2006 first-language 49:40
    "Dari, by the national first-language residual": f"{IR}.dari",
    "Pashtu, by the national first-language residual": f"{IR}.pashto",
    "Balochi and Dari, one figure": IR,
    "Turkmeni and Uzbeki, one figure": "turkic",
    "Other language": "other",
    # Languages the profiles never name, from cited speaker estimates (sources/af_mrrd.py
    # MINORITIES and BRAHUI, 2026-10-06; sources/af.md 3b). Shughni, Wakhi and Ishkashimi are
    # by.txt's nodes, Kyrgyz kg.txt's, Brahui tree.txt's, so colours match across the borders.
    # Rushani is folded into Shughni (Glottolog rush1239 is its dialect; no separate figure).
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
MODELLED = {k for k in NAMES if k.endswith("(speaker estimate)")}
SKIP = {"Not described"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"af2007: unmapped label {label!r}")
