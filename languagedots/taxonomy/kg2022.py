"""Kyrgyzstan 2022 census, native language by ethnic group (sources/kg_census.py) -> node.

Keyed by data/normalized/kg.csv's `source_category`: the table's named columns ("Kyrgyz",
"Russian", "Uzbek", "Dungan", "other languages") and "own language: <group>" for the column
"language of one's own ethnic group", which names a different language on every row.

CALLS.
  own language, Turk: Turkish. Kyrgyzstan's Turks are Meskhetian (Ahiska) Turks, deported from
    Georgia in 1944; their speech is an eastern Anatolian Turkish dialect and the census calls
    the group simply Turks. 11,534 people.
  own language, Kurd: Kurdish. Soviet Kurds speak Kurmanji; the node is the existing `kurdish`
    leaf, which no country has split. 9,052 people.
  own language, Karachay and Balkar: both Karachay-Balkar, one language with two names; the census
    keeps the peoples apart and they merge here. 1,015 people.
  own language, Moldovan: the existing `moldovan` node (Ukraine's), as the census names the people.
  own language, Roma: Romani (`romani.romani`), the leaf other countries use for Romani unsplit.
  own language, Chinese: `sinitic`, the node every country uses for Chinese not split by variety.
    37 people.
  own language, peoples of India and Pakistan: `other.india_pakistan`, a named leaf under `other`.
    The census groups the two countries' peoples (mostly medical students in Bishkek, Osh and Chui)
    and names no language; Hindi, Urdu, Punjabi, Malayalam, Tamil and more are all possible, and
    Indo-Aryan and Dravidian share no node short of the root. As Azerbaijan's "Jewish". 5,688.
  own language, other groups: the own language of every group a unit's rows do not list (and
    Naryn's printed "other" group row): `other`. 7,071 people, 0.10%.
  "other languages": `other`. It holds Uzbek in the volumes without an Uzbek column (Issyk-Kul,
    Talas, Chui, Bishkek: 837 people by Book II's oblast rows) and Dungan outside Naryn, so it
    is not guessed into either. 14,352 people.
"""
ND = "nakhdaghestanian"
SL = "indoeuropean.slavic"

LANGS = {
    "Kyrgyz": "turkic.kyrgyz",
    "Russian": f"{SL}.east.russian",
    "Uzbek": "turkic.uzbek",
    "Dungan": "sinotibetan.sinitic.dungan",
    "other languages": "other",
}
# group (sources/kg_census.py GROUPS' English) -> its own language's node
OWN = {
    "Kyrgyz": "turkic.kyrgyz",
    "Uzbek": "turkic.uzbek",
    "Russian": f"{SL}.east.russian",
    "Dungan": "sinotibetan.sinitic.dungan",
    "Tajik": "indoeuropean.iranian.tajik",
    "Uyghur": "turkic.uyghur",
    "Kazakh": "turkic.kazakh",
    "Turk": "turkic.turkish",
    "Azerbaijani": "turkic.azerbaijani",
    "Tatar": "turkic.tatar",
    "Kurd": "indoeuropean.iranian.kurdish",
    "Turkmen": "turkic.turkmen",
    "Korean": "koreanic.korean",
    "Ukrainian": f"{SL}.east.ukrainian",
    "German": "indoeuropean.germanic.continental.german",
    "Kalmyk": "mongolic.kalmyk",
    "Lezgin": f"{ND}.lezgic.lezgian",
    "Dargin": f"{ND}.dargwa",
    "Chechen": f"{ND}.nakh.chechen",
    "Agul": f"{ND}.lezgic.agul",
    "Karachay": "turkic.karachay_balkar",
    "Balkar": "turkic.karachay_balkar",
    "Kumyk": "turkic.kumyk",
    "Avar": f"{ND}.avarandic.avar",
    "Belarusian": f"{SL}.east.belarusian",
    "Moldovan": "indoeuropean.romance.moldovan",
    "Roma": "indoeuropean.indoaryan.romani.romani",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Chinese": "sinotibetan.sinitic",
    "peoples of India and Pakistan": "other.india_pakistan",
    "other groups": "other",
}
NAMES = {**LANGS, **{f"own language: {k}": v for k, v in OWN.items()}}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"kg2022: unmapped label {label!r}")
    return NAMES[label]
