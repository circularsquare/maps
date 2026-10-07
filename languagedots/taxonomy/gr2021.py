"""Greece, 2021: language labels written by sources/gr_build.py -> node.

Greece's census has asked no language since 1951, so every label here is the build's own, not a
census category (sources/gr.md says how each is derived):
  * "Greek": Greek citizens the minority figures do not place elsewhere, plus the foreign
    citizens the retention share leaves out.
  * minority languages: Turkish (Thrace, Rhodes, Kos), Pomak, Romani, Aromanian, Arvanitika,
    Macedonian and Bulgarian (the Slavic speakers of Greek Macedonia, split by place after
    Trudgill 2000).
  * immigrant languages: the 2021 census's foreign citizens, each country on its main language
    (France's table, sources/fr_build.py COUNTRY_LANG, with gr_build.GREECE_OVERRIDES).
  * "Other": foreign citizens of a country Eurostat does not name, and stateless residents.
"""
import fr2023

NAMES = dict(fr2023.NAMES)
NAMES.update({
    # Glottolog arva1236 Arvanitika Albanian: a sibling of Albanian, like Arbereshe in it.txt
    "Arvanitika": "indoeuropean.albanian.arvanitika",
    # Glottolog files Pomak (poma1238) as a dialect of Bulgarian; drawn as a sibling so that
    # Bulgarian stays a language node and is not washed out as a group
    "Pomak": "indoeuropean.slavic.south.pomak",
    "Aromanian": "indoeuropean.romance.aromanian",
    "Romani": "indoeuropean.indoaryan.romani.romani",
    "Other": "other",
})
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"gr2021: unmapped label {label!r}")
    return NAMES[label]
