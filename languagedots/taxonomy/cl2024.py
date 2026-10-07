"""Chile, Censo 2024: P30 "habla o entiende una de las siguientes lenguas indigenas u originarias"
-> node, keyed by INE's column heading in P3_Lenguas-indigenas.xlsx (sources/cl_censo.py).

One answer per person aged 5+, the indigenous language they speak best; Spanish is not asked.

  Mapuzungun (lengua mapuche)  Mapudungun, on br.txt's `araucanian.mapuche` ("Mapuche").
  Aymara, Quechua              the shared Aimara and Quechua nodes. Chile's Quechua speakers are
                               Quechua people of the north and Bolivian and Peruvian immigrants
                               (11,033 of 39,430 do not identify as indigenous); the census names
                               no variety, so the shared node, as Peru and Bolivia use.
  Rapa Nui                     `austronesian.oceanic.rapanui` (Glottolog rapa1244, East Polynesian).
  Ckunza                       Kunza, the Atacameño (Lickanantay) language, on ar.txt's
                               `isolate.kunza` (Glottolog kunz1244, an isolate, last native speakers
                               mid-20th century; these 2,616 are revival and heritage speakers).
  Kawésqar                     new node `kawesqar.kawesqar` (Glottolog family kawe1237).
  Yagán                        `isolate.yagan` (Glottolog: an isolate). The last native speaker died
                               in 2022; the 785 here speak or understand some of it.
  Otra lengua indígena de Chile  an unnamed remainder: on `americas_other`, since the census files
                               there only languages "of another indigenous people of Chile" (the
                               Diaguita, Colla, Chango and Selk'nam peoples have no listed
                               language) and whatever else respondents meant by it.
                               EXCEPT IN ALTO BIOBIO (08314), where it depends on place: 1,330
                               people, 23.5% of the comuna aged 5+ and 1,290 of them indigenous,
                               against at most 0.6% in any other comuna of over 1,000 people.
                               Alto Biobío is the Pewenche (Pehuenche) upper Biobío, where the
                               language is called Chedungun or Pewenche rather than Mapuzugun; 2,460
                               there ticked Mapuzungun. So there the remainder is very nearly all
                               Pewenche speech, which Glottolog files as a Mapudungun dialect, and
                               it sits on the narrowest node that holds it, `araucanian` (drawn as
                               an unnamed Araucanian language), not on the named Mapuche node,
                               since the census did not name it.
  No habla ni entiende ...     Spanish, tier `derived` in countries/cl.py (spec §3.5).
  Manejo ... no declarado      not stated (103,050): not drawn, in `gap`.
"""

ALTO_BIOBIO = "08314"

NAMES = {
    "Mapuzungun (lengua mapuche)": "araucanian.mapuche",
    "Aymara": "aymaran.aymara",
    "Quechua": "quechuan.quechua",
    "Rapa Nui": "austronesian.oceanic.rapanui",
    "Ckunza": "isolate.kunza",
    "Kawésqar": "kawesqar.kawesqar",
    "Yagán": "isolate.yagan",
    "Otra lengua indígena de Chile": "americas_other",
    "No habla ni entiende ninguna lengua indígena u originaria": "indoeuropean.romance.spanish",
    "Manejo de alguna lengua indígena u originaria no declarado": None,
}

EXTRA_NODES = ["araucanian"]


def resolve(label, geo_id=None):
    if label not in NAMES:
        raise KeyError(f"cl2024: unmapped label {label!r}")
    if label == "Otra lengua indígena de Chile" and geo_id == ALTO_BIOBIO:
        return "araucanian"
    return NAMES[label]
