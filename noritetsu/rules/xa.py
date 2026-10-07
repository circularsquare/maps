"""Abkhazia's rules for build_model.py (build_model.country_rules lists what it reads).

Every train in Abkhazia is Russian Railways', from Russia (caucasus_sources.md, "Abkhazia"):
FPC's Moscow - Sukhum (304) and St Petersburg - Sukhum (479/480), and the Dioskuria tourist
electric trains Olympic Park/Sirius - Sukhum and - Guma (925/926, 929/930). OSM maps each as
one route=train per train ("Туристический электропоезд 930Ж «Диоскурия»: Сухум → Сириус"),
and Russia's rules (rules/ru.py, a number of three digits and a letter) make every one of them
a named train; so they are here too, so the same relation is the same kind of line on both
sides of the border. Their track counts through the tariff sections.
"""


def looks_like_service(tags, name, name_en):
    return True
