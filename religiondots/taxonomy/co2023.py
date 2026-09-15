"""
LAPOP AmericasBarometer `q3c`/`q3cn` (Colombia, waves 2010-2023) -> religiondots taxonomy.

**The same card as `gt2023.py`, `sv2023.py` and `ec2023.py`.** Read `gt2023.py` for the shared
arguments and `ec2023.py` for the card change after 2016 (codes 6, 10 and 12 withdrawn, 77
added), which applies here unchanged: Colombia has no 2016 wave, so `Otro` is zero in 2010,
2012 and 2014 and Witnesses, Mormons and Jews are zero in 2018 and 2023. Only what is
different about Colombia is written out. Named for the last wave in the pool.

    pooled, weighted, all 26 sampled departments
    71.55%  Católico                                    -> christianity.catholic.latin
     8.93%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     7.58%  Evangélica y Pentecostal                    -> christianity.evangelical
     7.24%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     1.50%  Otro                                        -> other.co
     1.01%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.90%  Testigos de Jehová                          -> christianity.witnesses
     0.69%  Religiones Orientales no Cristianas         -> other.co
     0.50%  Religiones Tradicionales                    -> other.co   (NOT indigenous, see REVIEW)
     0.07%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism
     0.04%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday

Every row is `modelled` in §7's sense.
"""

EXCLUDED = {}

REVIEW = {
    "Religiones Tradicionales":
        "-> other.co, NOT `indigenous` as in gt2023.py, sv2023.py and ec2023.py. **In "
        "Colombia's pool this answer is not where indigenous Colombians are.** 18 of its 37 "
        "respondents are in Bogotá, the capital, on 17% of the sample, and 23 of the 37 are "
        "the 2014 wave; Cauca, Nariño and La Guajira, the sampled departments with large "
        "indigenous populations (Puracé, Cumbal, Manaure), contribute none. A box answered "
        "mostly in Bogotá and mostly in one wave reads as the card's "
        "other traditions, not as Nasa, Pastos or Wayuu practice, so drawing it on the "
        "indigenous family node would put a claim on the map the survey does not support. "
        "0.5%, drawn at the national rate inside each department's residual either way.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. 7.24%, nearly equal to the "
        "evangelical box beside it, and it FAILS the split-half at both the 22 departments "
        "(median +0.03) and the 6 design regions, so it is drawn at the national rate inside "
        "each department's residual. §11ap suggested testing it together with "
        "`Evangélica y Pentecostal` because Colombians move between the two boxes from wave "
        "to wave (code 2 runs 5.5-9.7%, code 5 5.4-9.9%). The union does pass (+0.58), but "
        "more weakly than the evangelical box alone (+0.69), and the two boxes' department "
        "shares are unrelated (Spearman +0.08), so the union's pass is the evangelical box's "
        "and the two are kept as the card offers them.",
    "Evangélica y Pentecostal":
        "-> christianity.evangelical, gt2023.py's argument. 7.58%, drawn on its own "
        "department shares (median split-half +0.69, chi-square 9e-21). Its wave series is "
        "the clearest movement in the country, 6.2% in 2010 to 9.7% in 2023.",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, gt2023.py's reason (the card's gloss is a believer without a church). "
        "8.93%, drawn on its own department shares (median +0.47, chi-square 4e-11).",
    "Testigos de Jehová":
        "-> christianity.witnesses. 0.90% and a floor, ec2023.py's withdrawn box. **It passes "
        "the rank test (p=0.048) and fails the spatial chi-square (p=0.29)**, so it is drawn "
        "at the national rate: spec §12's Sweden rule, a rank pass on a thin column with no "
        "difference between units behind it. 44 of its 66 respondents are the 2014 wave.",
    "Otro":
        "-> other.co. Exists from 2018 only and absorbs the withdrawn small denominations; "
        "ec2023.py. Kept apart from the other two answers on this node in `source_category`.",
    "Religiones Orientales no Cristianas":
        "-> other.co, following mx2020.py and gt2023.py: one bucket from Buddhism to Baha'i "
        "that cannot be split without inventing the split. 0.69%.",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 1.01%, against 8.93% believing without a church.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. Five respondents in 7,532, in the three waves that offered the box. Drawn "
        "because the partition is closed, not because the survey can see Colombian Jewry.",
    "Iglesia de los Santos de los Últimos Días (Mormones)":
        "-> christianity.latterday. Three respondents, and a floor for the withdrawn-box "
        "reason.",
}

MAP = {
    "Católico": "christianity.catholic.latin",
    "Evangélica y Pentecostal": "christianity.evangelical",
    "Protestante, Protestante Tradicional o Protestante no Evangélico": "christianity.protestant",
    "Testigos de Jehová": "christianity.witnesses",
    "Iglesia de los Santos de los Últimos Días (Mormones)": "christianity.latterday",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)": "unchurched",
    "Agnóstico o ateo (no cree en Dios)": "secular",
    "Judío (Ortodoxo, Conservador o Reformado)": "judaism",
    "Religiones Tradicionales": "other.co",
    "Religiones Orientales no Cristianas": "other.co",
    "Otro": "other.co",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, nothing was counted anywhere.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
