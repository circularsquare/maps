"""
LAPOP AmericasBarometer `q3c`/`q3cn` (Bolivia, single-country files, waves 2010-2023) -> taxonomy.

**The same card as `gt2023.py`, `ec2023.py` and `co2023.py`.** Read `gt2023.py` for the shared
arguments and `ec2023.py` for the card change, which reaches Bolivia a round later than Ecuador:
in 2016/17 the Bolivian file still offers Witnesses (code 12) and Mormons (6) beside the new `Otro`
(77), and from 2018 it offers only `Otro`. Jews (10) have no box from 2016. Only what is different
about Bolivia is written out. Named for the last wave in the pool.

    as drawn, all nine departments, post-stratified to the 2024 census
    70.57%  Católico                                    -> christianity.catholic.latin
    11.91%  Evangélica y Pentecostal                    -> christianity.evangelical
     7.36%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     6.20%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     1.10%  Otro                                        -> other.bo
     0.97%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.64%  Testigos de Jehová                          -> christianity.witnesses
     0.56%  Religiones Orientales no Cristianas         -> other.bo
     0.50%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday
     0.15%  Religiones Tradicionales                    -> other.bo   (NOT indigenous, see REVIEW)
     0.04%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism

Every row is `modelled` in §7's sense.
"""

EXCLUDED = {}

REVIEW = {
    "Religiones Tradicionales":
        "-> other.bo, NOT `indigenous` as in gt2023.py, sv2023.py and ec2023.py, and for the "
        "reason co2023.py gives on its own data. **Its 24 Bolivian respondents do not sit "
        "anywhere in particular**: every department has one to four of them and the spatial "
        "chi-square is p=0.86, and 10 of the 24 are the 2014 wave. The card's printed examples "
        "(Candomblé, Vudú, Rastafari, Maya religions, Umbanda) name no Andean or Amazonian "
        "tradition, so an Aymara or Quechua respondent who keeps the Pachamama rites has no box "
        "that says so. Drawing it on the indigenous family node would claim the survey found "
        "Bolivia's indigenous religion; it found 0.15% of respondents choosing a box written "
        "for another region. Drawn at the national rate inside each department's residual.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. 7.36%, and it PASSES the "
        "split-half at the nine departments on its own (median +0.575, chi-square 1e-26), so "
        "unlike Colombia it is drawn on department shares. **Bolivians move between this box "
        "and `Evangélica y Pentecostal` from round to round** (this one 4.6% in 2016, 10.5% in "
        "2018, 4.8% in 2023; the union rises smoothly from 14.1% to 22.7%), but not by place: "
        "the two boxes' department shares are unrelated (Spearman +0.03). The union passes "
        "more strongly (+0.875), and both boxes pass alone, so they are kept as the card "
        "offers them.",
    "Evangélica y Pentecostal":
        "-> christianity.evangelical, gt2023.py's argument. 11.91%, drawn on its own department "
        "shares (median +0.850, chi-square 1e-36). 8.5% in 2010 and 17.9% in 2023. The 1992 "
        "census's `evangélicos`, every non-Catholic Christian, orders the nine departments the "
        "way this box and the Protestant one together do (Spearman +0.917, sources/bo_checks.py).",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, gt2023.py's reason (the card's gloss is a believer without a church). "
        "6.20%, drawn on its own department shares (median +0.708, chi-square 9e-30).",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 0.97%, 105 respondents, and it passes both tests "
        "(median +0.815, chi-square 8e-17) with no sampling cluster holding more than 3% of "
        "it, so it is drawn on department shares: 2.2% of La Paz and 1.9% of Chuquisaca, none "
        "in Pando's 1,018 interviews.",
    "Testigos de Jehová":
        "-> christianity.witnesses. 0.64% and a floor: no box in 2018 or 2023. Fails the "
        "split-half (+0.225), national rate.",
    "Iglesia de los Santos de los Últimos Días (Mormones)":
        "-> christianity.latterday. 0.50%, a floor for the same reason, national rate.",
    "Otro":
        "-> other.bo. Exists from 2016 and absorbs the withdrawn small denominations from 2018; "
        "ec2023.py. Kept apart from the other two answers on this node in `source_category`.",
    "Religiones Orientales no Cristianas":
        "-> other.bo, following mx2020.py and gt2023.py: one bucket from Buddhism to Baha'i "
        "that cannot be split without inventing the split. 0.56%.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. Five respondents in 13,882, in the rounds that offered the box. Drawn "
        "because the partition is closed, not because the survey can see Bolivian Jewry.",
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
    "Religiones Tradicionales": "other.bo",
    "Religiones Orientales no Cristianas": "other.bo",
    "Otro": "other.bo",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, nothing was counted anywhere.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
