"""
INIDE VIII Censo de Población y IV de Vivienda 2005, variable P13 -> religiondots taxonomy.

**Branch-level mapping, like mx2020.py / ca2021.py / br2010.py.** No leaves are created; the
category's own name travels with the row in `source_category` (spec §2.4).

Eight categories at municipio, all measured at the level they are drawn on — nothing here is
allocated, so no row carries `tier=derived` and there is no `COLUMNS` dict. The whole file is
one screen because the question was one line on the form:

    Católica            2,652,985   58.47%   -> christianity.catholic.latin
    Evangélica            981,795   21.64%   -> christianity.protestant
    Ninguna               711,310   15.68%   -> unaffiliated
    Otra                   74,101    1.63%   -> other.ni
    Morava                 73,902    1.63%   -> christianity.moravian
    Testigo de Jehová      42,587    0.94%   -> christianity.witnesses
    Musulmán                  321    0.01%   -> islam
    Judaísmo                  199    0.00%   -> judaism
    ------------------------------------------------------------------
    Total               4,537,200            universe, EXCLUDED

**THE LIST IS THE FINDING, AND ITS SHAPE IS UNUSUAL.** Latin American censuses that go beyond
`Católica / Evangélica / Otra` normally add the same three: Adventists, Jehovah's Witnesses
and the Latter-day Saints — that is Mexico's list and Chile's. Nicaragua adds **Jehovah's
Witnesses and the Moravians**, and nothing else. A national census printing a box for a church
of 74,000 people, while printing none for Adventists or Anglicans, is a statement about the
Moravian Church's standing on the Caribbean coast rather than about its size: it is the
established church of the Miskito and Creole coast, it ran the schools and the clinics there
under both the British protectorate and the Somoza period, and its congregations are the
institution the RAAN and RAAS are organised around. The census asks about it because on half
of Nicaragua's territory it is the answer.

**AND IT IS THE ONLY CENSUS ON THIS MAP THAT ASKS.** `christianity.moravian` has been carried
by Jamaica (0.68%), Trinidad (0.27%) and the Caribbean small islands, always as a national
rounding. Nicaragua's 73,902 are 1.63% nationally and **53.3% of Prinzapolka, 50.9% of Puerto
Cabezas and 43.6% of Waspám** — the first place on this map where the Moravians are anybody's
plurality, and the reason the country was worth wiring (`sources.md` §11x).

`Evangélica` -> `christianity.protestant`, following mx2020.py exactly. In Nicaraguan usage
*evangélico* means non-Catholic Christian and is dominated by Pentecostals; it names no body,
so it lands on the family and not on a child. Splitting it towards Pentecostalism would be
inventing a composition INIDE does not publish (§14.4).

**THERE IS NO `no especificado` AND THAT IS REAL, NOT A GAP.** The eight categories sum to the
municipio total on all 153 rows and to 4,537,200 nationally, which is REDATAM's own count of
every record in the universe. P13 has no missing code — the tabulation covers everybody aged
5 and over. What is NOT covered is the under-fives, who were never asked; that is 604,898
people and countries.py carries it as `gap=`, not as a §3.5 undercount.
"""

EXCLUDED = {
    "Total": "the row's own universe (population aged 5+), not a religion category",
}

REVIEW = {
    "Evangélica":
        "-> christianity.protestant. 981,795 people, 21.6%, and the largest single "
        "Protestant answer in Nicaragua. As in Mexico, *evangélico* here means non-Catholic "
        "Christian rather than any named body, and it is majority Pentecostal in practice — "
        "but INIDE publishes no split, so it sits on the family node. Its geography is the "
        "interior agricultural frontier (Waslala 37.3%, Murra 36.5%, Paiwas 33.8%), not the "
        "cities.",
    "Morava":
        "-> christianity.moravian. The call itself is not in doubt; it is here because the "
        "SIZE of what it buys is easy to miss. This is the only census on this map that "
        "names the Moravians at a geography where they are a plurality, and it is the "
        "reason Nicaragua was drawn. See the module docstring.",
    "Católica":
        "-> christianity.catholic.latin, not the bare `christianity.catholic`. Nicaragua is "
        "Latin-rite; there is no Eastern Catholic jurisdiction here to keep the parent open "
        "for. Same call as mx2020.py and br2010.py.",
    "Ninguna":
        "-> unaffiliated. INIDE offers ONE no-religion box, so unlike Mexico and Czechia "
        "there is no `creyente sin adscripción` to separate and no reason to reach for "
        "`unchurched`. 15.7% and it is NOT an urban figure: it peaks in the northern "
        "mountains and the frontier (Santa María 39.8%, Murra 36.1%, Wiwilí de Jinotega "
        "32.9%) and bottoms out on the Moravian coast (Waspám 0.41%, Puerto Cabezas 1.22%) "
        "— the opposite of the usual shape and worth not mis-reading.",
    "Otra":
        "-> other.ni. 74,101 people whose geography is one coastline: Corn Island 44.1%, "
        "Laguna de Perlas 21.4%, Bluefields 17.5%. The strong reading is Anglicans and "
        "Jamaican-mission Baptists, neither of which has a box on this form, and it is NOT "
        "acted on — see the node's own text in branches.py, and §14.4.",
}

MAP = {
    "Católica": "christianity.catholic.latin",
    "Evangélica": "christianity.protestant",
    "Morava": "christianity.moravian",
    "Testigo de Jehová": "christianity.witnesses",
    "Judaísmo": "judaism",
    "Musulmán": "islam",
    "Ninguna": "unaffiliated",
    "Otra": "other.ni",
}

# No COLUMNS dict (spec §7a-i-1): every category is measured at the municipio it is drawn on,
# so no row is `derived` and nothing ever needs to roll up.


def resolve(category):
    """religiondots branch for an INIDE category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
