"""
INEI Censos Nacionales 2017, variable C5P26 -> religiondots taxonomy.

**Branch-level mapping, like ni2005.py / mx2020.py / cl2024.py.** No leaves are created; the
category's own name travels with the row in `source_category` (spec §2.4).

Eight categories at district, all measured at the level they are drawn on — nothing here is
allocated, so no row carries `tier=derived` and there is no `COLUMNS` dict.

    Católica           17,635,339   76.03%   -> christianity.catholic.latin
    Evangélica          3,264,819   14.07%   -> christianity.protestant
    Ninguna             1,180,361    5.09%   -> unaffiliated
    Cristiano             381,031    1.64%   -> christianity
    Adventista            353,430    1.52%   -> christianity.adventist
    Testigo de Jehová     173,602    0.75%   -> christianity.witnesses
    Mormones              113,659    0.49%   -> christianity.latterday
    Otra                   94,150    0.41%   -> other.pe
    ------------------------------------------------------------------
    Total              23,196,391            universe (aged 12+), EXCLUDED

**THE LIST IS THE FINDING, AND THE UNSD ORACLE DOES NOT KNOW IT EXISTS.** The oracle reports
FOUR categories for Peru, which is why `sources.md` §11t declined the country. Four is what
INEI forwarded to UNSD: its own headline release prints Católica, Evangélica, *otra religión*
and Ninguna, and that `otra religión` of 1,115,872 is exactly the bottom five rows above
added together. The microdata has all eight, at 1,874 districts, for free. §11y is the
finding and `sources/pe.py` asserts the collapse identity.

**PERU IS THE FIRST COUNTRY HERE TO CARRY `Mormones` AS A CENSUS CATEGORY OF ITS OWN.**
113,659 people named by a national census rather than inferred from a church's own returns
or buried in an `other` bucket. Chile and Mexico print an LDS line too, but Peru's is drawn
at 12,378 people per unit.

**AND `Adventista` IS THE REASON TO DRAW THE COUNTRY.** 353,430 people, 1.52% nationally, and
it is not a national figure at all — it is **two regions**, which is the part worth getting
right. The Adventist mission at Platería on the Puno shore of Lake Titicaca opened in 1898
and its schools are why the Aymara altiplano is Adventist today: San Antón 23.4%, Crucero
22.1%, Amantaní 20.9%, Huacullani 17.5%. The second cluster is 800 km north in the **Alto
Mayo** — Yantaló 17.8%, Omia 17.1%, San Fernando 16.8% — the San Martín and Amazonas
colonisation frontier. 399 districts have no Adventist at all. Nothing else in the Americas
on this map can show either cluster.

**`Cristiano` IS A SEPARATE ANSWER FROM BOTH `Católica` AND `Evangélica`, AND THAT IS WHY IT
LANDS ON THE FAMILY NODE.** See REVIEW.

Every category is an exact partition of the 12+ universe: the eight sum to the district total
on all 1,874 rows and there is no `no especificado` cell. What is NOT covered is the
under-twelves, 6,185,493 people who were never asked; countries.py carries that in `gap=`.
"""

EXCLUDED = {
    "Total": "the row's own universe (population aged 12+), not a religion category",
}

REVIEW = {
    "Cristiano":
        "-> christianity, the bare family node, and this is the call in this file that is "
        "most worth arguing. C5P26 offers `Católica`, `Evangélica` AND `Cristiano` as three "
        "separate boxes, so 381,031 people chose the third while looking at the other two. "
        "In Peruvian usage `cristiano` said in that position means non-Catholic Christian "
        "by someone who does not accept `evangélico` as a label for themselves — it names "
        "no body and asserts no denomination. Folding it into `christianity.protestant` "
        "would put a name on it INEI did not ask for and would merge an answer with the "
        "answer it was chosen INSTEAD OF; giving it a node of its own would invent a "
        "church. So it sits on the family. Its geography supports the reading rather than "
        "settling it: it is urban and coastal — Callao 4.5%, Villa el Salvador 4.0%, La "
        "Perla 4.2% — the Lima-Callao conurbation and not the evangelical countryside.",
    "Evangélica":
        "-> christianity.protestant. 3,264,819 people, 14.1%, following mx2020.py and "
        "ni2005.py exactly. In Peruvian usage *evangélico* means non-Catholic Christian and "
        "is majority Pentecostal in practice, but INEI publishes no split, so it lands on "
        "the family and not on a child (§14.4). Its geography is the Amazonian and Andean "
        "periphery rather than the cities: Elías Soplín Vargas 77.9% (San Martín), "
        "Uchuraccay 66.9% and Anchihuay 65.3% (Ayacucho), El Cenepa 65.1% and Río Santiago "
        "59.7% (the Awajún and Wampís districts of Amazonas). It is above zero in every one "
        "of the 1,874 districts, which no other category here manages.",
    "Católica":
        "-> christianity.catholic.latin, not the bare `christianity.catholic`. Peru is "
        "Latin-rite; there is no Eastern Catholic jurisdiction here to keep the parent open "
        "for. Same call as mx2020.py, cl2024.py and ni2005.py.",
    "Ninguna":
        "-> unaffiliated. INEI offers ONE no-religion box, so unlike Mexico and Czechia "
        "there is no `creyente sin adscripción` to separate and no reason to reach for "
        "`unchurched`. 5.09%, and like Nicaragua's it is NOT an urban figure — it peaks in "
        "indigenous Amazonia: Puerto Bermúdez 37.7% (Pasco), Awajún 30.5% and Pinto Recodo "
        "27.8% (San Martín), Raymondi 26.9% (the Asháninka district of Ucayali), Río "
        "Santiago 26.3%, Urarinas 24.6% (Loreto). **Reading that as secularity would be "
        "wrong.** The census gives Amazonian indigenous religions no box, and `Ninguna` is "
        "where a form with no box for your religion puts you. It is drawn as given and the "
        "country note says this in the open; §14.4 forbids splitting it on a reading.",
    "Adventista":
        "-> christianity.adventist. The call is not in doubt; it is here because the SIZE "
        "of what it buys is easy to miss, and because sources.md §11y overstated the "
        "national figure as '1.5M' when it is 353,430 — a percentage (1.52%) misread as "
        "millions. The case never rested on the total. See the module docstring.",
    "Mormones":
        "-> christianity.latterday. 113,659 people, 0.49%, and the first time a census on "
        "this map names the Latter-day Saints as a category of their own rather than "
        "leaving them inside an `other` bucket. Its geography is the southern coast — "
        "Pacocha 2.2% (Moquegua), Islay 2.1% and Mollendo 1.9% (Arequipa), Pocollay 1.7% "
        "(Tacna) — which is the mid-century mission field, and it is absent from 914 "
        "districts entirely. The peaks are low, so most of its dots draw as §4.3 presence "
        "rings rather than as a visible colour.",
    "Otra":
        "-> other.pe. Only 94,150 people, 0.41% — small because this census names five "
        "non-Catholic Christian bodies rather than sweeping them into one box. Its "
        "geography is the Amazon frontier and it is very sharp (Yavarí 20.7%, Tournavista "
        "19.2%, San Pablo 18.9%). The Israelitas del Nuevo Pacto Universal are the strongest "
        "candidate for what is in it and it is NOT split — see the node's own text in "
        "branches.py, and §14.4.",
}

MAP = {
    "Católica": "christianity.catholic.latin",
    "Evangélica": "christianity.protestant",
    "Cristiano": "christianity",
    "Adventista": "christianity.adventist",
    "Testigo de Jehová": "christianity.witnesses",
    "Mormones": "christianity.latterday",
    "Ninguna": "unaffiliated",
    "Otra": "other.pe",
}

# No COLUMNS dict (spec §7a-i-1): every category is measured at the district it is drawn on,
# so no row is `derived` and nothing ever needs to roll up.


def resolve(category):
    """religiondots branch for an INEI category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
