"""NIS 2019 GPCC religion (Table 2.5.1) -> religiondots taxonomy.

Four categories plus the universe total, at province. **The shallowest category list on this
map after Sri Lanka's six**, and unlike Sri Lanka's it is not paired with a fine geography —
Cambodia is §3.9's trade made badly in both directions at once, and it is NIS's ceiling
rather than a choice (see `sources/kh.md` §2).

**Nothing here needed a new node except the residual.** A census that distinguishes
Buddhist, Muslim and Christian and puts everything else in one box is asking the question at
the level of world religions, and the tree has had those since the first source. What
Cambodia adds to this map is not taxonomic depth: it is that the Cham Muslim belt of the
Mekong and the Tonle Sap, and the highland provinces where a fifth of the population answers
none of the three, are drawn at all.

**THE ONE DECISION WORTH ARGUING WITH IS `Other`**, which is 85% highland indigenous religion
by NIS's own account and is nevertheless filed on `other.kh`. The reasoning is in the node's
own note in `branches.py` and in REVIEW below.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the province's own population total, not a category.",
}

REVIEW = {
    "Buddhist":
        "-> buddhism, the PARENT, and deliberately not buddhism.theravada. 15,101,607 "
        "people, 97.1% of the country — the largest share any single category holds in any "
        "country on this map. Cambodian Buddhism is Theravada in overwhelming proportion "
        "and the Mahanikay and Thommayut orders are both Theravada, so filing it as "
        "Theravada would almost certainly be true. It is still not what the source says: "
        "NIS offers one cell labelled `Buddhist` and asks for a religion rather than a "
        "school, so the sub-school distribution is inferred from general knowledge and not "
        "measured, and §2 forbids inventing the distinction at ingest time. **This is "
        "lk2024.py's call, made for the same reason on the same tradition** — and in2011.py "
        "makes it a third time on India's Buddhists. The tree can hold the vehicles apart "
        "the moment a source separates them; no census on this map does. "
        "Its geography is the whole country: 99.8% in Kampong Speu, 99.7% in Svay Rieng, "
        "and below 90% in only three of twenty-five provinces — Mondul Kiri 70.4%, "
        "Ratanak Kiri 73.4% and Tbong Khmum 88.1%, which are exactly the highland and Cham "
        "provinces the other three categories live in.",
    "Muslims":
        "-> islam, with no branch, because the census gives none. 317,934 people, 2.04%. "
        "Cambodia's Muslims are the **Cham**, descendants of the Champa kingdom of central "
        "Vietnam who arrived in waves from the fifteenth century, plus a smaller Chvea "
        "(Malay-descended) population. They are overwhelmingly Sunni of the Shafi'i school, "
        "**and the one distinction a Cambodian source could usefully draw is one no census "
        "here draws**: the *Kan Imam San*, a minority tradition centred on Udong that keeps "
        "a distinctively Cham liturgy and prays once weekly, against the majority who "
        "follow mainstream Sunni practice. Nothing in Table 2.5.1 separates them and "
        "nothing here invents the split (§2.4). "
        "**Its geography is the single clearest thing on Cambodia's map.** Tbong Khmum is "
        "11.8% Muslim — 91,667 people, both the largest share and the largest absolute "
        "number in the country — followed by Kratie 6.6%, Kampong Chhnang 5.8%, Stung "
        "Treng 4.7% and Koh Kong 4.6%. That is the Mekong upstream of Phnom Penh and the "
        "Tonle Sap, which is where the Cham settled and still fish and farm. Against 0.1% "
        "in Svay Rieng and Kampong Speu. "
        "**Read the national figure as a floor for one specific historical reason**: the "
        "Cham were targeted for destruction under Democratic Kampuchea, losing an estimated "
        "third to a half of their population between 1975 and 1979, and the community's "
        "recorded size is still shaped by that.",
    "Christian":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row. 50,338 people, 0.32%. The category is "
        "defined by what it is not: NIS names no body at all, and the population is at "
        "least three unlike things — the Catholic church, historically Vietnamese-speaking "
        "and almost entirely destroyed in the 1970s; the Protestant and evangelical "
        "churches that grew fast after 1993; and the Korean and American mission churches "
        "of Phnom Penh. Filing it on `christianity.protestant` or `.catholic` would assert "
        "a body the source does not name. lk2024.py's call on Sri Lanka's `Other "
        "Christian`, for the same reason. "
        "**Its geography is the interesting part and it is not the capital.** In absolute "
        "terms Phnom Penh leads with 11,410, but by share the top two provinces are "
        "**Mondul Kiri at 4.0% and Ratanak Kiri at 2.1%** — twelve and six times the "
        "national rate — which is evangelical mission among the Bunong and Tampuan, the "
        "same highland peoples who make up most of the `Other` cell. The two categories "
        "are competing for the same population and the map shows both.",
    "Other":
        "-> other.kh. 82,332 people, 0.53%, and **the sharpest residual geography on this "
        "map**: Ratanak Kiri 23.2% and Mondul Kiri 21.2% hold 70,000 of it between them "
        "while fifteen provinces are printed as 0.0%. "
        "**This is Cambodia's highland indigenous religion and it is not filed as such.** "
        "NIS's own gloss is *\"mainly… the local religious system of the highland tribal "
        "groups and a few minority religious groups from other countries\"* — the Bunong, "
        "Tampuan, Jarai, Kreung, Brao and Kavet animist traditions. The argument for "
        "`other.kh` over `indigenous` is that the cell is demonstrably not only that: "
        "Kampong Chhnang has 4,739 in it (0.9%) with no highland population, and Phnom Penh "
        "2,282. NIS's *\"mainly… and a few…\"* says the same thing. Filing all of it as "
        "indigenous asserts a magnitude the source does not publish; splitting it invents "
        "one (§14.4). **The node wanted is an `indigenous` child for these traditions, and "
        "`branches.py`'s note records that it is wanted so a later source is a lookup** "
        "(§2.4). Per §3.11.",
}

MAP = {
    "Buddhist": "buddhism",
    "Muslims": "islam",
    "Christian": "christianity",
    "Other": "other.kh",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
