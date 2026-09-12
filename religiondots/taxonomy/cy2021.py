"""
CYSTAT Census 2021 religion classification -> religiondots taxonomy.

Twelve categories, from table 1891632E of CYSTAT-DB. Shallow, but the shallowness is
informative rather than lazy: Cyprus names the **Armenian church** and the **Maronite
church** as their own answers at 2,025 and 4,486 people, because those two and the Latins
are constitutional *religious groups* under Article 2 of the 1960 constitution, each electing
its own representative to the House. Almost no census anywhere else separates communities of
that size. What it does not do is split Orthodoxy, which is 74.5% of the country and holds
the Church of Cyprus, the Greek, Russian, Romanian, Bulgarian, Georgian, Ukrainian and
Serbian churches in one cell, or split Islam, which runs from Turkish Cypriots to Syrian and
Bangladeshi migrants to Iranian Shia.

**EVERY ROW IN THIS COUNTRY IS `derived` AND IT IS THE GEOGRAPHY THAT IS DERIVED, NOT THE
CATEGORY.** The twelve counts above are exact national census figures that match UNSD table
28 to the person. What `sources/cy.py` infers is only where in Cyprus they live, from the
citizenship-group composition of each community. So the usual reading of `derived` is the
wrong way round here: nothing is a guess about *what* anyone believes, and everything is a
guess about *where*.

**THERE IS DELIBERATELY NO `COLUMNS` DICT.** §7a-i-1 rolls a derived dot up to the category
its source counted at the SAME UNIT, and CYSTAT counted no religion at any unit at all -- only
population by citizenship group. So there is nothing at a community to fall back to, and
Cyprus is the shape `tools/check_rollup.py` calls case B, alongside China and Switzerland's
canton spread. It will report Cyprus as ORPHANED and that is the correct answer, not a
missing entry: under `inferred dots: not shown` the country empties, which is an honest
statement that the map cannot say where any Cypriot's religion is.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not recorded/Not stated (3)":
        "159,835 people, 17.3% of the country, the largest not-stated share of any European "
        "country on this map. The religion question was OPTIONAL in the 2021 census, which "
        "no other census here made it, and UNSD table 28 splits the same total into 146,943 "
        "'not specified' and 12,892 'not stated'. Not a religion and not 'no religion', "
        "which is its own category at 9,591. Excluded rather than scaled up, so Cyprus "
        "draws 763,546 of 923,381. It is not spread evenly either: 13.7% of Cypriots "
        "declined, against 28.8% of other EU citizens and 29.3% of non-EU citizens, so the "
        "communities with the most migrants are also the ones the map knows least about.",
}

REVIEW = {
    "Christian Orthodox":
        "-> christianity.orthodox, the branch, NOT christianity.orthodox.canonical.greek "
        "the way gr2024.py files Greece's. Two reasons. The node gr2024.py uses is labelled "
        "'Greek Orthodox Archdiocese of America' and is an ASARB leaf about a US body, "
        "which fits Greece only because Greece is 97% one church; and Cyprus's cell is "
        "genuinely mixed. 604,165 of the 688,075 are Cypriots and belong to the autocephalous "
        "Church of Cyprus, but 59,548 are other EU citizens (Greeks, Romanians, Bulgarians) "
        "and 24,021 are non-EU (Russians, Georgians, Ukrainians, Serbs, Moldovans), and the "
        "table does not say which church any of them attends. The branch is the honest node "
        "and it is what ca2021, de2022, hu2022, ie2022, mk2021, me2023, kz2021 and at2001 "
        "all use for the same shaped cell.",
    "Armenian church":
        "-> christianity.oriental.armenian. 2,025 people. The objection this note used to "
        "raise is fixed rather than argued around: the tree's only Armenian nodes WERE two "
        "ASARB rows labelled for American dioceses, and 2026-09-08 put a general "
        "`christianity.oriental.armenian` above them with the two catholicosates under it "
        "(ask/004-am). **It stops at the general node and does not go to "
        "`.armenian.cilicia`**, even though the Armenian Prelature of Cyprus is a Cilician "
        "jurisdiction, because CYSTAT's cell says `Armenian church` and nothing about a "
        "catholicosate. What is known about the community goes in this note; what the source "
        "printed is what gets drawn. bg2021, ee2021, ge2014, pl2021 and ro2021 land at the "
        "same node for the same reason.",
    "Maronite church":
        "-> christianity.catholic.eastern. The Maronite Church is one of the 23 sui iuris "
        "churches in full communion with Rome, which is exactly what that node holds, and "
        "au2021 and nz2023 both file 'Maronite Catholic' there. Not christianity.oriental: "
        "the Maronites are in communion and the Armenians are not, and putting the two "
        "together because both are Middle Eastern would erase the one distinction the "
        "Cypriot constitution itself draws between them. 4,486 people, 4,169 of them "
        "Cypriot citizens.",
    "Muslim":
        "-> islam, the root, and NOT islam.sunni the way gr2024.py files Greece's. Greece's "
        "Muslims are the Thrace minority and Albanians, both overwhelmingly Sunni. Cyprus's "
        "19,534 are only 3,242 Cypriot citizens (the Turkish Cypriots remaining in the "
        "government-controlled area, who are Sunni Hanafi); the other 15,723 are non-EU "
        "citizens, and the census's own country-of-birth table names Syria, Egypt, "
        "Bangladesh, Pakistan and IRAN among the largest origins. Iranians are Twelver "
        "Shia. Naming a school for this cell would invent the one fact the table withholds.",
    "Anglican/Protestant":
        "-> christianity.protestant, the 'answer, not category' node. This is the least "
        "comfortable call in the file, because the cell is a lumped one and the tree keeps "
        "christianity.anglican as a SIBLING of christianity.protestant rather than a child, "
        "so whichever is picked misfiles some of the 9,621. The Anglicans are probably most "
        "of it: 8,314 of the 9,621 hold non-EU citizenship, which since Brexit is what a "
        "British resident of Pafos is, and the Church of England has had chaplaincies here "
        "since 1878. But the rest are Filipino, Nigerian and Cameroonian evangelicals and "
        "Anglican would be plainly wrong for them, whereas 'Protestant, unspecified' is at "
        "worst imprecise for the Anglicans. Mapping to the christianity root instead was "
        "considered and rejected: it would throw away the one thing the cell does say.",
    "Other Religion":
        "-> other.cy, a new node on the standard per-country pattern, ~120 of which already "
        "exist. 4,545 people. It is a real residual and not a dumping ground: Cyprus names "
        "Buddhism, Sikhism and Hinduism separately, so what is left is Baha'i, Jewish "
        "(the ethnic/religious-group table counts a Jewish community and the language table "
        "885 Hebrew speakers, but the RELIGION table has no Jewish row at all), Yazidi, and "
        "the Alevi and Ahmadi answers that did not go in the Muslim cell.",
    "Atheist/No Religion":
        "-> unaffiliated. CYSTAT merges atheism with no religion into one answer, where "
        "Croatia and Czechia ask them apart, so `secular` cannot be separated out of it and "
        "the no-religion reading is the larger part. 9,591 people, 1.04% of the country, "
        "which is the lowest irreligious share of any EU member on this map, though the "
        "17.3% who declined the question makes that a floor rather than a measurement.",
}

MAP = {
    "Christian Orthodox": "christianity.orthodox",
    "Armenian church": "christianity.oriental.armenian",
    "Maronite church": "christianity.catholic.eastern",
    "Roman Catholic": "christianity.catholic.latin",
    "Muslim": "islam",
    "Anglican/Protestant": "christianity.protestant",
    "Buddhist": "buddhism",
    "Sikh": "sikhism",
    "Hindu": "hinduism",
    "Other Religion": "other.cy",
    "Atheist/No Religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
