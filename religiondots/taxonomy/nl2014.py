"""CBS `Religie en kerkbezoek naar gemeente 2010-2014` -> religiondots taxonomy.

Ten source categories, all of them boxes on the Enquete Beroepsbevolking's own card except
`Geen`, which is everyone the card did not place:

    Katholiek     -> christianity.catholic.latin
    Hervormd      -> christianity.reformed.continental.hervormd
    Gereformeerd  -> christianity.reformed.continental.gereformeerd
    PKN           -> christianity.reformed.continental.pkn
    Islam         -> islam
    Joods         -> judaism
    Hindoe        -> hinduism
    Boeddhist     -> buddhism
    Anders        -> other.nl
    Geen          -> unaffiliated

NOTHING IS EXCLUDED AND THERE IS NO REFUSAL COLUMN. The nine named shares and the published
religious total are all CBS prints, and `Geen` is the complement of that total, so the ten
partition each gemeente exactly. What the table does not print is nine gemeenten, which had
fewer than 150 respondents in the five-year pool; that is a hole in the geography rather
than in the category list, and `gap` says so.

THREE REFORMED ANSWERS ARE THE REASON THIS COUNTRY IS WORTH DRAWING AT THIS DEPTH, and the
three new nodes they need are the one thing here that was flagged to Anita rather than
decided (ask/009-nl-three-dutch-reformed-nodes-hervormd-gereform.md). The alternative was
to put all three on
`christianity.reformed.continental`, which is where South Africa's NGK and Australia's
Reformed already sit; it would cost the country everything interesting about it, because
`hervormd`, `gereformeerd` and `PKN` are not three names for one thing and their maps are
not the same map.
"""

EXCLUDED = {}

REVIEW = {
    "Hervormd":
        "-> christianity.reformed.continental.hervormd, a NEW node. In 2010-2014 the "
        "Nederlandse Hervormde Kerk has not existed as a separate body for six years: it "
        "went into the Protestantse Kerk in Nederland in 2004, and a respondent who says "
        "`hervormd` is either a member of a hervormde gemeente inside the PKN, a member of "
        "the Hersteld Hervormde Kerk which refused the union, or someone naming the church "
        "they were raised in. So the node is a TRADITION and cannot be a church, and its "
        "description says so. **It is not the same answer as `PKN` and CBS does not treat "
        "it as one**: the card offers both, 7.3% of the country picks this and 5.9% picks "
        "PKN, and the two have different geographies (Staphorst, Putten and Twenterand "
        "against Dongeradeel, Ferwerderadiel and Aalten). Merging them would be a "
        "reasonable-looking edit that destroyed the country's structure.",
    "Gereformeerd":
        "-> christianity.reformed.continental.gereformeerd, a NEW node, for the same "
        "reason and with the same caveat. The word covers the Afscheiding and Doleantie "
        "line: the Gereformeerde Kerken in Nederland (into the PKN in 2004), the "
        "Gereformeerde Kerken vrijgemaakt, the Christelijke Gereformeerde Kerken and the "
        "Gereformeerde Gemeenten. **The bevindelijk gereformeerde strongholds are inside "
        "this cell and are not separable from it**: Urk is 52.3% gereformeerd and "
        "Bunschoten 51.5%, which is what the answer looks like where the Gereformeerde "
        "Gemeenten are the local church, but nothing in the table distinguishes them from "
        "a Kampen vrijgemaakt or an Amsterdam GKN member.",
    "PKN":
        "-> christianity.reformed.continental.pkn, a NEW node, and the only one of the "
        "three that names a body. The Protestantse Kerk in Nederland is a real church with "
        "a real membership roll, it is the largest Protestant body in the country, and "
        "`Continental Reformed` describes it exactly. It is placed under the Reformed line "
        "rather than under `christianity.united` even though it is a 2004 union of three "
        "churches, because two of the three were Reformed and the Lutheran third was about "
        "1% of the result; Australia's Uniting Church is on `christianity.united` because "
        "there the strands really are equal partners.",
    "Katholiek":
        "-> christianity.catholic.latin. The Old Catholic Church of the Utrecht Union is "
        "Dutch, is the see the whole Old Catholic communion is named after, and is about "
        "5,000 people; it has no box here and its members will have answered either "
        "`Katholiek` or `Anders`. At 0.03% of the country either way it cannot be "
        "separated and is not worth an authored split.",
    "Islam":
        "-> islam, the root, and not islam.sunni. Most Dutch Muslims are Sunni, and the "
        "Turkish- and Moroccan-origin populations that make up most of the total certainly "
        "are. But the Netherlands has two sizeable communities the Sunni node would be "
        "wrong about: the Alevis, perhaps a tenth of the Turkish-origin population and "
        "organised in their own federation, and the Surinamese Ahmadiyya, who arrived with "
        "the Javanese and Indo-Surinamese migration of the 1970s and built the country's "
        "first purpose-built mosques. Finland's cell is a few dozen respondents and "
        "fi2024.py could reasonably call it Sunni; this one is 780,000 people and the "
        "parent is the honest place.",
    "Joods":
        "-> judaism, the root, not a denomination under it. CBS asks one word. The Dutch "
        "Jewish community is mostly the Nederlands-Israëlitisch Kerkgenootschap, which is "
        "Orthodox, but this cell is 0.13% of the country and the survey does not ask which "
        "kehilla anybody belongs to. **The figure is also not a count of Dutch Jews**: it "
        "is people who answer `joods` to a question about religious affiliation, which is "
        "a smaller set than the community, and 's-Gravenhage's and Amstelveen's shares "
        "should be read that way.",
    "Anders":
        "-> other.nl. The residual, 4.4% of the country and larger than any single answer "
        "except Catholicism and `Geen`. spec §3.11 keeps it whole rather than splitting it "
        "across `christianity`, `other` and the non-Christian roots, because the card has "
        "boxes for four religions and four Christian answers and everything else is in "
        "here together: evangelicals, pentecostals, Baptists, Orthodox, Old Catholics, "
        "Remonstrants, Doopsgezinden, Jehovah's Witnesses, Bahá'ís, Sikhs and people "
        "describing a belief with no body behind it. Nothing in the table separates them "
        "and no other CBS release of this survey does either.",
    "Geen":
        "-> unaffiliated, and it is the largest answer in the country at 47.2%. It is not "
        "a non-response: CBS publishes the religious total and this is its complement, so "
        "every person in it answered the question and said they belong to no denomination "
        "or philosophical grouping. branches.py's line is whether a POSITION is stated, "
        "and it is not: the Dutch humanist movement is large and organised but the EBB "
        "card has no atheist, agnostic or humanist box, so **nothing in the Netherlands "
        "reaches `secular` at all**. gr2024.py, fi2024.py and ge2014.py make the same call "
        "for the same reason.",
}

MAP = {
    "Katholiek": "christianity.catholic.latin",
    "Hervormd": "christianity.reformed.continental.hervormd",
    "Gereformeerd": "christianity.reformed.continental.gereformeerd",
    "PKN": "christianity.reformed.continental.pkn",
    "Islam": "islam",
    "Joods": "judaism",
    "Hindoe": "hinduism",
    "Boeddhist": "buddhism",
    "Anders": "other.nl",
    "Geen": "unaffiliated",
}

# spec §7a-i-1: the SOURCE'S OWN coarser column for a row that has to roll up. Nothing in
# this file is derived — every category is a printed column at the drawn gemeente — so the
# table is here only to say that the three Reformed answers have no coarser published
# parent. CBS prints no `protestant` total in the maatwerk sheet; the sum of the three is
# something this project would be computing, not something CBS counted.
COLUMNS = {}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
