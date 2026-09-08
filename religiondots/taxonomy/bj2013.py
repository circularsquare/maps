"""INStaD Benin RGPH-4 (2013) religion -> religiondots taxonomy.

Ten categories plus a computed residual, at commune. **The first source on this map to
count an African tradition under its own name**: `Vodoun` is a cell of its own, separate
from `Autres traditionnelles`, and the two have different geographies rather than being a
category and its leftovers.

The list is short but it is not shallow, and the two places it is deeper than any other
African source here are the two that matter for Benin: it separates Vodun from the other
traditional religions, and it separates the Celestial Church of Christ — an African
Instituted Church founded in Porto-Novo — from a generic `Autres chrétiens`. What it costs
is everything else: there is no cell for Buddhism, Hinduism, the Bahá'ís or Judaism at any
geography, and no branch given for Islam or for the Protestants beyond the Methodists.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Non déclaré (calculé)":
        "not a category INStaD prints. `sources/bj.py` computes it as the unit's population "
        "minus the ten published shares, because those shares sum to 98.81% nationally and "
        "the arithmetic should be visible rather than implied. It is non-response — RGPH-4's "
        "religion tabulations carry exactly ten categories wherever they appear — and §3.5 "
        "marks non-response rather than filling it, so 120,826 people, 1.21% of Benin, are "
        "counted and not drawn.",
}

REVIEW = {
    "Vodoun":
        "-> indigenous.african.vodun, a node added for Benin and the first named African "
        "tradition on this map. 1,160,279 people, 11.6%. "
        "**The mapping is easy and the reading is not, so the honest points first.** "
        "Vodun is not a residual here and not an outsider's word: it is the name the "
        "religion goes by in Benin, it is on the census form, and 10 January is a national "
        "holiday for it. That makes Benin the one African country on this map where the "
        "traditional religion is a named answer competing with the named churches rather "
        "than a `Traditionalist` box competing with a whole column of them. "
        "**Its geography overturns the obvious expectation.** The heartland is the Couffo — "
        "Djakotomey 69.1%, Toviklin 66.1%, Lalo 56.2%, Aplahoué 55.3%, Klouékanmè 50.5%, "
        "and the department as a whole 56.5% — which is Adja country in the south-west, not "
        "the Fon plateau. Abomey, the capital of the kingdom of Dahomey and the place a "
        "reader will look first, is 23.5%; Agbangnizoun beside it is 39.2%; Ouidah, the "
        "other name everyone knows, is in the Atlantique at 12.1% departmental. So the "
        "royal and Atlantic-trade sites are NOT where the census finds the most Vodun. "
        "**And 11.6% is a floor, for the reason sources.md §11b gives for the whole "
        "continent** — the box is exclusive of `Catholique` and `Islam`, and in Benin the "
        "same person is very commonly both. `sources/bj.md` §7 has the direct evidence, "
        "which this map has never had before: Benin's `Aucune` cell has an Atacora "
        "geography that irreligion cannot explain.",
    "Catholique":
        "-> christianity.catholic, the parent rather than .latin. 2,552,159 people, 25.5%, "
        "the largest single named body in Benin. Latin rite throughout and the census does "
        "not say so, which is the call ke2019.py and mw2018.py both make. Its distribution "
        "is urban and southern: Cotonou 51.2%, Abomey-Calavi 49.5%, and then the Collines "
        "— Bantè 49.4%, Glazoué 47.8% — where the Société des Missions Africaines worked "
        "from the 1890s. It falls to 1.2% in Karimama.",
    "Protestant méthodiste":
        "-> christianity.methodist. 342,442 people, 3.4%. The Église Protestante Méthodiste "
        "du Bénin, out of the Wesleyan mission that reached Ouidah in 1843 and is the "
        "oldest Protestant body in the country — which is why it has a cell of its own "
        "while every other Protestant is pooled. **Its geography is the Ouémé valley and "
        "the coast**: Aguégués 19.0%, Dangbo 15.3%, Sèmè-Kpodji 12.2%, Sô-Ava 10.4%, and "
        "Dassa-Zoumè 14.2% inland. Under 0.5% across the Muslim north.",
    "Autres protestants":
        "-> christianity.protestant, which holds an ANSWER and not a church. 340,906 "
        "people, 3.4%. Everything Protestant that is not Methodist and not filed by the "
        "respondent as `Autres chrétiens`: the Assemblies of God, the Baptists, the "
        "Adventists, the Union des Églises Évangéliques. No branch is inferred, because "
        "the census gives none and the four traditions in that list sit in four different "
        "places on the tree.",
    "Chrétien céleste":
        "-> christianity.africaninstituted, and it is only the SECOND source ever to feed "
        "that node. 676,032 people, 6.75%. The Église du Christianisme Céleste — the "
        "Celestial Church of Christ — founded by Samuel Oschoffa in Porto-Novo in 1947, and "
        "the largest of the Aladura churches of the Bight of Benin. The node's own note "
        "named the Aladura churches when Kenya created it and had nobody in them; this is "
        "the entry that fills it. "
        "**It is a single named body and it is 6.75% of a country**, which no other source "
        "here comes close to for an African Instituted Church — Kenya's node holds five "
        "bodies pooled. **Its geography is its founding**: Sô-Ava 30.2%, Akpro-Missérété "
        "25.8%, Bonou 24.4%, Avrankou 23.7%, Zè 22.7%, Dangbo 22.1% — the Ouémé valley and "
        "the lagoons around Porto-Novo, seventy years on and still centred where Oschoffa "
        "started. It is under 0.2% in the north.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 2,769,217 people, 27.7%, "
        "and **the most concentrated distribution in Benin by a long way**: Karimama 95.4%, "
        "Malanville 94.4%, Ségbana 92.3%, Kalalé 91.2% — the Niger valley and the "
        "Alibori — against 0.3% in Djakotomey, a three-hundred-fold range across 77 "
        "communes. Overwhelmingly Sunni Maliki, and Tijaniyya where a Sufi order is "
        "involved; `islam.sunni` would be an inference rather than a reading. "
        "The northern communes are the Dendi and Fulbe of the river trade and the Bariba of "
        "Borgou; Benin's Muslim south is a different and older thing, the Yoruba Muslim "
        "communities of Porto-Novo, and the census cannot tell them apart.",
    "Autres chrétiens":
        "-> christianity.other. 944,839 people, 9.4%, and it is the residual this source "
        "hands over. Smaller and less troubling than Ghana's or Malawi's identically-named "
        "cell, precisely because `Chrétien céleste` has already been lifted out of it: what "
        "remains is the rest of Benin's very large prophetic and independent sector — the "
        "Christianisme Céleste's many splinters, the Église du Christianisme Prophétique, "
        "the Cherubim and Seraphim, the Aladura bodies that are not Celestial, and the "
        "newer Pentecostal-charismatic ministries. "
        "**A good deal of it is `christianity.africaninstituted` and none of it can be "
        "moved there**, which is gh2021.py's and mw2018.py's position with a smaller cell. "
        "Its geography peaks at 22.8% in Za-Kpota and 22.1% in Pobè, on the Fon and Yoruba "
        "plateau rather than in the Celestial heartland — which is some evidence it is not "
        "simply more Celestial Church by another name.",
    "Autres traditionnelles":
        "-> indigenous.african, the node Ghana added, and the placement is the point. "
        "260,448 people, 2.6%. **This is not Vodun's leftovers — it is a different religion "
        "in a different half of the country.** Boukoumbé 54.1%, Cobly 42.1%, Tanguiéta "
        "36.6%, Toucountouna 20.0%, Natitingou 14.3%: every one of the top six is in the "
        "Atacora, the north-western highlands, where Vodun is 6.3% departmental. These are "
        "the traditions of the Bètammaribè (Otammari) and their neighbours — the people "
        "whose *tata somba* fortified houses are the region's landmark — and of the "
        "Boo and Yoa further east. "
        "It stays on the parent node rather than getting one of its own because no source "
        "names any of them individually (§2.4: a node earns its place by being countable), "
        "and because §6.6 lets a branch that carries dots be a category. **The REVIEW note "
        "a future source would want**: `indigenous.african.otammari` is the child to add, "
        "and until then those people are here. "
        "Read as a floor, like every African traditional cell — see the `Vodoun` note.",
    "Autres religions":
        "-> other.bj. 259,446 people, 2.59%. See the node's own note: an unusually honest "
        "tail, because the two large things that would normally hide in it — Vodun and the "
        "Celestial Church — have their own cells. Per §3.11.",
    "Aucune":
        "-> unaffiliated. 582,155 people, 5.8%. One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer — the call ke2019.py, "
        "mw2018.py, hr2021.py and mk2021.py all make. "
        "**AND THIS IS THE ONE MAPPING IN THE FILE THAT IS PROBABLY WRONG FOR PART OF ITS "
        "PEOPLE, WHICH IS WHY IT IS WRITTEN DOWN RATHER THAN ONLY MAPPED.** Its geography "
        "is not a national-irreligion geography at all: Toucountouna 45.2%, Kérou 26.9%, "
        "Cobly 20.4%, Tanguiéta 18.9%, Matéri 18.8%, Natitingou 17.3% — the Atacora again, "
        "the poorest and least urban department in Benin, and 19.0% of it against 2.9% of "
        "the Couffo and 2.8% of the Littoral. Cotonou, the one place a secularising urban "
        "population would show, is low. "
        "The Atacora is also where `Autres traditionnelles` is highest, so the same "
        "communes lead on both answers, and the straightforward reading is that a "
        "traditional practice with no church, no name on the form and no weekly assembly "
        "is being reported by some respondents as `Aucune`. **That cannot be corrected "
        "here** — moving people between two published cells would invent a magnitude "
        "(§14.4) — so it is drawn as INStaD published it and said plainly in "
        "`sources/bj.md` §7 and in the country's public note.",
}

MAP = {
    "Vodoun": "indigenous.african.vodun",
    "Catholique": "christianity.catholic",
    "Protestant méthodiste": "christianity.methodist",
    "Autres protestants": "christianity.protestant",
    "Chrétien céleste": "christianity.africaninstituted",
    "Islam": "islam",
    "Autres chrétiens": "christianity.other",
    "Autres traditionnelles": "indigenous.african",
    "Autres religions": "other.bj",
    "Aucune": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
