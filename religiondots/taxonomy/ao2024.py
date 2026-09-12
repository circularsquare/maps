"""INE Angola 2024 RGPH religion or spirituality -> religiondots taxonomy.

Twenty-one bodies plus the universe total, at municipality. **The deepest religion question
any African census on this map asks**, and the only one anywhere that names four
African-founded churches individually instead of pooling them: Tocoísta, Kimbanguista, Bom
Deus and Josafat between them are 1,204,617 people, 3.5% of Angola.

What makes the list unusual is that it is a DENOMINATION question that keeps `Protestante`
and `Evangélica` as answers beside the named bodies, so a Methodist, an Adventist and an
Assembly of God member each have a box and 2,239,792 people still answer only `Protestante`.
That is the same shape as Kenya's and Brazil's, with more named churches than either.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "População com 2 ou mais anos":
        "the unit's own population total, not a category.",
    "Não sabe/Não respondeu":
        "767,283 people, 2.22%, who were asked and did not answer or did not know. A "
        "non-answer and not a religion, so it is off the tree per spec §3.5; it is read "
        "and carried in ao.csv so that a municipality's bodies reconcile against its own "
        "universe, which is the check that catches a figure read into the wrong column.",
}

REVIEW = {
    "Sem religião":
        "-> unaffiliated. 4,173,740 people, 12.10% -- and **this cell is the one thing on "
        "the Angolan map that should not be read at face value.** Its geography is not an "
        "urban secular one. The peaks are **Iona 60.6%, Virei 53.3%, Curoca 48.6%** in "
        "Namibe and Cunene, with Quilenda 54.1% and Condé 45.9% in interior Cuanza Sul; "
        "Luanda province is 14.9% and its densest municipalities are below that. Those "
        "south-western units are the **transhumant pastoralist country of the Kuvale, "
        "Himba, Mucubal and Nyaneka-Nkhumbi**, some of the least missionised ground in "
        "the country, and Iona is a national park whose population is almost entirely "
        "herders. "
        "Beside it, `Animista` is **44,370 people, 0.13%**, which for a country of 34 "
        "million is not a credible count of traditional practice and is an order of "
        "magnitude below Ghana's 3.25% or Benin's. The likeliest reading is that Angola's "
        "form offers one box for religion, that traditional practice is not what the box "
        "is understood to mean, and that a herder who keeps the ancestors answers `no "
        "religion` rather than `animist`. "
        "**It is left on `unaffiliated` and not moved**, because moving it would be "
        "inventing a number the census did not produce, and because the same cell "
        "genuinely does hold Luanda's irreligious. What is asserted here is only that the "
        "two cells should be read together: see `indigenous.african`'s own note, which "
        "says the African traditional count is a floor everywhere, and spec §3.5.",
    "Animista":
        "-> indigenous.african. 44,370 people, 0.13%, and a floor rather than a count -- "
        "see the `Sem religião` note above, which is where most of Angola's traditional "
        "practice is likely to have been recorded. Its own geography is real and worth "
        "keeping: **Tomboco 5.6% and Cuimba 3.2% in Zaire**, on the Bakongo side of the "
        "lower Congo, and a second cluster in the Lunda-Chokwe east (Saurimo 1.3%, "
        "Chiluage 1.3%, Muangueji 1.2%). Two unrelated traditional complexes, which is "
        "the same shape Benin has and which INE gives one cell.",
    "Judaica":
        "-> judaism, and it is the least comfortable call in this file. **34,711 people, "
        "and its geography is the diamond belt rather than the capital**: Lunda Norte is "
        "0.40% against 0.10% nationally, peaking at Chitato 1.12%, Cazombo 0.96% on the "
        "Zambian border and Lucapa 0.71%, while Luanda -- where an actual Jewish community "
        "of Israeli and Portuguese business families does live -- is a rounding error. "
        "Angola has no historic Jewish population; a figure of thirty-five thousand "
        "concentrated in the Lunda mining districts is far more likely to be **Israelite "
        "and Judaising movements out of the Congo basin**, which are numerous and which no "
        "Angolan source names, or a coding artefact in a box most respondents will never "
        "have used. "
        "It goes to `judaism` because that is the box INE printed and spec §2.4 keeps a "
        "source's own category; nothing here is asserted about halakhic or communal "
        "Judaism, and the figure should not be quoted as Angola's Jewish population.",
    "Nova apostólica":
        "-> christianity.newapostolic, a node added for Angola. **515,929 people, 1.50%**, "
        "and the New Apostolic Church's largest published count anywhere on this map. It "
        "is an Irvingite body -- the Catholic Apostolic line of the 1830s, reorganised in "
        "Hamburg in 1863 around a living apostolate -- and it is neither Pentecostal, "
        "evangelical nor Stone-Campbell restorationist. "
        "**Its geography settles the identification**: Moxico Leste 8.00%, Moxico 5.39% "
        "and Cuando 4.93%, peaking at Ninda 23.8% and Lumbala Nguimbo 17.2%. That is the "
        "Zambian border, and Zambia is one of the New Apostolic Church's largest countries "
        "in the world. A church that had crossed from Portugal or Brazil would look like "
        "Luanda; this one looks like Mwinilunga. "
        "**Two peers are NOT on this node and were deliberately left where they are**: "
        "`ee2021.py` sends Estonia's New Apostolic Church to `christianity.restorationist` "
        "and `au2021.py` sends Australia's to `christianity.other`, both of which predate "
        "this node. Moving them is a one-line change in each file plus a re-scatter, and "
        "doing it here would silently alter two countries nobody asked about -- the same "
        "call `christianity.africaninstituted.harrist` made about Benin.",
    "Universal do reino de Deus":
        "-> christianity.pentecostal.charismatic, which is exactly where `br2010.py` puts "
        "the Igreja Universal do Reino de Deus in the country that founded it. 338,475 "
        "people, 0.98%: neo-Pentecostal, 1977, prosperity theology and no classical "
        "Pentecostal lineage, so it belongs with the later movements rather than beside "
        "the Assembleias de Deus. **Its geography is the Brazilian export pattern**, urban "
        "and Luanda-centred: Rangel 3.65%, Sambizanga 3.07%, Maianga 2.91% against 0.98% "
        "nationally, and near-absent in the rural east.",
    "Josafat":
        "-> christianity.pentecostal.charismatic. 105,383 people, 0.31%, and the entry in "
        "this list that needs its history explained. **Igreja Josafat is what the Igreja "
        "Maná operated as in Angola between 2009 and 2017** -- Maná is a Portuguese "
        "neo-Pentecostal church founded in Lisbon in 1984, it was banned in Angola in "
        "2008, and its congregations resumed the following year under an Angolan "
        "registration in Kilamba Kiaxi. The ban was lifted in 2017 and Maná resumed under "
        "its own name, but Josafat is still a registered body with its own congregations "
        "and the census still counts it, seven years on. "
        "The node is the same one Maná's parent tradition takes: neo-Pentecostal, "
        "prosperity-preaching, no classical lineage. **Urban and Luanda-shaped**, like "
        "IURD beside it -- Luanda province 0.66% against 0.31% nationally, peaking in "
        "Sambizanga and Rangel -- which is what a church planted from Lisbon into the "
        "capital in the 2000s should look like.",
    "Mensagem dos ultimos Tempos":
        "-> christianity.pentecostal, the parent and not `.trinitarian`. 62,134 people, "
        "0.18%, the smallest named body Angola counts. The Igreja Mensagem do Último "
        "Tempo dates itself to **Jeffersonville, Kentucky, 1933**, which is William "
        "Branham's town and the year he began preaching: this is the Message movement, "
        "the following that formed around Branham's healing campaigns and continued after "
        "his death in 1965. "
        "It is not put on `.trinitarian` because **Branham taught a non-trinitarian "
        "godhead and baptism in the name of Jesus only**, which is the Oneness position; "
        "it is not put on `.oneness` either, because the Message is its own restorationist "
        "movement rather than a body of the Oneness Pentecostal denominations that node "
        "was built for, and because INE's cell says nothing about what its members hold. "
        "The parent asserts only what the source does. "
        "**Its geography is Congolese**: Lunda Norte 1.63% against 0.18% nationally, "
        "peaking at Canzar 4.73% and Cambulo 3.36% on the DRC border. The Message is very "
        "large in the Congo, and this cell is its overspill.",
    "Protestante":
        "-> christianity.protestant. 2,239,792 people, 6.49%, and an ANSWER rather than a "
        "church: INE offers Methodist, Baptist, Adventist, Evangelical and the Assembleias "
        "de Deus as their own boxes, so this holds the Congregationalists of the Ovimbundu "
        "plateau, the Baptist mission churches of the north and everyone who named no body. "
        "**It is a rural eastern and southern answer**: Cuando 21.3%, Moxico 20.8%, Lunda "
        "Norte 14.7% and Bié 14.6%, peaking at Camaxilo 45.3% and Cuilo 44.3%, against "
        "2.9% in Luanda. Where the mission map is thinnest, the answer is broadest.",
    "Evangélica":
        "-> christianity.evangelical, on `ke2019.py`'s precedent: INE prints it beside "
        "`Protestante` rather than under it, so it is a distinct answer and not a "
        "sub-heading. 2,121,051 people, 6.15%. **Its geography is the southern and central "
        "highlands** -- Bié 12.0%, Cubango 11.4%, Huíla 11.3%, Cunene 10.2% -- which is "
        "the country of the Igreja Evangélica Congregacional em Angola and the Igreja "
        "Evangélica Sinodal, the successors of the American Board and Swiss missions.",
    "Católica":
        "-> christianity.catholic, the parent rather than `.latin`. **15,097,949 people, "
        "43.77%, the largest single body in Angola by a factor of four** and Latin rite "
        "throughout, which the census does not say -- the same call `ke2019.py` and "
        "`mw2018.py` make. "
        "**The 1600s are still on the map.** The Catholic share is 70.6% in Cunene, 69.5% "
        "in Benguela and 63.8% in Huíla, and reaches 97.0% in Viti Vivali and 95.3% in "
        "Mupa, against 35.8% in Luanda and under 20% across the Lunda east. That is the "
        "old Portuguese coastal and southern mission field, and its edge is where the "
        "Protestant missions got in first.",
    "Metodista":
        "-> christianity.methodist. 584,615 people, 1.69%, and **the sharpest mission "
        "geography in the country.** Bengo province is 17.15% Methodist against 1.69% "
        "nationally, and inside it **Quicunzo is 75.3% and Muxaluando 64.9%** -- the "
        "highest single-body concentrations anywhere in Angola. Nambuangongo is 36.7% and "
        "Bula-Atumba 20.4%. That is the Methodist Episcopal mission's Dembos and "
        "Nambuangongo field, opened from Luanda in 1885, and it has not moved in a hundred "
        "and forty years.",
    "Assembleia de Deus pentecostal":
        "-> christianity.pentecostal.trinitarian, which is where `br2010.py` puts the "
        "Assembleia de Deus in Brazil and where the Angolan body's own lineage runs from. "
        "**3,225,632 people, 9.35% -- the second largest single body in the country**, "
        "ahead of the Protestant answer and ahead of the Evangelicals. Luanda province is "
        "17.7% and Icolo e Bengo 18.3%, peaking at Gabela 28.0% and Belas 24.1%: this is "
        "the urban and near-coastal church, and it is the one that has grown.",
    "Adventista":
        "-> christianity.adventist. 1,827,359 people, 5.30%, which is a far larger "
        "Adventist share than any other country on this map records. **Malanje is 15.8% "
        "and Cuale 65.0%, Massango 53.5%, Calandula 34.6%** -- one contiguous block across "
        "northern Malanje -- with a second in Huambo and Moxico. INE's cell does not "
        "separate Seventh-day from the smaller Adventist bodies and none is inferred.",
    "Baptista":
        "-> christianity.baptist. 52,821 people, 0.15%, and **it is much smaller than "
        "Angola's Baptist history would lead you to expect**: the Baptist Missionary "
        "Society's São Salvador mission of 1878 is the oldest Protestant work in the "
        "country and the Bakongo Baptist churches were central to the north. The figure is "
        "0.15% in Uíge and Zaire as well, so this is not a northern cell that went "
        "somewhere else geographically. The likeliest reading is that the historic mission "
        "churches answer `Protestante`, and that `Baptista` catches only those who name the "
        "denomination; nothing here is moved on that reading.",
    "Testemunha de Jeová":
        "-> christianity.witnesses. 712,323 people, 2.07%, and the Watch Tower Society's "
        "own reporting makes Angola one of its largest fields in Africa. Urban and "
        "north-western -- Luanda 4.68%, Icolo e Bengo 4.82%, peaking at Cazenga 6.72% and "
        "Mulenvos 6.24%.",
    "Islâmica/Muçulmana":
        "-> islam, undivided, because INE prints one cell and says nothing about school. "
        "135,003 people, 0.39%. **Its geography is the diamond fields**: Lunda Norte is "
        "2.71% against 0.39% nationally, peaking at Chitato 4.68%, Lucapa 4.48% and Dundo "
        "4.15%, with Luanda second at 0.61%. That is the West African trading population "
        "of the mining districts -- Malian, Senegalese, Guinean and Mauritanian -- and a "
        "Lebanese and North African commercial community in the capital, which is what "
        "every account of Islam in Angola describes and which the census map draws "
        "unaided.",
    "Bom Deus":
        "-> christianity.africaninstituted.bomdeus, a node added for Angola. 339,044 "
        "people, 0.98%. The Igreja Fraternidade Evangélica de Pentecostes na África em "
        "Angola, known everywhere as Bom Deus, was begun by **Simão Lutumba in 1981** as "
        "an Angolan extension of the Congolese Nzambe Malamu and separated from it in the "
        "1990s. It is Pentecostal in practice and African-founded in origin, and the tree "
        "files it by the second: `christianity.africaninstituted` is defined by who "
        "founded a church and outside which mission, not by its worship. "
        "**It is the one of Angola's four named African churches with no regional "
        "geography** -- Cuanza Norte 2.55%, Icolo e Bengo 2.02%, Lunda Norte 1.85%, "
        "Malanje 1.82%, and no municipality above 6.4%. A church founded in Luanda in the "
        "1980s and spread nationally, against three that are still sitting on the mission "
        "and prophetic maps of the 1920s and 1940s.",
    "Outra religião":
        "-> other.ao. 435,663 people, 1.26%. **Its geography says what is in it**, on "
        "§9r's Chittagong rule: Zaire province is 7.62% against 1.26% nationally, peaking "
        "at Luvo 13.9%, Mbanza Kongo 11.0% and Cuimba 10.3%, with a second cluster in "
        "interior Cuanza Sul. Mbanza Kongo is the old capital of the Kongo kingdom and "
        "Luvo is the border post on the road to Matadi; that corner of Angola holds dozens "
        "of small Kongo prophetic churches that INE's twenty-one boxes do not name. Read "
        "the northern half of this cell as `christianity.africaninstituted` that could not "
        "be brought out. Per source, per spec §3.11.",
}

MAP = {
    "Católica": "christianity.catholic",
    "Protestante": "christianity.protestant",
    "Evangélica": "christianity.evangelical",
    "Metodista": "christianity.methodist",
    "Baptista": "christianity.baptist",
    "Adventista": "christianity.adventist",
    "Assembleia de Deus pentecostal": "christianity.pentecostal.trinitarian",
    "Universal do reino de Deus": "christianity.pentecostal.charismatic",
    "Josafat": "christianity.pentecostal.charismatic",
    "Mensagem dos ultimos Tempos": "christianity.pentecostal",
    "Testemunha de Jeová": "christianity.witnesses",
    "Nova apostólica": "christianity.newapostolic",
    "Tocoísta": "christianity.africaninstituted.tocoist",
    "Kimbanguista": "christianity.africaninstituted.kimbanguist",
    "Bom Deus": "christianity.africaninstituted.bomdeus",
    "Islâmica/Muçulmana": "islam",
    "Judaica": "judaism",
    "Animista": "indigenous.african",
    "Sem religião": "unaffiliated",
    "Outra religião": "other.ao",
}

# spec 7a-i-1: the level this source COUNTED each node at, so a dot inferred below it rolls
# up instead of vanishing. Angola measures every one of its twenty at the municipality it
# draws, so the column is the node itself and nothing rolls.
#
# **AND UÍGE'S AND MOXICO LESTE'S FILLED ROWS DELIBERATELY GET NO COLUMN**, which is the
# comment `tools/check_rollup.py` asks a country to leave rather than adding one. Their ten
# unprinted bodies are `derived` (sources/ao.py `fill()`), and neither province measures any
# ancestor of `christianity.methodist` at the municipality -- there is no Christianity cell,
# only the twenty-one bodies. Giving them a column would roll a Methodist dot up to a figure
# nobody counted; with none, `inferred dots: not shown` removes them, which is the honest
# behaviour. They will appear in check_rollup's `still gone` and that is correct.
COLUMNS = {v: v for v in MAP.values()}


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
