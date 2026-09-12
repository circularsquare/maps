"""INE Cabo Verde 2021 RGPH religion or spirituality -> religiondots taxonomy.

Fifteen answers on all twenty-two concelhos, for the population aged 15 and over. Twelve are
named religions, nine of them individual Christian churches; the other three are a catch-all,
`no religion` and a refusal. The list is a DENOMINATION question with no `Protestante` box
of its own: INE names the Church of the Nazarene, the
Adventists, the Assembly of God, the New Apostolic Church, the Universal Church of the
Kingdom of God, the Jehovah's Witnesses and the Latter-day Saints individually, and the only
place a generic Protestant answer could land is `Outra`, which is 1.16%.

**THE COUNTRY'S OWN RELIGION IS `Racionalismo Cristão`** — 6,129 people, 1.74%, the fourth
largest named religion after Catholic, Adventist and the Nazarenes, and the reason to want
this country on the map. See its REVIEW entry.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Não sabe / Não respondeu":
        "1,311 people, 0.37% of the 15+ population, who were asked and did not answer or "
        "did not know. A non-answer and not a religion, so it is off the tree per spec "
        "§3.5; it is read and carried in cv.csv so that a concelho's fifteen rows "
        "reconcile against its own printed total, which is the check that catches a figure "
        "read into the wrong column. **The much larger hole in this country is not this "
        "row**: INE asked nobody under 15, so 138,739 people, 28.24% of Cabo Verde, are "
        "outside the table entirely. That is `gap` in countries.py.",
}

REVIEW = {
    "Igreja do Nazareno / Protestante":
        "-> christianity.holiness.nazarene. 6,175 people, 1.75%, and the arguable part is "
        "the slash. The label could be read as *Nazarene OR any Protestant*, in which case "
        "the cell is a mixed one and belongs on `christianity.protestant`. **Three things "
        "say it is a gloss and not a merge.** First, every other slash in INE's list joins "
        "two names for ONE body -- `Islâmica / Muçulmano`, `Jesus Cristo dos Santos dos "
        "Últimos Dias / Mórmons` -- and the only slash that joins two different things is "
        "the refusal row. Second, **INE's own English transcription of this table, "
        "forwarded to the UN Demographic Yearbook, calls this row `Church of Nazarene` and "
        "nothing else** (`tools/oracle.py \"Cabo Verde\"`). Third, the Church of the "
        "Nazarene has been the Protestant church of Cabo Verde since 1901, brought back "
        "from New England by returning emigrants, and *protestante* is what Cape Verdeans "
        "call it; there is no separate Protestant box for the word to be competing with. "
        "**Its geography is the mission's own history**: Brava 8.10% and Mosteiros 3.90%, "
        "the Fogo-and-Brava emigration islands the first missionaries landed on, against "
        "0.07% in São Lourenço dos Órgãos in interior Santiago. A generic Protestant cell "
        "would not look like that.",
    "Racionalismo Cristão":
        "-> spiritualism. 6,129 people, 1.74%. **Christian Rationalism is a spiritualist "
        "doctrine founded in Santos, Brazil, in 1910 by Luís de Mattos**, out of a Kardecist "
        "group and then explicitly against Kardec on mediumship; it describes itself as a "
        "philosophy rather than a religion and has no clergy and no ritual. It reached Cabo "
        "Verde in 1911, and this is the one country on earth where it is a mass movement. "
        "**It sits on `spiritualism` and not on `spiritualism.kardecist`**, which is the "
        "call: same 19th-century movement, and a schism from Kardec rather than a branch of "
        "him, so a sibling rather than a child, in the same relation `br2010.py` uses when "
        "it separates Brazil's `Espírita` from its `Espiritualista`. A node of its own was "
        "considered and not taken -- it would be a legend row no other country uses "
        "(AGENT_BRIEF §3), and `spiritualism` is already drawn for Antigua, Australia, "
        "Brazil and Canada, so nothing is hidden by putting it there. "
        "**Its geography is the movement's own account of itself**: 3,988 of the 6,129 are "
        "in São Vicente, 6.86% of that concelho's adults, and the next four are Tarrafal de "
        "São Nicolau 4.41%, Boa Vista 3.54%, Paul 3.17% and Sal 2.38% -- São Vicente, São "
        "Nicolau, Boa Vista, Santo Antão and Sal, which is the island list the centres' own "
        "histories name. It draws nothing at all in São Salvador do Mundo or Santa Catarina "
        "do Fogo.",
    "Islâmica / Muçulmano":
        "-> islam. 4,616 people, 1.31%, and the reason it is here rather than in MAP is "
        "that **it is not a Cape Verdean population and the map should not be read as "
        "saying it is.** It is 3,668 men to 948 women, close to four to one, and its "
        "geography is Boa Vista 6.63% and Sal 4.37% -- the two tourist islands -- then "
        "Praia 1.88%, against 0.04% in Ribeira Brava and São Lourenço dos Órgãos. That is "
        "West African labour migration, mostly Senegalese and Guinean, arriving with the "
        "resort building of the last twenty years. The node is right and the reading is the "
        "thing worth writing down.",
    "Sem religião":
        "-> unaffiliated. 54,814 people, 15.55%, up from 10.8% in 2010. **Its geography is "
        "a city and a coast**: São Vicente 38.20%, which is Mindelo, then Tarrafal de São "
        "Nicolau 28.68%, Sal 24.34% and Boa Vista 20.11%, against 0.77% in São Salvador do "
        "Mundo and 1.20% in São Lourenço dos Órgãos, both interior Santiago. A fifty-fold "
        "spread across a country of half a million, and the same axis every other category "
        "here divides on: the northern and tourist islands against rural Santiago. It goes "
        "to `unaffiliated` rather than `secular` because INE's box is *sem religião*, an "
        "absence, and the form offers no atheist or agnostic option to sit beside it "
        "(branches.py's own distinction, and `lc2022.py`'s note on the two).",
    "Outra":
        "-> other.cv. 4,090 people, 1.16%, and a small residual because the twelve named "
        "boxes above it are unusually many for a country of half a million. **Its geography "
        "does not name what is in it**, on §9r's Chittagong rule: Boa Vista 2.73%, Santa "
        "Cruz 1.73%, Praia 1.62% and São Miguel 1.60% against 0.16% in São Lourenço dos "
        "Órgãos, a spread of about seventeen to one but with no cluster that points at a "
        "body. The two readings it will hold are a generic Protestant answer, which has "
        "nowhere else to go on this form, and the small Chinese, Hindu and West African "
        "communities of Praia and the tourist islands. Neither is separable, so per spec "
        "§3.11 it is drawn whole.",
    "Nova Apastólica":
        "-> christianity.newapostolic, and the label is INE's spelling, not a typo of this "
        "file's: `Apastólica` for *Apostólica*, uncorrected across all twenty-three "
        "workbooks (spec §2.4, transcribe). 1,719 people, 0.49%, and **almost all of them "
        "are on Fogo** -- Santa Catarina do Fogo 11.52%, Mosteiros 3.81%, São Filipe 2.83% "
        "-- with 0.00% in São Lourenço dos Órgãos and 0.01% in São Domingos. The New "
        "Apostolic Church is the same body `ao2024.py` maps for Angola, where it is also a "
        "Lusophone-Atlantic import, and Fogo is the pattern its missions take: one island, "
        "arrived at once.",
    "Assembleia de Deus":
        "-> christianity.pentecostal.trinitarian. 730 people, 0.21%, the second smallest "
        "row on the form. The Cape Verdean Assembleia de Deus descends from the Brazilian "
        "body rather than from the US Assemblies of God, which is why it goes on the family "
        "node and not on `...trinitarian.assemblies-of-god`, the USRC's American "
        "denomination; `ao2024.py` makes the same call for the same reason.",
    "Deus é Amor":
        "-> christianity.pentecostal. 300 people, 0.09%, the smallest named church here. "
        "**Igreja Pentecostal Deus é Amor**, founded in São Paulo in 1962, and the same "
        "body `py2002.py` maps for Paraguay. Classical Pentecostal, but nothing INE "
        "publishes says anything about its doctrine of the Godhead, which is what "
        "`trinitarian` and `oneness` divide on, so it stays on the family node.",
    "Judaica":
        "-> judaism. **23 people**, the smallest answer this census printed, across seven "
        "concelhos: Praia 7, São Vicente 6, Sal 4, Boa Vista 2, Brava 2, Maio 1, Santa "
        "Catarina de Santiago 1. Cabo Verde had a real Sephardic "
        "community in the 19th century, Moroccan Jews who came for the trade and are "
        "remembered in the restored cemeteries on Santo Antão and Boa Vista, and it "
        "assimilated. Twenty-three people is not that community and is not evidence about "
        "it; it is drawn because the census counted it, and at 1:1,000 it is a presence "
        "ring rather than a dot (spec §4.3).",
}

MAP = {
    "Adventista": "christianity.adventist",
    "Assembleia de Deus": "christianity.pentecostal.trinitarian",
    "Católica": "christianity.catholic",
    "Deus é Amor": "christianity.pentecostal",
    "Igreja do Nazareno / Protestante": "christianity.holiness.nazarene",
    "Islâmica / Muçulmano": "islam",
    "Judaica": "judaism",
    "Nova Apastólica": "christianity.newapostolic",
    "Racionalismo Cristão": "spiritualism",
    "Testemunha de Jeová": "christianity.witnesses",
    "Universal do Reino de Deus": "christianity.pentecostal.charismatic",
    "Jesus Cristo dos Santos dos Últimos Dias / Mórmons": "christianity.latterday.lds",
    "Outra": "other.cv",
    "Sem religião": "unaffiliated",
}

# spec 7a-i-1: the level this source COUNTED each node at, so a dot inferred below it rolls
# up instead of vanishing. INE measures every one of the fourteen at the concelho it draws
# them on, and nothing here is filled in from a coarser tier, so the column is the node
# itself and nothing rolls.
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
