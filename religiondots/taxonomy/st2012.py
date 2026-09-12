"""INE São Tomé e Príncipe, IV RGPH 2012, Quadro 8 religion -> religiondots taxonomy.

Thirteen answers on all seven districts, for the whole resident population of every age.
Ten are drawn: nine named churches and `Outras`. `Não tem` is no religion. The remaining two,
`Não declarou` and `Não sabe`, are 1,756 people between them, 0.98%, and are off the tree.

**THE FORM IS NINE CHURCHES AND NOTHING ELSE.** There is no `Protestante` box, no `Islão`,
no traditional-religion box and no `Outra religião cristã`: INE names the Adventists, the
Assembly of God, the Roman Catholic Church, Deus é Amor, the Jehovah's Witnesses, Maná, the
New Apostolic Church, the World Messianic Church and the Universal Church of the Kingdom of
God, and everything else is `Outras`, which is why that cell is 5.03% and the country's
third largest answer. Cabo Verde's 2021 form is the same shape and its `Outra` is 1.16%; the
difference is that Cabo Verde names the Nazarenes, which is where its Protestants are.

**FOUR OF THE NINE ARE BRAZILIAN AND ONE IS JAPANESE BY WAY OF BRAZIL.** Deus é Amor (São
Paulo, 1962), the Universal Church of the Kingdom of God (Rio, 1977) and the World Messianic
Church (Atami, 1935, and in Brazil from 1955) are all on the tree already because Brazil
counts them; Maná is Lisbon rather than Rio and is the one call here with no precedent to
follow. `br2010.py` and `cv2021.py` between them fix eight of the ten drawn rows, which is
what a shared taxonomy is for.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Não declarou":
        "488 people, 0.27%, who were asked and refused. Not a religion, so off the tree "
        "per spec §3.5. It is read and carried in st.csv so that a district's thirteen "
        "rows reconcile against its own printed total, which is the check that catches a "
        "figure read into the wrong column.",
    "Não sabe":
        "1,268 people, 0.71%, who did not know. Same reasoning, and INE keeps it separate "
        "from `Não declarou` and from `Não tem`, so this file does too. **Together the two "
        "are 0.98% of São Tomé**, which is small enough that leaving them out moves no "
        "drawn share by more than half a point; `countries.py`'s note says which way.",
}

REVIEW = {
    "Maná":
        "-> christianity.pentecostal.charismatic. **4,191 people, 2.34%, the fifth largest "
        "of São Tomé's nine named churches**, and the one row on this form with no "
        "precedent anywhere else "
        "on the map. **Igreja Maná** was founded in Lisbon in September 1984 by Jorge "
        "Tadeu, a Mozambican-born civil engineer who had been a member of the Apostolic "
        "Faith Mission of South Africa, and it is now in some eighty countries with its own "
        "television and radio stations in Portugal, Spain, Brazil, Mozambique and São Tomé "
        "and Príncipe. **The founder's lineage is classical Pentecostal and the church is "
        "not**: it dates from the 1980s, is built around a personal apostolate, media and "
        "prosperity teaching, and belongs with the Universal Church of the Kingdom of God "
        "rather than with the Assembly of God — which is the same cut `br2010.py` makes "
        "between IURD and Assembleia de Deus, and the reason this goes on the neo-"
        "charismatic child rather than on `...trinitarian`. **Its geography agrees**: 3.01% "
        "in Água-Grande and 2.79% in Cantagalo, the capital and the road along the eastern "
        "coast, against 0.62% in Lembá and 0.66% in Caué, the two remotest districts. It is "
        "a broadcast church and it is where the transmitters are, which is not the shape a "
        "mission takes. Cabo Verde's 2021 form has no Maná box at all, so nothing else here "
        "counts this body.",
    "Messiânica Mundial":
        "-> eastasiannew.japanese. **688 people, 0.38%, and the smallest drawn cell on this "
        "map's São Tomé**, at 1:1,000 a presence ring rather than a dot (spec §4.3). The "
        "Igreja Messiânica Mundial is Sekai Kyūsei Kyō, founded at Atami by Mokichi Okada "
        "in 1935, whose practice is *johrei*, the transmission of light by the raised hand. "
        "It reached Brazil in 1955 and the lusophone world from there, and `br2010.py` maps "
        "Brazil's on this same node. **A Japanese new religion with a measurable following "
        "in the Gulf of Guinea is not a thing this map has drawn before**, and it is the "
        "sharpest single argument for reading a small country's census in full: 688 people "
        "is 0.38% of São Tomé against **0.054% of Brazil** (103,736 of 190.8 million in "
        "2010), so the country the church came from is seven times less Messianic than the "
        "one it reached. Its own geography is the capital, 0.62% of Água-Grande against 0.10% in "
        "Lobata, which is where a church that arrives by air and stays urban would be.",
    "Assembléia de Deus":
        "-> christianity.pentecostal.trinitarian. 5,991 people, 3.35%, and INE's own "
        "spelling with the accent (spec §2.4, transcribe). The São Toméan Assembleia de "
        "Deus descends from the Brazilian body rather than from the US Assemblies of God, "
        "so it goes on the family's Trinitarian node and not on "
        "`...trinitarian.assemblies-of-god`, which is the USRC's American denomination; "
        "`cv2021.py` and `ao2024.py` make the same call for the same reason. **Its "
        "geography is the one that most needs saying**: 10.08% of Caué, the empty southern "
        "district, against 1.39% in Príncipe and 2.52% in the capital. Caué is the least "
        "Catholic district in the country, 38.22% against a national 55.71%, and this is "
        "the largest single piece of that 17.5-point gap; the New Apostolic Church, the "
        "Universal Church and no religion make up most of the rest.",
    "Nova Apostólica":
        "-> christianity.newapostolic. 5,177 people, 2.90%, the same Irvingite body "
        "`ao2024.py` and `cv2021.py` map, and of the three countries on this node São "
        "Tomé's is much the largest share: Angola 1.50%, Cabo Verde 0.49%. **It is a rural "
        "church here.** Caué 8.41%, "
        "Príncipe 7.39% and Lembá 5.15% against **0.89% in Água-Grande**, which is the only "
        "category on this form that is at its weakest in the capital. Cabo Verde's is "
        "concentrated the same way, on one island rather than in Praia.",
    "Outras":
        "-> other.st. 8,990 people, 5.03%, the third largest answer in the country, and "
        "large for the reason the node's own entry gives: the form is nine named churches "
        "with no family boxes, so it holds every Protestant, every Muslim and every church "
        "outside the nine at once. **The 2024 census prices the Muslim part of it**: an "
        "`Islâmico/Muçulmano` box added that year drew 354 people, 0.17% of the country, so "
        "Islam is about a twentieth of this cell and the rest is Christian. Its geography "
        "runs 6.16% in Príncipe to 3.07% in Lobata, a flat two to one with the capital, the "
        "island and the empty south together at the top, so §9r's Chittagong rule finds no "
        "cluster to read and spec §3.11 draws it whole.",
    "Não tem":
        "-> unaffiliated. **37,935 people, 21.22%, the second largest answer**, and its "
        "geography is the finding of this country. It runs **33.61% in Lembá**, the "
        "north-western district, and 27.66% in Mé-Zóchi and 27.19% in Caué, against "
        "**4.60% in the Região Autónoma do Príncipe** — a seven-fold spread inside a "
        "country of 179,000, and the two ends are an island apart rather than a city and a "
        "countryside. It goes to `unaffiliated` rather than `secular` because INE's box is "
        "*não tem*, an absence, and the 2012 form offers no atheist or agnostic option "
        "beside it. **The 2024 census settles that reading**: it splits the answer into "
        "`Sem religião` 20,075 and `Ateu` 1,473, so when São Toméans are offered the "
        "distinction, thirteen in fourteen of them take the absence.",
    "Deus é amor":
        "-> christianity.pentecostal. 1,432 people, 0.80%. Igreja Pentecostal Deus é Amor, "
        "founded in São Paulo in 1962, the same body `cv2021.py` and `py2002.py` map, and "
        "it stays on the family node for the same reason: classical Pentecostal, but "
        "nothing INE publishes says anything about its doctrine of the Godhead, which is "
        "what `trinitarian` and `oneness` divide on. **Lobata holds 321 of the 1,432**, "
        "1.66% of that district against 0.05% in Lembá and 0.16% in Cantagalo, which is "
        "one congregation showing through a national table.",
    "Católica Apostólica Romana":
        "-> christianity.catholic.latin. 99,570 people, 55.71%. On the Latin child rather "
        "than the `christianity.catholic` parent because INE's label says Roman "
        "explicitly, which is `br2010.py`'s reading of the identical Portuguese string; "
        "`cv2021.py` prints only `Católica` and sits on the parent. São Tomé has been a "
        "diocese since 1534, one of the oldest in Africa, and there is no Eastern-rite or "
        "independent Catholic body for the label to be competing with.",
}

MAP = {
    "Adventista": "christianity.adventist",
    "Assembléia de Deus": "christianity.pentecostal.trinitarian",
    "Católica Apostólica Romana": "christianity.catholic.latin",
    "Deus é amor": "christianity.pentecostal",
    "Jeová": "christianity.witnesses",
    "Maná": "christianity.pentecostal.charismatic",
    "Nova Apostólica": "christianity.newapostolic",
    "Messiânica Mundial": "eastasiannew.japanese",
    "Igreja Universal do Reino de Deus": "christianity.pentecostal.charismatic",
    "Outras": "other.st",
    "Não tem": "unaffiliated",
}

# spec 7a-i-1: the level this source COUNTED each node at, so a dot inferred below it rolls
# up instead of vanishing. INE measures every one of the eleven at the district it draws
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
