"""Stats SA Community Survey 2016 religion -> religiondots taxonomy.

Twenty-five categories at province: ten non-Christian religion answers from table 2.10a,
fourteen Christian denominations from table 2.10b, and the reconciling residual between
them. `sources/za.py` emits every one of them prefixed `Religion: ` or `Christian: `,
because **both tables have a row called `Other` and they mean different things** -- one is
another faith, one is another church. A bare join on the label would move 1,482,210 people
of other religions into Christianity and nothing would report it.

**THE CATEGORY THE COUNTRY IS DRAWN FOR IS `African Independent Church/African Initiated
Church`: 14,158,453 people, 25.77% of everyone who answered and 32.61% of the country's
Christians.** Before South Africa, `christianity.africaninstituted` and its children held
11,741,516 people across six countries -- Zimbabwe 6,112,503, Kenya 3,292,573, Angola's
three children 1,099,234, Benin 676,032, Eswatini 420,690, Côte d'Ivoire's Harrist 140,484.
So this one country is **1.21x the whole of the rest of the node and 2.32x Zimbabwe's cell,
which was the largest any single country had contributed**, and it takes South Africa to
54.7% of every African Instituted Church member on the map. (An earlier draft of this line
said "more than double" and that is wrong; it was checked against `countries.py`'s own
`counts()` for all seven countries.) South Africa is where the movement began.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

Every figure below reproduces from `data/normalized/za.csv`. Province percentages are of
that province's own drawn answers unless the line says "of its Christians".

**"OF ITS CHRISTIANS" IS NOT THE DENOMINATOR STATS SA PRINTS, and the difference is small
and systematic.** The reports compute table 2.10b's percentages against that table's own
total, which explicitly EXCLUDES the `Do not know` and `Unspecified` denominations; the
figures here are against those fourteen rows PLUS the not-reported residual, so every one of
them sits a few tenths of a percent relatively below the printed one (Limpopo's African
Independent row is 50.9 in Report 03-01-15 and 50.77 here; Western Cape's Pentecostal row is
19,6 in Report 03-01-07 and 19.45 here). The wider base is deliberate: it is the same
denominator in all nine provinces, and North West's printed total is defective (see below),
so computing on it would make that province's percentages incomparable with the rest.
"""

EXCLUDED = {}

REVIEW = {
    "Christian: African Independent Church/African Initiated Church":
        "-> christianity.africaninstituted, and this is the row that made South Africa "
        "worth building from a 2016 survey rather than the 2022 census. **14,158,453 "
        "people, 25.77% of everyone who answered and 32.61% of the country's Christians** "
        "-- larger than "
        "Zimbabwe's Vapostori cell (6,112,503), which was the biggest this node had ever "
        "held, and larger than every other single Christian answer in South Africa. "
        "Stats SA's own exemplars, printed in the table: *Zion Christian Church; Apostolic "
        "Church; African Nazareth Baptist Church/Shembe*. The ZCC at Moria in Limpopo "
        "draws the largest Easter gathering in Africa, and Limpopo is exactly where this "
        "row peaks: **50.77% of that province's Christians**, against 15.83% in Western "
        "Cape. KwaZulu-Natal is 42.98%, which is Shembe's country, and Mpumalanga 39.35%. "
        "**The mapping is not in doubt; what it cannot do is separate them.** These are "
        "thousands of distinct churches -- the ZCC and its St Engenas branch, the "
        "Nazaretha, the many Zionist and Apostolic bodies -- and one cell holds all of "
        "them. The 1996 and 2001 censuses did break them out (2001: Zion Christian Church "
        "4,971,932, Other Apostolic Churches 5,609,070, Ethiopian Churches 880,414, "
        "Ibandlalama Nazaretha 248,824), so a finer split of this node exists in principle "
        "at a twenty-five-year-old vintage; see `sources/za.md` §4.",
    "Christian: Pentecostal/Evangelistic":
        "-> christianity.pentecostal. 8,483,677 people, 19.54% of Christians and the "
        "second largest Christian answer. Stats SA's exemplars are *Assemblies of God; "
        "Born Again Church of God in Christ; Rhema Church; Apostolic Faith Mission; "
        "Prophetic Ministry*, which are Pentecostal and charismatic bodies throughout, so "
        "the node is right despite the label's `Evangelistic` half. **That word is worth a "
        "flag rather than a different node**: read strictly it would also take in "
        "non-Pentecostal evangelicals, who have no other box on this card and are "
        "therefore somewhere in this row or in `Other`. No branch is inferred, because the "
        "cell mixes the classical Pentecostal denominations with the newer independent "
        "prophetic ministries and the survey does not separate them. "
        "**Unusually for this map, it is the flattest large category in the country**: "
        "15.96% of Christians in KwaZulu-Natal to 25.69% in Limpopo, where most of the "
        "others move by a factor of three or more.",
    "Christian: Reformed church":
        "-> christianity.reformed.continental, NOT the `christianity.reformed` parent, "
        "which would then sit above the Presbyterian row and break §3.2's partition. "
        "Stats SA's exemplars are *Dutch Reformed Church; United Reformed Church; "
        "Christian Reformed Church*: the NG Kerk, the Nederduitsch Hervormde Kerk, the "
        "Gereformeerde Kerke and URCSA, which are Dutch Reformed in polity and descent and "
        "are continental rather than Presbyterian. 2,350,853 people, 5.41% of Christians. "
        "**Its geography is the Afrikaans-speaking interior**: 12.30% of Christians in "
        "Western Cape, 11.99% in Northern Cape and 11.34% in Free State against **1.37% in "
        "KwaZulu-Natal**, a factor of nine. That is the second sharpest gradient among the "
        "named mission denominations rather than the first; the Methodist row below beats "
        "it on both measures (15.3x and 14.2 points against 9.0x and 10.9 points).",
    "Christian: Presbyterian":
        "-> christianity.reformed.presbyterian, a sibling of the Reformed row above and "
        "not a child of it. 621,065 people, 1.43% of Christians. The Uniting Presbyterian "
        "Church in Southern Africa and the Presbyterian Church of Africa, both from the "
        "Scottish mission at Lovedale, which is why the row is an Eastern Cape one: "
        "**3.54% of that province's Christians against 0.39% in Mpumalanga**.",
    "Christian: Denomination not reported":
        "-> bare `christianity`, the answer `Christian` with nothing after it, which is the "
        "call at2001.py, au2021.py, bd2011.py, bg2021.py and bs2022.py all make for the "
        "same shape. 567,039 people, 1.31% of Christians. "
        "**IT IS NOT AN INVENTED ROW, IT IS THE RECONCILIATION.** Table 2.10b excludes its "
        "own `Do not know` and `Unspecified` and prints both figures, and the difference "
        "between table 2.10a's Christianity cell and table 2.10b's total reproduces them. "
        "Checked against the printed footnotes in all nine reports, the tally is: **exact "
        "in four** (Northern Cape 7,989, Free State 8,998, KwaZulu-Natal 18,874, Limpopo "
        "8,346), **within one person in three** (Eastern Cape +1, Gauteng -1, Mpumalanga "
        "-1), **unverifiable in one** because Report 03-01-07 prints no exclusion note "
        "under Western Cape's table at all, and **broken in one**, which is North West "
        "below. Western Cape's 46,055 is therefore inferred from table 2.11a rather than "
        "confirmed, and it is the largest residual of the eight sound provinces. "
        "Dropping the row would silently shrink Christianity in every province. "
        "**NORTH WEST IS 10.45% OF ITS CHRISTIANS HERE AGAINST 0.20-0.91% EVERYWHERE "
        "ELSE, AND THAT IS A DEFECT IN REPORT 03-01-11, NOT A FINDING ABOUT NORTH WEST.** "
        "That report's table 2.10b prints fourteen rows summing to 3,072,039 against its "
        "own printed total of 3,408,521, and its printed percentages sum to 90.1 rather "
        "than 100.0, so 336,482 people are in no denomination row at all. "
        "**TWO EXPLANATIONS FIT AND NEITHER CLOSES, WHICH IS THE REASON NOT TO GUESS.** "
        "One: the `Other` cell reads 21 873, character for character the `Do not know` "
        "figure in that same table's own footnote, and North West's `Other` comes out at "
        "0.64% of its Christians where the other eight run 4.43% to 17.25%; restoring the "
        "residual there would put it at 10.5%, inside that range. Two: North West is the "
        "only province whose footnote lists a THIRD exclusion, `Not applicable` (318,029), "
        "which is 94.5% of the shortfall, and if the rows were built excluding it while "
        "the total was not, these people would not be Christians at all. The first story "
        "misses nothing obvious; the second misses by 18,453. **Two non-closing stories "
        "are a better reason to leave it alone than one good one would be.** "
        "**And the cost of leaving it alone is bounded, which is the real defence.** Table "
        "2.10a independently counts 3,430,406 North West Christians, and `sources/za.py` "
        "refuses to build if any province's residual exceeds its own 2.10a Christianity "
        "cell. So whatever went wrong in table 2.10b, parking these people on bare "
        "`christianity` cannot be wrong about their religion; it can only be silent about "
        "their denomination, which is exactly what the row says. `sources/za.py` also "
        "asserts the defect is still present, so a reissued report fails the build rather "
        "than quietly changing the map.",
    "Christian: Other":
        "-> christianity.other. 3,509,156 people, 8.08% of Christians. Larger than it "
        "looks for a residual, and the reason is that South Africa's card names thirteen "
        "denominations and the country has thousands of independent congregations that "
        "match none of them. "
        "**Read the North West cell as missing, not as small.** It is 0.64% of that "
        "province's Christians against 4.43% to 17.25% elsewhere, and Report 03-01-11's "
        "table does not add up; see the `Denomination not reported` entry above. Every "
        "other province is a real measurement. The high end is Northern Cape at 17.25% "
        "and Western Cape at 13.37%.",
    "Christian: Anglican/Episcopalian":
        "-> christianity.anglican. 1,765,287 people, 4.07% of Christians. The Anglican "
        "Church of Southern Africa. Western Cape 8.26% of its Christians against 1.02% in "
        "Mpumalanga.",
    "Christian: Methodist":
        "-> christianity.methodist, the parent and no branch. 2,777,937 people, 6.40% of "
        "Christians. The Methodist Church of Southern Africa, and separately the several "
        "African Methodist bodies that split from it; the survey names neither, so nothing "
        "below the parent is inferred. **Eastern Cape is 15.18% of its Christians against "
        "0.99% in Limpopo**, which is the sharpest gradient of any named denomination in "
        "the country on both measures, 15.3x and 14.2 points, ahead of the Reformed row's "
        "9.0x and 10.9. Same Eastern Cape mission geography as the Presbyterian row.",
    "Christian: Seventh Day Adventist":
        "-> christianity.adventist, the parent rather than the `.sda` leaf. 311,271 "
        "people, 0.72% of Christians. The leaf is the named denomination and this map "
        "reserves it for sources that count that body specifically; a census answer box "
        "reading `Seventh Day Adventist` is what a respondent said, and other Adventist "
        "bodies would land in it. Two of the nine reports spell it `Seventh-Day`, folded "
        "in `sources/za.py`.",
    "Christian: Mormon":
        "-> christianity.latterday, the parent rather than the `.lds` leaf, for the same "
        "reason as the Adventist row. 114,807 people, 0.26% of Christians and the smallest "
        "Christian cell. Stats SA's exemplar is *Church of Jesus Christ of Latter Day "
        "Saints*, so the parent is barely wider than the leaf here, but the parent is what "
        "the answer supports.",
    "Christian: Just a Christian/non-denominational":
        "-> christianity.nondenominational. 2,501,384 people, 5.76% of Christians. **This "
        "is a different row from `Denomination not reported` above and the difference "
        "matters**: this one is an answer a respondent chose off the card, the other is "
        "the arithmetic gap left by people whose denomination was not established. Its "
        "geography is urban -- 8.64% of Western Cape's Christians and 8.29% of Gauteng's "
        "against 2.68% in North West -- which is the shape a non-denominational answer "
        "usually has.",
    "Religion: Traditional African religion":
        "-> indigenous.african, the node Ghana added. 2,454,887 people, 4.47%. "
        "**Read it as a floor**, for the reason sources.md §11b gives for the whole "
        "continent and `zw2022.py` gives next door: the box is exclusive of the Christian "
        "ones, and consulting a *sangoma* or honouring the *amadlozi* very commonly "
        "accompanies church membership here rather than replacing it. South Africa is a "
        "sharper case than most, because the African Independent Churches counted "
        "separately at six times the size grew out of exactly that overlap. "
        "Its geography is KwaZulu-Natal 7.36%, Eastern Cape 6.12% and Limpopo 5.47% "
        "against **0.50% in Northern Cape and 0.90% in North West**. No child node: no "
        "Stats SA release names a South African tradition individually (§2.4). "
        "**And note this is the category the two instruments disagree on most**: Census "
        "2022 puts it at 7.8% against 4.5% here, moving the opposite way to the "
        "no-religion cell. See `sources/za.md` §3.",
    "Religion: No religious affiliation/belief":
        "-> unaffiliated, with Atheism and Agnosticism kept separate below because the "
        "survey counts them separately. 5,964,893 people, 10.86%. "
        "**Its geography is not the urban-secular one a reader will expect.** Limpopo is "
        "the highest in the country at 17.25% and Northern Cape the lowest at 1.67%, with "
        "Gauteng at 13.82% and KwaZulu-Natal at 12.87%. Limpopo is also the most African "
        "Independent province, so this is not a religiosity gradient; a plausible reading "
        "is that where church membership is the dominant idiom, people outside a "
        "particular church answer `none` rather than naming an ancestral practice, which "
        "is Benin's `Aucune` warning in a different country. Nothing here resolves it. "
        "**The bigger caution is vintage.** Census 2022 puts this cell at 2.9% against "
        "10.86% here. That is a factor of nearly four in six years, which no trend "
        "explains, and it is the strongest single reason §3.1 forbids mixing the two "
        "releases; `sources/za.md` §3 has it.",
    "Religion: Atheism":
        "-> secular, and 'Religion: Agnosticism' goes to the same node. Together 85,542 "
        "people, 0.16%. Kept off `unaffiliated` because the survey asks them as separate "
        "boxes, which is the test au2021.py, ba2013.py, br2010.py and bs2022.py apply; "
        "`zw2022.py` sends its no-religion cell to `unaffiliated` alone precisely because "
        "Zimbabwe has only the one box. The two merge here because the tree has no child "
        "under `secular` for either and neither is large enough to want one. **Western "
        "Cape is far the highest on both** (0.30% atheist, 0.15% agnostic) and both round "
        "to 0.01% in three provinces. Their second places differ, which is why they are "
        "not described together: atheism runs KwaZulu-Natal 0.13% then Gauteng 0.12%, "
        "agnosticism Gauteng 0.10% then KwaZulu-Natal 0.05%.",
    "Religion: Other":
        "-> other.za. 1,482,210 people, 2.70%. See the node's own note: a genuine tail "
        "rather than a store cupboard, because Islam, Hinduism, Judaism, Buddhism, Bahaism "
        "and Traditional African religion all have their own boxes, and unusually flat "
        "across the provinces (3.87% Gauteng to 1.18% North West), so it names nothing. "
        "Per §3.11.",
    "Religion: Hinduism":
        "-> hinduism, with no branch, because the survey gives none. 561,269 people, "
        "1.02%. **The most concentrated religion in the country**: 3.99% of KwaZulu-Natal "
        "against 0.76% in Gauteng and 0.02-0.13% in the other seven provinces. That is the "
        "Durban Indian population, descended from the indentured labourers brought to the "
        "Natal sugar estates from 1860. Durban is often called the largest Indian city "
        "outside India; South Africa is NOT the largest Indian-descended population "
        "outside Asia, which an earlier draft of this line claimed, because the United "
        "States and the United Kingdom are both larger.",
    "Religion: Islam":
        "-> islam, with no branch, because the survey gives none and the two communities "
        "it holds belong to different ones. 892,684 people, 1.62%. **Western Cape 5.64% "
        "against 0.27% in Free State and Limpopo.** The Cape figure is largely the Cape "
        "Malay, Shafi'i Muslims descended from people exiled and enslaved from the Dutch "
        "East Indies from the 1650s; the Gauteng and KwaZulu-Natal figures are largely "
        "Hanafi Muslims of South Asian descent. Nothing in the survey separates them.",
    "Religion: Judaism":
        "-> judaism, with no branch. 49,471 people, 0.09%. Almost entirely Lithuanian "
        "Ashkenazi in descent and almost entirely in two cities: 0.22% of Western Cape and "
        "0.21% of Gauteng against 0.00-0.05% in the other seven provinces. Unlike "
        "Zimbabwe's cell next door, this figure and its geography are both what the "
        "community's history predicts, so nothing about it needs a second reading.",
    "Religion: Buddhism":
        "-> buddhism, with no branch. 24,807 people, 0.05%. Gauteng writes the label "
        "`Buddism` and the other eight write `Buddhism`; it is the CS 2016 codebook's own "
        "spelling rather than one report's typo, and `sources/za.py` folds it.",
    "Religion: Bahaism":
        "-> bahai. 6,880 people, 0.01%, the smallest cell in the country. **Northern Cape "
        "prints it as a bare dash rather than a zero** and `sources/za.py` reads that as "
        "zero rather than as a missing row; at this size the distinction is between none "
        "and too few to weight to one. Under a dot at the national dot value in most "
        "provinces, so it will draw sparsely and may ring (§4.3).",
}

MAP = {
    # --- table 2.10a, the ten non-Christian answers. Christianity is not here: it is
    # --- replaced by table 2.10b's fourteen denominations plus the residual.
    "Religion: Islam": "islam",
    "Religion: Traditional African religion": "indigenous.african",
    "Religion: Hinduism": "hinduism",
    "Religion: Buddhism": "buddhism",
    "Religion: Bahaism": "bahai",
    "Religion: Judaism": "judaism",
    "Religion: Atheism": "secular",
    "Religion: Agnosticism": "secular",
    "Religion: No religious affiliation/belief": "unaffiliated",
    "Religion: Other": "other.za",

    # --- table 2.10b, the fourteen Christian denominations.
    "Christian: Catholic": "christianity.catholic",
    "Christian: Anglican/Episcopalian": "christianity.anglican",
    "Christian: Baptist": "christianity.baptist",
    "Christian: Lutheran": "christianity.lutheran",
    "Christian: Methodist": "christianity.methodist",
    "Christian: Presbyterian": "christianity.reformed.presbyterian",
    "Christian: Pentecostal/Evangelistic": "christianity.pentecostal",
    "Christian: African Independent Church/African Initiated Church":
        "christianity.africaninstituted",
    "Christian: Jehovah's Witness": "christianity.witnesses",
    "Christian: Seventh Day Adventist": "christianity.adventist",
    "Christian: Mormon": "christianity.latterday",
    "Christian: Reformed church": "christianity.reformed.continental",
    "Christian: Just a Christian/non-denominational": "christianity.nondenominational",
    "Christian: Other": "christianity.other",

    # --- the reconciling residual between the two tables.
    "Christian: Denomination not reported": "christianity",
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
