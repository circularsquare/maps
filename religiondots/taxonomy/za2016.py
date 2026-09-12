"""Stats SA Community Survey 2016 religion -> religiondots taxonomy.

Twenty-five drawn categories on 213 local municipalities: ten non-Christian answers from the
`ReligionBelief` question, fourteen denominations from the `Christianity` question, and the
Christians whose denomination the survey did not establish. `sources/za.py` emits every one of
them prefixed `Religion: ` or `Christian: `, because **both questions have an answer called
`Other` and they mean different things** -- one is another faith, one is another church. A bare
join on the label would move 1,482,207 people of other religions into Christianity and nothing
would report it.

**NOT ONE NODE CHANGED WHEN THIS COUNTRY WENT FROM 9 PROVINCES TO 213 MUNICIPALITIES ON
2026-09-09.** The microdata's answer set is the published one -- `sources/za.py` asserts each
codebook label folds onto a published category -- so `MAP` below is character for character
what the province build shipped. What changed is the geography, the figures, and two things
the province build could only describe:

  * **North West's `Other` row is no longer a defect this map has to work around.** Report
    03-01-11's fourteen denomination rows sum to 90.1% of its own printed total and its
    `Other` cell reads 21,873, which is the `Do not know` figure from that table's own
    footnote. The province build parked the missing 336,482 people on bare `christianity`
    because two stories fitted and neither closed. The microdata puts North West's `Other` at
    358,355 -- 21,873 + 336,482 exactly -- so the row was mis-set and those people are
    Christians of another denomination. North West now reads 10.45% `Other` of its Christians,
    inside the 4.43% to 17.25% the other eight provinces run, and 0.64% `Denomination not
    reported` against 0.20% to 0.91% elsewhere. Both cells were wrong the other way round
    before, and both are right now.
  * **`Denomination not reported` is a measurement rather than an arithmetic gap.** It was the
    difference between two published tables, 567,039 people. It is now the survey's own
    `Do not know` (227,585) plus `Unspecified` (2,976) on the denomination question, 230,558
    after rounding, seen per person.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

Every figure below reproduces from `data/normalized/za.csv`. Municipality percentages are of
that municipality's own drawn answers unless the line says "of its Christians", and each one
names the number of unweighted CS 2016 records behind it, which the file carries as `cell_n`
in its note column. **That is not decoration.** This is a survey drawn at a fine tier, and a
share computed in a municipality of 12,000 people is a real measurement with a wide interval;
one computed in eThekwini is not. Municipal superlatives quoted here are restricted to the 175
units over 50,000 people unless the line says otherwise.
"""

# The two `ReligionBelief` answers that are not an answer. They are IN
# data/normalized/za.csv rather than dropped before it, so `tools/gap_share.py` computes South
# Africa's gap from the file instead of it being authored -- these people are 707,295 of the
# 55,653,634 the survey weighted, 1.2709%, and the province build could only reach that figure
# by adding up eight printed footnotes and inferring Western Cape's by difference.
#
# NOT DRAWN, per §3.5: nothing here knows what they would have said, and Stats SA's own tables
# exclude them from every religion total it publishes. Which way the hole leans is measured
# rather than guessed; see `countries.py`'s note_public and sources/za.md §3.5.
EXCLUDED = {
    "Religion: Do not know":
        "704,355 people, 1.27% of the survey population. Answered the religion question with "
        "`Do not know`. Stats SA excludes them from table 2.10a's own total in all nine "
        "provincial profiles and prints the figure under the table; this map does the same "
        "and says so in `gap`.",
    "Religion: Unspecified":
        "2,940 people, 0.005%. No answer recorded at all. Excluded with the row above and "
        "for the same reason; the two together are the country's `gap_share`.",
}

REVIEW = {
    "Christian: African Independent Church/African Initiated Church":
        "-> christianity.africaninstituted, and this is the row the country is drawn for. "
        "**14,158,461 people, 25.77% of everyone who answered and 32.61% of the country's "
        "Christians** -- larger than Zimbabwe's Vapostori cell (6,112,503), which was the "
        "biggest this node had ever held, and larger than every other single Christian "
        "answer in South Africa. Stats SA's own exemplars, printed in the table and carried "
        "in the CSV's note column: *Zion Christian Church; Apostolic Church; African "
        "Nazareth Baptist Church/Shembe*. "
        "**THE MUNICIPAL TIER IS WHAT THIS ROW WAS WORTH BUYING.** At nine provinces it ran "
        "15.83% of Western Cape's Christians to 50.77% of Limpopo's, a factor of 3.2. At 213 "
        "municipalities it runs **4.64% of Kai !Garib's Christians in the Northern Cape "
        "(n=158) to 75.56% of Mthonjaneni's in KwaZulu-Natal (n=2,609)**, a factor of 16, "
        "with Mfolozi 74.91% (n=4,437) and Maruleng in Limpopo 74.31% (n=4,248) beside it. "
        "As a share of ALL answers rather than of Christians the peak is Big Five Hlabisa at "
        "61.98% and Mfolozi at 61.62%: in those municipalities the African Independent "
        "Churches are not the largest denomination, they are the majority of the population. "
        "The Limpopo cluster is the Zion Christian Church's own country -- Moria, the ZCC's "
        "headquarters and the site of the largest Easter gathering in Africa, is in "
        "Limpopo's Capricorn district -- and the KwaZulu-Natal one is Shembe's Nazaretha. "
        "**The mapping is not in doubt; what it cannot do is separate them.** These are "
        "thousands of distinct churches and one cell holds all of them. The 1996 and 2001 "
        "censuses did break them out (2001: Zion Christian Church 4,971,932, Other Apostolic "
        "Churches 5,609,070, Ethiopian Churches 880,414, Ibandlalama Nazaretha 248,824), so a "
        "finer split exists in principle at a twenty-five-year-old vintage; sources/za.md §9.",
    "Christian: Pentecostal/Evangelistic":
        "-> christianity.pentecostal. 8,483,686 people, 19.54% of Christians and the second "
        "largest Christian answer. Stats SA's exemplars are *Assemblies of God; Born Again "
        "Church of God in Christ; Rhema Church; Apostolic Faith Mission; Prophetic Ministry*, "
        "which are Pentecostal and charismatic bodies throughout, so the node is right "
        "despite the label's `Evangelistic` half. **That word is worth a flag rather than a "
        "different node**: read strictly it would also take in non-Pentecostal evangelicals, "
        "who have no other box on this card and are therefore somewhere in this row or in "
        "`Other`. No branch is inferred, because the cell mixes the classical Pentecostal "
        "denominations with the newer independent prophetic ministries and the survey does "
        "not separate them. "
        "**At province it was the flattest large category in the country, 15.96% to 25.69% of "
        "Christians, and the municipal tier does not overturn that so much as sharpen it**: "
        "5.85% of Mthonjaneni's Christians (n=191) to 47.71% of Greater Giyani's in Limpopo "
        "(n=6,868), with Collins Chabane 42.02% (n=8,554) beside it. That is a factor of 8 "
        "where the African Independent row is 16 and the Reformed row is 200, so the "
        "comparison holds; what it adds is that the Limpopo and Mpumalanga lowveld is a "
        "genuinely Pentecostal region rather than a merely above-average one.",
    "Christian: Reformed church":
        "-> christianity.reformed.continental, NOT the `christianity.reformed` parent, which "
        "would then sit above the Presbyterian row and break §3.2's partition. Stats SA's "
        "exemplars are *Dutch Reformed Church; United Reformed Church; Christian Reformed "
        "Church*: the NG Kerk, the Nederduitsch Hervormde Kerk, the Gereformeerde Kerke and "
        "URCSA, which are Dutch Reformed in polity and descent and are continental rather "
        "than Presbyterian. 2,350,855 people, 5.41% of Christians. "
        "**This is the sharpest gradient in the country and the municipal tier multiplies it "
        "by twenty.** At province it was 12.30% of Western Cape's Christians against 1.37% in "
        "KwaZulu-Natal, a factor of 9. At municipality it is **41.89% of Bergrivier's "
        "Christians (n=1,391), 39.11% of Matzikama's (n=1,427) and 29.86% of Witzenberg's "
        "(n=1,629) against 0.21% in Mfolozi (n=13)**, a factor of 200. Those three are the "
        "Swartland and the Olifants River valley, which is the NG Kerk's own ground, and they "
        "are also the three municipalities where the African Independent row is lowest. The "
        "old settler/mission line is not a provincial boundary; it runs inside the Western "
        "Cape.",
    "Christian: Presbyterian":
        "-> christianity.reformed.presbyterian, a sibling of the Reformed row above and not a "
        "child of it. 621,063 people, 1.43% of Christians. The Uniting Presbyterian Church in "
        "Southern Africa and the Presbyterian Church of Africa, both from the Scottish "
        "mission at Lovedale, which is why it is an Eastern Cape row: **11.13% of Raymond "
        "Mhlaba's Christians (n=1,068) and 10.87% of Mnquma's (n=1,394) against zero in three "
        "KwaZulu-Natal municipalities**. Raymond Mhlaba contains Alice and Lovedale itself, "
        "which the province tier could not show.",
    "Christian: Denomination not reported":
        "-> bare `christianity`, the answer `Christian` with nothing after it, which is the "
        "call at2001.py, au2021.py, bd2011.py, bg2021.py and bs2022.py all make for the same "
        "shape. 230,558 people, 0.53% of Christians. "
        "**IT IS NOW A MEASUREMENT AND IT USED TO BE A SUBTRACTION, WHICH IS THE SINGLE "
        "BIGGEST THING THE MICRODATA CHANGED.** The province build could only reach this row "
        "as table 2.10a's Christianity cell less table 2.10b's fourteen rows, 567,039 people, "
        "and 336,482 of that was North West's defective report rather than anybody's "
        "unstated denomination. Here it is the `Christianity` question's own `Do not know` "
        "(227,585) and `Unspecified` (2,976), counted per person. "
        "**North West is no longer the outlier it was.** It read 10.45% not-reported against "
        "0.20-0.91% elsewhere, which was Report 03-01-11's arithmetic and not a fact about "
        "North West; it now reads **0.64%**, inside a national range of 0.20% (Limpopo) to "
        "0.91% (Western Cape). The 336,482 went to `Christian: Other`, where that report's "
        "own footnote says they belong. `sources/za.py` asserts the defect is still in the "
        "PDF, because the reconciliation it runs expects exactly that shortfall.",
    "Christian: Other":
        "-> christianity.other. 3,845,643 people, 8.86% of Christians, and **336,482 of that "
        "is North West's, restored from Report 03-01-11's mis-set row** -- see the entry "
        "above. Larger than it looks for a residual, and the reason is that South Africa's "
        "card names thirteen denominations and the country has thousands of independent "
        "congregations that match none of them. "
        "**Its geography is the arid west and it is not subtle.** 33.49% of Dawid Kruiper's "
        "Christians in the Northern Cape (n=2,371), 25.24% of Dr Beyers Naude's (n=1,022) and "
        "24.61% of Beaufort West's (n=834) against 0.38% in Umhlabuyalingana (n=40) and 0.48% "
        "in Nkandla (n=25). The high end is the Karoo and the Kalahari, where the named "
        "denominations on the card are the ones a Cape Town or a Durban questionnaire "
        "designer would name; the low end is deep rural KwaZulu-Natal, where almost everyone "
        "has a box that fits.",
    "Christian: Anglican/Episcopalian":
        "-> christianity.anglican. 1,765,279 people, 4.07% of Christians. The Anglican Church "
        "of Southern Africa. **33.00% of Hessequa's Christians (n=934) and 16.77% of Mossel "
        "Bay's (n=729) against 0.02% in Maphumulo (n=1)**; that is the southern Cape coast "
        "against the KwaZulu-Natal interior, and the province tier compressed it to 8.26% "
        "versus 1.02%.",
    "Christian: Methodist":
        "-> christianity.methodist, the parent and no branch. 2,777,934 people, 6.40% of "
        "Christians. The Methodist Church of Southern Africa, and separately the several "
        "African Methodist bodies that split from it; the survey names neither, so nothing "
        "below the parent is inferred. **33.82% of Ntabankulu's Christians (n=2,147) and "
        "33.41% of Umzimvubu's (n=4,723) against 0.16% in Greater Giyani (n=21)**. Those two "
        "are the old Transkei, and this is the row that shows most clearly what nine "
        "provinces cost: Eastern Cape as a whole is 15.18%, which is the average of a "
        "Methodist Transkei and a Reformed and Anglican coast.",
    "Christian: Seventh Day Adventist":
        "-> christianity.adventist, the parent rather than the `.sda` leaf. 311,267 people, "
        "0.72% of Christians. The leaf is the named denomination and this map reserves it for "
        "sources that count that body specifically; a census answer box reading `Seventh Day "
        "Adventist` is what a respondent said, and other Adventist bodies would land in it. "
        "Two of the nine provincial reports spell it `Seventh-Day`; `sources/za_profiles.py` "
        "folds that, and the microdata's own codebook spells it without the hyphen. Its peak "
        "is Ngqushwa in the Eastern Cape at 6.20% of that municipality's Christians (n=231).",
    "Christian: Mormon":
        "-> christianity.latterday, the parent rather than the `.lds` leaf, for the same "
        "reason as the Adventist row. 114,808 people, 0.26% of Christians and the smallest "
        "Christian cell. Stats SA's exemplar is *Church of Jesus Christ of Latter Day "
        "Saints*, so the parent is barely wider than the leaf here, but the parent is what "
        "the answer supports. **Nowhere reaches 2% of a municipality's Christians and 24 of "
        "the 213 have none at all**, so this row draws as scattered single dots and will "
        "often ring (§4.3).",
    "Christian: Just a Christian/non-denominational":
        "-> christianity.nondenominational. 2,501,373 people, 5.76% of Christians. **This is "
        "a different row from `Denomination not reported` above and the difference matters**: "
        "this one is an answer a respondent chose off the card, the other is the arithmetic "
        "left by people whose denomination was not established. Its geography is urban and "
        "Highveld -- 17.03% of Steve Tshwete's Christians (n=1,480), 15.62% of Midvaal's "
        "(n=567), 15.56% of Msukaligwa's (n=1,116) against 0.23% in Walter Sisulu (n=15). "
        "**And it carries this country's one large fieldwork anomaly.** Swellendam, a Western "
        "Cape municipality of 40,211 people, returns 57.34% of its answers in this row "
        "against 7.06% of its province's Christians, on 965 of its 1,685 records; its "
        "Pentecostal, Catholic, Methodist and Muslim cells are all far below the provincial "
        "rate at the same time. That is the shape of one enumeration team's habit rather than "
        "a town, nothing here can correct it (§14.4 forbids inventing the magnitude), and "
        "sources/za.md §2.3 records it beside the other one.",
    "Religion: Traditional African religion":
        "-> indigenous.african, the node Ghana added. 2,454,888 people, 4.47%. "
        "**Read it as a floor**, for the reason sources.md §11b gives for the whole continent "
        "and `zw2022.py` gives next door: the box is exclusive of the Christian ones, and "
        "consulting a *sangoma* or honouring the *amadlozi* very commonly accompanies church "
        "membership here rather than replacing it. South Africa is a sharper case than most, "
        "because the African Independent Churches counted separately at six times the size "
        "grew out of exactly that overlap. "
        "**The floor is much higher in one place than nine provinces could show**: 31.35% of "
        "Maphumulo's answers (n=2,026), 30.78% of Mkhambathini's (n=972) and 26.03% of "
        "Msinga's (n=2,751), all in the KwaZulu-Natal midlands, against zero in Walter Sisulu "
        "and Lekwa-Teemane. KwaZulu-Natal as a whole is 7.36%. No child node: no Stats SA "
        "release names a South African tradition individually (§2.4). **And note this is the "
        "category the two instruments disagree on most**: Census 2022 puts it at 7.8% against "
        "4.5% here, moving the opposite way to the no-religion cell. See sources/za.md §3.",
    "Religion: No religious affiliation/belief":
        "-> unaffiliated, with Atheism and Agnosticism kept separate below because the survey "
        "counts them separately. 5,964,889 people, 10.86%. "
        "**Its geography is not the urban-secular one a reader will expect, and the municipal "
        "tier makes that harder to explain rather than easier.** The top three are Okhahlamba "
        "in KwaZulu-Natal at 32.67% (n=3,067), Ephraim Mogale in Limpopo at 27.66% (n=2,473) "
        "and Blouberg in Limpopo at 26.78% (n=3,339) -- all rural, all in provinces where the "
        "African Independent Churches are strongest -- against 0.06% in Breede Valley (n=5) "
        "and 0.17% in Gamagara (n=4). Johannesburg is 12.83% and Cape Town 7.79%, both below "
        "the national figure. A plausible reading is that where church membership is the "
        "dominant idiom, people outside a particular church answer `none` rather than naming "
        "an ancestral practice, which is Benin's `Aucune` warning in a different country. "
        "Nothing here resolves it. **The bigger caution is vintage.** Census 2022 puts this "
        "cell at 2.9% against 10.86% here, a factor of nearly four in six years, which no "
        "trend explains; sources/za.md §3.",
    "Religion: Atheism":
        "-> secular, and `Religion: Agnosticism` goes to the same node. Together 85,546 "
        "people, 0.16%. Kept off `unaffiliated` because the survey asks them as separate "
        "boxes, which is the test au2021.py, ba2013.py, br2010.py and bs2022.py apply; "
        "`zw2022.py` sends its no-religion cell to `unaffiliated` alone precisely because "
        "Zimbabwe has only the one box. The two merge here because the tree has no child "
        "under `secular` for either and neither is large enough to want one. "
        "**AND THIS IS THE ROW THAT SHOWS WHAT A FINE TIER COSTS A SURVEY.** Cape Town holds "
        "30.6% of every atheist counted in South Africa and Johannesburg 15.6%, which is what "
        "a reader expects. **uPhongolo, a rural KwaZulu-Natal municipality of 141,248 people, "
        "holds 12.6%** -- 6,630 people, 4.76% of its own answers against 0.13% in its "
        "province and 0.10% nationally -- and it is not one household's weight: 384 of its "
        "8,121 records. Its Catholic (1.35% against a provincial 9.20%), Methodist (0.28% "
        "against 3.41%) and Anglican (0.40% against 1.82%) cells are all far below the "
        "province at the same time. That reads as one enumeration team coding an adjacent "
        "answer, and nothing here can establish it or correct it. Drawn as returned; "
        "sources/za.md §2.3.",
    "Religion: Other":
        "-> other.za. 1,482,207 people, 2.70%. See the node's own note: a genuine tail rather "
        "than a store cupboard, because Islam, Hinduism, Judaism, Buddhism, Bahaism and "
        "Traditional African religion all have their own boxes. At province it was unusually "
        "flat, 3.87% to 1.18%, and nothing was inferred from that; **at municipality it is "
        "12.07% of Nqutu's answers (n=1,525) and 11.39% of uMuziwabantu's (n=633) against "
        "0.14% in Mfolozi**, both of the high pair in KwaZulu-Natal. That is a real geography "
        "and it still names nothing, because a residual that varies is not a residual that "
        "explains itself (§9r's Chittagong rule). Per §3.11.",
    "Religion: Hinduism":
        "-> hinduism, with no branch, because the survey gives none. 561,268 people, 1.02%. "
        "**The most concentrated religion in the country**: 9.23% of KwaDukuza's answers "
        "(n=1,220) and 8.98% of eThekwini's (n=14,179) against zero in dozens of "
        "municipalities. KwaDukuza is Stanger, on the Natal north coast, and eThekwini is "
        "Durban; both are the population descended from the indentured labourers brought to "
        "the sugar estates from 1860, and the province figure of 3.99% is those two averaged "
        "with a Zulu interior that has almost none. Durban is often called the largest Indian "
        "city outside India; South Africa is NOT the largest Indian-descended population "
        "outside Asia, which an earlier draft of this line claimed, because the United States "
        "and the United Kingdom are both larger.",
    "Religion: Islam":
        "-> islam, with no branch, because the survey gives none and the two communities it "
        "holds belong to different ones. 892,688 people, 1.62%. **8.41% of Cape Town's "
        "answers (n=13,424), 3.59% of eThekwini's and 3.34% of Johannesburg's**, against zero "
        "in much of the Eastern Cape. The Cape figure is largely the Cape Malay, Shafi'i "
        "Muslims descended from people exiled and enslaved from the Dutch East Indies from "
        "the 1650s; the Gauteng and KwaZulu-Natal figures are largely Hanafi Muslims of South "
        "Asian descent. Nothing in the survey separates them, and at 213 units the two "
        "communities are at least in different municipalities rather than in the same "
        "province.",
    "Religion: Judaism":
        "-> judaism, with no branch. 49,467 people, 0.09%. Almost entirely Lithuanian "
        "Ashkenazi in descent and almost entirely in two cities: **0.48% of Johannesburg's "
        "answers (n=505) and 0.32% of Cape Town's (n=260)**, which between them are 78% of "
        "every Jewish answer in the country. Unlike Zimbabwe's cell next door, this figure "
        "and its geography are both what the community's history predicts, so nothing about "
        "it needs a second reading.",
    "Religion: Buddhism":
        "-> buddhism, with no branch. 24,805 people, 0.05%. Gauteng's provincial report "
        "writes the label `Buddism` and the other eight write `Buddhism`; **the microdata's "
        "codebook writes `Buddism` too**, so it is the survey's own spelling rather than one "
        "report's typo, and `sources/za_profiles.py` folds it for both readers.",
    "Religion: Bahaism":
        "-> bahai. 6,880 people, 0.01%, the smallest cell in the country and under a dot in "
        "most municipalities, so it draws sparsely and will often ring (§4.3). Northern Cape "
        "printed it as a bare dash rather than a zero in its provincial report; that trap is "
        "gone here, because the microdata carries the code and 62 municipalities return a "
        "genuine zero.",
}

MAP = {
    # --- the ten non-Christian answers to `ReligionBelief`. Christianity is not here: it is
    # --- replaced by the `Christianity` question's fourteen denominations plus the residual.
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

    # --- the fourteen Christian denominations.
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

    # --- Christians whose denomination the survey did not establish.
    "Christian: Denomination not reported": "christianity",
}

# NO `COLUMNS` DICT, DELIBERATELY, and this is the healthy case rather than an omission.
# COLUMNS exists for adapters that emit `derived` rows and need to say which measured column
# each one came out of (spec §7a-i-1). Every row South Africa draws is `measured` at the
# municipality it is drawn on -- Stats SA asked the question of those households, in that
# municipality -- so `inferred dots: not shown` removes nothing here and there is no column
# for a roll-up to name. `python tools/check_rollup.py za` says so.


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
