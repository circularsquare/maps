"""LSB 2015 PHC religion (village level, via K4D) -> religiondots taxonomy.

Six categories on 8,499 villages. Four of them are trivial and one is a residual; **the
whole of this module is about the fifth**, which is 31.45% of the country and which the
source labels `no religion`.

**THE CALL: `No religion` -> `indigenous.laos`, NOT `unaffiliated` and NOT `unknown`.** The
evidence is set out at length in the node's own entry in `branches.py`, and in short it is
four independent statements plus a geography:

  * the census defined religion as a spiritual system with WRITTEN DOCTRINES, so animism
    could not be recorded as one (Socio-Economic Atlas of the Lao PDR, §F.5);
  * the atlas, on the identical 2005 cell, says *"a more appropriate term for the 'other'
    category would be Animism"*;
  * the 2015 report's own summary calls the cell *"no religion or being animist"*;
  * Pew (2025) puts Laos's religiously unaffiliated at **under 0.1%** and its `other
    religions` at 34.2%, which is spec §3.11's external estimate naming what the census
    will not;
  * and it runs **96.5% in Dakcheung against 5.9% in Vientiane Capital**, with a rate of
    9.2% among Lao-Tai villagers against 65-79% among everyone else.

**IT IS ONE LINE TO REVERSE.** If the call is wrong the fix is `MAP["No religion"]` and
nothing else; no count, no geography and no other node depends on it.

**NOTHING ELSE HERE NEEDED A NEW NODE.** A census that offers Buddhism, Christianity, Islam
and the Baha'i Faith and nothing else is asking at the level of world religions, and the tree
has had those since the first source.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total Population":
        "the village's own population total, not a category. It is the denominator the two "
        "percentage services are read against and the residual is derived from, so it is "
        "carried in la.csv and dropped here.",
}

REVIEW = {
    "Buddhist":
        "-> buddhism, the PARENT, and deliberately not buddhism.theravada. 4,193,875 "
        "people, 64.7%. Lao Buddhism is Theravada in overwhelming proportion and the sangha "
        "is organised on Thai lines, so filing it as Theravada would almost certainly be "
        "true. It is still not what the source says: LSB offers one cell labelled `Buddhist` "
        "and asks for a religion rather than a school. **kh2019.py, lk2024.py and in2011.py "
        "make the same call on the same tradition**, and no census on this map separates "
        "the vehicles. "
        "**Its geography is the Mekong.** Champasak is 93.2% Buddhist, Vientiane Capital "
        "90.9%, Khammouan 86.5% and Savannakhet 80.1% — the lowland corridor where the Lao "
        "and Tai-Thay live and where the wat is the centre of a village. Against **20.3% in "
        "Oudomxai and 20.9% in Louangnamtha**, which is the same boundary the "
        "`indigenous.laos` node runs along from the other side. The two categories are the "
        "Laos map, and between them they are 96.2% of the country.",
    "Christian":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row. 112,194 people, 1.73%, and LSB names "
        "no body at all. The population is at least three unlike things: the Catholic "
        "vicariates of the Mekong valley, historically Vietnamese and French; the **Lao "
        "Evangelical Church**, one of only two Protestant bodies the state recognises and by "
        "far the larger; and the Seventh-day Adventists, who are the other. Filing on "
        "`christianity.protestant` would assert a division the source does not make. "
        "**Its geography is not the capital and not the lowlands.** Vientiane Capital is "
        "0.79%. The top provinces are **Bokeo 4.93%, Xaisomboun 3.91%, Bolikhamxai 3.47% "
        "and Vientiane province 3.33%** — that is Hmong and Khmu country, and it is the same "
        "upland population the `indigenous.laos` cell draws from. The two are competing for "
        "the same people and the map shows both, which is kh2019.py's Mondul Kiri finding "
        "one country to the south. "
        "**Read the figure as a floor.** Laos's constitution protects religious practice and "
        "its Decree 315 regulates it; Protestant congregations in the uplands have been "
        "closed and members detained, which the US State Department's religious freedom "
        "reporting documents year on year. A census answer given in that setting undercounts.",
    "Muslim":
        "-> islam, with no branch, because the census gives none. **1,603 people, 0.025%, "
        "and it is the smallest national Muslim population drawn anywhere on this map.** "
        "Laos's Muslims are a Vientiane community of South Asian and Cham descent with two "
        "mosques, plus Chinese-speaking Haw traders in the north. **503 are in Vientiane "
        "Capital and 401 in Xaisomboun**, the second being 0.47% of that province against "
        "0.02% nationally and much the sharpest Muslim concentration in the country; "
        "nothing published says what it is. Nothing separates Sunni from Shi'a and nothing "
        "here invents it.",
    "Baha'i":
        "-> bahai. 2,121 people, 0.033%, and **there are more Baha'is in Laos than "
        "Muslims**, which is true of very few countries and is why the 2005 questionnaire "
        "lists the Baha'i Faith as one of its four religions at all. The community dates "
        "from the 1950s. **It is spread rather than concentrated**: Vientiane Capital holds "
        "368 of the 2,121, then Savannakhet 242, Oudomxai 241 and Louangphabang 219, and "
        "the highest provincial share is Oudomxai's 0.08%. Published as a percentage rather "
        "than a count, recovered exactly (sources/la.py).",
    "No religion":
        "-> indigenous.laos. **2,038,393 people, 31.45%, and this is the one argued "
        "decision in the module.** The full case is in the node's entry in `branches.py`; "
        "the summary is in this module's docstring. Two things to hold onto when reading "
        "the map: **it is a ceiling and not a floor**, unlike every other node in the "
        "`indigenous` family, because it is a residual that also contains however many Lao "
        "genuinely report no religion; and **no tradition is named under it**, because no "
        "source names one. The Khmu, Hmong, Akha, Katu, Ta Oi, Brao and Lamet religions are "
        "in one box together and depth follows what a source counts (§2.4).",
    "Others/not stated":
        "-> other.la. 133,296 people, 2.06%, derived as a residual rather than published "
        "(sources/la.py). It pools an answer with a non-answer, and its shape says the "
        "non-answer is most of it: a 1-5% film across 3,934 of the 8,499 villages and a "
        "community nowhere. **Not evidence of Laos's other faiths**, because the four "
        "religions the instrument recognised are each drawn separately already. The same "
        "call as `other.mu` and `other.sk`, per §3.11.",
}

MAP = {
    "Buddhist": "buddhism",
    "Christian": "christianity",
    "Muslim": "islam",
    "Baha'i": "bahai",
    "No religion": "indigenous.laos",
    "Others/not stated": "other.la",
}

# spec §7a-i-1: the source column a derived row was split out of. Nothing here is derived —
# every row is LSB's own figure at LSB's own village — so there is no COLUMNS dict, and
# tools/check_rollup.py should report nothing for `la`.


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
