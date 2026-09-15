"""ESS `rlgblg` x `rlgdnno` -> religiondots taxonomy. Norway's Norwegian-citizen half.

ESS asks `rlgblg` (do you consider yourself as belonging to any particular religion or
denomination) and then, for Norway, `rlgdnno`: nine answers, in Norwegian even with
`metadataLanguage:"en"`, in every round from 5 to 11. `rlgdnano` and `rlgdnbno` do not exist.
`rlgdnno` is the harmonised `rlgdnm` with one answer split, and sources/no.py proves the nesting
from the data on every build.

    1  Den norske kirke                      -> christianity.lutheran
    2  Katolsk kirke                         -> christianity.catholic.latin
    3  Ortodoks kirke (gresk, russisk,       -> SPLIT TWO WAYS, see below
       andre)
    4  Andre protestantiske trossamfunn      -> christianity.protestant
       (f.eks. frikirker, anglikanske kirke,
       pinsevenner og andre)
    5  Andre kristne trossamfunn (f.eks.     -> christianity
       Jehovas vitner, mormonerne)
    6  Det mosaiske trossamfunn (jødisk)     -> judaism
    7  Islam (muslimsk)                      -> islam.sunni
    8  Østlige religioner (f.eks. ...)       -> other.no
    9  Andre ikke-kristne religioner         -> other.no
    (rlgblg = No)                            -> unaffiliated
    (refusal / don't know / no answer)       -> excluded, spec §3.5

ANSWER 3 IS SPLIT, ON SWEDEN'S RULE AND FOR A DIFFERENT REASON. sources.md §9cz: before sending
an undifferentiated Orthodox answer to `christianity.orthodox`, check the country. Norway's check
is the list of grant-counted members per faith community that the ministry publishes each year
(`antall tilskuddstellende medlemmer i tros- og livssynssamfunn`). regjeringen.no answers 403 to a
script and to WebFetch; the 2018 and 2024 PDFs were read from the Wayback Machine's copies, both
whole (%%EOF present). **The 2018 list is used**, because 2018 is the last year of the survey
rounds this answer is drawn from (rounds 5-9, 2010-2018).

    Eastern Orthodox     15,279   59.49%   Serbian, Russian (four parishes), Greek, Romanian,
                                           Bulgarian, the Norwegian-language parishes
    Oriental Orthodox    10,403   40.51%   Eritrean (most of it), Ethiopian, Coptic, Armenian

There is no Church of the East parish on the list and no Syriac one, which is where Norway
differs from Sweden. The non-Chalcedonian share is the Eritrean churches: about forty
congregations from Kristiansand to Nordreisa, most founded after 2000.

**Three things about the split that are named rather than fitted.**

  * **The list counts residents and this is a citizen cell, and in Norway that bias runs the
    OPPOSITE way from Sweden's.** Sweden's Syriac population is old and naturalised, so the
    borrowed ratio was conservative there. Norway's Eritrean population arrived mostly after 2008
    and largely after 2014, and naturalisation takes seven years, so in 2010-2018 many Eritrean
    Orthodox were not yet citizens and are drawn from the census half instead (origin_religion.py
    sends Eritrea's Christians to `christianity.oriental`). So 40.51% OVERSTATES the Oriental share
    of the citizen cell. Left unadjusted because nothing publishes the adjustment; the error is a
    few thousand people between two sibling nodes, uniformly across the counties.
  * **The card itself leans Eastern.** Its examples are *gresk, russisk*, and an Eritrean
    respondent may well have picked answer 5 instead. That moves people out of this cell, not
    between its halves.
  * **Three rows are assigned by naming convention and not by a stated church**: `ENGELEN SANKT
    GABRIEL ORTODOKSE KIRKE` (57), `ST. GEORGS ORTODOKSE, KRISTNE KIRKE` (198) and `ST.MICHEAL`
    (52). All three follow the Eritrean congregations' patterns (the dedication, the `ortodokse
    kristne kirke` phrasing) and none names a jurisdiction. 307 people, 1.2% of the total.
    Four `apostolisk` rows on the list are Pentecostal and are not here.

AND THE THING THIS FILE CANNOT DO. Norway's free churches are Pinsebevegelsen, Den
evangelisk-lutherske frikirke, Misjonsforbundet, the Baptists, the Methodists and more, and the
Church of Norway's own lay-mission movement (Indremisjonsforbundet, Normisjon, NLM) is inside
answer 1, which is Finland's Laestadian problem and Sweden's EFS problem again. ESS offers the
free churches one box, `Andre protestantiske trossamfunn`.
"""

MEMBERS_YEAR = 2018

# `Antall tilskuddstellende medlemmer i tros- og livssynssamfunn 2018`, regjeringen.no, read from
# web.archive.org/web/2019id_/... on 2026-09-14. Names are the list's own, in its capitals.
# The full URL (added in review, 2026-09-14): https://web.archive.org/web/2019id_/https://www.
# regjeringen.no/contentassets/8e328011f0e541d99e0b6243387e38dd/antall-tilskuddstellende-
# medlemmer-i-tros-og-livssynssamfunn-2018.pdf (26 pages, 1,003,099 bytes, %%EOF present).
GRANT_MEMBERS_2018 = {
    "christianity.orthodox": {
        "DET SERBISKE ORTODOKSE KIRKESAMFUND I NORGE HL VASILIJE OSTROSKI MENIGHET": 4507,
        "HELLIGE OLGA MENIGHET-DEN RUSSISKE ORTODOKSE KIRKE": 3375,
        "DEN GRESKE ORTHODOKSE MENIGHET EVANGELISMOS TIS THEOTOKOU": 1596,
        "HELLIGE DEMETRIOS AV THESSALONIKI": 1169,
        "HELLIGE ANNA MENIGHET - DEN RUSSISKE ORTODOKSE KIRKE": 1028,
        "DEN ORTODOKSE KIRKE I NORGE - HELLIGE NIKOLAI MENIGHET": 1000,
        "KRISTI ÅPENBARINGSMENIGHET DEN RUSSISK-ORTODOKSE KIRKE (MOSKVAPATRIARKATET)": 783,
        "DEN RUMENSKE ORTODOKSE MENIGHET I OSLO DE HELLIGE MARTYRER IOAN ROMANUL OG HALLVARD":
            753,
        # Russian Orthodox, Rogaland (Bryne), per the parish's own site at ortodoks.no.
        "HELLIGE IRINA MENIGHET": 384,
        "DEN BULGARSKE ORTODOKSE MENIGHET I NORGE ST. KIRIL OG ST. METODIJ": 282,
        "DEN RUMENSKE ORTODOKSE MENIGHETEN I HAUGESUND DEN HELLIGE MACARIE DEN STOR": 218,
        "RUSSISK ORTODOKSE HELLIGE TRIFON MENIGHET I SØR-VARANGER": 156,
        "DEN ORTODOKSE KIRKE I NORGE HELLIGE HALLVARD MENIGHET": 28,
    },
    "christianity.oriental": {
        "DEN ERITREISKE KOPTISKE KIRKE I NORGE": 1511,
        "DEN ERITREISK KOPTISK AMANUEL KIRKE I KRISTIANSAND": 988,
        "DEN ETIOPISKE ORTODOKSE KIRKE I NORGE": 877,
        "DEN ERITREISKE ORTODOKSE TEWAHEDO KIRKE DEBRE IYESUS I BERGEN": 643,
        "DEN ERITREISKE ORTODOKSE MENIGHET I ROGALAND": 566,
        "DEN ERITREISKE ORTODOKSE SAINT MICHAEL KIRKE I TRONDHEIM": 478,
        "DEN ERITREISK ORTODOKSE MENIGHET I BUSKERUD": 456,
        "ERITREISK ST. KIDANE MIHRET TEWAHDO ORTHODOKSES KIRKEN I GRENLAND": 402,
        "ERITREISK ORTHODOX TEWAHDO KIRKE DEBREHAWARYAT KDUS PETROS & PAULOS ASKER & BÆRUM": 304,
        "DEN ERITRISKE ORTODOKS TWAHDO I MOLDE": 282,
        "DEN ERITREISK ORTODOKSE MENIGHET I VESTFOLD": 267,
        "DEN ETIOPISK ORTODOKS TEWAHEDO MEDHANEALEM KIRKE": 262,
        "MENBERE LEUL ST. MICHAEL ETHIOPISKE ORTODOKSE TEWAHEDO MENIGHET I BERGEN": 243,
        "DET ARMENSKE APOSTOLISKE KIRKESAMFUNN": 198,
        "ST. GEORGS ORTODOKSE, KRISTNE KIRKE": 198,           # by naming convention
        "ST.RUFAEL ERITREISK ORTODOKSE TEWAHDO KIRKE HAMAR": 198,
        "VOLDA ERITREISK - ORTODOKSE KYRKJE": 170,
        "MEDHANIE-ALEM ORTODOKSE KRISTNE KIRKE": 149,
        "ERITREISKE ORTODOKSE TEWAHEDO KIRKE ABUNE TEKLEHAYMANOT I SOGNDAL": 147,
        "ST. GEORGE DEN ETIOPISK ORTODOKS TEWAHEDO KIRKE I TRONDHEIM": 146,
        "ERITREISK ORTODOKS TEWAHEDO MEDHANI ALEM KIRKE I HARSTAD": 137,
        "KIDANEMHRET KIRKE I FJELLREGIONEN": 129,
        "DEN ERITREISKE ORTODOKSE TEWAHDO KIDIST DINGEL MARIAM MENIGHET I STAVANGER": 122,
        "DEBRE TSION ST. MARIA ETIOPISK ORTODOKS TEWAHEDO MENIGHET KRISTIANSAND": 121,
        "DET ERITREISK OG ETIOPISK ORTODOKS KIRKE SAMFUNN I ELVERUM REGIONEN (DEEOKSIER)": 119,
        "ERITREAN ORTHODOX TEWAHDO": 110,
        "DEN ERITREISKE KOPTISK ORTODOKSE KIRKE I LARVIK": 94,
        "ORTODOKS KIRKE DEBRE SELAM KDUS MICHAEL KONGSVINGER OG OMEGN": 93,
        "KIDIST MARIAM KOPTISK ORTODOKSE MENIGHET NARVIK": 85,
        "DEN ERITREISKE ORTODOKSE TEWAHDO ABUNE AREGAWI MENIGHET PÅ JØRPELAND": 83,
        "DEN ORTHODOKSE TEWAHEDO MENIGHET I LEVANGER": 82,
        "DEN ERITREISKE ORTODOKSE KRISTEN ABUNE TEKLEHAYMANOT KIRKE I SANDNESSJØEN": 74,
        "ST. URAEL DEN ETIOPISKE ORTODOKSE KIRKE I HEDMARK": 68,
        "KIDANE MIHRET ERITREISK ORTODOKS KIRKE": 66,
        "DEN ERITREISKE ORTODOKSE SAINT KIDANE MHRET KIRKE I STEINKJER": 63,
        "DEN ERITREISKE KIDIST MARIAM ORTODOKS TEWAHDO KIRKE I FØRDE": 61,
        "DEN ETIOPISKE ORTODOKSE TEWAHEDOKIRKEN I TROMSØ": 57,
        "ENGELEN SANKT GABRIEL ORTODOKSE KIRKE": 57,           # by naming convention
        "ST.MICHEAL": 52,                                     # by naming convention
        "MEDHANEALEM KANONISK ERITRISK ORTODOKS KIRKE I OSLO": 47,
        "DEN ERITREISKE ORTODOKSE ST. GABRIEL KIRKE I STJØRDAL": 39,
        "DEN KOPTISKE-ORTODOKS KIRKE I NORGE": 37,
        "ABUNE AREGAWI ORTHODOKSE TEWAHDO MENIGHET NÆRØY": 36,
        "MEDHANIALEM KIRKE I VIKNA": 28,
        "ERITREISK ORTHODOKS TEWAHDO KDSTI DENGEL MARIAM I NORDREISA": 25,
        "DEN ERITREISK ORTODOKSE SAINT GABRIEL MENIGHET I GRIMSTAD": 22,
        "ERITRIESK ORTHODOX TEWAHDO ST. GABRIEL KIRKE": 11,
    },
}
assert sum(GRANT_MEMBERS_2018["christianity.orthodox"].values()) == 15_279
assert sum(GRANT_MEMBERS_2018["christianity.oriental"].values()) == 10_403

ORTHODOX_ANSWER = "Ortodoks kirke (gresk, russisk, andre)"

_ORTH_TOTALS = {node: sum(v.values()) for node, v in GRANT_MEMBERS_2018.items()}
_ORTH_ALL = sum(_ORTH_TOTALS.values())
ORTHODOX_SPLIT = {
    f"{ORTHODOX_ANSWER} (Eastern Orthodox)": _ORTH_TOTALS["christianity.orthodox"] / _ORTH_ALL,
    f"{ORTHODOX_ANSWER} (Oriental Orthodox)": _ORTH_TOTALS["christianity.oriental"] / _ORTH_ALL,
}

LUTHERAN = "Den norske kirke"
FREE_CHURCHES = ("Andre protestantiske trossamfunn (f.eks. frikirker, anglikanske kirke, "
                 "pinsevenner og andre)")
OTHER_CHRISTIAN = "Andre kristne trossamfunn (f.eks. Jehovas vitner, mormonerne)"
JEWISH = "Det mosaiske trossamfunn (jødisk)"
ISLAM = "Islam (muslimsk)"
EASTERN = ("Østlige religioner (f.eks. buddhisme, hinduisme, sikh, shintoisme, taoisme, "
           "konfutsianisme)")
OTHER_NONCHRISTIAN = "Andre ikke-kristne religioner"
NO_RELIGION = "No religion"

# The ten answers the pooled rounds produce, before the Orthodox split. sources/no.py asserts the
# pool against this set in BOTH directions (fi.py §8).
SOURCE = {
    LUTHERAN, "Katolsk kirke", ORTHODOX_ANSWER, FREE_CHURCHES, OTHER_CHRISTIAN, JEWISH, ISLAM,
    EASTERN, OTHER_NONCHRISTIAN, NO_RELIGION,
}

EXCLUDED = {
    "__refused__":
        "Everyone whose `rlgblg` is Refusal, Don't know or No answer, and everyone who answered "
        "Yes and then declined to name a denomination. spec §3.5 marks a refusal rather than "
        "filling it; sources/no.py prints the share and `gap` states it.",
}

REVIEW = {
    LUTHERAN:
        "-> christianity.lutheran, the branch, not a named node for the Church of Norway, which "
        "is fi2024.py's and se2024.py's call: a named node would be a single-country legend row. "
        "**This cell is self-identification and not membership.** SSB's table 12025 puts 67.7% "
        "of Norway on the church's rolls in 2020; this answer is much smaller, and sources/no.md "
        "prints the two side by side at the 11 counties. The lay-mission movement inside the "
        "church (Normisjon, NLM, Indremisjonsforbundet) is here too and has no separate answer.",
    ORTHODOX_ANSWER:
        "-> SPLIT between christianity.orthodox (59.49%) and christianity.oriental (40.51%) on the "
        "2018 grant-counted member list. The module docstring has the rows and why the ratio "
        "overstates the Oriental half of a citizen cell in Norway, which is the reverse of Sweden.",
    FREE_CHURCHES:
        "-> christianity.protestant, `Protestant, unspecified`, se2024.py's call for the same "
        "shape: the parenthesis is examples offered to the respondent (free churches, the "
        "Anglican church, Pentecostals, others), not categories counted. Pinsebevegelsen, the "
        "Evangelical Lutheran Free Church, Misjonsforbundet, the Baptists and Methodists are one "
        "cell here. The Evangelical Lutheran Free Church is Lutheran and is still filed as "
        "unspecified Protestant, because the respondent's box does not say which free church.",
    OTHER_CHRISTIAN:
        "-> christianity, the root, following se2024.py and fi2024.py. The examples are Jehovah's "
        "Witnesses and Latter-day Saints, which are two different branches, and the respondent "
        "named neither.",
    "Katolsk kirke":
        "-> christianity.catholic.latin. Norway's Catholics are the Oslo diocese and the two "
        "territorial prelatures of Trondheim and Tromsø, all Latin rite, and are largely Polish, "
        "Lithuanian, Filipino and Vietnamese in origin. Eastern-rite Catholics are not separable "
        "in this half.",
    ISLAM:
        "-> islam.sunni, se2024.py's and be2024.py's call. Norway's Muslims are majority Sunni "
        "(Pakistani, Somali, Iraqi Kurdish, Bosnian, Turkish, Syrian); the Shia minority "
        "(Iranian, Iraqi, some Pakistani) is not separable in this half, and the Shia dots come "
        "from the foreign half only.",
    JEWISH:
        "-> judaism, the root. Det Mosaiske Trossamfund is the Oslo congregation's own name; "
        "Trondheim has the other. The answer names the body and counts no movement.",
    EASTERN:
        "-> other.no with answer 9, for gr2024.py's and se2024.py's reason: the answer names six "
        "traditions and counts them as one box, and choosing Buddhism over Hinduism would invent "
        "a fact about a respondent. Buddhism is the largest of them on SSB's roll (21,555 members "
        "in 2020), mostly Vietnamese and Thai; the Hindu temples are largely Tamil.",
    OTHER_NONCHRISTIAN:
        "-> other.no. Sikhs are counted by SSB separately on the national roll; Baha'i too. This "
        "instrument offers one box.",
    NO_RELIGION:
        "-> unaffiliated. Everyone who answered NO to `rlgblg`. **Norway's life-stance "
        "organisations are real and state-funded and this map cannot see them**: the "
        "Human-Etisk Forbund is on the same grant roll as the churches, with 99,468 life-stance "
        "members on SSB's 2020 county table, and ESS never offers humanist or atheist as a "
        "denomination, so its members are in this cell with everyone who simply does not belong. "
        "Nothing in Norway reaches `secular`, which is Belgium's call for the same reason.",
}

MAP = {
    LUTHERAN: "christianity.lutheran",
    "Katolsk kirke": "christianity.catholic.latin",
    f"{ORTHODOX_ANSWER} (Eastern Orthodox)": "christianity.orthodox",
    f"{ORTHODOX_ANSWER} (Oriental Orthodox)": "christianity.oriental",
    FREE_CHURCHES: "christianity.protestant",
    OTHER_CHRISTIAN: "christianity",
    JEWISH: "judaism",
    ISLAM: "islam.sunni",
    EASTERN: "other.no",
    OTHER_NONCHRISTIAN: "other.no",
    NO_RELIGION: "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
