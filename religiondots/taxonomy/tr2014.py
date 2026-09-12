"""Türkiye'de Dinî Hayat Araştırması (Diyanet İşleri Başkanlığı + TÜİK, 2014) -> taxonomy.

**Eight drawn categories on twelve İBBS-1 regions, and the whole country is `modelled`.**
The source is a 21,632-person survey rather than a count, so §7b's test — were these people
COUNTED — fails everywhere, and `inferred dots: hidden` empties Türkiye completely. That is
the honest test of the country and it is worth performing rather than describing.

    from Table 4, page 42 (`ameli mezhep`, shares OF MUSLIMS, by region)
    77.5%  Hanefi                      -> islam.sunni.hanafi
    11.1%  Şafi                        -> islam.sunni.shafii
     1.0%  Caferi                      -> islam.shia.jaafari
     0.1%  Hanbeli                     -> islam.sunni.hanbali
     0.03% Maliki                      -> islam.sunni.maliki
     6.3%  Mezhep: Hiçbiri             -> islam
     2.4%  Mezhep: Bilmiyorum          -> islam
     0.9%  Mezhep: Cevap vermeyen      -> islam
     0.8%  Mezhep: Diğer               -> islam

    from Table 1, page 38 (religion, NATIONAL ONLY, laid on every region at the same rate)
     0.4%  Din: Diğer                  -> other.tr
     0.5%  Din: Cevap vermeyen         -> unknown

**FOUR COLUMNS COLLAPSE ONTO `islam` AND THAT IS THE MOST IMPORTANT DECISION IN THIS FILE.**
`Hiçbiri`, `Bilmiyorum`, `Diğer` and `Cevap vermeyen` are 10.4% of Turkish Muslims between
them. Everyone in all four answered *İslamiyet* at question 10, so their RELIGION is measured
and only their school is not; the parent node is exactly the right place for them, and it is
where Russia's *"I profess Islam, but am neither Sunni nor Shia"* already sits (`islam.shia`'s
note). None of them belongs on `unknown`, which is for people whose religion the source did
not establish.

**AND IT IS WHERE TÜRKİYE'S ALEVIS ARE, WITHOUT THIS FILE CLAIMING SO.** The report reprints
its own questionnaire, and question 11 offers Hanefi, Şafi, Maliki, Hanbeli, Caferi, Nusayri,
Bilmiyorum, Diğer, Hiçbiri and a refusal. There is no Alevi option, and the word does not
appear once in 293 pages. So an Alevi respondent's answer is in one of those four columns —
but so is a Hanafi who never thought about the question, and the source cannot tell them
apart. Drawing the four on `islam` says only what is true: these people are Muslim and the
survey did not establish which school.

**DO NOT MODEL ALEVIS OUT OF `Hiçbiri`.** It is 20.8% in Batı Marmara and 11.9% in the Aegean
against 6.5% in Orta Anadolu, which holds Sivas and Yozgat and is the Alevi heartland by every
settlement count there is. Whatever that column is measuring, its geography is the opposite of
Alevism's; §14.12 found which way this kind of split fails and this would fail that way.
KONDA's independent national survey puts Alevis at 5.02% and Nusayris at 0.1% (sources.md
§11ac), and **no source published anywhere gives either figure by region** — checked across
KONDA's whole archived library, the World Values Survey, and the ESS.

**`Nusayri` IS ON THE CARD AND NOT IN THE PUBLISHED TABLE.** Option 6 of question 11, folded
into `Diğer` in Table 4 — which peaks at 2.0% in Akdeniz, the region holding Hatay and Adana,
where the Nusayris are, against 0.8% nationally. That is a good internal check on the report
and it is not enough to draw from, because the column also holds every other write-in.

**THE TWO NATIONAL ROWS ARE SPREAD, AND `derived` WOULD BE THE TIER IF THE COUNTRY HAD ONE.**
Table 1 is published for Türkiye and for nothing smaller, so `Din: Diğer` and `Din: Cevap
vermeyen` sit at the national rate in every region: they carry no geography and must not be
read as saying Turkey's Christians are evenly spread. They are `modelled` like everything else
here, which is the weaker claim of the two, so nothing is lost by not marking them separately.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Toplam":
        "the region's own total, not a category. sources/tr.py drops the total column "
        "before it gets here; this entry exists so a layout change fails loudly.",
}

REVIEW = {
    "Mezhep: Hiçbiri":
        "-> islam. 6.3% of Turkish Muslims, and the single largest judgement in this file. "
        "The answer is `none of these schools`, given by someone who has already said they "
        "are Muslim, so the parent node is where they belong and `unaffiliated` would be a "
        "straightforward error. It is ALSO where an Alevi respondent lands, since the card "
        "offers no Alevi box, and the file's docstring is explicit that this node cannot "
        "separate the two and does not try.",
    "Mezhep: Bilmiyorum":
        "-> islam. 2.4%. `I do not know my school`. Their religion is established and their "
        "school is not, which is the parent node's meaning exactly. Not `unknown`, which is "
        "for people whose RELIGION the source did not establish.",
    "Mezhep: Cevap vermeyen":
        "-> islam. 0.9%. Declined question 11 having answered question 10. Same reasoning as "
        "`Bilmiyorum`: what was declined is the school, not the religion.",
    "Mezhep: Diğer":
        "-> islam. 0.8%. `Other (specify)`, and the card's own list puts NUSAYRI here, since "
        "option 6 exists on the questionnaire and has no column in the published table. It "
        "peaks at 2.0% in Akdeniz, which is Hatay and Adana. Drawing it as "
        "`islam.shia.alawite` was considered and refused: the column also holds every other "
        "write-in, the report never says how it splits, and inventing that split is §14 rule "
        "1. If the microdata is ever released this is the row that would move.",
    "Din: Diğer":
        "-> other.tr. 0.4% of the country, about 341,000 people. The report's own words are "
        "*belongs to a religion other than Islam OR belongs to no religion*, so this ONE CELL "
        "holds Türkiye's Christians, Jews and the irreligious together and cannot separate "
        "them. `other.tr` is the closest true node: a religion answer the tree cannot place. "
        "The alternative was not drawing it, which would have rendered Türkiye as a country "
        "with no non-Muslims at all, and §6.12 is about how a blank reads.",
    "Din: Cevap vermeyen":
        "-> unknown. 0.5%. Declined question 10, so the source counted them and did not "
        "establish their religion, which is `unknown`'s admission test word for word.",
}

MAP = {
    # Table 4, by region. Shares of Muslims; sources/tr.py scales them into the 99.2%.
    "Hanefi": "islam.sunni.hanafi",
    "Şafi": "islam.sunni.shafii",
    "Maliki": "islam.sunni.maliki",
    "Hanbeli": "islam.sunni.hanbali",
    "Caferi": "islam.shia.jaafari",
    "Mezhep: Diğer": "islam",
    "Mezhep: Hiçbiri": "islam",
    "Mezhep: Bilmiyorum": "islam",
    "Mezhep: Cevap vermeyen": "islam",
    # Table 1, national, laid on every region at the same rate.
    "Din: Diğer": "other.tr",
    "Din: Cevap vermeyen": "unknown",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
