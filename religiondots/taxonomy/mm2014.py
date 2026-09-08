"""DOP 2014 Census Volume 2-C religion -> religiondots taxonomy.

Seven religion categories plus the universe total, at State/Region — **and an eighth column
that is not a religion at all**, which is the reason this country is on the map.

**The whole published religion output of the 2014 census is two tables in a 17-page report.**
So the category list is world-religion depth with nothing under it: no Buddhist school, no
branch of Islam, no Christian body named, and one undivided cell for every traditional
religion in the country. Myanmar's value here is not taxonomic depth.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the state's own ENUMERATED population total, not a category — and note it is NOT "
        "the universe this country is drawn on, because the non-enumerated sit outside it. "
        "Enumerated 50,279,900 + non-enumerated 1,206,353 = 51,486,253, which is the "
        "report's own overall figure.",
}

REVIEW = {
    "Buddhist":
        "-> buddhism, the PARENT, and deliberately not buddhism.theravada. 45,185,449 "
        "people, 89.9% of the enumerated population. Myanmar is one of the historic centres "
        "of Theravada and its sangha is organised under nine state-recognised Theravada "
        "nikaya, so filing it as Theravada would almost certainly be true. It is still not "
        "what the source says: DOP offers one cell labelled `Buddhist` and asks for a "
        "religion rather than a school. **Third time this call has been made on this "
        "tradition** — lk2024.py on Sri Lanka and kh2019.py on Cambodia are the same "
        "reasoning, and in2011.py a fourth on India. The tree can hold the vehicles apart "
        "the moment a source separates them; no census on this map does.",
    "Christian":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row. 3,172,479 people, 6.3%. DOP names no "
        "body, and the population is at least three unlike things: the Baptist churches of "
        "the Kachin, Chin and Karen — American Baptist mission territory since Adoniram "
        "Judson in 1813 — the Roman Catholic church, and the Anglicans. Filing it on "
        "`christianity.protestant` or `.baptist` would assert a body the source does not "
        "name, and it would be wrong for the Catholic minority. lk2024.py's and kh2019.py's "
        "call. "
        "**Its geography is the ethnic minority uplands and it is the sharpest religious "
        "boundary in the country**: Chin is 85.4% Christian, Kayah 45.8%, Kachin 33.8%, "
        "against 1.1% in Magway and 1.1% in Nay Pyi Taw. Chin and Kachin are the two states "
        "where a world religion other than Buddhism holds a plurality.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 1,147,495 people, 2.3% of "
        "the enumerated population — **and this figure is the single most important thing to "
        "read carefully on this country.** It counts only the enumerated. The 1,090,000 "
        "people the census did not enumerate in Rakhine are, on DOP's own stated assumption, "
        "mainly Muslim; the report itself publishes a second national figure of **4.3%** on "
        "that basis. Neither number is wrong and they answer different questions. This map "
        "draws the 2.3% as `islam` and the non-enumerated separately as `unenumerated`, "
        "because DOP publishes the assumption at Union level only and this map draws states. "
        "Myanmar's enumerated Muslims are several unlike communities the census cannot "
        "separate: the Rohingya of northern Rakhine who WERE enumerated, the Kaman (a "
        "recognised Muslim group in Rakhine), the Panthay of Shan State, and the "
        "Indian-descended communities of Yangon and Mandalay.",
    "Hindu":
        "-> hinduism, with no branch. 252,763 people, 0.5%. The descendants of colonial-era "
        "migration from India, and its geography says so: **Bago 2.0% and Yangon 1.0%** are "
        "the plantation districts and the port, against near zero in the uplands.",
    "Animist":
        "-> indigenous.myanmar, a new national child on indigenous.philippine's precedent — "
        "a source giving its own country's traditions exactly one cell. 408,045 people, "
        "0.8%, and **93.9% of them are in Shan State**, which is 6.6% Animist against no "
        "other state above 1.9%. "
        "**Read it as a floor.** The box is exclusive of the Buddhist one, and nat "
        "propitiation is close to universal in Myanmar and accompanies Buddhism rather than "
        "replacing it — so this cell is the people for whom the traditional religion is the "
        "whole answer. §12's rule decided the node: the same word goes to `paganism` in "
        "Czechia and to `indigenous` in Sikkim, and the units it sits in are what tell you "
        "which — here it is concentrated in the upland minority states.",
    "Other religion":
        "-> other.mm. 82,825 people, 0.16%, one of the smallest residuals on the map because "
        "six categories already have boxes. Peaks by share in Kayah (1.20%) and Chin (1.11%) "
        "and by count in Shan (27,036) and Bago (12,687) — two different patterns, so §9r's "
        "rule says missing category rather than mixture and nothing is assigned. See the "
        "node's own note. Per §3.11.",
    "No religion":
        "-> unaffiliated. 30,844 people, **0.06% — the smallest no-religion share of any "
        "country on this map by a wide margin.** One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer. Its geography is "
        "Shan (24,767 of the 30,844, 80%), which is the same state that holds almost all the "
        "Animists — so some of it is likely to be traditional practice reported as 'no "
        "religion' rather than irreligion, which is Benin's `Aucune` warning again. The "
        "census does not say and nothing here resolves it.",
    "Estimated Non-enumerated population":
        "-> unenumerated, a NEW eighth member of §6.3a's grey family, and Anita's call "
        "(2026-09-07). 1,206,353 people: **Rakhine 1,090,000, Kayin 69,753, Kachin 46,600.** "
        "These are not a religion and not an answer — they are the state's own estimate of "
        "people it did not enumerate. DOP says why: *\"In Rakhine, an estimated 1.09 million "
        "people were not enumerated in the Census because they were not allowed to "
        "self-identify using a name not recognized by the Government.\"* "
        "**It is not mapped to `islam` even though the source suggests it**, because DOP "
        "applies that assumption at Union level only and this map draws states — assigning "
        "them would invent a magnitude at a resolution the source does not publish (§14 rule "
        "1). The assumption is quoted in `note_public` instead. "
        "**Every row is `modelled`** (§7): nobody was counted, and 1,090,000 is a round "
        "number in the source because it is an estimate. So `inferred dots: hidden` empties "
        "this node and shows the census exactly as the state published it, hole and all. "
        "See the node's note in branches.py for why not `unknown` and why drawing them at "
        "all is the point.",
}

MAP = {
    "Buddhist": "buddhism",
    "Christian": "christianity",
    "Islam": "islam",
    "Hindu": "hinduism",
    "Animist": "indigenous.myanmar",
    "Other religion": "other.mm",
    "No religion": "unaffiliated",
    "Estimated Non-enumerated population": "unenumerated",
}

# The one node on this country that is not a measurement (spec §7). countries.py reads it.
MODELLED = {"unenumerated"}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
