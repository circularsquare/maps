"""
SURS Popis 2002 religion classification -> religiondots taxonomy.

**Branch-level mapping, and a shallow one: five religion answers on 192 municipalities.**
Slovenia's census asked belief first and affiliation second, so the drawn table is a ladder
rather than a list -- a believer belonging to a named religion, a believer belonging to none,
a non-believer, a refusal, and an unknown -- and only the first rung is subdivided.

    57.82%  katoliška                                 -> christianity.catholic
    15.68%  Ni želel odgovoriti                       -> EXCLUDED (declined the question)
    10.15%  Ni vernik, ateist                         -> unaffiliated
     7.08%  Neznano                                   -> EXCLUDED (never established)
     3.50%  Je vernik, ne pripada nobeni veroizpovedi -> unchurched
     2.42%  islamska                                  -> islam
     2.34%  pravoslavna                               -> christianity.orthodox.canonical
     0.82%  evangeličanska in druge protestantske     -> christianity.protestant
     0.20%  druge veroizpovedi                        -> other.si   (a NEW node)

**THE SAME CENSUS PUBLISHED FOURTEEN CATEGORIES FOR THE COUNTRY AND FIVE PER MUNICIPALITY**,
which is spec §3.9's trade in its usual direction. sources/si.py keeps the fourteen in
si.csv at `geo_level=country_detail` because they decompose both catch-alls above to the
person, and every one of them is EXCLUDED here: they are the same 1,964,036 people a second
time, so resolving them would double the country. They are not a finer tier waiting to be
drawn. Nothing published puts a Slovenian Jew or a Slovenian Buddhist in a municipality.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Veroizpoved - SKUPAJ":
        "the unit's own population total, not a category.",
    "Opredeljeni po veroizpovedi - skupaj":
        "the subtotal of the five religion answers, 1,248,988 people. Drawing it alongside "
        "its own five parts would count 63.6% of Slovenia twice.",
    "Ni želel odgovoriti":
        "**307,973 people, 15.68%, who did not want to answer, and the single most "
        "important fact about the Slovenian map.** The 1991 census recorded 4.25% in the "
        "same cell, so the refusal nearly quadrupled in eleven years while the population "
        "barely moved. Not a religion and not a report of no religion, both of which are "
        "separate answers here. Article 41 of the 1991 constitution makes declaring one's "
        "religion voluntary and the census form said so, which is a fact about the "
        "question rather than about the country. Its geography is not uniform either: it "
        "runs from 2.0% in Rogašovci to 25.6% in Miklavž na Dravskem polju, and "
        "countries.py's note records which way that leans.",
    "Neznano":
        "139,097 people, 7.08%, for whom the answer was never established. SURS keeps this "
        "apart from the refusal above and so does this file, hr2021.py's and ba2013.py's "
        "call for the same pair. **Its cause is on the questionnaire**: questions 29 and 30 "
        "(ethnicity and religion) had to be answered by the person themselves, aged 14 or "
        "over, and no household member could answer them by proxy. An enumerator who found "
        "nobody in left form P-3/NV and a prepaid envelope, and 7.08% of the country never "
        "sent it back. It was 14.56% in 1991, so this one HALVED while the refusal "
        "quadrupled, which is two opposite movements inside one 22.8% hole.",

    # ---- the fourteen-category national table (geo_level=country_detail). These are the
    # same people as the rows above, published once more at a coarser geography and a finer
    # classification, and sources/si.py keeps them as a check level. Excluding them is not a
    # judgement about the categories; it is the only way not to count Slovenia twice.
    "Katoliška":
        "national-only check row, and it is `katoliška` above with a capital K. Not drawn.",
    "Evangeličanska":
        "national-only check row: 14,736 people, the Evangelical Lutheran Church of "
        "Slovenia. **This is the evidence for the mapping of the drawn compound** — see "
        "REVIEW — and it exists for the country and for nowhere in it. Not drawn.",
    "Druge protestantske":
        "national-only check row: 1,399 people, the other Protestant bodies, 8.7% of the "
        "compound the map draws. Not drawn.",
    "Pravoslavna":
        "national-only check row, capitalised. Not drawn.",
    "Druge krščanske":
        "national-only check row: 1,877 other Christians, the largest part of `druge "
        "veroizpovedi`. Not drawn.",
    "Islamska":
        "national-only check row, capitalised. Not drawn.",
    "Judovska":
        "national-only check row: **99 people, the Jewish community of Slovenia at the 2002 "
        "census**, against 199 in 1991. It has no municipality and cannot be drawn; it is "
        "inside `other.si`.",
    "Orientalske":
        "national-only check row: 1,026 people in `Oriental religions`, which is Buddhism, "
        "Hinduism and their neighbours in one cell and would go to `other.si` in any case, "
        "for hr2021.py's reason. Not drawn.",
    "Druge veroizpovedi":
        "national-only check row, capitalised: 558 people. The municipality cell of the "
        "same name is a wider thing — it is this plus four other national rows — which is "
        "why the two are not interchangeable and why both appear in this table.",
    "Agnostiki":
        "national-only check row: 271 people, 0.014%. The municipality table folds them "
        "into `druge veroizpovedi` rather than into the atheist cell, which is SURS's own "
        "statement that the two are not the same answer, and it is part of why the atheist "
        "cell resolves the way REVIEW says it does. There is no 1991 figure. Not drawn.",
    "Je vernik, vendar ne pripada nobeni veroizpovedi":
        "national-only check row, and the drawn cell's label with `vendar` in it. 1991 read "
        "3,929 against 2002's 68,714, a seventeenfold rise, which is the largest "
        "proportional movement in the whole table. Not drawn.",
}

REVIEW = {
    "Ni vernik, ateist":
        "-> unaffiliated, NOT `secular`, and it is the most arguable call in the file at "
        "199,264 people and 10.15% of the country. **The rule is about the label's shape "
        "and it is already written down three times.** ba2013.py routes Bosnia's bare "
        "`Ateist` to `secular` and says why: it has *'none of the \"not a believer\" gloss "
        "that sends Serbia's and North Macedonia's equivalents to unaffiliated instead'*. "
        "mk2021.py's `Не е верник (атеист)` and hr2021.py's `Nisu vjernici i ateisti` both "
        "carry that gloss and both go to `unaffiliated`; Slovenia's `Ni vernik, ateist` is "
        "the same construction in the same census family, and Croatia's is a border away. "
        "Every `secular` precedent is a BARE atheist label instead — Czechia's `ateismus`, "
        "Albania's `Atheists`, Brazil's `Sem religião - Ateu`, Spain's `Ateo/a`. **The "
        "strongest argument the other way is Albania's answer SET**, which like Slovenia's "
        "offers a believer-without-a-denomination box beside the atheist one and therefore "
        "splits the irreligious on a stated position about God, which is exactly what "
        "branches.py's `secular` is for; al2023.py decided that structure the other way. "
        "Slovenia's own form is some evidence against it, because SURS counted `Agnostiki` "
        "separately and did not put them here. Overturnable, and this is where.",
    "katoliška":
        "-> christianity.catholic, the PARENT, and not christianity.catholic.latin. The "
        "category is `Catholic` with no rite; hr2021.py and cz2021.py file their bare "
        "equivalents the same way. 1,135,626 people, 57.8% of the country and 75.0% of "
        "what the map draws, so this one node is most of Slovenia.",
    "evangeličanska in druge protestantske":
        "-> christianity.protestant, the 'named no body' node, and NOT christianity."
        "lutheran. The cell is a compound and the source says which way it splits: the "
        "national table has `Evangeličanska` 14,736 and `Druge protestantske` 1,399, so "
        "91.3% of it is the Evangelical Lutheran Church of Slovenia and the drawn cell is "
        "still not a Lutheran cell. Sending the whole thing to `christianity.lutheran` "
        "would assert that 1,399 people who told the census they were something else were "
        "Lutherans; `christianity.protestant` asserts only that all 16,135 are Protestants, "
        "which the label says. Nothing is allocated between the two (spec §14.4). "
        "**Its geography is one corner of the country**: Hodoš is 84.9% of what the map "
        "draws there, Gornji Petrovci 67.3% and Puconci 59.8%, all of them Prekmurje on "
        "the Hungarian border, which is the Reformation's map surviving three empires.",
    "pravoslavna":
        "-> christianity.orthodox.canonical, which is hr2021.py's, rs2022.py's and "
        "lt2021.py's call for a bare `Orthodox`. Slovenia's Orthodox are overwhelmingly "
        "the Serbian Orthodox Metropolitanate of Zagreb and Ljubljana, canonical "
        "throughout. **The wrinkle is the Macedonian Orthodox Church**, which was in "
        "schism from 1967 until 2022 and has had communities in Slovenia throughout, so "
        "some unknown part of the 45,908 was not in communion at the time. The census "
        "names no jurisdiction anywhere and SURS's municipality-level ethnicity table "
        "(`05W1002S.px`) does not name nationalities either, only whether a person "
        "declared one, so the share cannot be bounded from this census. Filing at the "
        "parent `christianity.orthodox` instead would be defensible and would make "
        "Slovenia the only country in the region doing it, which is the reason it does "
        "not.",
    "islamska":
        "-> islam, the parent, and not islam.sunni. Slovenia's Muslims are Bosniak and "
        "Hanafi almost to a person and the Islamic Community of Slovenia is Sunni, but the "
        "census names no branch and this file does not add one (§2.6). 47,488 people, "
        "2.42%, and the fastest-growing category between 1991 and 2002: 29,361 to 47,488, "
        "which is the Bosnian war inside a census table.",
    "druge veroizpovedi":
        "-> other.si, A NODE ADDED FOR THIS SOURCE. 3,831 people, 0.20%, and unusually its "
        "composition is published — for the country and not for any place in it. See the "
        "node's own text in branches.py. It contains 271 agnostics, which makes it the one "
        "`other.<source>` cell on the map holding a stated position rather than a religion; "
        "that is SURS's arrangement and not this file's, and splitting them out nationally "
        "while the geography cannot follow would be a division the map could not draw.",
    "Je vernik, ne pripada nobeni veroizpovedi":
        "-> unchurched, the node Czechia's `věřící nehlásící se k žádné církvi ani "
        "náboženské společnosti` created, and this is the same sentence in Slovene. 68,714 "
        "people, 3.50%, against 3,929 in 1991. Not `unaffiliated`, which is the answer "
        "above it, and not `secular`: the respondent reported belief and no institution, "
        "which is exactly what the node is for.",
}

MAP = {
    "katoliška": "christianity.catholic",
    "evangeličanska in druge protestantske": "christianity.protestant",
    "pravoslavna": "christianity.orthodox.canonical",
    "islamska": "islam",
    "druge veroizpovedi": "other.si",
    "Je vernik, ne pripada nobeni veroizpovedi": "unchurched",
    "Ni vernik, ateist": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
