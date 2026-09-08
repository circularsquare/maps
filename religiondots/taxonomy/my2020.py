"""Malaysia 2020 census religion (DOSM) -> religiondots taxonomy.

Seven categories at administrative district, 32.45 million people over 160 units.
`sources/my.md` and `sources.md` §11s are the source write-ups.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**ONE NEW NODE, `other.my`, AND FOUR CELLS THAT RESOLVE TO ROOTS.** A form offering
Islam, Christianity, Buddhism and Hinduism is asking at the level of world religions
and the tree has had those since its first source; §2 forbids adding at ingest the
branch the source declines to name. Malaysia's value here is not taxonomic. It is
that a country running from **97.3% Muslim in Terengganu to 50.1% Christian in
Sarawak** arrives sorted into 160 units, and that no other source on this map holds
that range inside one border.

**THE TWO HARD CALLS ARE BOTH IN THE GREY FAMILY**, and neither is about religion in
the ordinary sense:

  * `Tiada Agama` -> `unaffiliated` reads as secularity and in Malaysia it very
    largely is not. See the REVIEW note; it is the most misleading single cell in
    this source and `note_public` in `countries.py` carries the warning to the reader.
  * `Tidak Diketahui` -> `unknown` is 97% male and almost certainly non-citizen
    labour. Not distributed, per spec §3.5.
"""

EXCLUDED = {}

REVIEW = {
    "Islam":
        "-> islam, with no branch, because the census names none. 20,610,060 people, "
        "63.5% — the state religion, and constitutionally inseparable from Malay "
        "ethnicity: Article 160 defines a Malay as, among other things, a person who "
        "professes Islam, so the Muslim share and the Malay share are legally rather "
        "than merely statistically entangled. Malaysian Islam is Sunni of the "
        "**Shafi'i** school, which the state enforces — the religious authorities of "
        "several states have banned Shia practice outright — so `islam.sunni` would "
        "be very nearly true and is still an inference the form does not license. The "
        "same call bd2011.py makes for Bangladesh and et2007.py for Ethiopia. **The "
        "loss worth naming is Shia and Ahmadi**: both exist, both are small, both are "
        "legally suppressed, and a census that offers one Islam cell cannot show "
        "either — where pk2017.py has `islam.ahmadiyya` only because Pakistan's form "
        "counts Ahmadis separately in order to exclude them.",
    "Christianity":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots'. "
        "2,941,049 people, 9.1%, and **the single strongest reason to draw this "
        "country**: it is not spread thin, it is the majority religion of Sarawak "
        "(50.1%) and a quarter of Sabah (24.7%), against 0.3% in Terengganu. The "
        "district peaks are extraordinary — Tebedu 93.1%, Kapit 89.6%, Lubok Antu "
        "88.0%, Belaga 86.2%, Kanowit 84.8%, and Tambunan in Sabah 79.8%. No branch, "
        "because DOSM gives none, and the composition is genuinely mixed and worth "
        "recording as a loss: the Roman Catholic church is large in both Borneo "
        "states, the Anglicans came with Brooke rule in Sarawak, the **Sidang Injil "
        "Borneo** (Borneo Evangelical Church) is the largest Protestant body in the "
        "interior and appears on no other map here, and the Basel Christian Church "
        "carries the Hakka communities of Sabah. All of it is inside this one figure.",
    "Buddhism":
        "-> buddhism, no branch. 6,066,784 people, 18.7%, and **the largest Buddhist "
        "share of any country on this map outside East and Southeast Asia's Buddhist "
        "majorities**. It is almost entirely the Chinese community — Timur Laut "
        "(George Town) 52.5%, Kampar 45.8%, Seberang Perai Tengah 34.0%, Kinta 33.5%, "
        "Johor Bahru 33.1% — so the geography is the tin-and-trade geography of the "
        "west coast and the Perak valley. **The branch loss is real and is not a "
        "simple one.** Malaysian Buddhism is predominantly Mahayana of the Chinese "
        "tradition, with a substantial Theravada presence (the Thai communities of "
        "Kelantan and Kedah, the Sri Lankan and Burmese temples of Penang and Kuala "
        "Lumpur) and a modernist English-speaking layer that belongs comfortably to "
        "neither. More to the point, the boundary between 'Buddhist' and the Taoist "
        "and folk practice in `other.my` is one respondents draw differently from one "
        "another, so the Buddhist cell and the residual are not cleanly separable "
        "even in principle.",
    "Hinduism":
        "-> hinduism, no branch. 1,969,471 people, 6.1% — the descendants of Tamil "
        "labour brought to the rubber estates and the railways under British rule, "
        "and the geography still shows the estates: Bagan Datuk 21.5%, Port Dickson "
        "17.3%, Klang 16.9%, Seberang Perai Selatan 15.7%, Seremban 14.3%, Kulim "
        "13.8%. **`hinduism.tamil` exists on this tree** (in2011.py) and is not used "
        "here, because DOSM does not ask which Hinduism and the inference — however "
        "safe, and Malaysian Hindus are around 90% Tamil — would be ours rather than "
        "the source's. Same rule as everywhere else in this file.",
    "Others":
        "-> other.my, a new node. 285,152 people, 0.88%. The footnote names six "
        "traditions inside it and its geography resolves into two distinct missing "
        "categories — Chinese folk practice in the Chinese-majority west-coast "
        "districts, Orang Asli and interior Iban religion in the peninsular highlands "
        "and Sarawak. The node note in branches.py has the argument and the numbers; "
        "the short version is that **a tradition this map draws for China, Vietnam and "
        "Singapore cannot be brought out for Malaysia at all**, because DOSM's form "
        "gives temple practice no box. Per source, per spec §3.11.",
    "No Religion":
        "-> unaffiliated, AND IT IS THE MOST MISLEADING CELL IN THIS SOURCE. 271,799 "
        "people, 0.84%. Read as printed it says Malaysia is among the most religious "
        "countries on this map, which is true; read carelessly it says the places "
        "where it concentrates are the secular ones, which is false. **The geography "
        "is the opposite of a secular geography.** It peaks at **35.94% in Kecil "
        "Lojing, Kelantan** — a small, overwhelmingly Temiar Orang Asli district — and "
        "then runs Rompin 13.4%, Selangau 12.5%, Pekan 10.3%, Maradong 9.0%, Pakan "
        "8.9%, Cameron Highlands 8.8%, Sarikei 8.5%, Lipis 8.5%, Batang Padang 6.9%, "
        "Hulu Perak 6.4%. Every one of those is an **Orang Asli or interior indigenous "
        "district**, and none of them is Kuala Lumpur (0.9%) or Penang. §9r's rule "
        "applies to the grey family as much as to a residual: a sharp geography means "
        "a missing category. **The near-certain reading is indigenous traditional "
        "practice recorded as an absence of religion** — either by respondents with no "
        "box that fits, or by enumerators who did not count it as a religion — which "
        "makes this cell and `other.my` two different treatments of the same people in "
        "different districts. **It is mapped as printed anyway**, because remapping it "
        "would assert a magnitude DOSM does not publish (§14 rule 1) and the cell "
        "genuinely does also contain the country's small secular population. The "
        "honest fix is the reader-facing one: `note_public` says this outright, and it "
        "is load-bearing rather than decorative.",
    "Unknown":
        "-> unknown, and NOT distributed (spec §3.5). 303,070 people, 0.93%. **It is "
        "97% male** — 67,664 men to 25 women in Perak, 34,437 to 1 in Kinta, 7,316 to "
        "0 in Manjung — which is not a sex ratio any non-response process produces. "
        "The census counts total population including roughly 2.7 million "
        "non-citizens, overwhelmingly male labour in plantations, construction and "
        "manufacturing, and the concentration fits: W.P. Kuala Lumpur 5.1% (100,882 "
        "people), Hilir Perak 5.3%, Subis 4.9%, Kinta 3.9%, the Penang industrial "
        "districts 3-4%. **The near-certain reading is workers enumerated for a "
        "headcount without the religion item being taken.** DOSM states none of this, "
        "so it stays a reading rather than a claim, and the cell stays its own node — "
        "spreading it would invent religion for the one group the census did not "
        "assign one, and would put it in exactly the districts where the workforce is.",
}

MAP = {
    "Islam": "islam",
    "Christianity": "christianity",
    "Buddhism": "buddhism",
    "Hinduism": "hinduism",
    "Others": "other.my",
    "No Religion": "unaffiliated",
    "Unknown": "unknown",
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
