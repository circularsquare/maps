"""Iran census 1395 (November 2016) religion (Statistical Centre of Iran, Statistical Yearbook 1395,
Table 3-18) -> religiondots taxonomy.

Six columns at province, 79,926,270 people, 31 units, about 2.6 million each. sources/ir.md is the
write-up. The columns are Muslim, Zoroastrian, Christian, Jewish (`کلیمی`), other (`سایر`) and not
stated (`اظهارنشده`). The 1395 form was not found; *Amar* no. 21's Table 1 gives the 1390 and 1385
answers as Muslim, Christian (Assyrian or Chaldean, Armenian, other Christian), Jewish, Zoroastrian
and other, and the 1395 yearbook prints Christians as one column.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "اظهارنشده":
        "Not stated. 124,572 people, 0.156%. A non-answer, off the tree per spec §3.5 and in "
        "`gap`. It leans nowhere the table can show: 17.1% of it is in Khuzestan (0.45% of the "
        "province), 16.9% in Tehran and 12.8% in Razavi Khorasan. Ask 023 ruled that 1390's "
        "merged `سایر و اظهار نشده` (other and not stated) column stays on a coloured node rather "
        "than in `gap`; the 1395 yearbook prints the two apart, so not stated is handled as every "
        "other census's not stated and `سایر` keeps its node (below).",
}

REVIEW = {
    "مسلمان":
        "-> islam, no branch. 79,598,054 people, 99.589%. The table has one Muslim column; it "
        "does not split Shia from Sunni, and Iran's Sunni minority is concentrated in some "
        "provinces, so `islam.shia` would draw an inference onto exactly the provinces where it "
        "is wrong (spec §2.7a).",
    "مسیحی":
        "-> christianity, the bare branch. 130,158 people, 0.163%. One column in 1395. The 1390 "
        "table (Amar no. 21, Table 3) split Christians into `آشوری یا کلدانی` (Assyrian or "
        "Chaldean) and `مسیحی`, and the first was a flat 0.04-0.18% in every province. Across "
        "provinces the 1395 Christian share correlates +0.91 with that flat column and +0.44 with "
        "1390's `مسیحی` (sources/ir.py check 7), so the label does not identify Assyrians and "
        "neither column could put anyone on `christianity.oriental`. **The 1395 share is 0.08% "
        "to 0.16% in 28 provinces** and higher only in Tehran (0.33%, a third of all Christians), "
        "West Azerbaijan (0.23%) and Isfahan (0.17%), where the Armenian and Assyrian churches "
        "are. The census does not say who the low, even share elsewhere is; it is drawn as "
        "printed.",
    "زرتشتی":
        "-> zoroastrianism. 23,109 people, 0.029%. Yazd has the highest share (0.316%, 3,600 "
        "people) and Tehran the most people (8,579, 37.1%); Kerman is third (1,280). Same leader "
        "in 1390.",
    "کلیمی":
        "-> judaism. `Kalimi` is the census's word for Jewish. 9,826 people, 0.012%: Tehran "
        "5,067 (51.6%), Fars 2,816 (28.7%, the highest share at 0.058%), Isfahan 1,007. No "
        "branch: the census asks nothing finer.",
    "سایر":
        "-> other.ir. 40,551 people, 0.051%. SCI does not say what `سایر` holds. The census has "
        "no Baháʼí answer, so a Baháʼí who answered other is in this node and one who declined "
        "is in not stated; Anita's ruling on ask 023 was that this is the usual case for a small "
        "religion with no answer of its own and does not hold the build. Largest in Tehran "
        "(9,568), Isfahan (6,014) and Fars (4,649).",
}

MAP = {
    "مسلمان": "islam",
    "زرتشتی": "zoroastrianism",
    "مسیحی": "christianity",
    "کلیمی": "judaism",
    "سایر": "other.ir",
}

# spec §7a-i-1: every row is measured at the node it is drawn on.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return "".join(str(cat).replace("‌", "").split())


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
