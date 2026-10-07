"""Iran census 1395 (November 2016) religion (Statistical Centre of Iran, Statistical Yearbook 1395,
Table 3-18) -> religiondots taxonomy.

Six columns at province, 79,926,270 people, 31 units, about 2.6 million each. sources/ir.md is the
write-up. The columns are Muslim, Zoroastrian, Christian, Jewish (`کلیمی`), other (`سایر`) and not
stated (`اظهارنشده`). The 1395 form was not found; *Amar* no. 21's Table 1 gives the 1390 and 1385
answers as Muslim, Christian (Assyrian or Chaldean, Armenian, other Christian), Jewish, Zoroastrian
and other, and the 1395 yearbook prints Christians as one column.

The Muslim column is split by ir_split.py (2026-09-15, on Anita's ruling recorded 2026-09-16): a
Sunni row from Masaili's province estimates and a Shia row for the rest, both `derived` (the rest
on Shia by Anita's ruling of 2026-09-15, applied 2026-10-03). The two Sunni labels and the Shia
label below are that script's, not the census's.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

# ir_split.py writes these; they must stay identical to its LABEL, LABEL_LUMP and LABEL_SHIA.
LABEL_SUNNI = "Muslim: Sunni (Masaili 2023)"
LABEL_LUMP = "Muslim: Sunni, Tehran and the central provinces (Masaili 2023)"
LABEL_SHIA = "Muslim: Shia, census Muslims less Masaili's Sunnis"

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
        "-> islam, no branch, but no row reaches the map under this label: ir_split.py replaces "
        "the whole column with Masaili's 7,608,500 Sunnis and 71,989,554 Shia (79,598,054 people, "
        "99.589%), and this entry is the roll-up target in COLUMNS. REVERSED 2026-10-03 ON ANITA'S "
        "RULING OF 2026-09-15 (ask/RULINGS.md, 'ok lets record (a)'): the first build "
        "(d743fc47-irsect) left the remainder here, arguing that a Shia layer would be census "
        "Muslims minus an estimate, a number nobody published, and would carry Masaili's low lean "
        "onto Shia in the Kurdish provinces; the ma ir review (d743fc47-rev14) kept that outcome. "
        "Anita ruled the remainder onto `islam.shia` in every province, accepting those costs "
        "(LABEL_SHIA below).",
    LABEL_SHIA:
        "-> islam.shia. 71,989,554 people, every census Muslim in each province less Masaili's "
        "Sunnis there, and all of them in the 15 provinces his table gives no row (Khuzestan, "
        "Isfahan, East Azerbaijan, Ilam, Lorestan and ten more). On Anita's ruling of 2026-09-15: "
        "Masaili prints no Shia figure but says Twelver Shia are the majority, and the remainder "
        "goes on `islam.shia` rather than plain `islam`. THE ACCEPTED COSTS, as ruled: (1) his "
        "Kurdish count leans low by his own framing (Soltani 2015, a Sunni author, puts the Kurds "
        "about two million higher), and that difference is drawn as Shia in Kurdistan, West "
        "Azerbaijan and Kermanshah; (2) Kermanshah's Yarsan (Ahl-e Haqq), whom the paper counts "
        "among the province's 'Shia Kurds' and who answered Muslim on the census, are drawn on "
        "`islam.shia`, the Twelver node. Not decided: any later correction for either. `derived`, "
        "basis estimate, `parent_column=مسلمان`, so it rolls back to `islam` through COLUMNS like "
        "the Sunni rows; with inferred dots hidden Iran draws every Muslim on `islam`. Iran now "
        "draws `islam.shia`, so a national Shia estimate row for Iran would restate a drawn node "
        "(spec §15.3).",
    LABEL_SUNNI:
        "-> islam.sunni. 7,108,500 people in 14 provinces, Masaili's printed counts (Atlas-e "
        "Towsifi-ye Ahl-e Sonnat-e Iran, 2023, p. 59; Haft Aseman 26(88), Table 1): Kurdistan 82% "
        "of the population, Sistan and Baluchestan 64, Golestan 38, West Azerbaijan 35, Hormozgan "
        "35, Kermanshah 26, South Khorasan 15, North Khorasan 10, Gilan 7, Razavi Khorasan 5, "
        "Bushehr 5, Fars 4, Kerman 2, Ardabil 2. `derived`, basis estimate: one researcher's "
        "judgement from library and field work, applied inside a column the census counted at the "
        "same province, so it rolls back to `islam` through COLUMNS. The printed count is drawn "
        "rather than the integer percentage times census Muslims; the two differ by at most 2,469 "
        "(Kurdistan). Uniform inside a province: the book names Sunni-majority counties (Bastak, "
        "Jask, Qeshm and others in Hormozgan; the Avroman counties in Kermanshah) but prints no "
        "county figure, so Sunni and Shia dots are placed by the same population weight. No "
        "school (Hanafi, Shafi'i) is named, so nothing goes below `islam.sunni`.",
    LABEL_LUMP:
        "-> islam.sunni. Masaili's row 15, 500,000 Sunnis of 'Tehran and the central provinces of "
        "Iran', names no province. SPREAD OVER TEHRAN AND ALBORZ ONLY, by census Muslims (both "
        "3.1%; ir_split.py prints the two counts): Tehran is the one province named; Alborz was "
        "part of Tehran province until 2010 and Karaj is in the same city region; Soltani (2015) "
        "counts Alborz with Tehran. The other provinces a reader could call central (Qom, "
        "Markazi, Isfahan, Qazvin, Semnan, Yazd; 10.9M Muslims) get none: nothing read names a "
        "Sunni community in any of them, and spreading over all eight would put about 95,000 of "
        "the 500,000 in Isfahan. Changing the set is one dict in ir_split.py (LUMP_PROVINCES).",
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
    # ir_split.py's labels: Masaili's Sunnis, taken out of `مسلمان` (REVIEW).
    LABEL_SUNNI: "islam.sunni",
    LABEL_LUMP: "islam.sunni",
    LABEL_SHIA: "islam.shia",
}

# spec §7a-i-1: for a derived row, the node its SOURCE COLUMN names at the drawn unit. Every census
# column is measured at the province it is drawn on; the only derived rows are ir_split.py's Sunni
# and Shia rows, which carry `parent_column=مسلمان` and roll back to the `islam` the census counted
# there.
COLUMNS = {
    "مسلمان": "islam",
}


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
