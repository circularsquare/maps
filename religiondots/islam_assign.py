"""Muslims drawn as Sunni by assignment, in countries where every estimate of the other branches is small.

WHY THIS EXISTS
---------------
Anita, 2026-10-04, after the sweep in sources/branches.md ("Muslim-majority countries, how close
each is to one branch") and its reports in sources/branches_2026-10-04/: *"ok yeah, lets assign.
and remainder can be best minority figure per coutnry ig yeah. we can do libya too, same
treatment."* No census or survey in these countries asks which branch a Muslim follows, so the
branch is ASSIGNED, the way spec §2.6a assigns Theravada to census Buddhists: the source's own
Muslim count is split into `islam.sunni` and a remainder that stays on bare `islam`, spread at
one share across the country because nothing places the minority below national. A large
unspecified share would be fine (spec §2.7a); these are small.

THE REMAINDER is the best minority figure for each country, as a share of its Muslims (Anita's
call): a government or community figure where one exists, else the World Religion Database's
non-Sunni share (ARDA rankings ADH_505 and ADH_512), which is ethnic ascription and national.
Every figure, with its source, is in REMAINDER's comment and in the country's note.

WHOLE UNITS. Where the minority's home is one of the drawn units and it is a large share there,
the unit is left whole on `islam` instead (Ghardaïa's M'zab Ibadis, Gorno-Badakhshan's Pamiri
Ismailis, Libya's Nafusa and Zuwara Ibadis, by Anita's "same treatment"), or given its own
remainder (Médenine, for Djerba). These say only that the unit is not assigned; no Ibadi or
Ismaili node is drawn (spec §14: the 2017 Libyan fatwa, the M'zab clashes).

NOT HERE, AND WHY. Saudi Arabia (Anita left it to the builder): the one regional figure, the US
State Department's 25-30% Shia in the Eastern Province, has no traceable origin, Najran's
circulating Ismaili figures exceed its census Saudis, and the 2015 mosque bombings make a Shia
placement a §14 question, so the om/sa ruling of 2026-09-15 stands. Oman, the UAE, Qatar,
Afghanistan, Azerbaijan, Pakistan and Syria have real minorities nothing places, and stay as
they are.

TIERS. A `measured` Muslim row becomes an `assigned` Sunni row with `roll=islam` plus a
`measured` remainder. `assigned` (spec §7e, built 2026-10-04 on Anita's "ok lets add it") is the
tier for "the religion was counted, the branch was assigned": it rolls back to the census's own
Muslim count when a reader hides inferred dots, as `derived` does (spec §7a-i-1), and is not
counted in the "filled in" share, because nothing was filled in. A `modelled` or `derived` row
keeps its tier on both halves: nobody counted Morocco's Muslims either, so they stay modelled. The
country's `assigned` field says what was assigned whatever the tier. Basis is unchanged: the
source's own count is only divided, never added to.

Applied by countries.py at load, so every consumer of counts() (scatter, rollup, regions,
not_drawn) sees the split, and by coverage.py, which adds `islam.sunni` to these countries.
"""

NODE = "islam.sunni"
PARENT = "islam"

# cc -> share of Muslims left on bare `islam`, spread evenly over the country's units.
REMAINDER = {
    "ma": 0.0003,   # Shia ~8-12k incl. foreign residents, Ahmadis ~750 (State Dept IRF 2023, leaders' claims)
    "dz": 0.0,      # outside Ghardaïa: Ahmadis <200; Mozabite Ibadis are Ghardaïa's (WHOLE)
    "tn": 0.001,    # WRD Shia 0.10%; Djerba's Ibadis are Médenine's (UNIT_REMAINDER)
    "ly": 0.002,    # WRD Shia 0.16%; the Ibadi districts are WHOLE
    "mr": 0.01,     # a Shia leader's 45,000 claim, 2010
    "sd": 0.002,    # Arab Barometer / Afrobarometer Shia answers ~0.2%
    "so": 0.017,    # WRD Shia 1.16% + schismatics 0.58%
    "dj": 0.0,      # WRD 0; Ministry of Islamic Affairs via State Dept "almost all Sunni"
    "km": 0.02,     # State Dept: Shia, Ahmadis and Christians together <2%
    "ml": 0.005,    # Afrobarometer Shia 0-0.5% across rounds
    "ne": 0.01,     # Ministry of the Interior via State Dept IRF 2019: Shia <1%
    "bf": 0.0,      # no figure anywhere; WRD 0
    "gn": 0.0,      # WRD 0.02%
    "gm": 0.02,     # Ahmadiyya community claim ~50,000 (State Dept 2023)
    "gw": 0.005,    # WRD Shia 0.32% + schismatics 0.22%
    "td": 0.0,      # WRD 0; Pew 2012's 21% "Shia" is label confusion (branches.md)
    "eg": 0.01,     # State Dept, citing scholars and NGOs: Shia ~1%
    "sl": 0.065,    # 2004 census Ahmadi 5.0% of population = ~6.5% of Muslims (tables inconsistent)
    "jo": 0.026,    # WRD Shia 2.14% + schismatics 0.42%
    "ps": 0.001,    # WRD Shia 0.11% (its 11.66% "schismatics" is unexplained, not used)
    "uz": 0.0035,   # Uzbek government via State Dept 2023: 122,000 Shia
    "kz": 0.011,    # 145,615 Azerbaijanis in the 2021 census, the largest likely-Shia group
    "kg": 0.01,     # government: Shia <1% of Muslims; ~1,000 Ahmadis
    "tm": 0.006,    # 40,312 Azerbaijanis, Persians and Kurds, 2022 census (State Dept places Shia among them)
    "tj": 0.012,    # outside GBAO: 60-160k Ismailis (State Dept 3-4% less GBAO's 227,916), midpoint
    "id": 0.015,    # Shia 200k (Kemenag 2016) to 2.5M (IJABI claim); Ahmadis 80k-500k
    "bd": 0.025,    # Pew 2012 Shia 2%; Ahmadis ~100k (community)
    "my": 0.015,    # Shia 1,500 (JAKIM 2013) to 300k (community); Ahmadis ~2,000
    "bn": 0.01,     # Pew 2009 Shia <1%; Shia and Ahmadiyya banned
    "mv": 0.0,      # citizenship is Sunni-only by law
    "ba": 0.01,     # Pew 2009 Shia <1%; Pew 2012 0%
    "xk": 0.03,     # Pew 2012: Shia 1%, something else 2%
    "al": 0.015,    # KAS 2024: other Shia orders ~1.4% of non-Bektashi Muslims
}

# (cc, unit) left whole on `islam`: the minority's home, too large a share there to assign.
WHOLE = {
    ("dz", "DZ47"): "Ghardaïa",            # M'zab Ibadis, perhaps 40-55% (Minahan 150-300k)
    ("tj", "TJ-GB"): "Gorno-Badakhshan",   # Pamiri Ismailis
    ("ly", "LY0209"): "Nalut",             # Ibadi Amazigh, 300-400k nationally (Tmazight Congress)
    ("ly", "LY0216"): "Jabal al Gharbi",
    ("ly", "LY0215"): "Zuwara",
}

# (cc, unit) -> its own remainder, where the minority's home is one unit but a small share of it.
UNIT_REMAINDER = {
    ("tn", "TN52"): 0.12,                  # Médenine: Djerba's ~60,000 Ibadis of ~534,000 Muslims
}

# What was assigned, in words: the country's `assigned` field, which the viewer shows as its own
# row under the title (spec §7e). It names the split and carries no share, Anita's shape for the
# label (2026-09-14: "we should also specify what we assigned"). countries.py sets it.
ASSIGNED = "Sunni, from national estimates"      # short: it counts toward the 80-word top text
# Where whole units are left out, the row says so instead; the note says from what.
ASSIGNED_EXCEPT = {
    "dz": "Sunni outside Ghardaïa",
    "ly": "Sunni outside Ibadi districts",
    "tj": "Sunni outside Gorno-Badakhshan",
}


def assigned_text(cc):
    return ASSIGNED_EXCEPT.get(cc, ASSIGNED)

# One sentence or two per country for note_public: the evidence, named. countries.py appends
# _note(cc). Panel voice: no em dashes, bold only on a figure inside a sentence.
WHY = {
    "ma": "The US State Department's 2023 religious freedom report, citing community leaders, "
          "puts Shia at several thousand Moroccans and 1,000 to 2,000 foreign residents, and "
          "Ahmadis at about 750.",
    "dz": "No count of Algeria's Ibadis exists. Estimates of the Mozabites run from 150,000 to "
          "300,000 (Minahan's Encyclopedia of Stateless Nations, 2016), nearly all of them in the "
          "M'zab valley.",
    "tn": "Tunisia's Ibadis are put at about 60,000 in press reports, mostly on Djerba, and the "
          "World Religion Database puts Shia at 0.1%.",
    "ly": "Libya's Ibadi Amazigh are put at 300,000 to 400,000 by the Libyan Tmazight Congress, "
          "4.5 to 6% of Libyans, living in the Nafusa mountains and Zuwara.",
    "mr": "The only figure for other branches is a Mauritanian Shia leader's claim of 45,000 "
          "followers in 2010, about 1%.",
    "sd": "Surveys find about 0.2% of Sudanese Muslims answering Shia, and the larger claims of "
          "Shia groups are not borne out by them.",
    "so": "The World Religion Database puts Shia and other branches at 1.7% of Somalia's Muslims, "
          "and no other figure exists.",
    "dj": "Djibouti's Ministry of Islamic Affairs, quoted by the US State Department, describes "
          "its Muslims as almost all Sunni, and the World Religion Database records no other "
          "branch.",
    "km": "The US State Department puts Shia, Ahmadis and Christians together at under 2% of "
          "Comorians, mostly on Anjouan.",
    "ml": "Between 0 and 0.5% of Muslims answered Shia across the Afrobarometer's rounds in Mali.",
    "ne": "Niger's Ministry of the Interior, quoted in the US State Department's 2019 report, "
          "puts Shia at under 1%.",
    "bf": "No figure exists for Burkina Faso's small Shia and Ahmadi communities, and the World "
          "Religion Database records none.",
    "gn": "The World Religion Database puts other branches at 0.02% of Guinea's Muslims, and no "
          "other figure exists.",
    "gm": "The Ahmadiyya community puts its Gambian members at about 50,000, about 2% of Muslims.",
    "gw": "The World Religion Database puts Shia and other branches at 0.5% of Guinea-Bissau's "
          "Muslims.",
    "td": "The World Religion Database records no branch but Sunni in Chad. Pew's 2012 survey "
          "found 21% answering Shia, which no other source supports.",
    "eg": "The US State Department, citing scholars and NGOs, puts Shia at about 1%, and surveys "
          "find none.",
    "sl": "Sierra Leone's 2004 census counted Ahmadis at 5% of the population, about 6.5% of "
          "Muslims, though its district tables do not agree with its national one. The 2015 "
          "census did not ask.",
    "jo": "The World Religion Database puts Shia and other branches at 2.6% of Jordan's Muslims, "
          "and Pew's 2012 survey found no Shia.",
    "ps": "The World Religion Database puts Shia at 0.1% of Palestinian Muslims, and Pew's 2012 "
          "survey found none.",
    "uz": "Uzbekistan's government, quoted by the US State Department, counts about 122,000 Shia, "
          "0.35% of Muslims.",
    "kz": "No figure exists for Kazakhstan's Shia. The 145,615 Azerbaijanis the 2021 census "
          "counted, about 1.1% of Muslims, are the largest group likely to be Shia.",
    "kg": "Kyrgyzstan's government puts Shia at under 1% of Muslims, with about 1,000 Ahmadis.",
    "tm": "The US State Department places Turkmenistan's small Shia communities among its "
          "Azerbaijanis, Persians and Kurds, 40,312 people in the 2022 census, about 0.6% of "
          "Muslims.",
    "tj": "Ismailis are 3 to 4% of Tajikistan's people (US State Department), most of them Pamiris "
          "in Gorno-Badakhshan.",
    "id": "Estimates of Indonesia's Shia run from about 200,000 (a 2016 Ministry of Religious "
          "Affairs study) to the 2.5 million a Shia organisation claims, and of Ahmadis from "
          "80,000 to 500,000.",
    "bd": "Pew's 2012 survey found 2% of Bangladesh's Muslims answering Shia, and Ahmadis put "
          "their own number at about 100,000.",
    "my": "Shia teaching is banned in Malaysia. Estimates of its followers run from 1,500 (JAKIM, "
          "2013) to 300,000 (community figures).",
    "bn": "Shia and Ahmadi teaching are banned in Brunei, and Pew's 2009 compilation puts Shia at "
          "under 1%.",
    "mv": "Maldivian citizenship is open only to Sunni Muslims by law.",
    "ba": "Pew's 2012 survey found no Shia among Bosnia's Muslims, and Pew's 2009 compilation puts "
          "them at under 1%.",
    "xk": "Pew's 2012 survey found 1% of Kosovo's Muslims answering Shia and 2% something else, "
          "mostly Bektashi.",
    "al": "Bektashis are drawn apart, from the census's own answer. A 2024 Konrad Adenauer "
          "Stiftung survey found about 1.4% of Albania's other Muslims naming another Shia order.",
}

# Extra sentence for countries with a whole or separately-sized unit.
UNIT_NOTE = {
    "dz": "Ghardaïa province, home of the M'zab's Ibadis, is left whole as Muslim with no branch, "
          "because they may be close to half its Muslims.",
    "tn": "In Médenine, which includes Djerba, **12%** are left with no branch.",
    "ly": "Nalut, Jabal al Gharbi and Zuwara districts are left whole as Muslim with no branch.",
    "tj": "Gorno-Badakhshan is left whole as Muslim with no branch; elsewhere the share left "
          "allows for Ismailis living outside it.",
}


def _pct(r):
    p = r * 100
    if p == 0:
        return "0"
    return f"{p:.2f}".rstrip("0").rstrip(".") + "%"


def note(cc):
    """The paragraph countries.py appends to note_public."""
    r = REMAINDER[cc]
    if cc in UNIT_NOTE:
        tail = (f"Elsewhere **{_pct(r)}** of Muslims are left with no branch, spread evenly."
                if r else "Elsewhere every Muslim is drawn as Sunni.")
    else:
        tail = (f"**{_pct(r)}** of Muslims are left with no branch, spread evenly, for the "
                "other branches." if r else "No Muslims are left without a branch.")
    # The bold opening sentence is the viewer's paragraph break (index.html md()), not bold type.
    parts = ["**Muslims are drawn as Sunni by assignment.** No source here asks which branch a "
             "Muslim follows, so the branch comes from the estimates below, and the share they "
             "give to other branches is left with no branch.", WHY[cc]]
    if cc in UNIT_NOTE:
        parts.append(UNIT_NOTE[cc])
    parts.append(tail)
    return " ".join(parts)


def apply(cc, df):
    """Split every bare `islam` row of `df` into `islam.sunni` and the remainder
    (branch_assign.split, which buddhism_assign.py shares)."""
    if cc not in REMAINDER:
        return df
    import branch_assign
    return branch_assign.split(
        cc, df, PARENT, NODE, REMAINDER[cc],
        unit_share={u: r for (c, u), r in UNIT_REMAINDER.items() if c == cc},
        whole={u for (c, u) in WHOLE if c == cc})


def check():
    """Static checks on the tables, run by countries.py at load."""
    for k in (WHY,):
        missing = set(REMAINDER) - set(k)
        if missing:
            raise SystemExit(f"islam_assign: no WHY sentence for {sorted(missing)}")
    for (cc, _u) in list(WHOLE) + list(UNIT_REMAINDER):
        if cc not in REMAINDER:
            raise SystemExit(f"islam_assign: unit exception for {cc}, which has no REMAINDER")
        if cc not in UNIT_NOTE:
            raise SystemExit(f"islam_assign: {cc} has a unit exception and no UNIT_NOTE")
    for cc, r in REMAINDER.items():
        if not 0.0 <= r < 0.2:
            raise SystemExit(f"islam_assign: {cc} remainder {r} out of range")
        if "—" in WHY[cc] or "—" in UNIT_NOTE.get(cc, ""):
            raise SystemExit(f"islam_assign: {cc} note has an em dash")
    for cc in {c for c, _u in WHOLE}:
        if cc not in ASSIGNED_EXCEPT:
            raise SystemExit(f"islam_assign: {cc} leaves units whole and its `assigned` text "
                             "does not say so (ASSIGNED_EXCEPT)")
