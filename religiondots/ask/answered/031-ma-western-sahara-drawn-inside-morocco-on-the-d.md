# 031 — ma: Western Sahara drawn inside Morocco on the de facto rule

Summary: Morocco's census counts the Moroccan-administered part of Western Sahara (about 670,000 people). Built as part of Morocco, placed only west of the berm; the wash is untouched. Keep, give it its own entry, or leave it undrawn?

*Filed 2026-09-15 by session `d743fc47-ma`. Anita's call; nothing is waiting on it.*

## What I did

Morocco is drawn with the part of Western Sahara it administers, as four of its 73 units: the
people are HCP's 2024 census count, and dots are placed only west of the berm (Natural Earth's
`W. Sahara` feature B19, "Admin. by Morocco; Claimed by Western Sahara"). The Polisario-held part
east of the berm (B28) gets no dots. `country_shapes.py` is untouched, so Morocco's wash stays
Natural Earth's Morocco and the southern dots sit outside it.

## What it costs to reverse

Taking them out: drop four rows from `data/geo/ma/ma_lookup.csv` in `sources/ma.py` and re-scatter
`ma`, about 10 minutes plus the build tail. Giving them their own entry, as the settlers (`xs`) have:
an afternoon, since the survey's two southern regions would need their own shares.

## Why it is yours rather than mine

AGENT_BRIEF §3, first bar: whether a territory is drawn, and as whose. `ask/RULINGS.md` records the
de facto rule for Arunachal (spec §14.18) with "Not decided: disputed areas outside China's claims",
and Ukraine's occupied oblasts went the other way (ask 016), so neither settles Western Sahara.

## The detail

- **People.** Laâyoune-Sakia El Hamra 451,028 and Dakhla-Oued Ed-Dahab 219,965 (RGPH 2024), plus
  the Assa-Zag commune of Al Mahbass, 19,139 people in 170 households on the berm: 690,132 in the
  four units, 1.87% of the count. Tarfaya town and Akhfennir, north of 27°40'N, are in the Laâyoune
  unit because HCP counts them in Tarfaya province.
- **The census counts the berm garrisons in communes named for places east of it**: Tifariti 5,728
  people in 38 households, Amgala 4,102 in 83, and Lagouira, Aghouinite and Zoug (starred in the
  workbook). With B28 cut out, their dots go to the unit's populated hexes west of the berm.
- **The survey reaches it.** Arab Barometer samples both southern regions in every wave from V
  (Laâyoune-Sakia El Hamra 309 respondents over V-VIII, Eddakhla-Oued Eddahab 220), and its
  technical reports give the frame as Morocco's 2014 census.
- **Three options.** (a) keep as built; (b) an entry of its own with no territory, like `xs`,
  drawn from the same survey shares; (c) leave the four units off, 690,132 people undrawn.
- **The wash.** Natural Earth's `admin_0_countries` has Western Sahara as its own country, so
  Auto over Laâyoune picks Morocco only by the dot tally. Extending the wash is a
  `country_shapes.py` change that paints Western Sahara as Morocco; not done.


---

## Ruled 2026-09-15 by Anita: keep it in Morocco

*"keep in morocco fine"*

Option (a): the four Western Sahara units stay part of Morocco as built, placed only west of the
berm, with `country_shapes.py` untouched.
