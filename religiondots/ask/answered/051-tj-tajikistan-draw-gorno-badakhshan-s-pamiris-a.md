# 051 — tj: Tajikistan: draw Gorno-Badakhshan's Pamiris as Ismaili, or keep them on plain Islam?

Summary: Built with every Muslim in Tajikistan on plain Islam, Gorno-Badakhshan included. Drawing its Pamiri districts as Ismaili is a section 14 call: no source measures it, and the state has pressed the Pamiris hard since 2022.

*Filed 2026-10-03 by session `fafd1067-tj`. Anita's call; nothing is waiting on it.*

## What I did

Tajikistan is drawn at its five regions as an ethnicity model (`sources/tj.py`), and every Muslim in
it, the 227,916 people of Gorno-Badakhshan included, is on plain `islam`. No sect is drawn anywhere
in the country.

## What it costs to reverse

A new category for Gorno-Badakhshan's Ismaili districts in `sources/tj.py` and `taxonomy/tj2020.py`,
a district split of that one region (the 2020 census table on disk has its eight districts), and a
rescatter of `tj`: about an hour.

## Why it is yours rather than mine

§14: whether a group's safety is affected by publishing where they live. The Pamiris are a
religious minority (Nizari Ismaili) whose region the state put under heavy pressure after the May
2022 Khorog protests, and the brief that sent me here named it as the §14 part.

## The detail

- **Nothing measures it.** The 2020 census asked religion with one `ислам` box and published nothing
  (`sources/tj.md` §1-2). Its nationality card has no Pamiri code: Pamiris are counted as Tajiks. The
  Central Asia Barometer has an Ismaili code and in its one wave that asked the school (2017) recorded
  40 of Gorno-Badakhshan's 51 Muslim respondents as Sunni and 3 as Ismaili (`sources/tj.md` §4), which
  is not credible and is the only survey figure there is.
- **So drawing it would be ascription by district**, not a count: Khorog, Shughnon, Roshtqal'a,
  Ishkoshim and Rushon as Ismaili (about 154,000 people in 2020); Vanj and Darvoz mostly Sunni Tajiks
  and Murghob mostly Sunni Kyrgyz (about 73,000), from the general literature rather than any table.
- **The precedent goes the other way.** `taxonomy/cn2000.py` keeps China's Ismaili Tajiks of
  Taxkorgan on plain `islam`, because marking one community's sect forces the parallel claim for
  everyone around it; the om/sa ruling (2026-09-15) leaves an unplaced sect split undrawn. Here it
  would be placed (by district), which is the difference, and also the §14 worry.
- My recommendation is to leave it as built: the region is 2.4% of the country and its Muslims show
  as Muslims either way, and a map that marks where the Ismailis live adds little the region's name
  does not already say, at the cost of being the one source that draws them.
