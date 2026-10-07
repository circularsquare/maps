# 050 — sy: Syria drawn at one national mix; Druze, Alawites and Christians not placed (spec 14)

Summary: Syria built flat: Pew 2020's national mix in all 14 governorates, Druze and Alawites inside Islam. Placing Druze in Suwayda or Christians and Alawites by governorate needs a 2010 geographer's estimate and is held back for you.

*Filed 2026-10-03 by session `fafd1067-sy`. Anita's call; nothing is waiting on it.*

## What I did

Drew Syria at one national mix in every governorate: Pew Research Center 2020 (which for Syria is the
World Religion Database's figure), 94.2% Muslim, 3.8% Christian, 2.0% no religion, on the Central
Bureau of Statistics' end-2011 population estimate. The Druze, Alawites, Ismailis and Shia are all on
`islam`, because that source counts them as Muslims; the note says so and says the map does not show
where any community lives. Nobody is placed.

## What it costs to reverse

Leaving it: nothing. Placing the Druze in Suwayda (or more): a new estimate in `sources/sy.py`, a
`druze` row in `taxonomy/sy2020.py`, a rescatter and the build tail, about an hour. Taking Syria off
the map: drop `sy` from `ORDER` and retile.

## Why it is yours rather than mine

Spec §14, and the scout's row (`queue.md`, 2026-10-03) said any sect or Christian placement is
yours. In 2025 Alawites were killed on the coast (March), Druze in Suwayda (July) and Christians in
the Mar Elias church bombing in Damascus (June). A flat mix places nobody; any placement would draw
exactly those groups.

## The detail

- **The flat map is wrong in a known way.** Suwayda draws 94% Muslim; it is overwhelmingly Druze.
  Tartus and Latakia draw no Alawite signal (they are on `islam` either way, so this one is invisible).
  Christians are spread evenly instead of in Damascus, Aleppo, Homs, the Wadi al-Nasara and Hasakah.
- **What could place them, and how good it is.** No count or survey gives any community by
  governorate. The 1947 and 1960 censuses counted sects (Friedman, *BJMES* 2024), but that is your
  "too old" bar from Razmara. Arab Barometer IX (1,229 Syrians, late 2025) is unreleased. The best
  that exists is Fabrice Balanche's 2010 estimates (a geographer at Lyon 2; *Sectarianism in Syria's
  Civil War*, Washington Institute, 2018), e.g. Suwayda 90% Druze, 7% Christian, 3% Sunni. A
  geographer's estimate, pre-war, published by a think tank, and not read at source yet.
- **Options.** (a) Leave it flat (what is built). (b) Draw the Druze only, in Suwayda at Balanche's
  share, since that is the largest error and the least revealing fact (it is the Jabal al-Druze);
  the national Druze total would then need a figure too, which Pew does not give. (c) Wait for the
  Arab Barometer release and decide then. I would take (a) now and (c) when the data is out.
- **Not a §14 question but you may want to know:** the dots are 2011 positions, because the UN's
  current governorate figures on HDX are marked not for research use. The note says so, with UNHCR's
  4.7 million registered refugees abroad.
- Record: `sources/sy.md`; `sources.md` §sy-2026-10-03.
