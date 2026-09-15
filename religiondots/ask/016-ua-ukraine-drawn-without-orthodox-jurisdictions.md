# 016 — ua: Ukraine drawn without Orthodox jurisdictions, at the pre-war level over occupied territory

Summary: Ukraine: keep Orthodox as one colour, draw Crimea, Sevastopol and Donbas from pre-war answers with a note, and leave the level un-rescaled? Recommended: yes to all three.

*Filed 2026-09-14 by session `f95259a4-ua`. Anita's call; nothing is waiting on it.*

## What I did

Drew Ukraine from ESS rounds 2-6 (2005-2013) at all 27 COD-AB units, Crimea and Sevastopol
included, with the Orthodox answer as one colour (`christianity.orthodox`) and no split between the
Orthodox Church of Ukraine and the church of the Moscow Patriarchate. The population is Ukrstat's
last pre-invasion estimate (1 January 2022; Crimea and Sevastopol 1 January 2014), and the level is
2005-2013's everywhere: round 11 (2023-24) is printed as a witness and not used to rescale.

## What it costs to reverse

Any of the three is an edit to `sources/ua.py` or `taxonomy/ua2013.py`, a re-run (about a minute),
the scatter and the build tail, about 40 minutes in all. Taking occupied territory off the map is the
same plus a `gap=` line.

## Why it is yours rather than mine

AGENT_BRIEF.md §3, the §14 bar: which church a place is drawn as belonging to is politically loaded
in Ukraine now (the 2024 law on religious organisations affiliated with Russia's church, the transfer
of about 1,400 communities from the UOC to the OCU in 2022-2025), and so is how territory Russia
occupies is drawn. Both are about what the map says about a group and a place in wartime, which the
brief does not settle.

## The detail

**Three questions, one decision each.**

1. **Orthodox jurisdictions, not drawn.** The pooled rounds share only the harmonised card, which has
   one Orthodox answer. Ukraine's own card exists in rounds 4-6 and 11 and the two cannot be pooled:
   - rounds 4-6 (2009-2013): Moscow Patriarchate 47.9% of the Orthodox answer, Kyiv Patriarchate
     46.8%, other 3.8%, Autocephalous 1.5%; Donetsk and Luhansk 75% Moscow, Dnipropetrovsk,
     Zaporizhzhia and Kirovohrad 69% Kyiv;
   - round 11 (2023-24, government-controlled territory only): OCU 74.7%, Orthodox of no patriarchate
     15.0%, Moscow Patriarchate 10.3%. The Moscow share is highest in Odesa/Mykolaiv/Kherson (20%) and
     in the far west (Lviv/Ivano-Frankivsk/Zakarpattia/Chernivtsi, 19%, which is Chernivtsi and
     Zakarpattia).
   - The Kyiv Patriarchate and the Autocephalous Church merged into the OCU in 2018; Razumkov's own
     series has the Moscow Patriarchate at 20% of adults in 2013 and 5.4% in 2025.
   - DESS's 2024 register still has more UOC communities (10,586) than OCU (8,075), so the
     institution count and self-identification point opposite ways, which is §3.6's warning.

   Options: (a) as built; (b) round 11's split at the 8 macro-regions for the 23 oblasts it sampled,
   occupied units left as undivided Orthodox; (c) rounds 4-6's split, labelled 2009-2013. (c) would
   draw millions as Moscow Patriarchate who now say otherwise. My preference is (a), or (b) if you
   want the jurisdiction visible.

2. **Occupied territory, drawn at its last Ukrainian figures.** Crimea and Sevastopol are drawn from
   ESS rounds 2-6, which sampled Crimea (502 respondents; Sevastopol is never named and takes
   Crimea's composition), at Ukrstat's 1 January 2014 figures. Donetsk and Luhansk (including the
   parts occupied since 2014) are at Ukrstat's 2022 whole-oblast estimates. The precedent that does
   not fit is Georgia, where Abkhazia draws nothing because the census did not enumerate it; here the
   source did sample these places, before occupation. `note_public` says the map shows neither
   displacement nor change under occupation. The alternative is to leave Crimea and Sevastopol blank
   (2.35 million people) with a `gap=` line.

3. **The level, not rescaled.** On the 23 oblasts round 11 sampled, No religion is 33.9% against the
   map's 27.5% (+6.3 points), Orthodox 54.5% against 58.0%, Catholic 9.5% against 10.8%. Norway and
   Latvia rescaled above 3.5 points (spec §12). Here the only measurement of the change excludes
   Crimea, Sevastopol, Donetsk and Luhansk, so rescaling the 23 draws a vintage step on the 2014
   line and rescaling all 27 asserts a change nobody measured under occupation. Ask 015 is the same
   question for the LAPOP countries without the occupation problem.

What held up, for context: every round's oblast labels match the people behind them (Kyiv city the
most big-city, Galicia the most Catholic, Crimea or Donbas the most Russian-speaking before the war);
round 11 orders the oblasts like the pool for Catholic (+0.825) and No religion (+0.628); Razumkov's
four 2025 regions order Orthodox, Greek Catholic and no religion the same way. sources/ua.md has the
rest.
