# 045 — az: Karabakh drawn as the 2019 census left it; the 2023 refugees are on no map

Summary: Azerbaijan is drawn on the 2019 census, so the area outside government control then is empty and the 100,000+ Karabakh Armenians who reached Armenia in 2023 are on no map. Keep, or add them to Armenia?

*Filed 2026-10-03 by session `fafd1067-az`. Anita's call; nothing is waiting on it.*

## What I did

Azerbaijan is drawn on the 2019 census's de facto ("existing") population, inside today's de facto
borders (spec §14.18), so Karabakh is Azerbaijan's and the area outside government control from
1994 to 2020 is empty: nobody there was enumerated in 2019, the Karabakh Armenians have left since,
and the people resettled there since 2021 are drawn where they lived in 2019. The note says all
three things in plain words. The Karabakh Armenians are drawn nowhere: Azerbaijan never counted
them and Armenia's 2022 census came before they arrived.

## What it costs to reverse

Adding the refugees to Armenia: a refugee layer by marz for `am` from Armenia's registration
figures (not yet looked for), about a session. Drawing resettlement inside Karabakh instead: a
per-district resettlement figure, which I did not find (only a national "more than 48,000"), then
`sources/az.py` and a rescatter, 10 minutes.

## Why it is yours rather than mine

AGENT_BRIEF §3, first bar: who a map depicts. A displaced Christian population of about 100,000,
on no map at all, beside land drawn empty or Muslim; [[feedback_flag_ethics_for_discussion]].

## The detail

- **2019 census**, Volume A Table 3: permanent (de jure) 9,951,409, existing (de facto) 9,943,958.
  The eight districts outside government control (Kalbajar, Lachin, Gubadli, Zangilan, Shusha,
  Khojaly, Khojavend, Khankendi) have 268,468 permanent residents, all `temporarily absent`
  (displaced people registered at origin), and 0 existing; Aghdam, Fuzuli, Tartar and Jabrayil were
  split by the line. The footnote: the population of the occupied territories was not enumerated.
- **Placement** masks Natural Earth 4.1.0's (2018) Nagorno-Karabakh polygon, the 1994-2020 line,
  because Kontur 2023 still puts people on the ruins of Aghdam and Fuzuli and in Stepanakert.
- **Refugees**: UN News, 29 September 2023, quoting UNHCR's Filippo Grandi, "more than 100,000
  refugees had now arrived in Armenia from Karabakh". Armenia (`am`) is drawn from its October 2022
  census, so they are not in it.
- **Resettlement**: president.az/en/greatreturn (read 2026-10-03), "more than 48,000 people
  currently live and work in the liberated territories", undated, no district breakdown.
- **Precedents, none exact**: Ukraine's occupied oblasts drawn at pre-war figures (ask 016); Gaza
  drawn as counted in 2017 (ask 028); Sudan drawn at one share during the war with a refugee layer
  allowed later (ask 041). The closest is Sudan's "a refugee layer by state is allowed later".
- **Options**: (a) keep as built; (b) add the refugees to Armenia by marz, Armenian Apostolic, from
  Armenia's registration figures; (c) also draw the 48,000 resettled people in the districts, if a
  district split turns up.
