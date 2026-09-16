# 040 — sa: Saudi Arabia: split citizens into Sunni and Shia by region?

Summary: Saudi citizens are drawn on plain Islam. Evidence places Shia in the Eastern Province, Najran and Madinah, but they have been bombed and prosecuted. Draw a Sunni/Shia split by region, or keep one Islam?

*Filed 2026-09-15 by session `cb8b206e-sa`. Anita's call; nothing is waiting on it.*

## What I did

Drew all 18,792,262 Saudi citizens on `islam` in each of the 13 regions, with no Sunni/Shia split.
Non-Saudis carry no sect either (their Muslim branches are folded to `islam` in `sources/sa.py`).

## What it costs to reverse

Change the citizens block in `sources/sa.py` and the MAP in `taxonomy/sa2022.py`, rerun it and both
scatters, then the build tail: under an hour. The harder part is that only one region has a number
(below).

## Why it is yours rather than mine

AGENT_BRIEF §3, first bar (spec §14): whether publishing where a group lives affects its safety.
Islamic State suicide bombers attacked the Shia Imam Ali mosque in al-Qudaih, Qatif (22 May 2015, at
least 21 killed) and a Shia mosque in Dammam (29 May 2015, 4 killed), and an Ismaili mosque in Najran
(26 October 2015, claimed as an attack on "the rejectionist Ismailis"; Al Jazeera, 27 October 2015).
The US State Department's 2023 report records Shia citizens prosecuted, and sentenced to death, out of
proportion to their numbers.

## The detail

**Levels, from documents opened.**
- US State Department, *2023 Report on International Religious Freedom: Saudi Arabia*, Section I:
  citizens 85-90% Sunni; Shia "10 to 12 percent of the citizen population and an estimated 25 to 30
  percent of the Eastern Province's population"; Qatif "home to the country's largest Shia
  population".
- Human Rights Watch, *Denied Dignity* (2009): Shia 10-15% of the population; Twelvers "predominantly
  in the Eastern Province, and in Medina, home to the so-called Nakhawila".
- Human Rights Watch, *The Ismailis of Najran* (2008): Ismailis "widely believed to constitute a
  large majority of the Najrani population" (the 2004 census: about 408,000 in Najran).

**What a split by region would show, on the 2022 census.**
- 10-12% of 18,792,262 citizens is 1.88 to 2.26 million Shia.
- Eastern Province: 25-30% of its 5,125,254 residents is 1.28 to 1.54 million, which would be 43-52%
  of its 2,949,854 citizens. The report does not say whether its base is residents or citizens.
- Najran: "a large majority" of 394,976 citizens, with no figure. Madinah: named, with no figure.
- The rest of the 1.88 to 2.26 million would sit in the other ten regions with nothing saying where.
- The tree has `islam.shia` and its Ja'fari child and no Ismaili node, so Najran's Ismailis would sit
  on `islam.shia` or need a node of their own.

**Options.**
- (a) Keep one Islam, as built. Even without §14, `[[feedback_dont_draw_unsourced_breakdowns]]`
  would leave Najran and Madinah on the parent, and the only number is the Eastern Province's.
- (b) Split the Eastern Province only, at the State Department's share, everyone else on `islam`.
- (c) Split all three regions, with Najran and Madinah on figures that do not exist yet.

My lean is (a), on the thinness of the numbers as much as on §14. Nearby rulings: Iran was split from
Masaili's province estimates after a §14 flag (2026-09-16); Pakistan's partial Punjab Shia shares were
refused as partial (2026-09-16); Iraq's Sunni and Shia answers are drawn at governorate (ask 021).
Regions here average 2.5 million people; the Eastern Province is 5.1 million.
