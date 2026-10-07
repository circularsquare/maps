# 053 — er: Eritrea drawn at one national mix from the 2010 health survey; nobody is placed (spec 14)

Summary: Eritrea built flat: the 2010 survey's 37% Muslim in all 6 zobas, Pew's 52% (World Religion Database) a witness only. The state jails members of unregistered churches; a national mix places nobody. Keep, or take Eritrea off?

*Filed 2026-10-03 by session `fafd1067-er`. Anita's call; nothing is waiting on it.*

## What I did

Drew Eritrea at one national mix in all six zobas: the Eritrea Population and Health Survey 2010's
answers from 30,224 women and 4,299 men aged 15-49 (57.4% Orthodox, 37.2% Muslim, 4.4% Catholic, 0.9%
Protestant, 0.2% traditional), on the UN's 3,291,271 for 2020. The survey is the level; Pew's 51.7%
Muslim is printed in the note as a witness and not drawn.

## What it costs to reverse

Taking Eritrea off: drop `er` from `ORDER` in `countries.py` and run the build tail. Drawing it at
Pew's level instead: change the shares in `sources/er.py`, rescatter and tail, about 20 minutes.

## Why it is yours rather than mine

Spec §14 (`AGENT_BRIEF.md` §3, first bar), and the scout's row named it. The US State Department's
2023 report says more than 500 Christians from unregistered churches (most Pentecostal or
evangelical) and 36 Jehovah's Witnesses were in detention, some for nearly 20 years, and only four
bodies are registered. A national mix places nobody, so I think this is a keep, but whether a
persecuted country is drawn at all is yours.

## The detail

- **Pew's figure is not a measurement.** Pew's 2025 appendix of sources gives the World Religion
  Database as its only source for Eritrea in 2010 and 2020, and the State Department says "there are
  no reliable figures on religious affiliation" and quotes the same WRD 52/47. The WRD ascribes
  religion by ethnic group. The three surveys (1995, 2002, 2010) asked people and agree with each
  other (Muslim women 36.5% in 2002, 39.0% in 2010). They cover ages 15-49 and the households on the
  zobas' own village lists; children (47% of the population) are probably somewhat more Muslim, since
  fertility is higher in the Muslim-majority zobas (5.4-5.7 against Debub's 5.0 and Maekel's 3.4), and
  pastoral groups may be under-listed. Neither closes a 15-point gap.
- **The flat map is wrong in a known way**, as Lebanon's is: the highland zobas (Maekel, Debub) are
  mostly Orthodox and the lowland ones (Gash-Barka, the two Red Sea zobas) mostly Muslim, and every zoba
  draws 37% Muslim. No published table crosses religion with zoba or ethnic group; the DHS Program lists
  no Eritrean microdata at all. Nothing better can be built from open sources.
- **The Protestant dots (0.87%, about 28,500)** hold both the registered Lutheran church and the unregistered
  ones; they are spread evenly, which places nobody.
- **Population is a choice**, not a count: Eritrea has never held a census. UN WPP 2024 (3.29 million
  for 2020) against the US Census Bureau's 6.3 million (2023). The note gives both.
- Record: `sources/er.md`; `sources.md` §er-2026-10-03.
