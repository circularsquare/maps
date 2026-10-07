# Niger: RGP/H 2001 ethnic group read as language, moved by Afrobarometer R5-R6 retention

Drawn 2026-10-05 (session edd42a8c-ne). 17,138,707 people (RGP/H 2012 région populations, as
religiondots draws Niger), 8 régions, 10 nodes, every row `derived`. 17,134 dots at 1:1000, no
rings.

```
python sources/ne_census.py --fetch     # Niger's Afrobarometer rows from religiondots' .sav (read-only)
python sources/ne_census.py --rounds none|5,6,7,8,9   # the comparisons in section 2 (re-run plain after)
python taxonomy/build.py
python tools/check_country.py ne
python scatter.py --country ne
```

## 1. What exists

- **RGP/H 2012**: no language and no ethnic question. Its socio-cultural chapter ("État et
  structure", stat-niger.org) covers religion and nationality only; none of the 13 thematic
  reports or 16 regional monographs listed on stat-niger.org's RGPH 2012 page is about
  ethnicity or language (checked 2026-10-05; the coverage sweep had searched the structure
  PDF on 2026-10-03).
- **RGP/H 2001**: question C07 "nationalité ou ethnie" (a Nigerien gives their ethnic group).
  "État et structure de la population" (2001 analysis volume), Tableau 30, ethnic group by
  département in counts, Nigerien residents only (10,964,126). Copy from IREDA (CEPED's census
  inventory): `data/raw/ne/ner-2001-rec-o_etat-structure.pdf`, with the household questionnaire
  beside it. The 2001 départements are today's régions, same eight units. The questionnaire also
  asks the languages a person can read and write (literacy), not a spoken language. The widely
  quoted national shares (Hausa 55.4%, Zarma-Songhai 21.0%, Tuareg 9.3%, Fulani 8.5%, Kanuri
  4.7%) are this census's Tableau 29.
- **RGP 1988** (`ner-1988-rec-o1_etat_population.pdf`, also downloaded): no ethnic or language
  table in the état volume.
- **Afrobarometer** R5 (2013), R6 (2015), R7 (2018), R8 (2022), R9 (2022), about 1,200 each,
  all eight régions (R6 has none in Diffa). Each asks ethnic group and a language: R5-R6
  "Language of respondent", R7-R9 "Language spoken in home". Niger is not in R4.
- Not tried: DHS (registration off), MICS (UNICEF account). CLEAR Global's Niger layer is
  Afrobarometer-based, so not a second source.

## 2. Census ethnicity, survey retention, and the Hausa question

**The surveys sample Niger's minorities well.** Survey ethnic group by région, all rounds pooled,
against the 2001 census (%): Agadez Tuareg 59.2 / 60.1; Diffa Kanuri 56.0 / 60.2, Fulani 22.8 /
24.6, Tubu 4.4 / 6.2; Tahoua Tuareg 11.5 / 17.5; Zinder Kanuri 10.7 / 13.1, Tuareg 10.5 / 7.5;
Tillabéri Zarma 73.5 / 63.6 (`ne_census.py` prints all eight). Unlike Sudan and Algeria, the
minorities are in the sample.

**But their home-language answers lean to Hausa.** Interviews were in Hausa (63-70%), Zarma,
French, and a few in Tamasheq and Fulfulde; none in Kanuri save 8 in R8. Of Tuareg
respondents interviewed in Tamasheq, 97 of 101 named Tamasheq; interviewed in Hausa, 240 named
Hausa and 168 Tamasheq. Retention (own-language answer) by round, Tuareg / Kanuri / Fulani:
R5 40% / 44% / 54%; R6 85% / 82% / 90%; all five rounds pooled 47% / 58% / 57%. R6 is the odd round out:
its language answers almost copy its ethnic answers (Hausa 731 vs 725, Tamasheq 91 vs 96).
Retention also follows place: Tuareg in Agadez 84-100% in R5-R7, in Zinder 5-42% outside R6;
Kanuri in Diffa 71-92%, in Maradi 0 of 22.

**Drawn:** the census's ethnic shares per région, each group moved onto the languages its
members named in R5-R6 (the "language of respondent" rounds, the first-language reading of ask
018), per région, shrunk to the group's national R5-R6 answers with K = 10. Groups under 30
respondents (Arabe 5, Gourmantché 5, Toubou 5) are kept whole. Retention used: Tuareg 58.2%,
Kanuri 60.9%, Fulani 72.3%, Zarma 95.6%, Hausa 97.9%. Then times each région's 2012 population,
largest-remainder rounding; every région sums to its census population.

National results under each choice (the switch is `RETENTION_ROUNDS`, one line):

| | no move | R5-R6 (drawn) | R5-R9 |
|---|---|---|---|
| Hausa | 56.4% | 60.5% | 61.9% |
| Zarma | 19.4% | 21.2% | 20.7% |
| Tamajaq | 9.5% | 6.6% | 6.4% |
| Fulfulde | 8.5% | 6.6% | 5.0% |
| Kanuri | 5.1% | 4.0% | 3.6% |
| French | 0 | 0.04% | 1.2% |

Drawn per région (%), Hausa / Zarma / Tamajaq / Fulfulde / Kanuri: Agadez 35 / 6 / 51 / 2 / 3;
Diffa 25 / 3 / 1 / 13 / 50 (Tubu 6); Dosso 42 / 49 / 1 / 7 / 1; Maradi 91 / 1 / 1 / 7 / 0;
Niamey 34 / 57 / 3 / 5 / 1; Tahoua 83 / 1 / 13 / 2 / 0; Tillabéri 9 / 69 / 8 / 12 / 0; Zinder
77 / 1 / 5 / 7 / 9.

Why move at all, given the interview effect: every round but R6 finds the same thing, a large
Hausa-speaking share among Tuareg, Kanuri and Fulani living in Hausa country (Zinder, Maradi,
Tahoua), which is what one expects of Bouzou and sedentary Fulani there; and the brief asks
for retention where a source gives it. Why not R7-R9: their wording takes in the lingua franca
(ask 018), and they put 1.2% on French (82 answers, mostly from French interviews).

## 2a. Retention at R7's mother tongue (2026-10-05, session edd42a8c-r7e)

Anita ruled on ask 018: lingua francas at Afrobarometer R7's **mother tongue** question (Q2A).
The extract now holds Q2A for R7 (not Q2B, "language spoken in home"), and
`RETENTION_ROUNDS = (7,)`: each group's move onto Hausa or Zarma comes from R7's mother-tongue
answers alone (same K = 10 and MIN_N = 30). R7's mother tongue nearly copies ethnicity, as R6
did: Tuareg 94% Tamasheq (127 respondents), Kanuri 90% (59), Fulani 98% (57), Zarma 99.8%,
Hausa 97%. So the move onto Hausa almost disappears.

| national | before (R5-R6) | after (R7 mother tongue) |
|---|---|---|
| Hausa | 60.5% | 55.8% |
| Zarma | 21.2% | 20.6% |
| Tamajaq | 6.6% | 9.3% |
| Fulfulde | 6.6% | 8.4% |
| Kanuri | 4.0% | 4.8% |
| French | 0.04% | 0 |

The drawn map is now within a point of "no move" (section 2's first column) for every
language. R7's Q2A was asked mostly in Hausa interviews too (797 of 1,200), so the earlier
point stands that a Kanuri- or Tamajaq-language survey would settle how much shift is real;
R7's mother-tongue item simply does not record the shift that R5's home-language one did.

## 3. Mapping and tree (`taxonomy/ne2001.py`, `taxonomy/tree.d/ne.txt`)

- Djerma-Sonrai / Zarma-Songhay on ng.txt's `songhay.zarma` leaf: the group also holds
  Koyraboro Senni speakers (Ayorou, north-west Tillabéri) and Dendi (Gaya), which neither source
  separates; Zarma is most of it.
- Touareg / Tamasheq on a new leaf `berber.tamajaq` "Tamajaq (Tuareg)": Niger's Tawallammat
  (tawa1286) and Tayart (taya1257) are separate Glottolog languages from Mali's Tamasheq.
  Coloured one shade off Mali's Tamasheq.
- Kanouri-Manga on `kanuri` (Manga, Central and Tumari Kanuri together). Toubou on a new leaf
  `nilosaharan.tubu` (Dazaga and Tedaga, Saharan), steel blue. Peulh on Fula, Gourma on bf.txt's
  Gourmanchéma.
- Arabe split by place: Diffa and Zinder on Shuwa Arabic (the Mohamid and other Chadian Arabs),
  elsewhere on Hassaniya (Azawagh and Kounta Arabs). 66k people.
- French: one R5 answer, spread by the shrinkage to 6,611 people.

## 4. Calls someone might reverse

- Moving people by survey retention at all (section 2's table has the alternatives).
- R7's mother tongue for retention (§2a, Anita's ruling on ask 018); `RETENTION_ROUNDS` is the
  one-line switch.
- 2001 ethnic shares applied to 2012 populations: assumes the mix held for eleven years.
- Arabic split by région; Djerma-Sonrai all on Zarma.
- Diffa's Hausa share (25%, against 4.5% Hausa by ethnicity) comes from moved Kanuri and Fulani;
  R6 has no Diffa respondents, so it rests on R5's 40 plus the national vectors.

## 5. Room for improvement

The 2001 census microdata or a département-level table (63 départements of today) would sharpen
the grain; neither is online. A Nigerien survey interviewing in Kanuri, Tamajaq and Fulfulde
would settle how much of the move to Hausa is real.

## Terms

RGP/H 2001 and 2012: INS-Niger publications, quoted. Afrobarometer: free download, citation
requested. Glottolog CC BY; Kontur CC BY 4.0.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Tamajaq is in a Tuareg group with Mali's Tamasheq and Algeria's Tamahaq. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
