# Djibouti (dj): record

Drawn 2026-10-05 (session edd42a8c-est). The 2024 census does ask languages: Tome 4 of RGPH-3,
which religiondots had already downloaded for religion, tabulates "languages spoken" by region
(the parked handoff searched for a table and missed it; the estimate route turned out not to
be needed for the bulk). 1,003,800 people (ordinary and nomadic households, religiondots'
`dj_lookup.csv` `table_2024`), 6 regions, 8 nodes. Census rows `derived`, native-Arabic rows
`modelled`. 1,000 dots at 1:1000, no rings.

```
python sources/dj_rgph.py
python taxonomy/build.py
python tools/check_country.py dj
python scatter.py --country dj
```

## 1. The table

RGPH-3 2024, Tome 4 "Caracteristiques socioculturelles de la population" (INSTAD), religiondots'
`data/raw/dj/rgph3/` (read-only). Tableau 7 (pp. 39-40): residents aged 5+ in ordinary
households, by region, "parle en" yes/no for French, Arabic, English, Somali, Afar, Amharic,
Oromo, sign language and "other languages not listed" (variables SC08_*, one per language,
Tableau 2). Base 882,594. Transcribed into `sources/dj_rgph.py`; checks: every language's regions
sum to the national column; "speaks" + "does not speak" equals the region's base for French,
Arabic, Somali and Afar in every region. National: Somali 78.4%, French 41.8%, Arabic 26.6%,
Afar 25.8%, English 17.0%, Amharic 4.3%, Oromo 3.6%, other 0.8%, signs 0.3%.

No mother-tongue or ethnicity table: the report says the ethnicity variable "n'a pas ete
pleinement exploitee" for lack of completeness. Tableau 25 crosses the same languages by
nationality (Djiboutian 925,503; Ethiopian 51,481; Somali 12,537; Yemeni 3,781; Eritrean 1,679).

## 2. How the counts are made

AGENT_BRIEF section 2's multi-answer rule with the learned-languages rule:
- French and English folded (school and work languages; 3-8% among Ethiopian, Somali and
  Yemeni nationals, 39% among Djiboutians).
- Arabic folded too: 24.5% of Djiboutian citizens name it, a school and religious language for
  most. **It is also a real first language** for the Arab community of Yemeni origin, so that
  part is drawn from Joshua Project's people groups (data/raw/pg/joshuaproject_pgic.csv):
  "Arab, Yemeni" 41,000 (Ta'izzi-Adeni Arabic) and "Arab, Omani" 28,000, 3.53% and 2.41% of
  JP's Djibouti total of 1,161,200, applied to 1,003,800 = 35,442 and 24,205, all in
  Djibouti-Ville (the Arab quarter; JP's Omani point is there; spreading by the census's Arabic
  mentions would have put 5% Arabic speakers in Afar villages). Ask 019's ruling covers it.
- Per region: the native-Arabic carve-out, then the rest shared across Somali, Afar, Amharic,
  Oromo, signs and other by mentions (1.02 mentions per person in Ali-Sabieh to 1.20 in
  Tadjourah). Children under 5 and nomadic households take their region's shares.

Drawn: Somali 64.8%, Afar 21.8%, Amharic 3.6%, Yemeni Arabic 3.5%, Oromo 2.9%, Arabic 2.4%,
other 0.7%, signs 0.3%. By region, Somali 70% of the capital, 93% Ali-Sabieh, 83% Arta; Afar
82% Tadjourah, 83% Obock; Dikhil 50% Afar, 46% Somali.

**Check against the circulating figures** (Somali 48.7%, Afar 29.5%, Arabic 15.4%, French 2.1%;
axl.cefan.ulaval.ca, uncited): Afar comes out lower here. The census's own Afar mentions are
25.8% of people 5+, so the gap is not the method; the circulating figure has no source.

## 3. Calls someone might reverse

- Arabic folded and replaced by a JP estimate in the capital only (60k). If the supervisor or
  Anita prefers, Arabic mentions could be shared in instead (would draw about 20% Arabic).
- Amharic and Oromo shared in as named, although some Djiboutians name them as a second
  language (23,838 Djiboutian citizens name Amharic).
- French expatriates and the French military (JP: 26,000 French) are not separated.

## 4. Room for improvement

A mother-tongue or ethnicity table (the census collected ethnicity; INSTAD did not publish it),
or microdata giving the combinations of languages named.

## Terms

RGPH-3 Tome 4: INSTAD public report. Joshua Project dataset: free download. Kontur CC BY 4.0.
