# Cook Islands: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 13,874 people aged 5 and over on 12
islands, 2 nodes, 13 dots at 1:1000. Every row `derived` (multi-answer sharing). Scripts:
`sources/ck_census.py`, `sources/ck_geo.py`, `taxonomy/ck2021.py`, `taxonomy/tree.d/ck.txt`,
`countries/ck.py`. Geography is religiondots' Kontur hexes re-keyed to islands. No asks.

## 1. Source and question

Cook Islands Statistics Office, 2021 Census of Population and Dwellings, report with tables and
questionnaire (`data/raw/ck/census2021.pdf`, stats.gov.ck/2021-census-of-population-and-dwellings/).
Q18 (PDF p. 185): "What language(s) do you speak at home?  Cook Islands Maori? yes / no.  Any other
language(s)? yes / no." Two yes/no answers per person aged 5+, the other language not named. Table
3.07 (p. 98) gives both answers by usual residence: Rarotonga and its 11 districts, five Southern
Group islands, six Northern Group islands. Resident population 14,987; aged 5+ 13,874; the 1,113
under 5 are the gap. Earlier censuses (2016, 2011) are not used.

National: Cook Islands Maori yes 9,725 (70.1%), other language yes 12,289 (88.6%).

## 2. The combinations are recoverable

The report's text (p. 24) gives 29.9% other language only, 11.4% Cook Islands Maori only, 58.7% both;
these sum to 100, so nobody answered no to both. In any unit, with N people, a = Maori yes and
b = other yes: both = a + b - N, Maori only = N - b, other only = N - a. All 12 islands come out
non-negative, and the national totals reproduce 58.7 / 11.4 / 29.9 exactly. So the rule in AGENT_BRIEF
§2 ("microdata giving the combinations beats the scaling") is met without microdata.

## 3. Drawing, and the calls

- A person who answered yes to both is shared half and half (spec §3.6). Maori-only and
  other-only are whole. Result: 5,655 Maori (40.8%), 8,219 other (59.2%). By island Maori is
  0.35 of the mentions on Rarotonga, 0.5 to 0.66 on the other Southern islands and Manihiki,
  Rakahanga and Penrhyn, 0.81 on Pukapuka.
- "Other language" is drawn as English (`taxonomy/ck2021.py`). The census does not name it; English
  is the second official language and the language of schooling; the 2021 ethnicity table
  (Table 5) shows 2,138 residents of other ethnic origin and 1,246 part Cook Islands Maori, whose
  home languages would include others. A bare `other` would put 59% of people on the unnamed remainder.
  Reversing it means a share of `other` or of Fijian, Filipino, etc., for which nothing measured
  exists.
- Cook Islands Maori is one node: the question does not separate Rarotongan, Aitutaki, Manihiki
  and the other varieties. On Pukapuka all 386 people aged 5+ answered yes to it; Pukapukan is a
  separate Samoic language, so some of those answers are presumably Pukapukan, not separable.
- Learned vs home: this is a home-language question ("speak at home"), so unlike El Salvador's
  ability questions, English is not folded onto the main language.

## 4. Geography (`sources/ck_geo.py`)

religiondots' `ck_hexes.gpkg` is Kontur for the country as one unit. Each hex goes to the nearest
of the 12 inhabited islands' (rounded) centres; the farthest hex is 17.7 km from its centre
(Penrhyn's reef), islands are at least 40 km apart. One hex at Suwarrow (113 Kontur people; the
census table has no Suwarrow row) is dropped. Kontur against the census by island, normalised: 0.70
(Rakahanga) to 1.84 (Aitutaki), Palmerston 4.38 on 22 people; log correlation 0.954 against a best
of 0.780 over 500 shuffles. Rarotonga's 11 districts are in the table but there is no district
boundary file in religiondots; district-level placement would use the same table (Maori yes ranges
from 54% of people aged 5+ in Titikaveka and Ngatangiia to 72% in Takuvaine).

## 5. Room for improvement

Rarotonga district boundaries (Statistics Office), which would place the 11 districts' different
mixes; a named "other language" tabulation (the 2021 data hold it, the report does not print it).
