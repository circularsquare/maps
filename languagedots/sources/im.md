# Isle of Man: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 84,028 residents who gave a home language
(41 gave none), one unit, 12 nodes from 12 labels, 81 dots and 10 rings at 1:1000. Every row
`measured`. Scripts: `sources/im_census.py`, `taxonomy/im2021.py`, `taxonomy/tree.d/im.txt`,
`countries/im.py`. Geography is religiondots' `im_hexes.gpkg` (one unit `IM`, 567 hexes), read in
place. No asks.

## 1. Source and question

Isle of Man Census 2021, Report Part I, Table 2.11 "Language Mainly Spoken at Home" (PDF p. 28).
It comes from Q6 of the form (Part II, p. 43): "What language do you use most in the home?", with
tick boxes for English, Filipino, Bengali, Polish, Bulgarian, Punjabi, Chinese, Urdu and a
write-in. The report prints English 80,781, Polish 504, Filipino 435, Bulgarian 376, Chinese 349
(footnote: "includes all Chinese languages"), Portuguese 125, Afrikaans 99, Romanian 96, Spanish
79, Hungarian 77, Italian 72 and Other 1,035; data unavailable for 41 people. The report is
religiondots' copy (`religiondots/data/raw/im/`), read in place. Island-wide only: no table of
language by parish or town is published (the report's only small-area table is Manx ability).

## 2. Checks (`sources/im_census.py`)

The 12 rows sum to 84,028 = 84,069 residents less the 41 without an answer; each printed
percentage equals count / 84,069; 84,069 is the resident population on pp. 7, 9, 16 and 19.

## 3. Calls

- "Other" (1,035, 1.2%) is an unnamed remainder, on `other`. The form offered Bengali, Punjabi and
  Urdu boxes the table does not print, so those answers are inside it; so is any Manx. Nothing
  splits it (an office table would be needed, and the Economic Affairs Division published none).
- Manx (Q10: "Can you speak, read, or write Manx Gaelic?") is ability, not the language mostly
  used at home: 2,223 answered yes to some part (Table 2.13), 1,005 to speaking alone, 702 all
  three. It is not drawn, as a learned language is not a home language (AGENT_BRIEF §2).
- "Filipino" on the Filipino node, as printed; "Chinese" on the Sinitic group (variety not named).
- One unit: the dots follow where people live on Kontur's grid.

## 4. Room for improvement

A parish or town table of home language (Q6) would place the Polish, Filipino and Bulgarian
communities; the census holds it, the report does not print it.
