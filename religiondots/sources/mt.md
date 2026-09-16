# Malta (`mt`): 2021 census, religion by the 68 localities

Built 2026-09-15 by session `d743fc47-mt`. Code: `sources/mt.py` (the tables and their checks),
`sources/mt_geo.py` (units and placement), `taxonomy/mt2021.py` (mapping), `countries/mt.py`.

## 1. The source

National Statistics Office, *Census of Population and Housing 2021: Final Report: Population,
migration and other social characteristics (Volume 1)*,
`https://nso.gov.mt/wp-content/uploads/Census-of-Population-2021-volume1-final.pdf`, 175 pages,
6,820,290 bytes, `%%EOF` present. `nso.gov.mt` answers 403 to curl and to WebFetch, so Anita
downloaded it in a browser (ask 027); it is in `data/raw/mt/`. Census night 21 November 2021.

Chapter 5, *Religious Affiliation*, pp.159-168: Table 5.1 (sex and age), **5.2 (district and
sex)**, **5.3 (locality, pp.162-164, the table drawn)**, 5.4 (Maltese or not), 5.5 (citizenship),
5.6 (country of birth), 5.7 (racial origin). Malta's first census to ask religion, of residents
aged 15 and over (p.7), defined as self-identification "regardless of the level of religious
attendance or observance, or formal membership" (p.174). UNSD table 28 has no Maltese row.

Ten answers: Roman Catholicism 373,304 (82.64%), no religious affiliation 23,243 (5.15%), Islam
17,454 (3.86%), Orthodoxy 16,457 (3.64%), Hinduism 6,411, Church of England 5,706,
Protestantism 4,516, Buddhism 2,495, Judaism 1,249, other religious groups 911. Total 451,746.

## 2. The checks (all in `sources/mt.py::check`)

- Table 5.2's nation, island and district rows equal a transcription; persons = males + females;
  districts sum to islands and islands to the nation in all 11 columns.
- Table 5.3's 68 locality rows close and sum to its district rows, which equal Table 5.2's.
- Table 5.1's age rows sum to its totals, and its total is the national row (ages start at 15).
- Table 5.4: Maltese 347,577 + non-Maltese 104,169 = the national row in every answer; its
  Maltese no-religion cell is p.159's "7,254".
- **Every locality's religion total equals its population aged 15 and over in Table 1.5**
  (single years of age, one page per locality), whose totals equal Table 1.2's. The under-15s
  sum to 67,816, exactly 519,562 less 451,746.

One trap, now aliased in code: Table 1.5 heads Gozo's Żebbuġ page `Iż-Żebbuġ, Għawdex` (as
Table 1.1 and p.15's prose do) where Tables 1.2, 1.10 and 5.3 print `Iż-Żebbuġ`. The first run
found 68 pages but one name short, and the under-15s 423 short, which is that page's 3,303 less
its 2,880 aged 15 and over. Malta's own Żebbuġ is `Ħaż-Żebbuġ` everywhere, so no twin is at risk.

## 3. Universe, and what is not known about non-response

**The religion table has no not-stated answer, and its totals are the full population aged 15
and over in every locality.** So either everyone answered or answers were filled in. The
methodological note (p.171) says the census "could estimate information on persons who did not
participate directly in the Census" from administrative registers, and that this "made possible
the introduction of a weighting factor". Religion is not a register item, so some answers are
presumably estimated or weighted; the volume does not say how, or for how many. Recorded, not
undone (playbook: "record, never undo").

`gap` is the under-15s only: 67,816 people, **13.05%** of the 519,562 residents, hand-written
because they are in no religion table.

Not checked: the 2021 questionnaire (whether the ten answers are boxes or coded write-ins, and
whether the question was voluntary). `census2021.gov.mt` no longer resolves (2026-09-15), the NSO
site answers 403, and a MaltaToday article on the results answered 403 to WebFetch. Next place to
look: the Preliminary Report, `nso.gov.mt/wp-content/uploads/Census-of-population-2021-publication-web.pdf`
(a browser download), and the Wayback Machine for `census2021.gov.mt`.

## 4. Mapping

`taxonomy/mt2021.py`. Four REVIEW calls, each with the figures behind it: Roman Catholicism on
`christianity.catholic.latin`; Orthodoxy on `christianity.orthodox` (at most 1,282 of 16,457 are
of Asian, African or Arab origin, a ceiling on the Oriental Orthodox share); Protestantism on
`christianity.protestant`; other religious groups on a new `other.mt` (911; Indian citizens are
202 of it, and the census has no Sikh answer). Buddhism names no school and stays on `buddhism`.

## 5. Geography and placement

**Units.** GISCO LAU 2021 (`data/geo/lau2021/`, shared with Cyprus) has the 68 Maltese LAUs; the
volume says its localities are the LAU classification (p.172). The census prints names only, so
the join is on the name, one-to-one, with three witnesses the name does not decide:
(1) every LAU code's district prefix (MT011 ... MT026) matches the district Table 5.3 prints the
locality under; (2) GISCO `AREA_KM2` against Table 1.10's printed area runs 0.961 (L-Imdina) to
1.030 (Bormla), and the two possible same-name swaps would read 0.109 (Rabat as Rabat, Gozo) and
0.873 (Ħaż-Żebbuġ as Iż-Żebbuġ); (3) Eurostat's LAU population, a register figure of another date,
runs 0.78 to 1.30 of Table 1.2.

**Placement: measured against spec §8.2e, and Kontur kept on a cut layer.** The median locality
is 2.94 km2, 4.0 Kontur hexes, and 8 localities are smaller than one hex. A centroid join gives a
median of 3 hexes and leaves 4 localities with none (Ix-Xgħajra, L-Imtarfa, L-Isla, Ta' Xbiex),
which is the Saint Vincent shape. So the hexes are intersected with the localities and each
hex's people shared over its area inside the localities, which is its land, since every part of
Malta is in one: 799 pieces, median 9 per locality, minimum 2, every locality covered.
The bar was set before the numbers were read: keep the grid if its per-locality agreement with
Table 1.2 has p10 of at least 0.6 and p90 of at most 1.6 (Saint Vincent had 0.00 and 2.68). It
has p10 **0.68**, median 0.99, p90 **1.38**, Spearman 0.945, with Kontur inside the localities at
1.030x the census. Its worst misses are L-Imtarfa (0.49), Ta' Xbiex (0.55) and Tas-Sliema (0.58),
under-read, and L-Imdina, over-read at 2.36 on 193 people. The first build shared each hex by its
whole area, sea included, which dropped 29,261 of Kontur's 535,078 people and tilted the weight
inland in the harbour towns (p10 0.66, p90 1.45, L-Isla 0.28, Tas-Sliema 0.38); the review below
found it and it was fixed the same day. The weight only places a
locality's dots inside that locality, so those misses move nothing between localities; what it
buys is the large rural councils, Ir-Rabat (26.6 km2), Il-Mellieħa (22.6), Is-Siġġiewi (19.9) and
L-Imġarr (16.1), where uniform placement would put dots on the cliffs and garrigue.

## 6. What the table shows

Maltese citizens are 96.4% Roman Catholic (335,054 of 347,577) and 2.1% of no religion; the
104,169 non-Maltese aged 15 and over, 23.1% of that age group, are 36.7% Roman Catholic, 15.3% of
no religion, 15.1% Muslim and 14.5% Orthodox (Table 5.4). By locality: Islam is 14.0% of Il-Marsa
and 10.7% of Birżebbuġa; Orthodoxy 15.8% of San Pawl il-Baħar (4,427 people, 27% of Malta's
Orthodox); Hinduism 8.3% of L-Imsida; no religion 12.4% of Tas-Sliema and 12.3% of San Ġiljan;
Buddhism 4.3% of Iż-Żebbuġ in Gozo; Roman Catholicism 96.8% of Santa Luċija. Judaism, 1,249, is
largest in Tas-Sliema (178), and 627 of the 1,249 hold a citizenship the tables group as other.

## Review, 2026-09-15 (session `d743fc47-rev10`, full pass)

- **Checks.** `check_md.py` clean, both editions present, `check_rollup.py mt` all measured with
  nothing orphaned.
- **Figures.** Every locality figure in `note_public` recomputes off `mt.csv` (Il-Marsa 675 Muslims,
  14.0%; Birżebbuġa 10.7%; San Pawl Il-Baħar 4,427 Orthodox, 15.8% and 26.9% of Malta's; L-Imsida
  Hindus 8.3%; no religion 12.4% of Tas-Sliema and 12.3% of San Ġiljan; Santa Luċija 96.8% Roman
  Catholic), and each is the top locality for its answer. "Almost doubled since 2011" is p.15's own
  sentence.
- **Mapping agreed.** Roman Catholicism on `.latin` as be cy ie it gi; Orthodoxy on
  `christianity.orthodox` as at hu mk ke sc zm as ie (am ee ge xk use `.canonical`, each for a named
  national church); `other.mt` is a routine residual.
- **The placement exception is agreed, with one defect in how the hexes are shared.** Cutting is the
  right answer below the floor: a locality inside one hex gets one piece and is placed uniformly,
  which is what §8.2e would do anyway, and the large rural councils get real information. But
  `mt_geo.py` line 277 divides each piece's area by the whole hex's area, so the sea part of the 127
  coastal hexes (of 440) takes its people with it: 29,261 of Kontur's 535,078 (5.5%) are dropped, and
  inside a coastal locality the weight tilts inland. Recomputed with each hex's people shared over
  its area inside the localities (all Maltese land is in one, so that is its land area): the median
  locality moves 0.5% of its weight, p90 10.7%; L-Isla 22%, Tas-Sliema 18%, Ix-Xgħajra 17%, San
  Ġiljan 14%, San Pawl Il-Baħar 11%, Marsaskala 11%, Birżebbuġa 10%. The agreement with Table 1.2
  improves: p10 0.68, median 0.99, p90 1.38, Spearman 0.945, national 1.030x; L-Isla goes from 0.28
  to 0.93 and Tas-Sliema from 0.38 to 0.58. So the two worst misses in §5 were mostly the sea
  share, not Kontur. The error has a direction (away from the waterfront, where the dense towns are)
  but moves dots only inside localities of a few km2, so nothing was rebuilt. The fix is that one
  line: divide by the sum of the hex's piece areas instead of `hex_km2`, then `scatter.py` and the
  build tail. `xs_geo.py` and `ps_geo.py` share by the whole hex too, and are right to, because
  there the rest of the hex is other units' land.
  **Fixed 2026-09-15 (session `d743fc47-fixes`)** as described. The re-run of `mt_geo.py`
  reproduces these figures exactly (1.030x national; p10 0.68, median 0.99, p90 1.38, Spearman
  0.945; 535,008 of Kontur's 535,078 kept, the other 70 in hexes wholly at sea), and both editions
  were re-scattered: 447 dots and 1 ring at 1:1,000, 41 dots and 6 rings at 1:10,000. The build
  tail was not run, so the map keeps the old placement until the next one. §5 carries the new
  figures.
- **One `note_public` sentence reads badly.** "the census counts 84% of everyone else among the
  104,169 residents without Maltese citizenship": after a sentence about Maltese citizens,
  "everyone else" reads as non-citizens. The figure is right (about 65,900 of the 78,442 who are not
  Roman Catholic). Suggested: "84% of those who are not Roman Catholic are among the 104,169
  residents without Maltese citizenship, who are 36.7% Roman Catholic, ...". Not edited.
- **Screenshot.** Dots on land on both islands, densest round the Grand Harbour and Marsamxett,
  Muslim dots at Il-Marsa and Birżebbuġa; none in the sea at zoom 10.5.
