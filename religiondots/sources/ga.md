# Gabon (`ga`)

Drawn 2026-10-03 by session `fafd1067-ga`, reopened from `blocked` on Anita's priority-holes ruling
(ask/RULINGS.md 2026-09-15) and the Libya ruling (non-nationals estimated into `gap`). Code:
`sources/ga.py`, `sources/ga_geo.py`, `sources/ga_grid.py`, `taxonomy/ga2021.py`,
`countries/ga.py`. Every row is `modelled`.

## 1. What is drawn

Gabonese citizens only, 2,318,365 (RGPL 2026), at 9 provinces, 8 nodes. One national mix from
Afrobarometer rounds 6-9, except Woleu-Ntem's Christian share (92.5%) and Nyanga's None share
(27.6%), the two standouts. Christians divided at the DHS 2019-21 national ratio. Foreign residents,
1,200,256 (34.1%), are the `gap`, `gap_share` 0.34111.

National as drawn: revival churches 37.41%, Catholic 31.54%, none 13.02%, Protestant 9.79%, other
Christian 4.13%, Muslim 1.68%, traditional 1.30%, other 1.13%.

## 2. Census: still nothing on religion (REOPEN on a 2026 thematic volume)

- **RGPL 2013** *Résultats globaux* (UNFPA copy, `data/raw/ga/rgpl2013_resultats_globaux.pdf`, 97
  pp): §2.4 promises religion, prints language (sources.md §11aq). Used here for Tableau 5
  (province totals and densities, PDF p.33) and Tableau 24 (foreigners by province, p.50).
- **RGPL 2026**: certified by the Constitutional Court 2026-08-27, 3,518,621. Published so far: nine
  province totals (Gabon Media Time, "62 % de la population concentrée dans l'Estuaire", the only
  full list found; it sums to the certified total to the person; Estuaire, Haut-Ogooué and
  Ogooué-Maritime are repeated in a second Gabon Media Time piece, Estuaire in Gabonactu and Economie
  Gabon+), sex nationally (1,800,129 women, 1,718,492 men, Gabonactu 2026-08-27), and the
  nationality split nationally (2,318,365 Gabonese, 1,200,256 foreign; Minister of Planning on Gabon
  1ère, 1 Sept 2026, via Gabon Media Time). No department table, no questionnaire, no religion.
  Gabonreview ("le grand bond démographique qui demande des explications") lists what is missing and
  puts the count about a million above the World Bank's and UN's estimates.
- Searched 2026-10-03: the press above. INSTAT's site was not re-opened (the 2026-09-14 scout did).
  The queue row says REOPEN on a 2026 thematic volume with religion.

## 3. Survey: Afrobarometer R6-R9

- 4,790 respondents with an answer (7 non-answers dropped), 1,198-1,200 per round, fieldwork
  2015-09 to 2021-12. All nine provinces every round; REGION codes 1700-1708 stable, labels
  mojibake in R6 (handled by `cm.gkey`). Department column (R6, R7, R9) agrees with REGION for every
  respondent (2013-era names aliased in `DEPT_ALIAS`; R8 has no department).
- Held-out: r = +0.995 against the fitted Gabonese per province; none of 20,000 shufflings reach it.
- **Christian only** 20.9, 24.9, 45.1, 40.4% by round; Roman Catholic 36.7, 35.5, 21.7, 22.5. So the
  card's churches are not drawn from it; grouped to Christian, Muslim, traditional, none, other.
- Quota test passes (6 of 6 wave pairs). **Split-half passes nothing** at 9 provinces: Christian
  +0.433 against a null 95th of +0.467 (p 0.07), None +0.300 (p 0.16), the rest weaker. The spatial
  chi-square is significant for Christian, None, traditional and other, so provinces differ but the
  ranking does not repeat.
- **Standouts** (spec §12, Honduras; `tz.standouts`): Christian tops Woleu-Ntem and None tops Nyanga
  in both halves of all 3 halvings, chi-square < 0.05. Drawn at their own shares there.
- **Compose is not `tz.compose`.** The two standouts are complements; fixing each at the other
  provinces' pooled share everywhere puts Nyanga at Christian 82% + None 28% > 100%, a negative
  residual. `ga.py::compose`: a base mix (each category pooled where it is not a standout,
  renormalised); a standout province keeps its own share and scales the rest of the base to fill.
- Level: pooled against R8-R9 recomposed, worst None -2.1 points (bar 3.5).
- Swap check (traditional vs none by unit, early against late): no swap; Ogooué-Ivindo's None goes
  2.8 to 18.4 and Ogooué-Maritime's 10.5 to 22.7, which is the instability the split-half sees.

## 4. Population base and who is drawn (`ga_geo.py`)

- **RGPL 2026 province totals** are the base: the newest count, and the office's own (the
  geography playbook's rule over COD-PS 2022, a DGS projection from 2013). The count's size is
  disputed in the press; it is the certified figure, and the note says so.
- **Citizens per province is an estimate.** 9 x 2 IPF: rows the 2026 province totals, columns the
  2026 national Gabonese/foreign split, seeded with each province's 2013 foreign share (Tableau 24
  over Tableau 5; 2013's 65,236 undeclared nationality sit on the Gabonese side of the seed). Every
  province's foreign odds rise by x2.587. Result: Estuaire 39.7% foreign (2013 20.3%), Woleu-Ntem
  35.5%, Ogooué-Maritime 32.2%, Haut-Ogooué 29.5%, the rest 8.6-18.2%. One oddity: Haut-Ogooué's
  Gabonese come out at 0.97x their 2013 count. An alternative (2026 foreigners spread by the 2013
  distribution of foreigners) gives Haut-Ogooué 49% foreign, which is worse.
- **Boundaries**: COD-AB `cod-ab-gab` v01 (INC Gabon, valid 2026-06-01), GA01-GA09, names pinned.
  Area witness against Tableau 5's population over density: 0.890-1.056, except Ogooué-Lolo 1.151
  (pinned, `AREA_PINNED`) with Haut-Ogooué 0.926 beside it: COD's shared line sits east of the 2013
  densities, about 2,700-3,800 km2 of forest. Every GeoNames town of 5,000+ falls in its own
  province (Koulamoutou and Lastoursville in Ogooué-Lolo, Moanda and Franceville in Haut-Ogooué),
  and the per-province scaling absorbs whatever forest population moved.

## 5. Placement (`ga_grid.py`)

- Kontur GA 2023-11-01, 8,559 hexes, 2,448,255 people; 0.695 of the 2026 count. Snap within 2 km
  (coast), 14 hexes dropped beyond; 28 hexes that Equatorial Guinea's place layer already holds
  are left to it.
- **Raw Kontur is off by province**: Haut-Ogooué 3.17x its census share (every town 1.5-2.5x its
  GeoNames figure), Ogooué-Lolo 1.82x, Woleu-Ntem 1.51x, Estuaire 0.63x; 25.3% of Kontur's people in
  the wrong province. Scaled per province to the Gabonese count, which removes it.
- Rank witness over 9 provinces +0.933, 136 of 362,880 orderings reach it (bar 1 in 1,000).
- Seat check, five places under 10% within 5 km, reviewed and pinned (`KONTUR_HOLES`): Tsogni and
  Oyam (GeoNames figures that cannot be towns), Gamba (rounded point), Akanda (point on the
  peninsula tip; 491,678 Kontur within 20 km), and **Ntoum, a real thin spot** (5,065 within 5 km
  against GeoNames 62,445). No 2026 department count exists to fill it; its people are placed with
  the rest of Estuaire, mostly Libreville.
- No raw block at Kontur's cap; calibrated densest hex 26,213/km2 (Estuaire).
- Kontur includes the foreign residents, who live mostly in towns, so the citizens' dots lean a
  little urban within each province.

## 6. The Christian split (DHS 2019-21)

- DHS EDSG-III final report FR371 (`data/raw/ga/FR371.pdf`, open), Tableau 3.1 (PDF p.84),
  re-read and asserted on every run. Women 15-49 / men 15-59: Catholic 29.8/29.9, Protestant
  9.9/8.6, Église de réveil 41.9/28.6, other Christian 4.3/3.5, Muslim 8.2/15.2, traditional
  0.4/2.2, other 0.7/1.1, none 4.7/10.9. `Autres nationalités` 14.6/19.0 (an ethnicity row).
- Combined at the 2026 census's 48.8% men. Within Christians: Catholic 38.06%, Protestant 11.81%,
  revival 45.14%, other 4.98%; applied in every province.
- Check: Catholics are 29.85% of everyone in the DHS and 29.10% in the pooled Afrobarometer
  (`CATHOLIC_GAP_MAX` 3 points). The DHS includes foreigners; nothing splits the citizens'
  Christians alone.
- The microdata (would place the churches at province, with nationality to drop foreigners) are
  behind a DHS registration: asks 047 and 048 already put that to Anita. If she registers, Gabon
  2019-21 belongs on the same request.
- Fallback if the split is reversed: map all four Christian rows to bare `christianity`, as
  Cameroon.

## 7. Calls someone might reverse

- Drawing citizens only, foreigners in `gap` (34.1%), rather than everyone at the citizens' mix.
- The 2026 count as base, despite its disputed size; 2013 is the alternative (1,458,464 Gabonese).
- The IPF citizen split per province (one odds factor since 2013).
- The DHS national Christian split applied to citizens.
- `ga.py::compose` for two complementary standouts.

## 8. Not checked

- INSTAT's site for a 2026 release after 2026-09-14; the 1960-1980 censuses; the EGEP 2017 file
  (religion per person, 11 urban/rural strata only, IHSN 7826; not downloaded).
- Whether the RGPL 2026 questionnaire asked religion.

## 9. Review, 2026-10-03 (`fafd1067-rev8`)

Full pass. `ga.csv` matches the note: Woleu-Ntem 92.46% Christian, Nyanga 27.63% none, the other seven
provinces one mix, 2,318,365 in all. `Église de réveil` on `pentecostal` follows `cg2007.py`; `other.ga`
follows the per-source `other.<cc>` pattern (at least forty countries). Citizens-only with the
foreigners in `gap` is the Libya ruling as written, and the note says plainly that it takes most of
Gabon's Muslims off the map. Screenshot: dots on Libreville, Port-Gentil, Franceville, Oyem and the
river towns, none at sea. One fix: the note said the dots "are drawn desaturated"; nothing desaturates
any dot since spec §7 removed it on 2026-09-04, so the sentence now says they disappear when inferred
dots are turned off. `tiles.py --refresh-meta` run.
