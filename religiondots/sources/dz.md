# Algeria (`dz`)

Session `d743fc47-dz`, 2026-09-15, under a supervisor. Built on Anita's Maghreb rulings
(`ask/RULINGS.md`, 2026-09-15 and 2026-09-16) and the scouting in `sources.md`
§maghreb-2026-09-16. Code: `sources/dz.py`, `sources/dz_geo.py`, `sources/dz_grid.py`,
`taxonomy/dz2022.py`, `countries/dz.py`.

## 0. Outcome

Drawn at 48 wilayas, 34,080,030 people (RGPH 2008). Every row `modelled`.

| | people | share |
|---|---:|---:|
| Muslim (`islam`) | 33,932,974 | 99.568% |
| No religion (`unaffiliated`) | 77,257 | 0.227% |
| Christian (`christianity`) | 57,607 | 0.169% |
| Other (`other.dz`, new node) | 12,192 | 0.036% |

Kabylie (Tizi Ouzou, Béjaïa, Bouira) draws 71,389 non-Muslims in 2,735,767 (2.61%); the other 45
wilayas 75,667 (0.24%).

## 1. Sources

- **Arab Barometer**, waves V (2,332 Algerians), VI-1 (998), VI-2 (1,003), VI-3 (1,204) and VII
  (2,162), `data/raw/arabbarometer/`, shared. `Q1012` religion, `Q1` wilaya, `WT` weight,
  `Q1012A`/`Q1012A_MUSLIM`/`Q1012A_CHRISTIAN` follow-ups, `psu` in V and VII.
- **ONS, Annuaire Statistique de l'Algérie no. 31, chapter III** (`ons.dz/IMG/pdf/demographie.pdf`,
  2,223,697 bytes, saved as `data/raw/dz/ons_annuaire31_demographie.pdf`). Table 4: RGPH 2008
  resident population of ordinary and collective households by wilaya, 34,080,030. Table 29:
  density by wilaya at each census. The certificate chain does not verify from here; the fetch
  checks the PDF magic and the `%%EOF` trailer.
- **COD-AB `cod-ab-dza` v01**, ADM1, 48 features (GAUL lineage, UNICEF p-codes, valid from
  2021-01-20). P-codes DZ001-DZ048 run alphabetically, not in official order.
- **Kontur population `DZ` 20231101**, 193,089 hexes, 45,623,911 people.
- **GeoNames `DZ.zip`** (CC BY 4.0), the 56 PPLA/PPLC places, for the placement checks in §6.
- **Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), Algeria
  2020, as an outside level only: Muslims 98.38%, unaffiliated 1.27%, Christians 0.29%, other
  religions 0.04%, Hindus 0.02%.

## 2. The rulings applied

- 2026-09-15: draw the 99.x% Muslim countries with the best evidence for where non-Muslims are.
- 2026-09-16: Algeria's non-Muslims at one share across Kabylie; no IOM DTM; foreigners only where
  the state publishes a count by province, and Algeria's are worth looking for. The Mauritania
  ruling the same day: no presence rings for Jews or Ibadis anywhere in the Maghreb, so none is drawn.
- §14: Kabylie's Christians are a group the state never counted. The ruling settles the grain
  (Kabylie as one unit), so nothing finer is drawn and nothing was escalated.

## 3. The survey: pool, weights, card, contradictions

**The card.** `dz.py::card` reads each wave's `Q1012` labels. II, III and IV offer no box for
having no religion; V offers `Atheist`; VI-1, VI-2, VI-3 and VII offer `No religion`. In V-VII that
box is 19 of Algeria's 36 non-Muslim answers, so II-IV (3,636 answers, one non-Muslim: a wave IV
Christian in Tizi Ouzou) would mix two questionnaires into the level. They are in `OMIT` with that
reason. Wave I has no subnational column. Wave VIII has no Algeria.

**Atheist and No religion are one box.** Code 4 on their own cards, never both on one card;
`RECODE` merges them. `refused` (V) and `Refused` (VI-1) are merged into `Refused to answer` and
dropped (23).

**Wave VI's weights.** Algeria's `WT` averages 0.8396, 0.7801 and 0.8495 in the three parts, and
every other country in those files 0.99-1.00, so `ab.load` refused them. Labelled `Weight for the
probability of selection`; a constant does not change a share within the wave, only the wave's
weight in the pool. `ab.load(rescale_weights=)` was added (shared code; refuses a stale entry and a
wave outside the pool) and divides each by its mean. No other country passes it.

**Contradictions.** Wave V asks a denomination follow-up of everyone naming a religion, on one list
of Muslim schools and churches. Six answers contradict `Q1012`: the three `Jewish` answers give
Orthodox, Shafi'i and Sunni, and three `Christian` answers give Shia, Sunni and `Just a Muslim`.
Which item was mis-keyed cannot be told; all six are dropped (`contradictions`, the list asserted).
Drawn as recorded, the three Jewish answers alone would have been about 60,000 Algerian Jews. No
Jewish answer remains. Wave VII's follow-ups are consistent.

**Geography.** 62 `Q1` labels fold to 48 wilayas (`SPELLINGS`). 17 respondents in VI-2 and VI-3 have
`Don't know` for a wilaya, all Muslim, and leave the geography. The code witness: V's codes are
10000 + the official number and VI's 1000 +; 5,494 respondents agree and none disagree. VII uses
its own order and is checked by name only. Held-out: r = +0.981 between respondent shares and the
2008 census over 48 wilayas, best of 20,000 random pairings +0.672 (thinnest Tindouf 0.32x, fullest
Tissemsilt 1.18x). Quota test: p = 1 over 5 comparable pairs. Every wilaya is sampled in the pool
(VI samples all 48), so no wilaya is drawn without a respondent in its unit.

Pool as drawn: 7,699 answered, 7,676 after refusals, 7,653 after the contradictions and the
unknown wilayas (`NOTE`, asserted).

## 4. Kabylie, and nothing else apart

| wave | Kabylie | rest |
|---|---:|---:|
| V | 8 of 196 | 9 of 2,116 |
| VI-1 | 2 of 91 | 2 of 905 |
| VI-2 | 1 of 96 | 1 of 896 |
| VI-3 | 0 of 97 | 0 of 1,097 |
| VII | 5 of 133 | 8 of 2,026 |

- Pooled, 16 non-Muslims in Kabylie against 2.8 expected, exact within wave (hypergeometric per
  wave, convolved), P = 2.2e-09. Wave V alone: 8 against 1.4, P = 2.9e-05. VI-1 to VII alone: 8
  against 1.4, P = 2.2e-05. `kabylie_test` stops if either half fails.
- The largest PSU holds 3 of the 16 (19%, cap 50%); they come from all three wilayas and 9 PSUs.
- Weighted: **Kabylie 2.61%, the rest 0.24%.** The ruling's figures (all waves, unweighted) were
  2.10% and 0.23%.
- **Standouts outside Kabylie** (`standouts`): each wilaya with two or more non-Muslims, against
  the other 44, in both halves, Bonferroni over 3 (bar P < 0.0167). Algiers 4 against 1.0 (P
  0.011) then 2 against 1.2 (0.347); Sétif 2 against 0.6 (0.112) then 3 against 0.8 (0.040); Tipaza
  0 then 2 against 0.3 (0.031). None stands apart in both, so the 45 share one figure.
- **Composition.** 36 answers can not carry a mix per unit, so each unit's non-Muslim share is
  split at the pooled national weighted mix: no religion 52.5%, Christian 39.2%, other 8.3%
  (Kabylie on its own 55/42/4 on 16 answers, the rest 50/37/13 on 20).
- Witness, not drawn: wave VII's ethnicity item (`Q1012B`) has 9 of 13 non-Muslims
  Amazigh/Berber, including those in Algiers, Sétif and Tipaza.
- Level: the survey reads 0.44% non-Muslim; Pew 2020 has 1.27% unaffiliated and 0.29% Christian.
  The note calls the drawn figures floors.

## 5. Population base: RGPH 2008, unscaled

The sixth census (25 September to 16 October 2022) has published no wilaya results: La Nation,
27 August 2023 (*"Les résultats du Recensement général de la population (RGPH) de 2022 n'ont,
jusqu'à présent, pas été publiés"*), and ONS's home page, as fetched 2026-09-15 with an October 2025
banner, links only the 2008 results with a commented-out `rgph2020` link. ONS's collections page
for population is 2008 throughout.

Drawn as counted, not scaled. Scaling would need ONS's own national estimate, which was found only
in press summaries (46.3 million at 1 July 2023, 47.4 million at 1 January 2025), not opened, and a
national factor would still leave 2008's distribution. Kontur 2023 reads 1.333x nationally and
puts Kabylie at 6.98% of Algeria against 8.03% in 2008 (Table 4's own growth rates: Tizi Ouzou
0.2%, Béjaïa 0.6%, Bouira 1.0% a year 1998-2008 against 1.6%). So Kabylie's 71,389 is likely high
against today's Kabylie; the note says the share is probably smaller now.

## 6. Geography

**Join** (`dz_geo.py`): an authored table, official number -> COD p-code -> ONS spelling. Witnesses:
it covers COD's 48 p-codes; Table 4 and Table 29 both print the wilayas in official order and match
the table at every row; ONS area (Table 4 population over Table 29 density) against COD's area,
rho +0.984 with no permutation of 5,000 near it (best +0.496); the four wilayas under one person per
km2 (Illizi, Tamanrasset, Tindouf, Adrar) are COD's four sparsest. Table 4's sexes miss the printed
total by one person in 13 rows; the total is drawn.

**Five area disagreements, not settled** (`AREA_DISAGREE`): COD over ONS is Mila 0.37x, Djelfa
0.49x, El Oued 0.81x, Tizi Ouzou 0.83x, Chlef 0.84x. The rank test pins the pairing, so these are
either where wilaya lines run or Table 29's denominators; a sixth stops the build.

**Kontur** (`dz_grid.py`): 903 hexes (194,829 people) fall outside every wilaya. Per wilaya, Kontur
over census at the national ratio runs from Béchar 0.31 and Tlemcen 0.69 to Naâma 1.50 and Tindouf
1.67. Two fixes, both checked against GeoNames:

- **Béchar is a hole in Kontur.** 448 Kontur people within 5 km of GeoNames' Béchar (165,241); every
  other seat of 50,000+ has at least 0.27 (El Bayadh). `seat_check` asserts Béchar is the only seat
  under 0.10. A 3 km disc on GeoNames' point takes the wilaya's shortfall at the national ratio,
  247,688 of 359,994; before it, Béchar's dots would have gone to Abadla, Beni Abbès and the Aïn
  Sefra road.
- **Tindouf's heaviest hexes are 25-50 km south-east of the town, where the Sahrawi refugee camps
  are.** Kontur holds 109,299 in the wilaya against 49,149 counted. Only hexes within 15 km of
  GeoNames' Tindouf are kept (39,517, 0.60 of the census at the national ratio, band 0.5-2.0);
  69,782 are dropped.

`kontur_cap.py dz`: no stops. Scatter: no unplaced unit, 0.07% of placement area clipped as sea.

## 7. Not drawn, and REOPEN

- **Foreigners by wilaya. REOPEN on RGPH 2022's detailed results.** Scouted later on 2026-09-15 by
  `d743fc47-scout-dzfor` (`sources.md` §scout-2026-09-15-algeria-foreigners): the 2008 and 2022 forms
  both ask nationality, and no ONS release for 2008 or 1998 tabulates it. The builder's own search,
  kept as written: Anita expects a table (RGPH 2008 or 2022). Searched 2026-09-15:
  two web searches (French, RGPH 2008 foreign population by wilaya and nationality), ONS's
  population collection page (no `étranger` anywhere), ONS's home page. Not opened: ONS
  *Principaux résultats de l'exploitation exhaustive* (Données statistiques no. 527, 28 pages per
  wilaya, 2009), the `collections statistiques.htm` index on `ons.dz`, and the ONS `.doc` version of
  chapter III. The layer would be foreigners by wilaya times the national nationality mix times
  each origin's composition (`taxonomy/origin_religion.py`), under the ruling's small-or-predictable
  test.
- **Ibadis.** 7 Ibadi or Mozabite answers on the follow-up in the pool (V: Algiers, Batna, Djelfa;
  VII: Algiers, Mila, Ouargla 2), none in Ghardaïa. No magnitude anywhere. No ring: the Maghreb gets none (§2).
- **Afrobarometer**, witness only. R5 (1,204, 2013): region labels wrong (`playbooks/afrobarometer.md`).
  R6 (1,200, 2015): eight regions, no Christian or None answer, 14 `Other` in the South Eastern
  Region, likely Ibadis on a card with no Ibadi box. One `None` in the two rounds; neither adds to the
  non-Muslim share.
- **Jewish.** No answer survives §3's check.

## 8. Calls someone might reverse

1. Pool V-VII only: II-IV out on the card. Pooling all eight waves would lower both units' shares
   (the ruling's own 2.10% and 0.23%).
2. The six contradictory answers dropped rather than recoded (the Orthodox "Jewish" could be read as
   Christian).
3. 2008 census counts drawn unscaled.
4. One national mix inside both units, rather than each unit's own mix.
5. The Béchar disc and the Tindouf 15 km cut.
6. `Atheist` and `No religion` merged as one box.

## 9. Review, 2026-09-15 (session `d743fc47-rev12`)

- **The shared loader change moves no other country, measured.** `ab.load` as committed at HEAD
  (before `rescale_weights`) and as in the working tree, each called with exactly the arguments
  `eg.py`, `jo.py`, `iq.py` and `ye.py` pass, returned identical frames (exact compare) and
  identical printed logs: Egypt 6,840 rows, Jordan 14,917, Iraq 8,343, Yemen 3,600. `ma.py` does
  not pass the option either.
- **Rings have been ruled on since this was written.** `ask/RULINGS.md`, 2026-09-16, the Mauritania
  line: "no presence rings for Jews or Ibadis anywhere in the Maghreb ("lets not do rings")". §2
  and §7 here, the `note` in `countries/dz.py` ("presence rings (not ruled on)") and REVIEW
  `Muslim` in `taxonomy/dz2022.py` ("leaves presence rings undecided") predate it. Nothing drawn
  changes. The builder's text is left as written, and ask 030 is left for whoever closes asks. Since updated in all of them (cb8b206e-maint, 2026-09-15); ask 030 not closed.
- **Fixed:** note_public's last paragraph opened on a bold clause ending in a comma. The viewer
  makes a paragraph only of bold ending in `.`, `!` or `?` (`index.html`), so it rendered as a bold
  clause inside the Ibadi paragraph. Now "**The survey interviewed adults, and the shares are
  applied to everyone.** The people are the 2008 count, and Kabylie grew more slowly...". Reaches
  the viewer on the next build tail. check_md clean before and after; it does not catch this shape.
- **Wave V's keying, for the record, no change.** Five of the six contradictions read as Muslims
  keyed into another religion (three Christian and two Jewish answers naming a Muslim school), about
  1 in 460 of wave V. The check can see only a mis-key into a religion that gets the follow-up. The
  same slip into `Atheist` would pass, and wave V holds 12 of the pool's 19 no-religion answers.
  Wave VII's follow-ups show no slips, so this is wave V's fieldwork. It cannot touch Kabylie's
  contrast, and it runs against the floor reading of the level.

## 10. Sahrawi camps sized for the bar, ask 033 sweep (cb8b206e-gap, 2026-09-15)

On Anita's ruling (`ask/RULINGS.md` 2026-09-15); the sweep's record is `sources.md`
§gapsweep-2026-09-15.

- **Who.** The Sahrawi refugee camps south-east of Tindouf. RGPH 2008 counts 49,149 in Tindouf
  wilaya, Kontur 2023 holds 109,299 there, and §6 drops the 69,782 in hexes beyond 15 km of the
  town. So the camps are in neither the base nor the dots. Algeria's other foreigners (95,000 in
  2008) are inside the census count and drawn at the survey's shares, as Tunisia's are, so they are
  not a hole.
- **Figure.** *Sahrawi Refugees Response Plan 2024-2025* (UN in Algeria,
  `algeria.un.org/sites/default/files/2024-01/SRRP - English.pdf`, opened 2026-09-15). PDF page 7:
  "Based on an interagency study conducted in 2018, the Inter Sector Working Group like the host
  Government estimates that 173,600 people live in the camps." The planning table (PDF page 11,
  printed 14) uses 173,600 for 2024 and for end 2025. Its footnote 4: UNHCR uses 90,000 for the
  "most vulnerable refugees", which is a count for assistance, not the camps' population.
- **Share.** 173,600 / (34,080,030 + 173,600) = **0.00507**. The base is 2008 and the figure is from
  2018, so the two years differ; the note gives the study's year.
- **Note.** A paragraph added at the end of `note_public`; `gap` and `gap_share` set; the internal
  `note` points here. Nothing rebuilt.
