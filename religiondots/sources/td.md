# Chad — RGPH2 2009, religion by région

**Drawn 2026-09-14 at 22 régions** (sources.md §td-2026-09-14), after Anita ruled on
`ask/answered/017-td`: draw it as built. 6 categories, 10,941,682 censused people, 10,939 dots
at 1:1,000. Built to the mapping by session `f95259a4-td`, registered and scattered by
`d743fc47-td`; the entry is `countries/td.py`.

- `sources/td.py` -> `data/normalized/td.csv` (the PDF is `data/raw/td/rgph2_etat_structures.pdf`)
- `sources/td_geo.py` -> `data/geo/td/td_regions.gpkg`, `td_hexes.gpkg`, `td_lookup.csv`
  (COD-AB départements rebuilt to 2009 + Kontur 400 m)
- `taxonomy/td2009.py` -> the mapping; `other.td` is in `taxonomy/branches.py`
- `data/raw/td/rgph2_questionnaire.pdf`, `rgph2_resultats_sous_prefecture.pdf` (scan),
  `rgph2_manuel_agent.pdf` (the post-enumeration survey's manual, not the census's)
- sources.md **§11aq** is the scout's row; **§td-2026-09-14** is the build.

```
python sources/td.py     --fetch
python sources/td_geo.py --fetch
```

## 1. What INSEED publishes

| release | religion | tier |
|---|---|---|
| **RGPH2 2009, *Analyse thématique: État et structures de la population*** (Nov 2013, 210 pp) | **Tableau 5.07, région x religion, % to one decimal, with each région's population**; 5.06 counts by milieu and sex; 5.08 shares 1993 and 2009 | **22 régions** |
| RGPH2 *Résultats globaux définitifs* (158 pp, scanned) | no religion table (§11aq) | |
| RGPH2 *Résultats définitifs par sous-préfecture* (122 pp, scanned) | population only, by région, département, sous-préfecture | |
| RGPH2 microdata (NADA `anad.inseed.td` catalog 26) | B12 to sous-préfecture | data enclave, authorisation from the Director General |
| RGPH1 1993 synthesis | religion nationally, by sex and ethnic group (§11aq) | national |
| UNSD Demographic Yearbook table 28 | **Chad is absent** | |
| ECOSIT4 2018 (catalog 18) | `religion` in five codes, 23 provinces; listed public, the download page returned an empty body to a script (§11aq) | survey, not used |

The structure volume is only on the retired jdownloads store of the old `inseed.td`, through
Wayback's 2020-06-08 capture (`id_` form). Checked: `%PDF-1.5`, `%%EOF` 5 bytes from the end,
210 pages, text layer present.

## 2. The construction

    seed           Tableau 5.07 shares x Tableau 5.07 région populations
    row margins    Tableau 5.07 `Effectif`, 22 régions, sum 10,941,682
    column margins Tableau 5.06 `Ensemble`, six religions, sum 10,941,682

Both margins are the office's own counts of the same universe, so the seed is raked to both
(iterative proportional fitting) and integerised by largest remainder within each région.
Printed `0,0` cells stay zero. Result: every région equals its printed population, every
religion is within 2 people of 5.06, 12 of the 132 cells no longer round to the printed share,
and the largest move is 0.153 points. Guinea (§9dh) rescaled to UNSD instead; Chad has no UNSD
row, and here both margins are printed.

## 3. The checks (all in `sources/td.py`)

| check | result |
|---|---|
| Tableau 5.07 parsed off the page = the transcription | 22 régions + TCHAD, identical |
| Tableau 5.06, three blocks, parsed = transcribed | identical |
| every 5.07 row sums to 100 | within 0.10 pp |
| régions sum to the censused population | 10,941,682 exactly |
| 5.06's six Ensemble counts sum to it | exactly |
| 5.06's Urbain and Rural blocks | each +1 against its printed total; urban + rural per religion misses Ensemble by -72 to +44, men + women by -5 to +5 (separately rounded tabulations; bounded, not fixed) |
| shares x populations against 5.06 | -0.29% (no religion) to +0.19% (other); the four large religions within 0.08% |
| population-weighted shares against the printed TCHAD row | all within 0.06 pp |
| Tableau 2.02, 2.04, 2.13 figures on their pages | yes; see §5 for the footnote |

## 4. The questionnaire

The RGPH2 household form (NADA catalog 26, download 160, p3) asks **B12 RELIGION** of every
member: **ANI 1, CAT 2, MUS 3, PRO 4, AUT 5, SAN 6**, legend *ANImiste, CATholique, MUSulman,
PROtestant, AUTres, SANs*. Six boxes, the table's six columns in the same order. **Animists had
their own box**, so `Sans religion` is not the Mozambique or Laos case (spec §6.3a-ii): nothing
on the form sends traditional religion into it, and it maps to `unaffiliated`. The report
defines religion as "l'ensemble des croyances et pratiques qui régissent les rapports entre
l'Homme et Dieu" (p32). No enumerator instructions for B12 were found: the NADA item titled
*Manuel de l'agent enquêteur* is the post-enumeration survey's manual.

## 5. Universe and gap

- **In the table:** the censused population, 10,941,682, including 320,010 in collective
  households (Tableau 2.04), which the report says are mostly refugee camps, military camps and
  prisons. 235,183 foreigners were counted in refugee camps (text of 5.1.1), nearly all
  Sudanese, in the east.
- **Not in any table: 98,191 estimated** (Tableau 2.02), people in zones enumerators could not
  reach. `gap_share` = 98,191 / 11,039,873 = **0.008894**.
- **The footnote misprints Sila.** It names rural Sila 91,011 and Zouar (Tibesti) 4,180, which
  add to 95,191. Tableau 2.13 gives Sila 387,461 and Tibesti 25,483 in total against 293,450 and
  21,303 censused, so the estimated are **94,011** and 4,180; the sous-préfecture volume's
  Tableau 02 (p16) prints 94,011 as well. **24.3% of Sila was not counted.**
- **Placement caveat, not fixed.** Tableau 02 shows the uncounted Sila people all in Kimiti
  département: Moudeïna (22,992) and Tissi (24,036) sous-préfectures had nobody censused, Adé
  29,262 of 41,675 estimated, Mogororo 16,721, Goz-Béïda 1,000. COD-AB stops at départements, so
  Sila's 293,450 are spread over Kontur hexes that include those two sous-préfectures.

## 6. Boundaries: 22 régions of 2009 on COD-AB v01 (valid_on 2025-02-12)

COD's 23 provinces are rebuilt from its **70 départements**:

- **Ennedi** = Ennedi Est (Amdjarass, Wadi Hawar) + Ennedi Ouest (Fada, Mourtcha); the 2012
  split is in the structure volume's own footnote (p32).
- **Djourf Al Ahmar (TCD1402) goes back to Sila and Abdi (TCD2102) back to Ouaddaï.** On COD they
  sit the other way round. The sous-préfecture volume lists 2009 Sila as Kimiti + Djourouf Al
  Amar (p15) and 2009 Ouaddaï as Ouara + Abdi + Assoungha (p13). Nothing in the structure volume
  mentions it; **the 2009 areas give it away**: Tableau 2.13's population over density puts
  Ouaddaï at 30,049 km² and Sila at 35,876, against COD's provinces at 40,663 and 24,835, and
  29,771 and 35,727 after the move. A name join and every totals check pass on the wrong
  version.
- `Barh El Gazal` (census) = `BARH EL GAZEL` (COD), the one alias.

The area witness for the other 20, COD against 2009 (`td_geo.py` prints it): 14 within their
rounding interval widened 3%. Outside: **Borkou 0.79x** and Tibesti 0.84x (densities of 0.5 and
0.1 are too coarse to test, and 93,584 and 21,303 people, 98-99% Muslim, so a Borkou/Tibesti
line that moved would shift a few hundred Christians in the desert at most); **Lac 0.90x** (open
water of the lake, which has shrunk); **N'Djaména 436 km² against 500**; Hadjer Lamis 1.04x and
Mayo Kebbi Ouest 0.97x (digitising). None was acted on.

**Kontur** (2023-11) is 1.67x the 2009 count nationally, used only within régions. Low: Wadi
Fira 0.87x, Logone Occidental 0.89x; high: Salamat 3.51x, Borkou 2.95x, Ennedi 2.90x, Kanem
2.78x. Sila is 2.64x, which includes the uncounted Kimiti sous-préfectures and the refugee
arrivals since 2023. `python kontur_cap.py td`: no block at the density limit, no rows needed.

## 7. What the table shows

The seven southern régions (Logone Occidental, Logone Oriental, Mandoul, Mayo Kebbi Est, Mayo
Kebbi Ouest, Moyen Chari, Tandjilé) hold 90.3% of the Catholics, 88.6% of the Protestants,
94.4% of the animists, 94.0% of those with no religion and 91.4% of `Autres`, and 9.0% of the
Muslims. Twelve of the other fourteen régions are 98% Muslim or more; Guéra is 95.6% and Chari
Baguirmi 84.1%. **Mayo Kebbi Est is 32.0% animist**; **Mayo Kebbi Ouest is 15.2% no religion**;
Moyen Chari is 25.4% Muslim; N'Djaména is 70.7% Muslim and 27.9% Christian.

Tableau 5.08, 1993 to 2009: Muslim 55.0 to 58.4, Protestant 14.4 to 16.1, Catholic 20.5 to 18.5,
animist 7.5 to 4.0, no religion 3.1 to 2.4, other 0.5 both years. The report attributes the
Protestant growth to new revival churches.

## 8. §14 — held, `ask/017-td`

USCIRF's October 2025 issue update says JAS-Boko Haram and ISWAP victimised civilian
communities "including Christians" in Lac Province from 2020, and that in 2025 they and violent
Fulani herder groups "continue to target religious communities in Chad". The State Department's
2023 religious freedom report records armed groups attacking Christian communities in Logone
Oriental on 8 May 2023, killing 17 including a pastor and 12 congregants. Burkina Faso and Mali
were held for Anita on the same question the same day. The builder's recommendation, the
figures and the options are in the ask.

**Ruled 2026-09-14: draw Chad at its 22 régions as built** (option 1, "yes draww it"). Burkina
Faso and Mali were ruled the same way in ask 018. The ruling covers the published grain; nothing
finer than the région was asked or decided.

## 9. Gotchas

- **COD's current provinces are not 2009's régions**, two départements apart. Assert areas, not
  only names and totals.
- **Tableau 2.02's footnote misprints Sila's estimated population** (91,011 for 94,011).
- **Tableau 5.06's blocks do not add to the person** (urban + rural vs Ensemble, up to 72).
  Use the Ensemble block; the check bounds the rest.
- **Tableau 5.08's title says "entre 1964 et 2009" and its columns are 2009 and 1993.**
- The sous-préfecture results and the *Résultats globaux* are scans; render them.
- The report writes `N’Djaména` with a curly apostrophe in 5.07 and `N’Djamena` in 2.13; the
  folding join handles both.

## 10. Terms

INSEED's thematic reports and questionnaire are public downloads with no licence text; the
structure volume is cited from its Wayback capture. The census microdata's NADA terms require
the Director General's authorisation and were not used. USCIRF and State Department reports are
US government works.

## 11. Review, 2026-09-14 (`d743fc47-rev1`)

Read from `data/normalized/td.csv`, `taxonomy/td2009.py` and `countries/td.py` before this file.
Nothing to change, nothing to rebuild, no ask.

- **`note_public` recomputes off the CSV.** The seven southern régions hold 90.3% of the
  Catholics, 88.6% of the Protestants and 94.4% of the animists; Mayo Kebbi Est is 32.0% animist;
  N'Djaména is 70.7% Muslim and 27.9% Christian; twelve of the other fourteen régions are 98% or
  more. The note's top of "99.5%" is Kanem after raking (printed 99.4), so it matches what the map
  draws. The `other.td` legend note's 91% is 91.4%. `grain` and `gap_share` check.
- **The mapping follows precedent.** `Protestante`, `Animiste`, `Sans religion` and the residual
  box go where cg2007, cf2003, ci2021 and gn2014 send the same French boxes. `other.td` is a
  per-source residual under spec §3.11 like `other.gn` and `other.cg`, declared in
  `religions.json`; it shows on Chad's legend only.
- **The gap is not biased in a way that matters.** The 98,191 not counted are in Sila and Tibesti,
  both 98-99% Muslim, so leaving them out understates Chad's Muslim share by about 0.4 points.
  Spreading Sila's counted people over the uncounted Kimiti sous-préfectures (§5) moves no
  religion share, since Sila is 99.0% Muslim throughout.
- §14 is settled by ask 017.
- **Screenshot at 1:1,000:** dots on land only, the Christian south and the Muslim north and east
  where the table has them, the desert régions sparse. Nothing looks off.
