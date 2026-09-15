# Senegal — 1988 census, religion and Sufi brotherhood by région

**Drawn 2026-09-15** (session `d743fc47-sn`). 12 units (the nine other régions of 1988 and
Diourbel's three départements), 7 nodes, 6,896,808 residents of ordinary households, every row
`measured`. 6,893 dots at 1:1,000, 686 at 1:10,000.

- `sources/sn.py` -> `data/normalized/sn.csv` (three PDFs in `data/raw/sn/`, pinned by size and
  SHA-1)
- `sources/sn_geo.py` -> `data/geo/sn/sn_units.gpkg`, `sn_hexes.gpkg`, `sn_lookup.csv` (COD-AB
  Senegal v02 by pcode, Kontur 400 m, 2023-11)
- `taxonomy/sn1988.py` -> the mapping; `countries/sn.py` -> the entry; `taxonomy/branches.py`
  gains `islam.qadiriyya`, `islam.tijaniyya`, `islam.mouride`, `islam.layene` and `other.sn`
- `tools/check_mapping.py` `DEFAULT_LEVELS["sn"] = ["region", "department"]`
- sources.md **§sn-2026-09-15** is the summary; **§11aq** was the scout's row. Ask **029** is
  whether the brotherhoods stay four legend rows.

```
python sources/sn.py     --fetch
python sources/sn_geo.py --fetch
```

## 1. What exists

| release | religion | tier |
|---|---|---|
| **RGPH 1988, *résultats définitifs*** (DPS, June 1993, 76 pp; IREDA `sen-1988-rec-o1_rapport_resultats_definitifs.pdf`) | **Tableau 1.15, région x Musulmans, Khadriya, Layène, Mouride, Tidiane, Autres mus., Chrétiens, Autres; % of residents, one decimal** (printed p28, PDF p30). Also on `ansd.sn` as *Chapitre 1 - ETAT DE LA POPULATION.pdf* | **10 régions** |
| **RGPH 1988, Diourbel regional analysis report** (IREDA `sen-1988-rec-o2_diourbel.pdf`, 69 pp) | **Tableau 1.12, all eight codes by région, milieu and département, in counts** (printed p30, PDF p32) | **3 départements** |
| The other nine 1988 regional reports | same table presumably; East View Global Census Archive series `2540885YO`, institution-gated (§11aq). Not on IREDA: a search for `sen-1988-rec` finds only the national report, Diourbel and the enumerator manual, and the resource directory answers 403 | |
| *Manuel de l'agent recenseur* 1988 (IREDA `sen-1988-rec-m1_manuel_recenseur.pdf`) | P11's instructions (p26) | |
| RGPH-3 2002 | same item with Protestant apart (`QUESTRGPH2.pdf`); the 2013 Ziguinchor report quotes 2002 regional reports with département religion, none located (§11aq) | |
| RGPHAE 2013, RGPH-5 2023 | asked (2023: sixteen codes, reformist movements, Ahmadiyya, Shia); no table below the nation found; NADA microdata `licensed` (§11aq) | |
| UNSD Demographic Yearbook table 28 | absent | |
| Afrobarometer R4-R9 | 14 regions; places the orders (Mouride +0.95, Tijaniya +0.91), but the unnamed-Muslim share runs 22.7-64.2% by round (§11aq) | not used |

**Why 1988 and not the survey on a modern margin.** No census after 1988 publishes the orders at
any level, so a modern build would take their level from the survey, whose Muslim catch-all swings
by 40 points between rounds. The survey agrees with 1988 on the pattern (Diourbel the most Mouride,
Matam the most Tijani, Ziguinchor the most Christian), which is the scout's measurement, not a
check run here.

## 2. The form

P11 *Religion*, every person (national report PDF p75; manual p26): *"Pour les musulmans, encerclez
la confrérie déclarée"*: 1 KH Khadr, 2 LA Layène, 3 MO Mouride, 4 TI Tidiane, 5 AM *musulmans qui
n'appartiennent pas à ces confréries*; 6 CA Catholiques; 7 AC *autres chrétiens (protestants,
luthériens, témoins de Jéhovah etc.)*; 8 AR *autres (Juifs, Bouddhistes, Animistes etc.)*. No code
for no religion or for no answer. The report's prose (printed p27) calls the other religions
*animisme principalement*.

## 3. The construction

    shares        Tableau 1.15, région x 8 columns, one decimal        national report p28
    denominators  Tableau 1.2, résident population by région, counts   national report p9
    Diourbel      Tableau 1.12, département x 8 codes, counts          Diourbel report p30
    margins       Tableau 1.2 (départements) and 1.12's région column  Diourbel report p9, p30

Nine régions: each row's seven leaf shares divided by their printed sum (99.9 to 100.1) times the
région's population, rounded by largest remainder so every row closes. Diourbel: the three
départements' eight counts raked to Tableau 1.2's département populations and Tableau 1.12's
région column (the département columns are 1,387 short), then rounded within each département.
The units sum to 6,896,808 exactly. No per-religion rescale: there is no national count in persons,
and the printed Ensemble row is not the sum of the régions (§4).

`gap` is the "35000 personnes vivant dans la population comptée à part" of p8 (barracks, gendarmes,
fire brigades, prisons), 0.50% of 6,931,808, hand-written: no table carries them, so
`tools/gap_share.py` refuses.

## 4. The checks (`sources/sn.py::check`, `sources/sn_geo.py`)

| check | result |
|---|---|
| PDFs pinned | national 3,794,341 bytes `KOBVKWE7PFQBHNON3OKMWGNQ2G2NUHYL`; Diourbel 3,192,222 `S7QZ763272HTG2X65A6ZFE632N7PNN2Z`; manual 1,983,511 `NTJS2ZPMA34CIUABACZULSHA3S6DXFCO`; pages 76, 69, 46 |
| form and manual | P11's codes on the form's text layer; the manual's code 8 names animists and Jews |
| Tableau 1.15 off the text layer = transcription | 11 rows, `-` cells included (the text layer keeps them as their own lines) |
| rows close | orders sum to Musulmans within 0.10; Musulmans + Chrétiens + Autres = 100 within 0.10 |
| populations | Tableau 1.2's ten régions on p9, summing to Tableau 1.1's 6,896,808 |
| **Diourbel's row** | printed Khadriya `-`, Layène 3.7; the Diourbel report counts 3.70% and 0.04%; swapped, every cell agrees to the rounding, and the Ensemble Layène 0.6 comes back (0.60 swapped, 0.93 as printed) |
| **Ensemble row** | not reproduced: Musulmans 94.41 against 93.8, Khadriya 11.69 / 10.9, Mouride 29.70 / 30.1, Autres mus. 5.12 / 4.8, Chrétiens 4.46 / 4.3, Autres 1.15 / 1.6; only Layène and Tidiane within 0.15 (pinned). The row sums to 99.7 |
| Tableau 1.12 | 66 counts off the text layer = transcription; Muslim, Christian and grand totals close in all six columns; rural + urban = région; région 619,245 = national |
| Diourbel populations | men + women = each département (Tableau 1.2A), summing to 619,245; Tableau 1.12's départements short by 268 / 491 / 628 (pinned) |
| COD-AB | 14 régions and 46 départements, every pcode's name and parent asserted; the 1988 régions use all 14 once |
| areas | see §5 |
| Kontur | 63,855 hexes, 17,910,039 people; 604 offshore centroids snapped within 500 m (228,815 people), 84 dropped (15,710, 0.088%); ratio to 1988 2.595 |
| `kontur_cap.py sn` | no block at the cap |

## 5. Geography

The 1988 régions are today's by succession: Saint-Louis = Saint-Louis + Matam (2002), Tambacounda =
Tambacounda + Kédougou, Kaolack = Kaolack + Kaffrine, Kolda = Kolda + Sédhiou (all 2008). Diourbel
is its three départements, ADM2 SN0201-SN0203. **Areas against 1988's** (national Tableau 1.2's
whole-number densities bound each région's area; the Diourbel report prints its départements'):

| unit | COD km2 | 1988 | off |
|---|---:|---:|---:|
| Dakar | 542 | 550 | -1.5% |
| Ziguinchor | 7,331 | 7,309-7,446 | 0 |
| Saint-Louis | 47,860 | 42,599-45,537 | **+5.1%** |
| Tambacounda | 59,479 | 59,382-70,179 | 0 |
| Kaolack | 16,378 | 15,753-16,065 | +2.0% |
| Thiès | 6,578 | 6,559-6,605 | 0 |
| Louga | 25,636 | 28,004-29,702 | **-8.5%** |
| Fatick | 7,011 | 7,902-8,027 | **-11.3%** |
| Kolda | 21,118 | 20,766-21,521 | 0 |
| Bambey | 1,334 | 1,351 | -1.3% |
| Diourbel (dép.) | 1,287 | 1,175 | **+9.5%** |
| Mbacké | 2,242 | 1,833 | **+22.3%** |

The total agrees (196,795 against 1988's 197,000-odd). Louga's loss matches Saint-Louis-with-Matam's
gain, and Fatick's loss sits beside Diourbel's gain (the région is 4,862 km2 against 4,359), so
border land has most likely moved between régions since 1988; it may also be remeasurement. No 1988
boundary file was found to settle it. Not checked: geoBoundaries has no historical Senegal; the
1988 *répertoires des localités* (ten volumes, 1990-91) would place villages and could rebuild the
lines, which is far more work than the grain is worth.

**Placement is 2023's.** Kontur over the 1988 count runs 1.82x (Ziguinchor) to 5.18x (Mbacké, Touba),
so inside a unit the dots follow today's population. No unit's count moves.

## 6. Mapping calls (`taxonomy/sn1988.py` REVIEW)

- **The four brotherhoods are four new nodes** under `islam`, in LINEAGE's "Sufi orders" group after
  Bektashi: the form asks the order, not the school, and the Layène's Mahdist founding makes
  `islam.sunni` arguable for one of them. Ask 029 asks Anita whether four legend rows stay.
- `Autres mus.` -> `islam`: code 5, Muslims of no listed brotherhood (reformists, Shia, others).
- `Chrétiens` -> `christianity`; Diourbel's `CATHOLIQUE` and `AUTRES CHRETIENS` fold into it too, so
  one région's four Catholic dots do not add a legend row.
- `Autres` -> new `other.sn`, not traditional religion: code 8 also holds every other religion and,
  with no none or no-answer code, possibly people with none. The report's *animisme principalement*
  is in the node's text. `check_no_religion.py` has nothing to say (no box is a no-religion answer).
- Kaolack's Layène prints `-` and is drawn as zero (under 0.05%).

## 7. §14

Nothing raised. The 1988 form has no code for a reformist movement, Salafists, Ahmadis or Shia;
the scout's note (§11aq) was to raise before drawing any of those from 2023. The brotherhoods are
public, large and not a persecution axis. Ziguinchor's Christians and animists are drawn at région
tier, the published grain; the Casamance conflict is separatist, not religious.

## 8. What would improve it

- **ANSD publishing RGPH-5 2023's religion by région or département** (sixteen codes). That would
  replace this build, and its reformist, Ahmadiyya and Shia codes would need a §14 look first.
- The other nine 1988 regional reports (East View, gated) would take every région to département,
  the grain Diourbel has here.
- 2002's regional reports (département religion is quoted in 2013's Ziguinchor report).
- IPUMS holds 1988, 2002 and 2013 samples with religion; the account is blocked.

## 9. Traps

- **The national table misprints a cell in a way its row sum cannot see.** Diourbel's Khadriya sits
  under Layène; the five orders still sum to Musulmans. §11aq read it as a text-extraction shift; the
  rendered page shows the print itself. The table's own Ensemble Layène settles which reading is
  right, and so does the regional report.
- **The Ensemble row is not the regions' weighted sum** and not a partition (99.7). It is probably a
  different tabulation (the foreword mentions a 10% sample for the provisional results). Do not rake
  to it.
- **Diourbel's `AUTRES` and the national `Autres` differ only in case** and are different things;
  `sn1988.py` asserts they resolve apart.
- **OCR in the text layer**: `Ko Ida` for Kolda, `1'1`, `1 OO`, `2, 1`; the form garbles
  *Catholiques*. The readers normalise these and compare against the transcription.
