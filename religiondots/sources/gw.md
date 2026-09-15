# Guinea-Bissau — RGPH 2009, religion by região

**Drawn 2026-09-14** (session `f95259a4-gw`). 9 regiões, 5 categories, 1,213,509 Guinean
nationals who answered, every row `measured`, counts exact. New node `other.gw`.

- `sources/gw.py` -> `data/normalized/gw.csv` (the PDF is `data/raw/gw/caracteristicas_socio_cultural.pdf`)
- `sources/gw_geo.py` -> `data/geo/gw/gw_regions.gpkg`, `gw_hexes.gpkg`, `gw_lookup.csv`
  (COD-AB ADM1 + Kontur 400 m)
- `taxonomy/gw2009.py` -> the mapping; `countries.py` `"gw"` -> the wiring;
  `taxonomy/branches.py` `other.gw` is the one new node
- sources.md **§9dx** is the short write-up; it corrects **§11w**'s Guinea-Bissau paragraph and
  one detail of **§11aq**'s.

```
python sources/gw.py     --fetch
python sources/gw_geo.py --fetch
```

## 1. What INE publishes

| release | religion | tier |
|---|---|---|
| **RGPH 2009, *Características socioculturais*** (INE, 92 pp) | **Anexo Quadro 3, região x religion, counts and shares** (PDF p72), repeated as Anexo Quadro 7 (p82); body Quadro 3 by sex, Quadro 4 região shares, Quadro 5 etnia x religion | **9 regiões** |
| the nine regional booklets (`Regiao_de_*_RGPH_2009.pdf`, `SAB_RGPH_2009.pdf`) | none (§11w) | |
| *Estado e estrutura da população* (115 pp) | none (§11aq) | |
| RGPH 1991, `Resultados_rgph_1991.pdf` | Tabela 6.6, national, Catholic and other Christian separate (§11aq) | national |
| UNSD Demographic Yearbook table 28 | Guinea-Bissau 2009, six categories, total only | national |
| RGPH 2025 | the individual form (`RGPH2025/Enumeraçao Indivíduo.pdf`, Wayback 2026-04-11) is probably an image; whether it asks religion is **not verified**; no results yet | |

**§11w's "the one analytical volume that mentions religion does so in passing" was this
volume.** Its religion section (3.4, PDF pp28-32) has four tables, and the annex prints the
região table in counts. The office site is live and serves the file with a plain GET.

## 2. The table and its checks (all in `sources/gw.py`)

Anexo Quadro 3 is drawn as printed: no share is multiplied by anything.

| check | result |
|---|---|
| file | 2,725,536 bytes, SHA-1 `3SHQDVYZG7WACAPWVML6UUN4ZZZSEKXH`, byte-identical to Wayback's 2023-03-14 capture and every later full one |
| Anexo Quadro 3 and Quadro 7 parsed off the page = the transcription | 7 rows x 10 columns, identical on both pages |
| rows and columns close | every religion over 9 regiões, every região over 6 answers, nation 1,442,227 |
| Quadro 7's printed shares | all 70 are the counts' shares within 0.049 pp |
| body Quadro 4 | equal to Quadro 7 cell for cell |
| região totals | the same nine in Quadro 1A (p22) and Anexo Quadro 2 (p71) |
| UNSD table 28 | all six columns equal to the person; `Unknown` is ND |
| ethnic witness (§5) | Muslim r = +1.000, animist +0.894, Christian +0.879; no visible região swap fits better |
| COD-AB name join | 9/9; `SAB` = `Bissau`, the rest differ only by accents |
| Kontur 2023 vs census 2009 | 1.50x overall; SAB 0.55x, the rest 1.72x (Tombali, Cacheu) to 1.96x (Biombo) |

## 3. The universe, and the gap

**Guinean nationals in ordinary households, as enumerated.** The report's own arithmetic:

| | people | where |
|---|---:|---|
| enumerated in ordinary households | 1,449,230 | Quadro 1, PDF p20; also p3 |
| of whom Guinean nationals (**the table**) | 1,442,227 | Quadro 1 |
| of whom foreign nationals | 1,933 | Quadro 1; Anexo Quadro 1 by country (Guinea 535, Mauritania 361, Senegal 354) |
| of whom nationality not recorded | 5,070 | the remainder; no row prints it |
| enumerated in collective households | 3,696 | p3, footnote *orfanatos e casas religiosas* |
| enumerated, all | 1,452,926 | |
| with the post-enumeration survey's 4.6% omission | 1,520,830 | p3, the headline figure |

**ND is non-response**, 228,718 or 15.86% of nationals: *"uma percentagem relativamente
importante (15,9%) dos entrevistados não responderam à esta pergunta"* (PDF p28), which the
report puts down to data quality or to religion being personal. Not drawn (spec §3.5).

`gap_share` is **0.1648**: ND plus the 10,699 outside the table, both as shares of the
1,452,926 enumerated (15.74% + 0.74%). `tools/gap_share.py` computes ND alone as 15.86% of the
table's universe by both routes; the authored figure is larger because it adds the people in
no table, which the tool cannot see.

**The Guinea precedent (§9dh) applied**: there, UNSD's `Unknown` was the collective
households and went to `gap`. Here UNSD's `Unknown` is the ND column, a genuine non-answer,
and the collective households are outside the table altogether; both go to `gap`, and the
sentence names each part.

**Not corrected for the 4.6% omission.** The p3 note recommends its per-região weights for
population totals and says this report used the uncorrected population. The weights are
per-região scalars (1.038 Cacheu to 1.061 SAB), so they would change dot counts and no share;
drawn as printed so every count equals the office's table and UNSD's.

## 4. The questionnaire (Anexo 2 of the same volume)

P.14, asked of *todos os recenseados*: **"Qual é a sua Religião?"**, a blank line and a
two-digit code box. **No printed answer list.** So:

- no no-religion box that names animism, and the Mozambique/Laos ruling (spec §6.3a, §9dn)
  does not apply: `Sem religião` is people who said they had none, drawn as `unaffiliated`;
- no card whose column order could be swapped (Zambia, §9db); the coded answers are the
  table's rows;
- one answer per person. The report says (PDF p28) *"existe uma percentagem significativa da
  população que pratica duas religiões, o que não foi comtemplado no estudo"*.

## 5. The ethnic witness, and two misprints in Anexo Quadro 2

**The prose on PDF p30 swaps Gabú and Bafatá** (*"77,1% e 86,5% respectivamente"*) and prints
Oio as 47.1% Muslim. **§11aq attributed this to Quadro 4; it is the sentences above Quadro 4.**
Quadro 4 agrees with the annex.

To decide which way round the regiões are without trusting the same tabulation, `gw.py`
predicts each região's religion from its ethnic mix: Anexo Quadro 2 (região x etnia, counts)
times Quadro 5's national religion rates per etnia.

| região | Muslim predicted / printed | animist | Christian |
|---|---|---|---|
| Bafatá | 76.0 / 77.1 | 4.4 / 3.9 | 6.7 / 6.8 |
| Gabú | 84.2 / 86.5 | 1.4 / 0.3 | 2.9 / 2.6 |
| Biombo | 8.1 / 6.3 | 28.3 / 40.1 | 40.1 / 30.2 |
| Cacheu | 15.8 / 14.8 | 25.8 / 34.0 | 37.7 / 30.7 |
| SAB | 35.5 / 34.2 | 17.2 / 7.9 | 28.5 / 40.2 |

Gabú is more Muslim than Bafatá in both. The prediction uses national rates per etnia, so it
cannot see that a Papel or Balanta in Bissau is more often Christian than one in a village,
which is what caps the animist and Christian correlations (+0.894, +0.879). Muslim fits at
+1.000. Of the 36 two-região swaps, only Tombali/Oio fits better, by less than 0.0001 of
0.0966, and those two are printed within 3.3 pp of each other on every answer.

**Anexo Quadro 2 has two misprinted cells, each a dropped digit.** As printed, the Fula row
and the Oio column are both exactly 21,000 short, and the Mandinga row and the Cacheu column
both exactly 10,000 short; everything else closes. Fula in Oio is printed 2,980 for 23,980,
Mandinga in Cacheu 1,460 for 11,460, and the printed shares (0.7% each) were computed from the
misprints. Other splits across the four cells would also close; only this one is a single
slipped digit, and it is what takes the Muslim correlation from +0.993 to +1.000. The drawn
table is not affected. Quadro 5 and Anexo Quadro 2 also print Balanta and Fula one person apart
(323,949 / 323,948 and 410,559 / 410,560).

## 6. Boundaries and placement

COD-AB Guinea-Bissau (`cod-ab-gnb`, version 01, from SALB, valid 2021-06-09, CC BY-IGO), ADM1:
eight regiões and Bissau. SAB is 83 km² holding 362,699 nationals; Gabú is 9,043 km² at
22.6/km².

**The coastline snap** (spec §12, the archipelago rule). COD-AB's shoreline is coarse against
Kontur's 400 m hexes: 330 hexes holding 79,221 people (3.67%) had centroids in no região,
**48,348 of them on Bissau's own waterfront, 47 to 1,180 m outside the SAB polygon**, the rest
along the Bijagós and the mangrove estuaries. 52% of that population was within 400 m, 95.7%
within 1 km and 99.9% within 2 km, so `gw_geo.py` snaps to the nearest região within 2 km and
drops 3 hexes (90 people). Without it Bissau would have lost a quarter of its placement
weight, all of it on the shore.

**Kontur is thin in Bissau**: 199,572 people in SAB after the snap, 0.55x the 2009 census
nationals, while every other região is 1.72x to 1.96x. Kontur has placed part of the
capital's growth outside the SAB line or not at all. It is a within-região weight, so no count
moves; the shape inside the city is what it sets.

**Kontur's density cap**: `python kontur_cap.py gw` finds no block at the cap (13,414 cells),
so no `kontur_cap.csv` row. Bissau is not a core Kontur reaches 46,200/km² on.

## 7. What the table shows

Of those who answered: Gabú 96.7% Muslim and Bafatá 86.9%; Biombo 50.7% animist and 38.2%
Christian; Cacheu 41.2% animist and 37.2% Christian; SAB 46.9% Christian and 40.0% Muslim.
As printed, over all nationals: Biombo 40.1% animist, Cacheu 34.0%, Gabú 0.3%; SAB 40.2%
Christian and 45.8% of the country's Christians; Cacheu 29.1% of the animists and 67.4% of
`Outra religião`.

ND is not flat: Bolama/Bijagós 25.7%, Quinara 21.5%, Biombo 20.8%, against Gabú 10.5%. By etnia
(Quadro 5) it is 25.5% of Balanta, 24.0% of Bijagó and 19.8% of Papel against 10.7% of Fula and
10.8% of Mandinga, so it runs highest among the peoples with the most animists. Read the animist
share as a floor, on that and on the one-answer rule.

## 8. §14 was considered and no ask was filed

Religion here follows ethnicity closely (Quadro 5), which is the §14.5 concern; the ethnic
table is used only as a check and is not drawn. Nine units of 160,000 people on average, the
office's own published table, on its live site. Guinea-Bissau's instability has been military
and political rather than religious, and nothing here locates a group more finely than INE
already has.

## 9. Gotchas

- **A truncated Wayback capture opens.** Captures of this URL with digest `UXLWBN4E...` are the
  first 1,048,576 bytes of the file: `%PDF-1.5` header, no `%%EOF`, and MuPDF still reports 92
  pages (with xref errors). `fetch()` checks the trailer and the SHA-1.
- **The Quadro 3 page cuts off the SAB share column** at the right margin (`7,`, `34`, `40`);
  take shares from Quadro 7.
- **A cell wrapped mid-number**: `100,` then `0` on the next line in Quadro 3's TOTAL row.
- **The census writes `SAB`** and `B. Bijagós` / `Bolama/Bijagós`; COD-AB has `Bissau` and
  `Bolama/Bijagos`.
- The prose disagrees with its own tables (§5). Use the annex.

## 10. Terms

INE Guiné-Bissau's reports are public PDFs on its own site with no licence text. UNSD
Demographic Yearbook data is public. COD-AB is CC BY-IGO; Kontur Population is CC BY 4.0.
