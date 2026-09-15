# Argentina (`ar`)

**Drawn 2026-09-14** from CEIL-CONICET's *Segunda Encuesta Nacional sobre Creencias y Actitudes
Religiosas en Argentina* (2019), at the survey's six regions, on INDEC's 2022 census population.
45,892,285 people, 6 units, 6 nodes, every row `modelled`. `sources/ar.py` is the construction,
`sources/ar_geo.py` the regions, `sources/ar_grid.py` the placement, `taxonomy/ar2019.py` the
mapping.

## 1. Why a survey, and why this one

- **The census does not ask.** Absent from the UNSD oracle; §11ae's *"South America is
  exhausted"* names Argentina. CEIL's own editorial (Sociedad y Religión 55, 2020) says the
  national census *discontinued* the question; the 2019 report's comparison chart uses the 1947
  and 1960 censuses as its last census points.
- **LAPOP carries no Argentine religion** (§11ad): Argentina is only in the pre-2010 waves and
  `q3c` starts in 2010.
- **CEIL-CONICET ran the same survey twice**, 2008 (2,403 cases) and 2019 (2,421), designed for
  six-region representativeness. Adults 18+ in localities of 5,000+ (2010 frame); 89 localities as
  PSUs stratified by region and city size with PPS, census radios by PPS, dwellings systematic,
  respondents by sex and age quota; ±2% at 95%.

## 2. The documents, and which one is the source

All open, no account, fetched by `sources/ar.py --fetch` into `data/raw/ar/`:

| file | what | used for |
|---|---|---|
| `mallimaci_2020_sociedad_y_religion_55.pdf` | Mallimaci, Esquivel & Giménez Béliveau (2020), ri.conicet.gov.ar handle 11336/144739, 31 pp | **Tabla 5 p8 is the source**: six answers by total and region, including cells under 2%. Tabla 1 p4 for the national no-religion split |
| `ceil_ii25_2019.pdf` | CEIL Informe de Investigación 25 (2019), 72 pp | **the check**: p18 charts the regional split but prints only values over 2%; Católica, Sin religión and Evangélica agree with Tabla 5 to the decimal in all six regions. p14 for the national evangelical split |
| `ceil_encuesta1_2008.pdf` | the 2008 first-survey report, 29 pp | p7, the 2008 regional split, for the stability test only |
| `c2022_tp_c_resumen.xlsx` | INDEC Censo 2022 definitive, total population by jurisdiction | magnitude |
| `c2022_bsas_est_c1_2.xlsx` | INDEC Censo 2022, Buenos Aires Cuadro 1.2, population by partido | the printed `24 Partidos del Gran Buenos Aires` row (10,849,398) that splits the province |

**censo.gob.ar serves an incomplete TLS chain** (WebFetch: *unable to verify the first
certificate*); `fetch()` turns verification off for that host only. **The spreadsheets are not in
the page's link list as HTML anchors a fetcher sees**; the URLs are in the raw HTML of
`censo.gob.ar/index.php/datos_definitivos_total_pais/` and `.../datos_definitivos_bsas/`.
**INDEC's sheets do not start in column A**, so a positional reader silently finds no rows; the
parser reads by content.

### The trap: Tabla 7 of the same article

Tabla 7 prints 2008 and 2019 side by side by region. **Its Cuyo rows swap `Sin filiación
religiosa` and `Evangélica` in BOTH years**, against Tabla 5 for 2019 and against the 2008 report
for 2008, whose Patagonia bar is labelled in full and so fixes the stacking order. Its Patagonia
2019 column also reads 51.3/24.4/24.4 against Tabla 5's 51.0/24.3/24.4, and the CEIL report's own
p20 repeats that. `check_table7()` prints all 17 disagreeing cells on every run. **Use Tabla 5 for
2019 and the 2008 report for 2008.** The 2019 report also gives NEA 2008 as 84.8/10.8 where the
2008 report says 84.0/11.8; the ranks are unaffected either way.

## 3. What the regions are, which nobody writes down

Neither report, the article, CEIL's survey page nor the dataset record lists the provinces in
each region. **The one statement is the map on CONICET's 2019 infographic**
(`conicet.gov.ar/wp-content/uploads/Infografía-Encuesta-Religión-1.pdf`), and because it is a
vector PDF the regions are filled paths with a colour each. Read with PyMuPDF's
`page.get_drawings()`, each province path placed by its bbox:

| fill | region | provinces |
|---|---|---|
| (0.701, 0.770, 0.807) | NOA | Jujuy, Salta, Tucumán, Catamarca, Santiago del Estero, **La Rioja** |
| same fill | Patagonia | Neuquén, Río Negro, Chubut, Santa Cruz, Tierra del Fuego |
| (0.478, 0.581, 0.634) | NEA | Formosa, Chaco, Misiones, Corrientes, **Entre Ríos** |
| (0.516, 0.615, 0.666) | Cuyo | San Juan, San Luis, Mendoza |
| (0.370, 0.485, 0.542) | Centro | Córdoba, Santa Fe, **La Pampa**, Buenos Aires province |

NOA and Patagonia share a fill because they do not touch. **Entre Ríos in NEA is not INDEC's
grouping** and it is CEIL's own map, so it is followed.

**AMBA is the Ciudad Autónoma plus the 24 partidos of Gran Buenos Aires.** The 2008 report labels
the region `Capital y GBA`, and the 2019 survey repeats the 2008 regions for comparison. The
infographic's AMBA inset is ambiguous: its north ends at Tigre and its west edge is straight,
which fits the 24, but its south-east end looks large enough to include Gran La Plata (La Plata,
Berisso, Ensenada, about 0.9M people). The 2008 label decided it; if the microdata shows
otherwise, about 0.9M people move from Centro's shares to AMBA's.

## 4. Which answers carry their own regional shares

The house gate is the split-half (§9bi), which needs microdata. **CEIL deposited the microdata**
(`Segunda_Encuesta_..._base_excel.xlsx` plus the applied questionnaire, CC BY 2.5 AR) at
**ri.conicet.gov.ar/handle/11336/249205, embargoed until 2026-12-31.** The §11 note that "no
microdata is offered" was true of the report and is now out of date.

So the stand-in: **the 2008 wave is an independent sample of the same six regions by the same
team with the same design.** Spearman over six regions against `spearman_null.critical_rho(6)`,
the exact one-sided 95% bar, **+0.8286** (§9ct; not 1.96/√5 = +0.877, which is a standard
deviation), with the exact permutation p over all 720 orderings beside it. Per-region n is not published, so the chi-square runs under
population-proportional and equal allocations:

| answer | ρ 2008→2019 | perm p | χ² prop, p | χ² equal, p | drawn |
|---|---:|---:|---:|---:|---|
| Católica | +0.943 | 0.008 | 52.6, 4e-10 | 73.1, 2e-14 | own shares |
| Sin filiación religiosa | +1.000 | 0.001 | 101.4, 3e-20 | 118.5, 7e-24 | own shares |
| Evangélica | +0.600 | 0.121 | 32.2, 6e-6 | 37.3, 5e-7 | **own shares, OVERRIDE** |
| Testigos de Jehová/Mormones | −0.086 | 0.599 | 16.0, 7e-3 | 16.6, 5e-3 | national rate in residual |
| Otras | +0.132 | 0.394 | 8.5, 0.13 | 14.3, 0.01 | national rate in residual |
| No sabe | no 2008 cell | | 3.0, 0.69 | 4.7, 0.46 | national rate in residual |

**The stand-in is conservative, not neutral**: real change lowers ρ exactly as noise does.
Evangélica is the case. NOA went 3.7% → 16.7%, the change the authors lead with, and one region
moving four places sinks a six-unit Spearman; Patagonia is top in both waves and Centro is in the
bottom two in both. The override is one named answer with its reason printed on every run, and it
refuses to apply unless both chi-squares clear p < 0.001. The bar was not moved. The chi-square
ignores the cluster design; the authors' own ±2% national margin is the simple-random-sample
figure for n = 2,421, so they are not claiming a design effect either.

**When the embargo lifts**: per-region n, a within-2019 split-half, and a proper design-based
test replace all of this. The survey's PSUs are 89 localities, not provinces, so it will still not
give a provincial map.

## 5. Construction

Region population × Tabla 5 share for the three gated answers; the three others divide each
region's residual at their national proportions (1.4 : 1.2 : 0.3); largest remainder to whole
people. Population is INDEC 2022 definitive total population (45,892,285), with Buenos Aires
province split on the printed 24-partido row, asserted equal to its 24 partido rows. 2019 shares
on 2022 people (§3.4). The survey's universe is urban adults and the shares are applied to
everyone.

| region | people | Católica | Sin filiación | Evangélica |
|---|---:|---:|---:|---:|
| AMBA | 13,971,105 | 56.4 | 26.2 | 15.0 |
| Centro | 14,422,270 | 65.7 | 18.6 | 11.3 |
| NEA | 5,654,172 | 67.4 | 7.0 | 23.1 |
| NOA | 5,859,115 | 76.0 | 5.0 | 16.7 |
| Cuyo | 3,408,462 | 69.6 | 13.2 | 14.5 |
| Patagonia | 2,577,161 | 51.0 | 24.3 | 24.4 |

## 6. Mapping (taxonomy/ar2019.py)

Every cell at the depth the **regional** table prints (§2.7). `Católica` → `christianity.catholic.latin`;
`Sin filiación religiosa` → `unaffiliated` (holds Atea 6.0, Agnóstica 3.2, Ninguna 9.7, national
only; precedent cy2021, ca2021, es2026); `Evangélica` → `christianity.protestant` (Brazil's
`Evangélicas`: the 2008 report defines the box as Pentecostal, Baptist, Lutheran, Methodist,
Adventist and IURD; 13 of its 15.3 points are Pentecostal nationally, not drawn);
`Testigos de Jehová/Mormones` → `christianity` (one answer in 2019; the two nodes' common parent);
`Otras` → new `other.ar`; `No sabe` → `unknown`. Splitting either national breakdown across the
regions at a fixed ratio was considered and refused.

## 7. Geography

- **COD-AB's two layers do not share a CRS**: ADM1's `.prj` is Web Mercator and its coordinates
  are metres; ADM2's is geographic. `ar_geo.py` reprojects each on its own and asserts the bounds
  land on Argentina. A union without that would have joined metres to degrees.
- ADM2 pcode `AR006028` is INDEC's `06028`: the 24 partidos join on code and their names are
  asserted one by one. 135 Buenos Aires partidos, 24 to AMBA, 111 to Centro.
- **COD's Tierra del Fuego is the Isla Grande only**; there are no Malvinas, South Georgia or
  Antarctic parts to drop, and the drop is kept in the code in case a later COD adds them.
  Kontur's extract has **no hexes on the Malvinas**; `ar_grid.py` asserts none would join.
- Kontur 400 m: 355,945 hexes, 0.995x the census nationally; per region 0.81x (NEA) to 1.08x
  (Patagonia). The shuffled null is weak at six units and is printed as such.

## 8. Not asked, not filed

No ask. Nothing here is §14: a Catholic-majority country with no persecuted group whose location
this reveals, at 7.6M people a unit.

## 9. Review, 2026-09-14 (f95259a4-arrev)

Read from the PDFs and the CSV, not from this file. The drawn numbers stand and nothing needs
rebuilding. One wording fix, three things for the record.

**Regions: confirmed unwritten, and the one source for them can no longer be re-read.** Neither
report, the article nor the dataset record (handle 11336/249205, full view) lists provinces. The
infographic's recorded URL now redirects to CONICET's home page, the Wayback `id_` snapshot is an
HTML page, and no copy was saved in `data/raw/ar/`. So the Entre Ríos reading rests on the
builder's parse and cannot be repeated; if a copy turns up, keep it beside the reports. What rides
on it: Entre Ríos is about 1.4M people, and NEA against Centro is 23.1 against 11.3 evangelical and
7.0 against 18.6 no religion, so about 170,000 people change colour in each of those two answers if
it moves. Gran La Plata (about 0.9M, AMBA 26.2 against Centro 18.6 no religion) is the smaller call.

**The 2008 stand-in is not only conservative.** §4 notes that real change lowers rho. Two things
push the other way. Each wave is a full sample of about 2,400, less noisy than a split-half's halves
of about 1,200. And with 89 localities drawn PPS, the big agglomerations (Gran Buenos Aires,
Córdoba, Rosario, Mendoza, Tucumán, La Plata) are near-certain in both waves, so a quirk of one city
shows up in 2008 and 2019 alike and passes, the same shape as Tashkent in the LiTS Uzbekistan
split-half. The verdicts still look right: Católica and Sin filiación differ by region at p < 0.004
under both allocations even with the chi-square divided by a design effect of 3, and the three small
answers would fail any test.

**The Evangélica override: keep it, but its printed reason misstates the alternative.** `OVERRIDE`
says that drawn flat it would call Centro at 11.3% and Patagonia at 24.4% the same. Without the
override Evangélica is 84% of each region's residual and comes out AMBA 14.6, Centro 13.2, NEA
21.5, NOA 16.0, Cuyo 14.5, Patagonia 20.8: same ends, not flat. The override changes the colour of
about 276,000 people (0.6%), at most 3.6 points, in Patagonia. Its p < 0.001 gate also assumes no
design effect: at a design effect of 2 it gives p = 0.007 (proportional) and 0.002 (equal) and would
refuse. The better reason to keep it is that it draws the cell the source printed, which is what
`note_public` quotes, where the alternative is a modelled figure pulled toward the national mix.
With NOA left out, the 2008 to 2019 rank correlation is +0.900, so the builder's account of why it
failed is right. The article's prose (p8) gives NOA 2019 as 16.9% against Tabla 5's 16.7%; the same
class of slip as Tabla 7.

**Fixed in `note_public`:** it said Jehovah's Witnesses and Mormons, other religions and don't know
were "drawn at the national rate in every region". They are drawn at national proportions inside
each region's residual, which puts Jehovah's Witnesses and Mormons at 0.14% of Patagonia and 2.12%
of Centro. The sentence now says their regional figures were too small to use. It reaches the
viewer at the next `tiles.py`.

Checked and fine: a lone `Evangélica` is `christianity.protestant` in mx2020, ni2005, pe2017 and
br2010; `other.ar` follows the existing `other.<cc>` nodes; 76.5% (2008) and NOA's 3.7% against the
PDFs. A screenshot shows dots on land following population and none on the Malvinas. No ask.
