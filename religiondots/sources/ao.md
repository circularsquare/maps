# Angola — INE, Recenseamento Geral da População e Habitação 2024, Quadro 7

Wired 2026-09-08. 34,492,888 people aged 2 and over, 326 municipalities, 20 drawn bodies.

| | |
|---|---|
| source | Instituto Nacional de Estatística, **RGPH 2024**, Quadro 7.1 onwards of the **21 provincial volumes** (January and February 2026) |
| basis | `self_id`, religion or spirituality, population aged 2 and over |
| geography | **326 municipalities** of Lei 14/24 — ~106,000 people each |
| categories | **21** bodies plus the universe total; 20 drawn, `Não sabe/Não respondeu` off the tree |
| drawn | **32,806,184 people**, 95.1% of the 2+ universe and 90.7% of the census count |
| licence | INE publication, free to download and cite |

**The deepest religion question any African census on this map asks, and the only one
anywhere that names four African-founded churches one at a time.** Tocoísta, Kimbanguista,
Bom Deus and Josafat are 1,204,617 people between them, and three of the four have a
geography that dates them to the year they were founded.

Two things about it are unusual enough to lead with. **The country was redrawn between the
fieldwork and the publication**, so these figures exist on a tier no boundary file had. And
**two of the twenty-one provincial volumes are missing half their religion table**, which
is a defect in the published PDFs rather than in the parse.

---

## 1. Where the map came from, and why the obvious source was the wrong one

The route in was a Wikimedia Commons map, *Religious map of Angola, 2014*, drawn by
User:BorysMapping in March 2026 from "the 2014 Census" with citypopulation.de cited as the
visualisation. **Neither of those is the source.** citypopulation's Angola pages carry
population and nothing else; the 2014 census's own *Resultados Definitivos* (March 2016,
204 pages) prints religion exactly once, as **Gráfico 16 on page 52**, a national bar chart
with seven categories and no geography at all.

**The 2014 census is not the current one.** Angola ran a new RGPH on 19 September 2024 —
extended to 20 December for coverage — and INE published:

| | published | what it has |
|---|---|---|
| national volume | 20 Nov 2025 | Quadro 7.1-7.2, 21 bodies **by province** |
| 21 provincial volumes | Jan-Feb 2026 | Quadro 7.x, the same 21 bodies **by municipality** |

The provincial volumes are the source. `sources/ao_pdfs.py` holds the filename of each.

**The filenames cannot be derived.** Every INE publication lives at
`/Arquivos/arquivosCarregados/Carregados/Publicacao_<18-digit .NET tick stamp>.pdf`, where
the stamp is the upload time and carries no meaning. The index is the publications sidebar
rendered into every `ine.gov.ao/publicacoes/detalhes/<base64 id>` page, which lists every
volume with its title; parsing one such page gives the whole set.

**`ine.gov.ao` answers a plain scripted GET with a browser User-Agent and needs nothing
else** — no key, no login, no referrer. §11p recorded Angola as one of the 28 African
offices that answer; this is the first time anything was taken from it.

---

## 2. Lei 14/24, and why no standard boundary file works

The census was **collected** on Lei 18/16's division: 18 provinces, 164 municipalities, 562
communes, 33,318 bairros and aldeias, 55,788 census sections. **Lei n.º 14/24 of 5 September
2024** replaced it, splitting Luanda, Moxico and Cuando-Cubango to create Icolo e Bengo,
Moxico Leste and Cuando, and taking the country to **21 provinces, 326 municipalities and
378 communes**. INE retabulated onto the new division before publishing, and says so in the
national volume (page 32).

So the drawn tier is about eighteen months old, and:

| source | what it has | usable? |
|---|---|---|
| OCHA COD-AB (HDX, `cod-ab-ago`) | 18 / 161 / 539, 2018 vintage | no |
| geoBoundaries ADM2 | "Municipality", represents **2006** | no |
| OpenStreetMap `admin_level=6` | Luanda's 9 old municipalities, a dozen in Huíla and Malanje, **nothing else** | no |

**Dissolving COD-AB's communes into the new municipalities does not work either, and it is
worth writing down so nobody spends the afternoon on it.** The idea is sound — most new
municipalities are promoted communes — but COD's ADM3 attributes are wrong exactly where
Angola's people are:

* Luanda's `Belas` contains `Viana`, `Viana Sede`, `Kilamba Kiaxi` and `Mussulo`;
* `Cazenga` contains `Kikolo`, which is in Cacuaco;
* Bengo's `Dande` contains `Funda`, which is in Luanda;
* `Uíge` municipality has one commune where the census counts several;
* the spellings are its own (`Kikabo` for Quicabo, `Muxiluando` for Muxaluando,
  `Kiaje` for Quiage).

Only **282 of the statute's 538 leaf units** match one of COD's 539 communes by name.

### 2.1 What was used instead

**An ArcGIS Online feature service, `Nova Divisão Administrativa de Angola`**, item
`4233a339ad9d482c83f617660dea2303`, owner `kuyengap_msugis`, public, digitised from *Diário
da República I série n.º 171 of 5 de Setembro de 2024* — Lei 14/24 itself. The statute
describes every provincial, municipal and communal boundary as a named sequence of rivers,
roads and watersheds, and the layer follows it.

```
https://services.arcgis.com/uHAHKfH1Z5ye1Oe0/arcgis/rest/services/
    Nova_Divisao_Administrativa_de_Angola_WFL1/FeatureServer/0   # 326 municipalities
    .../FeatureServer/1                                          # 21 provinces
```

`Query` is enabled and no token is needed. The polygons carry the statute's river courses at
full detail, so the layer is **66 MB for 326 features** and must be fetched 25 objectIds at
a time; one request for the lot times out.

**It is one person's work and is treated as such.** `sources/ao_geo.py` checks it three ways,
none of which is derived from it:

1. **Against the statute.** The angolex.com transcription of Lei 14/24 was parsed
   independently into province → municipalities (`data/raw/ao/lei_14_24.json`). It gives 21
   provinces and **326 municipalities**, and the layer matches it **province by province and
   name by name on all 326**. Two names carry a parenthesised alias the census drops
   (`Boa Entrada (Cadá)`, `Gangula (Kuvu)`); `fold()` strips brackets, so no alias table.
2. **Against the census.** Quadro 9 of the national volume gives municipalities per province
   (10 Cabinda, 23 Uíge, 16 Luanda, 7 Icolo e Bengo, 27 Malanje…). The layer matches all 21.
3. **Against Natural Earth 10m**, which is nobody's derivative of it: area 1,247,384 km²
   against Angola's 1,246,700 (**+0.1%**), bounding box within **0.045°**.

And the join to the census is **326 of 326 by name**, with every province name folding
identically on both sides.

---

## 3. The table, and why the parser reads headers instead of positions

Quadro 7 is *População com 2 ou mais anos por área de residência e município, segundo a
religião ou espiritualidade*. Its rows are the province, an urban/rural pair and the
municipalities. Its columns are the universe and 21 bodies:

> Católica · Bom Deus · Islâmica/Muçulmana · Animista · Judaica · Protestante · Universal do
> reino de Deus · Nova apostólica · Tocoísta · Kimbanguista · Josafat · Assembleia de Deus
> pentecostal · Testemunha de Jeová · Metodista · Evangélica · Adventista · Baptista ·
> Mensagem dos ultimos Tempos · Sem religião · Outra religião · Não sabe/Não respondeu

**Nothing about the layout is constant between volumes**, and every one of these was found
the hard way:

| what moves | example |
|---|---|
| how many panels the 21 are split over | Bengo 12+9, Benguela 7+7+7, Moxico 10+11, Huíla 11+10 |
| which panel a body is in | Moxico puts `Bom Deus` in panel 2; everyone else puts it 6th in panel 1 |
| whether a panel repeats the universe column | Namibe's second panel goes straight into Kimbanguista |
| whether a panel has a title | Huambo, Huíla, Moxico Leste and the national volume start one at the top of a page with no `Quadro` line |
| the spelling of a heading | Malanje writes `Outro` for `Outra religião`; Benguela `Testemunhas` for `Testemunha` |
| the title itself | `Quadro 7. 1-` with a space, in Moxico and Namibe |

So the parser:

* finds panels by walking forward from a titled page while each page still prints a table
  whose first row is the province, and **merges a page that repeats the same headings**
  (pages 117 and 118 of the national volume are one table);
* takes the **universe column** to be the one holding the largest figure in the province row
  anywhere in the volume, because the bodies partition it and none can reach it;
* matches columns to bodies by letting **each header word carrying a body's stem vote for
  the column whose figures are printed nearest it**, then solving the whole volume as one
  assignment under the constraint that each body is used exactly once. That constraint is
  what settles Cuanza Norte, where `Mensagem` sits closer to the Baptista column than to its
  own.

### 3.1 A blank cell is not a zero, and reading left to right silently transposes a row

Bié's `Belo Horizonte`, `Luando` and `Umpulo` print 11, 10 and 11 figures where the panel has
12 columns; Lunda Sul does the same in 17 cells. **Read in order, Belo Horizonte's 10,588
Protestants land in the Universal do Reino de Deus column and every figure after them moves
one place** — §12's shape 2, with no total anywhere disagreeing.

Figures are therefore placed by the **right edge of the printed number**, which is the edge
INE aligns a column on, and a column with nothing in it is recorded as zero and counted.
That the row then reconciles against its own universe is what confirms the blank was a zero.

### 3.2 Three other traps in the same table

* **A wrapped municipality name leaves the data row unlabelled.** `Maquela do Zombo` prints
  as three lines — `Maquela do`, the figures, `Zombo` — and the figure row's own label is
  empty. The nearest header line above and below are joined back on.
* **Seven provinces contain a municipality of their own name** (Benguela, Huambo, Malanje,
  Uíge, Cabinda…), and a panel running over two pages prints the province row again at the
  top of the second. Dropping every row labelled with the province loses a municipality of
  788,380 people; the province row is instead the one whose universe *is* the province's.
* **Huambo prints the same panel twice**, on pages 110 and 111, figure for figure, under a
  `Continua na página seguinte` that continues nothing. Exact repeats are dropped.
* **Huíla follows its table with a second, coarser one** on page 117: the same 23
  municipalities against a nine-column summary that groups the bodies differently and heads
  two adjacent columns `Protestante`. A panel naming a body an earlier panel already named
  is not part of this table.

---

## 4. Uíge and Moxico Leste are missing half their table

**Page 112 of the Uíge volume and page 86 of the Moxico Leste volume are blank** — zero
characters of text, zero images, zero drawings. Both volumes print the universe and 11
bodies on the page before and then move to chapter 8. The words `Metodista` and `Adventista`
occur nowhere in either volume outside its list of tables.

| province | 2+ population | bodies printed | printed bodies cover | remainder |
|---|---|---|---|---|
| Uíge | 1,895,001 | 11 of 21 | 61.3% | 733,662 |
| Moxico Leste | 382,707 | 11 of 21 | 47.2% | 202,180 |

That is **2,277,708 people, 6.6% of Angola**, with no Methodist, Baptist, Adventist,
Evangelical, Pentecostal, Jehovah's Witness, Mensagem, no-religion or other-religion count
of their own.

### 4.1 The ten are filled in, because omitting them makes the stronger false claim

Drawn on the printed eleven alone, **Uíge comes out 39% short on people** — and
compositionally wrong with it: Catholicism reads as 55% of the province's drawn dots against
a true 33.8%, and the province appears to hold **no Evangelicals** (347,084 of them, its
second largest body) and **nobody with no religion**. A hole is not the neutral option.

**Both margins of the hole are published numbers**, which is the whole case for filling it:

* the **row** margin is each municipality's own unexplained remainder, `universe` minus the
  eleven bodies its volume printed, straight off the provincial table;
* the **column** margin is the province's total for each unprinted body, straight off Quadro
  7 of the **national** volume, which does print all twenty-one by province.

**And they agree without being made to**, which is the evidence that they are the same
quantity. Moxico Leste's municipal remainders sum to 202,180 against the national volume's
202,180 **exactly**; Uíge's to 733,662 against 734,842, a ratio of **0.9984** — the residue of
the same Cabinda/Uíge revision §5 describes. The column margins are scaled to the row total so
each municipality's remainder is consumed to the person.

**What is assumed, stated plainly: the MIX is uniform inside a province.** With two margins
and nothing in the interior the maximum-entropy fill is the outer product, so:

* the **amount** a municipality receives is measured and varies — Lucunga's remainder is 29.3%
  of its people, the city of Uíge's 44.2%;
* the **split** of that amount between Methodist and Adventist is not, and is the same
  everywhere in the province. Uíge's is 47.2% Evangélica, 12.6% Assembleia de Deus, 11.6% Sem
  religião, 7.8% Metodista; Moxico Leste's is 47.3% Sem religião, 15.8% Adventista, 11.7%
  Assembleia de Deus.

This passes [[feedback_proxy_residual_nameable]]'s test rather than dodging it: the
non-matching part is a published number to weight by, not a correlation.

The 320 rows are `derived` (`sources/ao.py` `fill()`, `tier=derived` in the note). They never
ring, and they carry `rollup.NOWHERE` rather than no roll at all — see §4.2.

### 4.2 `roll = NOWHERE`, and the whole-country `measured` set

A derived row with no recorded roll makes `rollup.py` walk the religion tree for an ancestor
the country measured. **`measured` is a set for the WHOLE COUNTRY, and that is wrong here.**
The walk takes Uíge's derived Assembleia de Deus dots to `christianity.pentecostal` — which
Angola does measure, from `Mensagem dos ultimos Tempos`, **in the nineteen provinces that
printed it and in neither of these two**. So 116,174 people who should disappear under
`inferred dots: not shown` drew as Pentecostals instead.

That is §7a-i-1's failure one level up, so `rollup.py` gained a sentinel on 2026-09-08:
`roll == rollup.NOWHERE` means *this adapter knows there is no measured ancestor at this
unit*, and the walk is skipped. Angola is the first user. It is additive — a country that
does not emit it behaves exactly as before — and `taxonomy/ao2024.py`'s `COLUMNS` carries the
matching comment, which is the one `tools/check_rollup.py` asks for.

### 4.3 What to check for, and what was already checked

**INE has issued an errata before** — *Errata das Províncias de Luanda e Icolo e Bengo*, which
§5 confirms is already in the provincial volumes. A corrected Uíge volume would be worth far
more than the fill.

Checked 2026-09-08, all negative:

| | |
|---|---|
| a Uíge or Moxico Leste volume newer than February 2026 | none; the listing still serves `Publicacao_639120325078684256.pdf` |
| an errata publication for either | none |
| a religion **thematic report** (the preface promises them) | none yet |
| the **10% microdata** the preface promises six months after publication (so ~May 2026) | no catalogue: `/microdados`, `/nada`, `nada.ine.gov.ao`, `microdados.ine.gov.ao` and `censo.ine.gov.ao` are all 404 or unresolvable, and a nonsense path returns the same empty 404, so these are genuine misses rather than a soft-404 wall |

The check is one GET of `ine.gov.ao/publicacoes/detalhes/NDc0MTE=` and a look at the sidebar.

---

## 5. The provincial volumes revise the national one, and the revision is systematic

The national volume (20 Nov 2025) and the provincial volumes (Jan-Feb 2026) do not agree.
Every province's municipalities reconcile with **its own volume's** province row, body by
body, to within INE's own arithmetic — so this is a revision and not a parse error.

| province | universe: provincial vs national |
|---|---|
| Luanda | −144,745 |
| Icolo e Bengo | +144,746 |
| Cabinda | +7,050 |
| Uíge | −7,052 |
| Benguela / Namibe | +29 / −29 |
| every other | 0, or ±1 rounding |

Luanda/Icolo e Bengo is the published errata. Cabinda/Uíge is not documented anywhere found.

**Inside the revised provinces the movement is one-directional and is the more interesting
half.** Cabinda's `Sem religião` rises 19,316 while every other body falls a little;
Luanda's rises 115,182. Nationally the provincial volumes put **about 311,000 more people in
`Sem religião`** than the national volume did. Something was reclassified into no-religion
between November and February, and INE does not say what.

The provincial volumes are the later word and are what is drawn. `sources/ao.py` reports
both, and its hard check is the internal one.

---

## 6. What the categories turn out to be

Four of the twenty-one needed identifying before they could be mapped. All four were settled
by their own geography as much as by the literature — see `taxonomy/ao2024.py` for the calls.

| category | what it is | the tell |
|---|---|---|
| **Tocoísta** | Igreja do Nosso Senhor Jesus Cristo no Mundo, Simão Toco, Léopoldville, 25 July 1949 | Uíge 6.2%, and **11.1% in Maquela do Zombo, the municipality Toco was born in** |
| **Kimbanguista** | Église de Jésus-Christ par son envoyé Simon Kimbangu, Nkamba, 1921 | Lufíco 34.7%, Nóqui 28.1% — both on the Congo opposite Matadi |
| **Bom Deus** | Igreja Fraternidade Evangélica de Pentecostes na África em Angola, Simão Lutumba, 1981, out of the Congolese Nzambe Malamu | no region at all; nothing above 6.4% anywhere |
| **Josafat** | what the Portuguese **Igreja Maná** operated as in Angola from 2009, after its 2008 ban; still registered separately since the ban was lifted in 2017 | urban Luanda, like IURD beside it |
| **Nova apostólica** | the New Apostolic Church, Irvingite, Hamburg 1863 | Moxico Leste 8.0%, Ninda 23.8% — **the Zambian border**, and Zambia is one of the church's largest countries |
| **Mensagem dos ultimos Tempos** | the Message: IMUT dates itself to **Jeffersonville, Kentucky, 1933**, William Branham's town and year | Lunda Norte 1.6%, Canzar 4.7% — the DRC border |

### 6.1 Two figures that should not be read at face value

**`Sem religião` is 12.10% and it is a rural south-western answer.** Iona 60.6%, Virei
53.3%, Curoca 48.6% — Kuvale, Himba and Mucubal transhumance country — against Luanda
province's 14.9%, seventh of nineteen. Beside it **`Animista` is 44,370 people, 0.13%**,
which for a country of 34 million is not a credible count of traditional practice and is an
order of magnitude below Ghana's 3.25%. The likeliest reading is that a herder who keeps the
ancestors answers *no religion* rather than *animist*. Both cells stay where the census put
them; the note in `ao2024.py` says to read them together.

**`Judaica` is 34,711 people and its geography is the diamond belt**, not the capital:
Lunda Norte 0.40%, Chitato 1.12%, Cazombo 0.96%, Lucapa 0.71%, while Luanda — where an
actual Israeli and Portuguese Jewish community does live — is a rounding error. Angola has
no historic Jewish population. This is far more likely to be Israelite and Judaising
movements out of the Congo basin, or a coding artefact in a box almost nobody uses. It goes
to `judaism` because that is the box INE printed (§2.4), and **the figure should not be
quoted as Angola's Jewish population.**

### 6.2 `Islâmica/Muçulmana` draws its own explanation

135,003 people, 0.39%, and Lunda Norte is **2.71%** — Chitato 4.68%, Lucapa 4.48%, Dundo
4.15% — with Luanda second at 0.61%. That is the West African trading population of the
diamond districts and a Lebanese and North African commercial community in the capital,
which is what every account of Islam in Angola describes and which the census map draws
unaided.

---

## 7. Four new taxonomy nodes

Added to `branches.py` 2026-09-08:

| node | people | why |
|---|---|---|
| `christianity.africaninstituted.kimbanguist` | 409,254 | second named child of the AIC node; Harrist was the first |
| `christianity.africaninstituted.tocoist` | 350,936 | the largest church founded by an Angolan |
| `christianity.africaninstituted.bomdeus` | 339,044 | African-founded, Pentecostal in practice; the node is defined by the first |
| `christianity.newapostolic` | 515,929 | Irvingite, and neither Pentecostal, evangelical nor Stone-Campbell |
| `other.ao` | 435,663 | the residual |

`christianity.newapostolic` has **two counted peers elsewhere on the tree** — `ee2021.py`
files Estonia's New Apostolic Church on `christianity.restorationist`, which is the wrong
restorationism, and `au2021.py` files Australia's on `christianity.other`. Both predate the
node and were deliberately not moved, on the precedent
`christianity.africaninstituted.harrist` set with Benin's Celestial Church: each is a
one-line change plus a re-scatter of that country, and doing it here would alter two
countries nobody asked about.

---

## 8. Build

```
python sources/ao.py --fetch          # 22 PDFs, ~250 MB
python sources/ao.py                  # -> data/normalized/ao.csv
python sources/ao.py --report         # per-municipality residuals
python sources/ao_geo.py --fetch      # 326 + 21 polygons over the REST API, ~90 MB
python sources/ao_grid.py --fetch     # one 17 MB gzipped Kontur gpkg
```

`data/raw/ao/lei_14_24.json` is the parsed statute and is optional; without it `ao_geo.py`
skips check 1 and says so. It was produced from the angolex.com transcription of Lei 14/24,
which is the full text including every boundary description.
