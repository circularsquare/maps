# São Tomé and Príncipe — IV RGPH 2012, the seven district reports

Instituto Nacional de Estatística, São Tomé. Drawn: **10 nodes on 7 districts, 176,983 people,
99.02% of the country**. `sources/st.py`, `sources/st_geo.py`, `sources/st_grid.py`,
`taxonomy/st2012.py`, `taxonomy/branches.py` (`other.st`).

The country is worth drawing for the shape of its form. **INE's 2012 census asks which of nine
individual churches you belong to and offers no family boxes at all**, so São Tomé prints the
Igreja Maná, the Igreja Messiânica Mundial and the New Apostolic Church as separate national
figures, and does it for 178,739 people.

---

## 1. The queue priced it national-only, and there is a whole census report per district

`sources.md` §11w (2026-09-07) ranked São Tomé fifth of Africa's undrawn countries on
`tools/oracle.py`'s category depth and closed the row as **"not chased"**. That was a statement
about the oracle, whose rows are national and urban/rural.

`www.ine.st` has a folder called **`Dados Distritais e Nacional Recenseamento 2012`** holding
eight PDFs: one for each of the seven districts and one national. Each district report reprints
Quadros 1 to 26 for that district alone, so the thirteen-category religion table exists seven
times over at the tier below the one the oracle shows. This is the third day running that
`AGENT_BRIEF`'s highest-yield question, *does the office publish a volume per unit of the tier
below the one you expect*, has landed a country finer than its queue row.

### The route is an open autoindex, which is a shape worth naming

`ine.st` is Joomla with Phoca Download, the same CMS family §11p's rule-1 id sweep was written
for. It needs none of that. **LiteSpeed serves the download tree with directory listing left
on**, so a recursive walk of

    https://www.ine.st/phocadownload/userupload/Documentos/

prints all 223 files with their sizes and modification dates in 68 requests. No plugin
catalogue, no REST base, no search endpoint.

**One trap: `/phocadownload/` and `/phocadownload/userupload/` are NOT indexes.** Both return
200 with a Joomla page rather than an `Index of` listing, so a walker that starts at the
component root concludes the whole tree is closed. The autoindex begins at
`.../userupload/Documentos/`, one level further down. Six subdirectories inside it are also not
indexes (`Atlas`, `IPC_Ano`, two spellings of `Informações Estatísticas/Educação`, two of the
MICS folder); everything else walks.

### What else is in the tree, so nobody re-walks it

* **`Recenseamentos/2012/Relatório temáticos`** (12 PDFs) is the analytical series: population
  dynamics, structure, migration, the elderly, economic activity, women, youth, nuptiality,
  children, housing, disability, education. Religion appears in none of them as a table; the
  word occurs only as *casamento religioso*.
* **`Recenseamentos/2001/RelatóriosTemáticos`** (9 PDFs) is the same series for the III RGPH.
  The oracle has 2001 at six categories (Catholic 96,684, Unknown 26,913, Evangelical 4,616,
  Other 4,212, Non-Apostolic 2,756, Adventist 2,418) and its `Unknown` is 19.6%, so 2012 is
  better in every way and 2001 was not chased further.
* **`DADOS_LOCALIDADE_PROJECOES`** holds the locality publication (§3) and *Projecções
  Demográficas 2012-2035*.
* **`Recenseamentos/2024`** holds the V RGPH results (§4) and two large videos.

## 2. The parse, and the four ways it closes

`sources/st.py` reads **Quadro 8**, *Repartição da população residente segundo sexo e
nacionalidade por religião*, whose first row is the district total across all thirteen
categories. Quadro 6 has the same figures but runs to eight pages and puts its total row in a
different place in each report; Quadro 8 is two pages, or one for Água-Grande, and always opens
with the total.

The table is too wide for the page, so INE breaks it after `Maná` and repeats the row stubs.
The parse anchors on the **last cell of each header** and takes the next seven figures in
reading order, which is [[reference_pdf_table_geometry]] in its cheapest form. Nothing in
either header is numeric, so over-reading is the only failure mode, and `_numbers` raises if a
line carries more figures than the row has cells.

**That guard earned itself immediately.** PyMuPDF gives one cell per line for every page of the
district reports, but the locality publication's widest rows put the last two cells on one line
(`17.503 39.691`), and a line-at-a-time reader walks straight past them into the next row and
returns a locality's figures as the country's.

The four checks, in `check()`:

1. **Each district's thirteen categories sum to its own printed total.** Seven independent
   identities, all exact.
2. **The seven districts sum to the national report** in every one of the thirteen categories
   and in total, 178,739.
3. **INE's 2016 locality publication reproduces it**, folded to seven columns. See §3.
4. **UNSD Demographic Yearbook table 28 reproduces the national row**, all thirteen categories
   to the person. That return is a transcription INE forwarded to New York and shares no
   lineage with these PDFs. [[reference_unsd_religion_oracle]]

The thirteen, with the Yearbook's English:

| INE 2012 | UNSD | people | % |
|---|---|---:|---:|
| Católica Apostólica Romana | Roman Apostolic Catholic | 99,570 | 55.71 |
| Não tem | No Religion | 37,935 | 21.22 |
| Outras | Other | 8,990 | 5.03 |
| Adventista | Adventist | 7,239 | 4.05 |
| Assembléia de Deus | Assembly of God | 5,991 | 3.35 |
| Nova Apostólica | New Apostolic | 5,177 | 2.90 |
| Maná | Maná | 4,191 | 2.34 |
| Igreja Universal do Reino de Deus | Universal of the Kingdom of God | 3,568 | 2.00 |
| Jeová | Jehovah Witness | 2,202 | 1.23 |
| Deus é amor | God is Love | 1,432 | 0.80 |
| Não sabe | Unknown | 1,268 | 0.71 |
| Messiânica Mundial | Messianica | 688 | 0.38 |
| Não declarou | Not Specified | 488 | 0.27 |

**The oracle's thirteen rows are thirteen categories here**, which is worth saying only because
Cabo Verde's sixteen were fifteen plus the under-15s the day before. São Tomé asked everybody:
Quadro 6 prints the religion table by five-year age band starting at 0-4, and those bands sum
to the same 178,739. There is no age cut in this country.

## 3. The locality table is finer AND shallower, which is the opposite of the usual trade

INE's 2016 **`Publicação dos Resultados sobre Localidades - IV RGPH 2012`** prints twenty-five
tables by district and locality, hundreds of localities, and **Tabela 3 is religion**. It is
the finest religion geography this country has, and it is not what is drawn, for two reasons.

**It has seven columns instead of thirteen.** Adventista, Assembleia de Deus, Católica, Nova
Apostólica, Igreja Universal, `Outras religiões` and `Não tem`. The two catch-alls are exact
sums of the district reports' categories:

    Outras religiões  17,503 = Outras 8,990 + Maná 4,191 + Jeová 2,202
                                + Deus é amor 1,432 + Messiânica Mundial 688
    Não tem           39,691 = Não tem 37,935 + Não sabe 1,268 + Não declarou 488

So drawing on it would cost Maná, the Witnesses, Deus é Amor and the World Messianic Church,
which is four of the ten drawn nodes, and would silently fold the non-response into no religion.

**And a locality never earns a dot.** A dot is 1,000 people and the localities average a few
hundred, so the finer composition would be invisible at 1:1,000 while costing four nodes.
`scatter.py` sets each node's dot count from its NATIONAL total and spreads it by a Hilbert
carry, so the unit grain changes where 171 dots land and not how many there are; over seven
districts and 1,001 km² that difference is small.

Both identities are asserted in `check()`, so the two publications are each other's witnesses
even though only one is drawn. This is the same relation Tonga's G 18, G 19 and G 20 have,
except that here the coarser table is the finer geography.

## 4. The 2024 census exists, tabulates religion by district, and is not drawn

**The V RGPH was taken in November and December 2024 and published in July 2025**, and its
`Resultado_VRGPH 2024.pdf` carries religion by district and sex in **sixteen categories**, a
deeper list than 2012's: it adds `Islâmico/Muçulmano` (354 people, 0.17%), splits `Ateu`
(1,473) out of `Sem religião` (20,075), and adds `Nenhuma resposta` and `Não sei` beside `ND`.
It is not in `tools/oracle.py`, which has São Tomé at 2012 and 2001 only.

**It is not drawn because 56,200 of its 209,161 people, 26.87%, are `ND`.**

That is the shape Cabo Verde had the day before, and the answer here is different. Cabo Verde's
28% residual was provably the under-15 population, closing to the person on all 22 concelhos.
São Tomé's does not close. `ND` runs **1.026 to 1.037 times each district's under-10
population** (national 56,200 against 54,605; Água-Grande 22,505 against 21,711; Príncipe 2,655
against 2,585), which is close enough to say the religion question was probably asked from
about age 10 and far enough that the remainder is unexplained, and **the report never says who
was asked**. A 26.9% hole that cannot be characterised is worse than a 0.98% one that can, so
2024 is used as a witness and 2012 is drawn.

### The 2024 report has been round-tripped through machine translation

This is worth recording because it will mislead anyone reading it cold, and because the figures
are unaffected.

| printed | means |
|---|---|
| `Pintura1.8` | *Quadro 1.8*, table. Portuguese `quadro` to English "picture" and back |
| `Copa do Mundo` | *Messiânica Mundial*. `Mundial` read as the World Cup |
| `A vontade de Jeová` | *Testemunhas de Jeová*, Jehovah's Witnesses |
| `Reino Universal de` | *Igreja Universal do Reino de Deus*, truncated |
| `POR FAVOR` | the total row, from `Junto` to "together" to "please" |
| `população em idade sindical` | *idade de união*, marriageable age. `união` to "union" to trade union |

The magnitudes identify the mangled rows unambiguously: `Copa do Mundo` 638 against 2012's
Messiânica Mundial 688, `A vontade de Jeová` 2,133 against 2,202, `Reino Universal de` 3,695
against 3,568.

### What 2024 says about 2012, which is why it is worth having

* **The absence is an absence.** Offered `Sem religião` and `Ateu` as separate boxes for the
  first time, 20,075 São Toméans took the first and 1,473 the second, thirteen in fourteen. So
  2012's single `Não tem` box belongs on `unaffiliated` and not on `secular`.
* **Islam is about a twentieth of `Outras`.** The 2024 Islam box drew 354 people, 0.17%, while
  `Outra religião` stayed at 5,348, 2.6%. So `other.st` is overwhelmingly Christian bodies the
  2012 form does not name.
* **Every district grew, and none of them grew strangely**, 1.13x to 1.35x against a national
  1.17x. That is `st_geo.py`'s witness 3.

## 5. The join, the geography, and the witness that does not quite work

**São Tomé has no twins.** Seven districts, seven distinct names, six folding to COD-AB's
spelling exactly and the seventh being COD's English gloss `Príncipe (Autonomous Region)` for
INE's `Região Autónoma do Príncipe`. There is nothing for
[[reference_name_join_wrong_neighbour]] to bite on, unlike Cabo Verde's three pairs the day
before, so the name join is the join and the rest is confirmation.

**COD-AB and COD-PS disagree about what level this tier is and their pcodes do not join.** The
2026 boundary bundle publishes the seven districts as **ADM1**, pcodes `ST11` and
`ST21`-`ST26`; the 2022 population bundle publishes the same seven as **ADM2** under an ADM1 of
two provinces, pcodes `ST0101` and `ST0201`-`ST0206`. Neither file mentions the other's scheme.
Anything joining the two on a pcode gets an empty frame; anything joining on `adm1_pcode` alone
pairs a district with a province. `st_geo.py` uses COD-AB's throughout and never reads COD-PS,
because the 2024 census is a better witness than a projection of the 2012 one.

**Witness 2 is area, and it is worth reading for what it cannot do.** The 2024 report prints
each district's area beside its population, and measuring each COD-AB polygon in UTM 32N
reproduces all seven to within 8.8%:

| | measured | printed | |
|---|---:|---:|---:|
| Caué | 276.8 | 267.0 | +3.7% |
| Lembá | 215.9 | 230.0 | −6.1% |
| Príncipe | 144.3 | 142.0 | +1.6% |
| Cantagalo | 128.9 | 118.5 | +8.8% |
| Mé-Zóchi | 120.8 | 122.0 | −1.0% |
| Lobata | 99.1 | 105.0 | −5.6% |
| Água-Grande | 17.3 | 16.5 | +4.6% |

That looks like a geometric proof of the pairing and it is not quite one. **Cantagalo and
Mé-Zóchi are inside each other's error, so swapping them fits the printed areas slightly better
than the truth does**: exactly 1 of the 5,039 other permutations beats the identity and it is
that swap. `check_areas` enumerates all of them and asserts that everything which beats the
truth moves only `{ST22, ST26}`, so a third district joining that set stops the file. In a
country where the names *were* ambiguous this witness would not have been enough on its own,
and that is the thing worth carrying forward: **an area check on a small unit set pins most of
a join and can leave a specific pair open, and it is cheap to say which pair.**

**Witness 3 closes the pair.** Swapping Cantagalo and Mé-Zóchi gives growth ratios of 2.95x and
0.46x against a band of 1.05 to 1.50, so it is rejected by a factor of several. The file
asserts that too.

## 6. Placement, and why the strays are snapped

`sources/st_grid.py` writes **546** Kontur 400 m hexes, the smallest grid in the project.
**49 of them, 17,186 people, 7.4% of the grid's population, have a centroid outside every
district polygon**, all within 700 m, and all are snapped to the nearest rather than dropped:
[[reference_archipelago_grid_snap]]. That is an eleventh of the grid's cells against a
seventeenth of Cabo Verde's (123 of 2,156), because São Tomé is two small islands where every
district but Mé-Zóchi
touches the sea and the settlements are the coast. Dropping them would walk the dots uphill
into the Obô forest. Nothing remains unplaced.

The units need the weighting for the same reason Cabo Verde's do. Caué is 267 km² of the
southern massif and the national park with 7,400 people on a coastal strip; Lembá is the
western slope. Kontur against the 2012 census over seven districts gives **r = 0.9868**, which
1 of 2,000 random pairings reaches, and every district is inside a factor of 2 (0.76 Lobata to
1.20 Cantagalo, normalised).

**The vintage gap is eleven years**, counts 2012 and grid 2023, the largest of any country
here. It moves dots within a district and never between districts, but the country grew 17%
over that span and unevenly, so the grid over-weights Água-Grande's newer periphery relative to
2012.

`water.py` reports **3 of the 546 hexes losing more than 95% of their area to the sea and
being left unclipped**, which is its stilt-village rule doing what it is meant to on cells that
are almost entirely water. Three cells out of 546 is small and they are among the snapped
coastal ones.

## 7. The mapping, and the two calls with no precedent

Full reasoning in `taxonomy/st2012.py`'s `REVIEW`, which has eight entries. Eight of the ten
drawn rows follow `br2010.py` and `cv2021.py`; the two that do not:

**`Maná` → `christianity.pentecostal.charismatic`.** 4,191 people, 2.34%, the fifth largest of
the nine named churches, and **the only census category on this whole map that names it**: a
sweep of every file in `data/normalized/` finds `Maná` as a category in `st.csv` and nowhere
else.
Igreja Maná was founded in Lisbon in September 1984 by Jorge Tadeu, a Mozambican-born civil
engineer previously of the Apostolic Faith Mission of South Africa, and is now in some eighty
countries with its own television and radio in Portugal, Spain, Brazil, Mozambique and São Tomé
and Príncipe. **The founder's lineage is classical Pentecostal and the church is not**: 1980s,
personal apostolate, media, prosperity teaching. It goes where `br2010.py` puts the Universal
Church rather than where it puts the Assembleia de Deus. Its geography agrees, 3.01% of
Água-Grande and 2.79% of Cantagalo against 0.62% in Lembá and 0.66% in Caué: a broadcast church
sitting where the transmitters are, which is not the shape a mission takes.

**`Messiânica Mundial` → `eastasiannew.japanese`.** 688 people, 0.38%, the smallest drawn cell
here and a ring rather than a dot at 1:1,000. Sekai Kyūsei Kyō, founded at Atami by Mokichi
Okada in 1935, in Brazil from 1955 and in the lusophone world from there; `br2010.py` maps
Brazil's on the same node, so this is a precedent rather than a new call, but the *fact* is
worth the entry. **0.38% of São Tomé against 0.054% of Brazil** (103,736 of 190.8 million in
2010): the country the church reached is seven times more Messianic than the country it came
through. Its own geography is the capital, 0.62% of Água-Grande against 0.10% in Lobata.

New node `other.st`, 8,990 people, 5.03%, the country's third largest answer. It is large
because the form is nine churches with no family boxes: no Protestant option, no Islam, no
traditional religion. Its geography names nothing, 6.16% in Príncipe to 3.07% in Lobata with
the capital, the island and the empty south together at the top, so §9r's Chittagong rule finds
no cluster and spec §3.11 draws it whole. **Djambi is not in it.** The Forro communal
possession ritual and the ritual specialists of the roças have no box on any São Tomé census
and the people who take part answer Catholic, the pattern `other.do` records for Dominican Vodú
and `other.ht` for Haitian Vodou.

## 8. The §3.5 lean check, with leave-one-out, and why it mostly says nothing

The hole is `Não declarou` 488 plus `Não sabe` 1,268, **1,756 people, 0.98%**. Its share per
district runs 2.01% in Caué, 1.18% Cantagalo, 0.98% Água-Grande, 0.95% Mé-Zóchi, 0.84% Lobata,
0.83% Lembá, 0.60% Príncipe. Correlated against each drawn category's share across the seven
districts, with a leave-one-out on each:

| category | r | leave-one-out |
|---|---:|---|
| Assembleia de Deus | +0.852 | +0.104 to +0.928 |
| Universal do Reino de Deus | +0.787 | −0.149 to +0.930, **sign flips** |
| Adventista | −0.676 | −0.801 to −0.670 |
| Católica Apostólica Romana | −0.591 | −0.764 to +0.092, **sign flips** |
| Jeová | −0.533 | −0.743 to +0.005, **sign flips** |
| Deus é amor | −0.455 | −0.653 to −0.345 |
| Não tem | +0.374 | +0.100 to +0.645 |
| Nova Apostólica | +0.353 | −0.817 to +0.707, **sign flips** |
| Maná | −0.232 | −0.424 to +0.730, **sign flips** |
| Outras | +0.178 | −0.338 to +0.509, **sign flips** |
| Messiânica Mundial | −0.109 | −0.222 to +0.234, **sign flips** |

**Six of the eleven change sign when one district is dropped**, which is the whole reason the
leave-one-out is run: with seven units a single district manufactures a lean, and Caué is the
one doing it. Catholic's r of −0.591 becomes +0.092 without Caué. Only Adventista, Deus é amor,
Assembleia de Deus and `Não tem` keep their sign, and Assembleia's collapses from +0.85 to
+0.10, so it is Caué as well.

**And the magnitude is negligible either way.** Excluding the 1,756 moves no national share by
more than **0.55 points** (Catholic, 56.260% of the drawn population against 55.707% of
everybody counted), and every other category by less than a quarter of a point. `note_public`
says which way it leans in one sentence and does not dress it up.

## 9. What the map shows, in one paragraph

Catholicism is 55.71% and the districts run from **68.31% in Cantagalo to 38.22% in Caué**. No
religion is 21.22%, and its two ends are an island apart: **33.61% in Lembá** on the
north-western coast of São Tomé against **4.60% in Príncipe**, a sevenfold spread inside a
country of 179,000. Príncipe is the most Catholic district after Cantagalo and the most
Adventist anywhere, 9.63% against a national 4.05%. **Caué, 6,031 people in the plantation
country under the southern forest, is where the newer churches have taken most ground**: the
Assembly of God 10.08%, the New Apostolic Church 8.41% and the Universal Church 5.50%, each
their highest figure in the country, and together most of the reason it is the least Catholic
district. Maná runs the other way, 3.01% in the capital against 0.62% in Lembá; so does the
World Messianic Church, 0.62% in Água-Grande against 0.10% in Lobata. At 1:1,000 the country
draws **171 dots and one ring**, the ring being the World Messianic Church's 688 people, and
`other.st` is 5.03% because the form has nowhere else to put a Protestant.

## 10. Files

    data/raw/st/nacional2012.pdf                 IV RGPH 2012, national, 5.5 MB
    data/raw/st/d_*.pdf                          the seven district reports, 2.2-5.0 MB each
    data/raw/st/localidades2012.pdf              the 2016 locality publication, 15.9 MB
    data/raw/st/resultado_vrgph2024.pdf          V RGPH 2024 results, 1.9 MB
    data/raw/st/desdobravel_vrgph2024.pdf        the 2024 leaflet, 5.6 MB, no religion
    data/raw/st/stp_admin_boundaries.shp.zip     COD-AB, HDX
    data/raw/st/stp_admpop_adm1_2022.csv         COD-PS, downloaded and NOT used (§5)
    data/raw/st/stp_admpop_adm2_2022.csv         COD-PS, downloaded and NOT used (§5)
    data/raw/st/kontur_population_ST_20231101.gpkg
    data/normalized/st.csv                       91 rows, 7 districts x 13 categories
    data/geo/st/st_districts.gpkg                7 polygons, with island and both censuses
    data/geo/st/st_lookup.csv                    the same, without geometry
    data/geo/st/st_hexes.gpkg                    546 Kontur hexes

---

## 11. Review, 2026-09-08

Second pass, read from the PDFs rather than from §1-§10. Everything substantive above was
re-derived and holds. What follows is the four things that did not, plus the two source calls
confirmed from the sources themselves because they are the interesting ones.

### 11.1 The 2012 census has no age cut, confirmed three ways

Quadro 8 in `nacional2012.pdf` closes on 178,739 across nationality (173,027 + 3,075 + 2,637)
and across sex (88,867 + 89,872), and the thirteen categories sum to it exactly. Quadro 6 on
page 29 prints the religion table by five-year band **starting at `0-4 anos`, 27,720 people,
of whom 13,119 are Catholic**, so the under-fives were asked and answered. This is the check
Cabo Verde failed the day before, and São Tomé passes it in the strongest available form: the
youngest band is not merely present, it is populated across every religion column. 99.02%
coverage and `gap_share=0.00982` are both exactly what they say.

### 11.2 The 2024 refusal is right, and the reason is stronger than §4 gives it

§4's arithmetic reproduces. Working from the report's own `Tabela 1.2` totals and the
one-decimal ND percentages, ND runs **1.0215 (Mé-Zóchi, Lembá) to 1.0401 (Água-Grande) times
each district's under-10 population**, a slightly wider band than §4's 1.026-1.037 but the
same finding: consistently a few points above the under-tens and never equal to them, and no
clean age boundary fits (under-10 is 26.05% of the country and under-11 is 28.85%, against an
observed 26.9%).

Two things sharpen this.

**`ND` is not non-response, and the note should not imply it is.** The 2024 religion table
already carries `Nenhuma resposta` (0.9%) and `Não sei` (2.4%) as separate rows, which are
2012's `Não declarou` and `Não sabe`. `ND` sits beside them, so it is the not-asked bucket,
not the refused one. And `ND` is a generic marker in this report rather than a religion
finding: `Tabela 2.1`, the age table, has an `ND` row of its own carrying **200 people**. So
the 26.9% is specific to the religion variable and is almost certainly children.

**The report is a French original machine-translated into Portuguese**, which is worth
recording more precisely than §4's "round-tripped through machine translation". The evidence
is the untranslated French left in the age pyramid on page 23: `0 - 4 ans`, `95 ans et plus`,
`Homme`, `Femme`. That identifies the mechanism behind every mangled label §4 lists, because
French *tableau* means both "table" and "painting", which is exactly how `Quadro 1.8` became
`Pintura1.8`. It makes §4's reconstruction of `Copa do Mundo` and `A vontade de Jeová` safer
rather than less safe, but it is also a third reason not to draw 2024: **the legend would
rest on reverse-engineered category names.**

So the refusal stands, and on three independent grounds rather than one: a 26.9% residual
that cannot be characterised, category labels that have to be inferred, and **counts that do
not exist at all** — the 2024 report publishes religion only as percentages to one decimal
place (`Pintura1.8` is captioned `DISTRIBUIÇÃO (%)`), where 2012 publishes exact people. What
2024 would have bought is one ring (`Islâmico/Muçulmano`, about 356 people) and the
`Ateu`/`Sem religião` split. That is not a trade worth making, and §4 reached the right answer.

**One figure to correct.** §4 and `sources/st.py`'s docstring both say *"56,200 of its 209,161
people"*. The report's resident population is **209,607** (`Tabela 1.2`, and `st.py`'s own
`TOTAL_2024 = 209_607` at line 416 has it right), which puts national ND at about 56,384. The
percentage is unaffected, which is why it went unnoticed; the prose figure is stale.

### 11.3 The locality table costs three nodes, not four and not six

§3's reading of the 2016 publication is exactly right and was re-read from `Tabela 3` on page
41: seven religion columns, and both catch-all identities close to the person
(`Outras religiões` 17,503 = 8,990 + 4,191 + 2,202 + 1,432 + 688; `Não tem` 39,691 = 37,935 +
1,268 + 488). The count of what it would cost does not close, and the two records disagree
with each other:

* §3 says *"four of the ten drawn nodes"*. Four **categories** would be lost, but Maná shares
  `christianity.pentecostal.charismatic` with the Universal Church, which the locality table
  keeps. So the node loss is **three**: `christianity.witnesses`, `christianity.pentecostal`
  and `eastasiannew.japanese`.
* `countries.py`'s internal `note` says *"six of the eleven nodes"*. There are ten drawn nodes
  and eleven mapped categories, and neither loses six of anything.

Both left as written rather than edited into someone else's section. **The conclusion is
unaffected and arguably strengthened by the correct number**, because the real cost of the
locality table was never the node count: it is that `other.st` would nearly double from 5.03%
to 9.79% and `unaffiliated` would silently swallow the 0.98% who refused or did not know.
Those two are stated in §3 and are the load-bearing reasons.

### 11.4 One wrong superlative in `note_public`, fixed

`note_public` said of Maná: *"It is the one church on the form that is strongest in the
capital."* Three of the eleven categories peak in Água-Grande, not one — Maná at 3.01%, the
Jehovah's Witnesses at 1.48% and the World Messianic Church at 0.62% — and §7 of this file
says so itself about the Messianic Church (*"Its own geography is the capital"*). Maná is not
even the most capital-tilted of the three: its capital-to-floor ratio is 4.9x against the
Messianic Church's 6.2x.

Changed to *"It is at its strongest in the capital"*, which is true and which the sentence
after it (the New Apostolic Church as the mirror image) still works against. `check_md.py`
clean and `countries.py` imports. **The edit will only reach readers when `tiles.py` next
rewrites `counts.json`**, which the next builder's `build_tail.py` does; not run here, to
avoid taking the lock off two live builders.

Every other figure in `note_public` was recomputed from `data/normalized/st.csv` and is
correct against people, including all eight superlatives: Cantagalo 68.31% and Caué 38.22% as
the Catholic ends, Lembá 33.61% and Príncipe 4.60% for no religion, Príncipe 9.63% Adventist,
Caué's 10.08% / 8.41% / 5.50% each genuinely that category's national maximum, the gap at
2.01% in Caué and 0.60% in Príncipe, and the largest shift from excluding the gap at +0.553
points (Catholic). Brazil's Messianic share checks at 103,736 of 190,755,799, 0.0544%, so
0.38% is 7.1 times it. A corpus sweep of every `data/normalized/*.csv` finds `Maná` as a
source category in `st.csv` and nowhere else, as §7 claims.

### 11.5 The §3.5 lean check reproduces exactly

Recomputed independently from `st.csv`: all eleven correlations and all eleven leave-one-out
ranges in §8 match to three decimals, and the same six flip sign. Caué is the district doing
it, as §8 says. Nothing to add.

### 11.6 Two placement observations, not diagnosed

`tools/check_rollup.py`, `check_mapping.py`, `gap_share.py`, `check_md.py` and
`built_countries.py --check` are all clean. No screenshot was taken; the two checks a shot
would have given are better done in code on a country this small, and both pass. **All 171
dots fall inside a district polygon**, none in the sea, and every one of the 546 hex centroids
sits inside the district it is assigned to. §6's grid correlation reproduces: normalised
hex-population-per-census-person runs 0.76 in Lobata to 1.20 in Cantagalo, exactly as stated.

Two things a human eye may want, neither investigated further per the review brief:

* **Caué draws 3 dots against the 6 its 6,031 people imply**, and Água-Grande draws 75
  against 69.5 while Mé-Zóchi draws 38 against 44.8. Água-Grande is a 17 km² enclave
  entirely surrounded by Mé-Zóchi, so 400 m hexes straddling that boundary can drop a dot on
  either side of it; the rest looks like the Hilbert carry at small N. It is worth knowing
  because **the longest and most emphatic paragraph in `note_public` is about Caué**, which is
  the district drawing the fewest dots and drawing half what its population implies.
* Maná and the Universal Church share `christianity.pentecostal.charismatic`, so the two
  bodies whose opposite geographies `note_public` contrasts (Maná in the capital, the
  Universal Church in Caué) are one colour and seven dots on the map. The note is a census
  reading rather than a caption for what is visible, which is the house convention, but at
  171 dots the distance between the two is larger here than usual.
