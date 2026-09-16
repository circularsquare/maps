# Mongolia — sources

Status: **in progress**, session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-mn`, 2026-09-08.

## 1. The question exists, and the office asks it directly

Mongolia's **2020 Population and Housing Census** (Хүн ам, орон сууцны 2020 оны улсын ээлжит
тооллого, enumerated January 2020) carries religion as **question P29, `Та шашин шүтдэг үү?`**
— *do you practise a religion?* — asked of the population **aged 15 and over**. That is read
off the NSO's own DDI for the census, not off a search summary:

    http://web.nso.mn/nada/index.php/catalog/ddi/175      3,647,800 bytes, HTTP 200

whose variable `V98 P29_SP10_RELIGION` has exactly six answer categories, in this order:

| code | label | gloss |
|---|---|---|
| 1 | `Шүтдэггүй` | does not practise |
| 2 | `Будда` | Buddhist |
| 3 | `Христ` | Christian |
| 4 | `Ислам` | Muslim |
| 5 | `Бөө` | Shamanist |
| 6 | `Бусад` | other |

**The oracle is wrong-by-omission here.** `tools/oracle.py mn` reports Mongolia ABSENT from
UNSD's table 28, which per §11r's standing reading proves only that no tabulation was
forwarded to New York. The census asks the question and has done since 2010.

**The frequencies in that DDI are a sample and must not be used.** Study 175 is a public-use
microdata *sample*: V98's six categories sum to 23,221 people and V3 (AIMAG) sums to the same
order, against a census population of 3,296,866. The DDI is evidence for the *category list*
and for the *codes*, and for nothing else. Microdata itself is gated (`get_microdata`), and
per `[[reference_ipums_account]]` that route is not open to this project.

## 2. Geography is solved, and the join is by code

**COD-AB Mongolia** (`cod-ab-mng`, OCHA FISS, refreshed 2026-01-26, CC BY-IGO) — one GET from
HDX, 5,757,562 bytes, no wall:

    https://data.humdata.org/dataset/a9b0a8a6-cb14-448e-b35c-aa5eb51b0557/resource/
      2ec00922-5b9b-47fc-aa36-0a8b59a877df/download/mng_admin_boundaries.shp.zip

It ships `mng_admin1.shp` (**22** aimags and the capital) and `mng_admin2.shp` (**339** soums
and Ulaanbaatar's 9 düüregs), both with English and Cyrillic names.

**The pcodes ARE the Mongolian official aimag codes, so there is no name join at all.** COD's
`adm1_pcode` is `MN` + the two-digit aimag code, and every one of the seventeen aimag codes the
census DDI happens to print (11 Улаанбаатар, 21 Дорнод, 22 Сүхбаатар, 23 Хэнтий, 41 Төв, 42
Говьсүмбэр, 43 Сэлэнгэ, 44 Дорноговь, 45 Дархан-Уул, 46 Өмнөговь, 48 Дундговь, 61 Орхон, 62
Өвөрхангай, 63 Булган, 64 Баянхонгор, 65 Архангай, 67 Хөвсгөл) matches COD's code **and its
Cyrillic name, exactly, with zero mismatches**. `adm2_pcode` is `MN` + aimag + soum, its first
four characters always equal the parent `adm1_pcode`, and all 339 are unique.

This is the one thing that usually goes wrong here
(`[[reference_name_join_wrong_neighbour]]`) and in Mongolia it cannot: the source and the
boundary file share a numeric key.

## 3. Placement: Kontur, and Mongolia needs it more than almost anywhere

    https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/
      kontur_population_MN_20231101.gpkg.gz          10,588,936 bytes, HTTP 200

**121,265 400 m hexagons totalling 3,452,426 people**, against a 2020 census 3,296,866 — a
ratio of 1.047 for a 2023 grid over a 2020 census, which is what Mongolia's growth rate
predicts and is well inside the tolerance `kz_grid.py` uses.

Mongolia is 1.56 million km² with 3.3 million people, a density of **2.1/km²** — a third of
Kazakhstan's, which `kz_grid.py` calls the strongest case for §8.2's grid in the project. Half
the country lives in Ulaanbaatar, and Ömnögovi is 165,000 km² holding about 70,000 people. An
equal-share-per-polygon wash would paint the Gobi in dots of one colour. Kontur is not optional
here.

## 4. What the 1212.mn statistical database does NOT have, and how its API works

The database at `www.1212.mn` is a Next.js SPA over PxWeb `.px` tables. Its internal API,
recovered by grepping the JS bundle (`[[reference_spa_hidden_apis]]`):

    /api/sectorname?lng=en
    /api/subsectorname?subsectorname=<sector>&lng=en
    /api/sectortablename?sector=<s>&subsector=<ss>&lng=en      -> DT_NSO_####_###V#.px ids
    /api/catalogue?list_id=<id>
    /api/download?info=<subsector>&lng=en&type=report|reportSector|updatereports
    POST /api/elastic_search   body {"values": "<term>"}

**`1212.mn` and every `*.nso.mn` host serve a broken TLS chain** — a missing intermediate, the
same failure as `stat.gov.pl` and Ghana's `statsbank`. `curl` needs `-k`, `requests` needs
`verify=False`. A cert error on these hosts is not the host being down. `web.nso.mn` is
HTTP-only and refuses HTTPS outright. `opendata.1212.mn` (the old documented open-data API host)
no longer resolves, and `www2.1212.mn` (the legacy ASP.NET database) times out or returns 500.

**Its religion tables are about institutions, not people**, and are the wrong quantity for this
map: `СҮМ ХИЙДИЙН ТОО` (temples and monasteries by religion type and aimag), `ХУРЛЫН ЛАМ НАРЫН
ТОО` (monks), `ШАШНЫ СУРГУУЛЬ ДАЦАНД СУРАЛЦАГЧДЫН ТОО` (religious-school pupils). Searching the
database for the census wording (`шашин шүтлэг`) returns nothing: the census religion
tabulation is not in the statistical database.

The census landing page `https://www.1212.mn/mn/statistic/fun-statistic/census2020` is **only a
Tableau embed** — workbook `PopulationMongolia2022/Dashboard1` on `tableau.1212.mn`, loaded
through a trusted-authentication ticket minted server-side, so there is no static payload to
read and the workbook is a population dashboard rather than a religion one.

## 5. Ethics (§14): nothing here needs a flag

Mongolia's state publishes religion itself, in its own census, and the one strongly
geographically-concentrated minority — the Kazakh Muslims of Bayan-Ölgii — is a openly counted,
uncontroversial and constitutionally protected population whose concentration is published by
the office and is not secret. COMMANDS.txt's §14 gate ("a country whose state does not publish
religion, or whose religious minorities are persecuted") does not catch this country.

## 6. The tabulation is in twenty-two separate volumes, one per aimag

**The national report has no religion geography at all.** Its chapter 3 gives religion by sex,
by age group and by ethnicity, nationally, and stops. So does the 2010 national report, whose
`NSO_Negdsen_dun_ENG.pdf` (38.8 MB, 290 pages, via NADA `catalog/89/download/702`) was checked
page by page.

**What does reach the aimags is the per-aimag census volume**, `<Aimag>ХАОСТ Нэгдсэн дүн`, one
per unit, published on the static host `https://downloads.1212.mn/`. Nineteen follow the
pattern `<Aimag>_XAOCT_Negdsen_<dun|Dun>.pdf`, inconsistently cased, with Darkhan-Uul using a
space and Tuv a space plus a doubled dot. **Three do not follow it at all** and were found by
sweeping the Wayback CDX of the whole `1212.mn` domain for the retired
`BookLibraryDownload.ashx?url=<filename>` links, whose `url=` parameter is exactly the filename
on `downloads.1212.mn`:

| Dundgovi | `downloads.1212.mn/Dundgovi.pdf` | 34,758,309 bytes |
| Khentii | `downloads.1212.mn/CENSUS-2020_KHENTII_MAIN_REPORT.pdf` | 44,470,917 bytes |
| Khovd | `downloads.1212.mn/Khovd.pdf` | 24,536,796 bytes |

273 spellings of the nineteen-file pattern were tried for these three first and every one 404s.

**`Khentii.pdf` also exists, returns 200, and is the wrong census.** It is the 2010 volume, and
its text layer is a legacy non-Unicode Cyrillic font that extracts as mojibake. Do not use it.

**The religion tables reach aimag and stop there.** Every volume carries soum breakdowns for
population, sex, migration, education and housing, and its religion tables are aimag-wide. That
is not an oversight: the question went to a 10% sample, which cannot support 339 soums. There
is no soum-level religion figure anywhere and there is unlikely ever to be one.

## 7. What is not drawn, and how it could be

**Darkhan-Uul and Dundgovi are the two aimags left out**, 148,869 people, 4.7% of Mongolia. In
both cases the volume exists, is the right year, and downloads fine; the tables inside it are
**pasted-in images**. Dundgovi's whole file is a scan (its producer is `iLovePDF`, and the only
text on any page is the running header and the page number). Darkhan-Uul's is subtler and worse
to diagnose: the prose and the captions are real text, so the file looks fine, and only the
tables are pictures. Its two religion pages carry nine image objects and two extractable rows.

Two ways to bring them in, neither taken here:

1. **OCR.** Mongolian Cyrillic in a table, at 300 dpi, is within reach of tesseract with `mon`
   traineddata, but it is a new dependency and a new class of silent error for 4.7% of one
   country.
2. **Transcribe four numbers by hand.** Each aimag needs only its `Шүтдэг` share and the five
   type shares, six figures, from one page each. That is the same kind of act as `sources/hu.md`
   §2's hand-made exports, and it would be checkable against the same identities `sources/mn.py`
   already asserts (the pair sums to 100, the five sum to 100, and the national reconstruction
   moves toward NSO's published figures rather than away). The denominators are already on disk:
   `read_population()` covers all 22 aimags because it reads the national report.

**Khovd was nearly a third casualty and is drawn.** Its volume publishes no religiosity table,
only the type-of-religion one, and states the split in a sentence instead. `sources/mn.py`'s
`OVERRIDE` takes the two figures from that sentence, and the sentence checks itself: its third
figure (49.3% of all adults Buddhist) is the product of the two the override uses (60.6%
religious) and the table's own Buddhist share (81.3%). Khovd matters because it is the second
Kazakh aimag; dropping it would have misstated where Mongolia's Muslims live.

## 8. What the parse had to survive, and how it is anchored

Twenty-two statistics departments typeset twenty-two volumes and agreed on almost nothing.
Table numbers differ (3.4/3.5 in Bayan-Ölgii, 3.7 in Khövsgöl, 3.10/3.11 in Bayankhongor).
Captions differ in wording, in declension, in whether the aimag's own name is prefixed, and
Govi-Altai spells `ХҮН` as `ХУН`. Six volumes merge the two religion tables into one.
Ulaanbaatar transposes its type table. Selenge prints 2020 alone where everyone else prints two
years, and misspells `Шүтдэггүй` as `Шүтлэггүй`. Sükhbaatar omits the `Ислам` row entirely
rather than printing a zero, and Zavkhan omits `Бөө`. Övörkhangai prints the religiosity rows
twice on one page. Govi-Altai uses `-` as an in-band zero.

**So no table is found by its caption.** Each is found by an identity only it satisfies: the
two religiosity shares must sum to 100.0, the five type shares must sum to 100.0, and the
population bands must sum to their own printed total. A wrong page fails the sum; a wrong
caption match returns numbers.

**Two traps cost real time and are worth repeating.**

*The year columns.* Reading 2010 as 2020 is the one error nothing downstream catches, because
Mongolia's shares barely moved between the censuses. The first attempt compared the x positions
of every `2010` and `2020` token on the page, which is wrong: Govi-Altai's caption ends `...,
2010 ОН, 2020 ОН`, the caption wraps, and `2020` therefore starts a line further left than the
`2010` above it. That silently produced Govi-Altai's 2010 figures. The rule now is to trust the
universal convention (2010 left) and confirm it only against a genuine spanner row, one whose
whole label is the two years.

*The denominator.* The national report's appendix table 1.1 carries its own continuation block
on the same page, so every aimag appears twice on it, once with `Total, 0-4, 5-9, 10-14` and
once with `35-39, 40-44, ...`. Taking the last occurrence read four age columns as a total and
three child bands and made every 15+ figure negative. Caught only because the sign was absurd,
which is the argument for the plausibility assertion that now sits beside it.

## 9. The check that says the whole thing is right

Summed over the 20 aimags drawn, the reconstruction gives:

| | this map | NSO published, national |
|---|---:|---:|
| Buddhist | 51.81% | 51.7% |
| No religion | 40.44% | 40.6% |
| Muslim | 3.33% | 3.2% |
| Shamanist | 2.44% | 2.5% |
| Christian | 1.31% | 1.3% |
| Other | 0.66% | 0.7% |

Every category within 0.15 points, from a completely independent path: two chained percentage
tables in twenty Mongolian-language PDFs, times a denominator out of an English appendix table.
That agreement is what says the year columns, the chaining and the denominators are all right
at once, and it is the reason this country was drawn rather than parked.

Individual volumes were also checked against their own prose, which states the Buddhist share
in a sentence beside the table: Bayan-Ölgii 92.5% Islam, Selenge 85.1% Buddhist, Övörkhangai
99.1%, Govi-Altai 95.1%, Khövsgöl 91.3%. All match.


## 10. Review pass, 2026-09-08

Reviewer session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-mn-rev`, on `.claude/commands/rd-review.md`.

Everything structural came back clean: `check_md.py`, `built_countries.py --check`, and
`check_rollup.py mn` (2,067,715 measured, no derived rows, no orphans). The taxonomy is not a
concern either. `indigenous.northeurasian` is shared with Russia (`taxonomy/ru2012.py`) rather
than being a single-country legend row, `other.mn` is the standing per-country convention that
94 countries already use, and `Будда` -> `buddhism.vajrayana` is filed the same way
`cn2000.py`, `np2021.py` and `usrc2020.py` file the same body. A screenshot of the country
draws correctly: dots on land, the Ulaanbaatar cluster sitting in its enclave inside Töv,
Bayan-Ölgii visibly Muslim, Darkhan-Uul and Dundgovi blank exactly as `gap` says, and the
legend totals equal to the CSV to the person. §9's national reconstruction reproduces from
`data/normalized/mn.csv` on all six categories.

**Every figure in `note_public` was recomputed from the normalized CSV.** Most are exact: 92.5%
and 13.6% Islam as a share of the religious in Bayan-Ölgii and Khovd, 88.7% and 59.6%
religious, 10.9 / 7.1 / 0.4% shamanist of the religious, 40.4% no religion, Govisümber's sample
of about 1,100 (11,443 adults at 10%, and it is indeed the smallest aimag), 4.7% for the two
missing aimags, and `grain`'s 103,000 adults per aimag. Three did not reproduce, and were
corrected in `countries.py` on this date:

1. *"the most religious unit in the country, 88.7%"*. Övörkhangai is 89.6%, so Bayan-Ölgii is
   second. Both numbers are the source's own published `religious_pc` and both sit in the CSV
   note column. Now reads "the second most religious unit drawn ... behind Övörkhangai's 89.6%".
2. *"2.4%, which is more than twice the Christian share"*. Shamanism is 2.44% of drawn adults
   against Christianity's 1.31%, a ratio of 1.86, and the ratio is identical on the
   religious-only denominator (4.09 against 2.20). Now "nearly twice".
3. *"almost nothing in the other twenty aimags"*. Only 20 aimags are drawn, so after Bayan-Ölgii
   and Khovd there are eighteen. The other two inside that twenty are Darkhan-Uul and Dundgovi,
   which the same note says cannot be read at all. Now "the other eighteen drawn".

**Was: "About 240,000 people answered."** That figure was in `note_public`, appeared
nowhere in this file or in the CSV, and did not reconstruct. Ten per cent of the 2,067,715
adults drawn is about 207,000; grossing up to all 22 aimags on the note's own 3,197,020 resident
population at the same 15-and-over share gives about 217,000. Reaching 240,000 needs a 15+
population of 2.4 million, which is 75% of Mongolia aged 15 and over against a real share nearer
68%. The sentence's point, that this is a larger respondent count than any survey on the map,
survives at either number.

### 10.1 Resolved 2026-09-08: the NSO does not state a sample size, so the note now carries a reconstruction

**Looked for the office's own figure first, and there is not one.** Four places were opened
rather than inferred:

- `Census2020_Main_report_Eng.pdf`, every page. It states the *design* three times and a count
  nowhere: p31 *"a long-form questionnaire (traditional method) was used for 10 percent of the
  total households"*, p61 *"the same questions were asked in this census from 10 percent sample
  of the population aged 15 and over"*, p158 the same sentence again for the abroad chapter.
  No absolute respondent or household figure accompanies any of them.
- The NADA study-175 **Sampling** tab, `web.nso.mn/nada/index.php/catalog/175/sampling`. Its
  entire *Sampling Procedure* field reads *"Ulaanbaatar city and all provinces, sums, districts,
  teams and committees."* There is no sample size, no design and no fraction. (The host is
  HTTP-only and needs a browser UA; `curl -k` gets it, WebFetch cannot because it upgrades to
  HTTPS.)
- The 1212.mn statistical database, per §4: it carries no census religion tabulation at all.
- Web search in English and Mongolian: the 897,427 household total is everywhere, the
  long-form count is nowhere.

**So the figure in `note_public` is now a reconstruction and says so.** `read_population()`
gives all 22 aimags' 15-and-over population from appendix table 1.1 as **2,170,573**, and a
tenth of that is **217,057**. The note reads *"The office never prints how many people that
was; a tenth of Mongolia's 2,170,573 adults is about 217,000"*. Two things about that number
worth keeping: it is 10% of the whole country and not of the 20 aimags drawn, whose own 15+
universe is 2,067,841 and whose tenth is 206,784; and it is an expectation rather than a count,
because the sample was drawn on **households** and every 15+ member of a selected household
answered, so the realised adult count moves with how many adults the selected households had.

**One softer note, left alone.** The shamanism sentence quotes 0.4% in Övörkhangai as the low
end, but the true minimum over the 20 drawn aimags is Zavkhan at 0.0%, and Övörkhangai is the
lowest non-zero. The sentence does not claim to be a minimum and the geographic point it makes
holds, so it stands.

**The gap reads as honest.** Darkhan-Uul and Dundgovi are missing because their volumes are
scans, which has no plausible correlation with religion, and §9's agreement with NSO's published
national figures to within 0.15 points is itself evidence the hole does not bias the total.

**These corrections are in `countries.py` only.** `tiles.py` bakes `note_public` into the counts
JSON, so the text a reader sees updates on the next build tail run, which any country landing
after this will do. No build was run from this session.


## 11. Ulaanbaatar drawn by düüreg, 2026-09-15

Session `cb8b206e-mn`, on the lead in `sources.md` §scout-2026-09-15-upgrades. Code:
`sources/mn_ub.py` (the reader), `sources/mn.py::read_ub` (the city checks and the rows),
`sources/mn_grid.py::split_capital` (placement).

**What is drawn now.** 19 aimags and the capital's 9 düüregs, 28 units, 2,067,774 adults on the
same six nodes; 2,064 dots at 1:1,000 and 204 at 1:10,000. `mn.csv` carries no city-wide
Ulaanbaatar row (geo_level `duureg` replaces it), so `tools/check_mapping.py::DEFAULT_LEVELS` has
`mn: aimag, duureg`. §9's national reconstruction moves by 0.05 points at most (Buddhist 51.81% to
51.86%).

**Source.** `data/raw/mn/aimags/Ulaanbaatar_XAOCT_Negdsen_dun.pdf`. Зураг 3.5 (PDF p.54): the
share of people aged 15 and over who practise a religion, by düüreg. Зураг 3.6 (PDF p.55): the
religious by type, by düüreg. Nothing else in the volume crosses religion with düüreg.

**Measured off the drawing, not by pixel.** Both charts are vector: every bar and segment is a
filled rectangle (figure 3.6: Будда `#d8c0cc`, Христ `#965777`, Ислам `#b08199`, Бөө `#7c2d55`,
Бусад `#808285`, read off the legend swatches; figure 3.5: one fill, `#be96aa`). The value
labels, row names and legend words are glyph outlines with no text layer. So:

- shares are widths at one scale per chart, fitted by least squares through the origin on the
  printed labels, which were read by eye off 300 dpi renders. The worst label against its own
  segment is 0.013 points on 3.5 and 0.030 on 3.6. Songinokhairkhan's 3.5 label runs under the
  city-average callout and is checked but not fitted.
- the scale is fixed, not per row. Baganuur's bar is 0.09% wider than the others because its
  printed values sum to 100.1; normalising each bar to 100 read its Buddhist share as 85.4
  against a printed 85.5.
- the transcribed row names and legend words are checked by glyph path signature: 26 distinct
  outlines, each standing for one letter across all 23 strings on both charts.
- each printed label must lie inside the segment of the religion it is credited to.
- the city-average line on 3.5 measures 53.06 against its printed 53.7, so that callout is placed
  by hand and is not used.

| düüreg | aged 15+ | religious % | Buddhist | Christian | Muslim | Shamanist | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baganuur | 19,878 | 51.30 | 85.50 | 5.99 | 0.43 | 7.77 | 0.41 |
| Bagakhangai | 2,809 | 47.11 | 95.52 | 1.89 | 0 | 2.60 | 0 |
| Bayangol | 151,891 | 55.41 | 91.59 | 2.92 | 0.68 | 3.82 | 1.00 |
| Bayanzürkh | 244,078 | 50.51 | 87.99 | 3.99 | 0.93 | 5.49 | 1.60 |
| Nalaikh | 25,321 | 55.30 | 77.69 | 4.31 | 13.59 | 4.03 | 0.39 |
| Songinokhairkhan | 220,200 | 55.11 | 90.31 | 2.28 | 0.39 | 5.31 | 1.71 |
| Sükhbaatar | 99,850 | 52.70 | 88.99 | 3.60 | 0.39 | 6.42 | 0.61 |
| Khan-Uul | 122,927 | 53.19 | 89.10 | 3.32 | 0.57 | 5.42 | 1.60 |
| Chingeltei | 102,450 | 58.19 | 89.31 | 3.17 | 0.11 | 6.92 | 0.50 |

The type shares are of the religious. Against the scout's pixel reading, the widths move
Bagakhangai's Christian from 1.8 to 1.89 (its Muslim and other have no segment at all, so 0),
Nalaikh's other from 0.3 to 0.39 and Sükhbaatar's other from 0.5 to 0.61; the rest agree within
0.1. Baganuur's Muslim 0.43 and Khan-Uul's 0.57 sit within 0.03 of a rounding boundary, so the
chart cannot say which one-decimal value lies under them; the width is used, and the difference
is a few dozen people.

**Nalaikh's 13.6 is Islam.** The prose (PDF p.20) says Christian, and gives the Kazakhs living
there as the reason. The chart's segment is in the Islam fill, the `13.6` label sits inside it,
Christian is a separate 4.3, and the city check below fails on the prose's reading.

**Denominator: appendix table 1.1 (PDF p.211)**, resident population by düüreg and five-year age
group, printed in two blocks so every düüreg appears twice; which block is which is settled by the
fifteen bands summing to the total. Tables 1.2 (men) and 1.3 (women) are read the same way and add
to 1.1 in every cell, and the nine düüregs give 1,466,125 residents and 989,404 aged 15 and over,
the national report's Ulaanbaatar row exactly. **Trap:** table 1.2 prints Songinokhairkhan's men
aged 45-49 as `9145`, with no thousands space (PDF p.212). `sources/mn.py::_rows` reads a number
only as one-to-three-digit groups, filed the cell in the row label and shifted the row a column;
`mn_ub.py::_table_rows` takes four digits and the identities check it (18,789 less 9,644).

**The city check.** Weighting each düüreg's shares by its adults gives 53.76% religious (printed
53.7), and Buddhist 89.17 (89.1), Christian 3.26 (3.3), Muslim 0.90 (0.9), Shamanist 5.43 (5.4),
other 1.25 (1.3): worst 0.07 against a 0.15 bar. With Nalaikh's 13.6 as Christian it gives Muslim
0.65 and Christian 3.51, off by 0.25, and fails. None of 2,000 shuffles of the populations among
the düüregs passes. The düüregs are not raked to the city's printed tables: the differences are
inside those tables' one-decimal rounding.

**Join and placement.** COD-AB's nine children of MN11 carry exactly the Cyrillic names the charts
and table 1.1 print, and the volume's per-düüreg appendix tables 1.4 to 1.12 run in COD's pcode
order (`mn_ub.py::check_join`). All 3,401 Kontur hexes assigned to Ulaanbaatar have their centroid
inside a düüreg polygon, none snapped. Kontur against table 1.1 per düüreg, normalised by the
city's own ratio (1.290): Bayangol 0.77, Khan-Uul 0.80, Bayanzürkh 0.84, Sükhbaatar 0.94, Nalaikh
0.95, Baganuur 0.98, Bagakhangai 1.03, Chingeltei 1.17, Songinokhairkhan 1.40. None is outside a
factor of two (the bar was at most one, set before reading), against a median 6 of 9 on a shuffled
join. `kontur_cap.py mn`: no stops.

**Sampling.** The 10% long form puts about 280 of Bagakhangai's 2,809 adults in the sample and
about 130 religious answers under its type split, so its 1.89% Christian is two or three people.
Baganuur, at about 2,000 adults sampled, is the next smallest.

**Also fixed.** `taxonomy/mn2020.py` REVIEW quoted Ulaanbaatar's Christians as 4.9% of the
religious. That is the capital's 2010 column; the 2020 figure, which the map has always drawn, is
3.3.

**Still open.** Darkhan-Uul and Dundgovi (§7). The note's "about 280" is an expectation on a
household sample, as §10.1 says of the national figure.
