# Andorra (`ad`)

Session `cb8b206e-ad`, 2026-09-15. Drawn: World Values Survey wave 7 (2018), one unit, every row
`modelled`. Code `sources/ad.py`, mapping `taxonomy/ad2018.py`, entry `countries/ad.py`, node
`other.ad` in `taxonomy/branches.py`. Found from `sources.md` §scout-2026-09-15-europe's row.

## 1. Source

- **No census has asked religion.** `tools/oracle.py Andorra`: absent from UNSD table 28.
- **WVS-7 Andorra 2018, read from the IHSN catalogue, not from the WVS download form.**
  `https://catalog.ihsn.org/catalog/11550` (`AND_2018_WVS-W7_v01_M`, data file
  `WVS_Wave_7_Andorra_Stata_v5.0`). Each variable page prints the file's unweighted category
  counts; the catalogue's study description gives the producers (Joan Rafel Micó Ibáñez, Núria
  Seguès Daina, Pepita Batalla Salvadó; funded by the Institut d'Estudis Andorrans and Fundació
  Julià Reig), fieldwork 2018-01-06 to 2018-09-22, universe 18 and over, and says the data file
  itself comes from the WVS site. The pages read, saved in `data/raw/ad/`:
  - `Q289` (V338), *Religious denominations - major groups*: no answer 2, none 302, Catholic 641,
    Protestant 10, Orthodox 17, Jew 1, Muslim 11, Hindu 9, Buddhist 6, Other 5; sum 1,004, valid
    1,004.
  - `Q289CS9` (V339), *detailed list*: the same ten counts on 10100000 `Roman Catholic; Latin
    Church`, 20000000 `Protestant; nfd`, 30100000 `Eastern Orthodox; nfd`, 40000000 `Judaism`,
    50000000 `Islam; nfd`, 60000000 `Hindu`, 70000000 `Buddhist`, 90000000 `Other; nfd`,
    100000020 `Non-religious`, -2.
  - `W_WEIGHT` (V30): one category, `No weighting`, 1,004.
  - `N_REGION_ISO` (V16): Canillo 63, Encamp 197, La Massana 51, Ordino 31, Sant Julià de Lòria
    138, Andorra la Vella 295, Escaldes-Engordany 229. `N_REGION_WVS` (V17) has the same counts
    under its own codes.
- **Documents**, from the same catalogue entry, in `data/raw/ad/`: questionnaires in Catalan
  (F00006597) and English (F00006598), the methodology report (F00008607) and the sample design
  note (F00010377). French and Spanish questionnaires (F00006599, F00006600) and the team sheet
  (F00010374) were not saved.
- **The card** (Q289, Catalan): `No, no pertanyo a cap religió`, `Sí, catòlic`, `Sí, protestant`,
  `Sí, ortodox (rus/grec/etc.)`, `Sí, jueu`, `Sí, musulmà`, `Sí, hindú`, `Sí, budista`,
  `Altra, quina?` (code 8, a write-in). The English card prints `Roman Catholic` for code 1. The
  file holds the write-in as code 9.
- **Design.** Sample design note: target population 64,483 aged over 18; 1,000 interviews;
  random routes, 10 interviews per PSU, quotas on sex, age and nationality within each parish;
  face to face in Catalan, Spanish, French or English; no weighting "since the quotas enabled the
  collection of proportionally represented sample in terms of sex, age and main nationalities".
  Methodology report Q39 ("Census 2017, 66.629 inhabitants" aged 18 and over) prints population
  against sample: female 48.88 / 49.3, Andorran 38.97 / 39.1, Spanish 30.00 / 30.2, French
  5.87 / 5.4, Portuguese 14.49 / 15.3, other 10.67 / 10.7; age 60 and over 24.16 / 20.3. Q21's
  limitations: people who have left still on the census, trouble finding Portuguese and Spanish
  people over 65, second homes of non-residents replaced.

## 2. Population base

- **76,177, the Department of Statistics' estimated resident population for 2018.** Read off
  the Observatori Social's *Evolució de la població estimada d'Andorra*
  (`observatorisocial.ad/files/153/Poblacio-dAndorra/2021/1-Evolucio-de-la-poblacio-d-%27Andorra-2025.pdf`,
  "Font: Departament d'Estadística"), series 2010-2025: 2017 74,794, 2018 76,177, 2019 77,543,
  2024 87,097, 2025 89,058. El Periòdic d'Andorra quotes the office's 2024 year-end as 87,097, so
  the series reads as 31 December; 2025 was not checked that way. `sources/ad.py::check_population`
  pins 2018, 2024 and 2025 off the PDF's text.
- **Why 2018 and not 2025.** The fieldwork year, as `sources/bq.py` chose for its survey. The
  country grew 17% to 2025, mostly in `other nationalities` (10.7% of the 2018 quota; 16.9% of
  residents in February 2025, office note `NP_A001_A003_20250313` table 1.3, with Argentine,
  Colombian and Peruvian growth named), people the quotas barely reached. Puerto Rico (`pr`) laid
  2018 shares on 2024 estimates instead; either is defensible, and the dot count moves from 71 to
  about 84.
- **Kontur** (`kontur_population_AD_20231101`): 341 populated hexes, 80,143 people, 1.05x the
  2018 base. Placement only.

## 3. Mapping (`taxonomy/ad2018.py`)

Catholic to `christianity.catholic.latin` (Q289CS9 says Latin Church); none to `unaffiliated`;
Orthodox to the `christianity.orthodox` parent, not `.canonical`, since the card names no
jurisdiction; Protestant to `christianity.protestant`; Muslim, Hindu, Buddhist, Jew to their
roots; Other to a new `other.ad`, because Andorra's write-ins are `Other; nfd`, not `Other
Christian; nfd` as Puerto Rico's were; no answer excluded, the `gap` (0.20%,
`tools/gap_share.py` agrees). Reasons in the module's `REVIEW`.

## 4. Grain: one unit

- **Drawn as one unit**, under Anita's microstate ruling (2026-09-08, `ask/RULINGS.md`), which
  sets no size cut-off; Andorra at 76,000 is Isle of Man-sized, and Gibraltar and the Isle of Man
  are drawn the same way.
- **The file has parishes but the public route has no religion by parish.** The catalogue pages
  are one variable at a time. A religion by parish table would need the data file (the WVS form,
  which Puerto Rico's download treated as Anita's) or the WVS online analysis tool
  (`worldvaluessurvey.org/WVSOnline.jsp`, no registration, crosses Q289CS9 with N_REGION_WVS in
  unweighted counts, read with headless Chrome in `sources/branches.md` Pakistan). **Not tried.**
  At seven parishes of 5,000 to 25,000 people and 31 to 295 interviews, a split-half would have
  almost no power (Anita's Turkmenistan ruling said as much of six units), and the whole country
  is 71 dots.
- **The interviews are not spread as the population is**, although the design note says
  proportional: La Massana and Ordino are 8.2% of interviews and 20.0% of residents (February
  2025 estimate; no 2018 parish table was read), Encamp and Escaldes-Engordany 42.4% against
  33.2%. The nationality quota held nationally (above), so the national share rests on that. If
  religion differs by parish, the unweighted national share carries it; not measurable from the
  catalogue.

## 5. Build

`python sources/ad.py --fetch`; `check_mapping.py ad` 10 categories, 9 nodes, 0 unmapped; 1:1,000
scatter 71 dots on 43 hexes (Catholic 48, none 22, Orthodox 1) and 6 rings (Islam, Protestant,
Hindu, Buddhist, other, Judaism); 1:10,000 edition built. No Kontur cap block, no water clip.
`country_shapes.py` needs no entry: Natural Earth's `Andorra` feature has `ISO_A2` AD.

## 6. Not checked, with where to start

- **WVS wave 8**: Andorra fieldwork from autumn 2025, results expected 2026
  (`ari.ad/en/projects/world-values-survey-wvs-andorra`). When its file or catalogue entry
  appears, it replaces 2018.
- **WVS wave 5** (2005-06): `catalog.ihsn.org/catalog/8992`. Its variable listing did not show a
  religion variable on the first page read; its frequencies would be a witness of level.
- **ARI's reports** *Andorra a l'Enquesta Mundial de Valors. Onada del 2018* (Pagès Editors) and
  the 2005-06 volume: not read; they may print religion by nationality.
- **The WVS online tool** for religion by parish and by nationality (§4).
- **The Department of Statistics' portal** (`estadistica.ad`, `sig.govern.ad`): not searched for
  any religion item. Institut d'Estudis Andorrans CRES surveys: not searched.
- A Govern PDF `govern.ad/documents/d/guest/uea_2018` (Universitat d'Estiu 2018, *Els valors*)
  came up in search and was not opened.

## Review, 2026-09-15 (cb8b206e-rev4, full pass)

Nothing to change. `check_md.py` clean; `built_countries.py --check` has `ad` in both editions;
`check_rollup.py ad` all modelled, nothing orphaned; `gap_share.py --check` agrees at 0.20%.
Read without starting from this file: the saved `Q289` page's third column is headed `Cases`
(not a weighted or percent column), `sources/ad.py` asserts the parse against the pin, the sum
and `Q289CS9` one to one, and every figure in `note_public` recomputes from the 1,004. `other.ad`
is in `branches.py` and `religions.json`, a standard §3.11 residual that the overview folds into
the one grey Other row, so no new legend row. Mapping agreed: a plain Orthodox answer on the
parent matches at2001, as2015, ie2022, cy2021 and uk2021; the ESS countries that went to
`.canonical` are already named in the REVIEW text. Screenshot at the entry's `view`: dots along
the valleys, densest in Andorra la Vella and Escaldes, none outside the border.

A witness of level, not opened: the US State Department's 2022 religious freedom report on
Andorra (`state.gov/reports/2022-report-on-international-religious-freedom/andorra`, 403 to a
fetch; figures from a search excerpt) says there is no census data on religious membership, and
gives government officials' 2019 estimate of 92% Catholic, Muslim leaders' figure of about 2,000
and the Jewish community's of about 100. The survey draws 835 Muslims (11 respondents) and 76
Jews; the 92% is an estimate by the state, not what people answered, so it does not contradict 63.8%.
