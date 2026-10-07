# Canada: the record

Drawn 2026-10-04 (agent d9e44929-ca). 36,643,409 people on 56,389 dissemination areas (DAs),
256 languages, 36,527 dots and 93 single-dot languages.

## Source

Statistics Canada, Census of Population 2021, **Census Profile 98-401-X2021006** (DAs, with the
country, provinces, census divisions and subdivisions above them), Statistics Canada Open
Licence. Six regional zips of one long CSV each, latin-1. The coverage sweep's lead (DA level,
about 200 mother tongues) was right; WebFetch got a 403 from StatCan in the sweep, but the zips
were already on disk: `ancestrydots/canada/data/raw/profile_<region>.zip`, downloaded
2026-07-13 for the ancestry map, read in place and never written to. `--fetch` downloads them
into `data/raw/ca/` if that copy is gone (URL pattern in `sources/ca_census.py`).

The question: **mother tongue**, "the first language learned at home in childhood and still
understood", asked on the short form, so it is 100% data and everyone except institutional
residents. Characteristics 393-723. The DA profile carries the full 2021 classification:
English, French, 70 Indigenous leaves and about 190 others, 264 single-response leaves in all,
plus five multiple-response rows. Group rows ("Cree languages") are only used for checks.

`python sources/ca_census.py` writes `data/normalized/ca.csv` (leaves and multiple rows at
country, province, subdivision and DA level, non-zero only) and `ca_units.csv` (each
geography's total, population, TNR and data-quality flag, and each DA's subdivision).

## Checks (numbers from the run)

1. The block is where it should be: characteristic ids 393, 394, 718, 719, 723, 724 checked by
   name. Every group row equals the sum of its members nationally, within 5 per member (worst
   0.5 of that slack).
2. Each DA's subdivision comes from row order (a CSD row, then its DAs) and every pair agrees on
   province. 13 provinces and territories, 5,161 subdivisions, 57,936 DAs.
3. Home-language total equals mother-tongue total in every DA (same population base).
4. DA totals sum to 36,612,565 against Canada's 36,620,955 (0.9998). Every province's DAs sum to
   the province row.
5. **The DAs' cells lose small counts.** Per DA, leaves + multiple rows fall short of the DA's
   own total by a median 15 (p99 55); summed over DAs the cells hold 0.974 of the total, and per
   language a median 0.754 of the national figure (Swedish 0.309: 5,890 people nationally, 1,820
   in DA cells). Subdivision cells hold 0.9987 and province cells 1.0000. Random rounding to 5 is
   unbiased, so this is StatCan dropping small cells at DA level, not rounding. Fixed in
   `countries/ca.py` (below); after it every language of 5,000+ sums to 0.999-1.010 of Canada.
6. Scatter: every counted DA has a polygon in religiondots' 2021 cartographic DA file
   (57,932 polygons, DGUID join, no misses); 1,543 polygons have no rows (empty DAs and the
   reserves below).

## How the counts are built (`countries/ca.py`)

- **Small cells put back (derived).** For each language, the province figure minus its
  subdivisions' sum (53,995 people) is shared over the province's subdivisions by their residual
  (total minus published cells); then each subdivision's figure minus its DAs' sum (992,241) over
  its DAs the same way. 622 people found no DA with a residual. These are the census's people;
  only where they sit inside the subdivision is inferred. Proportional, so it writes fractional
  rows (9.9M rows reach the scatter; it runs in about two minutes).
- **Multiple responses, 1.48M people (4.0%).** Shared 1/k across the languages named (spec §3.6;
  the row names the combination, so this is the exact split, not the scaling). The English and
  French shares keep the row's tier. The "non-official language(s)" share (630,683 people) names
  no language: it is shared over the DA's own single-response non-official languages (630,207),
  else its subdivision's (319), else its province's (158), `derived`. A person giving English and
  two non-official languages is treated as English and one; the profile does not say.
- Total drawn 36,643,409, 1.0008 of the DA totals (rounding). Derived: 1,588,355.

## Calls someone might reverse

- **Putting back the small cells.** Without it a quarter of most minority languages vanish at DA
  level and the map under-draws them unevenly. With it, 1.05M people are placed by residual inside
  their subdivision, tier `derived`, so the viewer's inferred-dots switch returns the published
  DA cells exactly.
- **The non-official share of multiple answers on the DA's own mix.** The alternative, `other`,
  would draw 630,000 grey dots concentrated in Toronto and Vancouver. StatCan publishes the
  detailed combinations nationally (not fetched here: 403 to automated requests in the sweep); a
  later session could check the national shares against them.
- **"X, n.o.s." on the group** where StatCan splits X into rows of its own (Cree 38,535, Slavey,
  Tutchone, Low German, Malagasy, Chinese, Creole), as us2024 and uk2021 put unspecified Chinese.
  Cree is the large one: 73% of Cree speakers gave no variety, so most Cree is drawn as "Cree,
  language not named". **Dene, n.o.s.** (8,060) gets a leaf "Dene", because no row names Dene
  Suline (Chipewyan), the language "Dene" mostly means; **Ojibway, n.o.s.** goes on us.txt's
  Ojibwe leaf, with the Ojibwe varieties (Algonquin, Oji-Cree, Chippewa, Odawa, Saulteaux) as
  siblings under Algic so the US's Ojibwe leaf does not turn into a group.
- **Merina moved** (2026-10-05, edd42a8c-merina) from `austronesian.malagasy.merina` to a sibling leaf `austronesian.merina`, so "Malagasy, n.o.s." now sits on the `austronesian.malagasy` leaf: as a group it made Madagascar draw washed out as "language not named" (§3); ru.txt does the same for Mari and Mordvin.
- **Persian**: "Iranian Persian" (179,425) and "Persian (Farsi), n.o.s." (25,975) both on
  Persian (Farsi), the US and UK node; Dari separate. "Oriya, n.o.s." merged into Odia.
- **Mina** under Chadic as StatCan files it, though many may speak Gen (Gbe).
- Four new roots (Iroquoian, Salishan, Wakashan, Tsimshianic), Haida and Ktunaxa as isolates,
  Michif under Algic (Glottolog): ask 005.

## Geography

Religiondots' `data/geo/ca/da/lda_000b21a_e.shp` (StatCan 2021 cartographic DAs, coastline-
clipped), read-only, `unit` = DGUID, one polygon per unit and no weighter. Religiondots draws
Canada on the same file. Its cached sea clip was reused. Large northern DAs spread their dots
over the whole polygon rather than on the settlement; there is no Kontur extract for Canada on
disk, and a hex layer for 57,936 units was not worth it at one dot per 1,000 people.

## Gap

- Institutional residents, 371,000 (1.0%), not asked the mother-tongue question in the profile.
- **63 incompletely enumerated Indian reserves and settlements** (Kahnawake, Akwesasne, Six
  Nations, Listuguj, Kanesatake, Lac-Rapide among them): StatCan publishes no counts, so these
  communities have no dots. This mostly hides Mohawk and other Iroquoian speakers.
- 1,773 DAs with no published total (8,877 people by their population row).

## Colours

New families and the Indigenous languages likely to meet on the ground are hand-picked in
`taxonomy/tree.d/ca.txt` (Cree yellow, Ojibwe and Oji-Cree greens, Dene green-teal, Inuktitut
pale blue, Innu red; in BC Salishan gold, Wakashan pink, Tsimshianic salmon). Checked pairwise in
OKLab: the remaining close pairs are far apart on the ground (Wolastoqey/Slavey, Thompson/
Gitxsan). Not touched: Punjabi and Korean are close (0.056) and meet in Surrey/Burnaby; both are
other countries' hand-picks.
