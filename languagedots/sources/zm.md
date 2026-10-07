# Zambia: the 2022 census language volume, at 156 constituencies split town and countryside

**Drawn 2026-10-04** (d9e44929-zm). 17,075,639 people drawn of 18,292,402 counted, 86 nodes, 17,036
dots, every row `derived`.

```
python sources/zm_census.py --fetch   # PDF -> data/normalized/zm.csv, all checks
python sources/zm_geo.py              # religiondots' hexes, cut town/countryside -> data/geo/zm/
python taxonomy/build.py
python tools/check_country.py zm
python scatter.py --country zm
```

Files: `sources/zm_census.py`, `sources/zm_geo.py`, `taxonomy/zm2022.py`, `taxonomy/tree.d/zm.txt`,
`countries/zm.py`; raw PDF in `data/raw/zm/`.

## 1. The source

ZamStats, *2022 Census of Population and Housing, Series C1: Language Descriptive Tables* (May
2026, 330 pp.),
`https://www.zamstats.gov.zm/wp-content/uploads/2026/05/Languages-Descriptive-Tables.pdf`. Open,
no login.

**The question** (household questionnaire P19, `wp-content/uploads/2023/12/2022-Census-Population-
Housing-Qre_FV_5June2022-1.pdf`, page 4): *"What is (NAME)'s predominant language of communication
at home?"*, one written answer, coded; "if unable to speak or hearing impaired" has its own code.
The tables call it "the widely spoken language of communication". So: home language, one per
person. Universe: the de facto household population, 18,292,402.

| table | content | grain |
|---|---|---|
| C1.0-C1.10 | 93 rows: ~80 Zambian languages in sub-groups A-L, English, 9 foreign rows, Sign Language, Other African, Other Language, babies, unable to speak; total, male, female x rural, urban | Zambia, 10 provinces |
| C2.0, C2.3, C2.4 | 9 "major language groups" | province, district, 156 constituencies, 1,858 wards; all, rural, urban |
| C2.1, C2.2 | the same by sex | not used |

The queue row said 1,853 wards; the volume prints 1,858 (the 2021 delimitation's count).

## 2. The nine groups are unions of languages, and not of C1's sub-groups

Matched on the national figures, then asserted for every province x residence (check 4, largest
gap 7 people, from starred C1 cells): Bemba = Group A less Kunda and Chikunda; Nyanja = Groups I
and J plus Kunda, Chikunda and Fungwe; Tumbuka = Group L less Fungwe; Tonga = Group K less Totela
and Subiya; Western = C1, C2 (less Mbowe and Wina), Totela, Subiya, Nkoya, Mashasha; North Western
= B, D, E, plus Mbowe, Lukolwe, Lushangi; Mambwe = F, G, plus Wina; English = English; Other
Languages = babies + unable to speak + sign + foreign + Other African + Other Language (1,287,553,
exact). Wina in Mambwe and Mbowe in North Western are odd (both are Luyana varieties), but they hold
in all twenty province x residence cells, so they are how ZamStats coded them.

## 3. The spread, and why every row is `derived`

For constituency c in province p, residence r, group g, language l:
`count(c, l) = C2_r(c, g) x C1_r(p, l) / C1_r(p, g)`. Measured: each constituency's nine groups, town
and countryside. Borrowed: the mix of languages inside a group, from the province's towns or
countryside. Every province x residence x language re-aggregates exactly (check 6). This is the
religiondots Zambia allocation (province denominational mix inside constituency Christians), and
the Bulgaria test in Claude's memory: the magnitudes are the publisher's, only the distribution is
borrowed, and nothing inside a constituency is invented.

**What it cannot show.** Where one group holds several big languages in one province, every
constituency gets the province's split. The worst case is **Eastern Province's Nyanja group**
(Chewa 906 dots, Nsenga 405, Nyanja 195, Ngoni 112 in the crop): Petauke's Nsenga and Chipata's
Chewa come out as the same mix. Others: Lala vs Bemba in Central's Bemba group; Kaonde vs Lunda vs
Luvale in North-Western's North Western group (Solwezi and Mwinilunga get one mix); Tonga vs Ila vs
Lenje in Southern's and Central's Tonga group. The note says so.

**Why not wards.** C2 goes to 1,858 wards, but no 2022 ward boundaries are open: HDX's DMMU
"Zambia - Administrative Boundaries" (CC BY, 2024) is constituencies only despite its
description (downloaded and checked, then deleted), and Stanford's ward layers are 1991-2010.
Wards would sharpen the groups' placement but not the within-group split, which is the bigger
weakness.

## 4. Suppressed cells (`*`), all accounted for

- C1: 173 cells recovered exactly from their row (total = rural + urban; sub-group totals too);
  523 shared equally from their sub-group's residual (2,175 people); 24 whose sub-group total is
  starred too shared from the table's residual (71 people). Province tables then sum to their
  totals within 1 person and to Zambia within 12.
- Central prints `Other Language Groups`' one row as **"English"** (228 people), a slip; read as
  `Other Language`.
- Western prints its **Sign Language** group with no figures at all, though Western's total
  includes them: Zambia less the other nine provinces, 284 (239 rural).
- C2 constituencies: 100 cells from the other residence's table and the total table, 29 from the
  row total, 75 shared from a row residual. 0 left.

## 5. Checks (all pass; numbers from the last run)

1. C1 province languages sum to their totals per residence (largest gap 1); provinces to Zambia
   (largest 12, starred cells).
2. Every province prints C1.0's 93 rows.
3. C2: wards nest in constituencies, constituencies in districts, districts in provinces, provinces
   in Zambia, on 1,909 / 1,848 / 2,345 unstarred sums (total / rural / urban), none differ; rural +
   urban = total on every unstarred cell.
4. C2 province rows = C1 languages summed into the nine groups (largest gap 7).
5. **Second table**: the 156 constituencies join one-to-one to religiondots' COD-AB layer (seven
   aliases: religiondots' five plus `Mpongwe Central` -> Mpongwe and `Sinjembela` -> Shangombo,
   both asserted to be their district's sole constituency). The religion volume's de facto count
   less this volume's, per constituency, is never negative (min 9, median 132) and sums to
   **exactly 47,941**, the people in institutions. Largest: Kamfinsa 3,026 and Bwacha 2,771
   (Kitwe's and Kabwe's prisons), Lusaka Central 2,086.
6. The spread re-aggregates to C1 x C2 on all 1,860 province x residence x language cells.

## 6. Placement: town and countryside inside each constituency

`sources/zm_geo.py` reads religiondots' 245,547 Kontur hexes (read-only) and labels the densest
hexes of each constituency urban until they hold the census's urban share (C2.4 / C2.0) of its
people. Achieved share against the census: median gap 0.001, largest 0.034. 103 constituencies
split, 34 all rural, 19 all urban; 7,982 town hexes. Units are `<id>-U` and `<id>-R`. It matters:
English is 97% urban, Nyanja 85%, Chewa 73% rural.

## 7. Mapping calls (taxonomy/zm2022.py has the full reasoning)

- Every printed label is a node, dialects included (Ngumbo, Unga, Mukulu; Senga, Fungwe, Yombe;
  Ndembu; the Luyana varieties).
- **Nsenga under Nyanja-Sena**, the conventional N.41 grouping; Glottolog has it under Sabi beside
  Bemba. **Lozi under Sotho-Tswana**, as Glottolog and convention agree.
- `Other African` (38,293) on a new root **`africa_other`**, kept off `other` per Anita's rule on
  indigenous remainders. American, European, Indian, Asian, Oceanian and Other Language on
  `other` (each names a region, and no narrower node holds every language it could be).
- `Swahili Tanzania` -> the existing Swahili node; `Swahili Congo` -> new `Congo Swahili`.
- Not drawn, in `gap`: babies not yet able to speak (1,108,980) and unable to speak (107,783).
- The au agent, building at the same time, adopted this fragment's `nigercongo.bantu.bemba` group
  and `nyanja_sena.nyanja` for Australia's Bemba and Nyanja answers.

## 8. Colours

Hand-picked in the fragment for the neighbours on the ground (its header lists them). Changed
after looking at the dots: Kaonde from red to blue-violet (it sat on Solwezi's Nyanja pink), Lunda
paler, Mbunda greener (against Lozi's blue in Kalabo and Lukulu), Lenje bluer (against Tonga). Weak
spot: Chewa (light pink) and Nsenga (violet) are distinct but not far apart; since they are drawn
as the same provincial mix anyway, that edge does not exist on this map yet.

## 9. Known limits

- **21 languages under 1,000 people draw nothing** (8,801 people): Wina 67, Yombe 91, Wandya 92,
  Mbumi 97, Mashasha 181, Mandarin 182, Iwa 230, Lushangi 230, Subiya 231, Twa 305, Ndembu 425,
  Lumbu 468, Lukolwe 532, Mukulu 557, Mbowe 656, Luano 674, Lima 700, Lundwe 746, Luyana 751,
  Kwandi 756, Simaa 830. scatter.py rings only `measured` rows, and every row here is derived. Their
  province figures are measured; a ring per language at its largest province would need scatter.py
  to accept a measured row on a coarser unit. Not changed (scatter.py is Anita's).
- De facto, not de jure: the census's 19,693,423 usual residents are not what the tables count.

## 10. What would improve it

- **The 2010 census district tables** as a witness or a within-province split. The 2010 provincial
  *Descriptive Tables* (Series A-D, 2013) have district tables (religiondots read Lusaka's for
  religion); if they carry predominant language by district, Eastern's Chewa/Nsenga split and
  Central's Lala/Bemba split could be checked against them. Only six of ten volumes are linked from
  the ZamStats census page (religiondots/sources/zm.md §2). Not opened for language.
- 2022 microdata (gated: the Director's written authorisation).
- 2022 ward boundaries, if ZamStats or the Electoral Commission publishes them.
