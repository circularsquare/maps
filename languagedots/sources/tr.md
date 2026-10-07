# Türkiye (tr): record

Drawn 2026-10-05 (session edd42a8c-tr). 88,298,651 people on 81 provinces, 14 nodes, every row
`modelled` except the Syrians (`derived`). 88,292 dots.

Files: `sources/tr_konda.py` (counts), `sources/tr_geo.py` (placement layer),
`taxonomy/tr2006.py`, `taxonomy/tree.d/tr.txt`, `countries/tr.py`, `data/normalized/tr.csv`,
`data/geo/tr/tr_units.gpkg`, `data/geo/tr/tr_hexes.gpkg`, `data/raw/tr/` (two KONDA PDFs, the
Wikipedia wikitext of the 1965 table, the DGMM image and its transcription, COD-AB admin 1),
`data/geo/kontur/kontur_population_TR_20231101.gpkg(.gz)` (copied from religiondots' raw/tr).

## 1. What exists, and why these sources

No census has asked mother tongue since 1965; the register census (ADNKS, since 2007) has no
language item. Searched and ruled out:

- **TNSA / Turkey DHS 2003, 2008, 2013, 2018** (Hacettepe). The women's questionnaire asks
  mother tongue (Q114: Turkish, Kurdish, Arabic, other), but **none of the four main reports
  prints a mother-tongue table**, by region or otherwise: full-text searched (2018 FR372 and
  2003 FR160 from dhsprogram.com; 2013 and 2008 from the Wayback copies of hips.hacettepe.edu.tr).
  The word appears only in the questionnaire appendix. The microdata is an application to HÜNEE,
  so gated. The brief's "likely floor" does not exist as a published table.
- **TAYA** (family structure survey): no mother-tongue table found; religiondots' record notes
  its reports were full-text checked for sect, and nothing in them suggested a language cross-tab.
- **KONDA** (archived library via the Wayback CDX API). *Biz Kimiz?* 2006 (47,958 adults,
  79 provinces, sample stratified on the 12 İBBS-1 regions, Tablo 2) has the only published
  mother-tongue partition with every answer named (Tablo 7, p.19, national only). Its 2008
  report *Kürtler ve Kürt Sorunu* p.5 prints, from the same survey, the **share of Kurds and
  Zazas in each of the 12 regions**, children included (Toplam 15.7, the whole-population
  figure on p.4). The 2010 and 2011 reports reprint the same regional table (the 2010 survey,
  n=10,393, gives mother tongue nationally only: Turkish 84.0, Kurdish 12.7, Zazaki 1.4, Arabic
  1.2, other 0.7). No KONDA report prints mother tongue by region.
- **1965 census**, mother tongue by province (67 then). Census counts, 13 languages in the
  provincial table on Wikipedia ("1965 Turkish census", citing Ahmet Buran 2012, from Dündar's
  compilation of the published volumes). 60 years old and before the Kurdish migration west
  (KONDA 2006: 17.5% of Turkey's Kurds live in İstanbul; 88% of İstanbul's Kurds were born
  elsewhere). As a count it would draw 31 million people. Used here only to **place**.

Choice: the 2006 survey for the numbers (largest sample, all answers named, 12 regions for the
Kurdish geography), the 1965 census for where people sit inside those numbers (§4.4), the 2025
register for the population, DGMM for the Syrians.

## 2. The model (sources/tr_konda.py)

Population: the **province populations DGMM prints beside its Syrian counts** (ADNKS, sum
86,092,168, the register's 31 Dec 2025 total), so base and Syrians come from one table.

1. **Kurdish + Zazaki** per region = population × KONDA's regional Kurd-and-Zaza share × 0.9687.
   0.9687 = (11.97 + 1.01) / 13.40, adults' Kurdish-or-Zazaki mother tongue over adults' Kurd or
   Zaza identity, both 2006. KONDA's identity count already includes anyone with that mother
   tongue, so mother tongue is a subset and the ratio is a retention rate; it is applied
   uniformly (it is surely lower in the west, and nothing says by how much).
2. **Zazaki's part**: the 1965 ratio Zazaki/(Kurdish+Zazaki) in Kuzeydoğu (0.009), Ortadoğu
   (0.099) and Güneydoğu Anadolu (0.074), the 1965 national ratio (0.064) in the nine western
   regions, all × 1.065 so the national figure is KONDA's 1.01/12.98.
3. **Inside a region**: in the three eastern regions an IPF over provinces × (Kurdish, Zazaki,
   rest), seeded with 1965 rates × today's population; in the nine others plain population
   (the speakers there are migrants, and 1965's handful of Kurdish villages in Konya or Sakarya
   would mislead).
4. **The twelve smaller answers**: KONDA's national share × 86.1M, spread by the 1965 rate of
   the matching language × today's population (Arabic, Armenian, Greek, Jewish, Laz,
   Circassian; Balkan by Pomak + Bosnian + Albanian; Kafkas by Georgian); Türki Diller, Kıptice,
   Batı Avrupa, Diğer by population.
5. **Turkish** is each province's remainder (asserted non-negative).
6. **Syrians under temporary protection**, 2,206,483 by province, added on top as Arabic,
   `derived`. ADNKS leaves them out (DGMM's own table adds them to get "people living in the
   province").

1965 parents for provinces created since: Osmaniye→Adana, Kırıkkale→Ankara, Düzce→Bolu,
Karabük and Bartın→Zonguldak, Bayburt→Gümüşhane, Ardahan and Iğdır→Kars, Aksaray→Niğde,
Karaman→Konya, Kilis→Gaziantep, Yalova→İstanbul, Batman and Şırnak→Siirt+Mardin pooled.

## 3. Results and checks

| | count | share of register |
|---|---:|---:|
| Turkish | 69,788,155 | 81.1% |
| Kurdish | 13,058,473 | 15.2% |
| Arabic (citizens) | 1,188,074 | 1.38% |
| Arabic (Syrians, TP) | 2,206,483 | 2.56% |
| Zazaki | 1,101,851 | 1.28% |
| eleven smaller answers | 958,615 | 1.1% |

Kurdish + Zazaki come to 16.4% of all ages against KONDA's 13.0% of adults: the regional shares
include children (KONDA's own whole-population figure was 15.7% in 2006) and are weighted here
by 2025 regional populations, in which the east and İstanbul have grown fastest.

Checks in the script: Tablo 7 sums to 100.00, Türkçe 84.54 and Kürtçe 11.97 pinned; the regional
table's 12 rows, TR1 14.8, TRB 79.1, Toplam 15.7 pinned; 1965 table 67 rows, 13 named columns,
every 1965 province used as a parent and every parent found; DGMM: syrians + il nüfusu = toplam
on all 81 rows, sums 2,206,483 (matches the headline figure in DGMM's yearly chart) and
86,092,168; the output sums to 88,298,651. The 1965 provincial table sums to 2,212,216 Kurdish
and 152,008 Zazaki against the national table's 2,233,071 and 150,644 (different compilations,
within 1%).

Geography (tr_geo.py): COD-AB admin 1 (2022, 81 provinces) joined to DGMM by ASCII name, 81 = 81
both ways. Kontur 400 m hexes, 454,936 kept; 1,032,319 Kontur people (1.2%) fall outside every
province (coast and border hexes). Kontur/register 0.986 nationally, per province p10 0.79,
median 0.93, p90 1.08, none outside a factor of 3; log r 0.994 against a best of 0.399 over 500
shuffles. One Kontur cap block is unreviewed (7 km from İzmir, 4.6% of the province, Buca and
Karabağlar): religiondots registered it `unreviewed`, and registering it `real` here clashes with
that row and stops the scatter, so it is left as religiondots has it (drawn as Kontur has it).

## 4. Calls someone might reverse

- **KONDA, a private pollster, as the source.** It is the only published mother-tongue
  partition since 1965, its sample is the largest of any Turkish survey asking the question, and
  its national figures replicate across its 2006, 2008 and 2010 waves within a point.
- **Identity shares carry the Kurdish geography.** The regional table is "Kürt ve Zaza", not
  mother tongue; scaled by one national retention ratio. A regional mother-tongue table (TNSA
  microdata could give one) would replace step 1.
- **Ortadoğu Anadolu's 79.1%** forces Malatya and Elazığ to about half Kurdish or Zazaki, more
  than most accounts give them, because the IPF has to fit it while Van, Hakkari, Bitlis and
  Muş are already near 90%. That is the survey's regional figure, kept as is.
- **Tunceli** comes out 57% Kurdish, 5% Zazaki, 37% Turkish: the 1965 census put 77% of Tunceli
  down as Turkish and 1.5% as Zazaki, so it seeds the split badly. Dersim is mostly Zazaki in
  every account; nothing countable says by how much. 85,000 people.
- **Batman and Şırnak** take pooled Siirt+Mardin rates, so 17% Arabic, probably too high for
  Şırnak.
- **Kars, Ardahan and Iğdır** share 1965 Kars's rates (29% Kurdish each); Iğdır's Azerbaijani
  speakers sit inside "Türki Diller", which is placed by population nationally.
- **Syrians all drawn as Arabic**; DGMM gives no language or ethnicity.
- **Balkan and Batı Avrupa on `indoeuropean`**, Türki Diller on `turkic`: answers that name a
  group of languages, drawn as "language not named".

## 5. Room for improvement

- TNSA microdata (2008/2013/2018, a HÜNEE application, so gated) has each woman's mother tongue
  and NUTS-1 region: a measured regional Kurdish/Arabic split for the women's sample.
- Dündar's full 1965 table (all ~50 languages by province, and district-level tables) would
  place Azerbaijani, Abkhaz, Chechen, Bulgarian and Hemshin properly.
- A newer KONDA identity table by region (its 2010+ barometers) would update the 2006 shares.

## Moved from countries/tr.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- Within each region, and for the smaller languages across the country, people are placed where the 1965 census found speakers of the same language, which is why Arabic sits in Hatay, Mardin and Şanlıurfa and Laz in Rize and Artvin.
