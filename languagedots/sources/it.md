# Italy (it): the record

Drawn 2026-10-05 by session edd42a8c-it. 58,943,457 people (ISTAT residents 1 Jan 2025), 107
provinces (NUTS 3), 142 languages, 58,886 dots. Measured rows: South Tyrol's 2024 language
groups and Trentino's 2021 minority declarations; everything else `derived`.

Build: `python sources/it_istat.py --fetch` (ISTAT 2024 tables and report, ASTAT WFS CSV, ISPAT
PDF into `data/raw/it/`), then `python sources/it_istat.py` (writes `data/normalized/it.csv` and
the placement weights `data/geo/it/it_weights.csv`). Comune lists and survey constants:
`sources/it_regional.py`. Mapping `taxonomy/it2025.py` (France's labels plus Italy's), nodes
`taxonomy/tree.d/it.txt`, entry `countries/it.py`. Population and placement are religiondots'
files, read-only: `data/raw/it/Dati_RCS_cittadinanza_2025.zip` and `data/geo/it/it_lau.gpkg`.

## 1. The rule

Italy's census asks no language. Anita's 2026-10-05 ruling for rich countries with no language
question (AGENT_BRIEF §2): Italian, plus (a) regional languages from surveys, (b) immigrant
languages by citizenship, proxy rows derived.

## 2. Sources

- **ISTAT, "L'uso della lingua italiana, dei dialetti e delle lingue straniere", anno 2024**
  (indagine I cittadini e il tempo libero, May-Sept 2024, ~16,950 households, 6+; published
  27 Jan 2026). Newer than the 2015 edition the brief named, same questions; the 2015 report is
  kept in `data/raw/it/` for reference. Used: tav. 3 (language usually spoken in the family by
  region, Bolzano and Trento apart: only/mainly Italian, only/mainly dialect, both, another
  language), tav. 11 (by mother tongue), tav. 12 (by age), tav. 14 (knowledge of the Law 482
  languages by region), tav. 1 (the 6+ base, 55,751,000).
- **ISTAT RCS 2025**: residents by comune and citizenship (196 citizenships). Comuni joined to
  LAU 2021 by code; 6 merged or split comuni since 2021 placed by predecessor or province
  (printed); Montecopiolo and Sassofeltrio stay on Pesaro-Urbino's unit, where the layer has them.
- **ASTAT, Sprachgruppenzugehörigkeit 2024** (CC0, the province's WFS): German 68.6 / Italian
  27.0 / Ladin 4.4% of declarations, by comune (116, all matched). Applied to each comune's
  Italian citizens: German 326,855, Italian 134,436, Ladin 20,499.
- **ISPAT, Rilevazione minoranze 2021** (Trentino): declared Ladin 15,775 (incl. Val di Non),
  Mòcheno 1,397, Cimbrian 1,111, by comune (tables 2, 7, 12 read from the PDF; comuni sum to the
  printed totals). "Altri comuni" spread on the province's remaining comuni.
- The brief's 2011-12 foreigners survey was not needed: tav. 11 of 2024 gives the retention
  share directly.

## 3. Method (per province)

1. **Measured minorities** first (South Tyrol, Trentino), as counts.
2. **Immigrant languages**: foreign citizens, each citizenship on its main language
   (`fr_build.COUNTRY_LANG`, Algeria and Morocco split as in France), scaled per region so the
   total equals tav. 3's "another language in the family" x population. The scale runs 0.45
   (Basilicata) to 0.95 (Veneto) of foreign citizens, against tav. 11's national 61.5% retention
   among non-Italian mother tongues: the survey share also holds naturalised citizens. In
   Sardinia, Friuli, Aosta, South Tyrol and Trentino, where "another language" also holds the
   regional language, foreign citizens x 61.5% instead. 3.99M of 5.37M foreign citizens drawn
   on their language; the rest is Italian.
   Italy differs from France on purpose: India -> Punjabi (Italy's Indians are Punjabi farm
   labour), Sri Lanka -> Sinhala (Negombo-coast migration), Switzerland -> German,
   Belgium -> Dutch, Canada -> English. China stays on the Sinitic group though most are Wenzhou
   Wu speakers: the source names a country.
3. **Local languages**: tav. 3's dialect share, "both" counting half, among those not speaking
   another language: lambda = (D + M/2)/(I + D + M); in the three regions above,
   (D + M/2 + A - immigrant share)/(1 - immigrant share). Applied to everyone left in the
   province. The survey is 6+: the under-6s (5.4%) get the 6-14 ratio (0.408), a x0.968 factor.
4. **Slovenian, Griko, Calabrian Greek**: tav. 14 knowledge x 49.1% (the report's share of those
   knowing a protected language who use it always or often in the family): Slovenian 7.6% of
   FVG -> 44,529; Greek 0.2% of Apulia -> 3,808 (Griko); 0.3% of Calabria -> 2,702 (Greko).
   Taken out of the province's local-language count, placed on their villages.
5. **Italian**: everyone else.

## 4. What "dialetto" is drawn as (`it_regional.py`)

Anita's rule is every named language its own node, and Glottolog treats the Italo-Romance
"dialects" as languages, so the dialect answer is drawn as the language where the person lives
(a place-dependent label split by geography). By region, with province and comune exceptions:
Piedmontese (Novara and Verbania Lombard), Franco-Provençal (Aosta), Ligurian, Lombard (Mantova
Emilian), Venetian (Veneto, Trento: Glottolog files Trentine under Venetian; Trieste; FVG's
Bisiacco, Gorizia, western Pordenone and lagoon comuni), Friulian (rest of FVG), Emilian,
Romagnol (Ravenna, Forlì, Rimini, Pesaro), Neapolitan = Glottolog's Continental Southern Italian
(Abruzzo but L'Aquila, Molise, Campania, Apulia but the Salento, Basilicata, Cosenza, Fermo,
Ascoli, Frosinone, southern Latina), Sicilian (Sicily, Lecce, Brindisi, Calabria but Cosenza),
Sardinian (Gallura Gallurese, Sassari area Sassarese, Alghero Catalan, Carloforte and Calasetta
Ligurian/Tabarchino). **Tuscany, Umbria, most of Lazio, L'Aquila, Ancona and Macerata: Glottolog
files these dialects under Italian itself, so they are drawn as Italian** (Umbria's 22% dialect
answers included). Minority villages take their region's dialect rate on their own language:
Arbëreshë (41 comuni, 27,117), Occitan (50 Piedmont valley comuni, 4,610), Franco-Provençal
outside Aosta (27 comuni), Molise Croatian (3 comuni, 429). That rate is an assumption with no
survey behind it; said in note_public. Sardinian is one node (the survey and Law 482 name
"sardo"); Logudorese and Campidanese are not split.

Not drawn: Walser, Cimbrian outside Trentino, Ladin in Belluno, Friulian-area German (Sauris,
Timau), Gallo-Italic of Sicily and Basilicata, Lunigiana, Brigasc.

## 5. Checks and result

Comuni sum to RCS provinces exactly; every province sums to its 2025 population (rows under 0.5
dropped: 7 people); ISPAT comuni sum to the printed totals; ASTAT shares sum to 100 per comune.
check_country ok; scatter 58,886 dots on 6,749 comuni, 50 rings (all derived), 57,457 people
under one dot per language.

National: Italian 71.6%, Neapolitan 3.93M (6.7%), Sicilian 2.61M, Venetian 2.19M, Lombard 1.33M,
Romanian 856,000, Emilian 590,000, Piedmontese 525,000, German 364,000, Albanian 336,000,
Sardinian 306,000, Romagnol 293,000, Friulian 265,000. Italian by region: Umbria 92%, Lazio and
Tuscany 90%, Veneto 52%, Calabria 54%, Trento 55%, FVG 56%, South Tyrol 29%.
Beside other figures: Wikipedia-style speaker counts (Neapolitan 5.7M, Sicilian 4.7M) are
ability; home use with "both" halved is lower, as expected.

## 6. Colours

Hand-set in `tree.d/it.txt` so neighbours part: Lombard mid blue against Piedmontese pale
aqua, Emilian light teal and Venetian sky; Neapolitan blue-violet against Sicilian light cyan
(they meet in Calabria and the Salento); Romagnol, Friulian (violet) and Sassarese (lilac) pulled
away from Emilian, Venetian and Gallurese. Neapolitan, Sicilian and Friulian were defined bare in
`pl.txt` (tiny Polish counts). Pinning these siblings moved France's generated Occitan from
#6095d4 to #85bbfc; worth a glance on France.

## 7. Room for improvement

- Survey shares are regional, so every province of a region gets the same dialect rate (Veneto's
  Belluno as Padua); provincial figures do not exist in the 2024 tables.
- Arbëreshë, Occitan and Franco-Provençal village rates borrowed from the region's dialect rate.
- Chinese drawn as a group; Nigerians as Hausa (France's mapping), though Italy's are mostly
  from Edo state; Ukrainian citizens all as Ukrainian.
- Naturalised citizens are carried by the regional scale on today's foreign-citizen mix.

## Moved from note_public (2026-10-06 sweep)

- Indian citizens are drawn 70% from Punjab, the share a University of Trento study found.
- Slovenian in Friuli-Venezia Giulia and Griko in the Salento: the survey's knowledge of a
  minority language, times the half of those who use it in the family.
