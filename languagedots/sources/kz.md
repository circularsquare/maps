# Kazakhstan: 2021 census native language, 218 rayons and cities

Drawn 2026-10-05 (session edd42a8c-kz). Queue status was `ruling` (tier D, "national only"); the
sweep's lead was wrong, the census does have native language below national level, so this is a
normal tier-A build and needed no ruling.

| | |
|---|---|
| source | BNS, National Population Census 2021, the census dashboard's Qlik engine (`qap.stat.gov.kz`, app `4c82a5bb-b3c9-49ba-bf4a-2eceb365084f`), open to anonymous callers |
| question | native language (rodnoy yazyk), one answer per person; field `Родной язык` |
| categories | 17 named languages + `Другой язык` (another language, 1.21%) |
| grain | 218 rayons and cities of oblast significance (and the districts of Astana, Almaty, Shymkent), the 2021 vintage; median 41,583 people, smallest 5,468 (Egindykol) |
| geography | languagedots' own `data/geo/kz/` (2026-10-05, session edd42a8c-kzr): COD-AB 2023 ADM2 put back to 2021 with four OSM relations; Kontur 400 m hexes, town-calibrated (see Geography) |
| script | `sources/kz_census.py` (`--fetch` ~1 min), `sources/kz_geo.py` (~3 min), `taxonomy/kz2021.py`, `taxonomy/tree.d/kz.txt` (no new nodes), `countries/kz.py` |
| result | 19,186,015 people, 18 nodes, 19,177 dots |

## How it was found

The coverage sweep opened the printed volume («Национальный состав, вероисповедание и владение
языками», Astana 2023) and was right about it: table 6 (PDF pp. 316-319) prints native language
only nationally, by nationality, as "own nationality's language" / "another nationality's
language", without naming the other language. Nothing below national in print.

religiondots (`../religiondots/sources/kz.py`) had already found that the census dashboard
(stat.gov.kz/ru/instuments/dashboards/28424/) is a Qlik Sense app whose data model is the person
records themselves (`Население 2009_2021`, 35,195,612 rows, 2009 + 2021) and that the engine
cross-tabulates any fields on request. Listing its fields (GetFieldList) turned up two
native-language fields:

- `Родной язык` on the person table, 18 values, 2021 only (every 2009 row reads `-`, so there
  is no 2009 alternative here);
- `Родной язык2` on the linked `Владение_языками` table, own / other.

`Q13` (4,860 values) looked like a write-in but is the multi-answer "languages spoken" question
(`Казахский,Русский,Английский`), not native language. Not used.

The engine also cuts at 218 rayons (`КАТО РАЙОН`), 2,454 rural districts and 4,531 villages.

## Checks (all asserted in kz_census.py)

1. `Национальность_краткая` x `Родной язык2` reproduces printed table 6's national block, all
   18 nationality rows, own and other, **to the person** (Kazakhs 13,380,107 / 117,784 ...
   Kyrgyz 26,264 / 7,920). The engine is the census and its language answer is the one printed.
2. 17 oblast totals equal religiondots' census oblast totals to the person (all distinct); the
   join to KATO is on population, then checked by name. The one allowed name difference is the
   capital, enumerated as Nur-Sultan and called Astana in the engine.
3. The 18 answers sum to 19,186,015.
4. The 218 rayons sum to their oblasts, language by language.
5. Printed only: the named field agrees with the own/other flag for 19,044,665 people, 99.26%.
   The rest are mostly people whose named language IS their group's but who are flagged
   "other" (8,258 Uzbeks, 2,028 Germans, 112 Ukrainians) and people of "other nationalities"
   (where "own" cannot be checked against a name). The named field is what is drawn; the flag
   is what the volume printed. Their national counts differ by under 1% per group.

## Calls

- **Grain: 218 rayons.** First drawn at 17 regions the same day, then moved to rayons because
  Anita works in Kazakhstan and wanted it good. No ask needed for language: religiondots' worry
  about the rayon grain was about prosecuted religious minorities, and religiondots stays at 17.
- **`Другой язык` -> `other`.** By nationality it holds Armenian, Moldovan, Ingush, Bashkir,
  Karakalpak, Georgian, Greek, Chinese and more, plus 46,865 Ukrainians and 26,349 Kazakhs whose
  language it does not name. No narrower node holds all of it. Engine field values for the
  remainder's actual languages were not found (no finer native-language field exists).
- **Turkish** for `Турецкий` (Meskhetian Turks), **Kurdish** unsplit, **Dungan** as Kyrgyzstan's
  node. No new nodes.
- **Placement**: Kontur population inside each rayon, with each census town given its own count
  (see Geography). Brief §4.4 would allow placing each language by settlement (the engine has
  `Село КАТО` x language for 4,531 villages), which needs village coordinates; not done.
- **Colours**: Kyrgyzstan's hand-picked Turkic set is reused. Kazakh is a pale rose (0.84 0.09 15)
  chosen for a minority in Kyrgyzstan; here it is 71% of the map, beside Russian's green. Worth a
  look once the tiles are up.

## Geography (sources/kz_geo.py)

**No KATO-coded rayon layer was found in the open.** religiondots joined nothing below the 17
oblasts (its `карта_район` lead has 190 rows for 218 units). OSM's 227 admin_level=6 relations in
Kazakhstan carry no `ref:kato` (Overpass, 2026-10-05, which answered that day once given a tool
User-Agent). The engine's `КАТО РАЙОН` is its own number (1-220), not KATO; `РАЙОН КАТО` is a
half-filled name string; `OWNER3_NAME` is the rayon's name and is what is used (cached as
`kz_qlik_rayon_names.csv`). So the join is by name.

**Boundary file: COD-AB 2023 ADM2, 218 polygons, matching the census count only because four
changes cancel.** After the census (2022), and in COD: Aksuat district carved from Tarbagatay,
Samar from Kokpekti; both dissolved back (the census puts Aksuat village in Tarbagatay and
Samarskoye in Kokpekti, asserted from `kz_qlik_rayon_villages.csv`). Before the census (2021), and
not in COD: Kosshy city administration out of Tselinograd, and Sauran district out of Turkestan
city administration's rural okrugs. COD's "Turkestan" is the city alone (282 km2) and its
"Kentau" (7,699 km2) also holds Sauran's ground. Rebuilt from OSM relations (polygons.openstreetmap.fr,
cached as `data/raw/kz/osm_rel_*.geojson`): Kosshy (15594335, 141 km2) cut from Tselinograd;
Turkestan city (5496366, 196 km2) and Kentau (17322798, 674 km2) cut from COD's Turkestan +
Kentau, and Sauran (7,110 km2) is the rest. OSM's own Sauran polygon overlaps its Kentau by 519
km2 and contains the city, so it was only a check. Sauran comes out 57.4% Uzbek, which fits the
old Turkestan rural okrugs (Ikan, Karashyk). The 2022 oblast reform moved whole rayons, so each
COD ADM1 maps to its 2021 oblast. Astana's Nura and Shymkent's Turan (both 2022) are not in COD,
which is right for 2021.

**The join, inside each oblast:** the Russian name transliterated, `район`/`Г.А.` and adjectival
endings stripped, city administrations marked so Kostanay city and Kostanay district (or Pavlodar
city and district) cannot swap; scored by string similarity, assigned one-to-one best first. 190
pairs agree exactly after normalising; the other 28 are pinned in `PINNED` (renames: Baiterek =
Zelenov, Akkuly = Lebyazhye, Kapchagay = Qonayev, Beimbet Mailin = Taran; spellings: Uil = Oiyl,
Burlin = Borili, Zhangala = Zhanakala, Temirtau = COD's "Termitau", and so on). A pin does not
steer the scorer: an inexact pair passes only if the scorer reached it unaided and it is pinned.
**218 both ways**, 19,186,015 people.

**Witness (Kontur, which no name decides):** log correlation of census against Kontur per rayon
r = 0.902; best of 500 national shuffles 0.252; **best of 1,000 shuffles made only inside oblasts
0.546** (the null that fits a within-oblast name join); 26 rayons outside a factor of 2 against a
shuffled minimum of 83. Kontur nationally 1.022 of the census. The outliers look like Kontur's,
not the join's: suburbs counted outside their city (Tselinograd 5.3 round Astana, Bukhar-Zhyrau
3.6 round Karaganda, Kostanay district 3.1, Ulan 3.6 by Ust-Kamenogorsk) and towns Kontur nearly
loses (Stepnogorsk 0.20, Kurchatov 0.17, Taraz 0.21, Almaty's Auezov 0.26).

**Town calibration (placement only, brief §4.4).** The census counts all 86 towns (`Г.` rows of
`Село КАТО`). In a disc round each town (radius for 2,000/km2, at least 3 km), Kontur's share of
the rayon against the town's census share was median 0.56, min 0.22 (Lisakovsk; Aksu 0.25,
Shakhtinsk 0.26, Stepnogorsk 0.34): uncorrected, a mining town's dots would sit in its villages.
In 70 rayons holding a town and other people, the disc's hexes are scaled to the town's census
share and the rest to the remainder (unit totals asserted unchanged); 13 one-city rayons are left
alone. Towns located from GeoNames' cities file (`maps/data/geonamescities.csv`, Russian
alternate names, the point required to fall in its own rayon); Kaskelen, Usharal and Serebryansk
by hand (`TOWN_XY`, same containment check); Zhem (1,355) not placed. Densest hex after: 12,792
per km2 (Kontur's own max 11,834), far under the 46,200 cap. Raw Kontur stays in the layer as
`kontur`.

## Regional picture (shares of each region, %)

| region | Kazakh | Russian | Uzbek | Uyghur |
|---|---|---|---|---|
| North Kazakhstan | 37.3 | 55.3 | 0.1 | 0 |
| Kostanay | 44.0 | 46.9 | 0.2 | 0 |
| Pavlodar | 56.6 | 36.8 | 0.2 | 0 |
| Almaty city | 66.2 | 22.0 | 0.5 | 4.8 |
| Almaty region | 74.8 | 11.6 | 0.3 | 7.3 |
| Shymkent | 71.6 | 7.1 | 16.3 | 0.2 |
| Turkistan | 76.1 | 1.6 | 17.6 | 0.1 |
| Kyzylorda | 96.3 | 2.2 | 0.2 | 0 |

At rayon grain: Russian is highest in Ridder (77.1%), Shemonaikha (76.3%) and Altai (71.1%);
Kazakh is above 99.8% in Kyzylkoga, Isatay and Aral; Uzbek 66.3% in Sayram and 57.4% in Sauran;
Uyghur 56.8% in Uyghur district and 30.3% in Panfilov; Dungan 30.8% in Korday.

## Second sources

None needed for the counts: check 1 is the printed table itself. The coverage sweep's fallback
(ethnicity by settlement x national native-language-by-nationality) is unnecessary.

## Reproduce

    python sources/kz_census.py --fetch
    python sources/kz_geo.py --fetch
    python taxonomy/build.py
    python tools/check_country.py kz
    python scatter.py --country kz

## Record (moved from countries/kz.py, 2026-10-06)

Text of `how` and `note_public` before the 2026-10-06 data-text sweep shortened them (kz.py had
no `gap`). The facts cut from the public note were the retention shares among Soviet-era settled
and deported peoples (63% of Ukrainians, 59% of Germans, 45% of Koreans and 72% of Poles named
Russian as native language), the 58,791 ethnic Kazakhs who named Russian (still in the note),
that the dashboard is backed by the individual census records, the examples in "other", and
that dots follow population, not speakers, inside a rayon.

`how`:

    census, 2021, native language; placed inside each rayon by Kontur population, with each
    town given its census count

`note_public`:

    The 2021 census asked each person's native language, which in the countries of the former
    Soviet Union leans towards identity rather than everyday use. 13.7 million people named
    Kazakh and 3.5 million Russian, and only 58,791 ethnic Kazakhs named Russian, though Russian
    is the everyday language of many Kazakh families in the cities and the north. Among the
    peoples settled or deported here in the Soviet years the answer often moved to Russian: 63%
    of Ukrainians, 59% of Germans, 45% of Koreans and 72% of Poles named it. The Bureau of
    National Statistics prints native language only for the country as a whole. The figures for
    the 218 rayons and cities come from the census dashboard, which is backed by the individual
    census records and reproduces the printed national table exactly. It names 17 languages; the
    other 1.2% (Armenian, Moldovan, Ingush, Bashkir and more) are drawn as other. Rayons are
    drawn as they were at the 2021 census, before Abai, Jetisu and Ulytau regions were formed.
    Within each rayon the dots follow where people live, not where the speakers of each
    language live.
