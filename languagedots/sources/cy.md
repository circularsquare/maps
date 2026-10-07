# Cyprus: Census of Population and Housing 2021, native language

Drawn 2026-10-05 (session edd42a8c-cy). 912,270 people, 5 districts, 31 nodes, 896 dots.
North added the same day (edd42a8c-cy2): 1,198,048 people, 197 units, 1,182 dots in all.

## Source

CYSTAT-DB (Statistical Service of Cyprus), PxWeb JSON API at
`https://cystatdb.cystat.gov.cy/api/v1/en/8.CYSTAT-DB/` (not `/pxweb/api/v1/`; religiondots'
`sources/cy.py` found that). Folder *Population / Census of Population and Housing 2021 /
Population / Population - Language, Religion, Ethnic Religious Group*:

| table | content | used for |
|---|---|---|
| 1891616E | language x district x sex, 33 named + Other + Not stated | the counts drawn |
| 1891610E | language x district x urban/rural x sex, 20 named + Other + Not stated | check |
| 1891613E | language x citizenship group x district x sex, same 22 | check, and placement |
| 1891213E | citizenship group x municipality/community x sex (folder *Country of Citizenship*) | placement |

Reference day 1 October 2021; universe the whole enumerated population, 923,381. Nothing is
published below district for language. The 2011 branch of CYSTAT-DB was not searched (2021 is
newer and the same grain at best).

**Which question.** The questionnaire (census2021.cystat.gov.cy, `Census 2021 QST-EN_.pdf`) asks
Q11(a) "Which language does ... speak fluently?" and Q11(b) "What is ...'s native language?",
both write-in. The tables' footnote reads "the language that the respondent stated that s/he
speaks best". A single-answer table with 923,381 rows summing exactly can only be 11(b), and the
footnote looks carried over from older wording; `how` says native language. If CYSTAT ever
documents otherwise, change `how` only.

## Checks (all asserted in `sources/cy_census.py`)

- 1891616E: languages sum to each district total; districts sum to 923,381.
- 1891610E and 1891613E agree with 1891616E on all 20 shared named languages, Not stated and
  Total, in every district, to the person.
- 1891613E's `Other languages` = 1891616E's 12 extra named languages (Hebrew to Dutch) + its own
  `Other languages`, every district. So the 12 are a split of that row only.
- 1891213E's 396 communities summed by district equal 1891613E's citizenship group totals per
  district, all four groups, to the person.
- `cy_place.csv` sums back to 1891616E for every (district, label).

## Geography

religiondots' `data/geo/cy/cy_hexes.gpkg` (Kontur hexes keyed by LAU code, 396 communities,
joined and checked there). District = the LAU code's first digit (1 Lefkosia, 3 Ammochostos,
4 Larnaka, 5 Lemesos, 6 Pafos; 2 Keryneia is not enumerated). The language tables spell
districts by name (one as `Lekfosia`); mapped explicitly.

**Placement inside a district** (AGENT_BRIEF §4.4, citizenship rung): for each label L and
community c, `E(L, c) = sum over groups g of P(L | g, district) x N(g, c)`, with P from 1891613E
(Cypriots, other EU, non-EU, not stated) and N from 1891213E. The 12 languages missing from
1891613E use the `Other languages` profile scaled to their own count. Inside a community, by
Kontur hex population. District counts never change. Effect: across communities of 1,000+,
Russian runs 0.6% to 10.5%, Arabic 0.6% to 8.9%, Greek 43% to 94%. Romanian leans to Lemesos
town (51% of the district's Romanian speakers against 41% of its people). Turkish follows the
Cypriot citizen profile, which is right for Turkish Cypriots living in the south and spreads them
evenly, since nothing finer says where they live.

## Mapping calls (`taxonomy/cy2021.py`)

- `Indian` (8,370) and `Sri Lankan` (6,060) name countries; each could be Indo-Aryan or
  Dravidian, so `other` (as us2024 and zm2022 do). Most Sri Lankans in Cyprus are probably
  Sinhala speakers, but nothing published says so.
- `Yugoslavian` on Serbo-Croatian, `Moldovian` on Moldovan, `Chinese` on Sinitic (unwashed),
  `Nepalese` on Nepali, `Persian` on Persian.
- Greek is not split into Cypriot Greek; Cypriot Maronite Arabic and Kurbetcha are not printed.
- No new nodes, no tree fragment. Colours left as generated: Greek (purple), English, Russian,
  Arabic, Romanian and Bulgarian are well apart; Georgian is a lighter purple near Greek but small.

## The north (added 2026-10-05, session edd42a8c-cy2; ask 015, Anita: yes)

Drawn as Turkish from the 2011 census held in the north, the only one there: 285,778 of its
286,257 usual residents (*sürekli ikamet eden nüfus*) on 192 village and town units, every row
`tier="derived"`, node `turkic.turkish`. 286 dots. Builder: `sources/cy_north.py`.

**Source.** istatistik.gov.ct.tr, page *Temel İstatistikler / Nüfus Sayımları / Nüfus Sayımı
2011*, files under `/Portals/39/`: Tablo-1 (ilçe), Tablo-3 (ilçe, bucak, belediye, mahalle; 250
rows, the finest published), Tablo-5 (ilçe x citizenship). No language, religion or ethnicity
question in any of the eleven tables. Checks, asserted: mahalles sum to 286,257; per ilçe to
Tablo-1; 32 belediye subtotals equal their mahalles; Tablo-5 rows sum to the total.

**Citizenship (Tablo 5), recorded, not used.** Northern Cypriot citizens 190,494 (only that
136,362; also Turkish 38,085; also another 16,047); Turkish only 80,550; UK 3,693, Turkmenistan
1,760, Nigeria 1,280, Iran 1,152, Pakistan 1,075, Bulgaria 920, Azerbaijan 835, other 4,498. So
15,213 (5.3%) hold neither citizenship, and 3,693 Britons (mostly Girne district, 3,057) are
the largest group drawn as Turkish who are almost certainly not Turkish speakers. Per the
ruling they stay on Turkish; `note_public` gives the figures. Maronites of Koruçam (Kormakitis)
and the Greek Cypriots of Dipkarpaz (Rizokarpaso) are likewise drawn as Turkish.

**Geography.** No codes, Turkish names. OSM (Overpass, `data/raw/cy/north2011/osm_boundaries.json`)
maps the north's villages at admin_level 8 (229 inside the line) and town quarters at 9 (73).
Each census row goes to an admin-8 village: by name within its 2011 ilçe (177 rows), via an
admin-9 quarter of that name and the village holding it (53), via a fixed list (8: Malatya -
İncesu spans two villages; Zeytinlik Kesim/Köy; Edremit = OSM Erdemit; Kılıçarslan = Kılıçaslan;
Karaman (Yukarı Karmi); Yukarı Girne, whose three OSM quarters are all inside Girne; Kapalı Maraş
to Gazimağusa), a point (Kantara, OSM node 6029501335), or its belediye's village (10 town
quarters with no polygon: four Lefkoşa walled-city quarters, Canbolat, three Tatlısu quarters,
Aşağı Karaman, Cevizli). 2011 ilçe Güzelyurt = OSM Güzelyurt + Lefke (split 2016). 36 OSM
villages have no 2011 row; the 32 mostly inside the line join the neighbouring used unit of the
same ilçe with the longest shared border (placement only), 4 (Erenköy/Kokkina, Mansur, Mosfili,
Akaça) are left out. The join's witness: log correlation of 2011 population with Kontur per
unit 0.722 against a best of 0.234 over 500 shuffles. Kontur/census 1.08 overall; town units
read low (Girne 0.34, Ortaköy 0.29, Lefkoşa 0.45), which is Kontur under-seeing towns plus 12
years' growth in the villages, not a misjoin (every quarter's polygon nests in the village it
was assigned to). Karakum and Doğanköy have no populated Kontur hex and use their polygon.

**The line.** North = OSM relation 2514541 (Northern Cyprus) minus 3263909 (UN Buffer Zone); the
two do not overlap. GISCO's 396 enumerated south communities run 170 km2 over that line in 22
places, because they are de jure community lands: Lefkosia (1000) holds the north half of the
walled city, Agios Dometios holds Metehan, Acheritou, Achna and Pergamos are north villages.
religiondots' hexes (read-only) therefore put 147 south hexes, 25,341 Kontur people (19,096 of
them in north Nicosia), north of the line. `data/geo/cy/cy_hexes.gpkg` is now languagedots'
own copy of those hexes with the 147 dropped and 75 hexes on the line cut back to the south side,
plus 2,705 north hexes clipped to the north side. Every south community keeps hexes. After
scatter: 286 dots in the north, all Turkish; no north dot in the buffer zone; 19 south dots fall
in OSM's buffer zone (Pyla, Athienou, the Nicosia strip), which are south communities the 2021
census enumerated there, unchanged.

## Gaps

- The north is ten years older than the south (2011 against 2021) and has no language question.
- Pyla (Pile): 479 people the northern census counted in the buffer zone; not drawn, since they
  would sit on the buffer zone or on the south's Pyla community.
- Not stated: 11,111 (1.2%), not drawn.
- 16,270 people (1.8%) are in languages under one dot's worth nationally and draw no dot
  (scatter.py writes 12 rings for them).

## Files

`sources/cy_census.py`, `data/raw/cy/*.json`, `data/normalized/cy.csv`,
`data/normalized/cy_place.csv`, `taxonomy/cy2021.py`, `countries/cy.py`, `ask/015-cy.md`;
the north: `sources/cy_north.py`, `data/raw/cy/north2011/`, `data/normalized/cy_north.csv`,
`data/geo/cy/cy_north_units.gpkg`, `data/geo/cy/cy_hexes.gpkg`.
