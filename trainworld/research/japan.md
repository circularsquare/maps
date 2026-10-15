# All of Japan: coverage, sources and a staged plan

T-420, 2026-10-15. Design only: nothing built or downloaded beyond one e-Stat mesh file opened to
check access. Anita: "maybe try doing a full dense area like maybe all of japan." Sources marked
"checked" were opened this pass; figures marked "memory" are to be confirmed at build time.

## 1. Coverage

**What a pack costs is its cells, not its people.** Cells are H3 resolution 9 (0.1 km²) and exist
only where people or jobs are, so a pack's bytes, its opening time and most of a worker's fixed
memory follow its settled area. New York: 208k cells, 22.0M people, an 18.3 MB pack (about 88 bytes
a cell all in), 256 MB peak and 222 MB steady a worker after T-398. Shanghai: 78k cells for 24.9M;
Mumbai 47k for 28.3M. Tokyo is dense in the same way: the 2020 census 500 m mesh file for first
mesh 5339 (Tokyo, Kawasaki, south Saitama, west Chiba) has 16.7k populated half-meshes, about
4,200 km² (checked). Tokyo, Kanagawa, Saitama and Chiba (36.9M, memory) have about 13,600 km² of
land, roughly 9,000 km² of it habitable (memory).

**The Greater Tokyo pack** (the four prefectures, Tokyo's Izu and Ogasawara islands left out):
about 90-120k cells, so a pack of about 8-11 MB raw (half New York's), about 2,500-3,000 zones of
2 km (New York 6,407). A worker's cell-driven memory should come out below New York's (estimate:
150-200 MB steady with a New York-sized network). What Tokyo would stress is the network a player
builds there, not the pack: strategy tables, the commute pair loop and `set_network` grow with
stations and route nodes, and a Tokyo-like network could have several times the real New York
save's stations. Nothing hard-coded breaks: node ids fall back from u16 to u32 past 65,535
(`kernel.rs`), gravity grid coordinates are u16 at 2 km, one region per pack. The 2 km gravity
holds: zone totals are large in central Tokyo (a Marunouchi zone may hold several hundred thousand
jobs) but nothing in the solve depends on size; the curve is fitted on Japan's own flows (section 2).

**Japan as a whole**: roughly 1.0-1.3M cells (Japan's habitable land is about 122,000 km² of
378,000, memory; cells cover roughly that: estimate), so 90-115 MB of packs in all, opened one zone at a time.

| Option | Packs | Largest pack | Verdict |
|---|---|---|---|
| A. Metro packs of whole prefectures + one zone per other prefecture (Hokkaido by subprefecture) | about 45 | Tokyo, ~100k cells, ~10 MB | **Recommended** |
| B. Metro packs + secondary medical areas (KSJ A38, about 335 units of whole municipalities, memory) as zones | about 240 | same | finer zones, 5x the packs, seams and files; obscure names ("Seiban", "Kita-Mino") |
| C. One pack per prefecture | 47 | Hokkaido (83,000 km²); Tokyo-to 14M | cuts Tokyo in four: 2.9M daily commuters into Tokyo-to (memory) would ride the regional layer at 6 km grain |
| D. A grid (say 50 km squares) | ~150 | - | follows no data unit; outlines and names odd |

Option A's metros: **Greater Tokyo** (Tokyo, Kanagawa, Saitama, Chiba, 36.9M), **Keihanshin**
(Osaka, Kyoto, Hyogo, Nara, 18.2M), **Chukyo** (Aichi, Gifu, Mie, 11.3M), **Fukuoka** (the
prefecture, Kitakyushu included, 5.1M), **Sapporo** (Ishikari subprefecture, 2.4M), **Sendai**
(Miyagi, 2.3M), **Hiroshima** (the prefecture, 2.8M): about 79M people in 7 packs. The other 32
prefectures are a zone each (0.55-3.6M; Shizuoka and Ibaraki biggest, perhaps 40-60k cells, which
open bare in 0.3-1.5 s), and Hokkaido's 13 other subprefectures merge into about 6 zones by the
under-100k rule. Prefectures are Japan's county-level unit for zones.md (municipalities are too
small, average 72k). Commutes over a pack's edge (Ibaraki to Tokyo, Shiga to Kyoto)
cross packs through the joint gravity, as Mercer and Windham do.

**Clusters and the long-trip market.** With the market's near pairs under 250 km (T-362 decision
6), Tokyo's whole pack (about 150 km across) is within one near-pair radius of itself, and Tokyo -
Nagoya (about 260 km) is a far pair. A served Tokaido corridor has perhaps 1,500-2,500 served 6 km
zones with hundreds of near neighbours each: a few times the Northeast market's pairs, within what
T-398's served-zones-only rows handle. The real issue is clusters: clusters merge through track, and
a Shinkansen line from Tokyo to Fukuoka would join every pack along it into one solve (each edit
re-solving Tokyo, Chukyo and Keihanshin together, several times New York's 11-22 s). Japan is where
T-362's loose coupling (other packs' loads on shared trains as a background) stops being optional.

## 2. Sources

All open; e-Stat and KSJ files download with no login (an e-Stat mesh zip fetched with a bare GET:
`.../gis/statmap-search/data?statsId=T001141&code=5339&downloadType=2`, checked). The e-Stat API
needs an app id; the file downloads below avoid it. No request carries Anita's name or email.

| Input | Source | Licence | Replaces |
|---|---|---|---|
| Population | 2020 census mesh statistics, 250 m and 500 m (500 m with 5-year ages), JGD2011, per first mesh or per prefecture (since Oct 2025), CSV in zip, e-Stat statistical GIS (https://www.e-stat.go.jp/gis/statmap-search?type=1, checked) | e-Stat terms: Government Standard Terms 2.0, usable under CC BY 4.0 (https://www.e-stat.go.jp/terms-of-use, checked) | GHSL scaled to districts (`population` hook). Mesh squares come from the codes, no polygons needed. Tiny meshes are merged into a neighbour (columns HTKSAKI, GASSAN): follow them |
| Jobs at work | 2021 Economic Census for Business Activity, all-industry persons employed by 1 km and 500 m mesh, government establishments included this round, 62.4M persons (https://www.stat.go.jp/data/mesh/map.html, summary r3gaiyo.pdf, checked); download through the same e-Stat GIS (statsId not looked up) | same | tier 0's proxy (`jobs` hook). Individual farms and household services are missing: scale each municipality to the census's workers by place of work (next row) |
| Commute flows, workers at home | 2020 census, commuting tables: table 3 and 4, workers and students by municipality of residence and of work or school, national and per prefecture, Excel and DB (statInfId 000032214305 national, https://www.e-stat.go.jp/stat-search/files?toukei=00200521&tstat=000001136464&tclass1=000001136469, checked); "working at home" is its own row | same | the borrowed curve: fitted as Toronto's and Paris's (`t277_fit.py` style), one fit for Greater Tokyo, one national for the rest; at-home workers make no commute |
| Work from home (telework) | MLIT telework population survey 2024: employed teleworkers 24.6% nationally, Greater Tokyo 37.5%, Kinki 24.4%, Chukyo 19.6%, elsewhere 16.8% (https://www.mlit.go.jp/report/press/toshi03_hh_000165.html, via press, checked) | government publication | the default 0: a share at home on a given day, about these times 2 days in 5 (judgement), per pack |
| Schools | KSJ P29 schools, 2023: every school as a point with its type (elementary, junior high, high, special, university, college), per prefecture or national (https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P29-2023.html, checked); no pupil counts | CC BY 4.0 (2021 and 2023 editions; 2013 is non-commercial) | OSM school sites (`schools` hook). Pupils from the census mesh's ages 5-14 near each school, or MEXT's School Basic Survey totals by prefecture and type (e-Stat, not checked) |
| Universities | P29 campuses; students per institution are not in the School Basic Survey (memory) | as above | the sites only; Wikidata counts and the national total stay tier 0 |
| Boundaries | KSJ N03 administrative areas, 2025-01-01 edition, 603 MB national GML zip, also per prefecture (https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2025.html, checked) | CC BY 4.0 | the drawn outline: municipal polygons dissolved per prefecture or subprefecture |
| Medical areas (option B only) | KSJ A38, 2020: primary, secondary and tertiary medical areas with their municipalities (https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-A38-2020.html, checked), 1.1 GB national | CC BY 4.0 | - |
| Stations and lines | not needed (the player builds them) | | |
| Water, everyday places, medical, parks, routes | OSM from Geofabrik, as tier 0 | ODbL | nothing |

Credit lines go in the pack header: "e-Stat (Statistics Bureau of Japan)" and "National Land
Numerical Information (MLIT), processed". Everything is national and general: the same module serves
every Japanese pack through tier 0's hooks, no city-specific source. The 2025 census's commuting
tables come out later (memory: 2027); 2020 was taken in October 2020, with no lockdown in force.

## 3. Basemap and names

OSM in Japan has 230k `place` objects (taginfo for the Geofabrik Japan extract, checked): 98.8%
carry `name` (kanji), **57.6% `name:en`**, 7.4% `name:ja-Latn`, and 57.6% a hiragana reading
(`name:ja-Hira`). `places.py`'s `latin_names` (Shanghai's) takes `name:en` or the Latin name, so
about 60% get romaji at once; the big places nearly all do, the shortfall is hamlets and
neighbourhoods. A kana reading becomes romaji by a fixed table (Hepburn), unlike kanji, which is a
general method for Japan: it would lift the share to perhaps 75-85% (to measure on the extract in
stage 1). The rest show their kanji. The basemap tiles keep OSM's `name`; T-057 is the font
(Zen Maru Gothic) for those kana and kanji labels, P3 and independent of the packs.

## 4. Build cost

- **OSM**: one Geofabrik Japan extract, 2.4 GB (regions: Kanto 495 MB, Chubu 488, Kansai 336,
  Kyushu 301, Tohoku 295, Chugoku 227, Hokkaido 180, Shikoku 87; checked 2026-10-15). Read the
  whole extract once rather than region by region: Chukyo and Keihanshin straddle regions.
- **Disk while building**: the extract 2.4 GB, its feature, water and road caches perhaps another
  3-4 GB (England's points and areas were 920 MB), e-Stat meshes and KSJ files about 1-2 GB:
  8-10 GB at the peak against about 32 GB free. All go once stage 3 is verified (the playbook's
  clean-up), leaving the small derived tables.
- **Machine time** (estimates from London and Paris): reading the extract 30-45 minutes at the CPU
  cap; Tokyo's build about 10 minutes and `routes.py` about 8; each other metro 3-7 plus 2-4; each
  zone 1-3 plus 1-2. Japan's joint commute gravity: 25-40k zones of 2 km (today's whole world is
  23k), about 3 minutes since T-297 (a Japan-sized case solved in 189 s, Japan apart from the
  rest as its own group). About 4-6 hours of machine time in
  all, spread over the stages.
- **Tokens**: a tier 0 city costs about 3M weighted (notes/city-costs.md). Building Japan's 45 packs
  one agent each would be about 135M (15% of a week). With a country module most of that goes: stage
  1 (the module and Tokyo) about 6-8M, like London's two stages; then a pack is a config row and a
  build, so the six other metros in one batch agent about 4-6M and the 38 zones in one or two batch
  agents (one per few Geofabrik regions) about 4-6M. Japan in all: about 15-20M, 2% of a week.

## 5. What the game needs first

1. **Nothing for stage 1.** Tokyo alone is a sampler city, outside the market (`market: -1`).
   Measure in that stage what a dense network costs: a synthetic Tokyo-sized network (500+
   stations) on the pack, worker memory and edit time, before stage 2.
2. **Before stage 2 (more than one Japanese pack)**: commutes between packs travel in the market,
   which today is one Northeast pack. So T-377 (per-pack market, Toronto and London join) and T-409
   (the market split by network) must land, or Japan joins the Northeast's cluster.
3. **Before stage 3**: T-297 or a joint gravity per group of touching zones (Japan solved apart from
   the Northeast); T-301 closing packs, since a pan along Japan opens dozens of zones; and T-362
   stage 3's loose coupling of packs joined by long lines (section 1). The fixed links for "no car
   between landmasses" (T-375) must know Seikan, Kanmon and the three Honshu-Shikoku routes.
4. **Japan tailoring, later**: crush loads (Tokyo's peak trains carry far more than the game's 160 a car), the Inter-Regional
   Travel Survey for long trips, the person trip surveys for rates. Not calibration now (2026-10-15).

## 6. Staged plan

| Stage | Work | Effort |
|---|---|---|
| 1 | `pipeline/japan.py` (N03 outlines, census mesh population, economic census mesh jobs scaled to census workers by municipality, the curve fitted on the flows, P29 schools, telework share, kana romaji in `places.py`); Greater Tokyo built and checked; the dense-network measurement | one agent, 2-3 hours, 6-8M tokens; machine about 1.5 hours |
| 2 | after T-377 and T-409: Keihanshin, Chukyo, Fukuoka, Sapporo, Sendai, Hiroshima in one batch; Japan's joint gravity; Japan joins the market | one agent, 2-3 hours, 4-6M; machine 1-2 hours |
| 3 | after T-297 (or a split solve), T-301 and the coupling: the 32 prefecture zones and Hokkaido's 6, batch-built with automatic checks; clean-up | one or two agents, 3-4 hours, 4-6M; machine 2-3 hours |

## 7. Decisions for Anita

1. **Coverage**: metro packs of whole prefectures plus one zone per other prefecture (option A,
   about 45 packs, Tokyo's about 10 MB). Recommended over medical-area zones (5x the packs) and one
   pack per prefecture (cuts Tokyo apart).
2. **Model the whole country**, not only zones between cities (T-300's rule, kept for the US), all
   47 prefectures including Okinawa; Tokyo's Izu and Ogasawara islands left out. Recommended: it
   is the point of the exercise, and it costs little once the module exists.
3. **Names**: romaji where OSM has it, else romaji made from the kana reading, else kanji.
   Recommended; the alternative is kanji with romaji in a hover.
4. **Order**: stage 1 now, ahead of Berlin, and stages 2-3 only after T-377, T-409 and T-297.
   Recommended, since stage 1 needs no game work and shows what a dense network costs.
5. **Japan's census inputs only, no Japan-specific behaviour yet** (crush loads, travel-survey
   rates): recommended under the no-calibration rule; tailoring as its own task after stage 2.
