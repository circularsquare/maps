# Station ridership (average day)

A small grey line in the station view, "About 2.7 million a day (2024)", from published
station counts joined onto the built stations. Nothing is rebuilt: `tools/station_riders.py`
reads `dist/data/<cc>/stations.json` and writes `dist/data/<cc>/riders.json` and
`dist/data/riders_sources.json`. The app side is a ready patch in
`handoff_notes/station_riders_app.md` (not applied). Research behind the source choices:
`flavour_stats_checklist.md` §2.

    python tools/station_riders.py              # every source, about a minute
    python tools/station_riders.py sbb          # one source's country (all its sources)

Each source is one module in `tools/riders/` (the docstring of `station_riders.py` lists what
a module defines); downloads are in `data/raw/riders/<key>/`, and every run writes a match
report per source to `data/raw/riders/_reports/<key>.txt`: the biggest source stations and the
id each went to, every miss with the reason, every match with how it was made.

## What the number means

- **n is people getting on plus people getting off (entries plus exits) on an average day**,
  the meaning almost every source has. A source counting only boardings or only entries is
  doubled, and says so in `riders_sources.json` "counts" (Belgium, Victoria, Istanbul, and
  Mexico City once its file is in). Annual totals are divided by the days of the year; monthly or daily files by the
  days they cover.
- Where a source gives only an average working day (NS in the Netherlands), "per": "weekday"
  in riders_sources.json and the app says "a weekday". Belgium's weekday, Saturday and Sunday
  counts are combined into a week's average, (5 x weekday + Saturday + Sunday) / 7.
- **year** is the year the figure is for. A fiscal or April-March year is named by its first
  year (Japan's FY2024 and ORR's 2024-25 are both 2024). A part year (Taiwan 2026, the Korean
  city metros 2026) is named by its year; the module docstring says which months.
- **A station served by several operators** gets the sum of their counts when they are
  separate counts of separate gates or platforms (Japan's S12 group; Korea's Korail
  intercity + Korail commuter lines + Seoul Metro at 서울; KRIC's per-line rows at a transfer
  station). That counts a passenger changing between them twice, the usual way such station
  totals are quoted (Shinjuku's "busiest station in the world" figure is one). When the parts
  are of different years the entry carries `y0` and the app shows "2023-25".
- **One operator's figures are never added across years** (S12's FY2019 change of joint-
  station reporting is why: until FY2018 every operator of a shared station reported the same
  joint figure; from FY2019 one carries it with an "X を含む" remark and the others report 0).
- **No figure is ever invented, spread or copied.** A station the source does not name, or
  that no match reaches, gets nothing, and the app shows nothing.
- **A complex split over several ids**: the figure sits on the id the source matched; another
  id of the same complex, with the same whole name within 0.6 km and served by a line the
  source counts, gets `{"at": <id>}` and the app shows the figure it points at (池袋's Seibu
  group g003379, 東京 g003785, 武蔵小杉 g004301; Zürich Hauptbahnhof n1236383343 -> Zürich HB
  c8503000). Tram stops named after the station ("Zürich Flughafen, Bahnhof", "Gare de
  Bègles") do not point at the station's figure.

## Matching

By code where the station ids are made of the source's own code (ch "c<BPUIC>" = SBB's UIC;
fr "fr<UIC>" = SNCF's code_uic_complet; jp "g<code>" = the 国土数値情報 group and station
codes S12 shares with N02), else by name: normalised (case, accents, "Station", "Gare de",
"Bahnhof", "Hbf"/"HB", 駅/站/역, brackets, 臺/台, ヶ/ケ) within the source's radius of its
position (1.5 km by default, 1 km for S12 and ORR; a contained or close spelling only within
0.6 of it), nearest first. A source without positions borrows them (ORR from Wikidata's CRS
codes, SNCF from its own station list, Korea from riders/koreariders' metro layer) or matches a
name unique in the country or a box. Candidates are limited to stations a line of the classes
the source counts calls at (`MODES`), so a tram stop never takes a railway station's count.
Hand decisions are `FORCE` entries in the modules, each with its reason.

## Per country

As of the run of 2026-10-09. "Of" counts stations without junction points; "at" is ids of a
split complex pointing at another id's figure. Match reports have the detail.

| cc | stations with a figure | of | at | sources: stations (years) |
|---|---|---|---|---|
| jp | 8,394 | 9,073 | 5 | s12 (FY2024; 539 stations' latest figure is older, back to FY2011, where an operator stopped publishing) |
| gb | 2,749 | 3,237 | 1 | orr 2,513 (2024-25), tfl 236 (2025) |
| fr | 2,866 | 5,157 | 0 | sncf (2024; a few closed stations older) |
| ch | 1,168 | 2,466 | 1 | sbb (2025 mostly) |
| kr | 1,106 | 1,216 | 1 | kric 414 (2024), korail_gw 301 (2025), kr_metro 259 (2025-26), korail 226 (2023) |
| au | 667 (9 banded) | 2,186 | 0 | vic 312 (2024-25), nsw 301 (2025-26), nsw_lr 60 (2025-26) |
| be | 509 | 1,234 | 0 | wd_be (October 2022) |
| my | 297 | 328 | 0 | my_rapid 154 (2025; Shah Alam line 2026), my_ktm 151 (2025) |
| nl | 267 | 1,117 | 0 | ns (2024, average weekday) |
| tr | 257 | 1,356 | 0 | ibb (2025, Istanbul only) |
| tw | 255 | 519 | 0 | tra 240 (2026 to date), taichung 18 (2026 to date) |
| th | 8 | 822 | 0 | th_drt (2024, Airport Rail Link only) |

Match rate per source (source stations matched / in the source): s12 8,399/8,565 (the 166
misses are closed stations and lines: 十和田観光電鉄, JR Hokkaido's abolished halts, the Ueno
Zoo monorail); orr 2,513/2,586 (73: gb has no node for Billericay, Goole, Guiseley, Hatch End
and others, see handoff_notes/missing_stops.md); tfl 297/307 (Underground stations merged into
the National Rail station's id, which keeps ORR's figure); sncf 2,870/3,005 (closed or
coach-only stops, and tiny halts fr has no node for); sbb 1,169/1,178; ns 267/275 (the Hoek
van Holland line, now RET metro, last counted by NS in 2016); wd_be 509/510; tra 240/241;
taichung 18/19; korail 226/226; korail_gw 321/324 (연천, 신이매, a 9-passenger row "백");
kric 478/478; kr_metro 261/262 (Daegu's 대공원); nsw 302/302; nsw_lr 62/65; vic 313/316;
my_rapid 166/166; my_ktm 209/209; th_drt 8/8; ibb 349/351 (Laleli, Sultançiftliği: not on the
map).

Sources, licences, what n counts (also in riders_sources.json):

| key | source | licence | n |
|---|---|---|---|
| s12 | MLIT 国土数値情報 駅別乗降客数 S12-25 (FY2011-2024) | 国土数値情報 terms, attribution | 乗降客数 per day, operators added |
| orr | ORR Estimates of station usage 2024-25, table 1410 | OGL v3 | entries + exits / 365 |
| tfl | TfL annualised station counts 2025 (Underground, DLR only; fills ids ORR has no figure for) | TfL open data (OGL based) | entries + exits / 365 |
| sncf | SNCF Gares & Connexions, Fréquentation en gares | Licence Ouverte 2.0 | voyageurs (on + off) / 365; RATP passengers included at shared stations |
| sbb | SBB Passagierfrequenz | SBB open data, cite source | DTV, on + off per day over all days |
| ns | NS in- en uitstappers t/m 2024, via Provincie Zuid-Holland | CC0 | on + off, average working day |
| wd_be | SNCB October counts via Wikidata P1373 | CC0 | boardings, week average, doubled |
| tra | TRA 每日各站點進出站人數 (data.gov.tw 8792) | OGDL Taiwan 1.0 | gate entries + exits, daily mean |
| taichung | 臺中捷運各站旅運量 (data.gov.tw 175718) | OGDL Taiwan 1.0 | entries + exits, months with both |
| korail | 철도통계연보 2023 via riders/koreariders | 공공누리 type 1 | 승하차 / 365, intercity trains |
| korail_gw | Korail 광역철도 역별 승하차 2025 via koreariders | 공공누리 type 1 | 승하차 일평균, commuter lines |
| kric | KRIC 역별 승강차실적 2024 via riders/seoulriders | KRIC public statistics | 승하차 / days, capital-area city railways |
| kr_metro | Busan, Daegu, Daejeon, Gwangju, Busan-Gimhae gate files (data.go.kr) via koreariders | 공공누리 type 1 | 승하차 daily mean over all days |
| nsw, nsw_lr | Transport for NSW Opal station usage (monthly) | CC BY 4.0 | entries + exits / days |
| vic | Victoria annual metropolitan and regional station entries 2024-25 | CC BY 4.0 | entries, doubled / 365 |
| my_rapid, my_ktm | data.gov.my ridership OD (Rapid Rail, KTM) | CC BY 4.0 | trips from + trips to / days |
| th_drt | DRT drt2566_02 (Airport Rail Link) via the MOT datastore | Open Data Common | arrivals + departures / days |
| ibb | İBB Raylı Sistemler İstasyon Bazlı Yolcu Sayıları 2025 | İBB open data licence | validations (entries), doubled / days |

## Spot checks (2026-10-09)

Each landed on the id it should; interchange decisions noted.

- **jp**: 新宿 g003700 2,713,386 (JR East 1,333,618 + Keio 728,874 with Toei's lines inside it
  + Odakyu 450,952 + Tokyo Metro 199,942, FY2024; Seibu-Shinjuku is its own station, 147,390).
  渋谷 2.90M, 池袋 2.36M (g003379, an N02 group of its own, points at it), 東京 1.26M (g003785
  points at it), 横浜 1.94M, 大阪 751k with 梅田, 大阪梅田, 東梅田, 西梅田 each their own
  group's count, 京都 639k, 博多 505k, 札幌 172k, 二月田 726.
- **gb**: Clapham Junction 66,979 (ORR 24.4M a year / 365), London Waterloo 192,848 (Waterloo
  East its own, 18,742), Liverpool Street 268,536, Stratford 141,023, Birmingham New Street
  100,339, Glasgow Central 69,298, Edinburgh Waverley 62,343 (ORR calls it "Edinburgh").
  Underground ids get TfL's count where ORR has none: Bank 112,656 (TfL's Bank and Monument
  row), Paddington's Underground/Elizabeth line id 158,815 (TfL counts its gates, Elizabeth
  line included; ORR's London Paddington id keeps 191,499). Hammersmith's two stations each
  their own row.
- **fr**: Paris Gare du Nord 704,176 (SNCF's figure for the whole station, RER B and D
  included), Magenta its own id 106,222, Gare de Lyon 310,203, Saint-Lazare 312,585, Gare
  Montparnasse 188,836, Lyon Part-Dieu 116,260. Châtelet - Les Halles has nothing: RATP runs
  it and SNCF's file does not have it.
- **ch**: Zürich HB 449,800 on c8503000, and OSM's "Zürich Hauptbahnhof" n1236383343 (the
  40-line node people will click) points at it; Zürich HB SZU (Sihltal line, not counted by
  SBB) nothing. Bern 177,000, Basel SBB 103,800, Lausanne 97,500, Genève 83,100, Zermatt 11,900.
- **nl**: Utrecht Centraal 175,179, Amsterdam Centraal 159,833, Schiphol Airport 74,382 (all
  average weekday 2024). Metro and tram stops: nothing.
- **be**: Bruxelles-Midi 88,743, Bruxelles-Central 83,469 (2022: weekday 49,476, Saturday
  24,143, Sunday 20,618 boardings, week average doubled), Gent-Sint-Pieters 82,495,
  Antwerpen-Centraal 57,121. A first version took whichever of the three day types Wikidata
  listed first; the qualifier is now read.
- **tw**: 臺北 121,170 (TRA only; the Taipei MRT id 台北車站 has nothing), 板橋 49,698, 高雄
  31,815; 烏日 TRA + Taichung MRT on one id, 2,757; 高鐵臺中站 MRT 18,671.
- **kr**: 서울 340,953 (Korail intercity 108,587 for 2023 + Korail commuter 53,612 for 2025 +
  Seoul Metro 1/4 and AREX 178,753 for 2024, shown "2023-25"), 강남 188,579, 잠실 193,891,
  서면 130,469, 부산역 110,624 (Korail + Busan Metro), 동대구 107,412, 수원 128,240. Names
  shared nearby decided by hand: 동해선 동래/좌천 vs Busan Metro's, 5호선 양평 vs 중앙선 양평.
- **au, my, th, tr**: in the helper reports summarised under "Per country" (Sydney Central
  143,429 trains and metro, the light rail's Central Grand Concourse its own 36,121; Flinders
  Street 107,580; Southern Cross 116,176 = Metro + V/Line; KL Sentral 73,537 = Kelana Jaya +
  monorail + Komuter + ETS + Intercity; Bukit Bintang 76,871; Phaya Thai 27,114, the Airport
  Rail Link only on an id shared with BTS; Yenikapı 391,860, Üsküdar 239,037).

## Not done, and why

- **Taipei Metro**: the only per-station file is the monthly hourly OD (data.gov.tw 128506),
  312 MB a month. Not downloaded (over the size asked about). Streaming one or a few months
  and keeping only station totals would work if that is acceptable.
- **Mexico City** (`cdmx.py`) and **Poland** (`utk.py`): modules written, `READY` false until a
  file is in `data/raw/riders/cdmx/` or `data/raw/riders/utk/`. datos.cdmx.gob.mx and
  metro.cdmx.gob.mx time out from here; dane.utk.gov.pl answers with a bot check (Incapsula).
  Untested against the real files.
- **Queensland**: data.qld.gov.au downloads answer with an AWS bot check.
- **Thailand** beyond the Airport Rail Link: DRT publishes BTS, MRT and the Red Lines only as
  operator totals. The full drt2566_02 CSV (2025 on) is on drt.gdcatalog.go.th, which did not
  connect; the datastore copy used stops in February 2025.
- **IDFM validations** (Paris RATP-run RER and metro stations, Châtelet - Les Halles): not
  done; validations are entries only and the files are large.
- **Belgium**: 40 small halts with only a weekday count for 2022 (no weekend service, probably)
  are left out rather than assuming zero weekend passengers.
- **Japan**: 539 stations show an older year, where an operator has stopped publishing
  (mostly JR East and JR Hokkaido unstaffed stations). Groups where one operator's latest year
  is older than another's lose that operator's figure (the one-year rule above).
- **Istanbul**: transfers through a gate (Marmaray to the metro at Yenikapı, Üsküdar, Ayrılık
  Çeşmesi) count as entries there, which the doubling then also counts as exits; the busiest
  interchanges read high by that.
- **Australia**: NSW months under 50 taps are published as "Less than 50"; 9 quiet stations
  get a band computed from those limits ("0-4"), not a published band.
- Not looked at: the US, Canada, Spain, Italy, Sweden, Ireland, Argentina and the rest of
  `flavour_stats_checklist.md` §2's moderate and hard rows.
