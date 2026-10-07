# Türkiye sources (surveyed 2026-10-03)

Region code `tr`. Turkey is not in ERA RINF. The register is OSM's named track, the recipe of
kr_register by way of gb_register's repairs (`tr_register.py`, whose docstring is the design).

## Commands

```
# the extract (managing session): Geofabrik europe/turkey-latest.osm.pbf, 0.65 GB
python extract.py --region tr --pbf data/raw/turkey-latest.osm.pbf
python tr_register.py --construction data/raw/turkey-latest.osm.pbf  # construction way ids (needs the .pbf)
python tr_register.py --clip          # after every extract: out abroad, non-services, construction
python tr_stations_pdf.py             # the network statement's station table -> CSV (reference only)
python tr_register.py --names         # the folded line names with track-km
python build_model.py --region tr --register tr_register:data/raw/tr     # ~45 s
python build_tiles.py --region tr                                          # ~15 s
python check_model.py --region tr
```

`--clip` rewrites `data/proc/tr` and is safe to run twice. After a new extract, run
`--construction` first (it reads the .pbf; the result, `data/proc/tr/construction.json`, is kept
so a later `--clip` does not need the .pbf).

## The build (2026-10-03)

- **41 register lines, 8,038 km**, 1,376 stations, 145 lines in all (104 OSM lines: 45 train,
  37 tram, 19 subway, 2 funicular, 1 light rail). 15 named trains.
- TCDD's network: 13,919 km of track (TCDD Taşımacılık 2024 statistics, table 1.1), of which
  **9,495 km is route** (first main line: 8,402 conventional + 991 YHT + 102 "hızlı"); the
  rest is second, third and fourth main lines and station tracks. The register's 8,038 km is
  the passenger part of the 9,495: left out are freight-only branches (FREIGHT, ~360 track-km), the
  lines with no passenger train (Fevzipaşa - Malatya ~250 km, Narlı - Başpınar 67, the
  Syrian-border Karkamış - Nusaybin 325, Uzunköprü - border), and the two stretches OSM maps
  as construction (below).
- **Check** (`check_model.REGISTER["tr"]`, 32 lines against TCDD's network statement section
  table, Wikidata and the 2026 decree): 28 within 5%, the other four explained. Exact or close: Eskişehir -
  Konya 1.00, İzmir - Afyonkarahisar 1.00, Irmak - Zonguldak 1.00, Samsun - Kalın 1.00,
  Alayunt - Balıkesir 1.00, Afyon - Karakuyu 1.00, Fevzipaşa - Kurtalan (Malatya - Kurtalan)
  1.00, Ankara - Kars 0.99 (against the 1,360 km of the Ankara - Kars public-service line),
  İstanbul - Ankara 0.99, İzmir - Eğirdir 0.99, Polatlı - Konya YHT 0.99, Marmaray 0.99,
  Ankara - Sivas YHT 0.98. Off, each explained in the check's note: the Ankara - İstanbul
  high-speed line 0.92 (OSM has the YHT on the old line Karaköy - Yayla and Doğançay - Arifiye,
  where the new track is not mapped under its name), Menemen - Aliağa 1.55 and Torbalı - Ödemiş
  1.06 (both legs of a triangle kept), Başkentray 1.05.
- **The named trains as an outside check**: the Doğu Ekspresi's OSM line is 1,356 km against the
  decree's Ankara - Kars 1,360; the Güney Kurtalan Ekspresi 1,262 against Ankara - Kurtalan
  1,264; the Konya Mavi Treni 688 against Konya - Basmane 688.
- Borders: Kapıkule - Svilengrad ends at RINF's eEU00212 (0.6 m from OSM's track; Bulgaria's
  line 1 already ends there), section Kapıkule - border 1.3 km. Uzunköprü - Pythio ends at
  eEU00186 (3 m off the track), but no passenger train crosses there (no OSM route), so the
  Uzunköprü - border section is dropped as junction-ended and unridden. **No borders.EXTRA
  entry is needed.** A trial build of bg and gr after Turkey's build (tools/ab.py) changed
  nothing in either.

## The line unit: what was measured

| candidate | what it is | verdict |
|---|---|---|
| **OSM track `name`** (chosen) | "Ankara-Kars demiryolu", "Irmak-Zonguldak demiryolu", "Fevzipaşa-Kurtalan demiryolu", "Ankara - İstanbul yüksek hızlı demiryolu", "Marmaray": the railways as tr.wikipedia and Wikidata name them | 98.4% of main + branch rail km named (`probe_kr_ways.py --region tr`), one name per way, so a partition; 82 names after folding |
| TCDD's line codes | 101 İstanbul - Demirköprü ... 205 Kayaş YHT - Sivas YHT (TCDD Taşımacılık 2024 statistics, tables 2.2.9 and 2.3.3, with train-km per code) | official, but TCDD publishes no length or geometry per code; used as the evidence of which lines carry passengers |
| OSM route=railway relations | 70 TCDD lines ("Kalın - Samsun Tren Hattı", "Irmak - Kayseri hattı"), split differently from the track names | coarser coverage, overlapping; not needed |
| Wikidata line items | 159 items, with P402 OSM relation ids and P2043 lengths; the same railways as the track names | the English names and some check lengths |
| TCDD network statement Ek-3.3 | 233 sections A - B with whole-km route lengths and speeds | the check lengths (`data/raw/tr/sb2025_ek33_sections.csv`) |

The named railways are bigger than TCDD's codes (the Ankara - Kars railway is TCDD 108, 110,
115 and 117) but they are what a rider reads, as the UK's Cotswold Line is.

## Sources

| file (data/raw/tr) | what | from |
|---|---|---|
| `tcdd_station_pairs.json` | TCDD Taşımacılık's ticket-sales station list, 593 entries (name, code, often a point) | `https://cdn-api-prod-ytp.tcddtasimacilik.gov.tr/datas/station-pairs-INTERNET.json` |
| `tcddt_2024_istatistik.pdf` | TCDD Taşımacılık 2024 statistics: network km, train-km by line code | adminapi.tcddtasimacilik.gov.tr |
| `sb2027_ek331_hat_uzunluklari.pdf` | Network statement 2027, Ek-3.3.1: line lengths (13,919 km) | static.tcdd.gov.tr/.../sebekebildirimi/2027/331110.pdf |
| `sb2025_ek33_teknik.pdf`, `sb2025_ek33_sections.csv` | Network statement 2025, Ek-3.3: route sections with km (CSV parsed from it) | .../2025/33910.pdf |
| `sb2026_ek3313_istasyon.pdf`, `tcdd_istasyonlar.csv` | Network statement 2026, Ek-3.3.1.3: 928 stations, sidings, halts with status, chainage, "Yolcu İşletme" (+/-), daily passengers (`tr_stations_pdf.py`) | .../2026/3313101.pdf |
| `rg_20260918_khy_hatlar.pdf` | Cumhurbaşkanı Kararı 11800 (Resmî Gazete 33374, 2026-09-18): the 47 public-service passenger lines with km | resmigazete.gov.tr/eskiler/2026/09/20260918-2.pdf |
| `wd_lines.json` | Wikidata railway-line items with P17 Türkiye | query.wikidata.org |
| `tr_boundary.geojson` | OSM relation 174737 (Türkiye) | polygons.openstreetmap.fr |
| `trwiki_hatlar_listesi.wikitext` | tr.wikipedia "Türkiye'deki demiryolu hatları listesi" (historical openings) | reference only |

**Timetables**: TCDD Taşımacılık publishes no open GTFS. The Mobility Database (feeds_v2.csv,
2026-10-03) has 23 Turkish feeds, none national rail: İZBAN's own (mdb-1828,
izban.com.tr/gtfs/rail-izban-gtfs.zip, marked inactive since 2024-02), Metro İzmir (mdb-1824)
and Tram İzmir (mdb-1829), İstanbul's IETT (buses), Flixbus Türkiye (coaches), Kocaeli and
KentKart city bus feeds. The ticket list's `pairs` field is empty in the public file. No
`gtfs_served` check.

## What the reader does

See `tr_register.py`'s docstring. In short:

- **The names**: spellings folded (`fold_key`, NAME_ALIAS); a structure's or a junction's
  name on the track ("Batıbel Tüneli", "Yapı YHT Kavşağı") read as no name or as its line's;
  the İstanbul-first spellings of the high-speed line on Gebze - Köseköy given to the old
  line (TCDD's own "Gebze - Köseköy HT", which every train shares); the Sivas approach given to
  the Ankara - Kars line. Unnamed runs of track through stations take the one line round them
  (`fill_runs`: the Alayunt - Balıkesir railway was 14 pieces without it).
- **Left out of the register** (they stay drawn, and OSM lines where OSM has routes): metro,
  tram and light-rail names on rail track (Sirkeci - Kazlıçeşme T6, Gebze - Darıca metro);
  usage=industrial; FREIGHT (TCDD codes with no passenger train-km in 2024: Muratlı - Tekirdağ,
  Çobanisa - Kemalpaşa, Kayseri's northern bypass, BTK Kars - Georgia, Akçagöze - Başpınar,
  Bozdemir - Mazıdağı, Kütahya - Seyitömer, Tavşanlı - Tunçbilek, Samsun - Azot, Hanlı -
  Bostankaya).
- **Passenger stations**: an OSM station is one when TCDD sells tickets to it (the ticket list,
  matched by name; TCDD's own points are often 0,0 or wrong, so a name with one OSM station is
  that station), when an OSM passenger route stops at it (Marmaray, Başkentray, İZBAN,
  Gaziray, Adaray), or when a route's track ends at it. 377 OSM rail stations are no passenger
  stop (crossing loops). A station goes on every named line within 400 m (a high-speed line
  only YHT stations), and on a line whose track ends within 2 km of it.
- **Junctions**: gb_register's junction ends, plus a line end that reaches another line over
  up to 5 km of unnamed track; but a line end within 1.5 km of a passenger station ends at the
  station (Kars - Akyaka leaves the Ankara - Kars line 0.8 km east of Kars).
- **Suspended** (greyed): Van - Kapıköy ("Van-Sufiyan demiryolu"): TCDD's statistics show no
  passenger train on line 122 in 2024; the Tehran trains are not running.

### The extract clip (`--clip`)

- Out of Türkiye (OSM's boundary): 70 ways (Greek Alexandroupoli - Svilengrad, Iran, Georgia).
- NOT_SERVICE route relations (27): planned and building lines tagged as routes (Antalya -
  Kayseri, Erzincan - Erzurum, Sivas - Zara - Erzincan, the Mersin - Adana - Gaziantep high-
  speed project, KonyaRay, Siirt - Kurtalan, Çukurova airport branch, Kocaeli and Ankara metro
  projects), closed lines (Sütlaç - Çivril, the pre-1971 Karaağaç route), infrastructure
  tagged route=train (Gebze - Halkalı, Konya - Karaman), Samsun - Çarşamba (no passenger
  train), and the **Mersin - Adana regional train, suspended since 2024-04-22** for the
  rebuilding (TCDD; the 2026 network statement has Mersin, Tarsus and Taşkent closed to
  passengers). Metro and tram routes with half their track railway=construction are dropped
  too (Konya's Adliye - Şehir Hastanesi tram).
- Construction ways no remaining route runs over are dropped; KEEP_CONSTRUCTION keeps the
  Yenice - Osmaniye stretch of the Mersin - Adana - Gaziantep line, being doubled under
  traffic.

## Lines vs named trains (`rules/tr.py`)

100 route=train relations, 73 TCDD Taşımacılık's. **Lines**: every YHT service (Ankara -
İstanbul about 20 a day each way, Ankara - Konya about 10, Ankara - Sivas 4 or more; the
İstanbul - Sivas and İstanbul - Karaman pairs, once or twice a day, are the same product and
stay lines too), the regional "Bölgesel" trains (even Basmane - Uşak, once a day: the regional
service of its corridor), Marmaray, Başkentray, İZBAN, Gaziray, Adaray, AYBAN. **Named trains**:
the "Ekspresi" and "Mavi Tren" anahat trains (Doğu, Turistik Doğu, Ankara, Ege, Erciyes, Güney
Kurtalan, Van Gölü, Pamukkale, Toros, Göller, Güller, 6/17 Eylül; İzmir and Konya Mavi Treni),
each once a day or a few times a week, and the international Istanbul - Sofia/Bucharest train.

OSM maps the YHT and the named trains as ways only, with no stop members, so build_model
dropped them all; `extra_route_stops` in rules/tr.py gives such a route the ticket-list
stations within 300 m of its own track (a YHT route only YHT stations). And each service is
two route relations, one per direction, with no route_master: `--clip` adds one over each pair
whose names are the same ends swapped (`pair_directions`; 10 pairs), so there are 9 YHT lines,
not 17.

## Open

- **Adana - Ceyhan (48 km) and Karaman - Ulukışla (~110 km)** are missing: OSM maps the rebuilt
  line as railway=construction in pieces that do not join, and extract.py keeps a construction
  way only under a route. Trains run on both (the Toros and Erciyes Ekspresi; Adana -
  İskenderun). Their track counts through the named trains' OSM lines meanwhile. The
  Mersin - Adana - Gaziantep line is left in three pieces for it ("Lines in pieces" below).
- **Fevzipaşa - Malatya** (TCDD 123, ~250 km) has no passenger train in TCDD's 2024 statistics
  and none in OSM; the 2026 decree names an Elazığ - Adana public-service line over it, so the
  service may come back. It is left out (dropped as junction-ended and unridden).
- **Gaziantep - Karkamış**: the decree names it, TCDD's 2024 statistics show a little passenger
  traffic on line 124, but Karkamış is not on the ticket list and the 2026 network statement has
  it closed to passengers: the register stops at Nizip.
- **İslahiye, Kahramanmaraş, Bahçe**: open to passengers in the network statement, not on the
  ticket list, no OSM route: not stations here.
- The 2026 network statement's "Yolcu İşletme" column says (-) for 200 stations TCDD sells
  tickets to; it seems to mean staffed passenger handling, not whether trains stop, so it is
  not used.
- The YHT runs on the old line between Karaköy and Yayla, and from Doğançay through Arifiye
  to Sapanca; since 2026-10-04 those stretches are sections of the high-speed line, borrowed
  where they lie on the old line's track ("Lines in pieces").
- No `colours/tr.csv` yet (OSM colours on 113 route relations, mostly city lines).
- Kocaeli's Gebze - Darıca metro and İstanbul's M12 have no service yet (construction).

## Lines in pieces (2026-10-04)

Anita, 2026-10-04 ("yes, we can continue doing bridge over shared track"): the UK fix
(gb_sources.md "Lines in pieces") carried to Türkiye. A trip is entered station to station on
a line's strip diagram, so a register line whose sections do not all connect cannot be ridden
across its gap. tr_register now has the `split_pieces` hook build_model calls after
`drop_unridden_sections`, through the shared `pieces.py` over gb_register's track graph (with
Türkiye's settings, as everything else gb_register does here): a gap is bridged over the track
between the pieces where trains run across, the `borrowed` sections crediting the line whose
track it is; what cannot be bridged would become one line per piece.

**Measured** on the build shipped 2026-10-03: 3 of 41 register lines in pieces, 198 km outside
each one's biggest piece. By what lies in each gap (the track found between the pieces in
data/proc/tr):

| line | pieces (km) | gap | cause | done |
|---|---|---|---|---|
| İstanbul - Ankara demiryolu | 444.7 + 96.9 | Köseköy - Sapanca | the line's own track, 24.2 km of it, but it stops 0.2 km short of the rest at Sapanca, where the YHT's junction joint carries the high-speed line's name | bridged over its own track, Sapanca - Köseköy 24.4 km |
| Ankara - İstanbul yüksek hızlı demiryolu | 301.4 + 57.3 + 23.1 | Karaköy - Yayla; Pamukova YHT - Sapanca | shared track: the YHT runs over the old line between Karaköy and Yayla (through the 5.7 km connecting curves, unnamed here), and from Doğançay through Arifiye to its own track at Sapanca; the 23.1 km piece was Sapanca - Köseköy between two junctions, no stop on it | bridged: Karaköy - Yayla 11.6 km (the curves and its own track, not borrowed); Pamukova YHT - Arifiye - Sapanca junction 43.3 km (21.5 its own name), Arifiye - Sapanca junction borrowed (8.8 km) |
| Mersin - Adana - Gaziantep yüksek standartlı demiryolu | 39.3 + 17.3 + 2.9 | Şehitlik - Şakirpaşa, Adana - Ceyhan | no track in the extract: OSM maps the rebuilt line between as railway=construction pieces that do not join ("Open", Adana - Ceyhan); trains run through | left whole in pieces (`KEEP_WHOLE`) |

Türkiye's settings beside the UK's (`tr_register.rules()`): `SLACK` 2.0, because the YHT
from Pamukova runs north to Doğançay and onto the old line, then back west through Arifiye to
Sapanca, 42.8 km for 22.8 km crow-fly; the line's own named track costs half and counts as
under a route (`OWN_COST`), so the old line's Köseköy - Sapanca gap is bridged over its own
track rather than the YHT's beside it (first tried without: the bridge took the YHT's ways,
and since the old line's own ways lie beside them the borrowed section credited itself for 22
of its 24 km); `dense`, so Pamukova YHT, on straight track with no OSM vertex within 150 m,
joins the track graph (without it the bridge started 48 km back at the Bilecik junction and
ran past Pamukova on the old line); and a high-speed line's bridge stops only at YHT stations
(`bridge_stop_ok`, tr_lists' `hs_ok`): Arifiye, not Karaköy or the old Sapanca station.

**Why Mersin - Adana - Gaziantep is left whole, not split**: the gaps are track OSM does not
have (or the extract leaves out), on one line the Toros and Erciyes Ekspresi and the Adana -
İskenderun trains run through. Split, the Şehitlik - Yenice and Şakirpaşa - Adana pieces would
be lines of their own whose ids vanish again once the rebuilt track is mapped, and a ride
across the gap could not be entered on either kind. Revisit when OSM maps the line as open.

**Crediting**: the only borrowed section is the YHT's Arifiye - Sapanca junction, 8.8 km on the
old line's track, and riding it credits İstanbul - Ankara demiryolu, all 8.8 km. The other new
sections are on their own line's track (or unnamed curves) and own it; 0.9 km of ways beside the
borrowed section have no other register line's section near and go to the YHT (logged). The
country's owned total (build_regions.owned_totals) goes 8,683.5 -> 8,717.0 km: the 33.5 km is
track newly owned (the old line's Köseköy - Sapanca, the YHT's Pamukova - Doğançay and the
curves at Karaköy, where an OSM line or nobody owned it), the borrowed km counted once.

**Before -> after** (trial 2026-10-04): register lines 41 -> 41, 8,038.5 -> 8,117.7 km
(borrowed 8.8 among them); lines in pieces 3 -> 1 (Mersin - Adana - Gaziantep, kept whole).
check_model: İstanbul - Ankara demiryolu 0.99 -> 1.04 (566.0 of 545; it now has Köseköy -
Sapanca), Ankara - İstanbul YHT 0.92 -> 1.05 (436.7 of 414; the stretches on the old line).
