# Sweden register sources (built 2026-10-02)

What the Swedish build reads, where each piece came from, and what is still wrong with it.
Sweden is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/se.py`. Downloads live in `data/raw/rinf/se/` and `data/raw/gtfs/se/`
(gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch se                     # RINF and Wikidata, a few seconds
python extract.py --region se --pbf data/raw/sweden-latest.osm.pbf     # delete the .pbf after
python gtfs_served.py --fetch se              # Samtrafiken's feed via Transitous, 44 MB -> 2.4 MB
$env:OMP_NUM_THREADS=2; python build_model.py --region se --register rinf:data/raw/rinf/se   # ~70 s
python build_tiles.py --region se             # ~25 s
python check_model.py --region se
python rinf.py --dry se                       # the reader alone, with its full log
```

The first build (2026-10-02) ran with two shared changes not yet landed, sent to the managing
session as diffs: the `se` branch of `build_model.looks_like_service` and `SPLIT_STRONG` in
`gtfs_served.py` (both below). Without them Nattåg 93 counts as a 1,418 km line, and
Kolmården - Åby södra (Nyköpingsbanan, 11.7 km) and Kungsör - Valskog (Svealandsbanan, 9.0 km)
are dropped.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-02 (`sections.json` 1,184 sections, one version each; `points.json` 1,114 points,
  all with coordinates). 68 line ids, 10,800 km. Infrastructure managers: Trafikverket
  (`0074_IM`), Inlandsbanan AB (`3779_IM`, stråk 99), A-Train (`LQB6_IM`, Arlandabanan's tunnel
  section) and Øresundsbro Konsortiet (`3872_IM`, the bridge). Point types are Trafikverket's
  driftplatser: 487 "small station" (20), 601 junction (80), 2 station (10, Stockholm C and
  Göteborg C). Halts are not in it (Rönninge, Tullinge, Stuvsta).
- **Trafikverket's line names.** RINF's line id is Trafikverket's stråk number ("01" to "99"),
  and Trafikverket names its stråk: Järnvägsnätsbeskrivning 2013, bilaga 3.6 "Lutningar per
  stråk" (`bransch.trafikverket.se/contentassets/5b34a8ec0ccd41ba9106d6733f3c2d6a/bil_3_6_jnb_2013_lutningar.pdf`,
  read, not kept: "Stråk 1 Västra stambanan", "Stråk 22 Stockholm, Älvsjö-Ulriksdal,
  Sundbyberg", "Stråk 63 Bergslagspendeln"...), and sv.wikipedia's "Järnväg i Sverige" lists
  all 96 with Trafikverket's names. Names are facts; the network statement carries no open
  licence, sv.wikipedia is CC BY-SA 4.0.
- **sv.wikipedia line articles**, raw wikitext through the API, retrieved 2026-10-02: each
  line's infobox `längd` (the check below) and `persontrafik`, and article text where a line's
  passenger service was in doubt (Bergslagsbanan, Norra stambanan, Stambanan genom övre
  Norrland, Västerdalsbanan, Skånebanan, Dal-Västra Värmlands Järnväg, Mälarbanan, Ådalsbanan).
- **Samtrafiken's national timetable** (Trafiklab "GTFS Sverige 2", mdb-2661), through
  Transitous's open mirror `api.transitous.org/gtfs/se_Trafiklab.gtfs.zip`, fetched
  2026-10-02 and slimmed to rail: 26,391 trips, 24 agencies (SJ, Mälartåg, Öresundståg,
  Skånetrafiken, Västtrafik, SL, Norrtåg, Tåg i Bergslagen, Krösatågen, Tågab, Snälltåget, Vy,
  VR...), calendar 2026-09-09 to 2027-08-31, trips to the December timetable change.
  Trafiklab's terms (gtfs_sources.md).
- **Wikidata**, `wikidata.json` from `--fetch` (707 rows): route numbers (P1671) on Swedish
  lines. Not used for names in the end (below).
- **OpenStreetMap**, Geofabrik `sweden-latest.osm.pbf` (2026-10-02), ODbL: 27,564 track ways,
  215 route relations (155 train), 85% of main-line km under a route. Swedish OSM tags
  driftplatser `railway=site`, which the extract does not keep, so a halt with no
  `railway=station/halt` node (Torneträsk, Krokvik, Rensjön, Södra Vi) has no OSM station.

## Choosing the name source

The brief said Sweden's RINF ids are "track-section numbers" needing a name map. They are
Trafikverket's stråk numbers, the corridors of its network statement, and Trafikverket names
them. Three candidates:

- **Wikidata P1671** (449 Swedish lines): its one- and two-digit numbers are the stråk numbers
  (1 Västra stambanan, 21 Malmbanan), but shared with unrelated items (21 is also Lidingöbanan,
  12 Nockebybanan, 10 four railways). Its three-digit numbers (001 Luleå - Boden, 336
  Stockholm C - Järna...) are a separate, older bandel numbering with `part of` (P361) links to
  the named lines, CC0; a useful reference, but not RINF's ids.
- **Trafikverket's bandel numbers** (411, 512, 811 in the network statement): finer, but RINF
  does not carry them.
- **Trafikverket's stråk names**: picked. They are the operator's own names for exactly RINF's
  ids, and they are the names riders use for the main lines.

A stråk is a corridor, though, not always one line. Stråk 2 holds Nyköpingsbanan, Nässjö -
Ekenässjön and Alvesta - Gemla; stråk 7 holds Luleå - Boden and Vännäs - Umeå; stråk 22, 23,
24 and 27 are the Stockholm, Göteborg and Malmö city areas. So `se.py` files each section
under a named line by chains of RINF points, and where a stråk is split the pieces take the
name of sv.wikipedia's article (Nyköpingsbanan, Citybanan, Citytunneln, Söderåsbanan,
Lommabanan, Trelleborgsbanan, Kontinentalbanan). Three names are mine, for pieces with no
article: "Vännäs–Umeå", "Nässjö–Vetlanda" (Trafikverket's stråk 81 is "Nässjö–Åseda", but the
line now ends at Vetlanda) and "Kristinehamn–Nykroppa" (stråk 69's own name, shortened).

## How `se.py` builds the lines

- **LINES**: 54 named lines as chains of RINF points; each chain claims the sections on the
  shortest path (RINF km) between consecutive points. A section on two chains is the first
  line's (Stockholm C - Karlberg is Ostkustbanan's, so Mälarbanan starts at Tomteboda).
  Sections no chain claims go to their stråk's own line (`DEFAULT`: parallel tracks and
  curves); stråk with no line here are left out.
- **Mälarbanan runs to Örebro C**, as Mälartåg's trains do. Trafikverket and sv.wikipedia end
  it at Hovsta (187 km) and file Hovsta - Örebro under Godsstråket; but Hovsta is no stop, and
  ending there left Arboga - Hovsta a junction-ended section only Arboga - Örebro's 50 km
  non-stop run crosses, which the timetable check does not take as evidence.
- **FREIGHT**, claimed first and left out: Råtsi - Svappavaara, Gällivare - Koskullskulle,
  Koijuvaara - Aitik (ore), Birsta - Fillan (Sundsvall harbour), Åstorp - Kattarp (Skånebanan's
  bypass, "trafikeras i regel inte"), Ockelbo - Storvik ("Samtliga persontåg på banan går
  numera via Gävle"), Jädersbruk - Fellingsbro - Frövi (Mälarbanan's old route; no train in the
  feed calls at Fellingsbro), Storuman - Gunnarn and Orsa - Kallholsfors (filed under
  Inlandsbanan's stråk, toward Lycksele and Bollnäs). Freight-only stråk (42 Piteåbanan, 43
  Umeå–Holmsund, 44 Hällnäs–Storuman, 46 Forsmo–Hoting, 47, 49, 51, 52, 54, 55, 64, 68, 74, 82,
  89, 92, 34 Grycksbobanan) have no line here.
- **FORCE**: Hallsbergs personbangård - Skymossen to Godsstråket (the Motala trains' track; by
  RINF km the way round through the rangerbangård is shorter).
- **LENGTH**: Hallsbergs personbangård - Skymossen is 10.78 km in RINF for 7.7 km of track
  (5.4 km as the crow flies), read as 7.7.
- **FOLD**: Lockarp's point is on the Trelleborg line north of the triangle there, and RINF
  routes Svågertorp through it to both Ystad and Trelleborg; the Svågertorp line joins the
  Trelleborg line south of it and has its own curve onto the Ystad line. Svågertorp - Lockarp
  is folded into Lockarp - Skabersjö and Lockarp - Västra Ingelstad, which then start at
  Svågertorp. Traced through the point, it doubled back over 2-4 km no Ystad train runs.
- **`cut_at_junctions`** (rinf.py, as Portugal): sections end where another line meets.
  Without it a branch ended partway along the section of the line it joins (Haparandabanan at
  Buddbyn, inside Malmbanan's Boden - Holmfors; Tjustbanan at Bjärka-Säby; 27 places), and the
  lines did not connect: Boden - Haparanda, Linköping - Västervik, Kristianstad - Karlskrona had
  no path over register lines.
- **Stations**: `osm_stops` (OSM stations a train route stops at become stops on the sections
  they lie on: RINF has no halts), and `fix` gives RINF's genitive names ("Halmstads central",
  "Bodens central", "Hallsbergs personbangård") the bare town name as a second name, so they
  match OSM's "Halmstad C", "Boden Central", "Hallsberg". 489 passenger-typed points, 451 an
  OSM station.
- OSM's `route=railway` relations are not read (`osm_rel` returns None).

## What is in the register

56 register lines (52 names; Sala–Oxelösund is in three pieces and Godsstråket genom
Bergslagen and Inlandsbanan in two, split by other lines' track), **9,124 km**; 1,013
stations in all, 609 on register lines. 100 OSM lines: Stockholm's metro, Tvärbanan,
Roslagsbanan, Saltsjöbanan, Lidingöbanan, Nockebybanan, the Göteborg, Norrköping and Lund
trams, and the train patterns (Pendeltåg 40-48, Pågatåg, Øresundståg, Västtåg, SJ's tables).

**Named trains** (the `se` branch below): SJ's Nattåg 93 Stockholm - Narvik and Inlandsbanan's
summer train, Tåg 37 Mora - Östersund (one a day each way, June - August). VR's Juna PYO 276
(Kolari) is not in the extract.

**Not running, greyed**: Dal Västra Värmlands Järnväg, Mellerud - Billingsfors (38.5 km). Its
only trains are summer tourist trains (sv.wikipedia: "Sommartid körs turisttrafik på sträckan
Bengtsfors - Mellerud ToR"), which are not in the feed; by Anita's seasonal rule it should be
drawn as running (question below).

**Inlandsbanan comes out as running**: 1,046.6 km, Mora - Brunflo and Östersund - Gällivare.
Its manager, Inlandsbanan AB, has no agency in the feed and nothing else calls at its stops,
so gtfs_served leaves its sections "unknown" (drawn as they are) rather than closed; no summer
snapshot was needed. Kristinehamn - Mora (the line's southern third) is not in RINF.

**Dropped as unridden** (junction-ended, no OSM route and no train in the feed): Stambanan
genom övre Norrland's Bräcke - Vännäs (sv.wikipedia: "Persontrafik: Nej (Bräcke–Vännäs)"),
Ådalsbanan's Västeraspby - Sollefteå - Långsele ("Nej (Långsele–Västeraspby)"),
Västerdalsbanan (passenger trains ended 2011), Skelleftebanan, Markarydsbanan's Markaryd -
Eldsberga, and junction curves.

**Kept with few trains**: Kil - Nykroppa - Hällefors - Ställdalen (Bergslagsbanan) and
Kristinehamn - Storfors - Nykroppa: Tågab's Falun - Ludvika - Kristinehamn train, Sundays
only in autumn 2026 (sv.wikipedia, "Bergslagsbanan"), served in the feed.

## Check

`python check_model.py --region se`: against RINF's own section lengths, 56 lines, median
0.995, none off by more than 5%. Against sv.wikipedia (WP) infobox lengths, 36 lines in
`REGISTER`, every one within 4% or off by exactly what its note explains:

| line | built | WP | ratio | |
|---|---|---|---|---|
| Södra stambanan | 481.0 | 483 | 1.00 | |
| Ostkustbanan | 397.0 | 400 | 0.99 | |
| Mittbanan | 360.2 | 358 | 1.01 | |
| Dalabanan | 264.2 | 265 | 1.00 | |
| Norge/Vänerbanan | 294.8 | 300 | 0.98 | |
| Kust till kust-banan | 401.5 | ~406 | 0.99 | WP text: ~350 + 56 |
| Godsstråket genom Bergslagen | 299.3 | 311 | 0.96 | Hovsta - Örebro is Mälarbanan's |
| Botniabanan | 183.9 | 185 | 0.99 | |
| Bohusbanan | 177.0 | 180 | 0.98 | |
| Svealandsbanan | 113.4 | 115 | 0.99 | |
| Nyköpingsbanan | 108.1 | 109 | 0.99 | |
| Västra stambanan | 483.3 | 455 | 1.06 | also the old line via Södertälje (31 km) |
| Bergslagsbanan | 396.4 | 337 | 1.18 | also Frövi - Ställdalen (63 km) |
| Malmbanan | 433.9 | 473 | 0.92 | WP runs on to Narvik (Ofotbanen, 43 km) |
| Stambanan genom övre Norrland | 281.5 | 626 | 0.45 | Bräcke - Vännäs has no trains |
| Ådalsbanan | 120.0 | 175 | 0.69 | Västeraspby - Långsele has no trains |
| Inlandsbanan | 1046.6 | 1288 | 0.81 | Kristinehamn - Mora not in RINF |

(and Jönköpingsbanan, Älvsborgsbanan, Kinnekullebanan, Fryksdalsbanan, Nynäsbanan, Rååbanan,
Vaggerydsbanan, Citybanan 1.00-1.01; Värmlandsbanan 1.03, Västkustbanan 1.04, Skånebanan
0.98, Viskadalsbanan 0.97, Haparandabanan 0.97, Blekinge kustbana 0.96, Ystadbanan 0.96,
Norra stambanan 0.97, Mälarbanan 1.06, Stångådalsbanan 1.11, Tjustbanan 0.82, each as its note
says.)

End to end along the built lines (shortest path over register sections), against WP:
Stockholm C - Göteborg C 452.3 (455, 0.994); Katrineholm - Malmö C 481.0 (483, 0.996);
Göteborg C - Lund C 279.9 (283, 0.989); Göteborg C - Kalmar C 350.1 (~350); Stockholm C -
Sundsvall C 394.0 (400, 0.985); Sundsvall C - Storlien 356.5 (358); Uppsala - Mora 263.1 (265);
Luleå C - Riksgränsen 430.5 (Wikidata 430); Kil - Gävle 333.9 (337); Storvik - Mjölby 307.3
(311); Boden - Haparanda 158.2 (159); Linköping - Kalmar 235.4 (235); Linköping - Västervik
115.5 (116); Laxå - Charlottenberg 201.8 (202); Kristianstad - Karlskrona 128.8 (130); Malmö
C - Ystad 69.1 (67 via Citytunneln, 1.03).

## Shared changes this build needs (sent as diffs, 2026-10-02)

- `build_model.looks_like_service`, `se` branch: named trains are service=night/car,
  "Nattåg", "Snälltåget", "Inlandståget", operator IBAB, and FI_TRAIN / EU_TRAIN; everything
  else (Samtrafiken's tables as OSM maps them) is a line.
- `gtfs_served.py`, `SPLIT_STRONG = {"se"}`: in the second pass over the split graph, a whole
  section crossed by a run that starts or ends at one of its ends, or is under LONG_RUN_KM and
  within STRAIGHT of the crow-fly line, is strong evidence, as in the main pass. Needed for
  Kolmården - Åby södra and Kungsör - Valskog, junction-ended pieces with no OSM route whose
  only trains are found in the split graph.
- `gtfs_served.FEEDS["se"]`: landed by the managing session.

## Still off, and why

- **The Öresund bridge**: Öresundsbanan ends at Lernacken. RINF's border point EU00141 is on
  Peberholm, which is Danish and 2.3 km past the end of the Swedish extract's track, so the
  bridge (stråk 98, Øresundsbro Konsortiet) is not built; the Øresundståg and SJ's Copenhagen
  trains run on beyond Hyllie with no border point ("no border point" in the build log). Needs
  Denmark built, or a hand-added point in `borders.EXTRA`.
- **Register gaps** the ownership step lists (track only an OSM pattern owns): 5.6 km at
  Södertälje syd under Tåg 80 (not looked into), 2.0 km at the new Kiruna station, 2.2 km near
  Tingvallsvägen (Västtåget), and short border stretches at Charlottenberg, Storlien,
  Riksgränsen and Ed, where OSM's track runs on past RINF's border point.
- **Dal Västra Värmlands Järnväg** is greyed though it has summer trains (above).
- **Haparanda - Tornio** (0.9 km to the border point): "border" in the feed check, left as it
  is; Finland needs a rebuild so the crossing joins.
- **Names**: lines have no English names (Wikidata's stråk items are not joined), and three
  piece names are mine (above). No `colours/se.csv`: Trafikverket publishes no line colours;
  48 OSM lines carry OSM's colours.
- **Södra stambanan at Lund**: Klostergården - Lund C and Hjärup - Åkarp trace 0.5-0.9 km off
  RINF's lengths (kept on their own track, logged as "length off").
