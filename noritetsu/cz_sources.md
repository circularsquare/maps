# Czechia register sources (built 2026-09-30)

What the Czech build reads, where each piece came from, and what is still wrong with it.
Czechia is built with `rinf.py`, the generic ERA RINF reader; how that reader works is in its
docstring, and the per-country settings are `rinf_countries/cz.py`. Downloads live in
`data/raw/rinf/cz/` and `data/raw/cz_*` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch cz                                               # RINF, ~20 s
curl -L -o data/raw/czech-republic-260929.osm.pbf https://download.geofabrik.de/europe/czech-republic-260929.osm.pbf
$env:OSMIUM_POOL_THREADS = 2; python extract.py --region cz --pbf data/raw/czech-republic-260929.osm.pbf   # 70 s; delete the .pbf after
python build_model.py --region cz --register rinf:data/raw/rinf/cz       # 80 s
python build_tiles.py --region cz                                        # 40 s
python check_model.py --region cz
python rinf.py --dry cz          # the reader alone, with its full log
```

The Geofabrik file is 0.95 GB; the dated name from https://download.geofabrik.de/europe/czech-republic.html
was used, as for the other countries. `rinf.py --fetch` had a name clash (`country`) that broke
it for every country; fixed in rinf.py on 2026-09-30.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-09-30 (`sections.json`: 3,905 sections, 653 line ids, one version each, all valid
  2025-01-01 to 2050-12-31; `points.json`: 3,677 points). RINF's point names are written
  without diacritics and abbreviated ("Usti n.L.hl.n.os.n.", "AHr Kacice z").
- **OpenStreetMap**, Geofabrik `czech-republic-260929.osm.pbf` (data to 2026-09-29), ODbL:
  track, stations, 887 route relations, and 769 `route=tracks`/`route=railway` relations,
  which carry the timetable numbers (below).
- **Wikidata**, CC0: items with a route number (P1671) and P17 Czechia, with labels and
  lengths (P2043), pulled once on 2026-09-30 into `data/raw/cz_wikidata.json` (356 rows;
  rinf.py's `Q_WIKIDATA` with Q213 and labels in cs and en). The
  build does not read it (`"wikidata": None` in cz.py); it is where the published lengths in
  `check_model.REGISTER["cz"]` come from. P2043 is the "délka" in the line's cs.wikipedia
  infobox; 040, 113, 190 and 200 were checked against the articles themselves.
- **cs.wikipedia** line articles, for the owners of the private lines (below) and the
  REGISTER spot checks. The articles and Wikidata disagree on 113 (article 34.817 km,
  Wikidata 36.884, RINF 36.9).

## Line ids and numbers

RINF's id is `540-00_1501`: Správa železnic's own line number (540), a suffix (-00 the line,
-01.. connecting and parallel tracks, -90.. siding leads) and SŽ's section number (traťový
úsek 1501). Riders never see it. **The number riders know is the timetable number (KJŘ):
"trať 010 Kolín - Česká Třebová"** heads each table of the national timetable, is what station
boards and cs.wikipedia use, and is Wikidata's P1671. So the ref is the KJŘ number, and the
name is the timetable's heading, "010 Kolín – Česká Třebová".

There is no rule from SŽ's number to the KJŘ one (SŽ 540 is 010, 220 is 190, 280 is 220), and
one KJŘ line is several SŽ ids (190 is 220-00_0401 and 360-03_0202). Numbers come from OSM's
`route=tracks` relations, which carry the KJŘ number. **OSM carries a second numbering on the
same `ref` key**: a batch of relations named "220 Nemanice - Plzeň hlavní nádraží" carries SŽ's
number, where "220 – Benešov u Prahy – České Budějovice" carries the KJŘ one. `cz_rel` in
cz.py keeps only the KJŘ ones, told apart by the name (a dash after the number, or en dashes
in it; SŽ's are "NNN A - B" with hyphens) and by range (KJŘ runs 010-346). It needed a hook
in rinf.py, `osm_rel(tags)`, since the ref alone cannot say which is which. Also dropped:
ČD's duplicates of five Jeseník lines, freight-only relations ("nákladní tratě"), disused
ones, Polish and Slovak relations over the border, and the seven corridor ("TŽK") relations.

Seven relations are corrected in `REL_FIX`, from Wikidata's P1671 on each line's item:
Nepomuk - Blatná is 191 (OSM 192), Křižanov - Studenec 252 (OSM 257), Šakvice - Hustopeče 254
(OSM 251). Hrušovany u Brna - Židlochovice (OSM 251) and Děčín - Schöna (OSM 083) have no
source for a number and keep only their names, instead of being grouped with a line at the
other end of the country. 145 and 147 are named in German in OSM and are given the Czech
names of the same places (Falkenau = Sokolov, Franzensbad = Františkovy Lázně).

**Siding ids.** SŽ's siding leads (-90..-99) and the private sidings' own ids (0xx-yy, and
ArcelorMittal Ostrava's V34-xx) lie beside their line's relation. Numbered with it, each
siding junction became a branch point of the line, and build_model then cut the pieces either
side of it out wherever OSM has no passenger route: 132 km, from the middle of lines such as
020 (Choceň - Újezd u Chocně) and 295 (Lipová Lázně - Vápenná). They now never take a number
(`no_ref`, a second rinf.py hook) and are left out of the build (`skip_line`), all 153 of
them except 032-64_1001 Velké Březno - Zubrnice, the Zubrnice museum railway, which OSM has a
passenger route on (T3). 000-00_2051 is SŽ's Vranovice - Pohořelice, not a siding.

Of the 503 ids built: 442 numbered from an OSM relation, 61 with none. 209 numbered lines
come out of the build, and 27 unnumbered ones named "first - last" or from a named relation.

## Infrastructure managers

RINF names none of them; `COUNTRY["im"]` is read off the lines each holds and their
cs.wikipedia articles:

- `0054_IM` Správa železnic, nearly everything.
- `3218_IM` JHMD, the 760 mm lines 228 Jindřichův Hradec - Obrataň and 229 - Nová Bystřice.
- `3324_IM` PDV Railway, 145 Sokolov - Kraslice and 045 Trutnov - Svoboda nad Úpou (the
  article says SŽ took 045 over on 2026-07-01; RINF still files it under PDV).
- `3325_IM` SART-stavby a rekonstrukce, the Železnice Desná (293, Petrov nad Desnou - Kouty
  and Sobotín; the municipalities own it, SART runs it).
- `3559_IM` AŽD Praha, 113 Čížkovice - Obrnice and 063 Dolní Bousov - Kopidlno.
- `3642_IM` Railway Capital, 245 Hrušovany - Hevlín and 256 Čejč - Uhřice u Kyjova.
- `3145_IM` PKP Cargo International (ex AWT), 313 Milotice - Vrbno pod Pradědem and the
  Ostrava colliery lines.
- 44 other codes are left unnamed. All hold only sidings (skipped), except `3366_IM`, which
  also holds the Zubrnice museum railway; which company that is was not established, so that
  line has no operator.

Not in RINF, so they stay OSM lines: the Prague metro (3 lines), trams in Prague, Brno,
Ostrava, Olomouc, Plzeň, Liberec - Jablonec and Most - Litvínov (83 lines), the Petřín and
Karlovy Vary funiculars, and every train service (ČD, RegioJet, Arriva, GW Train, Die
Länderbahn and the cross-border ones), which are operating patterns and named trains over the
register lines. The private railways with passenger trains (JHMD, PDV, SART, AŽD, Railway
Capital, the Zubrnice museum railway) are all in RINF, so no private railway stays OSM-only. No `colours/cz.csv`: SŽ publishes no line colours, and the
S-line colours in Prague, Brno and Ostrava belong to services, which keep OSM's.

## Counts (2026-09-30)

- 236 register lines, 9,101 km: 209 with a timetable number (8,931 km), 27 without (170 km,
  mostly short connecting lines in Prague and border stubs).
- 502 lines in all with the OSM ones (10 of them named trains: EC, EN, Nightjet and the like),
  3,544 stations. Tiles 4.1 MB.
- 2,751 RINF passenger-typed points, of which 2,662 are an OSM station (2,643 distinct), 351 by
  distance alone. All 351 were checked in the log; they are RINF's abbreviations ("Pardubice-
  Ros.n.Lab." for Pardubice-Rosice nad Labem). The 89 with no OSM station are mostly closed
  halts and freight points; see below for the real ones.

## Check

`python check_model.py --region cz`: against RINF's own section lengths, 229 lines of 2 km or
more, median 0.996, 7 off by more than 5% (all short: 133, 094, 195, 033, 311 and two Prague
connecting lines). Against the published figures, 30 lines:

| line | built | published | ratio | |
|---|---|---|---|---|
| 040 Trutnov – Chlumec nad Cidlinou | 101.5 | 101.9 | 1.00 | article checked |
| 086 Liberec – Česká Lípa | 58.6 | 59.0 | 0.99 | |
| 126 Most – Rakovník | 70.2 | 70.0 | 1.00 | |
| 137 Chomutov – Vejprty | 57.9 | 57.9 | 1.00 | |
| 145 Sokolov – Kraslice – Zwotental | 27.3 | 27.5 | 1.00 | PDV Railway |
| 160 Plzeň – Žatec | 106.2 | 107.0 | 0.99 | |
| 161 Rakovník – Bečov nad Teplou | 87.4 | 88.0 | 0.99 | |
| 183 Plzeň – Klatovy – Železná Ruda | 97.0 | 97.4 | 1.00 | |
| 190 Plzeň – České Budějovice | 132.8 | 136.0 | 0.98 | article's km table 135.7, RINF 133.1 |
| 198 Strakonice – Volary | 70.5 | 70.8 | 1.00 | |
| 200 Zdice – Protivín | 102.0 | 101.9 | 1.00 | article checked |
| 202 Tábor – Bechyně | 24.1 | 24.1 | 1.00 | |
| 203 Březnice – Strakonice | 49.3 | 49.1 | 1.00 | |
| 224 Tábor – Horní Cerekev | 69.1 | 69.4 | 1.00 | |
| 226 Veselí nad Lužnicí – Gmünd NÖ | 54.7 | 54.9 | 1.00 | |
| 227 Kostelec u Jihlavy – Slavonice | 53.3 | 53.5 | 1.00 | |
| 235 Kutná Hora – Zruč nad Sázavou | 35.5 | 35.9 | 0.99 | |
| 243 Moravské Budějovice – Jemnice | 20.7 | 20.8 | 1.00 | |
| 250 Havlíčkův Brod – Brno – Břeclav – Kúty | 195.7 | 191.0 | 1.02 | RINF 196.0 |
| 252 Křižanov – Studenec | 33.8 | 33.8 | 1.00 | |
| 255 Hodonín – Zaječí | 37.3 | 37.5 | 0.99 | |
| 262 Chornice – Skalice nad Svitavou | 68.1 | 69.0 | 0.99 | |
| 300 Brno – Přerov | 90.1 | 90.1 | 1.00 | |
| 303 Kojetín – Valašské Meziříčí | 60.8 | 61.0 | 1.00 | |
| 346 Újezdec u Luhačovic – Luhačovice | 9.7 | 9.6 | 1.00 | |
| 113 Čížkovice – Obrnice | 36.6 | 34.8 | 1.05 | article 34.817, Wikidata 36.884, RINF 36.9 |
| 199 České Budějovice – České Velenice – Gmünd NÖ | 48.5 | 52.0 | 0.93 | figure runs on to Gmünd |
| 180 Plzeň – Furth im Wald | 72.9 | 81.2 | 0.90 | figure runs on to Furth |
| 228 Jindřichův Hradec – Obrataň | 42.0 | 46.0 | 0.91 | first 1.3 km dropped, below |
| 229 Jindřichův Hradec – Nová Bystřice | 26.7 | 32.9 | 0.81 | first 3.6 km dropped, below |

Lines left out of REGISTER because the article covers a different stretch from the timetable
number: 030/031 (one article, Pardubice - Liberec), 090 (Praha - Děčín; the timetable's 090
starts at Kralupy), 170/171 (Praha - Plzeň against Beroun - Plzeň - Cheb and Praha - Beroun),
220/221, 225, 240, 241, 244, 246 and several whose Wikidata number is another country's line
(121, 125, 147, 291, 311).

## Still off, and why

- **Junction-ended sections with no OSM route are dropped**, as build_model does everywhere
  (a register section ending at a junction is kept only where an OSM passenger route runs over
  half of it). Czech OSM has no route relations on many regional lines, so 83 km of numbered
  lines is dropped, some of it real passenger track: 238's approach to Havlíčkův Brod through
  Odb Kubešův Mlýn (6.5 km), 292 around Bludov (4.5 km, plus two border stubs), 201 Nasavrky -
  Tábor (4.0), 080 Srní u České Lípy - Žizníkov (4.0), 063 (4.4), 135 Třebušice - Most (4.2),
  270 Přerov - Dluhonice (3.3), and JHMD's 229 and 228 into Jindřichův Hradec (3.6 and 1.3;
  OSM has no JHMD route at all). Whole lines with no OSM route and none built: 242 Dobronín -
  Polná, 253 Vranovice - Pohořelice, 332 Hodonín - Holíč, 345 Nemotice - Koryčany. Mapping
  the missing OSM routes, or a different rule in build_model, would bring these back.
- **OSM track gaps**: 097 Lovosice - Teplice has no path Radejčín - Chotiměř (5.1 km; not
  looked into whether that track is mapped as disused or is really missing), 122
  has no path Praha-Žvahov - Praha-Smíchov (5.0 km), 135 Osek město - Lom u Mostu traces 7.0
  km for 4.2 and is rejected, and Brno-Černovice - Brno hl.n. (807-00) traces 3.1 for 4.3.
- **One detour kept**: 120 Hostivice - Praha-Ruzyně is traced 6.51 km for RINF's 3.80 (crow
  flies 3.84). rinf.py keeps it because it lies on 120's own relation track, and that rule has
  no detour check; 120 reads 1.04 against RINF.
- **RINF's length is misallocated** in 33 places, kept because the trace runs on the line's own
  relation or two traces agree ("length off" in the log): 195 Lipno - Loučovice 2.83 km for
  1.40 (195 reads 1.06 against RINF but 0.99 against Wikipedia), 311 Břidličná - Velká Štáhle
  3.6 for 2.6.
- **Real stations RINF points miss**, because RINF's point is 200-500 m from the OSM station
  and its name is abbreviated past matching: Adamov zastávka, Jindřiš zastávka, Praha-Běchovice
  střed, Praha-Velká Chuchle, Krupka-Bohosudov, Červený Újezd u Votic zastávka, Praha-Jinonice
  (RINF still calls it Praha-Waltrovka). Writing out RINF's abbreviations ("n." nad, "zast."
  zastávka, "P.-" Praha-) would catch five of them but also merge Praha-Holešovice zastávka
  onto Praha-Holešovice 522 m away, so it was not done.
- **Unnumbered passenger lines**: Rumburk - Česká Lípa (45.6 km, SŽ 465, which runs along parts
  of 080 and 081 with no one relation covering it), Moravany - Borohrádek (part of 016's
  track), Litovel předměstí - Mladeč (OSM names it 308, tags it 274), Děčín - Schöna, and the
  Prague connecting lines (Praha-Libeň - Praha-Bubeneč, Praha-Běchovice - Praha-Vyšehrad).
