# One owner per piece of track: prototype and comparison

2026-10-01, for maps-ee. Prototype in this folder (`own.py`, `report.py`, `compare.py`,
`rides.py`); nothing in the project was edited and no build was run against project outputs.

**Status: prototype only, not adopted; waiting on Anita's answers to the decisions below.**
Copied here from the session scratchpad, which is temporary: the scripts may still point at
scratch paths, and the `out/` results they wrote are not kept.
Every number below comes from today's `dist/data/<cc>/` and `data/proc/<cc>/`, read only.

## What I built

`own.py <cc> [--bbox]` gives every drawn way one owner, then gives every section of every line
a **footprint**: the owner sections it runs over, as ranges `[lo, hi]` along them.
`compare.py` runs a fixed set of test rides through a Python port of today's app crediting
(`creditSpans`, `ownTrack` with `onRegister`, `countable`, `uniqueShares`, `countedPart`, all
from `dist/index.html`, over today's `credits.json`) and through the footprints, side by side.

Run on all of Switzerland and Belgium, and on slices: Tokyo (139.55-139.95 E, 35.55-35.82 N),
Paris (2.10-2.60 E, 48.70-49.02 N) and Bordeaux-Dax (for TER 52). In a slice, totals and
per-line figures cover only sections whose centre is inside the box. Russia was run whole for
sizes, build time and the fixed-rule counts only (no ride comparison).

Outputs: `out/<tag>/model.pkl` and `out/<tag>/footprints.json` (the shippable shape),
`out/compare_<tag>.txt` (all rides, lines that differ by over 1 km), `out/compare_ch_all.txt`
and `out/tokyo_all.txt` (every line either model credits).

## The design

### The unit: the owning line's section

A ride is still stored as line + from + to. It resolves to the ridden line's sections as now
(`pathBetween`). What changes is what a section credits: instead of "every section within 45 m",
it is the section's footprint, a list of `[owner section, lo, hi]`.

The counting unit is the **owner line's section** (station to station), with ways only
deciding which owner section a ride covers and how much of it. Why not ways themselves:

- A line's length has to come from the line. Double track is two ways, quadruple track four;
  summing ways doubles every double-track line. Both tracks of a pair project onto the same
  stretch of the one owner section, so riding either track covers that stretch once.
- Ways are split wherever a mapper split them, and change between extracts. Sections are
  already the stable, rider-facing unit: they have stations, km, `closed`, strip-diagram rows,
  and saved rides resolve to them.
- The app already reasons in "fraction of a section" (`creditSpans`, `sliceLine`), so the range
  representation and the 150 m gap closing carry over unchanged.

### How a way gets its owner (in this order)

1. **A register line, by geometry.** register_way_lines' test as it stands: at least 60% of the
   way within 40 m of the line, kind family must agree. Then: the way's own `name` naming a
   candidate settles it; a candidate whose high-speed flag (line or section) agrees with the
   way's `highspeed=yes` is preferred, but a disagreeing flag **no longer excludes** a line
   when it is the only one there; then the nearest line at the way's midpoint; candidates
   within 0.5 m of each other are re-measured by mean distance along the whole way (a midpoint
   tie is often two lines crossing there); only a true tie falls to the ref rule.
   Changed from today: tram, light rail and subway may own each other's ways on the same rails
   (8 m), never for a guided line. Lausanne's m1 is `railway=light_rail` on the ground and
   "subway" in the Swiss register, and without this its 7.7 km were counted twice (register
   line M1 and OSM line m1).
2. **Station throats** (new pass): a way no register line passed for, used only by operating
   patterns (OSM lines whose own route ways are at least 60% register-owned), lying at least
   60% within 150 m of a register line of its kind, goes to the nearest such line. Platform
   roads and crossovers at big stations: be 7.6 km of way length, ch 28.9, Tokyo slice 21.8,
   Paris slice 48.7. Without it every S-Bahn and IC "owned" a few hundred metres at each big
   station.
3. **The register line an OSM twin was merged into**, where that twin's relation used the way
   (Tokyo slice: 20 ways).
4. **One OSM line** (not a named train) whose route relation uses the way: lowest ref by
   natural sort ("2" < "10" < "A3"), empty refs last, then name, then line id. Deterministic
   and independent of build order.
5. **Nobody**: "svc" (only named trains run there), "none" (no route at all), "abroad" (way
   midpoint outside the country outline from regions.json, buffered 150 m: the neighbour's).

### How a section maps onto owners

An OSM section's geometry is cut from the very ways of its route relation, so each segment of
it is found on its way **exactly** (coordinates rounded to 1e-5 as the build writes them; a
3 m nearest-way fallback catches the rest). No buffer. The way says which line owns it; the
piece (densified to 50 m) is projected onto the nearest section of that line within 160 m,
which says which owner section and where along it. Pieces are merged per owner section.

- **Partial coverage**: the projected ranges, merged with the app's own rule (gaps under 150 m
  closed, capped at 10% of the section; ends snapped within the same tolerance).
- **Junctions**: each side of the junction is owned by its own line, so an IC over a junction
  credits the line it leaves up to the junction and the line it joins from there. A crossover
  of a few tens of metres gives a few tens of metres of the line it belongs to.
- **Parallel lines of different owners a few metres apart** (Belgium 36/36N, 25/27, 161/161A,
  50A/50C; Swiss CBT Est/Ovest; Shinkansen beside conventional): each way is decided on its
  own, nearest line wins, so a ride credits only the pair its relation runs on. Mini-Shinkansen
  over conventional ways credits the conventional line. Luxembourg's 4 beside 6 is the same
  case (not run).
- **Ways no line runs over** count nowhere, as today.
- **A register section owns itself whole** (footprint `[[self, 0, 1]]`). See "Register lines
  drawn on the same rails" for the one place that is not quite right.
- **Straight-line fallback sections** (no ways) match nothing and credit nothing: ch's IRE 3
  Basel Bad Bf - Erzingen (62.6 km chord through Germany), R42 Moutier - Sonceboz (18.6 km),
  the Parsennbahn's upper section. Today the first two count as 81 km of "own track".

### What counts

- Register line: completion = ridden share of its sections, as now.
- OSM line, its own route percentage "in full": per section, the share of its footprint that
  is ridden. A tram's own percentage includes the street it shares with a lower-numbered line.
- Named train: route percentage the same way, never in totals.
- Country and operator totals: register sections whole + each OSM owner section's owned spans
  (the parts of it on ways its line owns). Each way has one owner, so each piece of track is
  counted once by construction; `ownTrack`, `onRegister`, `uniqueShares` and `OP_UNIQ` go.

## Comparison with today

"Adds" is what the ride adds to the country total; "credits" is km of a line's own percentage.

### Belgium (country total today 3,552.6 km, ownership 3,582.4)

| Ride | Adds today | Adds ownership | Ridden |
|---|---|---|---|
| IC-01 Oostende - Eupen, all | 418.1 | 325.3 | 325.7 |
| L.36 Schaerbeek - Leuven (register line) | 69.4 | 26.3 | 26.3 |
| S1 Brussel-Noord - Mechelen | 58.9 | 20.2 | 20.5 |
| IC-12 Brussel-Noord - Leuven | 70.6 | 28.6 | 28.6 |
| IC-16 Brussel-Zuid - Namur | 102.2 | 64.6 | 64.8 |
| Tram 25, all | 10.9 | 11.1 | 11.3 |
| Metro 6, all | 15.7 | 15.3 | 16.5 |
| Tram 97, all | 9.2 | 9.6 | 9.6 |
| ICE 79 Brussel-Noord - Liege (named train) | 66.0 | 100.1 | 100.6 |
| Eurostar Brussel-Zuid - border (named train) | 74.5 | 87.8 | 87.7 |
| Kusttram Oostende - De Panne | 36.3 | 36.3 | 36.3 |
| Charleroi M2, all | 16.5 | 17.9 | 17.9 |
| All 12 together | 716.3 | 612.3 | |

Differences over 1 km, and which is right:

- **Parallel register lines (ownership right).** Today a ride completes both lines of every
  four-track corridor: riding L.36 Schaerbeek - Leuven completes 26.2 km of L.36N too; IC-12 on
  36N completes 26.2 km of L.36; IC-16 completes 161 (59.2) and 161A (26.7); S1 completes 25
  (20.2) and 27 (18.9). So the country total gains about 2 km per km ridden there. Under
  ownership each ride credits the pair its relation is mapped on: S1 7.0 km of 25 plus 11.9 of
  27, IC-16 48.5 of 161 plus 10.6 of 161A. Which pair is only as good as the OSM relation.
- **Station clutter (ownership right).** IC-01 today credits 1-3 km of 25 other register lines
  (66, 27, 35, 53, 34, Schaerbeek yard lines...) where they lie within 45 m at stations. 0 now.
- **High-speed flag (ownership right, known failure).** ICE 79 today credits only HSL 2
  (66.0 km): its one long section is flagged high-speed, 36N and 36 are not, so they never meet.
  Ownership: 36N 28.9 + HSL 2 65.0 + L.36 6.2 = 100.1 km of a 100.6 km ride. Eurostar the same:
  96N 12.6 km and 96 2.1 km were missing.
- **Metro tunnels side by side (ownership right).** Metro 6 credits 1.1 km of Metro 5 and 1.0 of
  Metro 1 today (parallel tunnels near Beekkant); 0 now.
- **Charleroi M2** shares most of its route with M1; today's total adds 16.5 km for a 17.9 km
  ride because `uniqueShares` gives parts of the shared track to neither; ownership gives it to
  M1 and the ride covers it: 17.9. Ownership right.

### Switzerland (today 5,667.5 km, ownership 5,637.5)

| Ride | Adds today | Adds ownership | Ridden |
|---|---|---|---|
| IC 1, all | 445.1 | 366.5 | 366.8 |
| IC 2 Zurich - Lugano | 219.9 | 178.6 | 178.6 |
| IC 6 Bern - Brig | 81.8 | 103.6 | 103.5 |
| S1 Zug - Luzern | 31.2 | 28.4 | 28.3 |
| 140 Visp - Zermatt (register line) | 34.8 | 34.8 | 34.8 |
| EC Basel - Milano, Bern - Brig (named train) | 82.1 | 103.7 | 103.8 |
| Glacier Express, all | 285.5 | 276.6 | 286.5 (276.3 register km) |
| Tram 7 Zurich, all | 13.8 | 12.2 | 12.3 |
| Tram 20 (Limmattalbahn), all | 12.8 | 12.8 | 12.8 |
| Tram 2 Zurich, all | 12.1 | 11.1 | 11.1 |
| S18 Forchbahn, all | 3.3 | 16.3 | 16.3 |
| Tram 15 Geneva, all | 7.0 | 8.8 | 9.0 |
| All 12 together | 1,087.0 | 1,009.9 | |

- **IC 1 (ownership right).** Today credits parallel and junction register lines: 160 Renens -
  Lausanne-Triage 4.8, 151 2.7, 718/708/719 at Zurich 2-3 km each, 450 Olten - Bern 19.0 where
  the new line runs beside it (ownership: 11.1, the stretch IC 1 really shares).
- **IC 2: one tube of the Ceneri base tunnel.** The Swiss register has each tube as its own
  line (580 CBT Est 15.7 km, 581 CBT Ovest 14.7 km). Today one ride completes both; ownership
  credits only 580, the tube the relation uses. Ownership is literally right, but a rider would
  call the tunnel one line: **a decision** (below). Same for about 16 Swiss single-track
  km-lines ("Gleis links", "binario destro"; 604, 605, 601, 607, 608...), about 58 km.
- **IC 6 and EC through the Lotschberg base tunnel.** No register line covers the base tunnel
  in today's ch build (Gotthard has 594/595 GBT; I did not find why Lotschberg is missing:
  possibly its register geometry is over 40 m from OSM's and the junction-ended sections were
  then dropped as unridden). Today the tunnel counts nowhere (IC 6 is under 40% own track);
  ownership gives it to IC 6 by the ref rule, 36.9 km counted. Also: R12 Spiez - Frutigen is
  credited 9-12 km by IC 6 and EC rides now; today 0 because the EC/IC 6 section is flagged
  high-speed and R12's are not. And today's false 8.3 km on 140 Brig - Visp - Zermatt (metre
  gauge beside the standard-gauge line) is gone.
- **S1 Zug - Luzern** credited only 8.7 of the 28.3 km IR 70 shares with it today: a short S1
  section never covers 15% of IR 70's long Luzern - Zug section (`min_frac`). Ownership: 28.3.
  Today's 2.1 km on 653 (parallel) is gone.
- **Glacier Express**: today's 8.3 km of 100 Lausanne - Simplon (SBB beside the MGB) is gone.
- **S18 Forchbahn (known failure type).** A light-rail route on register line 731 (rail):
  today credits nothing there (kind mismatch), adds 3.3 km. Ownership 16.3.
- **Trams**: Tram 7 today credits parallel streets within 45 m (13.8 for 12.3 ridden); Tram 15
  runs on a 2023 extension no register line has (owned by 15, counted now).

### Tokyo slice (today 924.6 km, ownership 931.6)

| Ride | Adds today | Adds ownership | Ridden (in slice) |
|---|---|---|---|
| Marunouchi (OSM line), all | 29.9 | 27.4 | 27.4 |
| Marunouchi (register line), all | 26.6 | 24.3 | 24.3 |
| Hanzomon - Den-en-toshi - Skytree through service | 23.2 | 50.6 | 52.1 |
| Metro through service onto Tobu Tojo | 5.3 | 5.3 | 5.3 |
| Metro through service onto Seibu Ikebukuro | 0.0 | 10.3 | 10.3 |
| Yamanote loop (OSM), all | 44.3 | 34.4 | 34.3 |
| Nozomi Tokyo - Shin-Yokohama | 25.5 | 25.0 | 25.5 |
| Chuo Rapid Tokyo - Mitaka | 26.8 | 23.9 | 24.1 |
| Narita Express Tokyo - Shinjuku | 20.4 | 17.0 | 17.3 |

- **Marunouchi crediting the Oedo (known failure)**: a ride on the OSM Marunouchi line today
  credits 0.40 km of the Oedo, plus Namboku 0.81, Chiyoda 0.51 and Ginza 0.36 (a ride on the
  register line: Oedo 0.41, Chiyoda 0.39, Fukutoshin 0.36, Ginza 0.31). All 0 now.
- **Through services (known failure)**: the Hanzomon through service (a subway route) credits
  Den-en-toshi 19.2 km and Isesaki 14.3 km now, 0 today; today it also falsely credits Ginza
  4.0 and Shinjuku 1.3. The Seibu through service: 0 today, 10.3 km of the Ikebukuro Line now.
- **Yamanote**: today credits 9.1 km of the Tokaido Line beside it and 2.3 of the Seibu
  Shinjuku Line; ownership 5.5 of Tokaido, on Tokyo - Shinagawa where the loop's tracks
  legally are the Tokaido Line, and nothing of the Seibu line. Ownership right.
- Chuo Rapid: today's 1.5 km of the Yamanote near Shinjuku gone. N'EX: today's figures include
  parallel track; ownership's are the tracks it uses.

### Paris slice (today 856.6 km, ownership 902.2)

| Ride | Adds today | Adds ownership | Ridden |
|---|---|---|---|
| Metro 1, all | 16.9 | 16.4 | 16.4 |
| Metro 14, all | 24.7 | 31.2 | 31.2 |
| RER A (in slice) | 72.5 | 69.9 | 70.1 |
| RER B (in slice) | 67.6 | 67.1 | 67.9 |

- **Metro 1 and 14 (known failure)**: today `uniqueShares` treats the two tunnels as one piece
  of track, so Metro 14 counts only 20.0 of its 31.2 km in the total and a full ride on it adds
  24.7. A Metro 14 ride also credits 2.0 km of Line 7 and 1.9 of Line 13. Ownership: 31.2, and
  0 on the others. Every Paris metro line gains 1-3 km in the total for the same reason (today
  merges tunnels within 45 m), which is most of the slice's +45.6 km.
- RER A: today 6.9 km of 340 000 (Saint-Lazare - Le Havre, beside the Cergy/Poissy branch),
  ownership 3.8 (the stretch the relation uses).

### Bordeaux - Dax (TER 52)

The TER 52 ride credits 655 000 Bordeaux - Irun about 108 km in both models in today's build, so
I could not reproduce "credits nothing" there. Under ownership the question does not arise:
a TER's own track is the ways no register line owns, worked out from the ways, not from what
its credits reach, so a high-speed flag cannot turn register track into "own track". What the
slice did show: **the fr build has no register line for Lamothe - Arcachon or the Medoc line**;
ownership gives them to TER 41 / TER 42 (39 km in the slice); today they count nowhere.

### The other known failures

Seoul Lines 1/3/4, PKM Gdansk, Athens M3 and the Austrian Railjets are not in my countries.
All four are the same mechanism as the through services and the Forchbahn above (a route's
kind or speed flag differs from the register line's), and that mechanism is gone: a route
credits whatever owns the ways it uses, whatever its own kind or speed.

## How often the fixed rules are needed

### Shared track no register covers (the "lower line number" rule)

Route-km = the owner sections' owned length that another OSM line also runs over, each stretch
once (double track once). Way length counts both tracks.

| | Route-km | Places | Way length | Notes |
|---|---|---|---|---|
| Belgium | 119.0 | 93 | 242 km | all trams and metros |
| Switzerland | 48.9 | 16 | 77 km | 37 km of it is the Lotschberg base tunnel (IC 6 over IC 8) and 4.2 km Geneva's CEVA (Leman Express L1 over L2-L4, L7, RE33): both look like register gaps. Real non-register sharing about 7 km |
| Tokyo slice | 8.8 | 15 | 47 km | all heavy rail the register test missed (Joban Line near Minami-Kashiwa, Tobu Skytree near Dokkyo-daigakumae, Keisei near Funabashi); real non-register sharing 0 |
| Paris slice | 9.6 | 15 | 26 km | all heavy rail near Villeneuve-Saint-Georges and on the Cergy branch (RER A over Transilien L); the metro and trams share nothing |
| Russia (rough, see caveats) | 2,267 | 1,278 | 3,775 km | two thirds trams (2,580 km of way length: city tram networks); the rest departmental and suburban railways outside the tariff guide ("Poezd No. 1/2/6" near Chernyshevka 83 km, Kerch - Anapa / Feodosiya - Anapa diesel trains 83 km, Sakhalin 6303/6304 39 km) |

Belgian examples (route-km a line gives to a lower-numbered one): Charleroi M2 13.9 (to M1),
Antwerp tram A3 12.6 (to 2 and 7), Brussels tram 93 10.6 (to 51, 62, 8, 92), Brussels Metro 6
10.3 (to 2), Antwerp tram 6 10.1 (to 1, 2), Brussels tram 97 7.9 (to 4, 82, 92), tram 25 7.3 (to
7, 8), Antwerp tram 10 7.3 (to 1, 4, 8), Brussels Metro 5 6.5 (to 1), Antwerp A9 6.0, Brussels
tram 62 5.8, Charleroi M3 4.1. Swiss: Limmattalbahn tram 20 gives 2.7 km to tram 2 at Schlieren;
Appenzeller Bahnen S21/S22 1.8 km to S20 at St. Gallen; Monte Bre funicular sections 1.1 km.

So the rule matters in cities with tram and metro networks outside the register (Belgium;
Russia's city trams at a much larger scale), and on railways outside the register that several
suburban patterns share (Russia's departmental lines), and almost nowhere else. In ch, jp and the Paris slice, where it fires it is nearly
always flagging a register gap rather than deciding a real shared street.

One side effect: under the rule Metro 6 owns 31% of its length, tram 25 33%, Charleroi M2 22%.
Today's `OWN_LINE_SHARE` (40% own track to be listed as a line) would drop them from lists and
operator totals. The listing test should use "share not on register track" (which the rule
does not touch), and only totals use ownership.

### Register lines drawn on the same rails

Two register lines both within 0.5 m of one way along its whole length; the ref rule decides
which a ride credits. Today both count whole in the totals, and so does the prototype (register
sections own themselves whole), so these km are counted twice in both.

| | Way length | Places | Examples |
|---|---|---|---|
| Belgium | 49.0 km | 90 | 161 / 161A near Genval and Rixensart (13 km: RINF's trace put both on the same pair), 50C / Brussels-Midi - Y.Ruisbroek, 21A / 35 at Hasselt |
| Switzerland | 5.4 km | 57 | short junction stretches where km-lines meet (664/665 at Zug, 590/594 at Pollegio, 400/450 at Rothrist) |
| Tokyo slice | 7.3 km | 19 | Seibu Ikebukuro / Toshima at Nerima (1.6), Yamanote / Tokaido at Kita-Shinagawa (1.0), Nambu / Tokaido at Kawasaki (0.9), Tokyu Shin-Yokohama / Toyoko at Hiyoshi (0.8) |
| Paris slice | 0.1 km | 5 | |
| Russia | 626.5 km | 357 | the parallel tariff sections: Adler - Roza Khutor / Imeretinsky Kurort - Roza Khutor 43 km, Sokhranovka - Millerovo / Chertkovo - Millerovo 24, Karymskaya - Borzya / Karymskaya - Kuenga 20, Bakaritsa - Arkhangelsk / Bakaritsa - Karpogory 15 |

Register lines merely near each other (within 8 m, nearest wins, no rule) are far more common:
be 470 km of way length, ch 303, Tokyo 42, Paris 17. That is exactly the track today's 45 m
buffer double-credits.

### Track only named trains run over

Owned by nobody, counted nowhere: be 0.2 km, ch 1.3, Tokyo 1.5, Russia 852 (near Iletsk,
Tsimlyanskaya, Petukhovo: long-distance-only lines or register misses), Paris slice 47.1 (LGV
Interconnexion Est near Mitry and CDG, about 22 km; station approaches at Gare de Lyon,
Montparnasse, Austerlitz). Likely register gaps or ownership misses in the fr build; worth a
look before rollout there.

## Rollout sketch

### build_model.py

- `build()` already has each OSM section's node path (`slice_path` / `trace` ids, used for
  `_digest`). Keep it: node pairs give the ways exactly, no coordinate matching.
- `register_way_lines` becomes `own_ways`: same candidate test, rekind and `route_share`
  unchanged, but it returns ONE owner per way (rules 1-5 above) instead of every line within
  8 m. High-speed becomes a preference. Tram/light rail/subway same-rails rule added.
- New `footprints()`: per section `[[owner gid, lo, hi], ...]`, and per OSM owner section its
  owned spans. Replaces `section_highspeed` and `build_credits`; `--buffer` and `--min-frac`
  go.
- Outputs: **credits.json goes away.** New `owners.json` (or two new keys in lines.json):
  `own` = owned spans of OSM sections that own less than all of themselves, `foot` = footprints
  of sections whose footprint is not just themselves. ways.json can carry the owner first.
- Sizes measured from the prototype: be 0.09 MB against credits.json's 0.26 MB (3,924 entries
  against 14,637 pairs); ch 0.21 MB against 0.61 (9,566 against 33,485); **Russia 3.5 MB against
  23.5 MB** (156,873 entries against 1,365,873 pairs; 95,317 of the entries are named trains').
  Today's pairs are mostly OSM-onto-OSM and named-train-onto-OSM (67% in ru), which a footprint
  never stores. Shipped per line beside geom/ and fetched only for lines with rides, a country's
  first load would carry only the owned spans.
- Build time today for `build_credits` (from data/logs): ch 25 s, be 9 s, jp 26 s, fr 100 s,
  ru 900 s. The prototype's footprint step: be 13 s, ch 108 s, ru 1,100 s (Python, matching
  coordinates). Ownership itself took 40 s for ru. Built on node ids with a per-line tree it
  should come in under today's; its cost grows with route length, not with how many lines
  share a corridor.

### dist/index.html

- `creditSpans`: push `FOOT[gid]` entries for each ridden gid (a register section's footprint
  is itself) instead of `COVERS`. The 150 m gap closing stays.
- Remove `COVERED_BY`, `ownTrack`, `onRegister`, `uniqueShares`, `UNIQ_REGION`, `OP_UNIQ`.
- `stats()`: region and operator totals = register sections whole + OSM owner sections times
  their owned spans; done = ridden spans intersected with owned spans.
- `lineKm`: register lines as now; OSM lines and named trains by footprint share ridden (their
  own route percentage in full, shared streets included).
- `countable`/listing: register always; named trains never; OSM lines listed by share not on
  register track (decision below), counted by owned spans.
- `paintRidden` unchanged.

### Saved rides

Unchanged format (line, from, to, whole, around, regions), unchanged resolution to sections, no
migration. What moves:

- Country totals **fall** where a ride also completed parallel register lines (be IC-01 end to
  end 418 -> 325 km; ch IC 1 445 -> 366; Yamanote 44 -> 34) and **rise** where a route's kind or
  speed flag hid the register line (ICE 79 +34 km, Forchbahn +13, Hanzomon through service
  +27, Seibu through +10, LBT +22 via IC 6). On my test sets: be -15%, ch -7%, Tokyo +17%,
  Paris +2%.
- Register line percentages drop for lines only a neighbour's ride touched (36N after an L.36
  ride: 87% -> 0%) and rise for lines a mismatch hid (36N after the ICE: 0% -> 95%).
- OSM line percentages rise where a local ride covers an express's long section (IR 70 after an
  S1 ride: 31% -> 100% of the shared stretch).
- Denominators: be +0.8% (parallel tram streets and tunnels no longer merged), ch -0.5% (the
  81 km of straight chords leave, the Lotschberg tunnel and tram extensions arrive), Paris
  slice +5% (metro tunnels).

## Decisions for Anita

1. **Register lines drawn on the same rails**: keep counting both whole in totals (as today;
   Russia's parallel tariff sections then count twice), or give the stretch to one line and
   shorten the other's counted length (strict one owner)? Either way a ride credits one of them,
   by the ref rule.
2. **Single-track register lines of one route** (Swiss CBT Est/Ovest, "Gleis links" km-lines,
   ~58 km in ch): one ride credits only the tube or track it used. Fine, or merge such pairs
   into one owner?
3. **Listing threshold** (`OWN_LINE_SHARE`): judge "is this a line" by share not on register
   track, so the ref rule cannot drop Metro 6 or tram 25 from the lists?
4. **Register gaps the model turns into counted OSM track**: Lotschberg base tunnel (37 km, ch),
   Geneva CEVA (4 km), fr's Arcachon and Medoc branches, the Paris LGV approaches ("svc", counted
   nowhere). Count them through the OSM line meanwhile (as built), or hold rollout per country
   until the register is fixed?
5. **Ref rule details**: natural sort, empty refs last, then name, then line id. OK as the
   deterministic tie-break? (It means Brussels Metro 2 owns the 2/6 trunk, M1 owns Charleroi's.)
6. **Four-track corridors** (Belgian 25/27, 36/36N, 161/161A): a ride credits the pair its OSM
   relation is mapped on. Acceptable, given nothing else says which pair a train used?
7. **Station throats** go to the nearest register line (the pass-2 rule). OK?

## Caveats

- Distances use one flat projection per run, fine for ch, be and the slices; for Russia it
  distorts by up to about 30% at the edges, so the Russia run is for sizes and counts only.
- In slices, routes leaving the box lose their outside part ("unmatched" in the report).
- Pieces of a footprint whose owner line has no section within 160 m are dropped ("far"): be
  14 km, ch 23 km of section length, mostly at borders and big stations.
