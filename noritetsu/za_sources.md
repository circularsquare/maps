# South Africa register sources (built 2026-10-03)

What `za_register.py` reads, what runs and what does not, the line calls and why, and how the
build checks out. Nothing here needed a login, a key or an account. Downloads: none beyond the
OSM extract (data/proc/za); the reader's own output is in data/raw/rinf/za/.

## The short answer

- **No open line register exists**, and OSM names only 22% of South Africa's main-line track
  (`python probe_kr_ways.py --region za`), so Korea's named-track recipe does not work. OSM does
  have 190 infrastructure relations naming the lines by their ends ("Cape Town–De Aar",
  "Salt River–Simon's Town", "Kaalfontein–Leralla") and complete Metrorail and Gautrain route
  relations with stops.
- **So the register is a hand-written list** (`LINES` in za_register.py), each line its ends and
  enough stations between to pin the path, written into rinf.py's input files and traced over
  OSM track by rinf.py, as balkans_register.py does. Each trace prefers the ways of the OSM
  infrastructure relation(s) the line names (`own`; most traces lie 77-100% on them).
  Stops: every OSM station a passenger route stops at that lies on a traced section
  (rinf.py's `osm_stops`), so Metrorail's and Shosholoza Meyl's own stop lists decide them.
- **53 register lines, 4,701 km**: 33 running (2,237 km), 20 greyed as not running (2,464 km).
  532 stations. 45 OSM lines on top (Metrorail's services, Gautrain's three), 17 of them named
  trains.
- **Path checks against published route lengths: 0.993 to 1.017** (below).
- **No GTFS** covers any of it (below), and **no passenger train crosses a border**.

## The line unit

Track pieces, cut so every piece of passenger track is on exactly one line and every line is
wholly running or wholly not (not_running.py greys a whole line or nothing). Names are the OSM
infrastructure relation's where the passenger part is the whole of it ("Salt River–Simon's
Town", "Kraaifontein–Malmesbury", "Pinelands–Bellville"), else its ends ("Eerste River–Du Toit"
running and "Du Toit–Muldersvlei" not, both on OSM's "Eerste River–Muldersvlei"). Place names
are the current official ones OSM uses: KuGompo City (East London), Gqeberha, Kariega
(Uitenhage), uMnambithi (Ladysmith), Komani (Queenstown), Ntabozuko (Berlin), KwaDukuza
(Stanger); the old name is the line's English name where a line is named after one.

Metrorail's own lines (Southern Line, Northern Line, the Gauteng colour lines, Durban's
KwaMashu Line...) are services over several of these pieces and stay OSM lines, as JR's
services do over Japan's legal lines. Their track counts through the register lines. They are
not register lines' English names either: two pieces of one service would show one name.

**Gautrain stays OSM lines** (North-South, Airport, East-West; Gautrain's own route relations):
its Park station's OSM record sits on Metrorail's platforms, 120 m off Gautrain's track, so
rinf.py cannot trace from it without a new hook, and the routes are complete. OSM measures the
North-South line 61.9 km and the Airport line 19.8 (Sandton - O.R. Tambo); WP gives the
system as 80 km (Park - Hatfield plus Marlboro - O.R. Tambo, about 76 here).

## What runs (checked 2026-10-03) and what the build does with it

Main sources: Parliament's Portfolio Committee on Transport report of 10 June 2026 (ATC no.
102, PRASA's Gauteng corridor table: trips, frequencies and "No service" per corridor;
`parliament.gov.za/storage/app/media/Docs/atc/01ls62wgdnvbh5n2rt6vejrbpyvl7vngka.pdf`, pages
22-24), GroundUp (1 Sept 2026 on Durban, 23 March 2026 on Shosholoza Meyl), Logistics Business
Africa (5 June 2026, Johannesburg - Lenasia), cttrains.co.za and nexttrain.co.za (current Cape
Town and national timetables), seat61.com.

### Metrorail Western Cape: everything runs but Du Toit - Muldersvlei

| line | status | in the build |
|---|---|---|
| Cape Town–Bellville (via Mutual) | Northern Line via Mutual | running |
| Cape Town–De Aar | Northern Line via Century City, Wellington, Malmesbury trains; Worcester one train each way on weekdays; the Blue Train and Rovos Rail beyond | running |
| Salt River–Simon's Town | Southern Line; Fish Hoek - Simon's Town weekdays and a Saturday shuttle | running |
| Maitland–Heathfield | Cape Flats Line (its trains run on to Retreat over the Southern Line's track) | running |
| Ysterplaat–Langa, Pinelands–Bellville, Bonteheuwel–Kapteinsklip, Philippi–Chris Hani | Central Line, reopened in stages to May 2025; Kapteinsklip trains run through to Cape Town (2026 timetable) | running |
| Bellville–Strand | Northern Line to Strand | running |
| Eerste River–Du Toit | Stellenbosch trains, which turn at Du Toit | running |
| Du Toit–Muldersvlei | Koelenhof closed since 2020; no train | greyed |
| Kraaifontein–Malmesbury | limited peak service | running |

### Metrorail Gauteng: PRASA's corridor table

Running in PRASA's table: Mabopane - Pretoria, Saulsville - Pretoria, Pienaarspoort -
Pretoria, Germiston - Leralla, Germiston - Johannesburg, Naledi - Johannesburg, Pretoria -
Kempton Park, De Wildt - Pretoria, Mabopane - Belle Ombre, Hercules - Koedoespoort,
Johannesburg - Midway (to Lenasia since June 2026), Germiston - Kwesine, Johannesburg -
Randfontein. "Vereeniging - Union: No service". The same report: "no train services on the
Daveyton, Springs, Nigel, Oberholzer and Jikeleza lines, including Midway to Vereeniging".

| line | in the build |
|---|---|
| Pretoria–Saulsville, Pretoria–De Wildt, Winternest–Mabopane, Belle Ombre–Technikon Rant, Hercules–Koedoespoort, Pretoria–Pienaarspoort, Germiston–Pretoria, Kaalfontein–Leralla, Germiston–Elsburg, Elsburg–Kwesine, Langlaagte–Lenasia, New Canada–Naledi | running |
| Germiston–Kimberley (Germiston - Johannesburg - Krugersdorp - Randfontein - Klerksdorp - Kimberley) | running: Metrorail to Randfontein, the Blue Train and Rovos Rail beyond |
| Lenasia–Vereeniging (Lawley - Vereeniging closed; Houtheuwel expected June 2027), Midway–Bank (Oberholzer line), Elsburg–Vereeniging (via Kliprivier), President–Elsburg (Kwesine trains run via Kutalo: PRASA's 15.5 km), Germiston–Springs, Dunswart–Daveyton, Springs–Nigel, and the Jikeleza loop: Germiston–New Canada (via Booysens), George Goch–Kaserne West, Booysens–Faraday, Crown–Westgate | greyed |

### Metrorail KwaZulu-Natal: GroundUp, 1 September 2026

KwaMashu - Durban - Umlazi runs, 17 return trips a day; the South Coast line to Winklespruit
(the rest waits on the Illovo bridge); Chatsworth (Merebank - Crossmoor) and the Old Main
(Rossburgh - Pinetown) on one track each; "the Bluff line remains closed"; "only sections of
the Northern Coast and the KwaMashu lines are operating" (the North Coast proper, Duff's Road -
KwaDukuza, has had no train since the 2022 floods; EWN March 2025). nexttrain.co.za lists
Berea Road - Bridge City, Durban - Cato Ridge, - Crossmoor, - Pinetown, - Umlazi, -
Winklespruit.

| line | in the build |
|---|---|
| Durban–Cato Ridge (New Main Line), Rossburgh–Pinetown (Old Main), Rossburgh–Winklespruit (South Coast), Reunion–Umlazi, Merebank–Crossmoor (Chatsworth), Durban–Bridge City (KwaMashu Line) | running |
| Winklespruit–Kelso, Clairwood–Wests (Bluff), Umgeni–KwaDukuza (North Coast) | greyed |

### Metrorail Eastern Cape

| line | in the build |
|---|---|
| KuGompo City–Ntabozuko (East London - Berlin) | running (nexttrain's East London - Berlin; PRASA's 2026 report has "rolling stock currently in use" on the East London and Gqeberha corridors) |
| Gqeberha–Kariega (Port Elizabeth - Uitenhage) | running: one train each way on weekdays since 16 Oct 2023 (SAnews); no later suspension found |

### Long distance

| service | status | in the build |
|---|---|---|
| The Blue Train, Pretoria - Cape Town | about weekly each way in 2026 (three departures a month most months, published dates) | named train; its track counts: Germiston–Pretoria, Germiston–Kimberley, Kimberley–De Aar, Cape Town–De Aar are running register lines |
| Rovos Rail Pretoria - Cape Town (and rarer cruises to Durban, Victoria Falls, Dar es Salaam) | a few a month | named train (no OSM route) |
| Shosholoza Meyl: Johannesburg - Cape Town, - Durban, - East London, - Musina | suspended since October 2024 (GroundUp 23 March 2026: "no routes are currently operating"); PRASA plans their return in 2027 | named trains (OSM routes); the Durban, East London and Musina track greyed: Union–uMnambithi, uMnambithi–Cato Ridge, Elsburg–Vereeniging, Vereeniging–Springfontein, Springfontein–Ntabozuko, Pretoria-Noord–Musina |
| Shosholoza Meyl Johannesburg - Gqeberha and - Komatipoort | not run since 2020 | not built (their OSM routes are named trains over no register line) |
| Premier Classe | not run since 2020 (seat61) | nothing |
| Kei Rail, Mthatha - Amabele - East London | no service found since the late 2010s | not built; OSM's route a named train (stopgap) |
| Tourist trains (Ceres Rail, Apple Express, Hexpas Express, Umgeni Steam, Sandstone) | excursions, weekly or rarer | not counted (named trains where OSM has a route) |

**The Blue Train decision**: it is a luxury cruise train, but it is scheduled with published
dates about once a week each way, with Rovos Rail on the same track a few times a month, so the
track gets more than about one train a week, Anita's rule. As with Australia's Ghan, the
trains are named trains and their track counts through the register lines (1,588 km Pretoria
- Cape Town). If that is wrong, set `suspended=True` on "Kimberley–De Aar" and split
"Cape Town–De Aar" at Worcester and "Germiston–Kimberley" at Randfontein.

**Borders**: no passenger train crosses. CFM's Maputo trains turn at Ressano Garcia on the
Mozambican side of Komatipoort; no scheduled train crosses at Beitbridge, Mahikeng, Maseru
or Golela (only Rovos Rail's occasional cruises to Victoria Falls and Dar es Salaam). No
`borders.EXTRA` point.

## Timetable feeds

None. Transitous has no South African feed; the Mobility Database lists two (Stellenbosch
minibus taxis, Algoa Bus); Gautrain and Metrorail publish no GTFS (gtfs-data-exchange's last
Metrorail file is from 2011). No `gtfs_served.FEEDS["za"]`.

## Checks

The section lengths are the reader's own traces, so there is no chainage check (`no_chain`).
Outside numbers, from the build log's `path check` lines (shortest path over the named register
lines) and `check_model.REGISTER["za"]`:

| path | built | published | ratio |
|---|---|---|---|
| Cape Town - Worcester | 175.4 | 174 (Metrorail WC, WP) | 1.008 |
| Cape Town - Malmesbury | 79.3 | 79.4 (WP Malmesbury Line) | 0.999 |
| Cape Town - Simon's Town | 36.0 | 36 (WP Southern Line) | 1.000 |
| Cape Town - Heathfield (Cape Flats) | 22.2 | 22.2 (WP 23.8 to Retreat less 1.6) | 1.002 |
| Pretoria - Cape Town | 1,588 | 1,600 (the Blue Train, WP) | 0.993 |
| Pretoria - Polokwane | 289.3 | 284.4 (Pretoria - Pietersburg, SAR 1902 length) | 1.017 |

Shosholoza Meyl's OSM route Johannesburg - Cape Town measures 1,559.7 km over the same track.

Fixes in the reader, with what they caught:
- Two OSM records of one station (Cleveland/Clevenland, Medunsa/Mudunsa, Eersterus/Eersterust,
  Ellis Park/Ellis Park Station, two Deerness, two Oosterzee, Metrorail's and Gautrain's
  Rhodesfield) gave sections of 16-126 m: `merge_twins` makes them one station.
- The Kimberley line leaves the Mafikeng line short of Fourteen Streams (OSM: Veertien Strome),
  which no passenger train calls at, so the main line runs Germiston–Kimberley and
  Kimberley–De Aar rather than through Fourteen Streams.
- The Natal main line shares Germiston - Elsburg - Union with the Kwesine and Vereeniging
  lines, so its greyed piece starts at Union; Belle Ombre's trains reach Hercules over the De
  Wildt line from Technikon Rant.

## Known faults

- **OSM services over greyed track count as running there.** ownership.py gives a closed
  register section's ways to whatever OSM line runs over them. Where a service runs today in
  part, its OSM route still goes all the way, so: the South Coast Line over Winklespruit -
  Kelso (33.4 km of way length), the Northern Line over Du Toit - Muldersvlei (13.3), the Red
  Line via President over President - Elsburg (4.6). Wholly stale services are flagged named
  trains in rules/za.py (stopgap, as Mexico's Línea Z), so they own nothing. Fix proposed to
  the managing session: closed register sections keep their ways from OSM lines.
- Stations two lines share at a junction overlap a few hundred metres where one line's trace
  runs into the other's station (Midway - Bank 1.3 km into Bank, Pinelands - Bellville 2.9 km
  over the Strand line near Bellville); ownership gives each to one line.
- Tourist funiculars (Cape Point) have no OSM route and are not built.

## Commands

    python za_register.py --dry            # traced km per line, stations passed, shared track
    python za_register.py --convert        # writes data/raw/rinf/za/ (build_model needs it to exist)
    python build_model.py --region za --register za_register:data/raw/rinf/za    # ~70 s
    python build_tiles.py --region za                                            # ~20 s, 2.9 MB
    python check_model.py --region za

## Watch for

The Metrorail recovery (PRASA restores corridors every few months: Springs, Daveyton, Lenasia -
Houtheuwel by June 2027, Elsburg - Vereeniging "this financial year", Durban's North Coast after
the Transnet agreement, Winklespruit - Kelso after the Illovo bridge): set `suspended=False` in
`LINES`, and drop the service from `NAMED_NAME` in rules/za.py. Shosholoza Meyl's 2027 return.
