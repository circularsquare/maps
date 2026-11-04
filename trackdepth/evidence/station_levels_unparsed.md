# Machines not placed in a chain

Written by `station_levels.py`. Each machine below has no usable rise, or could not be
joined to a chain that reaches the street. A later pass (more rises, a layout drawing, a
barometer ride) could pick these up.

## Rises that needed reading or could not be read

| machine | station | raw | what was done |
|---|---|---|---|
| EL219 | 14 St-Union Sq | `10in- 2in` | reinterpreted '10in- 2in': '10in' is a typo for 10 ft (sibling EL220 reads 10' - 2in) |
| EL302 | Atlantic Av-Barclays Ctr | `19'=10in` | reinterpreted '19'=10in': '=' read as '-' |
| EL343 | Euclid Av | `10in - 2 3/8in` | reinterpreted '10in - 2 3/8in': '10in' is a typo for 10 ft (mezzanine-to-platform elevator) |
| EL428 | Queens Plaza | `16'-4in 3/4in` | reinterpreted '16'-4in 3/4in': read 16' 4 3/4in |
| EL514 | Westchester Sq-East Tremont Av | `14.8` | reinterpreted '14.8': no unit; read as 14.8 ft (Pelham Bay Park's similar elevator is 14'8in) |
| ES108 | Intervale Av | `30'- 5/5in` | reinterpreted '30'- 5/5in': 5/5 in is garbled; took 30' 0.5in (at most 1 in off) |
| ES202 | 96 St | `5137mm (16'  6 1/2in)` | mm and feet-inches disagree (5137 mm vs 16' 6 1/2in = 5042 mm); used mm |
| ES213 | 59 St | `38-11 5/8in` | reinterpreted '38-11 5/8in': missing foot mark; read 38' 11 5/8in (sibling ES214 is 39') |
| ES243 | Lexington Av/53 St | `56' - 2/25in` | reinterpreted '56' - 2/25in': read 2/25 as 2.25 in, as its siblings 2.025 / 21.4 / 21-4 at the same bank |
| ES244 | Lexington Av/53 St | `56' 21-4in` | reinterpreted '56' 21-4in': read 21-4 as 2 1/4 in |
| ES245 | Lexington Av/53 St | `56' - 2.025in` | reinterpreted '56' - 2.025in': read 2.025 as 2.25 in (same bank reads 2/25, 21.4, 21-4) |
| ES246 | Lexington Av/53 St | `56' - 21.4in` | reinterpreted '56' - 21.4in': 21.4 in is impossible; read as 2 1/4 in (siblings 2/25, 2.025, 21-4); 2.14 in would differ by 0.1 in |
| ES285 | 72 St | `4953 mm (16' 1/4in)` | mm and feet-inches disagree (4953 mm vs 16' 1/4in = 4883 mm); used mm |
| ES286 | 72 St | `4953 mm (16' 1/4in)` | mm and feet-inches disagree (4953 mm vs 16' 1/4in = 4883 mm); used mm |
| ES287 | 72 St | `4953 mm (16' 1/4in)` | mm and feet-inches disagree (4953 mm vs 16' 1/4in = 4883 mm); used mm |
| ES321 | High St | `39' 11/5in` | reinterpreted '39' 11/5in': 11/5 read as 11.5 in: the same bank's ES350 reads 39' 11 1/2in |
| ES322 | High St | `39' 11/5in` | reinterpreted '39' 11/5in': 11/5 read as 11.5 in: the same bank's ES350 reads 39' 11 1/2in |
| ES350 | High St | `40'39' 11 1/2in` | reinterpreted '40'39' 11 1/2in': two values run together; took the precise one, 39' 11 1/2in |
| ES375 | South Ferry | `7408 mm (24' 6in)` | mm and feet-inches disagree (7408 mm vs 24' 6in = 7468 mm); used mm |
| ES405 | Lexington Av/63 St | `25'11 3/4` | reinterpreted '25'11 3/4': no inch mark; read 25' 11 3/4in |
| ES407 | Lexington Av/63 St | `18'10 5/8` | reinterpreted '18'10 5/8': no inch mark; read 18' 10 5/8in |
| ES636 | 34 St-Hudson Yards | `83.3` | reinterpreted '83.3': no unit; read as 83.3 ft (the parallel bank at this station is 81'8in) |

## Coordinates replaced by the station's (more than 400 m from it)

| machine | station | listed lat, lon | distance |
|---|---|---|---|
| ES316 | Smith-9 Sts | 40.71271, -74.00486 | 4.4 km |
| EL100 | 149 St-Hostos | 40.69082, -73.98929 | 15.1 km |
| ES314 | Smith-9 Sts | 40.692024, -73.98709 | 2.2 km |

## Paths that disagree (misfit over 2.5 sd)

- complex 238: ES349 (upper mezzanine to intermediate mezzanine (9 st&smith st)) off by +1.14 m (sd widened from 0.14 to 0.57); ES315 (upper mezzanine to intermediate mezzanine (9 st&smith st)) off by -0.08 m (sd widened from 0.15 to 0.21); ES316 (upper mezzanine to intermediate mezzanine (9 st&smith st)) off by -0.08 m (sd widened from 0.15 to 0.21)
- complex 306: street pin ES103 off by +0.07 m (sd widened from 0.58 to 0.72); street pin ES102 off by -4.72 m (sd widened from 0.55 to 2.36); street pin ES101 off by +1.11 m (sd widened from 0.66 to 1.24)
- complex 421: street pin ES104 off by +1.92 m (sd widened from 0.51 to 0.96); street pin ES105 off by -1.36 m (sd widened from 0.51 to 0.81)
- complex 607: ES231 (mezzanine (N/Q/R/W) to downtown B/D/F/M platform) off by -0.60 m (sd widened from 0.15 to 0.30); ES232 (mezzanine to downtown B/D/F/M platform) off by -0.42 m (sd widened from 0.15 to 0.21); ES236 (mezzanine to downtown B/D/F/M platform) off by +0.19 m (sd widened from 0.14 to 0.16); ES229 (mezzanine (N/Q/R/W) to uptown B/D/F/M platform) off by -0.32 m (sd widened from 0.15 to 0.19); EL210 (mezzanine (N/Q/R/W) to uptown B/D/F/M platform) off by +0.52 m (sd widened from 0.14 to 0.26); EL209 (mezzanine (N/Q/R/W) to downtown B/D/F/M platform) off by +0.23 m (sd widened from 0.14 to 0.17)
- complex 611: ES218 (upper mezzanine (41 st&7 av) to 7 platform) off by -0.51 m (sd widened from 0.16 to 0.26)
- complex 635: street pin ES373 off by +2.15 m (sd widened from 0.51 to 1.07)

## Machines with a rise that were not placed

| complex | station | machine | rise | notes | why |
|---|---|---|---|---|---|
| 58 | Coney Island-Stillwell Av | EL701 | `27'-9in` | N platform to overpass for access to D platform. | chain never reaches the street and has no plain mezzanine to hang a default on |
| 58 | Coney Island-Stillwell Av | EL702 | `16' - 3in` | D platform to overpass for access to mezzanine via N platform | chain never reaches the street and has no plain mezzanine to hang a default on |
| 103 | Bowery | ES334 | `23' 8 9/32in` |  | Notes don't say 'A to B' |
| 135 | Livonia Av | EL772 | `17’-3 ¼”` | Passageway to Manhattan-bound platform | chain never reaches the street and has no plain mezzanine to hang a default on |
| 164 | 34 St-Penn Station | EL227 | `12' 6in` | Penn Station concourse to A platform for service in both directions | chain never reaches the street and has no plain mezzanine to hang a default on |
| 345 | Crown Hts-Utica Av | EL323 | `26' - 2 3/4in` | mezzanine (eastern entrance near Utica Ave) to platforms for service in both directions | left out: Crown Hts-Utica Av: 'mezzanine to platforms for service in both directions' serves both levels of a two-level station; can't tell which the rise is to |
| 359 | Flatbush Av-Brooklyn College | EL319 | `51' 4in` | Nostrand Ave & Flatbush Ave (SE corner) to platform for service in both directions | left out: Flatbush Av-Brooklyn College: 51'4" street-to-platform is the exact rise of Grand Central's EL244 and implausible for this shallow terminal (Church Av next door: 17'7"); probably a copy error |
| 471 | 34 St-Hudson Yards | ES633 | `28'` | upper mezzanine (Hudson Blvd E & 35 St) to lower mezzanine | left out: listed as 35 St upper mezzanine to lower mezzanine at 28 ft, but ES636 joins the same levels at 83.3 ft and the main bank does 81'8"; must serve an intermediate landing |
| 471 | 34 St-Hudson Yards | ES634 | `28'` | upper mezzanine (Hudson Blvd E & 35 St) to lower mezzanine | left out: as ES633 |
| 471 | 34 St-Hudson Yards | ES635 | `24'` | lower mezzanine to upper mezzanine for Hudson Blvd E & 35 St exit | left out: as ES633 (24 ft) |
| 477 | 72 St | ES290 | `20828 mm (68'4in)` | 69 St & 2nd Ave (NE corner) to upper mezzanine | left out: 72 St (Q): street-to-upper-mezzanine escalator carries 68'4", the rise of the lower-to-upper escalators beside it (copy error; the 70 St entrance does the same step in 12'5") |
| 602 | 14 St-Union Sq | EL217 | `12'-6 1/4in` |  | left out: 14 St-Union Sq EL217: no Notes, and its coordinates are near 59 St-Columbus Circle |
| 603 | 149 St-Hostos | EL101 | `28' 2in` | Transfer mezzanine to Woodlawn-bound 4 platform and Bronx-bound 2/5 platform. | left out: 149 St-Grand Concourse: elevator serves the 4 platform and the 2/5 platform, which are on different levels; can't tell which landing the rise is to |
| 603 | 149 St-Hostos | EL102 | `28'-4 1/2in` | Transfer mezzanine to Manhattan-bound 4 platform and Manhattan-bound 2/5 platform. | left out: as EL101 |
| 611 | Times Sq-42 St | EL229 | `16' - 1in` |  | Notes don't say 'A to B' |
| 614 | 59 St-Columbus Circle | EL622 | `17'-2 5/8in` | Mezzanine to downtown A/B/C/D & 1 platforms | left out: 59 St-Columbus Circle: 'mezzanine to downtown A/B/C/D & 1 platforms' serves two platforms on different levels; can't tell which the rise is to |
| 616 | 74 St-Broadway | EL420 | `39' - 1in` |  | left out: 74 St-Broadway: no Notes, 3 landings (street, mezzanine, platform?); can't tell its ends |
| 616 | Jackson Hts-Roosevelt Av | EL421 | `19'-3 1/4in` | transfer mezzanine to Manhattan-bound E/F/M/R platform and mezzanine for access to Queens-bound E/F/M/R service | left out: Jackson Hts: 3-landing elevator (transfer mezzanine, Manhattan-bound platform, mezzanine for Queens-bound); can't tell which two landings 19'3" spans, and reading it as mezzanine-to-platform puts the two E/F/M/R platforms 7 m apart |
| 616 | Jackson Hts-Roosevelt Av | EL422 | `10'-7 1/8in` | mezzanine to Forest Hills and Jamaica-bound E/F/M/R platform | left out: Jackson Hts: 'mezzanine to Forest Hills-bound platform', but which mezzanine is unclear (its coordinates are at Sutphin Blvd) and it doesn't join a chain from the street |
| 619 | B'way-Lafayette St | ES300 | `23' 7 3/4in` | uptown B/D/F/M platform to uptown 6 platform and access to street | chain never reaches the street and has no plain mezzanine to hang a default on |
| 624 | World Trade Center | EL326 | `13'2in` | Park Pl & Church St (SE corner to E Line access to downtown R/W | left out: World Trade Center EL326: Notes garbled ('Park Pl & Church St (SE corner to E Line access to downtown R/W'); can't tell the lower end |
| 628 | Fulton St | ES364 | `20' 2 1/5in` | lower concourse to downtown 4/5 platform and access to street | chain never reaches the street and has no plain mezzanine to hang a default on |

## Machines with no rise in the inventory (390)

They still say what connects to what; listed only where they would complete a chain that
has rises elsewhere in the same complex.

- 57 St-7 Av EL281: 57 St & 7th Ave (SW corner) to mezzanine for service in both directions
- 59 St EL389: mezzanine to platform for Coney Island-bound N and Bay Ridge-bound R service
- Coney Island-Stillwell Av EL399: 3 Landings
- Livonia Av EL771: Street inside station house to passageway to Canarsie-bound platform
- 181 St EL517: Mezzanine to platform for Inwood-207 St-bound service.
- 181 St EL518: Mezzanine to platform for Downtown and Brooklyn-bound service.
- 125 St EL142: 125 St & St Nicholas Ave (SW corner) to mezzanine for service in both directions
- 34 St-Penn Station EL228: Penn Station concourse to downtown C/E platform
- 21 St-Queensbridge EL405: 21 St & 41 Ave (NE corner) to mezzanine for service in both directions
- 21 St-Queensbridge EL406: mezzanine to Manhattan-bound platform
- 21 St-Queensbridge EL407: mezzanine to Jamaica-bound platform
- Roosevelt Island EL403: West Rd to Manhattan-bound platform
- Roosevelt Island EL404: West Rd to Jamaica-bound platform
- Lexington Av/63 St EL401: 63 St & Lexington Av (NW corner) to mezzanine for service in both directions
- Lexington Av/63 St EL402: 63 St & Lexington Av mezzanine to platforms for service in both directions.
- Lexington Av/63 St EL440: 63 St & 3 Av mezzanine to platforms for service in both directions.
- Lexington Av/63 St EL441: 63 St & 3 Av mezzanine to platforms for service in both directions.
- Lexington Av/63 St EL442: 63 St & 3 Av mezzanine to platforms for service in both directions.
- Lexington Av/63 St EL443: 63 St & 3 Av mezzanine to platforms for service in both directions.
- Lexington Av/63 St EL444: 3rd Ave & 63 St (NW corner) to mezzanine for service in both directions
- Northern Blvd EL485: Street (SE corner of 54 St and Broadway) to Forest Hills-bound platform
- Jamaica Center-Parsons/Archer EL412: Archer Ave & Parsons Blvd (SW corner) to mezzanine for service in both directions on both lines
- Jamaica Center-Parsons/Archer EL413: mezzanine to platforms for E/J/Z service in both directions
- Sutphin Blvd-Archer Av-JFK Airport EL411: mezzanine to platforms for E/J/Z service in both directions
- Sutphin Blvd-Archer Av-JFK Airport EL448X: Sutphin Blvd & Archer Ave (SW intersection) to mezzanine for service in both directions
- Sutphin Blvd-Archer Av-JFK Airport EL449X: Sutphin Blvd & Archer Ave (SW intersection) to mezzanine for service in both directions
- Sutphin Blvd-Archer Av-JFK Airport EL450X: Sutphin Blvd & Archer Ave (SW intersection) to mezzanine for service in both directions
- Sutphin Blvd-Archer Av-JFK Airport EL454: Yard To Storage Room
- Sutphin Blvd-Archer Av-JFK Airport ES432X: LIRR and AirTrain Terminal (SW corner of Sutphin Blvd & Archer Ave) to mezzanine for E/J/Z service
- Sutphin Blvd-Archer Av-JFK Airport ES433X: mezzanine to Sutphin Blvd & Archer Ave (SW corner) for connection to LIRR and AirTrain
- Jamaica Van Wyck EL408: mezzanine to platform for service in both directions
- Jamaica Van Wyck EL409: 89 Ave & Van Wyck Expwy (SW corner) to mezzanine for service in both directions
- Greenpoint Av EL764: Mezzanine to Court Sq-bound platform
- Greenpoint Av EL765: Mezzanine to Church Av-bound platform
- 137 St-City College EL166X: 3 Landings
- 137 St-City College EL167X: 3 Landings
- 137 St-City College EL169X: 3 Landings
- 34 St-Penn Station EL214: concourse level of Penn Station to platform for uptown 1 service
- 34 St-Penn Station EL215: concourse level of Penn Station to platform for 2/3 service in both directions.
- 34 St-Penn Station EL34X: LIRR entrance - south side of 34 St (west of 7th Ave) to Penn Station concourse for service in both directions on all lines
- Cortlandt St EL11X: West Concourse balcony to West Concourse with access to 1 service in both directions
- Cortlandt St EL12X: West Concourse balcony to PATH level with access to 1 train service in both directions
- Cortlandt St EL13X: Oculus main level  to PATH level with access to 1 train service in both directions
- Cortlandt St EL14X: Greenwich St to Oculus Balcony with access to uptown 1 and downtown R/W service
- Cortlandt St ES353: north mezzanine to platform for uptown 1 service
- Church Av EL320: Nostrand Ave & Church Ave (SE corner) to Manhattan-bound platform
- Westchester Sq-East Tremont Av EL164: Passanger Yard Elevator At Phelham Inspection Barn
- Westchester Sq-East Tremont Av EL174: Passenger Yard Elevator At Westchester Yard
- Westchester Sq-East Tremont Av EL512: Street (NW corner of Westchester Av and Lane Av) to mezzanine.
- Westchester Sq-East Tremont Av EL513: Mezzanine to Pelham Bay Park-bound platform.
- Bowling Green EL710: plaza south of Bowling Green Park to uptown platform and lower mezzanine for Brooklyn-bound service
- Bowling Green EL711: lower mezzanine to  Brooklyn-bound platform
- Gun Hill Rd EL182: Gun Hill Rd to Wakefield 241 St-bound platform
- Gun Hill Rd EL183: Gun Hill Rd to Manhattan-bound platform
- Pelham Pkwy EL136: White Plains Rd & Pelham Pkwy South (SW corner) to mezzanine for service in both directions
- Pelham Pkwy EL137: mezzanine to Manhattan-bound platform
- Pelham Pkwy EL138: mezzanine to Wakefield 241 St-bound platform
- Flushing-Main St EL414: East of Main St & Roosevelt Ave to Manhattan-bound platform.
- 61 St-Woodside EL415X: Roosevelt Ave & 61 St (NE corner) to mezzanine for LIRR and 7 service in both directions
- 61 St-Woodside EL416X: mezzanine to Manhattan-bound 7 platform
- 61 St-Woodside EL417X: mezzanine to Flushing Main St-bound 7 platform
- 61 St-Woodside EL418X: mezzanine to Penn Station-bound LIRR platform
- 61 St-Woodside EL419X: mezzanine to Long Island-bound LIRR platform
- 34 St-Hudson Yards EL600: mezzanine to Flushing Main St-bound platform
- 34 St-Hudson Yards EL601: upper mezzanine to lower mezzanine for Flushing Main St-bound service
- 34 St-Hudson Yards EL602: upper mezzanine to lower mezzanine for Flushing Main St-bound service
- 34 St-Hudson Yards EL603: 34th St (between 10th Ave & 11th Ave) to upper mezzanine for Flushing Main St-bound service
- 96 St EL247: mezzanine to downtown platform
- 96 St EL248: 95 St & 2nd Ave entrance (SW corner) to mezzanine for downtown service
- 86 St EL249: mezzanine to platform for service in both directions
- 86 St EL250: 86 St & 2nd Ave (SW corner)  to mezzanine for service in both directions
- 72 St EL259: mezzanine to platform for service in both directions
- 72 St EL262: 2nd Ave & 72 St (SE corner) to mezzanine for service in both directions
- 72 St EL264: 2nd Ave & 72 St (SE corner) to mezzanine for service in both directions
- 72 St EL265: 2nd Ave & 72 St (SE corner) to mezzanine for service in both directions
- 72 St EL266: 2nd Ave & 72 St (SE corner) to mezzanine for service in both directions
- 72 St EL267: 2nd Ave & 72 St (SE corner) to mezzanine for service in both directions
- 72 St EL269: Lower Mezzanine-first Upper Mezzanine-second Upper Mezzanine
- 14 St-Union Sq ES219: L platform to mezzanine for access to street and 4/5/6 & N/Q/R/W service
- 14 St-Union Sq ES257X: 4/5/6 mezzanine to 14 St & 4th Ave (NE corner)
- 14 St-Union Sq ES258X: 14 St & 4th Ave (NE corner) to mezzanine for 4/5/6/L service and access to N/Q/R/W
- 168 St EL139: 168 St & St Nicholas Ave (SE corner) to mezzanine for A/C service in both directions and access to non-ADA 1 line elevators
- 168 St EL140: mezzanine to platform for uptown A service and C terminal
- 168 St EL141: mezzanine to downtown A/C platform
- 34 St-Herald Sq EL200X: 6th Ave & 32 St (NW corner) to mezzanine for PATH service and access to subway service via stairs or non-ADA ramp
- 34 St-Herald Sq EL212: mezzanine to uptown N/Q/R/W platform
- 34 St-Herald Sq EL213: 6th Ave & 34 St entrance (SW corner) to mezzanine for service in both directions on all lines
- Grand Central-42 St EL204: 42 St (between Park Ave & Lexington Ave) to mezzanine for 4/5/6/7 service in both directions
- Grand Central-42 St EL205: mezzanine to downtown 4/5/6 platform
- Grand Central-42 St EL206: Mezzanine To Uptown 4/5/6 Platform
- Grand Central-42 St EL296: 42 St & Lexington Av to mezzanine for 4/5/6/7 service in both directions
- Grand Central-42 St EL606X: 1 Vanderbilt lobby on the Vanderbilt side directly across from Grand Central. 5 landings: street/lobby level, SUMMIT/subway/grand central terminal level, subway shuttle/GCT food concourse level, private (offices), GCM LIRR.
- Grand Central-42 St EL607X: One Vanderbilt mezzanine level to platform level for Shuttle service and connection to 4/5/6 and 7 via passageway
- Grand Central-42 St EL608X: 42 St between Vanderbilt and Madison Ave (inside One Vanderbilt building) to mezzanine level for service on all lines
- Grand Central-42 St ES608X: 4/5/6 mezzanine to Park Ave & 42 St (SW corner)
- Grand Central-42 St ES609X: Park Ave & 42 St (SW corner) to mezzanine for 4/5/6 service
- Grand Central-42 St ES610X: Intermediate mezzanine to Bowery Building (SE corner of 42 St & Park Ave)
- Grand Central-42 St ES637X: Platform level for Shuttle service to 42 St and Vanderbilt Ave (NE Corner).
- Grand Central-42 St ES638X: Platform level for Shuttle service to 42 St and Vanderbilt Ave (NE Corner).
- Grand Central-42 St ES639X: 42 St and Vanderbilt Ave (NE corner) to platform level for Shuttle service.
- Grand Central-42 St ES640X: Shuttle platform to LIRR “Madison Concourse” below.
- Grand Central-42 St ES641X: LIRR “Madison Concourse” to Shuttle platform.
- Times Sq-42 St EL231X: 7th Ave & 42 St (SE corner) to mezzanine for service in both directions on all lines.
- Times Sq-42 St EL232: nan
- Times Sq-42 St EL233: nan
- 42St/Port Authority-Bus Terminal EL288X: Port Authority Bus Terminal (8th Ave between 41st St & 42nd St) to mezzanine for A/C/E service in both directions via ramps
- 42St/Port Authority-Bus Terminal EL289X: Port Authority Bus Terminal (8th Ave between 41st St & 42nd St) to mezzanine for A/C/E service in both directions via ramps
- 42St/Port Authority-Bus Terminal EL290X: intermediate landing to mezzanine for A/C/E service in both directions
- 42St/Port Authority-Bus Terminal EL291X: 8th Ave & 44 St entrance (SW corner) to intermediate landing for A/C/E service in both directions
- Times Sq-42 St EL619: 43 St & Broadway (NW corner) to mezzanine for service in both directions for all lines.
- Times Sq-42 St ES261X: 42 St & 7th Ave (SW corner) to mezzanine for 1/2/3 service and access to rest of complex
- Times Sq-42 St ES262X: 1/2/3 mezzanine to 42 St & 7th Ave (SW corner)
- Times Sq-42 St ES267X: 42 St & 7th Ave (SE corner) to mezzanine for N/Q/R/W service and access to rest of complex
- Times Sq-42 St ES268X: N/Q/R/W mezzanine to 42 St & 7th Ave (SE corner)
- 42St/Port Authority-Bus Terminal ES276X: Mezzanine for A/C/E service in both directions via ramps to Port Authority Bus Terminal (8th Ave between 41st St & 42nd St)
- 42St/Port Authority-Bus Terminal ES606X: A/C/E mezzanine to intermediate mezzanine for 42 St & 8th Ave exit
- 42St/Port Authority-Bus Terminal ES607X: A/C/E intermediate mezzanine to 42 St & 8th Ave (SE corner)
- 51 St EL201: Uptown 6 platform to underpass downtown 6
- 51 St EL202: Downtown 6 platform to underpass for access to uptown 6 and E/F service in both directions
- Lexington Av/53 St EL203X: 52 St & Lexington Ave (NE corner) to mezzanine for E/F/6 service in both directions.
- Lexington Av/53 St ES253X: Downtown 6 underpass to mezzanine for E/F service and 52 St & Lexington Av exit.
- Lexington Av/53 St ES254X: E/F platform to mezzanine for 6 service and 53 St & Lexington Av exit.
- Lexington Av/53 St ES600X: 6/E/F mezzanine to 52 St & Lexington Av (NE corner).
- Atlantic Av-Barclays Ctr EL300X: Hanson Pl to LIRR concourse for access to Manhattan-bound 2/3 platform rest of complex
- Atlantic Av-Barclays Ctr EL306: Manhattan-bound 2/3 platform to mezzanine for access to rest of complex
- Atlantic Av-Barclays Ctr EL307: mezzanine to B/Q platform for service in both directions
- Atlantic Av-Barclays Ctr EL700X: Plaza entrance in front of Barclays Center to accessible passageway for access to Manhattan-bound 2/3 platform and rest of complex
- Atlantic Av-Barclays Ctr EL737X: Hanson Pl to LIRR concourse for access to Manhattan-bound 2/3 platform rest of complex
- Atlantic Av-Barclays Ctr ES358X: Barclays Center plaza (SE corner of Atlantic Ave & Flatbush Ave) to mezzanine for B/Q service and access to Manhattan-bound 2/3, 4/5, and rest of complex
- Atlantic Av-Barclays Ctr ES359X: B/Q mezzanine to Barclays Center plaza (SE corner of Atlantic Ave & Flatbush Ave)
- B'way-Lafayette St EL328: Houston St & Lafayette St (NW corner) to downtown 6 mezzanine and access to rest of complex
- B'way-Lafayette St EL329: downtown 6 mezzanine to uptown B/D/F/M platform
- B'way-Lafayette St EL330: downtown 6 mezzanine to downtown B/D/F/M platform
- Bleecker St EL331: nan
- Bleecker St EL332: nan
- Court St EL308: mezzanine (Clinton St & Montague St) to intermediate mezzanine for R service in both directions
- Court St EL309: mezzanine (Clinton St & Montague St) to intermediate mezzanine for R service in both directions
- Borough Hall EL318: mezzanine to platforms for 2/3 service in both directions and access to 4/5 service via Manhattan-bound 2/3 platform
- Broadway Junction EL352: Freight Elevator At Eny Depot
- Broadway Junction EL353: Freight Elevator At Eny Depot
- Cortlandt St EL16X: Oculus main level to Fulton Center level, balcony, and mezzanine for downtown R/W service
- Cortlandt St EL17X: Oculus main level to Fulton Center level, balcony, and mezzanine for downtown R/W service
- Cortlandt St EL18X: Church St to mezzanine for downtown R/W service
- Cortlandt St EL22X: Church St to South Concourse balcony and mezzanine for downtown R/W service
- World Trade Center EL23X: Church St to E line with access to downtown R/W, balcony & Oculus main level
- Cortlandt St EL726X: Fulton Center level to downtown R/W platform and access to rest of complex
- Cortlandt St EL727X: Fulton center level to uptown R/W platform and access to rest of complex
- Cortlandt St ES380X: underpass to mezzanine (Cortlandt St & Church St) for access to street and uptown R/W service
- Franklin Av EL339: Fulton St & Franklin Ave (SW corner) to Euclid-bound C platform, S platform, and footbridge for access to Manhattan-bound C service
- Franklin Av EL340: footbridge to Manhattan-bound C platform
- Fulton St EL714X: Transit Center
- Fulton St EL715X: Transit Center
- Fulton St EL716: Transit Center concourse to uptown 4/5 platform and access to Brooklyn-bound J/Z platform and rest of complex
- Fulton St EL717X: Transit Center (near Broadway & John St) to uptown 4/5 platform and concourse for access to rest of complex
- Fulton St EL718X: Transit Center (near Broadway & John St) to uptown 4/5 platform and access to rest of complex
- Fulton St EL719: mezzanine (Nassau St) and Broad St-bound J/Z to A/C platform for service in both directions and access to rest of complex
- Fulton St EL720: concourse (Brooklyn-bound J/Z) to A/C platform for service in both directions and access to rest of complex
- Fulton St EL721: 2/3 mezzanine to A/C platform for service in both directions and access to rest of complex
- Fulton St EL722: Nassau St & Fulton St (NE corner) to mezzanine for Broad St-bound J/Z service and access to A/C platform and rest of complex
- Fulton St EL723: William St & Fulton St (SW corner) to mezzanine for 2/3 and A/C service in both directions and access to rest of complex
- Fulton St EL724X: Corbin Building
- Fulton St EL725X: Corbin Building
- Fulton St EL728X: Dey St & Broadway (SW corner) to downtown 4/5 platform, access to rest of Fulton St complex, and connection to Oculus complex
- Fulton St EL729X: Transit Center concourse to downntown 4/5 platform and access to rest of complex
- Fulton St EL732: 2/3 platform for service in both directions to mezzanine for access to A/C platform and rest of complex
- Fulton St EL733X: Transit Center (near Broadway & John St) to upper retail floors
- Fulton St EL734X: Transit Center
- Fulton St ES360X: Transit Center entrance (SE corner of Fulton St & Broadway) to uptown 4/5 platform and access to rest of complex
- Fulton St ES361X: uptown 4/5 platform level to Transit Center exit (NE corner of John St & Broadway)
- Fulton St ES362X: uptown 4/5 platform level to Transit Center exit (Fulton St east of Broadway)
- Fulton St ES363X: concourse to uptown 4/5 platform and access to Transit Center exit
- Fulton St ES365X: Transit Center entrance (John St east of Broadway) to lower concourse for A/C/J/Z/2/3 & downtown 4/5 service and connection to WTC complex
- Fulton St ES366X: lower concourse to Transit Center exit (John St east of Broadway)
- Fulton St ES376X: Nassau St (between Maiden Lane & John St) to Broad St-bound J/Z platform and access to rest of complex
- Fulton St ES377X: Broad St-bound J/Z platform to Nassau St (between Maiden Lane & John St)
- Fulton St ES383X: uptown 4/5 platform level to Transit Center exit (SE corner of Fulton St & Broadway)
- Myrtle-Wyckoff Avs EL396: Gates Ave & Wyckoff Ave (SE corner) to mezzanines for L and M service in both directions
- Myrtle-Wyckoff Avs EL397: mezzanine to L platform for service in both directions
- Myrtle-Wyckoff Avs EL398: mezzanine to M platform for service in both directions
- South Ferry EL730: plaza in front of Staten Island Ferry Terminal (east side) to mezzanine for uptown service
- South Ferry EL731: mezzanine to platform for uptown service
- Jay St-MetroTech EL327X: Bridge St & Willoughby St (SE corner) to mezzanine for R service in both directions
- Jay St-Metro Tech EL706: Jay St & Willoughby St (NW corner) to mezzanine for A/C/F/R service in both directions
- Jay St-Metro Tech EL707: mezzanine to Manhattan-bound A/C/F platform
- Jay St-Metro Tech EL708: mezzanine to platform for Coney Island-bound F and Queens-bound A/C service
- Jay St-MetroTech EL709: nan
