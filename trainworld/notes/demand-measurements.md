# Demand model: cost and accuracy measurements

The numbers behind the demand docs' rules, by section; the rules are in the docs.

## Kernel cost and accuracy, moved from docs/demand.md 4.4 (T-588, 2026-10-17)

Moved in the T-588 tidy: the numbers behind the kernel's cost, taken on the real New York pack with the hand-made network and on the combined save. The rules they led to are in docs/demand-kernel.md and docs/demand-runtime.md.

**Cost.** One period on the 202,337-cell New York pack and the hand-made network, one stage (the
free-flow estimate or one crowding round), one thread: 0.25-0.3 s native, 0.27 s in WASM (0.23-0.24
before T-090's wider reach: the extra origin subzones each build their station row and visit the
destination zones, and the drive costs a little more per pair; a day is 15% longer in WASM). An
edit reruns everything except gravity. Periods are independent, so the game runs them in a pool
of workers ([architecture.md](architecture.md) 7): with three workers in Chrome the mode split updated 0.49-0.55 s after a
new network reached them and the third crowding round landed by 1.9-2.0 s at T-078; T-090 adds
about 15% (not measured in Chrome). Each worker holds its own copy of the city (one for its commutes and everyday layer together): 18 MB of WASM
memory open, about 23 MB after solving (notes/T-026.md). New York's real 867-station network is
13.8M pairs, 1.0-1.7 s a stage in WASM, 24-25 s for a whole day on one worker (20 s before T-090).

**Accuracy of subzones** against one subzone per cell with the same model (T-090, hand-made
network, free flow): rail total within 0.2%, station entries and exits within 5.0% and 6.3% (sum
of absolute differences), segment loads within 3.0%. The logit roughly halved the station error
of all-or-nothing choice at any grouping, which is why the 5-minute walk bands are gone; what is
left is the grouping (a subzone's walk is one figure for all its cells). A plain mean of the
cells' perceived walks put rail 3.3% under, since the walk's cost is convex.

**Cost on the real combined network** (T-119: the 109-line New York save plus one Philadelphia
line, 875 demand stations, 5,039 subzones, 13.9M subzone pairs a period; native, one period, free
flow or one round, three populations): 1.75 s before. Strategy searches took 0.28 s for each of
the three populations (Philadelphia's has 35 subzones but searched towards all 875 stations); New
York's mode choice 0.81 s; the market's pair loop 0.06 s. Now 1.25 s: one search (0.29 s), the
mode choice, and Philadelphia's 2 destinations. Results are identical to floating point noise
(max difference 2e-16 of each array's largest value over line loads, boardings, entries and the
summaries of all five periods). Since T-172 (fixing T-121) the market's free-flow estimate also
routes with the distance fare, as the local populations' does, so round 0 shares their search
too; before, its own round-0 search towards every station a far zone can reach took 1.8 s natively
a period on the 109-line save (against 0.1 s for a crowding round), and in five Chrome workers
at once made round 0 four times as long as a crowding round (10 s against 2.3 s).

In Chrome on this machine (16 threads), the real network plus the Philadelphia line:

| | before (3 workers) | 3 workers | 5 workers |
|---|---|---|---|
| free-flow estimate published | 4.5 s | 3.2 s | 1.7 s |
| round three published | 17.2 s | 11.0 s | 6.1 s |
| bubbles drawn | 17.4 s (after round three) | 3.4 s | 1.9 s |
| WASM memory per worker | 129-132 MB | 120-134 MB | 120 MB |

Each period's stage in a worker takes about 1.5 s (2.1-2.3 s before). Historical single-city
measurements above retain their context.

**Memory follow-up** (T-123). The regional JSON pack now parses into flat numeric rows, skipping
metadata; temporary rows are consumed before constructing pairs. Immutable pair and lookup
vectors release their unused capacity. This preserves all trip parameters and results. On the
current full-network fixture, opening the five solving workers plus the auxiliary uses
360 MB against 596 MB before; each solving worker after the solve uses 127–128 MB against
137 MB. Estimated pool memory falls by 46 MB (6.4%) to about 668 MB, including the auxiliary's
opening allocation. These are WASM linear-memory measurements, not browser RSS. Live network,
strategy and period arrays still set the full-network peak. Comparison: notes/T-123.md.

## Where trips come from, moved from docs/demand-sources.md (T-940)

**Merged cells** (T-044, T-466, T-510, T-850b). Together the merge rules removed a median 68% of
a city pack's res-9 cells (52-78% for most) and 78% of a countryside zone's (71-85%), 78% of all.

**Workers' homes** (T-054). With LODES RAC at the home end, county totals came within 10% of ACS
commuters; with population the Bronx had been 23% over.

**People in institutions** (T-234). Over the nine Northeast packs the rule took 362,000 people
(0.8%) and 66,000 job holders (0.3%) off homes; LODES had put 676 job holders on Rikers Island's
blocks, where 27 people live outside the jails. The park draw's nearby-residents term moved under
1% overall, but a small park beside a big prison drew up to about four times too much. Per city:
notes/T-234.md.

**Parks** (T-196). Counted rather than by density, Harriman State Park (19,000 ha) outdrew
Central Park five times; by density Central Park draws 17, Harriman 15. ATUS 2023: 7.6% of
leisure outings end "outdoors away from home" (95% interval 6.4-8.7%); friends' homes 23% and
restaurants, gyms and other places 70%.

**Everyday decay** (T-637). A single exponential at 2 km zones gives trips whose mean is about 1.2
times their median, where surveys' means are about twice their medians, so no scale matches both;
the shorter scale fits better where most trips are.

**Schools** (T-140, T-150, T-181). NCES 2023-24: 230 to 1,370 schools in each of the other
cities; New York 5,064 schools with 2,825,244 enrolled pupils, charter schools included; 100% of
eligible enrollment matched to cells, with at most two snaps (New York one, 237 m). Until T-181
New York's five boroughs used NYC DOE building enrollment (844,956 pupils in 1,213 buildings);
NCES has 975,570 there, the difference mostly charter schools (141,775 pupils), which the DOE
building report did not carry (notes/T-181.md). The New York pack holds about 3.87 million
estimated child-home weights.

**Universities** (T-152, T-142). IPEDS 2023: New York 261 institutions and 836,714 eligible
students; the other cities 8 to 76 institutions, 48k to 282k students. Splitting Connecticut State
Community College over its campuses gave Hartford 5 of the twelve, New Haven Gateway and Naugatuck
Valley, and New York's Fairfield County Housatonic and Norwalk (5,330 students New York had not
counted). About 1.0 million university trip legs a day in New York.

**Airports** (T-142). O&D shares (DB1B 2024 Q2, domestic): 0.66 BWI, 0.74 Dulles, 0.82-0.83
Philadelphia and National, 0.90 the three New York airports, 0.98-1.00 elsewhere. Pack shares: BWI
0.66 to Baltimore, the rest to Washington; Bradley 0.57 to Hartford; Tweed 0.47 to New Haven; the
big hubs 0.94-1.00. Every airport with over 1,000 outings a day kept its total within 0.4% after
the 150 m exclusion and the taper, except Westchester (4% short) and Long Island MacArthur (11%
short), whose catchments reach past the 80 km cut.

**Medical** (T-142). 0.04 outings a resident a day is 0.08 legs, against NHTS 2022's 0.091
medical and dental legs a person a day (4.02% of 2.27 trips). Two destination samples a home cell
against sixteen move the walk share at most 0.3 points and the mean length 0.1 km.

## Worldwide rules, moved from docs/demand-worldwide.md (T-940)

**Canada** (T-277). The curve fitted on Canada's municipality flows, d^-1.2 exp(-d/20 km): mean
commute 11.0 km, 5.6% of commutes misplaced between municipality pairs against 15.1% at the US
curve.

**Tier 0 on Toronto and London** (T-287, T-288, T-314, T-360). Population: 2.0% misplaced between
Toronto's 28 municipalities. Jobs proxy: 18.1% (Toronto) and 26.1% (London) of jobs misplaced
between 2 km zones, both in-sample. London's proxy gives Westminster and the City 483k jobs
against BRES's 1.51M and Greater London 4.36M against 5.88M; no blend, height power or
concentration exponent fixes it without wrecking Toronto (the height weight that finds Toronto's
core misses London's). Pack jobs against the real total: London -19%, Toronto +9%. Where workers
live (population x one rate): 3.8% misplaced between municipalities in Toronto. Borrowed curve
(at the time, `fitted_curves.json` 59 packs in 8 countries): held out a country at a time it cost
2.9 points on average by country, against 3.3-3.9 for nearest by population and vehicles, by
income group, a world median or a length scaled to the city's size; the US packs lose 4-14 points
under any borrowed curve. The zone curve over 38 fitted Japanese prefecture zones: 0.3 points over
each zone's own on average; Japan's zones keep 52-91% of commuters in their own municipality.
Road distance factor: Mumbai's routed drive factor at the band edges 1.30-1.45 (the near zone
pairs' unweighted median of 1.62 is not the commuters' factor). Schools: -6% in Toronto, 5.3%
misplaced; London -8% against GIAS's pupils on roll (which include sixth forms), 21% misplaced
between 2 km zones and 14% between districts. Population by small area: London 3.3% misplaced
between 2 km zones against Census 2021 by LSOA, 13% between LSOAs; GHSL put 12-19% too many
people in Westminster, Kensington and Chelsea, Camden and Hammersmith before scaling.
Universities: Wikidata counts on 4 of Toronto's 63 OSM campuses; UK 0.037 tertiary students a
resident in 2018, Canada 0.044; London's counted institutions are dense central buildings (1,925
students a hectare at the median), which sized suburban campuses at 50-150k students and the pack
at 2.4M against about half a million before T-288's national total. Airports: Wikidata's
patronage matched StatCan's within 0.3% for Toronto's three airports.

**Country packs** (each country's notes have the full tables). Paris (T-289): tier 0's proxy
alone gave Paris 60% of its INSEE jobs (the 8th arrondissement a quarter) and Ile-de-France 89%,
17% misplaced between 2 km zones against London's 27.5%; on the flows, Toronto's curve scores as
Paris's own, tier 0's one-parameter rule (power 1.4, length to the flows' median of 6.3 km) within
half a point, London's d^-1.85 two points worse and the US curve five. Shanghai (T-291): GHSL 24%
too many people overall, 23-34% in the central districts and Chongming; the proxy alone gave the
seven central districts 19% of jobs against 36% and Baoshan and Jinshan nearly double;
CAUPD-Baidu 9.8 km against the survey's 10.2 km; OSM 10 airport terminals, one hospital with beds,
18k job points and 22k job areas, 551 medical sites. Mumbai (T-292): GHSL's total within 3% of the
grown census, the island city 14% high, the Thane suburbs 10-24% low; the proxy gave Greater
Mumbai 45% of jobs against its 52% of 2011 workers by residence, overdrawing Uran, Khalapur and
Palghar; 12 of 1,460 places had `name:en`; 803 school sites for 6.2M pupils; median drive factor
1.62 between near zones. Lagos (T-459): GHSL E2020 13.8M; UN WUP 2018 14.4M for 2020, Africapolis
13.8M plus 1.6M Ikorodu (GHSL's Ikorodu LGA 0.7M); OSM land use 81% of the OSM job signal, the
23 km² Dangote refinery would have taken 425k jobs (6.5%) uncapped; 0.09 errand and 0.04 leisure
places per 1,000 residents (Mumbai 0.29 and 0.19, London 5.9 and 4.0), in 2% of cells holding 7%
of the people; borrowed-curve mean 9.0 km against Osoba's 2010 survey mean trip of 12 km by road.
Sao Paulo (T-458): GHSL within 5% of the census in most municipalities, 15-23% low in Barueri,
Cajamar, Cotia and Santana de Parnaiba; before Brazil's own everyday rates (T-509) social outings
ran 10.8 km against the survey's 4.9 and sent twice its share to the centre (decay lengths were
per country then: 2.3, 1.9 and 3.7 km). Berlin (T-290): GHSL within 2.5% in Berlin, 6-10% low in
most ring Kreise; the proxy alone gave Berlin 78% of its jobs and the ring Kreise 1.3-1.9 times
theirs (14% misplaced between Gemeinden); Paris's and Toronto's curves score as Berlin's own,
London's two points worse, the US curve three; 1.67M internal Berlin commuters in one cell; the
national tertiary rate gives 209k students; OSM 70k job points, 190k job areas, 1,737 school sites,
4,024 medical sites, median drive factor 1.42. Seoul (T-457): GHSL Hanam +65%, Gimpo +57%, Gwangju
+62%, 15-27% high in old dense districts; mean commute 20.4 km along the route, 15.7 km
straight-line, which at power 1.4 needs a 156 km length; Wikidata counts for 3 of 217 university
sites. Japan (T-420, T-431, T-432, T-426): the economic census counts 9% more than the census, the
per-municipality scaling moves single wards up to 38% (Yokohama Nishi-ku down, Saitama Chuo-ku
up); the fit is flat from power 1.85 to 2.3, the census 34.9% working in their own municipality;
regional metros' best fits misplace 8-12% between municipalities, Tokyo's curve 9-12%; non-city
zones 52-91% in their own municipality, misplacing 3-11%. Work from home means: Greater Tokyo
0.23, Keihanshin 0.17, the zones 0.14-0.23. Commutes between packs: at factor 1 the gravity sent
1.9 times the census's; 1.6% of Japan's commuters cross a pack line; the spread over pack pairs is
a quarter off the census's (Shiga into Kyoto and Osaka at a third, Shiga to Nagoya at 3x). Names:
`name:en` 64% of Greater Tokyo's places, kana romaji 1.2%, dictionary 35%, kanji 0.5%. After
T-636 dropped Japan's +0.5 rail constant, every Japanese test cross carried 0.4-0.7 of its rail
trips against before.

**The US distance curve** (T-162, T-185, T-198). d^-1 exp(-d/40 km) (T-162) became d^-0.8
exp(-d/30 km). On the nine packs the flatter near end helped most in the small and middle-sized
packs (New Haven, Hartford, Springfield, Providence, Baltimore: trips misplaced down 5-9 points),
less in Philadelphia, Boston and Washington, and was about neutral in New York (16.2% to 16.7%; its
own-county share fell 2 points below LODES). Job-centre residents' trips lengthened by about the
same 0.6-1.3 km everywhere and stayed 9-19% short; bedroom areas' trips were too long before and
still were; in New Haven, Hartford, Springfield and Boston the mean trip ran 0.1-1.1 km over LODES.
London on the 2011 flows' own margins fits d^-1.5 exp(-d/20 km); the US curve misplaces 15% of
Toronto's and 32% of London's commutes.

**The US border factor** (T-171, T-185, T-198). At the flatter curve the fitted factor stayed the
same (New York 0.228, Philadelphia 0.207). On five packs' LODES: land borders with no water or
tolls show the same cut; half of New York's cross-state commuting is inside one urban centre;
per-pair factors only separate city-states (DC 0.61-0.66) from real states (0.17-0.37).

**Rail propensity** (T-509, T-636). T-509 had fitted 0.4 (8 perceived minutes), everyday trips 1.1
lower and Japan +0.5 to six real networks. With no constant (T-636) the model carried x3.1 New
York's station entries and x2.0 London's (notes/T-636.md; before T-918's ratio curve).

**Walking the whole way** (T-558, T-636). T-558's version (2.80 and 3.21, worth 56 minutes at the
old scale, and 4 a minute past 10 minutes) had been set to keep New York and London walking as
before; T-636 replaced it. New York's real trips walk 58% within 2 km (NHTS 2017); the model
walked 55% there with the real save's rail network (at T-587).

**Everyday scales** (T-637). Each country's scale fitted to its survey's mean put too few trips
where people walk (New York 23% of errands under 2 km against NHTS's 50%; London's everyday trips
walked 18-27% against LTDS's 41-44%). New York's NHTS everyday rail legs: errands 4.7%, social
7.2%, against the game's 13% at the time.

**Everyday destination sources** (T-456). Japan's mesh with wholesale left in failed the
concentration test (11% in Tokyo).

**The world market's open-border area** (T-809). London - Paris about 28k a weekday against
about 26k published (rail and air); the land pairs come out low (Paris - Brussels about two thirds
of its rail alone, Amsterdam - Brussels about half).

**Air door to door** (T-712). Published fares: the US $200-220 each way at 400-2,000 km (DOT's
Consumer Airfare Report Table 6, 2025, taxes in, bags out); JAL's domestic revenue a passenger
about ¥15,700 or $108 at 770 km (FY2023); DLR's Low Cost Monitor €81-141 one way (2026). On ideal
access rail held about 99% of rail and air at 3 hours on the train, 92% at 4 and 56% at 5
(European studies: 70%, half, a fifth); over whole corridors the station's reach brings it down.

**Commutes between zones** (T-268, T-539). Against LODES jobs between packs: Baltimore -
Washington 0.89x, Boston - Providence 0.82x, New Haven - New York 0.92x, Hartford - New Haven
0.99x, New York - Philadelphia 0.56x (LODES counts long office-address links). Pairs more than
50 km apart in a shared gravity carried 0.3% (Northeast) and 0.5% (Japan) of the commuters
crossing a pack line.

**Zone pairs in chunks** (T-780). All of Japan: 12.4 million longer-trip zone pairs to 3.1
million.

**Weekends** (T-260). England's NTS 0504b weekend per weekday: commuting 0.28, shopping 1.26,
leisure 1.57, education 0.03.

## Workers, zones and clusters, moved from docs/demand-runtime.md (T-940)

**Which cities are open** (T-149). A worker check solved a Midtown line with all nine packs open
and with New York alone; every line load, boarding, crowding figure, New York and market summary
and corridor was bit-identical, and the other eight cities had zero rail (notes/T-149.md).
Opening a commute pack natively: 16 ms New York, 1-7 ms the others. Everyday purposes natively
since T-397's sample counts: New York 0.6 s, Washington 0.3 s, Philadelphia 0.2 s, Boston 0.2 s,
Baltimore 0.1 s, the four small ones 0.02-0.06 s; in Chrome, measured before T-397 with about
twice the samples, New York 1.3-1.9 s in each of five workers, Philadelphia 1.1-1.2 s and
Washington 1.7-1.8 s. A market composition: 0.7-3.5 s in Chrome on a big game (T-889). Natively
New York's everyday layer holds 2.3M samples, about 18 MB with their pair indices (T-397). A
solving worker on the real save holding three periods: about 156 MB of heap after a solve and
188 MB at its peak; with all nine Northeast cities open, 282 and 318 MB (T-399). T-172's grid
table for the gravity decay agreed with the old per-pair square root and exp to 7 significant
figures in every purpose's trips, walk share and trip length; a single long WASM call had made
the opening 3-5x slower in Chrome than natively.

**Bare zones at startup** (T-351). A new game's loading card went 2.6-2.8 s after navigation
(dev server, warm), against 6.6-7.1 s before; the bare opening 2.0-2.2 s of it, 0.4 s more
without the everyday warm-up. A saved game's network arrives about 2.1 s after navigation against
0.65 s for its viewed pack (the real New York save); waiting for the reach would cost every save
that reaches its viewed zone about 1.4 s. The first station: `prefetchViewed` 2.4-2.6 s for New
York; a first line's first estimate came 2.0-2.1 s after the build, against 4.4-4.6 s with no
tool picked first.

**Bare zones on a pan** (real New York save, five solving workers, a 13-step pan): each solving
worker stayed at 354 MB, against 855-955 MB when every zone on the way opened in every worker
and was solved (10 solves); the auxiliary worker grew from 76 to 477 MB; the pool from about
4.6 GB to 2.25 GB. A bare zone's bubbles came 1.5-3.6 s after arriving, against a final result
16.6-22.6 s after. A click on a bare zone's bubble drew its far end in 155 ms. Opening bare took
0.7-1.4 s (Toronto, Philadelphia) in Node.

**Everyday warm-up** (T-173). Five workers opening New York's everyday layer at once from
baseline code took 7-8 s each instead of 1.3-1.8 s in most page loads (one worker alone 1.6 s,
three 4.1 s each: as if they shared one core; never with Chrome's `--no-wasm-dynamic-tiering`;
not the module cache, efficiency mode or the tiering budget). The warm layer opens in about
150 ms, ready 0.36-0.66 s after navigation, about when New York has downloaded. With it New York
opened in 1.3-1.8 s in every worker in 10 of 10 page loads, against 3.4-7.3 s in 7 of 10 without
(`?perfOff=stagger`, same build).

## The world market (T-377, T-118), moved from docs/demand-market.md (T-940)

Opening a composition of all 20 places, visiting no pair: 0.66 s natively. A period's regional
solve: about 8 ms on the small real-pack profile (3.7 ms at an earlier, unrecorded state).

## The trip explorer (T-195), moved from docs/demand-views.md (T-940)

Taken when the explorer shipped, under the mode choice of the time (before T-636 and T-918), so
the shares are history, not the current model's. The WASM worker's answers for the three
`trip_real_network` trips equalled the native ones to 1.7e-7 relative (rail and drive chances
identical). On the real save in Chrome a trip was answered in 3-14 ms. Park Slope to Midtown in
the morning peak: train 91%, drive 9% (the drive 36 minutes plus parking that counted as 43); the
busiest route the F from 7 Av to 47-50 Sts-Rockefeller Ctr with 17% of the trip's train riders,
the rest spread over neighbouring stations at both ends and the 42 St stations. Jersey City
(Journal Square) to Wall St: train 62% (PATH to World Trade Center, then the 1 or a walk). Mineola
to Hicksville: drive 63%, train 37%.
