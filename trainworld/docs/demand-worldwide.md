# Demand assumptions to recheck for cities around the world

Started 2026-10-12 at Anita's request. Every demand rule was built and fitted on nine US Northeast
cities. Before the first city outside the US (and whenever a rule changes), go through this list
and ask: does it hold for a city with a very different density pattern, commute culture, car
ownership or income? Each entry says what the game does now, what it was fitted on, why it might
not carry over, and what would check it. Add entries as new rules appear; strike through (do not
delete) ones that have been checked, with the date and outcome.

## The first country outside the US: Canada (Toronto, T-277, 2026-10-13)

What Canada's data changed, rule by rule (sources in notes/T-277.md, the adapter in
`pipeline/canada.py`, the substitution table in docs/adding-a-city.md "Outside the US"):

- **Institutions** are not taken off homes: Canada publishes people in collective dwellings only
  for larger areas, not by dissemination area. Its incarceration rate is about a fifth of the
  US's, so the error is smaller than in a US pack; a big care home or prison stays a home.
- **Job locations** come from the census's workers by census tract of work, not an employer
  register. The 2021 census was taken in a lockdown (35.6% of the Toronto CMA's workers at home,
  counted at home): their jobs are put back at workplaces by employment income band, and "no
  fixed workplace" workers added in proportion. Downtown Toronto is probably still under its
  pre-pandemic job count; for today's commuting with 24.7% at home it is plausible. Not
  reweighted by OSM offices (DECISIONS 2026-10-13).
- **Work from home** is one share for the whole pack (Labour Force Survey, May 2024, Toronto CMA
  24.7%), applied as "the same fraction of 2021's at-home workers commutes again" at both ends,
  Hamilton and Oshawa included (28.3% and 27.3% at home in 2021; their own current share is
  probably a little lower). US packs use ACS by tract and industry.
- **The distance curve** fitted on Canada's census flows between municipalities is much steeper
  than the US one: d^-1.2 exp(-d/20 km), mean commute 11.0 km, 5.6% of commutes misplaced between
  municipality pairs against 15.1% at the US curve (notes/T-277.md). Partly real (a strongly
  centred city, as the "distance curve shape" entry below predicted), partly the lockdown: the
  commuters of May 2021 were the ones still going in, more local than office commuters. Recheck
  on the 2026 census flows.
- **Everyday destinations** are OSM places counted per dissemination area (shops and services for
  errands; restaurants, bars, venues and hotels for leisure), not jobs by sector: a place is a
  place whatever its size. Where OSM is thin, so is the draw.
- **Job places within a tract** are 80% OSM job places (with commercial and industrial land at one
  place per 5,000 m², judgement) and 20% population.
- **Airports** have no connecting share from a national source (no DB1B): Pearson takes 0.75 by
  judgement, the share of the US hubs DB1B measures (Dulles 0.74), the others 1.
- **Schools and universities** come from provincial files (Ontario's), so each province needs its
  own; part-time college students are not counted.
- **Borders**: Ontario is one region with no factor; the pack does not reach the US border.

## Tier 0: global sources only (`pipeline/tier0.py`, T-287, 2026-10-14)

The adapter for countries with no small-area census inputs (Shanghai, Mumbai first). Measured by
building Toronto from tier 0 against the real pack (notes/T-287.md has every table):

- **Population** is GHSL GHS-POP (100 m) scaled to district totals: 2.0% misplaced between
  Toronto's 28 municipalities, but municipalities built up after 2020 come out 15-27% short (the
  GHSL epoch lags). Use the latest epoch and the newest district totals for fast-growing cities.
- **Jobs** are a proxy: population x the World Bank employment rate, spread 50% by GHSL
  non-residential volume weighted by height^1.5, 30% by OSM job places, 20% by population. 4.0%
  misplaced between municipalities, 18.6% between 2 km zones; the centre right within 10-25%.
  Without the height weight the centre had a third of its jobs. Why it may not carry over: the
  height power was picked on Toronto (in-sample); where much work is informal or at home (Mumbai),
  buildings miss it, so raise the population share; where industry is large, industrial land
  overdraws (Brampton +40%). Check: London (T-288) against BRES jobs by LSOA.
- **Where workers live** is population x one rate (no source by residence): 3.8% misplaced
  between municipalities in Toronto. Student and retiree areas send too many commuters.
- **The distance curve without flows** is the weakest step: fitting only its length (power 1) to
  a published median commute (Toronto 2016, 10.5 km) gave d^-1 exp(-d/95 km), far too flat:
  17.5% of commutes misplaced against 11.3% with Toronto's own fitted curve on the same tier 0
  inputs, and 22% fewer riders on a test line. Borrowing the US curve did better (15.7%). Check:
  where a distance distribution exists (India's census bands), fit both parameters; with a single
  figure, borrow the most similar fitted city's curve.
- **Schools** are OSM sites sized by the square root of site area with pupils from the national
  share aged 0-14 (-6% in Toronto, 5.3% misplaced); public and private alike.
- **Universities** are OSM campuses with Wikidata student counts, which few OSM campuses carry
  (4 of 63 in Toronto), so most are sized by area. Since T-288 the total is population x the
  national tertiary students a resident (UNESCO enrolment via the World Bank, SE.TER.ENRL over
  SP.POP.TOTL; UK 0.037 in 2018, Canada 0.044), and the counts and areas only spread it: London's
  counted institutions are dense central buildings (1,925 students a hectare at the median), which
  sized suburban campuses at 50-150k students and the pack at 2.4M against about half a million.
  Why it may not carry over: students gather in university cities, so a national rate undercounts
  a city with many campuses (Toronto -20% against Ontario's files; Boston or Oxford would be far
  under) and overcounts one with few. Check: London against HESA (T-360).
- **London's first reading** (T-288, tier 0, notes/T-288.md): GHSL put 12-19% too many people in
  central boroughs (Westminster, Kensington and Chelsea, Camden, Hammersmith) before scaling to
  Census 2021 districts, so the district totals matter in old dense centres. The jobs proxy
  draws far too little to the centre: 483k jobs in Westminster and the City against about 1.3-1.4M
  published, 4.36M in Greater London against about 6M, partly because the job total is the
  national rate x residents (a commuter city's core has more jobs than residents). Check: T-360
  against BRES by LSOA.
- **Airports**: Wikidata's patronage matched StatCan's within 0.3% for Toronto's three airports.

## Homes (docs/demand.md 4.1)

- **People in institutions are not homes** (T-234). The US packs take the 2020 census's
  institutional group quarters (prisons and jails, juvenile facilities, nursing homes, other
  institutions; PL 94-171 P5 per block) off every block's population, keep dorms, military
  quarters and other non-institutional places, and cap LODES job holders on those blocks at the
  people left. 0.5-1.3% of each pack's people. Why it may not carry over: censuses differ in
  where they count these people. Some count prisoners at their home address, not the prison
  (several US states now reallocate them for redistricting, though the 2020 PL file does not);
  register-based censuses (the Nordic countries, the Netherlands) count people at their
  registered address, which for long-stay inmates may be the institution or the old home; some
  publish institutional population only for large areas or not by type. Where no small-area
  figure exists, a global fallback is OSM `amenity=prison` and `amenity=nursing_home` polygons
  with a capacity estimate, and a population grid (GHSL, WorldPop) that spreads census totals by
  built-up area will put prisoners and residents of big care homes wherever the census put them,
  usually the institution's district. Elsewhere, where nursing homes are rare (most of Asia and
  Africa, where the old live with family), the rule matters less; where incarceration is far
  lower (Europe, Japan), prisons matter less too. Check, per country: does its census or register
  count prisoners at the prison, and is there an institutional population by small area; for one
  big prison per country, compare the population grid's cells over it with its capacity.

## Commute flows (SPEC 4, docs/demand.md 4.4)

- **Distance curve shape.** Work gravity uses d^-0.8 exp(-d/30 km) (T-198, from T-185; was
  d^-1 exp(-d/40 km), T-162), one value for the nine US packs, carried in each pack. What the nine
  showed (notes/T-198.md): the flatter near end helped most in the small and middle-sized packs
  (New Haven, Hartford, Springfield, Providence, Baltimore: trips misplaced down 5-9 points), less
  in Philadelphia, Boston and Washington, and was about neutral in New York, the densest and most
  centre-heavy (16.2% to 16.7%; its own-county share fell 2 points below LODES). Job-centre
  residents' trips lengthened by about the same 0.6-1.3 km everywhere and stay 9-19% short;
  bedroom areas' trips were too long before and still are; in New Haven, Hartford, Springfield and
  Boston the mean trip now runs 0.1-1.1 km over LODES. So the gain is not tied to density as such: in every pack, people
  living among many jobs still take jobs about as far down the list of nearest ones as anyone
  else (two-earner households, matching by skill), which a steep near end misses. Why it may not
  carry over: all nine are US metros with high car use and dispersed suburban jobs. In a strongly
  centre-heavy city (Paris, Tokyo, Seoul) the centre's pull may dominate and New York's neutral
  result hints the steeper curve could fit as well or better; in sprawling car cities (Los
  Angeles, Houston) both parameters may need to be longer; informal-job cities have no workplace
  register to fit to. Check: one non-US commuting-flow table per new country (England and Wales
  WU01EW census flows, Japan's census commuting tables, Korea's), fit the two parameters, compare
  with T-185's relative measures (local job ratio, nearest-job rank), not absolute density.
- **Border factor.** Cross-state US commutes are multiplied by 0.23 (Washington 0.45), fitted to
  LODES; outside the US the policy is national borders only, weaker inside Schengen (T-181). It
  does not stop NJ to NYC commuting (Hudson to Manhattan stays near the real figure); it removes
  the excess the plain gravity produced. Anita is unsure state lines matter much in practice.
  T-185 and T-198 showed the distance curve does not explain the cross-border gap: at the
  flatter curve the fitted factor is the same (New York 0.228, Philadelphia 0.207). T-171
  tested what could replace it on five packs' LODES: water or toll-crossing terms cannot (land
  borders with no water or tolls show the same cut), a factor only between metros cannot (half
  of New York's cross-state commuting is inside one urban centre), and per-pair factors only
  separate city-states (DC 0.61-0.66) from real states (0.17-0.37). So in the US the effect is
  the state line itself, inside metros too. Check: for each new country, whether its regional
  borders show any measurable effect in its flow data, and whether a city-state (Berlin, Vienna,
  Mexico City) behaves like DC (a weak factor) rather than a state.
- **Doubly constrained balancing.** Every zone's jobs are filled from homes. Holds where job
  counts are reliable and most workers commute within the pack. Weaker where the pack's edge cuts
  a commuter shed or where jobs data is a proxy (GHSL, T-012).
- **Job locations from registers.** LODES places jobs at the reporting establishment; head-office
  lumps (T-043) exist in other countries' registers too, and many countries have no workplace
  data at all (the GHSL proxy would replace it).

## Mode choice

- **Driving alternative.** Drive times and costs assume US car ownership and parking. Car
  ownership varies by a factor of five or more between cities (Manhattan vs Houston vs Tokyo vs
  Lagos); without it, transit share is wrong everywhere outside the US Northeast. Check: a car
  ownership or motorisation rate per country or city, and whether the driving alternative should
  be unavailable to a share of people.
- **Routed drives** (T-275, docs/demand.md 4.3). A drive's road is the straight line times a
  per-pack routed factor over OSM roads (every public road but service ways, shortest distance),
  times one scale, 1.3 / 1.210, that puts New York's commute-weighted factor back at the 1.3 its
  density speeds were fitted with. The nine US packs come to 1.17-1.26 before the scale. Why it
  may not carry over: the scale ties every city to New York's speed fit; shortest-distance routes
  undercount more where drivers keep to a sparse motorway grid (Los Angeles, Houston) and less in
  old street grids (European centres); OSM's classification differs by country (rural roads
  tagged `track` are not in the drive graph), and residential
  coverage is thin in parts of Africa and Asia, where the nearest road joined can be kilometres
  off. Ferries that carry cars (Hong Kong, Istanbul, Seattle, the Baltic) are not roads here, so
  a city across a strait has only its bridges and tunnels. Check: per new country, the
  commute-weighted factor the pipeline reports and the median snapping distance (US packs:
  94-213 m); a city's ACS-like car commute times against the model's, as T-090 did.
- **Long drives** (T-370, docs/demand.md 4.3). Each end's density speeds last its own first 10 km
  of road (2 local, 8 main); anything beyond is open highway at 100 km/h. Why it may not carry
  over: 100 km/h assumes a motorway between the ends (US, Canada, western Europe); where long
  roads are two-lane or congested end to end (much of India, Indonesia, Lagos) the middle is far
  slower, and a sprawling metro whose dense core is wider than 10 km (Mexico City, Jakarta) is
  slow for longer. Check: a few typical door-to-door drives of 50-130 km per new country against
  the model's (notes/T-295.md has the US, Toronto and London ones).
- **Routed walks** (T-275). A walk to a station goes round water on a 50 m raster from the pack's
  water mask, with every OSM way a person may walk on that is a bridge opened. Why it may not
  carry over: the water mask is TIGER's in the US and OSM's elsewhere (T-041; within 0.6% of TIGER at New York); OSM's `foot=` tagging
  on bridges is good in Europe and patchy elsewhere, and trunk bridges are assumed to have no
  footway unless `foot=yes`, which drops real footways in countries that tag them on the road
  itself; walking ferries (Istanbul, Hong Kong's Star Ferry, Venice) are no walk; a bridge
  passing over an island joins the island (Roosevelt Island over the Queensboro Bridge).
  Barriers other than water (motorways, rail yards, gated estates, rivers in deep cuttings) are
  not barriers. Check: the probes in notes/T-275.md for a few crossings per city
  (`pack_walk --probe`).
- **Walking curve.** Fitted to New York (ACS B08301 walked, T-058/T-090). Walking tolerance
  differs by climate, culture and street design.
- **Value of time and fare sensitivity.** One value for everyone, from US figures. Incomes differ
  hugely between and within countries (see Anita's "income dependence" idea in her TODO).
- **Waiting, transfer and crowding penalties.** From transit literature that is mostly US and
  European. Riders in Tokyo or Mumbai tolerate crowding very differently.
- **Riders left behind** (T-188, docs/demand.md 4.4 step 7). A boarding's cost includes the
  average extra wait for full trains, worked out per period with the boarding account's rule:
  arrivals peaking mid-period at the US busiest-hour factors, trains taking riders up to crush
  (160 a 20 m car, a New York car), the extra wait weighted like any platform wait (1.37) and
  capped at 60 minutes. Why it may not carry over: crush loads differ (Tokyo and Mumbai pack far
  more than 160 into a car, pushers included, so "full" comes later; many European systems run
  emptier); where riders are routinely left behind (Mumbai suburban, Lagos, Chinese metros with
  queue control at the station entrance) they plan for it, and the perceived weight of that wait
  may be lower, while elsewhere one missed train feels much worse than its minutes; the peak
  shape within a period is US commute timing. Check: published crush capacities per car for the
  country's stock, and left-behind or platform-queue surveys where they exist (London's and Hong
  Kong's operators publish some).

## Other purposes

- **Everyday trip rates.** Shopping/errands 0.299 and social/leisure 0.200 outings per resident a
  day, New York's rates applied everywhere (T-142). Destinations from LODES sector jobs (NAICS
  retail, food, entertainment), population and parks (social 46 / 46 / 8%, T-196). ATUS 2023
  puts only 23% of US leisure outings at someone else's home against the model's 46% population
  draw (the population draw also stands for neighbourhood leisure LODES misses), worth a look
  with any non-US survey. Rates and destination proxies need a non-US source (national
  travel surveys; OSM shops and amenities as destinations where no jobs-by-sector exist).
- **Parks** (T-196). OSM parks take 8% of the social/leisure destination draw, from ATUS 2023
  (7.6% of US leisure outings end outdoors away from home); each place draws sqrt(area) x
  D / (D + 2,000), D the residents per km² within 2 km of its edge; nature reserves count fully only when
  named as parks, else at a quarter. Why it may not carry over: the outdoor share depends on
  climate and culture (Nordic and Australian cities likely more, hot or very dense ones less, and
  ATUS is a year's average with no season); the reserve rule reads English names ("Park",
  "Recreation Area"), so other languages need their own words or the quarter for every reserve;
  OSM tagging differs by country (UK commons and recreation grounds are counted, but whether a
  country maps its urban parks as `leisure=park` at all needs a look); the half-point of 2,000
  residents per km² is absolute (about a US inner suburb), so in dense cities (Mumbai, Dhaka,
  Seoul) almost every park saturates and only size separates them, which is probably right; in
  US exurbs a big state park is discounted to a tenth. The size exponent 0.5 and the 2 km reach are
  judgement, not fitted. Check: a national time-use survey with a location code (the UK and most
  of the EU have HETUS-style diaries) for the share; park visit counts (mobile-phone or ticketed
  parks) in one city per country for the size and population terms.
- **School and university.** NCES and IPEDS are US-only. Abroad: national school registers where
  they exist, OSM school and university points with an enrolment estimate otherwise.
- **Airport and medical** (T-142): FAA enplanements are US-only; OurAirports and OSM hospitals
  are global but need a passenger or visit figure.
- **Trip timing.** Period shares and departure profiles come from US data (ACS B08302, NHTS).
  Work hours, siesta patterns, school hours and late evenings differ by country.
- **Weekends** (T-260, docs/demand.md 4.2). A weekend day's trips per weekday trip by purpose come
  from NHTS 2017 (work 0.30, shopping and errands 1.22, social and leisure 1.60, school 0.08,
  university 0.22, medical 0.24, trips of 50 miles or more 1.58) and TSA checkpoint counts
  (airport 1.01); weekend timing is the weekday's shifted by NHTS's weekend/weekday ratio per
  period. England's NTS 0504b gives nearly the same (commuting 0.28, shopping 1.26, leisure 1.57,
  education 0.03), so the shape holds at least for rich car-owning countries. Why it may not carry
  over: the weekend is not Saturday and Sunday everywhere (Friday and Saturday in much of the
  Middle East, Friday alone or Friday and Saturday elsewhere; six-day working weeks in parts of
  Asia and Africa), so the calendar must become per country (T-262); where Saturday work and
  school are common (Japan's and Korea's Saturday schools until recently, retail and informal work
  everywhere) weekend work and school run higher; where shops close on Sundays (Germany, Austria,
  parts of Switzerland) Sunday shopping collapses while Saturday's rises, so a pooled weekend hides
  two very different days; and where public holidays cluster (Golden Week, Chuseok, Chinese New
  Year) the long-distance factor is a holiday matter, not a weekend one (T-264). The calendar's
  game day 1 being a Monday is a game choice, not a rule. The weekend also keeps the weekday's
  crowded route choices, which matters more where weekday crowding is extreme (Tokyo, Mumbai).
  Check: each new country's national travel survey by day of week and purpose (most publish it:
  MiD Germany, ENTD France, the Japanese person trip surveys, the Korean household travel survey),
  and its legal weekend days.

## Network and stations

- **Station merge and walking transfer distances** (100 m merge, 300 m walks). Fine for US
  stations; very large hubs (Shinjuku, Châtelet) have transfers over much longer underground walks.
- **Complexes as transfer points** (T-161, with the player's links, T-197). Any two stations of a
  complex are a walking transfer however far apart, costed like every walk between stations:
  centre-to-centre distance x 1.3 detour at 1.2 m/s, weighted 1.39, plus the boarding's usual
  wait and 5-minute transfer penalty, with no extra fixed penalty for the walk. Set by judgement
  to agree with the existing walks, not fitted; on New York's real save it moves nothing (every
  complex's demand stations are within the 300 m walk already). Why it may not carry over: in
  deep, crowded hubs (Shinjuku, Châtelet, Moscow's linked stations, London's Bank) a long
  interchange is slower than 1.2 m/s on the straight line x 1.3 (stairs, escalators, crowds), and
  riders weigh it more heavily; in the other direction, a cross-platform design (Hong Kong,
  Singapore) makes a change nearly free, which the flat 5 minutes misses. Check: published
  interchange times for a few big hubs (London's TfL interchange walk times, Tokyo station transfer
  guides) against what the rule gives for the same platforms.

## Intercity

- **The regional market's trip rate and curve** (T-276, docs/zones.md 5). 0.107 non-commute
  departures per resident a weekday, shared by attraction x onset(25-50 km) x max(d, 25)^-2.37
  among every zone and an outside world, fitted on NHTS 2017 Northeast residents' weekday trips by
  ground, 50-1,000 km (research/intercity/market_fit.py; an exponential length went to infinity in
  the fit). Why it may not carry over: long trips per person rise with income and car ownership
  and fall with distance differently where long distances are flown or taken by coach (Europe's
  and Japan's national surveys have far fewer long car trips per person; China's and India's
  holiday travel is seasonal, T-264); the power law comes from one dense, car-owning region. Check:
  each country's national travel survey by distance band (MiD Germany, ENTD France, NTS England,
  Japan's Inter-Regional Travel Survey), refit R and the power with that country's outside world.
- **The outside world** (T-276). Kontur population outside every zone, at the zones' mean
  attraction per person (1.98 in the Northeast: people + 2 x jobs). Where jobs per person differ
  (poorer countries, retirement regions) the factor is off for the outside alone; harmless while
  the outside is only somewhere trips leave to.
- **The longer trip's car** (T-276). Drive 15 min + 1.25 x the great circle at 85 km/h, plus 15
  US cents a road km (AAA 2024 fuel and upkeep) and the local layer's parking at the destination,
  both split over two occupants, at 3 perceived minutes a dollar. Fuel prices, tolls (T-270),
  motorway speeds, occupancy and car ownership all vary by country; Europe's fuel costs about twice
  the US's per km, so cars there would cost more. Check: a fuel price and toll regime per country.
- **Commutes between zones** (T-276, docs/zones.md 4). One gravity over every zone pack, each
  pack's own curve and border factor inside it, region pairs across packs from the pack holding
  both, else the country's admin-1 default (US 0.23), national borders 0.05 (US-Canada, judgement).
  Checked, not fitted, against LODES jobs between packs: Baltimore - Washington 0.89x, Boston -
  Providence 0.82x, New Haven - New York 0.92x, Hartford - New Haven 0.99x, New York - Philadelphia
  0.56x (LODES counts long office-address links, notes/T-268.md). Why it may not carry over: where
  commuting crosses national borders freely (Schengen: Basel, Geneva, Luxembourg) 0.05 is far too
  strong; where a country's admin-1 lines do not cut commuting (T-181) the default is 1. Check: the
  country's census commuting flows between its zones.
- **Zones** (T-276, docs/zones.md 2). Non-city zones are county-sized units touching two city zones;
  the unit (US county, Canadian census division, NUTS-3) differs in size by country, and where
  admin-2 units are tiny (French communes, Japanese municipalities) a coarser statistical unit is
  needed.
