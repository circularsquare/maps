# Long-distance trips: moonraker's report, and why they are a purpose

2026-11-28. Written for Anita. Reading and writing only: nothing in the game was changed or run.

moonraker's four screenshots (saved by the Discord reader):

- C:\Users\anita\projects\maps\artifacts\trainworld\feedback\discord\bug-reports\1576229424242892830-1-busrouteridership.png
- C:\Users\anita\projects\maps\artifacts\trainworld\feedback\discord\bug-reports\1576229424242892830-2-demandcirclesfromduluthcentre.png
- C:\Users\anita\projects\maps\artifacts\trainworld\feedback\discord\bug-reports\1576229424242892830-3-demandcirclefromregion.png
- C:\Users\anita\projects\maps\artifacts\trainworld\feedback\discord\bug-reports\1576229424242892830-4-clickondemandcircle.png

## a. What moonraker saw, and what causes it

moonraker (bug reports, 11-28 08:32) runs a bus between Duluth and Superior, one town split by a zone line in
the game, and gets four different answers: "Depending on what I look at, the game shows very different data",
ending with "I think some of this is down to confusion in terms, but there might be some inconsistencies in what
does vs doesn't get counted as intercity by different game systems". He guessed that the last panel counts only
trips of 25 km or more. That guess is right.

The game has three kinds of trips between two zones: commutes, everyday trips (errands, visits, school, medical,
airports) and the "Intercity trips" purpose, which is only visits, business and leisure trips of 25 km or more.
Superior is in the zone the game calls Wausau, so almost everything on his bus is the first two kinds. Each
panel counts a different mix of them:

| Screenshot | Figure | What it counts |
|---|---|---|
| 1, the bus on the bridge | 6,400 riders a day | everyone on the bus over that stretch, both ways, every kind of trip |
| 2, his Duluth bubbles selected, then a spot in Superior | 1k commute legs (53% transit) and 5k everyday legs (49%), so about 3k on the bus | only people who live in the selected bubbles, their legs there and back; people from Superior coming over are not in it, so the bus's 6,400 is roughly this doubled |
| 3, zoomed out, Duluth selected, hover on Wausau | 17k trips, about a tenth by transit | the whole Duluth zone with the whole Wausau zone, both ways: intercity trips (5.5k) plus commute legs (the other 11.5k or so). Everyday trips across the line are left out, and they are most of his riders, so the transit part looks far too small |
| 4, a click on that bubble | 5,525 intercity trips, 312 km apart, none by transit, "There are no stations near either place" | only the intercity purpose, from Duluth's centre to Wausau's centre |

So three of the four differences come from counting different things under one word, and one is a bug:

- **Bug: a bus-only network counts as "no station".** Between two whole places, the trip explorer first checks
  whether any station is within about 40 km of either place. Plain bus stops are not stations in the save, so
  with only buses it decides no transit is in play and answers from a quick table: intercity trips only, no
  commutes, no everyday trips, no transit, and "There are no stations near either place". With a train station
  there it would have listed all three kinds and come out near the hover's figure. A cheap fix (count bus stops
  too).
- **The zoomed-out hover between two zones leaves out everyday trips**, while the explorer it opens counts them.
  Since everyday trips across zone lines arrived (11-17), "Click to explore these trips" opens a different total
  from the one on the hover.
- **"312 km apart"** is right by the game's rules: zones whose centres are over 250 km apart pair place to place,
  centre to centre, and the zone holding Superior has its centre in Wausau. This is the same family as qiallo's
  Guilin and Lingui and Edison's Macau and Zhuhai reports: a zone line through one built-up area.
- **A lone "Car 100%" row with no heading** at the top of screenshot 2 is the intercity part between those
  bubbles and that spot, under half a trip a day. The panel drops the heading under half a trip but keeps the bar.
  Small display bug.

## b. What a long-distance trip is in the game

- **Who makes one.** Every resident: about 0.107 one-way trips a weekday at the US's income per head, so about
  one every nine weekdays, fewer in poorer countries (income per head to the power 0.7). On top of that a thin
  stream of far trips, more of them from dense places.
- **How far.** None under 25 km, phased in from 25 to 50 km. The chance of going somewhere falls with distance to
  the power -2.37; between 300 and 700 km that hands over to a much flatter tail (power -0.7), mostly between big
  places.
- **Where to.** Anywhere in the world, by people and jobs. Trips to places with no zone leave the map: counted,
  never routed. Crossing a border cuts them (to half inside the EU and UK area, much more elsewhere). Big cities
  draw extra past 300 km (not for people from the US, Japan, the UK, Korea, Russia or Turkey), and Japanese resort
  towns draw their tourists.
- **How they go.** The player's network against the car (fuel, tolls, parking, two people sharing) and, past
  250 km between places that both have airports, the plane. No walking. A dollar counts for less than on a local
  trip, people are split into three income groups, and the first wait is a planned one, 30 minutes at most.
- **When.** Mostly by day, more at weekends, twice a weekday's on the days before and after a long holiday.
- **How it is counted and shown.** One-way trips a day. Between 6 km areas when two places are under 250 km apart,
  place to place beyond that. It shows as the "Intercity trips" purpose in the purpose filter, in the bubbles at
  each end, in the trip explorer and in the city panel's "To other places". A long trip between two parts of one
  big zone counts in that zone's "Trips within".
- **How it differs from the other purposes.** The others are named by where people go (a job, a school, shops, a
  doctor, an airport, an attraction) and are made inside each zone from that country's travel survey, with short
  reaches (the chance halves every 0.7 to 14 km). Commutes are one rule at every distance: a 100 km commute is
  still a commute. "Intercity" is the only purpose named by how far it goes. It is made by one world rule fitted
  on the US 2017 household travel survey's long trips, and it has its own way of picking a mode.

## c. Why it is a purpose and not a trait

**The reason on record.** The intercity review before T-276 (research/intercity-review.md, 2026-10-13; you picked
its option 2) said the layers should split the work "by purpose, not by distance or pack": commutes get one rule
at every distance, and everything else is a local outing up to the onset and a market trip beyond it. So the
purpose is really "everything that isn't a commute, past 25-50 km". Its option 3 split the long trips into
business, leisure and visiting friends, each with its own pull, timing and value of time. It was set aside as
"large, mostly design and tuning" with "the most numbers to explain in the panels", and the review called the
business and leisure split "mostly flavour for the game". DECISIONS 11-13 has your "intercity demand is rough on
purpose".

**What the model gains.**

- The everyday rules can't make long trips. They are built inside one zone's map with reaches of a few km, and a
  300 km trip ends on another zone's map or off the map entirely. Long trips need their own curve with a long
  tail, the whole world as destinations, and income and border effects.
- They choose differently: the plane, a car with running costs shared by two, money counting less, a planned
  wait. With one bucket, those rules have one place to live.
- The data fits one bucket. The fit is by distance (US survey, flights, trips abroad) and the checks are city-pair
  rail counts, none of them by purpose. There is no worldwide split of long trips by purpose to fit to.
- (Being able to switch them off on phones, and loading them only when needed, comes from the separate layer, not
  from the purpose. That would stay either way.)

**What it costs.**

- "Intercity" means two things. As a purpose it is the 25 km+ trips. Elsewhere it means trips crossing a zone
  line in general: the zoomed-out hover ("intercity trips to and from Duluth"), the city panel, and the setting
  "Calculate intercity trips", which also switches off 5 km commutes across a zone line. The "Grounded"
  achievement ("trips to other places", which players call the intercity one) counts yet another mix: long trips
  plus commutes between zones, no everyday trips.
- A zone line through one town makes short trips "intercity" in one view and not in another (moonraker, and the
  Guilin and Macau reports).
- The purpose list mixes where people go with how far: the same visit to family is Leisure at 15 km and Intercity
  at 60 km.
- Panels count different mixes, as in the table above.

**What making it a trait would change, two sizes.**

1. **Keep the model, fix the words and counts** (what I'd do). One meaning per word: the purpose could become
   "Long trips" (hover "Visits, business and leisure trips of 25 km or more"), and "intercity" or "to other
   places" would mean everything between two zones. The zoomed-out hover would break its figure into long trips,
   commutes and everyday trips, and count the everyday ones. The bus-only bug and the headless bar get fixed.
   Roughly 6-10 hours of agent time, plus your review of the strings.
2. **Make long-distance a real trait of every purpose.** Split the long trips into business, visiting family and
   friends, and leisure. Fold them into the purpose list (business into Work or a new Business purpose, the rest
   into Leisure), and add "25 km and more" as a filter that works on every purpose, commutes included. To mean
   anything, each part needs its own pull (jobs for business, people for visits), timing and value of time, from
   sources that exist mainly for the US. Without that the split is just labels nothing checks. Each country's
   everyday survey rates would have to be squared with the long-trip rates so no trip is counted twice. Every
   zone's market file gains numbers per purpose (a world data rebuild), the solve carries more purposes (memory),
   and the panels, goals, translations and checks follow. It adds a complicating factor to demand, so it waits for
   your word. My rough guess is 20-30 hours of agent time, about the size of the everyday-trips-across-zone-lines
   work (T-1163, three stages) plus a world rebuild, mostly fitting and checks.

My read: the purpose is a sound modelling choice. The confusion comes from the name and from panels counting
different mixes. Size 1 fixes what moonraker hit; size 2 is a design question for later, if ever.

## d. Draft reply to moonraker

Her voice; she decides whether to send it. It promises the bug fix and a wording cleanup: drop the last line if
you'd rather not promise.

> thanks, great writeup, you're right that it's confusing. part naming, part a real bug
>
> trips between zones come in three kinds: commutes, everyday trips (errands, visits, school) and intercity trips, which in the game means only visits, business and leisure trips of 25 km or more. superior is in the wausau zone, so your bus mostly carries the first two
>
> the 6,400 is everyone on the bus there, both ways. the bubble selection only counts people living in your bubbles, there and back, so people coming over from superior roughly double your ~3k. the 17k between duluth and wausau is commutes plus intercity trips. it leaves out everyday trips, so its transit bar is too small
>
> clicking it should list all three, but it only showed intercity trips and said there were no stations. that's the bug: it only looks for train stations, so a bus-only network counts as nothing. the 312 km is measured from wausau's centre
>
> i'll get the bug fixed and clean up where the game says "intercity", since right now it means two different things

## Follow-ups, for the orchestrator to file if she agrees

1. Bus-only networks read as "no station" in the trip explorer between two whole places: count bus stops in the
   station check. About 1-2 hours. (P2: it hides transit and two kinds of trips.)
2. The zoomed-out hover between two zones leaves out everyday trips across the line, so it disagrees with the
   explorer it opens; count them, and break the hover into its parts. About 2-4 hours. Related: T-298 (P4,
   "Inspectors mix commutes into longer trips").
3. The trip explorer shows the intercity bar without its heading when there is under half a trip. Under an hour.
4. A wording pass on "intercity" (size 1 above): one meaning per word, strings listed for her review.

## Side note for agents

docs/regional.md 5.1 ("Pairs") says local outings taper over 40-80 km so the two layers meet with no gap or double
count, and docs/zones.md 5 says everyday outings taper "by the complement of the market's onset". In the code the
market's onset is 25-50 km (`ONSET_KM` 25 in sim/src/demand/intercity.rs) and the everyday taper is still
40-80 km (`local_share` in sim/src/demand/everyday.rs), so the two overlap between 25 and 80 km. Everyday outings
are nearly gone by 25 km (the slowest, into-town errands, halve every 4.2 km), so the effect should be tiny, but
the docs and the code comment are out of step. Not measured.

Code behind section a: the explorer's station check is `stationNear` and `rowsAnswer` in
app/src/game/marketRule.ts, used by `exploreMarket` in app/src/map/tripView.ts (bus stops are never in
`save.stations`, game/types.ts); the zoomed-out pair figure is `zoneRowOf` in app/src/game/corridorFigures.ts
(corridors only: long trips plus commute legs); the headless bar is in app/src/ui/inspector/MarketTrip.tsx
(the heading needs `all >= 0.5`, the bar only `all > 0`).
