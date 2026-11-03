# What players love and hate in games like trainsworld

Research for T-2074, 2026-11-03. What players say about comparable transit and rail games, and
ideas for trainsworld drawn from it. Mechanics of Subway Builder and NIMBY Rails are in
`research/competitors.md` and `research/nimby-rails.md` and are not repeated here.

Tags: **[DEV]** the developer said it (site, changelog, devblog, postmortem). **[PLAYER]** players
said it (Steam reviews and threads, forums). **[PLAYER, summary]** a review-summary site's grouping
of many Steam reviews (VaporLens, GameBrain): useful for which complaints are common, but the
wording is the site's, and its numbers are its own mention counts or scores, not percentages of
players. **[CRITIC]** a press review. **[GUESS]** my inference.

**Method and gaps.** About 33 web searches and 40 page fetches. Reddit could not be fetched at all
(the fetch tool refuses reddit.com and old.reddit.com) and search returned no reddit threads, so
there are no reddit claims here; Steam discussions stand in for them. Behind a wall or erroring:
railfans.ca's NIMBY review (403), Greater Greater Washington's Subway Builder article (403),
Twinfinite's A-Train review (403), the Paradox forum thread on CS2's patch 1.1.12f1 (500). VaporLens
has no summary yet for Mini Metro, NIMBY Rails or A-Train Tourism (it needs a login to queue one).
"Infra" turned up nothing transit-related and is left out.

---

## 1. Game by game

### Subway Builder (2025 web, Steam July 2026)

Loved:
- **Fun and addictive** (score 0.31), **real data driving riders** (0.19), **detailed data and
  analysis** (0.13), **easy to learn with a short tutorial** (0.13) [PLAYER, summary].
  https://vaporlens.app/app/4039140/subway_builder.md
- Seeing demand respond: "It is absolutely thrilling to build a new rail line and start to see
  those red bubbles change colors." "cars are the final boss of Subway Builder." [CRITIC]
  https://punishedbacklog.com/subway-builder-review/
- Blueprints and the automatic route map are listed among the positives (0.10) [PLAYER, summary],
  same VaporLens page.

Complaints:
- **Always online** (0.34, the top complaint), **performance** (0.26, worse as lines are built),
  **poor UI** (0.25: cluttered, no scaling, no shortcuts), **price** (0.22), **tutorial** (0.12)
  [PLAYER, summary], VaporLens above.
- **Demand "seems randomized and disconnected from city logic"**; **economy**: "hard to reach
  profitability, even with experience in similar games" [PLAYER, summary], VaporLens above.
- **Line editing**: adding a station can mean recreating the line (0.06); no track upgrades
  without demolishing (0.06) [PLAYER, summary].
- "The Demand Stats ... are extremely helpful, but they vanish as soon as you start building." The
  tutorial: "only moderately helpful; I was still a bit confused by the end of it." [CRITIC]
  punishedbacklog above.
- "Simulation speed that feels too slow" is among the early friction points [PLAYER, summary].

Why people stop: refunds over bugs, always-online and polish; "refunded after 30 mins"
[PLAYER, summary]. Players who stayed "became engaged after finishing the tutorial or watching a
YouTube walkthrough" [PLAYER, summary].

What the developer added after launch, which says what players asked for [DEV]
(https://www.subwaybuilder.com/changelog): bonds and loans for the early game (v0.8); ridership by
hour charts (v0.10); **daily challenges with a leaderboard** (v1.3) and a second daily mode,
**transit share** (v1.4); an annotations tool to sketch before building (v1.3); a route loads map
tool (v1.3); **a tutorial that teaches demand, "Tutorial steps tell you why, not just which button
to press"** (v1.4.17); achievements view (v1.4.5); new games start paused (v1.4.17); unconnected
track marked on the map and route errors that name the stations (v1.4.17, v1.6); duplicate route
(v1.6); distance and zone fares (v1.6, v1.7); a measure tool showing travel time previews for each
train type (v1.7); a lifetime stats view (v1.7); Cmd+K search through menus (v1.7.2).

Player suggestion threads on Steam (https://steamcommunity.com/app/4039140/discussions/): induced
demand around new stations ("usually a new station is built alongside a bunch of new urban
development",
https://steamcommunity.com/app/4039140/discussions/0/588433527532489346/), a suspend service
button, a screenshot key, a light theme, an offline mode (25 replies, the busiest) [PLAYER].

### NIMBY Rails (early access since 2021)

Loved:
- Building anywhere on the real world map; the "YIMBY fantasy" of no opposition [CRITIC] (Vice,
  via https://www.vice.com/en/article/this-game-lets-you-build-the-transit-system-of-your-dreams-in-the-real-world).
- "The core gameplay blends creative freedom with management strategy"; "Replay value is high";
  "Funny station names" from OSM [PLAYER, summary]. https://gamebrain.co/game/nimby-rails

Complaints:
- "Steep learning curve with unintuitive UI/UX", "Everything is unnecessarily tedious", "There
  isn't a tutorial at the moment or story missions" (no missions, achievements or guided goals),
  "It has very simple graphics and no soundtrack" [PLAYER, summary], GameBrain above.
- **Complexity added by updates drove a veteran off.** "A tutorial system is becoming a necessity"
  (400+ hours): "the added complexity with each update is dampening, rather than enhancing the
  experience"; another stopped because relearning after updates was too much [PLAYER].
  https://steamcommunity.com/app/1134710/discussions/0/3379410661802232139
- **Timetables**: one player said a single line's timetable took over an hour plus another to fix
  [PLAYER, search summary of a Steam thread]; the developer agreed timetables were "buried in a tab
  in the train editor" and proposed a top-level editor [DEV] https://carloscarrasco.com/page/13/
- **Slow money**: "Treat it like an Idle game"; "the game needs some more non-building gameplay in
  it to offset the long build up of cash" (Zam!) [PLAYER].
  https://steamcommunity.com/app/1134710/discussions/0/3109142047898036098
- A "no cash" mode was "one of top requested features" in 2020 [DEV].
  https://carloscarrasco.com/page/58/
- The 1.12 demand rewrite cut riders 4-20x in existing saves (in `research/nimby-rails.md`).

### Mini Metro and Mini Motorways (Dinosaur Polo Club)

Loved:
- Mini Motorways: **addictive and replayable** (99 mentions), **relaxing yet engaging** (83),
  **polished minimal look** (41), "strategic depth in simplicity" (22) [PLAYER, summary].
  https://vaporlens.app/app/1127500
- Mini Metro's daily challenge: "something I can do to relax but which still requires some brain
  power"; the procedurally generated music [CRITIC/blog].
  https://www.gamedeveloper.com/design/let-s-talk-about-mini-metro
- The studio's own account: the free **web-playable build** is why the studio survived; the game
  "communicates the idea of itself very well" because it looks like a transit map; "a game
  defined by its scope" [DEV].
  https://gamedeveloper.com/audio/postmortem-dinosaur-polo-club-s-i-mini-metro-i-
- **Wuselfaktor**, the pleasure of watching many small agents: "Just watching the cars leave from a
  house and go somewhere else was pretty satisfying"; "it should be incredibly easy to start
  playing" [DEV]. https://www.thumbsticks.com/hustle-and-bustle-how-mini-motorways-built-its-wuselfaktor/

Complaints:
- Mini Motorways: over-reliance on randomness (30), clumsy road tools with "accidental deletions
  and limited undo" (27), buildings spawning in "illogical ... or unfair locations" (24), "lacks
  depth compared to ... Mini Metro" (20) [PLAYER, summary], VaporLens above.
- Mini Metro: stations "too random and not arranged the way a city would be, with denser centres
  and sparser outskirts"; too many circles [PLAYER].
  https://steamcommunity.com/app/287980/discussions/0/1473095331494774723
- A transit planner's view: it teaches the frequency against coverage trade-off well, but free
  demolition and rebuilding is "a major unreality" (Aaron Priven, comment) [CRITIC/planner].
  https://humantransit.org/2014/03/the-lessons-of-mini-metro.html

Modes added on request: Endless ("stations never overcrowd") on mobile after launch, and a
"much-requested" Creative mode for Mini Motorways in 2025 [DEV].
https://nzgda.com/news/mini-motorways-celebrates-nine-million-players-as-dinosaur-polo-club-launches-new-creative-mode

### Transport Fever 2

Loved: modding (74 mentions), replayability (53), the campaign as a tutorial (23), and
"transport-focused city building ... indirectly influence city growth" (14) [PLAYER, summary].
https://vaporlens.app/app/1066780

Complaints:
- Campaign and tutorial "unclear, frustrating, and lacking proper guidance" (48), cluttered UI
  (40), routing and AI "unrealistic, inconsistent" (37), performance and late-game slowdown (35,
  21) [PLAYER, summary], VaporLens above.
- **Money stops mattering.** "plays as if we just used a cheat for unlimited money"; "there's no
  information about which variables determine the payment for completed transport" [CRITIC].
  https://www.gamepressure.com/editorials/reviews/transport-fever-2-review-good-sim-poor-tycoon/z520c
  "Very pretty, too easy" [CRITIC] https://www.dedoimedo.com/games/transport-fever-2.html
  Players on Steam: running costs "simply too low", "way too easy" even on hard [PLAYER].
  https://steamcommunity.com/app/1066780/discussions/0/4625853071149245248
- New lines "take several months" before people use them; a city bus feeder filled a station
  "almost immediately" [PLAYER].
  https://steamcommunity.com/app/1066780/discussions/0/3418808280402543203

### OpenTTD

Loved: "fun and fulfilling" (56), mods (43), runs on anything (28), multiplayer (18), "homely"
graphics (15). 94% positive [PLAYER, summary]. https://vaporlens.app/app/1536610
Cargodist (destinations for passengers) is "one of the key requirements for me in a transport
game" [PLAYER] https://www.tt-forums.net/viewtopic.php?p=773593 ; its link graph overlay shades
links short of capacity [DEV] https://wiki.openttd.org/en/Manual/Passenger%20and%20cargo%20distribution

Complaints: **lack of in-game explanations, "forcing players to consult external resources like
wikis"** (26), cluttered interface (22), overwhelming at first (21). About 2 hours to get hooked
[PLAYER, summary], VaporLens above. Cargodist's slow reaction to changes confused players; the
developers explained the link graph's time budget is set for the worst case [DEV, via
tt-forums search results].

### Cities: Skylines II (transit side)

Players' top complaints are crashes, late-game performance and a sequel thinner than the first
game; public transit is not singled out in the review summary [PLAYER, summary].
https://vaporlens.app/app/949230 On the forums: stops where some waiting citizens never board;
trams and trains that stop leaving stations; and after patch 1.1.12f1 "public transport usage was
effectively cut in half" [PLAYER, search summaries of
https://admin-forum.paradoxplaza.com/forum/threads/patch-notes-1-1-12f1.1716824/page-6 and
https://steamcommunity.com/app/949230/discussions/0/3937895635388858248]. Thin evidence: I could
not open the Paradox thread.

### Railway Empire 2

Loved: a big step up from the first game (78), engaging and addictive (69), streamlined track and
automatic signals (36), large maps (29); "steep learning curve, good tutorial" (15) [PLAYER,
summary]. Complaints: bugs and crashes (94), regression in polish (89), **clunky parallel track and
junction building** (73), train AI and signals (44), DLC (30). Some players want a hard mode with
less starting money; sandbox has infinite funds [PLAYER, summary].
https://vaporlens.app/app/1644320

### A-Train (All Aboard! Tourism, A-Train 9)

Critics praise depth and towns growing around stations ("people gather around stations, gradually
developing the surrounding town") but find it "just about as complex as management sims get ...
far too complicated for your average player" and its interface "far from user-friendly"; OpenCritic
68, 40% recommend [CRITIC, via search summaries]. https://opencritic.com/game/11053/a-train-all-aboard-tourism/reviews
Twinfinite's headline: "a deep railway management sim dragged down by poor UI" (page 403).
A-Train 9's store copy sells "the city begins to grow organically around your network" [DEV].
No player review text could be read.

### Rail Route

Loved: fun and relaxing (30), many maps (15), addictive (14), Endless mode's "zen" (12) [PLAYER,
summary]. Complaints: **tutorial "slow, confusing, condescending, and incomplete"** (28), cluttered
UI (25), steep curve (14), unclear scheduling (14). Why people stop: tutorial contracts that cannot
be finished (soft locks), tedious early manual work, clutter [PLAYER, summary].
https://vaporlens.app/app/1124180
The developers: "Most of the negative feedback we received after we entered the Early Access period
was regarding the tutorial"; they rebuilt it and watched "randomly selected players (that have never
played our game before) play through the tutorial whilst thinking out loud", which "was super
helpful" [DEV, interview quotes as given by search; the pages are
https://intoindiegames.com/reviews/previews/rail-route-interview-build-a-rail-network-and-become-a-train-dispatcher/
and https://railadvent.co.uk/?p=142324]. The 2022 tutorial rework is
https://bitrich.itch.io/railroute/devlog/113845/rail-route-the-levels-update

### Newer and smaller games (2023-2026)

- **Station to Station** (2023): praised as "a relaxing, cozy railway planner with clear goals";
  criticised as short (8-10 hours) and light on content [CRITIC].
  https://opencritic.com/game/15595/station-to-station/reviews
- **National Transit** (early access June 2026, 95% of 69 reviews positive): a nation's road,
  rail, water and air on procedural maps, no fail state, and "a history feature for revisiting past
  months or branching new timelines" [DEV]. https://store.steampowered.com/app/4606620
- **Metro Architect** (early access July 2026, 70% of 17): metro networks on twelve real-inspired
  cities, three playstyles from casual to rush-hour pressure [DEV].
  https://store.steampowered.com/app/4733420
- **Metro Dreamin'** (web): a fantasy transit map site where every shared map shows its ridership,
  cost, stations and lines, with comments and stars. https://metrodreamin.com
- **Brand New Subway** (web, 2016): redesign the New York subway from scratch or from real maps of
  several eras; goals are lower fares, more riders, less crowding.
  https://viewing.nyc/this-game-lets-you-redesign-the-new-york-city-subway-from-scratch
- No 2025-2026 browser game that builds rail on a real world map turned up besides trainsworld.

---

## 2. What repeats across games

1. **Tutorials are the most common complaint after performance**: Transport Fever 2 (48), Rail
   Route (28, and why people quit), OpenTTD (26, "consult ... wikis"), Subway Builder (0.12), NIMBY
   (none at all). The fixes that worked: Subway Builder's "why, not which button", Rail Route's
   think-aloud tests.
2. **Clear goals are praised where they exist** (Station to Station, Mini Metro's daily, Subway
   Builder's dailies) and missed where they don't (NIMBY: "set their own goals").
3. **Seeing demand move is the hook**: bubbles changing colour, cars leaving a house, "cars are
   the final boss".
4. **Opaque demand and money breed distrust**: "seems randomized", "no information about which
   variables determine the payment".
5. **Money pacing fails at both ends**: too slow early (NIMBY idle), meaningless late (TF2).
6. **Complexity creep and cluttered UI** follow every deep game (NIMBY, A-Train, OpenTTD, TF2,
   Rail Route).
7. **Performance that worsens as the network grows** is the top complaint of Subway Builder, CS2,
   Railway Empire 2 and TF2's late game.
8. **A demand change that guts existing networks** (NIMBY 1.12, CS2 1.1.12f1) is remembered.
9. **Relaxed modes are asked for and kept** (NIMBY no-cash, Mini Metro Endless, Mini Motorways
   Creative, Rail Route Endless).
10. **Showing off** is part of the hobby: Metro Dreamin' exists for it; Subway Builder players ask
    for a screenshot key; Mini Metro spread through a free web build.

---

## 3. Ideas for trainsworld, ranked by expected value to a player

Each: what it is; the evidence; whether trainsworld has it (from SPEC.md and docs/ui-*.md); fit
with Anita's rules (CLAUDE.md "Working here"); size. None is filed as a task.

### 1. A second tutorial that teaches demand
After the six button steps, a second short card that teaches reading the game: find the red
bubbles, open one trip in the trip explorer, add trains to cut the wait, reach 5% train share.
- Evidence: Subway Builder 1.4.17 "Tutorial steps tell you why, not just which button to press"
  and "teaches demand" [DEV]; tutorials are the top non-technical complaint in four games;
  OpenTTD's players learn from wikis [PLAYER, summary].
- Has it? Partly. The Tutorial card (T-1822) is six mechanical steps (track, stations,
  construct, line, stops, a train). Nothing teaches bubbles, the trip explorer, frequency or fares.
- Fit: the card is the established exception to "no explanatory text"; steps stay one short line
  with a counter, as now. The ripple can point at the demand selector and trip tool.
- Size: medium.

### 2. A ladder of goals per city, and a few for the world
Beyond the 10-40% share ticks: a handful of goals per city (serve the airport, 100k trips a day,
profit for 7 days) and world ones only trainsworld can offer (cross a border, link two cities
300 km apart, run trains in three continents). Ticks in the city view; world goals in the Cities
tab's World row.
- Evidence: NIMBY's lack of goals is a standing criticism ("players must set their own goals")
  [PLAYER, summary]; Station to Station praised for "clear goals" [CRITIC]; Subway Builder added
  achievements [DEV]; TF2's campaign praised as a tutorial [PLAYER, summary].
- Has it? Partly: the share goals and the star (T-685, T-1437); points that "unlock nothing yet".
  T-685's city stamps are open.
- Fit: goals are states that change, so they belong in panels. Keep each a plain count.
- Size: medium.

### 3. "What this edit did" for every edit
When an edit's final result lands, the passing note shows the city's change too ("Train share
12.4% to 13.1%", "+38k trips a day"), and the bubbles whose colour moved most pulse once.
- Evidence: "thrilling ... to see those red bubbles change colors" [CRITIC]; Wuselfaktor [DEV];
  NIMBY's players disliked demand that ignores the network [PLAYER].
- Has it? Partly: the daily rides note after a line's stops change (T-1478) and the opening day
  count-up (T-1434). Frequency, fare, track and station edits get no note; bubbles change in one
  step with nothing marking which moved.
- Fit: a passing note is already the pattern; motion allowed where it helps (2026-10-29).
- Size: small for the note; medium with the bubble pulse.

### 4. One plain reason for a quiet line
For a line well below what its trains could carry, a Trouble list row or a line panel line names
the biggest reason from the solve: "most trips along it drive: rail is 2.1x slower door to door",
"waits average 14 min", "fares above the car's cost".
- Evidence: "seems randomized" (Subway Builder) [PLAYER, summary]; "I would have settled for a
  simpler demand scheme" (NIMBY) [PLAYER]; TF2 "no information about which variables determine the
  payment" [CRITIC]; OpenTTD's "lack of in-game explanations" (26) [PLAYER, summary].
- Has it? Partly: the trip explorer answers it for one pair of cells, and hovers carry formulas;
  the Trouble list covers crowding and capacity, not empty lines.
- Fit: a row is a state, so it fits "panels show what changes"; the reason comes from figures the
  solve already has (perceived minutes per part). Nothing new in demand.
- Size: medium.

### 5. A weekly challenge with a leaderboard
One city and rule set a week for everyone (a start budget, 30 game days, scored by train share or
points), rotating round the world.
- Evidence: Subway Builder added daily challenges and a transit share mode [DEV]; Mini Metro's
  daily is the mode a reviewer still plays [CRITIC]; Mini Motorways has daily and weekly
  [DEV, via search].
- Has it? Partly: leaderboards per city by points for signed-in players (T-972, T-973). No shared
  start or end, so boards compare games of any age.
- Fit: rotation shows off the whole world; the rules are a few fixed numbers. Needs shonei-server
  support for a challenge id.
- Size: medium to large.

### 6. Riders estimate for blueprints before paying
In the blueprint total, an estimate for the blueprint's lines: riders a day and profit a day if
built, worked out as the fare estimate is.
- Evidence: Subway Builder's players find profit hard to reach [PLAYER, summary], and it added
  a measure tool with travel times and an annotations tool for planning [DEV]; Mini Metro's
  unreality is free rebuilding, so mistakes should cost, which makes a forecast worth more
  [CRITIC/planner].
- Has it? No, as far as the docs say: blueprints show their price, demand counts built stations
  only. The fare graph has an estimate (T-1300) that is the model for this. Worth checking what a
  blueprint line does in demand today before designing.
- Fit: it is the forecast kind of figure, the same as the demand views; label it as an estimate.
  Costly in CPU: a trial solve per blueprint change, so likely only on request.
- Size: large.

### 7. Share a picture, then a link
A button that saves a clean picture of the map in view (no panels; the city name, train share and
trips a day small in a corner), then T-1446's read-only link.
- Evidence: Metro Dreamin' is built round sharing maps with their ridership; Subway Builder players
  asked for a screenshot key [PLAYER]; Mini Metro's free web build spread the game [DEV].
- Has it? No picture; the link is T-1446 (P3, open). F2 hides the interface, so players can screenshot by hand.
- Fit: small and self-contained; the link fits saves being inputs only.
- Size: small for the picture, medium for the link.

### 8. Your network against the real one
Per city, the train share the real network gets in trainsworld's own model, drawn as a mark on the
city's share bar ("Real network: 23%"), so a player redrawing their own city has a fair target.
- Evidence: Subway Builder's audience is people redrawing their own city (the GGWash piece's
  title, "turns Metro enthusiasts into transit designers",
  https://ggwash.org/view/101667/subway-builder-turns-metro-enthusiasts-into-transit-designers;
  the page itself was 403) [CRITIC]; Brand New Subway started from
  real maps; trainsworld's own sandbox starts from real networks in seven cities.
- Has it? The pieces: the real network layer, `realnet.py`, the `calib` bin that solves a real
  network through the game's demand. No figure is shown to players.
- Fit: general sources (operators' GTFS, else OSM routes), and the target comes from the same
  rules, so no fitted constants. Only cities with a real network built; others show none.
- Size: medium.

### 9. A choice of start money
At New game: less money, the standard $6B, or sandbox. Leaderboards count the standard only.
- Evidence: Railway Empire 2 players want a mode with less starting money [PLAYER, summary]; TF2
  "too easy" even on hard [CRITIC, PLAYER]; NIMBY's "no cash" mode was a top request [DEV].
- Has it? Normal ($6B) and sandbox.
- Fit: one number; simple.
- Size: small.

### 10. Faster speeds that are easy to find
Make the speeds past 16x findable (the slider is behind a double-click), and check how long the
4x cap during recomputes holds on big games.
- Evidence: "simulation speed that feels too slow" [PLAYER, summary, Subway Builder]; NIMBY
  players run Cheat Engine for speed and the developer planned "10000x" [PLAYER].
- Has it? Yes, hidden: 0.5x-100x on double-click of 16x (T-1664); the cap (T-1665) with quiet
  edits exempt (T-1803).
- Fit: no new text beyond a hover.
- Size: small.

### 11. Watch a newcomer play, thinking aloud
Before each round of UI work, one person who has never played starts a game and talks through it;
the agents read the notes.
- Evidence: Rail Route's developers rebuilt their tutorial this way and keep doing it [DEV].
- Has it? No; Anita playtests, and she knows the game.
- Fit: a practice, not a feature. Anita picks the person.
- Size: small.

### 12. Long history and lifetime totals
Finances history past 30 days (weekly columns beyond a month), and a few lifetime totals: trips
carried, km of track, cities served, the date each city's first line opened.
- Evidence: Subway Builder added a lifetime stats view (v1.7) and a 48-hour finance view [DEV];
  National Transit sells revisiting past months [DEV].
- Has it? 30 days and 48 hours of charts (T-447).
- Fit: figures that change; no forecast.
- Size: small.

### 13. Follow one rider's trip
Click a rider at a platform or on a train to follow them door to door: the walk, each train, the
change, the arrival, with the camera along.
- Evidence: Wuselfaktor quotes [DEV]; CS's line panel showing who waits and rides [DEV].
- Has it? Partly: trains show who is on board, Life at stations faces, the trip explorer for a
  pair of cells. Riders are flows, so a followed rider would be one drawn from a flow.
- Fit: say it is a sample. Nothing new in demand.
- Size: medium.

### 14. A diagram of the network
A schematic map of the player's lines (straight lines at 45 degrees), to look at and to save as a
picture.
- Evidence: Subway Builder's automatic route map among its positives (0.10) [PLAYER, summary];
  Mini Metro's success rides on looking like a transit map [DEV]; Metro Dreamin'.
- Has it? No. The line panel has a per-line diagram (not to scale).
- Fit: a view, no rules.
- Size: large (schematic layout is a hard problem; published algorithms exist).

### 15. Towns that grow round good stations
Homes and jobs near well-served stations grow slowly over game months up to a cap.
- Evidence: A-Train's whole loop [DEV]; TF2's "indirectly influence city growth" praised (14)
  [PLAYER, summary]; Subway Builder suggestion thread [PLAYER]; Mini Metro's growing demand keeps a
  game moving [DEV].
- Has it? No: demand follows the census.
- Fit: **against "demand is simple on purpose"** unless kept to one rule (a share a month, a cap)
  and proposed to Anita first; it also moves figures away from real data, and makes saves depend
  on their history.
- Size: large.

### 16. Music that follows the network
A quiet background that grows with the network, built from the arrival chimes already there.
- Evidence: Mini Metro's procedural music praised [CRITIC, DEV]; NIMBY "no soundtrack" a
  complaint [PLAYER, summary].
- Has it? Partly: per-line arrival notes on 12 semitones, build and construct sounds, load
  chime; train sounds are T-1776.
- Fit: under the existing Sound switch.
- Size: medium.

### 17. "Where next" on the world view
Zoomed out, the zones ranked by the trips a player could win (big car flows with no rail), so a
player looking at the whole world has a starting point.
- Evidence: NIMBY's build-anywhere freedom praised, its lack of direction criticised [PLAYER,
  summary]; Mini Metro's pressure tells a player where to act [DEV].
- Has it? Partly: one bubble per zone zoomed far out, the biggest car flows as arcs (T-1435), the
  Cities tab.
- Fit: figures the market already has.
- Size: medium.

### 18. Check that money stays meaningful in a long game
Play (or script) a long game and chart cash against what there is to spend it on; if cash runs
away, the answer may be goals that cost (idea 2) or dearer far cities, not new rules.
- Evidence: TF2 [CRITIC, PLAYER]; NIMBY's idle start [PLAYER].
- Has it? Upkeep (T-867), loans (T-2022), refusals by profit. No measurement of a long game seen.
- Fit: measurement first.
- Size: small to measure.

---

## 4. Pitfalls those games fell into

1. **A demand change that guts existing games.** NIMBY 1.12 took players from about 100k to 20k
   riders; CS2's 1.1.12f1 "cut in half" transit use. trainsworld retunes demand often. Only Anita
   has saves now, but once players do, a retune will move their numbers overnight. Possible
   guards: measure every retune on a big save's line totals, and tell a loaded game when its
   figures moved because the data changed, not because of the player. **trainsworld is already at
   risk here.**
2. **Complexity creep.** NIMBY's 400-hour player: "the added complexity with each update is
   dampening, rather than enhancing the experience". A-Train, OpenTTD, TF2 and Rail Route all
   carry "cluttered". trainsworld's panels are dense (the line panel, Finances, Settings), and new
   rows land most days. A periodic pass asking "would a new player miss this row?" (idea 11)
   is the cheap guard. **Already at risk.**
3. **Tutorials that teach buttons, or that soft-lock.** Rail Route's uncompletable contracts;
   Subway Builder's reviewer "still a bit confused by the end". trainsworld's card ticks on any
   track and any line, so it cannot lock, but it teaches buttons only (idea 1).
4. **Performance that sinks as the network grows** (Subway Builder, CS2, TF2, Railway Empire 2).
   trainsworld's first priority is exactly this; keep testing the biggest saves, as now.
5. **Money that stops mattering, or grinds.** See idea 18.
6. **Online walls.** Subway Builder's always-online requirement is its top complaint (0.34).
   trainsworld runs in a browser and loads packs from the network, so it is online by nature, but
   its account is optional. Keep it that way, and let a running game carry on through a dropped
   connection where the packs are already loaded.
7. **Randomness that feels unfair** (Mini Motorways spawns, "seems randomized"). trainsworld's
   demand is real data and deterministic, a strength; the rough figures that jump when the detailed
   solve lands are the one place it could read as random.
