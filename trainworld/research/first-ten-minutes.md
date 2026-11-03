# The first ten minutes (T-2087)

Research report, 2026-11-03. What makes a new player's first ten minutes go well in rail and city
sims, what loses them, how trainsworld's start compares today, and ideas ranked for Anita to pick
from (none filed as tasks). Screenshots: `artifacts/trainworld/t2087/`.

**[DEV]** = the developer said it. **[PLAYER]** = player reports (Steam reviews, forums; the
vaporlens.app summaries are machine-made digests of Steam reviews, so read their weights as a rough
signal). **[STUDY]** = research. Pages behind a login or bot wall were skipped and are named where
it happened.

## 1. How comparable games start a new player

### Mini Metro and Mini Motorways (Dinosaur Polo Club)

- The first game is three stations and a few lines. Drag from one station to another and a train
  appears on the line at once, running back and forth: one gesture gives a running line.
  [PLAYER/review] https://www.androidcentral.com/mini-metro-retro-review,
  https://www.gamesradar.com/elegant-simplicity-and-impatient-passengers-you-should-be-playing-mini-metro
- The tutorial is nearly wordless; one reviewer says the layout is so plain a tutorial is hardly
  needed, while TouchArcade's review says the wordless tutorial does not tell you everything about
  basic functions (adding trains and carriages), and a forum player "had absolutely no idea what
  I'm doing beyond trying to link stations with lines". The repeated mouse demo for adding cars
  annoys later. [PLAYER] https://toucharcade.com/2016/12/30/mini-metro-review-train-braining/,
  https://toucharcade.com/community/threads/mini-metro-by-dinosaur-polo-club.295652/,
  https://milesintransit.com/?p=918
- The team's stated aims: rules "transparent to the player", "seemingly simple systems that are
  complex in unintuitive ways", no "insurmountable wall of difficulty at Week X". [DEV]
  https://gamedeveloper.com/audio/-i-mini-motorways-i-and-the-delicate-art-of-marrying-complexity-and-minimalism
- Mini Motorways runs an interactive lesson before the first city: colour-matched houses and
  workplaces, so the goal (join the green house to the green building) needs no words. [PLAYER/review]
  https://www.pocketgamer.com/mini-motorways/tips-for-beginners

### Subway Builder (the closest game to trainsworld)

- Normal mode starts with $3B; the brief tutorial was "only moderately helpful", the reviewer
  "still a bit confused by the end of it", and learned in sandbox first. A first line down Park
  Avenue hit Grand Central at $2.3B to clear. Demand stats "vanish as soon as you start building",
  which he found frustrating. [PLAYER/review] https://punishedbacklog.com/subway-builder-review/
- Steam digest (170 reviews, "Mixed"): "The tutorial is insufficient and not helpful, leaving new
  players confused about basic mechanics" (weight 0.12) against "easy to learn with a short
  tutorial" (0.13); "confusing track/station connections", sim speed too slow, early economy hard
  to make profitable; players get past it with YouTube or by experimenting; Honolulu suggested as
  the gentle first city. [PLAYER] https://vaporlens.app/app/4039140/subway_builder.md
- The wiki marks cities by difficulty (Honolulu "Easy": a small island needing few lines).
  https://en.wikipedia.org/wiki/Subway_Builder; the wiki's Getting Started Guide
  (https://wiki.subwaybuilder.com/index.php/Getting_Started_Guide) answered 502 when fetched.

### NIMBY Rails

- No tutorial in the game; the official wiki's quickstart is the onboarding: "NIMBY Rails is a
  sandbox game and as such it requires a high degree of self motivation to play. There is no
  campaign mode and no in-game progression prompts or unlocking of any kind." Its steps: track
  (a free blueprint, then Build now), stations, a line in the line editor, Add train, then set the
  line's Service Level to Full, because "lines start closed, and passengers only plan trips on open
  lines". [DEV/wiki] https://wiki.nimbyrails.com/Quickstart_guide
- A 400-hour player: "I'm browsing forums and guides, only picking up pieces, and can't figure
  it out"; replies: "I just couldn't wrap my head around it", YouTube tutorials are outdated, a
  volunteer "tutorial task force" suggested. [PLAYER]
  https://steamcommunity.com/app/1134710/discussions/0/3379410661802232139
- vaporlens has no digest for NIMBY Rails (its queue needs a login; skipped).

### Transport Fever 2

- The campaign doubles as the tutorial. Steam digest: "campaign and tutorial" unclear and
  frustrating is the biggest complaint (weight 0.48), "steep learning curve ... little early-game
  onboarding" (0.28), early maintenance costs; on the other side the campaign "introduces gameplay
  mechanics gradually" (0.33). [PLAYER] https://vaporlens.app/app/1066780/transport_fever_2.md
- A Steam thread: the first chapter does not say which station to use; "among the worst tutorials"
  the poster had seen. [PLAYER] https://steamcommunity.com/app/1066780/discussions/0/3037103480419247850

### OpenTTD

- No in-game tutorial; the wiki tutorial (a bus route, then a coal train set to wait for a full
  load) takes "around 30 minutes". Players steer each other to one coal mine and one power station,
  or three bus routes in the biggest town, as the first profit. [PLAYER/wiki]
  https://wiki.openttd.org/en/Manual/Tutorial/Trains,
  https://nylon.net/ttd/starting.htm

### Cities: Skylines II

- [DEV] A fixed task list first (utilities, roads, zoning), then a short tutorial each time a
  panel opens for the first time; new features get a green dot on their icon; a closed panel
  mid-tutorial gets a blinking green outline back to it; guidance "shrinks as the city grows";
  everything optional, kept in an Advisor panel (a question mark) for later.
  https://colossalorder.fi/news/behind-the-scenes-6-tutorials-advisor/
- [PLAYER] "The tutorials are not effective"; "very basic"; "definitely not learner-friendly",
  citing its "overlapping objects" notices; another found them clearer than the first game's.
  https://steamcommunity.com/app/949230/discussions/0/3877096256093655965

### Rail Route

- Praised as "one of the best tutorials ever made in game", framed as an apprentice coached by an
  old dispatcher [review] https://boilingsteam.com/rail-route-review/; elsewhere slow,
  "condescending" and incomplete, and tutorial contracts that cannot be met soft-lock the player
  [PLAYER] https://vaporlens.app/app/1124180/rail_route. Early players asked for "a couple more
  tutorial boxes popping up in the corner to get me through the first bit" and complained of
  tiny text [PLAYER] https://bitrich.itch.io/railroute/devlog/95517/how-was-your-experience-with-rail-route.
  The devs' own write-up (gamedev.net) answered 403; skipped.

### Factorio (a complex sim's tutorial, as a reference)

- [DEV] The old tutorial took "about 30-45 minutes to get to automation", the thing the game is
  about; objectives "the player does it because they are being told to"; information split over
  three places. Principles for the new one: teach the core idea in the first level with as little
  else as possible, goals from the game's own systems, experimentation over arbitrary tasks, one
  place for information. https://factorio.com/blog/post/fff-241
- [DEV] Lessons after cancelling it: players dislike being told to restart; the helper should be
  present but not do things for them; "self-discovery of new mechanics matters more than using
  them". https://direct.factorio.com/blog/post/fff-329

### Kingdoms and Castles (teaching with no tutorial)

- [DEV] Two problems at the start (homelessness, hunger), each building with one plain purpose,
  wood only until those are met, threats held back until a population or time mark, advisors for
  when the player is unsure. https://gamedeveloper.com/design/teaching-players-without-tutorials-in-i-kingdoms-and-castles-i-

### Browser games (Poki's guidance for web games)

- "For first-time players especially, skip splash screens, title screens, and level selects."
  "Reveal actions and buttons over the first few levels instead of front-loading everything."
  "Overwhelmed players end their session; eased-in players stay." "Drop-off during loading is
  brutally underestimated." [platform] https://developers.poki.com/guide/easy-access
- Poki counts "conversion to play" (visitors who reach gameplay): above 70% solid, above 80%
  exceptional. https://poki.com/blog/how-we-made-cannon-clash-load-fast-and-boosted-conversion

### Studies

- **Andersen et al., CHI 2012** (45,000+ players, three games): tutorials raised play time by up
  to 29% only in the most complex game, and did nothing measurable in the two simpler ones; their
  section headings: "context-sensitivity can improve engagement", "tutorial freedom did not affect
  player behavior", on-demand help "harmed and helped" retention. [STUDY]
  https://grail.cs.washington.edu/projects/game-abtesting/chi2012/
- **Cheung, Zimmermann and Nagappan, CHI PLAY 2014** (247 first-hour reviews, six industry
  interviews): players want clear directions and the freedom to discover, both; without directions
  they feel stupid and quit ("I don't like it when games make me feel stupid"); one missed detail
  can be a deal-breaker ("an information problem rather than a game-play issue"); it quotes Bruce
  Shelley, "a player must be actively engaged by a new game within 15 minutes of starting or we
  risk losing the player forever", and Sid Meier on rewarding the first fifteen minutes with "a
  foreshadowing of all the cool stuff". [STUDY]
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/First20Hour20-20CHIPlay20201420-20preprint2.pdf

### What the sources agree on

1. **Time to the first running thing.** Mini Metro gives a moving train for one drag; Factorio's
   complaint about its old tutorial was the 30-45 minutes before the core idea; Poki stresses the
   first couple of minutes and the load. The games most criticised (NIMBY Rails, Transport Fever 2,
   Subway Builder) need several separate steps before anything moves, and each step is a place to
   get lost (NIMBY's "lines start closed" is the classic).
2. **Teach in context, at the moment of use**, not up front (Andersen's context-sensitivity,
   Cities: Skylines II's first-open tutorials, Poki's "reveal buttons over the first levels").
3. **A visible reason for the next step.** Kingdoms and Castles' two problems, Mini Motorways'
   colours, Factorio's "goals from the game's own systems". Arbitrary tasks ("place 5 km") are what
   Factorio named as its old tutorial's weakness.
4. **The payoff has to be seen.** Players stay when the result of what they built is legible
   (Subway Builder's bubbles turning from red to mixed is the moment its reviewer remembers; the
   same reviewer's complaint is demand stats vanishing while building).
5. **A gentle first place.** Subway Builder players steer each other to Honolulu; OpenTTD players
   to one mine and one power station. The biggest, densest city is not the easiest first map.
6. **Don't block, don't nag, don't soft-lock.** Rail Route's unmeetable contracts and Mini Metro's
   repeated demo are the warnings; Factorio's "present but not doing things for them".
7. **Money is the early quitter's trap.** Subway Builder's and Transport Fever 2's digests both
   name the early economy; OpenTTD players warn not to build every time $500 arrives.

## 2. Trainsworld today, walked as a stranger

The live game (https://anita.garden/trainsworld/, build 797), headless Chrome at 1400 x 900 with a
fresh profile, so a first visit; also at 390 x 844 (a phone) and with no city picked. The script
(`scratchpad .../t2087/walk.mjs`, not kept) clicked what a new player would: the tutorial's targets,
screen points up Manhattan for track and stations. Screenshots are
`C:\Users\anita\projects\maps\artifacts\trainworld\t2087\<name>.png`; logs of the visible text
beside them (`walk-*.log`).

What went well, worth keeping:

- **Fast in.** The page opens straight on the new game sheet over the logo (no title screen, as
  Poki asks), and a New York game was playable 3 s after Start. Nothing to download first.
- **The tutorial card works.** Six ticked steps with live counters, each target rippled; a
  stranger who follows it in order gets a running line. Ticks came at once (`nyc-12-stations-placed`,
  `nyc-19-trains-added`).
- **Good previews while building.** The track ghost names its length, cost multiplier, elevation,
  speed and terrain (`nyc-09-track-drawing-ghost`); the station ghost says "1.2M trips within 1km"
  (`nyc-11-station-ghost`), which is the Islanders-style "what this would earn" preview the genre
  praises. The stop editor hides the bubbles and says "Click a circle to add or remove a stop.
  Press Enter when done." (`nyc-15-stop-hover`).
- **It pays.** A 7-station, 10.4 km line up Manhattan cost $1.94B of the $6B and, with 4 trains,
  had 148k riders and $135M expected profit in its first game morning (`nyc-21-running-60s`).

Where a stranger stalls, in the order they meet them:

| # | Point | What happens | Screenshot |
|---|---|---|---|
| S1 | **Start with nothing picked** | "Choose on map" is lit by default and Start is the big centred button, so the quickest click lands on the whole world. The tutorial says "Place 5km of track" with no place in view, the top bar says "World –", and in the headless run no zone bubbles had drawn after 50 s (only grey outlines; may be headless GL, worth a look in a real browser). The clock runs. | `default-start-01-first-visit`, `default-start-03-world-start-later` |
| S2 | **Opens on the wrong tab** | A new game opens on the Lines tab: "No lines yet." and a lit "New line N" button, while the card's first step needs the Build tab's Track tool. The ripple points at the Build tab icon, but the obvious button on screen is New line, which makes an empty line before any track. | `nyc-05-game-start` |
| S3 | **Six separate steps before a train moves** | Track, stations, construct (Build tab), then New line, stops, trains (Lines tab), each a different control. New line opens a name box selected for typing and a colour picker before any stop (`nyc-14-new-line`). Mini Metro's one drag gives a moving train; NIMBY Rails' long chain is the one its players complain of. | `nyc-14-new-line` |
| S4 | **Trains at high demand only** | "Add trains to the line" ticks when one train is set at any level, and its ripple points at the High row. Medium and Low stay 0 ("none"), so the line stops at 20:00 and runs nothing all weekend (weekend days run at medium), quietly, while upkeep goes on. Nothing on the card says so. Seen at 21:36 on day 1 (20:00 comes about 12 real minutes after the first train at normal speed, 3 at 4x): the Lines tab's Trains cell reads 0, the Medium row (in force) has 0 trains, the card still all ticked. | `nyc-19-trains-added`, `evening-27-evening-line` |
| S5 | **The payoff is small on screen** | After $1.94B the top bar moves from 0.0% to 0.6%. The goals (10% to 40%, the star) are only ticks on the city panel's bar, not on the top bar or the card; the card ends with "Done!" and nothing next. The first line gets no "+N daily rides" note (that note needs an earlier figure), so the only sign is the Riders cell counting up. At the start zoom the built track is a 1-2 px line under big bubbles and the trains are specks. | `nyc-21-running-60s`, `nyc-13-constructed` |
| S6 | **Words a stranger has to decode** | "Blueprint", "Construct blueprints", "High / Medium / Low" demand rows, "Headway", elevation "+1" with "1.0x cost", "Riders left at platform 14k/161k". Each has a hover or is standard in the genre, but they come all at once. The "Assign stops" counter reads 0/1 before construction and 0/3 after. | `nyc-07-build-tab`, `nyc-17-line-panel` |
| S7 | **Money with no yardstick** | The first line is a third of the cash and the panel shows upkeep in red ("$610k / day upkeep"); nothing says whether $135M a day is good. Early economy confusion is a named quit reason in Subway Builder's and Transport Fever 2's reviews. Here the first line does pay, so this is a reading problem only. | `nyc-12-stations-placed`, `nyc-23-finances` |
| S8 | **Phone** | Upright, the map keeps the top half under three rows of demand buttons and the card; the dock opens on Lines beside an empty inspector half. Playable, but the first step's tool is two taps away. | `phone-02-phone-game` |

## 3. Ideas, ranked

The shape of the problem: the start is a checklist of UI verbs, and the distance from "I want a
train here" to a moving train is six controls on two tabs, with a silent trap at the end (S4) and
little to see when it works (S5). The ideas below shorten that path with defaults, make the payoff
visible, and give the next goal. Ranked by how much of a stall they remove for their size. None is
filed; Anita picks. Wording is a draft for her.

1. **Start in a city by default** (small). The sheet lights New York (or the located city, once a
   location answers) instead of Choose on map, so Start reads "Start in New York"; Choose on map
   stays as the third way. Fixes S1. Evidence: Poki ("skip ... level selects", "drop-off ... is
   brutally underestimated"); Subway Builder players steering newcomers to one easy city. Rules: no
   new text; the button label already exists.
2. **Trains at every level from the first click** (small). On a new line (the T-1899 "new line"
   window), the first + on the High row sets Medium and Low to the same count, until the player
   sets either of them. Or, smaller
   still, step 6 needs trains at all three levels, counter "0/3", words "Add trains at high, medium
   and low demand". Fixes S4. Evidence: NIMBY Rails' "lines start closed" is its best-known trap;
   Rail Route's soft locks; Cheung et al. on one missed detail being a deal-breaker. Rules: no panel
   text; the 40% line drawing and its hover "No trains at medium demand" already exist.
3. **A tutorial game opens on the Build tab** (small). With the card showing, the dock opens on
   Build, not Lines (Lines is the default tab today, and a returning player's last tab stays as it
   is). Fixes S2.
   Evidence: context-sensitive help (Andersen et al.), Cities: Skylines II's "green outline back to
   the panel". Rules: nothing new on screen.
4. **After Done!, the card shows the next goal** (small to medium). The Done! button stays; above it
   one more row, "Reach 10% train share" with "0.6%/10%" and the star, its tick when the city's star
   is earned; then 20%, 30% and 40% in turn while the card is open. The top bar's split could carry
   the 10% tick too (it was left off in T-1551; Anita's call). Fixes the "and now what?" of S5.
   Evidence: Factorio (goals from the game's own systems, not arbitrary tasks), Kingdoms and Castles
   (two problems to solve), Sid Meier on rewarding the first fifteen minutes. Rules: the card is the
   one floating tutorial surface already allowed; no new panel text.
5. **Show the first line opening** (small). Give a line's first opening the same top-centre note as
   a stop edit, "+148,000 daily rides" with its chip, counted from 0. No camera move and nothing
   else. Fixes S5's quiet moment. Evidence: Mini Metro's instant train;
   Subway Builder's reviewer remembering the bubbles change colour. Rules: reuses an existing note
   and its wording.
6. **New line goes straight to stops** (small to medium). New line opens in stop editing with the
   name box not focused and the colour picker folded until the pencil is pressed; the line keeps
   "Line 1" and its colour. Fixes part of S3. Evidence: Factorio (core idea first, little else);
   NIMBY's line editor as a place players get lost. Rules: fewer things on screen.
7. **A default schedule for a new line** (medium). When a new line's stop editing first ends with
   two stops or more and no trains set, it gets trains for a 6 minute headway at high, 10 at medium
   and 15 at low (the cars bought as any edit buys them, refused like any edit when the cash is
   short), so "Create a line" and "Assign stops" end with a moving train, as in Mini Metro. The
   tutorial's step 6 then ticks by itself. Bigger than 2 and changes what a new line is, so it needs
   Anita's yes. Fixes S3 and S4 together.
8. **The player's network readable over the bubbles** (small to medium). While the Build tab is
   open, or at city zoom generally, blueprint and built track and station dots draw wider (about
   3 px at zoom 11) with the casing, over the bubbles. Fixes S5's "a line under bubbles". Evidence:
   Mini Metro's whole look; Subway Builder's demand view as the moment players remember. Rules: a
   drawing change only.
9. **Step 1 says where** (small; her wording call). "Place 5km of track" becomes "Place 5 km of
   track through the big bubbles" (wording only, no rule behind it, as she decided on 2026-10-29).
   Fixes the blank "where?" of the first step. Evidence: Mini Motorways' colour matching, Factorio's
   "the player does it because they are being told to".
10. **Payback on the first line's profit** (small). The line panel's Profit hover adds "Pays back
    its build in about 15 days" (its construction over its expected profit a day, while positive).
    Fixes S7. Evidence: early-economy complaints in Subway Builder and Transport Fever 2 reviews.
    Rules: a hover on a number, a fact, no method.
11. **Count where new players stop** (medium; her call on privacy). The page sends one anonymous
    count per tutorial step reached (no id, no save, nothing else), to the same server that counts
    game plays today; a tiny table on its stats page. Every idea above is a guess until this exists.
    Evidence: GameAnalytics' and Bugnet's tutorial funnels, Poki's "conversion to play".
    https://www.gameanalytics.com/blog/exploring-gaming-funnels,
    https://bugnet.io/blog/how-to-measure-and-reduce-tutorial-drop-off
12. **Fewer words at once in the Build tab while the card shows** (medium). The first tutorial game
    hides the rows a first track does not need (Bus, Select, Quad, High speed, Target max speed,
    Platform length) until step 3 is ticked, as Poki's "reveal actions and buttons over the first
    few levels" and Cities: Skylines II's fading guidance. Fixes S6. Changes what the Build tab is
    for a while, so it needs her yes; risk of a player not finding a tool they want.
13. **Phone: the first step's tool in reach** (small). In the upright layout with the card showing,
    the dock opens on Build with the Track tool in its half, not an empty inspector beside Lines.
    Fixes S8.

Overlap with `research/player-voices.md` (T-2074, the same night): its idea 1 (a second card that
teaches reading demand, after Subway Builder 1.4.17's "Tutorial steps tell you why, not just which
button to press") and idea 2 (a ladder of goals per city) are the bigger versions of idea 4 here;
idea 4 is the small first rung of either.

Not proposed: a tagline or welcome text on the start sheet (she removed it, T-1710); a forced
tutorial or locked tools (Factorio's lesson: players dislike being made to; Andersen: tutorial
freedom made no difference); a video or wall of text (Poki: "walls of text intimidate").
