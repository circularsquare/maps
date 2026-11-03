# Late-game things to spend on

Research and proposal for T-2167, 2026-11-03 (agent t2167). Anita, on player-voices' money pacing point:
"not sure we can do anything about money pacing tbh, it seems a bit inherent to this type of game. though
perhaps we can think about some super expensive stuff to add late game? idk. i feel like high speed rail and
long distance kinda hits that for me personally."

Nothing here is built or filed. Section 4 is the short version.

**The shape of the problem.** Late money only matters if it turns into something the player wants. In
trainsworld that can be three things: **riders** (capacity, speed, lower fares, which also mean points and
share goals), **reach** (new places, new crossings), or **recognition** (a name on the map, an achievement,
something to show). A sink that only burns money (rising upkeep, decorations with no effect) feels like a
tax, and a sink that turns money into more money (cheaper running) makes the pile grow faster. The good late
purchases are expensive, and each one buys a big step in riders, reach or recognition.

---

## 1. What a rich player can buy today, and what a big network earns

### Prices (track-and-service.md 6.4, SPEC 6)

| Thing | Price | Upkeep a game day |
|---|---|---|
| Double track, per route-km | $37.5M at ground, $75M at +1/-1, $120M at -2, $150M at +3/-3 | $20k ground, $25k viaduct, $35k cut and cover, $40k bored |
| High-speed track | 1.5x the above (320 km/h cap) | 1.5x |
| Quad track ("Double up", also on built track) | two double tracks | 2x |
| Hills, mountains | +0.5, +1.2 or +3 on the level's multiplier (not at -2, -3) | none |
| Over or under water | +2.5 on the multiplier: a +1 bridge $262.5M/km, a -2 tunnel $307.5M/km; viaducts and cut and cover refused past 30 km of water | as the level |
| Road crossed at ground | $6M each | none |
| Station (200 m) | $75M at ground to $175M at -3, 2x over water | $50k |
| Station upgrades | terminal (1 km of track), park and ride $20M, waiting room $5M, passage $10M + $90M/km | none |
| Cars | $2.5M, high-speed $3.75M; never sold | running $4/train-km + $2/car-km (HS $5 + $3), x100 |
| Buses | $1M a road-km, $0.6M a bus | $9/bus-km, x100 |
| Loan | one at a time, $1B, $5B or $10B, 60 daily payments | |

Other ways money already goes: removing built track refunds nothing (so rebuilding is a sink); built
track cannot be reshaped, re-levelled or switched to high speed (remove and draw again); and **lowering
fares** turns money straight into riders, points and share goals. Construction is instant. There is no
research, no ageing, no rival.

### What a big network earns

From T-2104's measurements (notes/T-2104.md, weekday, game dollars):

| Save | Fares a day | Profit a day |
|---|---|---|
| Real New York (109 lines) | $3.45B | about $1.95B (sum of line profits) |
| Tokyo, Seoul, Fukuoka, Busan (65 lines) | $6.65B | not summed; the Shinkansen line alone +$786M |

A game day is 24 minutes at normal speed, 6 at 4x, 90 seconds at 16x and 14 seconds at the slider's 100x
(`SPEEDS` in game/clock.ts). So a New York-sized network makes about **$2B every 90 seconds at 16x**.
Building that whole real network at today's prices would cost roughly $0.5 trillion (my estimate from its
2,848 km in tunnel and 1,651 km elevated, notes/T-1897.md, plus its stations), about 250 game days of its own
profit, or about six hours of play at 16x.

How fast a normal game gets there is not measured (player-voices idea 18). Both saves above are imported
real networks.

### What megaprojects cost at today's prices

Using the current rules, -2 for tunnels, +1 for bridges; real costs are from the linked pages and rounded.

| Project | Length | Game price, ordinary / high speed | Real |
|---|---|---|---|
| [Channel Tunnel](https://en.wikipedia.org/wiki/Channel_Tunnel) | 50.5 km, 37.9 under sea | $13.2B / $19.7B; upkeep $3.0M a day (HS) | about £4.65B at 1985 prices |
| [Seikan Tunnel](https://en.wikipedia.org/wiki/Seikan_Tunnel) | 53.9 km, 23.3 under sea | $10.8B / $16.2B | about ¥538B (1988) |
| [Gotthard Base Tunnel](https://en.wikipedia.org/wiki/Gotthard_Base_Tunnel) | 57 km under mountains | $6.9B / $10.3B | about CHF 12B |
| [Fehmarnbelt tunnel](https://en.wikipedia.org/wiki/Fehmarn_Belt_fixed_link) | 18 km immersed (-1 under water) | $4.7B / $7.1B | about €7B |
| [Strait of Messina bridge](https://en.wikipedia.org/wiki/Strait_of_Messina_Bridge) | 3.3 km span | $0.9B / $1.3B | about €13.5B (planned) |
| [Helsinki-Tallinn tunnel](https://en.wikipedia.org/wiki/Helsinki%E2%80%93Tallinn_Tunnel) | about 90 km, proposed | $28B / $42B | studies €15-20B |
| [Japan-Korea tunnel](https://en.wikipedia.org/wiki/Japan%E2%80%93Korea_Undersea_Tunnel) | about 200 km, 128 under sea, proposed | about $72B (HS) | |
| [Chuo Shinkansen](https://en.wikipedia.org/wiki/Ch%C5%AB%C5%8D_Shinkansen), Tokyo-Nagoya | 286 km, most of it tunnel | about $49B as HS (no maglev in the game) | ¥11 trillion, 2025 estimate ([Rail Journal](https://www.railjournal.com/news/chuo-maglev-costs-soar/)) |
| [Beijing-Shanghai HSR](https://en.wikipedia.org/wiki/Beijing%E2%80%93Shanghai_high-speed_railway) | 1,318 km | about $84B at ground, $148B on viaduct | about ¥221B |
| [Crossrail](https://en.wikipedia.org/wiki/Crossrail) central tunnel | 21 km, 10 stations | about $4B | about £19B |

What this shows:
- **Megaprojects already exist, priced by the ordinary rules**, at roughly real-world prices for long
  tunnels and high-speed lines (Gotthard, Chuo), and far under real for city tunnels (Crossrail) and long
  spans (Messina: no depth or span rule).
- **To a New York-sized network they are 5 to 45 game days of profit**, minutes to an hour of play at 16x.
  Expensive, not huge.
- **They don't pay back.** The Tokyo save's Japan-Korea high-speed line makes +$5.9M a day (T-2104); at
  about $72B that is 12,000 game days. So the biggest projects are bought for reach and pride, not profit,
  which is how real ones mostly work too.
- **The time scale.** Fares and running costs count 100x real days, construction doesn't, so a real
  project's 30-50 year payback is 110-180 game days here. Prices are not the problem; what the money is
  for, once the network is built, is.
- **Long distance has to feel worth it.** Anita's own sink only works if a long high-speed line fills.
  T-1504 found China's ten high-speed lines nearly empty, and T-1599 (Beijing-Shanghai at player scale,
  waiting on her) is the check for that. Any of the ideas below that lean on long distance depend on it.

---

## 2. How other games handle late money

Tags as in player-voices.md: **[DEV]**, **[PLAYER]**, **[CRITIC]**, **[GUESS]**.

### Games where the late pile is the known complaint

- **Transport Fever 2.** "There was a tipping point where you were making so much money that money was no
  longer an object as such"; "Problems can be solved by throwing money at them" [PLAYER, thread "Running
  costs are too cheap", 2019]
  https://steamcommunity.com/app/1066780/discussions/0/1740010344366988160 . The usual fix offered is
  maintenance that rises each year ("maybe 10% increase every year to everything"), and the replies are
  split: it "only changes how long it takes for your bank balance to take off", and "Why penalise people
  for putting in the work?" [PLAYER, 2020]
  https://steamcommunity.com/app/1066780/discussions/0/2872668209264027820 . A critic: "plays as if we
  just used a cheat for unlimited money" [CRITIC]
  https://www.gamepressure.com/editorials/reviews/transport-fever-2-review-good-sim-poor-tycoon/z520c
- **NIMBY Rails.** "this is an easy game if you just want to make money"; "Wait a couple of days and the
  many millions a day will start rolling in"; what players then do with it: "when I made enough to repay
  the loans and have money left, I built between the cities" [PLAYER, "What's The Secret?", 2021]
  https://steamcommunity.com/app/1134710/discussions/0/3148556875507046807 . The answer the game itself
  gives is a no-money mode, a top request [DEV] https://carloscarrasco.com/page/58/ .
- **Cities: Skylines.** Monuments are its late purchase: six unique buildings unlock each, they cost
  650k-1.9M with upkeep, and each has a city-wide effect [wiki/guide]
  https://www.keengamer.com/articles/guides/8-cities-skylines-monuments-and-how-to-unlock-them/ . Players:
  "they should generate money, not cost money"; "you will already have a massive income if you done stuff
  correctly" [PLAYER] https://steamcommunity.com/app/255710/discussions/0/611701999524856515 .
  Cities: Skylines II's launch economy was too easy; the developer admitted it and reworked it in 2024
  (Economy 2.0), while still wanting the game "playable without paying too much attention to where and how
  money ... flow" [DEV, via press] https://www.pcgamesn.com/cities-skylines-2/broken-economy
- **Workers & Resources: Soviet Republic.** "Most of the mechanics make early game harder, but late game is
  always a breeze"; players asked for a space programme as a late project [PLAYER]
  https://steamcommunity.com/app/784150/discussions/0/4425436537577071474 . Its money is only a trade
  currency; the economy runs on resources [wiki]
  https://wiki.hoodedhorse.com/Workers_Resources_Soviet_Republic/Rubles
- **Railway Empire 2.** Late game "labor-intensive and repetitive", "tedious track building and
  reconstruction" [PLAYER, summary] https://gamebrain.co/game/railway-empire-2 ; sandbox has unlimited
  money [DEV] https://blog.kalypsomedia.com/en/?p=2028
- **Subway Builder.** Bonds and loans help the early game [DEV]
  https://www.subwaybuilder.com/changelog ; players find it "hard to reach profitability" [PLAYER, summary]
  https://vaporlens.app/app/4039140/subway_builder.md . No late-game spending found; its answer to long
  play is daily challenges with a leaderboard.

### Ways games give the late money a job

- **Megaprojects as the goal.** Satisfactory's Space Elevator takes deliveries in phases, each unlocking
  the next tier, and the last launches the ending [wiki] https://satisfactory.wiki.gg/wiki/Project_Assembly .
  Anno 1800's World's Fair is "the ultimate end-game goal", built in stages from growing production chains
  [CRITIC, via search; DEV] https://www.ludo.guide/guide/anno-1800/overview/quest-a-monument-to-progress .
  SimCity (2013)'s **Great Works** (Arcology, International Airport, Space Center) are built by every city
  of a region together, cost about 1M and serve the whole region [guide]
  https://primagames.com/eguides/simcity-eguide/essentials/great-works . What players like here is a named,
  visible thing that took the whole economy to make.
- **Spending on the towns.** OpenTTD's local authority sells advertising (station ratings up for a while),
  a statue of the owner, funding new buildings (the town grows; no direct return), exclusive transport
  rights and bribes [wiki] https://wiki.openttd.org/en/Manual/Local%20authority . A-Train lets the company
  build hotels, offices and resorts by its stations, which grow the town and earn rent [via search]
  https://forum.quartertothree.com/t/lets-take-the-a-train-an-all-purpose-series-thread/152528/126 .
  Subway Builder players ask for demand that grows round new stations [PLAYER]
  https://steamcommunity.com/app/4039140/discussions/0/588433527532489346/
- **Upkeep that grows with the network.** OpenTTD's infrastructure maintenance grows faster than the
  network, so short links can lose money, and players are told to remove what they don't need; some find
  it and inflation harsh enough to switch off [wiki, PLAYER]
  https://wiki.openttd.org/en/Archive/Community/Infrastructure%20maintenance . trainsworld has flat upkeep
  per km already (T-867).
- **Markets and rivals.** Railroad Tycoon's late money goes into stocks and taking over rival railroads
  [guide] https://strategywiki.org/wiki/Railroad_Tycoon_3/Gameplay . trainsworld has no rivals.
- **No money at all.** Mini Metro has none: every in-game week brings a new train and a pick of upgrades
  (a carriage, a line, tunnels or bridges, an interchange) [wiki]
  https://shapes.inc/fandom/mini-metro/upgrades . Scarcity is in the pieces, so there is no pile. Factorio
  has no money either; after the rocket, players set their own goals (megabases, science per minute) and
  stop "only because they hit performance limits" [PLAYER]
  https://forums.factorio.com/viewtopic.php?p=572387 ; continuing "can feel empty because the game has no
  official objective after that point" [PLAYER] https://forums.factorio.com/viewtopic.php?p=362832 .

### What carries over

1. Nobody has fixed the late pile by making things dearer. Rising maintenance is the most proposed fix and
   the most argued against.
2. The late purchases players remember are **named projects with a visible result**: a space elevator, a
   World's Fair, a Great Work. In a rail game on the real world, that is the real world's own megaprojects.
3. NIMBY players already do what Anita does: once rich, **build between the cities**. trainsworld's map is
   the whole world, so the next country is always there; that is its biggest sink, and it is free to add.
4. Spending that is the same click many times (OpenTTD's adverts and statues, TF2's vehicle renewal) is
   busywork; one big decision is not.

---

## 3. Ideas

Each: what it is, why a player would want it, rough cost in game money, what it needs, how it fits the
rules (simple predictable prices, demand simple on purpose, sensible over realistic). "Busywork" is said
where it applies.

### A. Upgrade built track in place
Switch built double track to high speed, or move it to another level (bury a viaduct, lift ground track
through a centre), keeping its lines, stations and trains. Today it means remove (no refund) and draw again.
- Why: the real progression for long lines (build cheap, upgrade when it pays), and exactly Anita's sink:
  a rich player turns a 160 km/h corridor into a 320 one. Subway Builder players list "no track upgrades
  without demolishing" as a complaint [PLAYER, summary].
- Cost: the new track's full price, as if built again (100 km of ground track to high speed: about $5.6B).
  The simplest rule, and as dear as today's remove-and-rebuild.
- Needs: sim (edit the high-speed flag and the level on built edges with service kept; upkeep follows);
  UI only enables what the Select bar already has for blueprint (Regular / High speed, elevation buttons) on
  built track, with the price in the hover. Curves stay as drawn, so a line drawn for 160 km/h keeps its
  slow curves (built track cannot be reshaped).
- Fit: no new price rule, no demand change. Not busywork: one decision per corridor.

### B. Named crossings: the world's megaprojects as goals
A fixed list of 20-30 real sites, built or only dreamed of: the Strait of Dover, Tsugaru Strait, Gotthard
and Brenner, the Fehmarn Belt, the Bosporus, the Strait of Messina, Helsinki-Tallinn, Gibraltar, the Korea
Strait, the Bohai Strait, Qiongzhou, a Tokyo-Osaka maglev route. When a running line uses one, it is
marked built: an entry in the achievements with the date, and the place named on the map.
- Why: the late goal other games give (Space Elevator, World's Fair, Great Works) in the real world's own
  terms. Most of these lose money, so the reason to build is the goal, which is honest.
- Cost: no new prices; the current rules give $1-70B each (table in section 1).
- Needs: a small hand-made data file (each site as two end areas or a corridor polygon), a check on line
  paths in the app, rows in achievements (T-2124's medal list; T-2150's Sea crossing is the first step).
  Only sites whose both ends are in covered zones are shown (Gibraltar waits on Morocco, Bohai on Liaoning
  and Shandong).
- Fit: data and UI only. A one-off hand list breaks "general sources only" a little, but it is game
  content, not demand data.

### C. Maglev
A third track and train kind: 500 km/h, its own track that joins nothing else, its own stations
(changes by passage).
- Why: the top of the speed ladder, for 400-1,000 km where today's 320 km/h loses to the plane. Shanghai
  runs one, the Chuo Shinkansen is being built.
- Cost: 2.5-3x the level's price (high speed is 1.5x): about $300-360M a km in a -2 tunnel, so Tokyo-Osaka
  is $120-150B. Cars about $6M, running cost about twice high speed's.
- Needs: sim (a speed tier, curve radii for 500 km/h, about 8 km minimum, a train kind and its fleet,
  capacity's `v / 0.6` term at 500), UI (track type, train type). No demand change: the time saving goes
  through the existing mode choice against car and plane.
- Fit: one more multiplier, simple. Risk of complexity creep (player-voices point 6) is small since it
  copies high speed. Depends on long trips riding rail at all (T-1504, T-1599).

### D. Modern signals
An upgrade on a stretch of track that lets trains run closer: the track section's fixed 70 s per train drops to
about 45 s, so at 80 km/h 34 trains an hour become 44, at 320 km/h 16 become 18.
- Why: the trunk lines of a big network hit the track's limit, and quad track is not always possible. Real
  cities pay a lot for this (New York's 7 line, about $600M for 16 km, about $37M a km,
  https://reinventalbany.org/2018/11/flushing-7-train-cbtc-signals-late-and-over-budget-what-lessons-will-the-mta-learn-for-systemwide-modernization/).
- Cost: about $40M a route-km, any level.
- Needs: sim (one flag per edge, read by capacity's occupancy), UI (an icon in the track inspector and the
  Select bar).
- Fit: one number and one price. Fits "no signals, for now" (SPEC 2.1), since capacity stays a figure.
  Busywork risk if it has to be bought piece by piece; buy it over a whole run, as Double up does.

### E. Big stations
A station upgrade with more platform tracks (4, then 8) and a hall: more trains an hour through it and
turning at it, a bigger platform crowd, a shorter change inside it.
- Why: a city's main station is where every long line ends; today every station has two platform tracks
  per pair. Real ones are megaprojects of their own (Stuttgart 21 about €11B, Berlin Hauptbahnhof about €1B).
- Cost: $1-2B at ground, $3-5B underground, for the 8-track one.
- Needs: sim (platform tracks per station as a count, which the terminus rule already divides by; the
  platform crowd's 5,000 cap), drawing (a wider box, a roof), possibly a smaller change penalty in the
  complex (a demand rule, like the waiting room's 0.8, so Anita's call).
- Fit: medium. Also a show-off piece.

### F. Driverless lines
An upgrade per line that cuts its running cost a train-km (say $4 to $2) for a capital price (Paris line
1 was automated while running, https://en.wikipedia.org/wiki/Paris_M%C3%A9tro_Line_1 ).
- Cost: about $20M a route-km.
- Honest note: it pays back, so it turns money into more money and makes the late pile grow faster. Good
  in the early-middle game, the wrong direction for this task.

### G. Construction over time for the biggest projects
A project over some size (say $10B) builds over game days instead of at once, its cost drawn as it goes,
drawn on the map as a site.
- Why: the Space Elevator and World's Fair feel comes from phases you watch fill. Also lets a player start
  a project before having all the cash.
- Needs: sim and UI, and it changes "construct builds at once" (Anita, 2026-10-09). Large.
- Fit: it adds waiting, not spending; only if she wants big builds to feel like events.

### H. Grow places round stations
Pay to grow homes and jobs within 800 m of a station (OpenTTD's "fund new buildings", A-Train's
subsidiaries, Subway Builder players' request, player-voices idea 15).
- Cost: say $1B a station for +10% people and jobs within 800 m.
- Needs: demand data edits at run time, a save field, a re-solve. Large, and it changes where demand comes
  from.
- Fit: against "demand simple on purpose" unless kept to one rule; needs Anita first.

### I. Cut fares as a goal
Already possible: the fare curve can go to $0, which spends money on riders, share goals and points.
What is missing is only recognition: an achievement for a city at 0 fares with 30% train share, say.
- Cost: what the fares were (New York: about $3.45B a day).
- Needs: one achievement row. Tiny.

### J. Faster cars on existing lines
A newer fleet with better acceleration, replacing a line's cars at full price.
- Busywork: this is TF2 and OpenTTD's vehicle renewal, one click per line many times, for a few seconds a
  run. Not recommended.

### K. Rising upkeep or ageing
Upkeep that grows each year, or trains that wear out.
- Busywork and a tax, and players split on it (TF2 thread above). trainsworld's flat upkeep a km is
  already the useful part. Not recommended.

### L. Showing off (later, as Anita said)
Money turned into something to look at or share:
- **Liveries and train names**: a colour and a name per line's trains, a price per fleet ($0.1-1B), only
  for looks.
- **Station halls**: E's roof without its capacity, a price by size.
- **A page of the player's crossings and records** (B's list, the longest line, the fastest train) to
  share by link, with leaderboards beside it (T-988 to T-990).
Pure sinks are fine here, because the result is the point. Needs drawing and share-link work; later.

### Not taken
Stock markets and rivals (no rivals), OpenTTD-style adverts, statues and bribes (busywork, no rivals to
beat), dearer far cities (breaks one price list worldwide), hyperloop (not sensible), bigger loans (money
is not short late).

---

## 4. Top picks, ranked

### 1. Upgrade built track in place (A)
The direct form of Anita's own sink. A player builds a cheap double-track corridor to reach a far city,
and once rich upgrades it to high speed or buries the noisy viaduct through the centre, without tearing
out the lines that run on it. One price rule (the new track's full price), no demand change, and the
Select bar already has the buttons for blueprint track. It also removes a remove-and-redraw step players
of other games complain about. About one sim session and a UI pass.
- **Anita decides**: the price, the new track's full price (my pick, simplest) or only the difference
  between the two plus a rebuild charge; and whether re-levelling is in or only high speed.

### 2. Named crossings (B)
A list of the world's straits, base tunnels and great lines, each marked when a running line uses it, with
its date and name in the achievements and on the map. It gives the late money a goal of the kind Satisfactory,
Anno and SimCity use, and in trainsworld's own terms: real places, built at the prices the game already
has, mostly at a loss, as real ones are. It grows from T-2150's Sea crossing and needs only a hand-made data
file and UI. The list grows with the world (Gibraltar when Morocco comes).
- **Anita decides**: whether a hand-made list of places is fine here (it is content, not demand data);
  whether proposed and never-built projects are in (my pick: yes, they are the fun ones); whether each
  marks a name on the map or only in the achievements.

### 3. Maglev (C)
The top of the speed ladder: 500 km/h on its own track at 2.5-3x the price, so Tokyo-Osaka is a $120-150B
project. It is where long distance goes past high speed, which is what Anita said hits the late game for
her. It copies high speed's pattern (a track kind, a train kind, a fleet), so the rules stay simple. It
only makes sense once long trips ride rail in numbers (T-1504; T-1599 waits on her), so it comes after that.
- **Anita decides**: whether to add a third kind at all; its own track only (my pick) or able to run on
  high-speed track at 320; the price factor (2.5x or 3x).

### 4. Modern signals (D)
One upgrade over a run of track, about $40M a km, that takes the track's limit from 34 to 44 trains an
hour at 80 km/h. It is for the trunk of a big network, where quad track is not possible, and turns money
into riders rather than more money. One flag per edge in capacity and an icon.
- **Anita decides**: whether trainsworld wants a capacity upgrade beside quad track; the size of the gain
  (45 s in place of 70 is my guess for a modern system).

### 5. Big stations (E)
More platform tracks and a hall at a city's main station, $1-5B. Capacity for the place every long line
ends, and the first show-off piece. Bigger in code than D (platform tracks as a count, drawing).
- **Anita decides**: whether a smaller change penalty inside it is in (that is a demand rule, like the
  waiting room's 0.8) or only capacity.

**Before any of them**, player-voices idea 18 (play or script a long normal game and chart cash against
what there is to buy) would say how big the late pile really gets and when. The figures above are from
imported real networks.

### Draft player-facing text for the picks (for Anita's reading, not final)

- A, Select bar on built track: buttons stay "Regular" and "High speed"; hover "Upgrade 12.4 km to
  high-speed track" then "$698M".
- B, achievements: a "Crossings" group, entries named by the place ("Strait of Dover", "Tsugaru Strait",
  "Gotthard"); the toast "A line under the Strait of Dover".
- C, Build tab track choice: "Maglev"; speed limit hover "Maglev trains up to 500 km/h".
- D, track inspector icon hover: "Modern signals" then "Up to 44 trains/hour each way, from 34".
- E, station upgrade hover: "Big station" then "8 platform tracks".
