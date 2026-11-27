# Track / platform / station design: how other games do it (T-3444, agent "games")

Web research only, ~40 calls. Items marked UNVERIFIED or NOT FOUND are gaps. Many Steam thread summaries come via a small
summariser model, so quotes are as reported by it; reddit.com is blocked for the search tool (no r/ links), tt-forums.net and
forum.the open-source transport sim.com and steamdb.info return 403/Anubis, wiki.the subway-building game.com returned 502.

## The track-drawing rail game (closest peer)
Track
- Track is a chain of single "track objects" (one control point each, linked to previous/next sibling). Tracks can be a branch or a
  PARALLEL track "parented" to another track: parented to a control point (offset 0-1000 m, can flip side) or to a point on the
  track (segment t 0-1, perpendicular displacement -1000..1000 m). So a "double" is just two parented tracks.
- Tracks menu has a "Double track mode" toggle (draws a second track beside); "the most common additional button".
- Platforms are track sections promoted to platform ("Promote track into platform"); a platform segment cannot have branches.
Station
- "A station is a set of one or more platforms whose footprints overlap on each other"; a lone platform is a station; no limit on
  platforms, any layer/orientation. Properties: name (auto-named by default), coverage area (passenger spawn), auto walk transfer
  range, platform passenger capacity 50-100,000, station-hall overflow. Coverage 0 = depot.
- Create stations tool (hotkey T): "Create new stations, or add platforms to an existing station". How to build track page.
Stop picking
- "Stops are always defined as two tracks, be it a station platform or a waypoint" (the two directions). Basic mode (default): you
  click platform ENDS; the game places the train. Hold Alt to force all platform options visible. Advanced mode (1.9+): click any
  point on a track; point on platform track = line stop, on normal track = waypoint; train nose stops at the point, a stripe shows
  reference train length.
- Platforms appear in the line editor as lettered options (A, B, C, D...). The editor hides directions it thinks are useless using
  direction signals; Dev (Weird and Wry): "Hold Alt to force the game to show all the platform options when creating lines."
  Complaint (Keiji, Apr 2022): editor offered A and C which could not be used and hid the usable D; "would have stopped playing
  without" the Alt trick (summariser's wording).
- Platform sharing (1.9 advanced): trains share a platform if they do not overlap while stopped; a 250 m platform fits two 80 m
  trains; a balise at the platform entrance overrides track reservation (then timetabling is on the player).
- Platform stop point: a signal placed on a platform track; trains stop at it in basic mode. Dev advice: build platform track a bit
  longer than needed, use platform stop signals for exact stop position.
- Alternatives/secondary stops: patch notes (via search snippet, steamdb blocked): 1.5 "to select secondary platforms for a stop
  you first need a path signal to serve as the platform distributor"; later: "stop selection signals" at station level replaced
  line-level secondary stop signals; a train whose path ends on a platform of a station including that signal may change its
  destination stop area among the line stop's main + secondary areas.  (403 to fetch),
(snippets only; exact wording UNVERIFIED)
- Path signal setup: entry signal (type AB) at station throat, platform exit signals (DB/DS), through-only platforms via one-way
  signals; "Entry signals control access: stopped if the destination platform is occupied". Passing tracks let expresses bypass.
- Dev advice: terminus with two platform stops forces trains to reverse and reposition; keep one stop per terminus where possible
  (via search summary of a Steam reply; url not fetched).
Passing / overtakes / quad
- Track layout guide (under construction): timed overtake in a short passing track; "Being smart with platform assignment can
  reduce path crossings"; quad track types: route-paired, speed-paired, direction-paired (inner-slow smaller footprint);
  "Quad tracking is often needed beyond about two slow and two fast trains per hour". Crossovers handled as plain track.
Complaints / praise
- Platform filter bug above. Players need layover yards "14 tracks past Kenosha" and "alternative platform functionality" to spread
  trains:
- Station stretches when a track is connected, hard to hold fixed platform length:
- Signalling a terminus: players ask if "same direction only" fixes exit-signal alerts; answer yes:
- Island platforms in 2022: a comment said the beta branch was needed (https://steamcommunity.com/sharedfiles/filedetails/comments/2775109598),
  the wiki's build-track page does not mention island platforms at all. NOT FOUND: any the track-drawing rail game multi-track "quad" tool beyond double mode.

## The subway-building game (casual-to-medium peer; most relevant recent changelog)
All from  (fetched). Wiki page
(only via search snippet: single/double/quad track stations cost 0.75x / 1.0x / 1.5x; quad = express/local).
- Station is a placed object with a track type: single, double, double island ("double track island platform"), quad (express
  layouts, "Platform layout system with quad-express layouts and retroactive track adjustment", v1.4.0). Platform width configurable
  (1.4.5). Stations can be shorter for fewer cars (cheaper), curved. Parallel tracks "always connect to the center two tracks of
  quad track stations" (1.6.0); later "you can connect to any valid pair of tracks, not just inner or outer" (1.7.0).
- Crossovers placed AUTOMATICALLY when drawing stations; advanced mode lets you place manually; "scissors crossovers can be
  placed anywhere along parallel tracks"; auto-crossovers in advanced mode now a toggle default off; "Crossovers don't need
  Advanced Mode anymore. They are needed to turn trains around at a terminus." Unbuilt crossovers visually flagged.
- Stations grouped into "station groups" (manual group editor 1.1.0, can have several marker dots with one label, rename all at once).
  Stations are separate objects; a group is the hub/name.
- Route editing: click stations while editing a route; "Right-clicking a station track while editing a route adds or removes all its
  stations" (1.7.0); route errors "name the affected stations and specify when a stop needs both platforms or manual ordering"
  (1.4.17); stations in the route list can be dragged to change order (manual routing, 1.7.0). "Train routes can turn around at any
  station track at their terminus, not just the assigned one" (1.6.0).
- Platform reservation (1.7.0): "trains no longer wait for an occupied platform when another platform is free" (a bug fix in how
  casual routes work: a route stops at a station, the game picks a free track).
- Tracks and stations can have directions of travel beyond forward/backward; "Switch track direction" option in advanced mode;
  arrows on tracks show direction (0.10.0). Traffic light icon flags grade conflicts. Parallel and interlined routes draw side by side.
- Advanced mode vs default: pattern is "automatic by default, manual in an Advanced Mode toggle" for crossovers and directions.
- NOT FOUND: player reviews/complaints of the subway-building game platform handling (no Reddit access, search returned nothing).

## The 3D transport game (and its sequel)
- Stations: prebuilt terminal templates and modular stations built from platform modules + track modules; "each section of a terminal
  needs both a platform module and a track module, and it can't use the regular tracks placed on the map" (160 m terminal = 4 + 4);
  "every track in a station requires an adjacent platform in order to operate"; capacity computed from lane layout.
   (via search summary)
- Stop picking: each line stop is bound to ONE terminal in the Line Manager ("every single vehicle on Line A will use Platform 1
  every single time"); optional "alternative terminal" per line per station: falls back when primary is occupied; works only if the
  train can physically reach it (crossovers/signals); unreachable shown with yellow/red chevrons.
- Praise/complaint (Dec 2019): Mordgier "Static routes platform to platform as before - micromanagement hell."; Actionjackson
  "The TpF solution is a lot less work for the player compared to the [another game] way of doing it" and in the classic transport sim/the open-source transport sim larger stations
  need elaborate signalling ("a big chaos" without). The rail-building game users say signal-based pathing uses any free platform.
- TF1: Dec 2017 request for trains to pick any empty platform; "[another game] lacks the dynamic platform assignment that TTD had".
   (via search summary)
- Mixed road station: terminal list shows only the line type of the last platform built (2019 bug):
- The 3D transport game: released 2026-09-29;
  announced as modular infrastructure with "line manager that combines management of lines, stations and vehicles in one view";
   (search snippet).
  NOT FOUND: any concrete TF3 platform/terminal-selection change; reviews barely mention stations.

## The classic transport sim and its patch pack
- Station = tiles of station track joined (platform tiles), a catchment-based object; stations have no "platform number" pick;
  platform choice is by signalling. Path signals (one at each platform exit + one before the throat): train reaching signal "looks for a
  free platform and gets green if one is free, otherwise red". Pre-signal / waiting-bay layouts for queuing; terminus: "trains head
  for the nearest free" platform.  station;;
- Pitfall: train waits for a free path while a platform is free because a signal behind is blocked
  (https://www.tt-forums.net/viewtopic.php?p=1228399, via search summary; fetch blocked).
- The classic transport sim: conditional orders can test number of free platforms; road vehicle orders can require a specific bay/direction;
  through-load for trains longer than platform; programmable pre-signals, no-entry signals, routing restrictions (hidden by default
  in the build window); fixes for platforms left partially reserved. Orders are per station, never per platform.
changelog
- NOT FOUND: a classic transport sim "via order to a specific platform" doc beyond via-state hotkey entries; tt-forums blocked.

## The open-source transport sim
- Stations ("halts") are groups of station tiles/stops; schedule entry is a tile (buses remember squares, not stations).
  Choose signal: "allow trains to automatically choose an empty platform"; train checks the platform in its schedule first, then
  others; "If none is free, it waits at the signal for its scheduled platform". End-of-choose sign bounds the search. Bug: with all
  platforms full trains ignore the choose signal and go to the default platform. Warning: don't use at drive-through stations where
  not every train stops.; forum.the open-source transport sim.com msg=20456, 171810, 176088
  (blocked, search summary). Proposal (not confirmed built): list all usable platforms in the schedule as alternatives.

## The rail tycoon games
- Stations have a fixed number of tracks (4/6 etc by upgrade); per stop in the route, pick platform ("Shift+click" opens station
  stop details; track 1 is nearest the station building). Gridirons (built outside station, third tab of track construction) let
  trains automatically take the first reachable free platform in RE2; no signal-controlled stations in RE2.
- Complaint (RE1, 2018): "lack of dynamic platform choice a dealbreaker", "has turned me off this game after about 30 hours";
  reassigning platforms is "a tedium simulator with time pressure"; "basic feature in this genre"; "the ONLY thing stopping me from
  buying"; counter: "I can accept that the player has to manage his platforms". Dev reportedly had technical reasons.
- RE2 "Auto platform selection" thread (May 2023): "the lack of signal-controlled stations a major loss"; workaround 4-track gridirons
  need 8-track stations to avoid clogging:
- The classic rail tycoon game: NOT RESEARCHED separately (no calls spent).

## The city-builder and its sequel
- The city-builder vanilla: a line's platform is set by where you click when building the line: start at a single station, click the multi-platform
  station, etc; wrong platform = delete and rebuild. Mods: Multi-Track Station Enabler / Advanced Stop Selection: hold Shift, hover
  platform ends until the stop dot lands on the right track; TMPE lane connectors to designate a track per line. Player: "When I place
  the stops they get crossed over", selector "want's to switch rails from the one I select".
- Metro Overhaul Mod: track menu mixes style, single/double, one-way/two-way; ground/elevated station types; island platform stations
  (v7); 4-track stations (v10): "Express Stations are 4 track stations with platforms on the outer 2 tracks. Dual Island Stations
  are 4 track stations with 2 island platforms servicing all 4 tracks"; station connectivity/pathfinding fixes for hubs.
  https://steamcommunity.com/sharedfiles/filedetails/changelog/816260433
- The newer city-builder: prefabricated stations (small, medium 2 platforms, central multi-platform, City Central 14 platforms: );
  dev diary: "Stations also feature pre-built tracks which create track switches automatically when tracks are connected to them,
  regardless of track type".
  Request (Nov 2023): "Train stations should be freely expandable with platform modules that can be added as many times side by side
  as you wish" (search summary). The newer city-builder wiki: use platforms at
  least 160 m. Modular station mod exists ("Modular Transport Station
  [Subway Freedom]",; page content not readable).
- NOT FOUND: how the newer city-builder vanilla picks which platform a line uses; no the newer city-builder platform-pick complaint located.

## The factory game (for the "station = name" idea)
- Train stops with the same name act as one station; train picks among enabled, not-full stops, then highest priority (0-255, default
  50), then closest by pathfinding distance; per-stop train limit; waits "destination full". Stackers are player-built queues.

## The Soviet-era city sim, the rail-building game, others
- W&R: block signalling (semaphores), dedicated platforms per direction more reliable; 4-track stations can deadlock
  ("unless the first train reverses out ... deadlock to break manually"). https://wiki.hoodedhorse.com/Workers_Resources_Soviet_Republic/Train_platform
  (search summary only)
- The rail-building game: platform choice via signals (user claim), no official source found. The indie rail game: per-stop "platform" button with a
  mouse icon and an "Any" platform option (a dev said occupancy of the chosen platform is then not displayed; this is from a search
  summary of; the "Platform Adjustments Upgrade" patch
  note could belong to another game, UNVERIFIED).
- The Japanese train-tycoon series: NOT FOUND (only store pages and one scenario guide note about two platforms, one line each side).
- The minimalist metro game, the rail puzzle game, Sim Rail: not researched, no calls spent.
