# T-1840 text review

Reply by number, e.g. "12: fine, 15: drop, 18: use yours". Anything you skip stays as it is. "Try" is my proposed rewrite, "drop" means remove the hover.
Strings changed in T-1835 to T-1844 appear only where they still break a rule. Brackets give file:line for whoever applies your answers.

## Start menu, new game, loading

1. Start menu, game row, and Saves panel; error under a game
   Now: `This game could not be downloaded.` / `This game could not be deleted.` / `This game could not be read.` / `This backup could not be downloaded.`
   Problem: passive "could not be", says what happened but not what to do.
   Try: `Could not download this game. Try again.` (same shape for the others)
   [ui/StartMenu.tsx:97,110,115; ui/SavesPanel.tsx:194,228,348]

2. New game, Money row, hover on Normal / Sandbox (also Settings > Game, same text)
   Now: `Construction costs money, and an edit is refused when the cash runs short` / `Infinite money, and the cash is hidden. Fares and running costs are still counted.`
   Problem: two clauses of one shape, "an edit is refused" is passive, "still counted" is passive.
   Try: `Building costs money. You can't build when you run out.` / `Unlimited money, and no cash shown. Fares and running costs still count.`
   [ui/NewGame.tsx:148-149; ui/SettingsTab.tsx:190]

3. New game, "Start with X's rail lines" checkbox, hover
   Now: `Start with Tokyo's rail lines as they run today, built and running`
   Problem: "built and running" says the same thing twice.
   Try: `Start with Tokyo's real rail lines, already built.`
   [ui/NewGame.tsx:180]

4. Loading failure screen
   Now: `No demand figures arrived in 3 minutes.` / `No answer came from the train simulation in 3 minutes.` / `The page stopped while starting (...)`
   Problem: things as subjects ("figures arrived", "an answer came"), no next step.
   Try: `Demand figures did not load after 3 minutes. Reload the page to try again.` (check whether a reload line already sits beside it)
   [ui/LoadingScreen.tsx:93,96]

5. Status tag by the top bar, hover; also the bracket notes on the train-share hover
   Now: `These figures are an early estimate. They settle when the forecast is finished.` / `These figures are from the last time this game was open. They are being worked out again.` / `These figures are for the network before your last change. They change once the new forecast is finished.` / `These figures are done. Demand elsewhere is still being worked out.` / `(rough while X is worked out in detail)`
   Problem: five lines open "These figures", each two sentences of one shape; "settle" is a metaphor; passives.
   Try: `Rough until the forecast finishes.` / `From your last session. Updating now.` / `From before your last change. New ones are coming.` / `Done for this area. The rest is still updating.`
   [ui/StatusTag.tsx:46-54; ui/TopBar.tsx:271]

6. Demand stopped warning, and the "too big for this device" dialog
   Now: `The demand model needs more memory than the browser gives it for this network.` / `This game reaches 12 zones and needs about 6 GB for each demand calculation. A browser gives each one at most 4 GB.`
   Problem: "reaches", "gives it", things doing the verbs; second sentence mirrors the first.
   Try: `This network is too big for the browser's memory.` / `This game covers 12 zones. Each demand calculation needs about 6 GB and browsers allow 4.`
   [ui/DemandStopped.tsx:9; game/memoryCheck.ts:43]

## Top bar

7. Top bar, speed buttons, hover while demand updates
   Now: `Held at 8x while demand updates, then 60x`
   Problem: telegraphic ("Held"), no subject.
   Try: `Slowed to 8x while demand updates, then back to 60x.`
   [ui/TopBar.tsx:202]

8. Top bar, speed slider, hover
   Now: `Clock speed, 1x to 3000x`
   Problem: repeats the label and the visible range.
   Try: drop (keep the aria-label)
   [ui/TopBar.tsx:222]

9. Top bar, train-share bar, hover
   Now: `24 billion trips/day in the world (8 billion people, about 3 trips each)` / `Train 0.4% of the 24 billion trips/day in the world`
   Problem: telegraphic ("Train 0.4% of"), "/day" shorthand.
   Try: `Trains make 0.4% of the world's 24 billion trips a day.`
   [ui/TopBar.tsx:234-239]

## Build tab

10. Build tab, ramp button, hover
    Now: `Ramp over track you cross at the same elevation, or under it from -1 down. Where the ramps don't fit, the crossing is flat.`
    Problem: first sentence is hard to parse, second repeats the idea.
    Try: `Ramp over track you cross, or under it from -1 down. Crossings stay flat where a ramp won't fit.`
    [ui/BuildTab.tsx:56]

11. Build tab, terrain button; Settings > Show > Terrain
    Now: `Show terrain. Construction is more expensive in steep terrain.` / `Construction is more expensive in steep terrain`
    Problem: "Show terrain" repeats the label; "construction is more expensive" is wordy.
    Try: `Steep ground costs more to build on.`
    [ui/BuildTab.tsx:59; ui/SettingsTab.tsx:111]

12. Build tab, snap to real network button
    Now: `Snap track to the real network`
    Problem: repeats the aria-label word for word.
    Try: drop
    [ui/BuildTab.tsx:70]

13. Build tab, Target max speed row, hover
    Now: `Long track pieces are laid out straight enough for this speed`
    Problem: passive, "laid out", "pieces".
    Try: `Track is drawn straight enough to reach this speed.`
    [ui/BuildTab.tsx:79]

14. Build tab, Platform length row, hover
    Now: `Platform length for new stations`
    Problem: repeats the label.
    Try: drop
    [ui/BuildTab.tsx:88]

15. Build tab, Select tool, hover
    Now: `Drag a box round track and stations to select several / Double click track to select it out to the junctions / Shift+click track to add the track on the way there`
    Problem: British "round", "select it out", "on the way there" is vague.
    Try: `Drag a box around track and stations to select them. / Double click track to select out to the next junctions. / Shift+click track to add the track between.`
    [ui/toolKeys.ts:15]

16. Blueprint panel, Disable button, hover
    Now: `Disable the selection so it will not be constructed (D)`
    Problem: two-part, passive.
    Try: `Leave these out when constructing (D)`
    [ui/Blueprint.tsx:214]

17. Track panel, disabled blueprint track, hover
    Now: `Will not be constructed unless enabled`
    Problem: passive, describes the visible state.
    Try: `Skipped when you construct. Enable it to build this.`
    [ui/inspector/TrackInspector.tsx:85]

## Lines tab and line panel

18. Lines tab (rows and totals) and line panel, riders and profit cells, hover while counting
    Now: `Riders are being worked out for the change`
    Problem: passive, "worked out" is used for 6 different things.
    Try: `Counting riders after your change.`
    [ui/LinesTab.tsx:48,95; ui/inspector/LineInspector.tsx:216,231]

19. Line panel, riders cell, hover
    Now: `The past 24 hours, counted train by train. Whoever does not fit on a full train waits for the next one.`
    Problem: "Whoever" is stiff, second sentence narrates.
    Try: `Past 24 hours, counted train by train. Riders who don't fit wait for the next train.`
    [ui/inspector/LineInspector.tsx:228]

20. Line panel, profit cell, hover (Cities tab has the same shape)
    Now: `Profit in the past 24 hours: $X, fares minus running trains and upkeep` then `Fares ...` `Running trains ...` `Upkeep ...` then `Upkeep of track and stations several lines use is shared by their trains per day.`
    Problem: the first line repeats the breakdown below it; last line is passive.
    Try: first line `Profit in the past 24 hours: $X`. Last line `Lines that share track or stations split the upkeep by trains per day.`
    [ui/inspector/LineInspector.tsx:233; ui/Cities.tsx:55]

21. Line panel and bus panel, load cell, hover
    Now: `Riders aboard on average, as a share of the 1,200 a train holds at crush, over every trip on a weekday. Each trip counts for as long as it runs.`
    Problem: one long sentence, "holds at crush", second sentence is hard to read.
    Try: `Average load on a weekday, as a share of the 1,200 riders a train holds at crush. Longer trips count for more.`
    [ui/inspector/LineInspector.tsx:238; ui/inspector/BusInspector.tsx:270]

22. Line panel, full trains and left behind, hovers
    Now: `3/40 trains left full` / `120 of the 3,400 riders who wanted a train were left at the platform by a full one`
    Problem: "left full" is telegraphic; second is passive.
    Try: `3 of 40 trains left full.` / `Full trains left 120 of 3,400 riders on the platform.`
    [ui/inspector/LineInspector.tsx:243-244]

23. Line panel, Trains and Headway rows, hover
    Now: `Set the number of trains on the line for each demand level` / `Set the headway for each demand level. The line runs the fewest trains that keep it over the round trip, and works them out again when the line or its track changes.`
    Problem: "Set the" narrates the control; the line is the subject doing the work; long.
    Try: `Time between trains at each demand level. The fewest trains that hold it are worked out for you, and again when the line or track changes.`
    [ui/inspector/LineInspector.tsx:938-939; BusInspector.tsx:404-405]

24. Line panel crowding row and bus panel figures row, hover while waiting
    Now: `Crowding appears when demand has loaded` / `Figures appear when demand has loaded`
    Problem: things as subjects, and two phrasings for one situation.
    Try: `Waiting for demand to load.`
    [ui/inspector/LineInspector.tsx:955; ui/inspector/BusInspector.tsx:420]

25. Line, bus and group panels, edit button, hover
    Now: `Edit the name, symbol and color (double click also opens it)`
    Problem: "double click also opens it" has the click as subject; bracket tail.
    Try: `Edit name, symbol and color. Double click works too.`
    [ui/inspector/LineInspector.tsx:895; ui/inspector/BusInspector.tsx:367; ui/inspector/GroupInspector.tsx:187]

26. Line panel, held-back trains hover; Trouble list, same fault
    Now: `These trains wait at the ends of the line instead of running, because no more trains an hour fit through the full spot.` / `The line's extra trains are held out of service at its ends`
    Problem: "instead of ... because" is the not-X shape; "full spot"; same fact written two ways.
    Try: `Extra trains wait at the ends of the line. The full stretch can't take more than 12 an hour.` (one wording for both)
    [game/capacityKinds.ts:63; ui/TroubleList.tsx:98; ui/inspector/LineInspector.tsx:592]

27. Bus panel, profit cell, hover
    Now: `Fares $X minus running cost $Y for the buses and upkeep of its route $Z: $N net`
    Problem: telegraphic run-on, and the train version is laid out as a list.
    Try: use the train layout: `Profit in the past 24 hours: $N` then Fares, Running buses, Upkeep on separate lines.
    [ui/inspector/BusInspector.tsx:265]

28. Notice when a platform is too short for a pasted schedule
    Now: `Trains of 8 cars on L: the platform at X fits no more.`
    Problem: colon-telegraphic; unclear what happened to the schedule.
    Try: `The platform at X is too short for 8 cars. L's trains stay at 6.` (check what the code really does first)
    [game/scheduleCopy.ts:91]

29. Notice after drawing a bus line or blueprint
    Now: `Bus line added: 12 km, $5M to construct.` / `Blueprint added: 12 km, $5M to construct.` / `Route changed: 12 km.`
    Problem: colon fragments.
    Try: `Added a 12 km bus line. It costs $5M to construct.` / `Added a 12 km blueprint. It costs $5M to construct.` / `Changed the route. It is 12 km now.`
    [map/busDraw.ts:665,675; map/tools.ts:774]

## Stations tab and station panel

30. Station panel, line row, "waiting" hover (the neighbour of the platform hover you disliked)
    Now: `Nobody waiting: no train calls here this hour` / `120 riders waiting for L now` then `A platform is full, or the last train left riders behind`
    Problem: colon fragment; second line is two clauses of one shape.
    Try: `No train stops here this hour.` / `120 riders waiting for L.` then `Some were left behind by the last train.`
    [ui/inspector/StationInspector.tsx:89-90]

31. Station panel, upkeep row, hover
    Now: `Each built station costs $X/day to keep up, at any elevation, whether trains stop or not. The lines stopping here share it by their trains per day.`
    Problem: stacked clauses, "/day", "the lines stopping here share it".
    Try: `Upkeep is $X a day for a built station, at any elevation, whether trains stop or not. Lines that stop here split it by trains per day.`
    [ui/inspector/StationInspector.tsx:329]

32. Station panel, Passage upgrade, hover
    Now: `Passage. Two stations up to 400 m apart joined by one are a complex, and riders change along the straight line on moving walkways. $X plus $Y a km, paid once both stations are built`
    Problem: three ideas in one sentence.
    Try: `Passage. Joins two stations up to 400 m apart into one complex, with a moving walkway between them. $X plus $Y a km, paid once both stations are built.`
    [ui/inspector/StationUpgrades.tsx:100]

33. Station panel, upgrade hovers (terminal, car park, waiting room)
    Now: `Trains turn in two sidings past the platform, up to 30 an hour, and the platforms stay free for other trains.` / `Riding to any of its stations costs 10 perceived minutes less.` / `Waits here count at 0.8 of their minutes when people choose how to travel.`
    Problem: "perceived" is a third name for effective time (see pattern 4); the last two are heavy.
    Try: `Riding from any of its stations feels 10 minutes shorter.` / `Waiting here feels 20% shorter when people pick how to travel.`
    [game/upgrades.ts:18-22]

34. Station complex map, platform hover
    Now: `L: 120 boardings, 300 riders on its trains here per day, 40 change to or from other lines, most with X (20)`
    Problem: one run-on line of figures; "per day" vs "a day" elsewhere.
    Try: break into lines: `L` / `120 boardings a day` / `300 riders on its trains here` / `40 change lines, most to X (20)`
    [ui/inspector/ComplexMap.tsx:185-190]

## Finances tab

35. Finances > Fares, high-speed multiplier, hover
    Now: `Rides on high-speed lines pay 1.5x the fare above the base fare. A trip that changes lines pays it only on its high-speed km.`
    Problem: "Rides pay" has things as subject; "pays it only on".
    Try: `High-speed lines charge 1.5x the fare above the base fare. A trip that changes lines pays that only for its high-speed km.`
    [ui/MoneyTab.tsx:115]

36. Finances > Fares, estimate hovers
    Now: `About 3,000 at these fares, 2,500 now. A guess until demand catches up.`
    Problem: "catches up" is figurative with demand as subject; fragment "About 3,000".
    Try: `About 3,000 riders at these fares, 2,500 now. Rough until demand finishes updating.`
    [ui/MoneyTab.tsx:151,154]

37. Finances > Trains and Buses, spare and running hovers
    Now: `Cars on the running lines at the demand level that needs most` / `Cars kept in the depot after a line shrank. A line that grows uses them before any are bought.` / `Cars are bought when a line needs more than the spare ones.`
    Problem: first is hard to parse; others use lines and cars as subjects and passives.
    Try: `Cars the running lines need at their busiest demand level.` / `Spare cars from lines that shrank. Growing lines use these before you buy more.`
    [ui/MoneyTab.tsx:243-271]

38. Finances > history chart, title and axis hovers
    Now: `Each game day, up to the past 30` / `The chart's top is $12M` / `The chart runs from -$2M to $9M` / `The chart's top is 40,000 train rides`
    Problem: the three chart hovers repeat the number printed beside them; the first is telegraphic.
    Try: `The last 30 game days.` and drop the three axis hovers.
    [ui/MoneyHistory.tsx:111-112,119,139,148]

39. Finances > fare graph, rate-step hover
    Now: `Where this rate ends and the next begins, km. Click to type a distance.`
    Problem: "begins, km" reads as a typo.
    Try: `Where this rate ends and the next begins, in km. Click to type a distance.`
    [ui/FareGraph.tsx:175]

## Cities tab, city panel, longer trips

40. City panel, Longer trips section, info hover
    Now: `Visits, shopping and business, for the areas the selection lies in, with the errands, visits, school and medical trips that cross a zone line nearby. Commutes are left out. The map's bubbles are their destinations outside this metropolitan area, on a separate scale from local trips.`
    Problem: one 50-word sentence, "are left out" passive, "on a separate scale".
    Try: `Visits, shopping and business trips from the selected areas, plus errands, school and medical trips that cross a zone line. No commutes. Bubbles on the map are the far ends, outside this metro area, on their own scale.`
    [ui/inspector/LongerTrips.tsx:149]

41. Longer trips and map view button; spelling
    Now: `Places outside every zone of the game; how these trips travel is not modelled` / `Nothing modelled here`
    Problem: British "modelled" (game says "modeled"?), semicolon, passive.
    Try: `Places outside the game's zones. How these trips travel isn't modeled.` / `Nothing modeled here`
    [ui/inspector/LongerTrips.tsx:61; ui/DemandSelector.tsx:36]

## Settings

42. Settings > Show > Track at capacity, hover (changed T-1843, still breaks)
    Now: `Marks where stations or track cannot smoothly handle scheduled train frequencies`
    Problem: "Marks" has the setting as subject; "smoothly handle scheduled train frequencies" is jargon.
    Try: `Places where track or stations can't fit the trains scheduled.`
    [ui/SettingsTab.tsx:93]

43. Settings > Show > Trains at capacity, hover
    Now: `Marks where trains are full and passengers cannot board`
    Problem: "Marks" again; "passengers" where the rest of the game says "riders".
    Try: `Places where trains are full and riders can't board.`
    [ui/SettingsTab.tsx:96]

44. Settings > Show > Drives, hover
    Now: `Show arcs for common driving trips`
    Problem: a command that repeats the label; says nothing the arcs don't.
    Try: `Arcs for common car trips, with the minutes a train would save.`
    [ui/SettingsTab.tsx:102]

45. Settings > Show > Real network, hover
    Now: `Draw real-world railways on the base map`
    Problem: command repeating the label.
    Try: `Today's real rail lines, drawn on the base map.` or drop
    [ui/SettingsTab.tsx:108]

46. Settings sliders (Size, Opacity, Volume, Panel and text size), hovers
    Now: `Bubbles at 100% of their usual size` / `Bubbles at 80% of their full opacity` / `All sounds at 50% volume` / `Panels and text at 100% of their usual size`
    Problem: restate the value the slider already shows; "full opacity" is ambiguous.
    Try: drop all four
    [ui/SettingsTab.tsx:49,50,132,219]

47. Settings > Demand bubbles > Detail, hover
    Now: `How many bubbles are aggregated together`
    Problem: "aggregated" is jargon, passive.
    Try: `How many nearby bubbles merge into one.`
    [ui/SettingsTab.tsx:53]

48. Settings > Station circles > day selector, hover
    Now: `Which day demand figures are shown for`
    Problem: passive, ends on a preposition.
    Try: `The day of the week the demand figures are for.`
    [ui/SettingsTab.tsx:74]

49. Settings > Units > US units, hover
    Now: `Miles, mph and feet. Off shows kilometers, km/h and meters.`
    Problem: "Off shows" has the switch state as subject.
    Try: `Miles, mph and feet. Kilometers, km/h and meters when off.`
    [ui/SettingsTab.tsx:149]

50. Settings > Units and performance > Use more cores, hover
    Now: `Up to 5 demand workers, using more of the processor and about 400 MB of memory each. Takes effect after a reload.`
    Problem: fragment openers, "Takes effect" has no subject.
    Try: `Runs up to 5 demand workers, each using about 400 MB of memory. Reload to apply.`
    [ui/SettingsTab.tsx:152]

51. Settings > Run in background, hover
    Now: `The clock, fares and running costs carry on at the chosen speed while this tab is hidden. Off, the game pauses until the tab is shown again.`
    Problem: "carry on" is British; two-part with "Off," as an opener.
    Try: `Keeps the clock, fares and running costs going while this tab is hidden. When off, the game pauses until you come back.`
    [ui/SettingsTab.tsx:155]

52. Settings > Game > switch to sandbox, confirm dialog
    Now: `Money cannot be turned back on in this game.` (buttons: Switch to sandbox mode / Stay in normal mode)
    Problem: passive.
    Try: `You can't turn money back on in this game.`
    [ui/SettingsTab.tsx:192]

53. Map view buttons (Lines / Height / Speed; Starts / Changes / Both), hovers
    Now: `Line colors, thickness by traffic` / `Track colored by its height` / `Riders at each station, blue where they start a trip and green where they change lines`
    Problem: colours as the subject and the last one is the closest kin of the hover that started this; narrates what the map shows.
    Try: `Riders at each station: starting a trip (blue) or changing lines (green).` The other two are fine or drop.
    [ui/DemandSelector.tsx:56-58,79-81]

## Map hovers and the trip explorer

54. Trip explorer, access and egress hovers; station upgrades
    Now: `On foot or by car, felt as 12 min on average over the area` / `crowding makes it feel 8 min` / `Effective time by train 40 min` / `10 perceived minutes`
    Problem: four names for one measure: felt, feel, effective, perceived.
    Try: pick "effective time" everywhere, or "feels like N min" everywhere.
    [map/tripView.ts:585,589,618; ui/inspector/TripInspector.tsx:36,58; game/upgrades.ts:20]

55. Trip explorer, route and fare hovers (about 25 in the panel)
    Now: `The model spreads this trip's train users over 6 pairs of stations near the two ends, the busiest pair taking 40%, and over the useful trains at each station. Routes under 0.5% of them are not listed.` / `Once a train is past 70% of its crush load, riders count each minute on board as more than one, 3 at crush and 5.4 at 120%.`
    Problem: model-explainer prose; one sentence each, several clauses, "the model" as subject. (The panel is for people who want the detail, so this may be fine.)
    Try: `Train users are split across 6 pairs of stations near the two ends. The busiest takes 40%. Routes under 0.5% are hidden.` I'd do a read-through of the whole panel with you rather than fix one by one.
    [ui/inspector/TripInspector.tsx:222,228,358; ui/inspector/MarketTrip.tsx:60,453]

56. Trip explorer / city panel, purpose source hovers (schools, airports, hospitals, attractions)
    Now: `Matched public schools and degree-granting colleges across the metro area. Private schools and exclusively online or nondegree students are not included. Homes, campus locations and typical-day attendance are estimated.`
    Problem: passives ("are not included", "are estimated") x4; reads like documentation.
    Try: `Public schools and degree-granting colleges in the metro area. No private schools, online-only or nondegree students. Homes, campuses and daily attendance are estimates.`
    [ui/inspector/PurposeBreakdown.tsx:78-81]

57. Train panel, "one rider" hover
    Now: `One rider picked from the riders aboard: where they got on and where they get off from the train's boardings, why they ride from this stretch's riders. Minutes are the trip explorer's effective times.`
    Problem: hard to parse, colon then two clauses of one shape.
    Try: `One rider picked at random from those aboard. Where they board and get off comes from the train's boardings, why they ride from this stretch's riders. Minutes are effective times from the trip explorer.`
    [ui/inspector/OneRider.tsx:109]

58. Train panel, load bar, hover
    Now: `412 riders aboard. The bar is full at 1,200 riders, and the tick marks the 300 seats.`
    Problem: bar and tick as subjects, narrates the visible.
    Try: `412 riders aboard. 300 seats, 1,200 at crush.`
    [ui/inspector/TrainInspector.tsx:107]

## Saves and account

59. Dock, red dot by Saved games, hover
    Now: `Saved only in this browser! / If your browser data is cleared, this game will be lost. You can make an account to save it online, or download a copy locally.`
    Problem: "!" on a warning, "locally", the second sentence is two options in one.
    Try: `Only saved in this browser. / Clearing your browser data deletes this game. Make an account to save it online, or download a copy.`
    [ui/KeepDot.tsx:13-15]

60. Saves panel, cloud icon on an account game, hover
    Now: `On your account. Open downloads it here.`
    Problem: "Open downloads" has the button as subject and reads oddly.
    Try: `On your account. Click Open to download it here.`
    [ui/SavesPanel.tsx:113]

61. Saves panel, game saved by a newer version, hover
    Now: `Make a copy to play on here. The copy leaves out what only the newer version can read.`
    Problem: "The copy leaves out" has the copy as subject.
    Try: `Make a copy you can play here. It drops what only the newer version can read.`
    [ui/SavesPanel.tsx:187]

62. Account panel, sign out and remove games, hover
    Now: `Games saved to your account leave this browser. A game with changes not on your account yet stays. The game you are playing closes.`
    Problem: three short sentences, each with a thing as subject.
    Try: `Removes games already on your account from this browser. Games with unsynced changes stay. The game you are playing closes.`
    [ui/AccountPanel.tsx:142]

63. Feedback form, after sending
    Now: `sent! thank you for feedback`
    Problem: "!" (you allow it for a real greeting), "thank you for feedback" drops a word.
    Try: `sent. thank you.` Probably yours, so only if you like it.
    [ui/FeedbackPanel.tsx:47]

64. Search box, hover and placeholder; Help panel
    Now: hover `Find a station, line or place (/ or Ctrl+F)`, placeholder `Search stations, lines and places`, Help panel `/, Ctrl K`
    Problem: "Find" and "Search" for one thing; hover and Help list different keys (the code accepts / , Ctrl+K and Ctrl+F).
    Try: `Search stations, lines and places (/, Ctrl+K or Ctrl+F)` in the hover, Help row `/, Ctrl K, Ctrl F`.
    [ui/Search.tsx:154,165; ui/HelpPanel.tsx:45]

## Errors, notices and Help

65. Build error when drawing track
    Now: `Track can only be single or double.`
    Problem: possibly out of date, since the Build tab offers Double, Quad and High speed.
    Try: check which case triggers it; if it is stale, `Track has to be double or quad.`
    [game/issues.ts:40]

66. Help panel and touch bar; British phrasing
    Now: `Pause or carry on` / `Take back the last piece of track or bus point` / `Take back the last point`
    Problem: "carry on" and "take back" are British; "bus point" is odd.
    Try: `Pause or resume` / `Undo the last piece of track or bus point` / `Undo the last point`
    [ui/HelpPanel.tsx:24,42; ui/TouchBar.tsx:20]

No items from Credits (licence wording is set by the data sources) or from privacy.html and terms.html (plain, no exclamation marks, no "we're").

## Patterns seen repeatedly

1. Hovers that repeat the label or the number beside them (items 8, 11, 12, 14, 38, 44, 45, 46, 58; about 12 hovers). Rule: drop, or say something the control doesn't.
2. A thing as subject doing a verb: "Marks where", "Off shows", "Open downloads", "The bar is full", "The copy leaves out", "Rides pay", "the line runs", "figures arrived" (items 4, 5, 23, 25, 35, 42, 43, 49, 58, 60, 61, 62; about 20 strings). Rule: name the person or use the plain statement.
3. Long explainer hovers with three or more clauses (items 10, 21, 23, 31, 32, 40, 55, 56, 57; about 30 hovers once the trip explorer is counted). Rule: one idea per hover, split the rest to a second line.
4. One measure, several names: effective / perceived / felt / feel (item 54); riders / passengers / rides / trips; Search / Find; worked out (used for about 8 different things).
5. Passive "could not be" errors with no next step (items 1, 4; about 12 strings). Rule: say what happened and what to try.
6. "/day", "per day", "a day", "an hour" mixed (items 9, 31, 34).
7. British or idiom words in an American-spelled UI: modelled, round, carry on, take back (items 15, 41, 51, 66).
8. "These figures ..." opens five lines in a row (item 5); "Set the ..." / "Show ..." commands on hovers for toggles and sliders (items 11, 23, 44, 45).
