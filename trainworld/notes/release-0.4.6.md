# Release 0.4.6 check lists

Candidate: master f63b105b3. Last publish: v0.4.5 (b0899150e). Served at http://localhost:8800/trainworld/ (checked with serve.py list). Written by the release-lists agent, 2026-11-24. Nothing was run; this comes from git, the changelog list and her asks log.

## 1. Playtest list

All on http://localhost:8800/trainworld/. Riskiest first within each area. "Phone" means her Android phone.

Track
1. Station tool: pick Auto, Train station, Tram stop, BRT station and Port from the Station kind choice. Hover over rail, a tram line, a bus road and open ground. Check Train station is the default in a fresh browser profile, and that BRT stations place beside a road with two clicks on open ground (T-3999, T-4105, T-4135).
2. Make a bus line, switch it to BRT in the line panel, Construct, then add a second BRT line on the same road. Lanes should draw dashed, be paid once, and the bus inspector should show Busiest lane and shared cost (T-3378, T-3944).
3. Draw a busway (bus-only track) off the streets and run a bus on it. Try to run a train or tram and to put a station on it: both should be refused with a message (T-3378 stage 4).
4. Draw track with big cursor lag in mind: on a big world, the ghost should follow the pointer from the first click. Draw track onto the end of other track that is almost in line (should not be refused wrongly) and remove single track on a one-way line (T-4136, T-4152, T-4000).
5. Watch trams, trains and buses round a sharp curve: they should stay on the track (T-4075).
6. Ctrl+click with the bus tool to place a stop exactly where you point; Shift+click delete or remove to skip the question (T-4069, T-4068).
7. Phone: place track by touch in Firefox on Android (T-4076).

Edit speed (performance, but shown in every edit)
8. On a big save (Asia or North America): place and delete a station, construct, undo, add and remove a line stop, change a line colour. Each should show at the click; try a refused edit (station on a ramp, station already there) and check the red flash and that cash comes back. Undo after Copy line or New line should remove the whole line (T-4145 to T-4153, T-4185).
9. Pan to a new area of a big world while drawing: no multi-second stall (T-4155). Search for a far town while the game runs: no freeze (T-3997, T-4004).

Demand
10. Make parallel express and local lines between the same two stations: check the express gets a real share and trams at street stops lose no boarding time (T-4041). Check a transfer where the next stop is where riders came from (T-4064).
11. Bus stops: place stops 150 m and 500 m apart. Stops within 150 m connect on foot with a dashed line; an Auto bus line counts every stop; people walk up to 1 km to a bus stop; passage hover reads "Saves 1 minute on transfers" (T-4079, T-3924, T-3995, T-4158, T-4092).
12. Select a Madison bubble: trips to La Crosse should show and open in the trip explorer. In an island zone with no road or flight, the city panel should say No route (T-4140, T-2216).

Scheduling
13. A reversal partway along a line: icon and hover should show the limit only where it is the thing limiting the line; check a train line and a tram line (T-4178). Change a line to high speed or tram and check it takes that kind's default headways unless you had changed them (T-4104).
14. In the line panel stop list, make a stop boarding only, then getting off only, save and reload (T-4013). Make a headway line with an express and a local through a passing-track station: the express should not run through the local (T-3929). Change the train count on full track: the frequency should follow (T-3655).

Fares
15. Open advanced fares: only the line kinds and boat sizes in use should show, in two columns, with a BRT fare multiple. A high-speed, ferry or bus line's multiple should not read 1.00x (T-3915, T-4040).

Schematic
16. Blueprints: tick and untick Blueprints in Settings, and on a community save check blueprints start hidden. Turn the demand bubbles off after a big load (T-4003, T-4002, T-4029). Map credits sit folded in the (i) button (T-3998).

Community
17. Community tab: bookmark a network, find the Bookmarked tab, sign in and reload to see it kept; liked cards marked; Report is a flag icon with a "Report this save?" popup; no Discord link; leaderboards link. Open a community save without Calculate trips ticked and check it loads fast and paused (T-4005, T-4078, T-3990, T-4077, T-3851, T-4002).

Performance
18. Load the Tokyo real network and the New York save: after an edit, lines move before all trips are done; try the Calculate trips in less detail setting (off by default on desktop, on for phones) and compare a busy line's riders. Loading should fill the bar and not stick at 90% (T-4117, T-2624, T-4051).
19. Phone: load a big saved game. Check the memory warning, the zoom buttons being off by default (turn on in Settings), the coarse setting being on, and that the day and night strip shares the row with bubble settings. Check a game that ran out of memory can be opened without calculating trips (T-3940, T-4138, T-4117, T-4128). Her phone is Android, so the iPhone warning wording can't be seen: check it appears for a phone or tablet.

Other
20. Phone: play for 2 hours, then check the install note at the bottom of the map: Android shows Install or the menu steps, the x hides it for good. Use a second session to be quick: it only shows after 2 hours of play, so this is the one item that may need a clock change or a long session (T-4106).
21. Station upgrades: waiting room, passage and station building should show Upkeep per day, and the Money tab a Station upgrades line. Open a trip help panel from the trip panel's ? button: effective time wording and the train against car graph (T-3922, T-2264).
22. Download any saved game from the start menu, including one that won't open; open a save from a newer build: it should be refused and left alone. Kill the train model (rare) only if it is easy: a reload banner should appear (T-4126, T-4014, T-4109).
23. Try the new zones: Hokkaido as four regions, Muzaffarpur as three, Manila smaller, and Georgia, Armenia, Azerbaijan and Afghanistan (T-3688, T-3764). Add a Kansai or Nagoya real network and look for the Keihan main line (T-3935).

Memory and speed items with no step (check with perf_bench, not by hand): T-4094, T-4158, T-4093, T-3925, T-2486, T-3160, T-3939, T-4179, T-4184, T-4103.

## 2. Strings that need her OK

Source: git diff of trainworld/app/src/i18n/messages/en.json between b0899150e and f63b105b3 (96 keys added, 6 changed, 4 removed). I also scanned the .ts and .tsx diffs for hardcoded English and found none shown to players (one stored note in workers/cloud.ts is routed through the keyed text). Changelog lines are not listed. Where a string says "new" the old was none.

Changed:
- issues.nodeHeading:
  old "Track has to leave a junction along the existing track."
  new "Track alignment error"
- lineInspector.reverseTip:
  old "{vehicle, select, tram {Trams reverse here and stand {time}\nAt most 12 trams an hour reverse on each side} other {Trains reverse here and stand {time}\nAt most 12 trains an hour reverse on each side}}"
  new "{vehicle, select, tram {Trams turn around here and wait {time}.\nThis limits the line to {n, plural, one {# tram} other {# trams}} per hour.} other {Trains turn around here and wait {time}.\nThis limits the line to {n, plural, one {# train} other {# trains}} per hour.}}" (reworded by Gemini, T-4196)
- reversalMarks.reverses:
  old "{name} reverses here"
  new "{vehicle, select, tram {Turning around here limits {name} to {n, plural, one {# tram} other {# trams}} per hour} other {Turning around here limits {name} to {n, plural, one {# train} other {# trains}} per hour}}" (reworded by Gemini, T-4196)
- stationUpgrades.buildingTip:
  old "Station building\nJoins every station it touches into one complex so that people can transfer.\nChanging trains inside it feels 1 minute shorter.\nWaiting in it for a train feels 20% shorter.\nCosts {price}\n{act, select, stop {Click to stop} built {Built. Click to change its outline} other {Click, then change its outline on the map}}"
  new "Station building\nJoins every station it touches into one complex so people can transfer.\nChanging trains inside feels 1 minute shorter.\nWaiting inside for a train feels 20% shorter.\nCosts {price}\nUpkeep {upkeep} per day\n{act, select, stop {Click to stop} built {Built. Click to change its outline} other {Click, then change its outline on the map}}" (reworded by Gemini, T-4196)
- stationUpgrades.passageTip:
  old "Passage\nJoins two stations up to {distance} apart into one complex so that people can transfer.\nCosts {range} depending on length\n{picking, select, true {Click another station on the map, or here to stop.} other {Click, then click the other station on the map.}}"
  new "Passage\nJoins two stations up to {distance} apart into one complex so people can transfer.\nCosts {range} depending on length\nUpkeep {upkeep} per day\n{picking, select, true {Click another station on the map, or here to stop.} other {Click, then click another station on the map.}}" (reworded by Gemini, T-4196)
- troubleList.reverses:
  old "{name} reverses here"
  new "{vehicle, select, tram {Turning around here limits {name} to {n, plural, one {# tram} other {# trams}} per hour} other {Turning around here limits {name} to {n, plural, one {# train} other {# trains}} per hour}}" (reworded by Gemini, T-4196)

Added:
- blueprint.brtStations: (none) -> "{count, plural, one {# BRT station} other {# BRT stations}}"
- blueprint.lengthLevelBusway: (none) -> "{distance} busway, {level}"
- blueprint.baseCostBrtStation: (none) -> "Base cost {price} per BRT station"
- busInspector.brtBuilt: (none) -> "BRT built on {name} for {price}."
- busInspector.brtShared: (none) -> "{shared} of that is shared with other BRT lines and paid for once" (reworded by Gemini, T-4196)
- busInspector.lane: (none) -> "Busiest lane"
- busInspector.laneLoad: (none) -> "{n}/{most} buses per hour"
- busInspector.laneFullTip: (none) -> "Above {most} buses per hour, lines on this lane run fewer buses." (reworded by Gemini, T-4196)
- busInspector.laneHeld: (none) -> "{n, plural, one {# bus held out of service} other {# buses held out of service}}"
- busLayer.brtBlueprint: (none) -> "BRT lanes in the blueprint"
- capacityKinds.laneTitle: (none) -> "BRT lane"
- clockClient.reopenFailed: (none) -> "This network could not be reopened." (reworded by Gemini, T-4196)
- clockClient.startTimeout: (none) -> "The game's files did not finish downloading. Check your connection and try again."
- clockWorker.copyBeforeLeaving: (none) -> "Make a copy of this game before leaving, or download it as a file and reload the page." (reworded by Gemini, T-4196)
- clockWorker.couldNotSaveLeaving: (none) -> "This game could not be saved. Download it as a file before leaving." (reworded by Gemini, T-4196)
- common.shiftSkip: (none) -> "Shift+click to skip confirmation"
- retryNote.text: (none) -> "This game ran out of memory, so it loaded again with fewer cores."
- helpTopic.graphTrain: (none) -> "train"
- helpTopic.graphX: (none) -> "train time compared with car time"
- helpTopic.graphY: (none) -> "share of trips"
- helpTopic.graphDot: (none) -> "The dot is this trip, for the purpose and time of day under By transit."
- issues.buswayJoin: (none) -> "Busways can only join other busways." (reworded by Gemini, T-4196)
- issues.buswayStation: (none) -> "Stations can't go on busways. Place a BRT station beside one."
- issues.buswayTrack: (none) -> "Trains and trams can't run on busways."
- lineInspector.boardOnly: (none) -> "Boarding only"
- lineInspector.alightOnly: (none) -> "Getting off only"
- lineInspector.setBoardOnly: (none) -> "Set to boarding only" (reworded by Gemini, T-4196)
- lineInspector.setAlightOnly: (none) -> "Set to getting off only" (reworded by Gemini, T-4196)
- lineInspector.setBoardBoth: (none) -> "Allow boarding and getting off"
- lineMoney.fareKind: (none) -> "{kind, select, hsr {High speed} ferry {Ferry} bus {Bus} other {BRT}} {value}"
- lineMoney.farePaid: (none) -> "Riders on this line pay {value}"
- links.busStop: (none) -> "Bus stops can't have passages."
- marketTrip.cutOff: (none) -> "No road or flight connects these two places, and transit costs over 4 times what driving across would, so nobody takes it." (reworded by Gemini, T-4196)
- moneyTab.brtFare: (none) -> "BRT fare" (reworded by Gemini, T-4196)
- moneyTab.brtFareTip: (none) -> "Fare multiple for rides on BRT lines" (reworded by Gemini, T-4196)
- moneyTab.upgradesUpkeep: (none) -> "Station upgrades {share} of their price per day"
- savesPanel.memoryWarnPhone: (none) -> "This game is big for a phone or tablet and may run out of memory while loading."
- savesPanel.memoryWarnDied: (none) -> "This game ran out of memory the last time it loaded."
- settingsTab.coarseDemand: (none) -> "Calculate trips in less detail"
- settingsTab.blueprints: (none) -> "Blueprints"
- stationInspector.brtStation: (none) -> "BRT station"
- stationInspector.brtUpkeep: (none) -> "BRT station upkeep is {price} per day."
- stationInspector.upgradesUpkeep: (none) -> "Upgrades add {amount} per day." (reworded by Gemini, T-4196)
- stationUpgrades.brtBuiltIn: (none) -> "Every BRT station has one"
- stationUpgrades.passageSaves: (none) -> "Saves 1 minute on transfers" (reworded by Gemini, T-4196)
- stopMarkers.addExact: (none) -> "Ctrl+click to place without snapping"
- toolBar.stationKind: (none) -> "Station kind"
- toolBar.kindRail: (none) -> "Train station"
- toolBar.kindTram: (none) -> "Tram stop"
- toolBar.busway: (none) -> "Busway"
- toolBar.priceTipBusway: (none) -> "{level}, busway"
- tools.notBesideRoad: (none) -> "Place BRT stations beside a road"
- tools.freeStart: (none) -> "{touch, select, true {Tap} other {Click}} to start a station here"
- tools.freeTurn: (none) -> "{touch, select, true {Tap} other {Click}} to set the station's direction"
- tools.freePiece: (none) -> "{track} of new track, {cost}"
- tools.nearBus: (none) -> "{n} trips within {distance}"
- tools.nearBusOne: (none) -> "1 trip within {distance}"
- tools.nearOne: (none) -> "1 trip within 1{unit}"
- topBar.viewingNoTrips: (none) -> "{name} isn't saved. Trips aren't calculated." (reworded by Gemini, T-4196)
- topBar.noTrips: (none) -> "Trips aren't calculated for this game." (reworded by Gemini, T-4196)
- topBar.calculateTrips: (none) -> "Calculate trips"
- trackInspector.nameBusway: (none) -> "Busway"
- trackInspector.kindBusway: (none) -> "busway"
- community.reportTitle: (none) -> "Report this save?"
- community.reportSubmit: (none) -> "Send report" (reworded by Gemini, T-4196)
- community.bookmarked: (none) -> "Bookmarked"
- community.bookmark: (none) -> "Bookmark"
- community.unbookmark: (none) -> "Remove bookmark"
- community.noBookmarks: (none) -> "No bookmarks yet"
- community.viewTrips: (none) -> "Calculate trips"

Removed (no longer shown):
- savesPanel.downloadNewer: "Download as a file, as saved on the newer game version"
- savesPanel.saveNewer: "This game was saved on a newer game version. Reload the page to update."
- tools.notOnTrack: "Place stations on track"
- tools.nearUnder: "Under 50 trips within 1{unit}"

## 3. Acknowledged or reused

Her own wording (from her notes or commit messages; no "text acknowledged" reply for 0.4.6 text is in asks-log.jsonl or handoff notes, so this is her wording, not a reply):
- helpTopic.timeTitle and the helpTopic.time* lines: her word for word text, notes/T-2264.md "Her wording 11-21". helpTopic.timeTitle is reused from tripText.effectiveTime.
- installBanner.*: commit d4eabb5c6 says it uses her wording (her 11-23 ask for the banner, T-4106).
- settingsTab.coarseDemandTip: commit f4ed4c19a, "coarse setting hover in her words".
- installBanner.iosSafari: (none) -> "Safari can delete locally stored saves. You can install trainsworld as an app: tap Share, then Add to Home Screen"
- installBanner.ios: (none) -> "Your browser can delete locally stored saves. You can install trainsworld as an app: tap Share, then Add to Home Screen"
- installBanner.android: (none) -> "Your browser can delete locally stored saves. You can install trainsworld as an app."
- installBanner.androidMenu: (none) -> "Your browser can delete locally stored saves. You can install trainsworld as an app: press ⋮ and tap Install and Create Shortcut"
- installBanner.install: (none) -> "Install"
- helpTopic.timeIntro: (none) -> "Effective time is how long a trip feels in minutes. Fares and other expenses are converted into minutes."
- helpTopic.timeTrain: (none) -> "Train trip effective time"
- helpTopic.timeWalk: (none) -> "Walking to station: 1.5 min / min"
- helpTopic.timeWait: (none) -> "Waiting for train: depends on train frequency"
- helpTopic.timeRide: (none) -> "Riding train"
- helpTopic.timeBoard: (none) -> "Boarding/disembarking train: 2 minutes each"
- helpTopic.timeChange: (none) -> "Transfer: 2 minutes, plus walking time and waiting for next vehicle"
- helpTopic.timeFare: (none) -> "Fare: conversion rate depends on average income in the country"
- helpTopic.timeCar: (none) -> "Car trip effective time"
- helpTopic.timeDrive: (none) -> "Driving: 1.33 min / min"
- helpTopic.timePark: (none) -> "Parking: 7 minutes"
- helpTopic.timeParkPrice: (none) -> "Parking prices: scales a lot with population density, up to max of 80 minutes"
- helpTopic.timeOwn: (none) -> "Car ownership cost: high fixed cost, scales partially with trip length and average income in the country"
- helpTopic.timeWalking: (none) -> "Walking trip effective time is just 1.5 min / min"
- helpTopic.timeChoose: (none) -> "People take into account effective time of all modes, and decide between them with soft thresholds."
- settingsTab.coarseDemandTip: (none) -> "Trips are calculated faster. Ridership can change by a few percent on some lines"

Reused word for word from an existing string:
- common.upkeepDay: "Upkeep {amount} per day" (word for word from trackHover.upkeepDay)
- helpTopic.timeTitle: "Effective time" (word for word from tripText.effectiveTime)
- helpTopic.graphCar: "car" (word for word from tripText.termCar)
- toolBar.kindAuto: "Auto" (word for word from busInspector.spacingAuto)
- community.reportCancel: "Cancel" (word for word from oneWay.no)


## 4. Changed after the list above (11-24 afternoon)

Disembarking wording (T-4200, her ask; landed a521ebd05):
- lineInspector.alightOnly: "Getting off only" -> "Disembarking only"
- lineInspector.setAlightOnly: "Set to getting off only" -> "Set to disembarking only"
- lineInspector.setBoardBoth: "Allow boarding and getting off" -> "Allow boarding and disembarking"
- marketTrip.atAirportsTip: "...then getting off and out at the other end" -> "...then disembarking and leaving at the other end"
- oneRider.toTip: "Gets off at {name}, ..." -> "Disembarks at {name}, ..."
- trainStops.gettingOff: "getting off" -> "disembarking"
- trainStops.lastTip: "Everyone on board gets off at {name}, ..." -> "Everyone on board disembarks at {name}, ..."
- trainStops.stopTip: "{off} of the {riders} riders on board get off at {name}, and {on} get on" -> "{off} of the {riders} riders on board disembark at {name}, and {on} board"

Reversal strings back to "reverse" at her word (61cbe8015): "Trams reverse here and wait {time}." / "Trains reverse here and wait {time}." and "Reversing here limits {name} to ... per hour" (the line panel, the map's marks and the Trouble list).

Pending in a running agent (T-4204), asked 15:12:
- "Time capped at {cap}x while updating demand" -> "Time capped at {cap}x while calculating trips"
- "{speed} (time capped at {cap} while updating demand)" -> "{speed} (time capped at {cap} while calculating trips)"
