# trainworld — Anita's todo

## Decide

- [x] No signals for now, with tangles discouraged by capacity and the door left open for signals
      later
- [x] First region: Northeast US, New York first
- [x] One train kind
- [x] Towns can become full cities mid-game
- [x] Rust for the simulation
- [x] Accounts and shared saves: shonei-server, for now
- [x] Name: anitabuilder (front-facing; trainworld stays the folder and code name)

## Do

- [x] Install Rust. Done 2026-10-08, Rust 1.99.
- [ ] Optional: a few screenshots of things that feel right for the look in
      `trainworld/moodboard/`, if the mock (below) misses something.
- [ ] Next time you play Subway Builder or NIMBY Rails, note the moments that feel laggy or
      unsatisfying. They become test cases.

## Name ideas

- Kippu (ticket)
- Tsugi (as in "next stop")
- Norikae (transfer)
- Ekimae (in front of the station)
- Little Lines
- Through Service

## Added by agents

- [x] Track (T-008): double track by default, single track as a setting
- [x] Water price (T-008): no ground level over water; bridges and tunnels 2x
- [x] Look (T-009): all three themes kept, Zen Maru Gothic, notes in DECISIONS.md
- [x] Trains layer (T-015): overlay preferred if the panning lag can be fixed
- [x] Station access: one access mode, a fast walk at about bike reach; no driving or buses for now
- [x] Train speed: faster than real is fine
- [x] Construct is final, no free undo
- [x] Money pace: economy at 100x
- [x] Play the loop (T-025): feels okay; build tools into the Build tab, track through the clicked
      points, junction setting unclear (all in batch 5)
- [x] Real New York save (T-007): looks good
- [x] Demand views (T-084): all of them (line width, station circles, train fill, commuter dots,
      station catchment)
- [ ] Look at batch 4 (screenshots `T-batch4-*.png`): do the side-by-side lines on shared track
      read right to you?
- [x] Commuter bubbles (T-097): RGB mix kept, sizes good (a size slider added, T-099); "Track" is
      fine as the tool's name; trains hiding busy stations is fine (zoom in), T-086 dropped
- Saves (T-029), for reference: the game continues where you left it. Settings (gear) has "Save
  to a file", "Load a file" and "New game"; `?new=1` in the URL starts fresh.
  `T-007-real-nyc.save` loads New York's real network the same way.
