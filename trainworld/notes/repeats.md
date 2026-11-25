# Repeat asks

Things Anita has raised more than once. Before filing anything she asks, grep this file and ARCHIVE.md for it. A match
is a repeat: add her new words to its entry. The next brief carries the whole entry and starts by saying why the last
fix didn't reach her, read from the code, before any new fix. A repeat closes when she says it's gone on 8800, not on an
agent's measurement. Don't ask her to test one until we think it feels right on 8800 (her 11-25); mention progress
meanwhile.

## The loading bar sits at 90% for minutes

- Her words: 11-25 02:26: "still fills to 90% and then stops for a while and i have to wait minutes for the demand.
  this is literally like the 10th time ive brought this up"; the 11-22 call to show 90% as half the bar "doesnt seem
  to actually apply here". 11-25 02:35: "the cities that sit at 90% are inconsistent. they tend to be larger cities,
  but its nothing about the cities themselves. any city can sometimes load fast and sometimes slow."
- Claimed fixes: T-793 (10-21), T-884 (10-21), T-1195 (10-24), T-2552 (11-05), T-2638 (11-07), T-3876 (11-20),
  T-3881 (11-20), T-4051 (11-24), the half-bar shift (11-22, app/src/game/loadBarCore.ts). Open: T-344 [P4].
- Why they didn't hold: Gemini's read, artifacts/trainworld/gemini/ask-0226/loadbar.md.

## Edits on her Asia save take about a second

- Her words: 11-24 18:50: 1 s between track clicks, stations 1 s to show. 11-25 00:51: "the game reworking the whole
  network after each edit is really the target". 11-25 02:26: "like the 10th round of placing station fixes ... the
  first station lands, subsequent stations dont ... partition where we're currently editing from the rest of the
  network so that nothing scales with the save size ... deleting track still takes like a second to respond". 02:44: "creating a
  new line should really be instant ... ideally it falls out of a general edit speed solution ... nothing is changing
  anywhere on the map. deleting a line also seems like it should be instant."
- Plans and fixes: the T-4144 plan (11-23: show each edit at the click, the checked version in a third of the time);
  DECISIONS T-4097 (11-25: root cause first, drawing at the click deprioritized). Each later stage cut one step
  (T-4240, T-4214 A, T-4153 s1-5, T-4150 s1); the steps left still scale with the save.
- Why: no design took every step of an edit off the save's size; each stage removed one.
- Now: Gemini's end-to-end design, artifacts/trainworld/gemini/ask-0226/editscale.md, goes to her before building.

## Orchestrators drift heavy and cycle fast

- Her words: 11-25 02:26: "this orchestrator did not last very long. are we sure we're following normal orchestrator
  practices to stay light? i feel like i have to ask this every 4 orchestrators, we quickly drift."
- maps-3e (about 2.5 h): read whole reports, diffs and Gemini reviews itself, fixed tools itself, wrote long briefs
  inline, retried failed landings.
- Light: briefs in files, report heads only, review findings go to the fix agent unread, tool fixes go to agents,
  landing with `finish` only.
