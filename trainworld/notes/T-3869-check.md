# T-3869 check (savechk), one round

Tree savefail (rebased on master), WASM built with agent_cargo.py wasm, `npm run check` clean. Vite 6175 (tree) and 6176 (main tree, read only; its WASM may be older than the tree's).
Scratch: scratchpad/savechk (dl.js, hook.js, log.js, cmp.py). Screenshots: artifacts/trainworld/t3869chk-download-error.png, t3869chk-after-replace.png.

## 1. Ordinary saves: PASS
T-007-real-nyc.save, loaded, trips final (895,752,175 trips/day, 12,558,064 rail, 109 lines, 1756 stations on both), then exportGame download.
Tree: 91,490 bytes, sha256 ac6133a5...; master: 91,490 bytes, same sha256. Byte for byte equal (twice on the tree, same hash).

## 2. Memory path (?debug=1&saveRoom=all): PASS
Game opened by "Open a file" (a real slot). Save, Make a copy and Download each succeeded after one worker replacement
(log: "a save found no memory; the demand workers are replaced" once per action, 2 new demand workers each time).
Download of the copy, taken after the trips were final again: 91,490 bytes, same sha256 as master. Trips after replacements: same
895,752,175 (paused game). Download taken before final differs from master only in stars/achievements/demandFine (not yet computed then).
Error string: with the page's "roomMade" answer swallowed (tw.client.post wrapped) Download fails after about 20 s, console "the game could not be downloaded",
and the banner "The browser ran out of memory saving this game. Close other tabs, then try again." shows. No other string appeared.
The debug mode cannot refuse after the replacement, so the string only shows with the answer blocked; the autosave failure shows the same string.

## 3. Autosave loop: PASS
Fresh game, clock running, debug mode: autosave replaced the workers at t=1 s; then failed every ~62 s (60 s autosave + the 1 s retry) without
replacing (shown as the same banner); the next replacement came at t=621.5 s (620 s later). Nothing spun. Forced saves (Save, copy, Download) replace every time, which also resets the 10 minutes.

## 4. Big save na-day-491.save: PASS (one explained header field)
Both on 175 lines, 2240 stations, 6877 edges, final 1,715,267,314 trips/day on both. Downloaded 729,204 (tree) vs 729,203 (master) bytes.
Decoded: network bytes identical (352,556), header identical except `level` 31 vs 30 (`TrackApi.save_level()` from the WASM build; the main tree's
WASM was not rebuilt after the rebase). Not from the codec. Rerun master with a rebuilt WASM if this needs to be certain.
Reopening the downloaded file was not tried; the content equals master's.

## 5. Workers after replacement: PASS with a gap
After each replacement the pool re-solved and reached final with the same figures (paused: exact; running clock: 895,750,305 vs 895,752,175, game time moved).
Four to five replacements in one session: still solved, never stuck (computeN halves to 1 and stays).
Gap: `restartForSave` does not set `memoryResolve`, so the "Calculating trips again after freeing memory" line did not show after a save-triggered replacement
(StatusTag shows it only from the paths that set memoryResolve). The ordinary calculating bar was also not visible in the one screenshot taken 1 to 3 s after
the replacement (solves were fast on the NYC save). Decide whether the line is wanted here.

## Other
- Each forced action halves the solving workers again until 1; fine, but a player who saves a few times in a short session keeps 1 worker.
- Skipped: nothing; reopening the downloaded files in a fresh game was replaced by decoding them and comparing with master.
