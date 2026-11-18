# Refactor survey, 2026-11-18 (T-3725)

Agent refactorlist. A survey only: no code changed. The areas below are the parts of the app that confused agents
or players most often in the last three weeks. They are ranked by evidence: bugs traced to the area (ARCHIVE.md since
2026-10-28, grepped), task notes, agents' reports, file size and how often each file was edited. Each one has a
rough plan and a size in hours of agent time. For comparison: T-2381 split two 5,000-6,000 line Rust files in one
agent session, and T-2848 plus T-2849 (two of T-2843's state machine fixes) took one agent one day.

Track is left out. The track model, the drawing tools and single track will be replaced by the rework (T-3444,
Anita 2026-11-18: "we might not want to refactor track cuz we're ognna fully rework track"). It is still the
busiest place for bugs. The single-track line alone had 25 archive entries in three weeks, and T-3647 was a
regression caused by T-3386. That code is skipped, and a rework will replace it: the sim's track files, most of the
track tool, the track tool's held runs and drags (T-2843 machine F and its findings 5 and 7), and the track parts of
the T-2383 and T-2374 splits (renderer and network drawing).

The agents' reports had little to say about getting lost: a word search found a few lines, none about a code area.
The evidence below comes mostly from what bugs were traced to.

## 1. Who may save the game, and which game the page thinks is playing

**What it is.** The clock worker decides when the open game may be written to its saved slot. Several flags each say
"don't write now" for a different reason: another tab has the game, another device has it, a newer build saved it,
storage is broken, a game is being swapped, a test page. Four separate places each check their own mix of those
flags. Separately, the page and the worker each keep their own idea of which game is playing, and during a load (a
file opened, a backup restored, a Community network viewed) the two disagree for a second or two.

**What goes wrong.** Saves are lost, or a game is marked wrongly on the server for good. Every fix so far added one
more flag or one more "which slot was this for" tag in one place.

**Evidence.** Today alone, four bugs here:
- T-3687: a normal game was latched as a sandbox on the server for good. The score send ran during a load, with the
  new game's money and the old game's slot.
- T-3611: an autosave from a tab that had lost the game overwrote the other tab's save.
- T-3698 (open): a refused manual save says the game could not be saved.
- T-3722: leaving during a file's open dropped the file.
- Earlier: T-3478 (2026-11-16, leaving while a game opens) and T-2843 findings 4 and 8.
- Finding 4 still looks unfixed: the leave check still tests "anything unsaved" where it should test "a change the
  player made". I read this, not ran it.

Most of these bugs were not caused by the four separate checks. What they share is a message or a write that did
not say which game, slot or edit it belonged to. T-3731's lost stop clicks were the same kind of bug outside saving.

**Plan, in small stages** (the clock worker file takes about 7 commits a day, so each stage lands on its own):
1. Make the two scratch probes into standing checks first: T-3611's two-tab save race and T-3687's score send during
   a load. This is the code that loses saves, so the checks come before the refactor. The save-safety check
   (`t1826-saves`, 17 items) is the place to add them. Both probes are awkward: T-3611's pauses the worker just before
   the write, and T-3687's end-to-end run needs a local account server, so a standing check may only reach the page's
   side of it. 1.5-2 h, a guess.
2. Add one "may I write, and if not why" answer, used by all four places (T-2843 machine C). This fixes finding 4,
   takes in T-3698 (a refused write and its message) and changes nothing else. 1-1.5 h.
3. Give every message about a game one stamp: which game it is and which load it came from. The stamp is taken when
   the message is made, and only a change of game drops a message. T-3731's bug was a guard that dropped clicks for
   less, so nothing else may drop one. This replaces the one-off tags added today (the money's slot from T-3687, the
   stop clicks' game count from T-3731). Every game opened reloads the page, so the window is the second or two of a
   load. 1.5-2 h, a guess: I did not count how many message kinds carry game state.
4. Move the save, slot, lease and cloud code out of the clock worker file into its own module: about 31 functions
   plus their state. 1-1.5 h.

**Size.** About 5-7 h in all. **Matches** T-2843 (machine C, with A's leftovers and E's "which game is loading"),
T-3698, and the save part of T-2374.

## 2. Which numbers are shown while trips are being calculated

**What it is.** After an edit, the demand pool sends partial results first (a quick local solve, a solve of just the
zone in view), then the full one. Each line's riders, fares and points are shown, and paid, from whichever result
is newest. Two questions are answered in many places: may this partial result replace what a line shows, and which
old figures are kept meanwhile? The first is answered in the demand pool's publish paths and in a fare rule on the
page. The second is answered by three separate kept copies, in today's riders, the clock client's income and the
money module. Two timers (8 s and 30 s) cover the gaps.

**What goes wrong.** Players see numbers that are wrong for a while, and are paid from them: fares dipping toward 0,
or a partial result's guess paid out as real money.

**Evidence.**
- T-3303 (P1, 2026-11-12): a fare exploit reported by a player. A quick local result was stamped with the new fares
  before its trips were solved at them. On the real save, cash went from 6e9 to 3e18 in 54 game hours.
- T-2793 (2026-11-09): a zone-in-view result gave a long line one zone's riders and was paid. Tokyo Hikari fell from
  $1,296M to $19.9M a day for about 110 s.
- Each was fixed with its own guard in a different place: one in the page's fare rule, one in the demand pool's
  publish paths.
- T-3644 (today) had to line the on-screen score up with the leaderboard send's own gate.
- T-3597 (today): seven separate calculating tags had to be worded alike.
- T-3719 (today): an agent's session went into "the panel shows the old count, then drops it" (not a bug).

**Plan.** Each result says what kind it is (local, zone only, full) and what fares it was solved at. One rule decides
whether it may replace a line's shown figures, and both the demand pool and the page use it. One module owns each
line's state: settled, waiting since edit number N, or solved but not yet counted. It also keeps the one copy of its
last settled figures, which the panels, money, score and edit note all read (T-2843 machine D, extended to the demand
pool's side). Before and after, compare the figures on the real save: they should match exactly. Keep the T-2793 and
T-3303 checks passing.

**Size.** 5-7 h, in two stages (the one replace rule first, then the one kept copy). Riskier than the rest, because
money reads it. **Matches** T-2843 machine D. Its precondition,
T-2793 finishing, is met.

## 3. Is demand working, and is a game open

**What it is.** A handful of signals together say "the game is loading", "demand failed", "the graphics stopped" and
"a game is open". One of them stores two unrelated facts: it holds "graphics" next to "memory" and "error".
Another, named `startup`, really means "the first game's trips were first calculated".

**What goes wrong.** Code reads a signal as meaning something it doesn't. A button goes missing, or a demand failure
is hidden behind a graphics notice.

**Evidence.**
- T-3722 (today): the exit to menu button was gated on `startup`, so it was missing for up to 100 s on a big save.
- T-2843 finding 3: T-2849 (2026-11-09) made a real failure replace the graphics notice. It also noted that the
  separate signals were never built, and one signal still holds both facts.
- These signals are read 40 times across 14 files, 17 of them inside the loading module.

**Plan.** One "game phase" signal (menu, opening, playing, failed) and one "demand health" signal (loading, fine,
stopped and why), plus a separate "graphics lost". Only the loading module writes them, and everything else reads
them (T-2843 machines B and E).

**Size.** 2-3 h. **Matches** T-2843 B and E. This goes well right after area 1, whose stage 3 touches the same load
signals, and it makes area 2 a little smaller.

## 4. The two giant files: the clock worker and the demand pool

**What it is.**
- The clock worker file (5,169 lines) holds the network, previews, money, autosave, cloud saves and save pictures.
- The demand pool (5,431 lines) runs on the main thread. It decides how many demand workers to use, which zones they
  open, and how answers come back.

**What goes wrong.** Agents have to read thousands of lines to change one thing, and changes collide. The demand
pool's solve rounds and levels of detail were never reviewed (T-2843, "Not covered").

**Evidence.**
- The clock worker is the most edited file in the project: 143 commits in three weeks. It grew 290 lines since
  2026-11-16 and has 145 variables at the top level of the file.
- The demand pool grew 440 lines since 2026-11-16.
- Rough count of bug-fix commits (commits come in batches, so this is loose): 31 touched the clock worker and 19 the
  demand pool.

**Plan.** For the clock worker: area 1's stage 4 takes out the save part first, then money and the ledger, then the
save pictures. Leave the edit and preview handling alone, since the track rework will rewrite it. Split the demand
pool by job: the worker pool and its memory, the solve rounds, the figures sent out, and the market rows. Use the
same method as T-2381's kernel split.

**Size.** 3-4 h per file, after area 1. **Matches** T-2374 (without its renderer part, which is track drawing).

## 5. The same rule written twice: the trip panel's route pick

**What it is.** The trip panel picks a route to list. The local trip panel and the intercity trip panel each have
their own copy of that loop.

**What goes wrong.** A fix to one copy misses the other, and the panel says "No route" while trains carry riders.

**Evidence.** T-1500 (2026-10-26) gave the intercity loop its rule and missed a case. T-3724 (today) fixed that case
in the intercity copy only. The local copy still has the same gap (T-3724's note, "Cousin"): T-3730 is open, and
agent localroute has it now.

**Plan.** One route pick in the demand code that both panels call, with T-3724's test run against both.

**Size.** About 1 h, best done inside T-3730 now. The code tour lists other doubled jobs: the schematic's sheet maths
in Rust and JavaScript, and car street speeds written twice. A test keeps the street speeds equal, and the sheet
maths is meant to match bit for bit. No bug was traced to either in three weeks, so leave them.

## 6. Stop edits from the line panel

**What it is.** Stop clicks on the map now wait in one queue and are sent in order (T-3731). The line panel's own
stop buttons still have their own path. They ignore a second click while one is out, and they build the stop list
from what is on screen.

**What goes wrong.** Quick edits in the panel can be dropped or overwrite each other.

**Evidence.** T-3731 (P1, today): players lost stop edits. The real save lost 1 of 8 quick clicks, and a big save
with a slowed worker lost 6 of 8. T-3732 (open) covers the panel.
T-2766 (2026-11-11) built the first queue.

**Plan.** Route every panel stop edit through T-3731's queue, with the edit worked out when it is sent. Agent
panelstops is doing this now in T-3732. What's left is to check that it ends with one path. Go no further: stop
picking is part of the track rework (numbered circles per track).

**Size.** 0-1 h beyond T-3732. **Matches** T-3732.

## 7. Checks that test old rules

**What it is.** About 320 standing checks, 283 of them named after the task that made them. Nobody can tell which
ones still test current rules.

**What goes wrong.** Agents spend sessions chasing "bugs" that are really out-of-date checks.

**Evidence.**
- T-3700 (today): an Opus session found no game bug, only a check still testing a rule removed on 2026-11-17.
- T-3697 (open): old-layout diagram checks have failed since T-2590.
- A track-drawing check has failed on master through three agents and is still not filed.

**Plan.** Run every check on master once. Retire or rewrite the ones that fail, and give each a line in the checks
index saying what it protects. Leave the track checks for the rework. This is not a code refactor.

**Size.** 2-3 h. **Matches** T-3697.

Not ranked: main.ts and the debug flags (T-2375). The flags file was edited 45 times in three weeks, but no bug was
traced to it.

No area matches T-2382 (pipeline and pack_gate small items) or T-2383 (sim refactors: T-1114 sim items plus the
market's `pass_window` and `realcorridors.build`). I did not assess either. `pass_window` is demand code, not track,
so the track skip does not cover it.

## Recommendation

**First: area 1, who may save the game and which game is playing.** It had the most bugs (four today, two more this
month). Its failures are the worst kind: a lost save, or a game latched on the server that only an inbox fix can
undo. Each fix so far added one more flag. Doing it also shrinks the clock worker file. Start with stage 1, the two
standing checks.

**Second: area 2, which numbers are shown while trips are calculated.** This is where players were paid wrong money:
the P1 fare exploit and the Tokyo fare dips this month. Each was fixed with its own guard in a different place, so
the next kind of partial result can slip past both. It is bigger and touches money, so it comes after area 1 has
settled. If there is room before it, area 3 (2-3 h) makes it smaller.

Do area 5 now inside T-3730, and check area 6 when T-3732 lands. Both are already in progress and cost little.
