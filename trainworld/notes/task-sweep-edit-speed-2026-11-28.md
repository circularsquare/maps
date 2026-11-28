# Edit-speed task sweep, 2026-11-28 (orchestrator brief, her 11-28 question)

Her words: "why are there 42 open edit speed tasks? ... do we really need this many tasks to represent this goal? are so
many of them really P1?"

## Count

- Before: 42 tasks tagged {L:edit-speed}: 8 P1, 17 P2, 12 P3, 5 P4 (37 counting as open; P4 does not).
- After: 24: 5 P1, 8 P2, 8 P3, 3 P4 (21 counting as open). No priority was changed; the three P1s that went were
  closed or folded into another P1.
- Why there were so many: T-4267's design (11-27) filed each of its roughly one-hour stages as its own task (T-4441 to
  T-4464), on top of the older parents (T-4097, T-4272) and leftovers of landed work (T-4309, T-4360, T-4429). Most
  folds below put a stage list back into one task per piece of work, as T-4153 holds its eleven stages.

## Closed (done or superseded)

- T-4097 Edits feel instant on big worlds: an umbrella for the whole line; every part is its own task, and its
  end-to-end measurement is T-4463's goal.
- T-1031 First ghost of a run takes 0.5-0.7 s: done by T-4143 (11-23), the first ghost shows at the first pointer move
  (8-95 ms) from a stand-in that never waits on the worker.
- T-4134 A line's time zone and fleet kind looked up once: already so on master. The shift is kept per line
  (`LineSvc.shift`, read by `line_shift` in wasm.rs) and worked out only for rebuilt lines (service.rs `build_on`;
  `reshift` only when the time zones are set at a load). T-4441 timed `cars_needed` at 0.8 ms and the whole
  `update_scoped` at 33 ms with no time-zone part, so T-4131's 75-120 ms is gone.

## Folded (closed into the task named; its fields rewritten to cover them)

- T-4272 (a click confirmed from what it touches) into T-4442: its goal, every click of a run within 100 ms, is what
  is left.
- T-4432 (owed settle split in two) into T-4442: the same wait (a later click behind the click before's settle), kept
  as the fallback; the design stays in notes/T-4281.md "Stage 4".
- T-4446 (trains noted only for lines reached, C4) into T-4442: it was after A1 and is in the same code (`pay_owed`).
- T-4464 (far lines settle a moment later, C7) into T-4442: built only if the trunk's train work stays over about 30 ms.
- T-4452, T-4453, T-4454 (passageway, Construct, New line answered at once) into T-4451 (Delete line): stages A2-A5
  are one piece, each edit answered right after its own train work.
- T-4429 (bands' last copies, B4) and T-4459 (sample tables per line, B5) into T-4153: the drawn network's last two
  stages.
- T-4309 (model calls, B1), T-4360 and T-4458 (rows leftovers, B2) into T-4291 (post, B6), retitled "The worker's state
  after an edit, what's left". T-4309 and T-4360 both named the line stamps.
- T-4350 (autosave at a quiet moment) into T-4457 (crash copy at a quiet moment), retitled "Whole-save copies taken at
  a quiet moment": both are a whole-save write a click can land on, with one fix.
- T-4448 and T-4455 into T-4447 (P4, paused): stages A6-A8, one click's confirm, all under 50 ms.

Tags brought in line: T-4445 now {after:T-4431} (the settle agent works on the same weekend capacity; was after
T-4444, landed); T-4451 {after:T-4442 and T-4445}; T-4462 {after:T-4153, T-4291 and T-4460}; T-4358, T-4352, T-4353
{after:T-4442} (were after T-4272; A1 changes the settle code they are about).

Left alone: T-4431 (agent settle) and T-4284 (agent pagestage5b), both taking.

## Proposed priorities (not applied)

Anita approved these 11-28 (through the orchestrator), applied the same day: T-4284, T-4153, T-4291 to P2; T-4457, T-4461 to P3; T-1211 and T-4142 to frame-rate, T-4151 to checks; T-4576 folded into T-4284.

First, one `edit_bench --steps` run on her Asia save on master: every figure the tasks quote is from 11-25 to 11-27,
and T-4444, T-4449, T-4504, T-4153 stages 10-11, T-4284 stage 5, T-4300 and T-4314 landed since. Whatever that run
shows under 50 ms goes to P4 by her 11-27 rule.

- P1: T-4431 (running): the owed settle a later click waits behind, 116-121 ms natively on the trunk.
- P1: T-4442: later clicks in a run, about 100 ms (p90 255, up to 350 on a 3-line trunk), the wait she meets most.
- T-4284 (running): P1 while pagestage5b runs, then P2 unless the Asia run shows the page's own work over 50 ms.
- T-4153 P1 to P2: stages 10-11 landed; left are 10-15 ms of copies and the sample tables.
- T-4291 P1 to P2: worker leftovers; the rows' 11-53 ms in Node is the biggest part.
- T-4451 stays P2, the first to move up after T-4442 and T-4445: a delete that breaks a line takes 2.2-2.6 s and
  Delete line 0.5-0.7 s, the slowest edits left.
- T-4445, T-4456, T-4460, T-4462, T-4463 stay P2.
- T-4457 P2 to P3 and T-4461 P2 to P3: occasional 60-100 ms waits, and 46-69 ms after a burst of edits.
- T-4576 (P3) folds into T-4284 once pagestage5b reports; it is in T-4284's "Left", and its cost grows with each edit
  since the snap grid's last whole build.
- T-4358, T-4352 stay P3 (bugs from T-4272's reviews, after T-4442). T-3804 stays P3; its snapshot item overlaps
  T-4461. T-3550 stays P3; its London figures are from 11-20, so one windrush run before anyone works it.
- T-1211 (panels open slowly), T-4142 (frames while drawing track during a calculation) and T-4151 (the bench's
  calculating pass) are not edit speed: proposed retag to frame-rate, frame-rate and checks. T-4142's
  {after:T-4114} is stale (T-4114 archived 11-23).
- P4 unchanged: T-4315, T-4353, T-4447.
