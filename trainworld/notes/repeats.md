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
  11-27 19:16: "is it expected that the selected city sometimes still takes > a minute to load? im experiencing that. on using oen core."
  11-28 10:00: "im waiting for like 3 minutes for chongqing demand bubbles to load. is that expected? (ive asked this before, ideally you should just be able to find the answer)". Cause and fix: notes/T-4501.md (a city a new line first reaches waited for the whole recalculation).
- Claimed fixes: T-793 (10-21), T-884 (10-21), T-1195 (10-24), T-2552 (11-05), T-2638 (11-07), T-3876 (11-20),
  T-3881 (11-20), T-4051 (11-24), the half-bar shift (11-22, app/src/game/loadBarCore.ts). Open: T-344 [P4].
- Why they didn't hold (T-4263, from her 11-25 diagnostics and the code): seven of them changed the bar's model, not
  the wait; T-4051 sped up a zone with no figures. Her Shenyang wait was a whole recalculation no question can answer
  from: 24 s after the load's first final the pool was cut 2 -> 1 (the load-time memory check left out the aux worker;
  fixed in T-4263) and everything was solved again on one worker for 4-6 minutes. T-4204 had stopped the same re-solve
  after a restart, not after a cut, and measured with "Use more cores" off, where this never fires; her tab had it on.
  Her, 11-25: "i feel like that might not be the whole issue, but its a good find." Still open: after any edit on her joined-up Asia save, a city's bubbles wait for the whole solve (T-4265).
  The bars themselves (T-4264): every bar ran a time guess into a 95-99% cap that the never-back rule then held, shown
  at 83-91%; the top bar's "Loading data…" for bubbles had its own 3 s guess to 95%, never the half-bar shift, and
  never looked at the running solve. Now a guess stops at half; past half only on the workers' own-pace time left.

## Edits on her Asia save take about a second

- Her words: 11-24 18:50: 1 s between track clicks, stations 1 s to show. 11-25 00:51: "the game reworking the whole
  network after each edit is really the target". 11-25 02:26: "like the 10th round of placing station fixes ... the
  first station lands, subsequent stations dont ... partition where we're currently editing from the rest of the
  network so that nothing scales with the save size ... deleting track still takes like a second to respond". 02:44: "creating a
  new line should really be instant ... ideally it falls out of a general edit speed solution ... nothing is changing
  anywhere on the map. deleting a line also seems like it should be instant."
- 11-25 04:18, on the hold stage: "i am not fully understanding it i think, but currently i am not a fan of this hold idea. if the finished look doesnt arrive until 500ms after a click or 5s after a click, thats basically not faster edit speed at all ... did we decide that theres no way to just silo off whatevers on screen and just have our edits only depend on the nearby stuff? why does that not work? i want to be able to tell within like 100ms whether there's are any conflicts with the click we've made, and i want it to confirm that the click goes through or it doesnt. any conflicts should be local, so there should only be a bounded amount of things to check. i dont understand why this still has to scale and take on the order of 500ms. if it's not possible, just say its not possible. i dont want you to hide things and say that you'll figure it out if we dont actually have a plan to fix it."
- Plans and fixes: the T-4144 plan (11-23: show each edit at the click, the checked version in a third of the time);
  DECISIONS T-4097 (11-25: root cause first, drawing at the click deprioritized). Each later stage cut one step
  (T-4240, T-4214 A, T-4153 s1-5, T-4150 s1); the steps left still scale with the save.
- Why: no design took every step of an edit off the save's size; each stage removed one. 11-25 the reworked design
  still put a timer (the hold) first and maps-c5 passed it on as a recommendation; her 04:2x: "i've literally asked as
  explicitly as i possibly could, that we should *only be targeting the root cause!!!! we should not be doing any of the
  cheap fixes!!"
- Now: Gemini's end-to-end design, artifacts/trainworld/gemini/ask-0226/editscale.md, goes to her before building.

## Orchestrators drift heavy and cycle fast

- Her words: 11-25 02:26: "this orchestrator did not last very long. are we sure we're following normal orchestrator
  practices to stay light? i feel like i have to ask this every 4 orchestrators, we quickly drift."
- maps-3e (about 2.5 h): read whole reports, diffs and Gemini reviews itself, fixed tools itself, wrote long briefs
  inline, retried failed landings.
- Light: briefs in files, report heads only, review findings go to the fix agent unread, tool fixes go to agents,
  landing with `finish` only.

## Too much review and testing during development

- Her words: 11-27 ~14:30: "i feel like we maybe rely too much on review, especially when we aren't actually ready to
  ship. this is a repeated complaint of mine. how might we make changes to speed up development time / require less
  review and testing during active development?" Earlier: 11-24 "review, not playtest"; the 11-24 testing budget.
- Adopted 11-27 (her yes to 1-5; notes/workflow-review-2026-11-27.md): development lands on its own tests, only save, slot and cash diffs get review, only save, edit-loss, crash and wrong-money findings are fixed then, no waiting for Gemini; one Gemini pass at release prep with notes/release-findings.md. Where: docs/orchestrator.md "Landing, commits, 8800", publish skill step 0, DECISIONS.md Tooling and process, T-4426 (occasional non-default settings check). Closes when she says development feels faster.
