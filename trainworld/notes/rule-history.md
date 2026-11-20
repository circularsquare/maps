# Where some rules came from

The rules themselves live in `trainworld/CLAUDE.md`, `docs/orchestrator.md` and the maps skills. This file keeps the
dates and the incidents behind them, moved out of those files on 2026-11-20 so the rules read shorter. Grep it;
don't read it whole. Add a line when a rule is added for a reason worth remembering.

## trainworld/CLAUDE.md

- Estimates, no deadlines unless she gives one: Anita, 2026-11-17.
- A player request alone is no reason to change what can be built (350 km/h high speed): Anita, 2026-11-12.
- Speed and waiting wording ("loads", "shows up", "calculating trips"): Anita, 2026-11-17.
- Changelog items are not asked about at landing; she reviews them on the board: Anita, 2026-11-19. A publish still
  waits for her OK on new player text (Anita, 2026-11-20; v0.3.1 went out on 2026-11-12 with lines she had not seen).
- "we need to be less willing to put stuff on the changelog": Anita, 2026-11-18.
- Changelog tense (past for the change, present for how things are now): Anita, 2026-11-17.
- No changelog item for a change too fiddly to describe (undo after a load's cap lowering): Anita, 2026-11-17.
- No changelog item for a tooltip, alert wording or diagnostic text: Anita, 2026-11-18.
- Text copied from her own message is not reviewed: Anita, 2026-11-11. Text reused from the game: 2026-11-18.
- "Manually", never "by hand" ("by hand is a bit weird"): Anita, 2026-11-18.
- "turn around", never "turn round" ("its a pet peeve of mine"): Anita, 2026-11-12.
- Task fields very short, written to her: Anita, 2026-11-16. Rewrite a task to the part that's left: 2026-11-18.
- Nothing may open a window on her screen (`windowsHide`, `CREATE_NO_WINDOW`): background children were flashing
  terminals, 2026-11-18.
- Lean testing ("a bit too test heavy"): Anita, 2026-11-17.
- Local demand factors allowed again where published figures show a gap: Anita, 2026-11-11.

## docs/orchestrator.md

- Edit the handoff with the turn's other bookkeeping, not once per message: Anita, 2026-11-09.
- No Fable as a model unless she asks: Anita, 2026-11-09 and 2026-11-17. Advisor trial: 2026-11-16.
- Reports capped at 15 lines: Anita, 2026-11-17; reports had been running 30-60 lines.
- Stop background work before the final report: on 2026-11-17 an agent left with background work sent its report
  five times.
- Three versions on one sheet when she asks for a look to change without numbers: Anita, 2026-11-17, T-3586.
- Cycling at 500k: Anita, 2026-11-16, number set 2026-11-17. Overnight, keep going past it: 2026-11-05.
- Never go quiet on her: 2026-11-17, she waited 25 minutes for an answer during a landing.
- Duplicates may be folded without asking ("its fine to fold in duplicates if you think its a duplicate, since
  we're not ognna be losing that much info"): Anita, 2026-11-16.
- Triage of player reports: Anita, 2026-11-12.
- Ask cards short, with a title: Anita, 2026-11-17 (cards were too long).
- For you carries only new asks ("stop repeating things unless i ask"): Anita, 2026-11-17; made the rule for every
  session in her global file on 2026-11-20.
- Player request groups that are clearly done or duplicates are settled without her ("just don't auto address plyer
  requests where we have to make an actual new design decision"): Anita, 2026-11-16.
- Checker before landing risky changes: Anita, 2026-11-16, T-3451. Concise checkers: 2026-11-17.
- Commit board-related serve.py changes ("yes please commit board-related serve changes from now on"): Anita,
  2026-11-16. Push regularly ("feel free to be pushing this regulrly"): 2026-11-16; one wording for every
  committing session (commit freely, a couple of pushes a session) on 2026-11-20.
- snap8800 and serve.py restarts allowed without a prompt ("serve.py restarts are totally fine. they only affect
  me"): Anita, 2026-11-16.
- She edits the changelog before a publish ("ill amost always want to edit it unless we only have like 2 things in
  the update"): Anita, 2026-11-18. Highlights: 2026-11-17, T-3591. Sort before her final check: 2026-11-19.
- Playtest list before every publish ("maybe we were a bit too hasty to deploy last update ... before every update
  deploy, please give me like a full list of things to test"): Anita, 2026-11-19.
- New zones image: Anita, 2026-11-18, T-3614. Open-source sync under the publish yes: 2026-11-12. PR reply style:
  2026-11-13.
- shonei-server push covered by the deploy yes: Anita, 2026-11-18.
- Feedback inbox read-only, and reading a player's saved games only when a report can't be settled otherwise:
  Anita, 2026-11-12.
- After a publish, look for what to start next ("sometimes we start publishing and then just kinda dont take on
  much work at all and mostly sit idle, when in reality theres lots to do"): Anita, 2026-11-20.
- Handoff file under 60 lines: Anita, 2026-11-20 (the skill had said 40 and the check 80).
- Deleting a tree folder by hand: on 2026-11-12 it deleted through the `node_modules` junction and the main tree
  lost `.bin` and 15 packages ("Cannot find native binding"); `npm install` from trainworld/app put them back.
- Staging part of a JSON file: a line-level stage put two keys after the closing brace and snapbuild failed
  (commit d4164171).
- One line to her when winding down: on 2026-11-17 she was getting the same news from two tabs.
- Keep the work moving ("there are times when we have lots of stuff to potentially work on but we're only doing a
  bit at a time, or stopping"): Anita, 2026-11-03.
