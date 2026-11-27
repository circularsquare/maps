# Workflow review: less review and testing during development (2026-11-27)

Her ask, 11-27: "i feel like we maybe rely too much on review, especially when we aren't actually ready to ship ... how
might we make changes to speed up development time / require less review and testing during active development?"
Proposals for her and the orchestrator to decide; no doc has been changed. Evidence is below them.

## Proposals

1. **Two modes.** Development is the default. Release prep starts when the orchestrator runs the `publish` skill (it
   already decides when an update is about ready; she isn't asked). The breadth review moves into release prep: one
   Gemini pass over everything since the last publish, split by area, then one fix round.
   Saves: most of the ~21 Gemini batch reviews and ~20 fix rounds of the last 4 days (see 3, 4). Risks: bugs sit on 8800
   for longer. Caught by: her play on 8800, the release review, smoke, the crash counter (T-4409).
2. **Review by risk class instead of by batch.** During development, review only a diff that changes what a save writes
   or reads (the save level, blocks), writes, clears or reloads a slot, or settles cash. Those keep the fresh Sonnet
   saves review. Everything else (edit speed, drawing, demand timing, UI, tools) lands on its own tests and goes into
   the release review. Saves: the "own review before landing" for "sim-core changes others build on" today covers almost
   all edit-speed work (review23-29 on 11-25). Dropping it takes 1-3 h of waiting off each such stage. Risks: a wrong
   figure on 8800 until release. Caught by: Rust tests, the figure diff in the budget, release review.
3. **Fix during development only what can't wait.** That is: a save that won't load, lost edits or data, a crash, or
   wrong figures or money that get written into a save. Every other finding (low, "heals itself", under ~70% sure, a
   debug flag, perf not measured, release tooling) is one line in the task's notes for the release review, with no fix
   agent and no task. No re-review of a fix round (review31, T-4080's second round) unless the fix touches saves or
   money. Saves: roughly two thirds of fix rounds (~3 of ~4.7 h of Claude fix-agent time plus Gemini fix agents in 4
   days), the follow-up tasks filed from leftovers (T-4352-4355, T-4407), and 2-4 orchestrator turns per batch.
   Risks: small bugs pile up. Caught by: the release review takes the list as its starting point.
4. **Never wait for Gemini during development.** If the quota is out, a batch skips review (it joins the release
   review) instead of waiting or moving to Sonnet. Save and data-loss reviews keep a Sonnet fallback. Saves: today's
   batch 1127e wait (16:24 to ~19:45 UTC) and the 11-25 to 11-27 weekly outage's moves to Sonnet. Risk: none beyond 1.
5. **Cheap checks for the bugs that actually reached players.** Both crashes players hit this week (T-4367 in 0.4.7,
   T-4390 on phones) came from settings off the default path (intercity off, one worker, low memory). Reading missed
   T-4367: batch 13 reviewed T-4237 and flagged it "minor". The 23 s phone step in smoke (T-4372) catches it. Add
   setting variants to smoke as bugs show them, and read the crash counter after each 8800 rebuild and publish,
   instead of asking reviews to hunt. Saves: nothing now; it targets what review isn't finding.
6. **No benchmark chases in development.** A master bench that moves when the batch didn't touch speed gets noted, not
   investigated (benchwhy, 48 min, plus T-4293, T-4294 and 5c504a742 were all the check's own fault). Benches stay
   once per batch only where speed is the task's point, and on all three saves at release, as now.

Kept as now, since they protect her saves and master: builder's typecheck or Rust tests and one run before landing;
smoke once after a batch (about 3 min, cheap, passed all 8 times on record); the saves review on save-level changes (it
caught the level-42 clash between master and tw/track, batch 1127d F1); revert a commit that breaks master; the
release list of things for her to test (her 11-19); 8800 rebuilt without waiting for review (11-21).

## Evidence (git log since 11-24, notes/agent-reports, artifacts/trainworld/gemini)

- 297 trainworld commits in 4 days: 132 bookkeeping (36 of them about review starts, findings or fix rounds), ~18
  standalone "review fixes" / "findings fixed" commits and ~6 landings carrying review fixes, no reverts, 8 smoke passes (no failure on record).
- Reviews: ~21 Gemini reviews (batch13-23, 1127 a-d, per-task ones), 27 gemini_audit runs including design reviews,
  about 10-15 min each; 16 Claude review agents, about 2 h in total (median 7 min). Reading is cheap. The cost is the
  fix round, re-review, filed leftovers and bookkeeping after it.
- Fix rounds: 14 Claude fix agents, 4.7 h (2-51 min, median ~17), plus Gemini fix agents (keepfix, fixweekend, busfix).
  Chains: T-4080 had two review rounds and three fix agents (11-24 18:49 to 11-25 06:22 local); T-4272 stage 2 went
  review30, buyfix, review31 (about 3 h from review to landing).
- Findings that were worth it during development (player hits it, it stays wrong or loses data), about 9 in 4 days:
  Reload losing edits (T-3932, 1127b), edits lost when a first load fails (T-4346, now T-4408), running cost wrong after
  every load (T-4299, high), no walk between rejoined stations (T-4048 F1), station buildings ignoring new track
  (T-4153 s8), free cars (T-4272), hidden track selectable for delete (T-4400), local figures cached as final (T-2188),
  the save-level clash. Almost all are in saves, money or persisted figures: the classes proposal 2 keeps.
- Findings that were mostly churn: review21 (three lows), review23 ("no way a player reaches it"), review24 (cost not
  measured), review28 ("heals itself"), review29 (a ?patch=0 debug flag, fixed anyway in da827cd34), review26 (publish
  tooling, a release concern fixed mid-development), review20 (65-70% sure), T-4380's first_off flag (a 3-step edge).
  The 11-24 Gemini triage kept 11 of 23 points (6 not worth it, 3 done, 3 already filed).
- Missed by every step: T-4367 (0.4.7 crash, player report, hotfix 0.4.7.1) and T-4390 (phone crash).
- Waits: Gemini weekly quota out 11-25 05:25 to 11-27 ~09:30; out again 11-27 16:24 to ~19:45 UTC with batch 1127e
  waiting. 8800 held once for a fix round (2be702b30), against the 11-21 rule.
- Orchestrators: 7 takeovers in 4 days. Tracking review states is a big share of their turns, and repeats.md already
  calls heavy orchestrators a repeat ("review findings go to the fix agent unread").
- notes/handoff-log.md has no entries after 11-23, so the diary added nothing for these days.

Her earlier words (notes/rule-history.md, handoff.md): 11-16 checker before landing risky changes (T-3451); 11-17
"a bit too test heavy"; 11-19 "maybe we were a bit too hasty to deploy last update" (the release test list); 11-20
batch checking (T-3902); 11-24 agents "get a lot more mileage by thinking about actual code than by trying to test",
"if it doesnt work, we can always go back later", "review, not playtest"; 11-27 this ask. Her rules so far cut where
checking happens (trees to master, play to reading). They haven't changed when it happens. Proposals 1-3 do.
