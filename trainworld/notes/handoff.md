# Orchestrator handoff, trainworld

Orchestrator: session f0b2a436-f42b-45c6-999f-26d970f36c1a, name maps-45, since 2026-11-17 18:15 UTC
Updated: 2026-11-17 22:15 UTC (maps-45 past 500k, starting its successor)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## State
- Live = v0.3.5. She wants v0.4.0 out TONIGHT if it works ("would rather do tonight, but dont worry about it").
  T-3565 (one-way pairs) is NOT in: she said 11-17 ~22:10 UTC "write down ideas and start fresh" (below). My
  proposed cut-off: ship without T-3565 if not solid by ~8 pm her time (asked, unanswered).
- master = 8800 = fa0eeaea + later commits (bookkeeping, es/pt/ca and zh-TW/ru/tr translations); pushed fa0eeaea.
- v0.4.0 highlights marked (c084 regions, c051 long-distance fix, c130 schedule panel). Changelog sorted, grouped,
  tidied (notes/changelog-sort-2026-11-17.md, changelog-tidy-2026-11-17.md); she edits on the board.
- Release steps (each needs her yes to that exact action, none overnight): finish translations (below); `changelog.py
  line` + `publish --version v0.4.0`; then translate the new updates.v040* keys (second l10n batch, 3 passes);
  run shonei-server's TestTwSaveCheckFollowsTheGame against master first (T-3557 added an LWE1 flag, T-3588 header
  fields after the server commits 7a23cbc/96e4f12/95356ed); server deploy; smoke8800 light and dark; deploy.py;
  website push; open-source sync + push; Renkon PR 1 / issue 8 replies (handoff-log 11-17 maps-b8). T-3464 freight
  tiles is {after:v0.4.0 publish} in TASKS but an old handoff listed it pre-ship: ask her if it matters.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit (commit with the night run). Outside
  trainworld, uncommitted, hers to decide (asks.json): .claude/skills/handoff, takeover, hooks/orchestrator.py, maps
  CLAUDE.md 500k line. serve.py board commits: allow rule `PowerShell(git commit * -- serve.py)` added on her word.
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Landing: `agent-tree.mjs finish` (handles already-archived tasks now; serve.py changes must be carried by hand:
  diff to a patch with cmd /c, checkout in tree, finish, git apply, commit, restart_serve.py).

## In flight (maps-45's subagents: reports come to maps-45, which relays them verbatim to its successor)
- extendfix (Opus, .tw-trees/extendfix, port 6179): T-3595 P1, her repro: Caracas Line 2 extended to San Martin
  breaks (artifacts/trainworld/t3595-caracas-sanmartin-broken.save; t3595-caracas-day45-earlier.save is 28 s before).
  Told: reproduce on master, compare with the onewaystops tree's build, then cause + small fix. Likely the same
  problem as T-3565.
- onewaystops (Opus, .tw-trees/onewaystops): T-3565 STOPPED on her word; writing its ideas, routing notes, failed
  models and repros into notes/T-3565.md (main tree copy too). Next: a FRESH Opus agent from that note with one brief:
  one-way stations ordered by where the run passes them, never click order, normal editing only; no save change; a
  Rust test over every click-order permutation; fixtures scen.mjs multi (scratchpad\onewaycheck), her Arismendi
  extension, her San Martin save. Advisor freely (her word). Then a checker. Remove the old tree after.
- farebasecheck (Sonnet, port 6174): checker for T-3578 (fare base, tree .tw-trees/farebase). On pass: land; its
  changelog item duplicates T-3567's unpublished fare-multiple item: reword that one to the end state, hide the new.
  Hover text approved by her. Commit translators' messages/ merges before landing (fareTip changes).
- arrows (Sonnet, .tw-trees/arrows, port 6180): T-3596 one-way arrows vanish a zoom sooner, thinning twice as far.
- Translation of v0.4.0 UI strings (~61 keys/language), docs/localization.md "Translating":
  - pass 1 done and committed: zh-TW, ru, tr (l10n-b; ru speedLimit clash noted for review), es, pt, ca (l10n-c).
  - pass 1 running: l10n-a (ja, ko, zh-Hans), l10n-d (de, fr, id). Commit each one's messages/+sources/ files on report.
  - pass 2 running: l10n-back-b (zh-TW, ru, tr) -> scratchpad\l10n-back-b\<lang>\back.json.
  - still to start: pass 2 for es/pt/ca and the rest; pass 3 review (fresh Opus, reads pass-1 doubts.md in
    scratchpad\l10n-<x>\<lang>\ and the back.json); zh-TW opencc check after zh-Hans is reviewed.

## Night run (she'd prefer the publish first; start after it, or at ~02:00 UTC if no publish)
- T-3511 enclave shifts + T-3455 coastal rows, plan notes/T-3455.md, ~8.2 h. Start: `powershell -File
  C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1
  C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log;
  PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID. 10 cores.
  Hold app/ landings during its gates.

## Next
1. Relay/land reports above; smoke on master per batch; rebuild 8800 (`snap8800.py <hash>` hidden, background); push.
2. Start the fresh T-3565 agent once notes/T-3565.md lands.
3. Finish translations; then the release steps with her yeses.
4. Leftover .tw-trees: onewaystops (after the fresh agent takes what it needs), plus old c1, ferry45, fixes, freight,
   hotfix031-target, hover, touchrule, trackrebuild, safari-save-035, shenzhen, tt2, ttstyle, trams: look first.

## Waiting on Anita
- asks.json (one open: committing the workflow files outside trainworld). Also unanswered in chat: ship 0.4 without
  T-3565 if it isn't solid by ~8 pm her time?

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines; agents reword an unpublished item instead of adding one; orchestrator merges or
  hides duplicates itself. Highlights (1-8) for updates over ~20 items.
- Redraws with no numbers: three versions on one sheet first.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
