# Orchestrator handoff, trainworld

Orchestrator: session aa768402-8365-45e0-85b8-0c27fe256637, name maps-93, since 2026-11-05 17:55
Updated: 2026-11-05 21:37

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **d5b02108** (Mexico committed). **Real save smokes 12,515,594** since farebase (T-2559).
- **8800 = build 1084 (d5b02108)**, with Mexico; smoke light and dark PASSED ~21:55, 12,515,594; Anita pinged. LIVE = 821. Publish only on Anita's word (release "later this week"; Mexico
  and Brazil before release).
- **Cores cap 12**; Anita ~21:45: "keep 4 cores free": real use, not just claims. Check with
  `Get-Counter '\Processor(_Total)\% Processor Time'` (75% = 4 free), not summed per-process CPU deltas (maps-93's 5 s
  sample over-counted cachehit's Chrome, which pins itself to its granted cores).
- **Pack lock free.** Mexico T-2436 done (105 zones, 3,200 -> 3,305 shipped, freeze 20261105-200211; backups deleted).
  Next country: Brazil T-2437 (no prep; brazil.py hook done). Disk 50 GB free.
- maps-17 (now named maps-03) handed over fully at 21:37: no agents of its own left.

## In flight (maps-93's agents)
- **cachehit** (Opus, tree .tw-trees/cachehit from 79d0c5ee, port 6183, ~360k tokens): T-2574 cache-hit bar, then
  T-2533 zone in view answers within ~20 s of a cache hit. Running North America batches; retire after its report.
- **marketsearch** (Opus, tree .tw-trees/marketsearch from 55ab534f, port 6181): T-1647 native Japan pass 15% faster,
  exact. Rust: landing rebuilds the main WASM.
- **sheet2** (Opus, read-only, scratch\sheet2, writes notes/T-2566.md "Round 2" and artifacts/trainworld/sheet2/):
  T-2577 diagram sheet round 2 from Anita's review (verbatim in notes/T-2566.md): aspect change costs most on land
  with lines, less on land without lines, much less on water; cores halfway back; more octilinear; line rivers with a
  higher minimum width, the Harlem River; stations always on their own side of zone borders. Pictures to Anita;
  T-2567/T-2568 wait on her.

## Landed this session (maps-93)
- uismall5 3d0939bc (T-2569..T-2572, Anita's evening TODO; also the map credits button, invisible since T-1435, now
  just left of the trouble button; Anita: "attr fix looks good"), farebase f3b3cb18 (T-2559, 598 -> 594), t409strong
  (T-2527, T-2573 filed), cleana4 97c6e153 (T-2512, T-2575 filed), kernelsplit 8696ae95 (T-2381), walkcost 83f433fd
  (T-2324: the walking rule costs 5.6% of New York's calib; 3.2% over T-1059 net of other tasks), followtrack 665dd025
  (T-1358, 0.55 accepted), toworld 3d0d0097 (T-2576), Mexico d5b02108 (T-2436; T-2578 border factor filed).
- From maps-17: watertoggle 7bba88c2 (T-2565), cachecard 9a68feb5 (T-2560; cache-hit reload playable ~16-25 s on her
  NA save), sheet's write-up db56555e (T-2566).
- Strings shown to Anita (no answer yet): "Show water" (new hover); "Nothing built" -> "Loading..." while an account
  game's picture loads; "No stations" (new hover, trouble row).
- Next free id **T-2579**.

## Anita 2026-11-05, in her words where it matters
- Diagram: "i love how japan looks"; "perhaps we could turn down distortion if its easy" (dial at 0.8, asked whether
  to keep it); "peninsulas dont hae to be huge especially if theres nothing on them. middle groud"; islands with no
  stations "can be either ommitted (if small relative to their surroundings) or at least not expanded".
- Launch: she'll speed up her own gameplay recording (T-2546 on hold); no Discord/subreddit; no Cloudflare Cache Rule
  yet; phone tested (NA save refused with the memory message, Japan loaded).
- Disk: "anything that nonnegligibly slows down our edvelopment process to gain stoage we probably dont need ot do".
- Batches: notes/todo-2026-11-05a.md, -05b.md, -05c.md (evening, T-2569..T-2572, all done).

## Next
1. As agents report: land trees (typecheck, smoke in the tree, `agent-tree.mjs land`), commit records by path, show
   Anita strings old -> new and review pictures (copy to artifacts/trainworld/review/).
2. After 8800's smoke: Brazil T-2437 (pack lock; Opus agent, docs/adding-a-country.md; one big region, so T-2070's read
   ahead still needs a multi-region country). Mind the 4 free cores: a country run at --jobs 4-6 only with little else.
3. Open and ready (cores allowing): T-2532 and T-2553 (demand worker; after cachehit lands, same files), T-2374 app
   splits (one file per agent; collides with any app work, so when few app agents run), T-2575 (regions.json once),
   T-2573 (t409 walker), T-2578 (MX-US border), T-2496 (WASM memory per structure; Rust, after marketsearch lands),
   T-2510 (Japan masks miss big rivers), T-2558 (Mexico everyday from DENUE). Diagram routing tasks (T-2529, T-2538,
   T-2530, T-2539, T-2551, T-2518, T-2498) wait on the sheet's direction (T-2577).

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\aa768402-8365-45e0-85b8-0c27fe256637\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py `<tasks.md> T-x=agent ...`,
  common-brief-tree.md for tree agents, lowdisk.ps1, splice_handoff.py). A new orchestrator copies these and fixes the
  path inside snapbuild-frozen.cmd.
- 8800: `cmd //c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (Bash), then claim 4 cores, smoke on
  http://localhost:8800/trainworld/ light and with `$env:SMOKE_SHADE='dark'`. Build number = `git rev-list --count`.
  If 8800 does not answer, restart serve.py detached (it dies with a VS Code restart).
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"` (rebases itself; rebuilds the
  main WASM when sim changed), then `remove <name>`. On a rebase conflict: resolve in the tree, `git -c
  core.editor=true rebase --continue`, typecheck, land again. ff fails if the main tree has uncommitted edits in a
  file the commit touches: commit those first.
- **Commit by path**: `git commit -m "..." -- <paths>` (other sessions stage too).
- Docs lint: `venv\Scripts\python.exe trainworld\checks\doc_lint.py`.
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`,
  website `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page
  --version <v>`. Afterwards prune publish/ and data/frozen/.

## Open questions to Anita (asked, unanswered)
- None open. Sheet round 1 answered ~21:30-21:40 (notes/T-2566.md "Anita's review"; sheet2 builds round 2). The three
  new strings were shown ~21:05; no answer yet means fine unless she names one.
- Earlier, answered ~18:15: water is experimental behind a button, default off (T-2565); sheet first, rough
  prototype outside the game (T-2566); "this sort of distortion scheme would solve the issue of region borders not
  having their right shape". Distortion default 0.5 (her pick).

## Lessons from this run
- A UI landing can silently break every pipeline gate: T-2365 made the top bar hover a grid and
  checks/city-browser.mjs's text wait timed out on every zone from 13:42 (fixed cf102c0a). Before the next
  country gate after a landing that touches the top bar or its hovers (profit2's T-2443 does), run
  city-browser.mjs on one zone first.
- An agent that left a background wait running re-sends its final report on every wake; TaskStop it once done.
- Check the clock (`Get-Date`) before writing times into the handoff.
- Commit with a pathspec; another session's staged moves get swept in otherwise (314776ed).
- Headless pages run over their core claims (a page took 7-8 cores on a 3-4 claim); briefs must say "cap the page's
  demand workers to the claim, claim with --pid, release between runs".
- An 8800 build during a pack rewrite can serve a mixed set (console errors); rebuild after pack writes end.
- A gate's browser checks run on a Vite over the main tree; joins re-gate without a browser, so app/WASM landings
  are fine during a join but not during a browser gate.
- Tree agents' pages read the main data junction: a tree made before a replan's game json commit asks for retired
  packs ("New York could not load").
- `world_run.py step <x> --help` may run the step for real (T-2398).
- serve.py (8800) dies with a VS Code restart; check it answers after one.
- Agents near 450-550k tokens: retire after their report; give follow-ups to a fresh agent.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk: delete orphaned temp files always; keep only the current build and the latest published ones; delete
  superseded backups once a later state checks clean.
