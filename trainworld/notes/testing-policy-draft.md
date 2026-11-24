# Testing and measuring policy (draft, 11-24)

From Gemini's review (artifacts/trainworld/gemini/testing-policy-answer.md), checked by maps-06. Replaces "Testing, kept lean" and the measuring parts of "Performance" in trainworld/CLAUDE.md, and the checker rules in docs/orchestrator.md.

## Why it kept not sticking
- The notes said "lean" in one line, then required full perf_bench numbers, before/after edit_bench and big-save timings 20 lines later. Agents follow the concrete rule.
- My briefs added more on top ("show it on asianita", "3 runs", "all perf_bench numbers").
- 7 agents kept the machine at 13-14 of 14 cores, so every timing was noisy, so agents re-ran, which kept the machine full.
- Agents built and benchmarked both the old and new version in every tree.
- The bugs you hit (0 riders, ghost blink) came from playing, which none of the benchmarks caught.

## The budget, by kind of change
| Kind | In the agent's tree | Orchestrator, once per batch on master | Release only |
|---|---|---|---|
| UI, text, CSS | typecheck + 1 screenshot or game run (5 min). No benchmarks. | 8800 rebuild | smoke |
| App logic | typecheck + nearest check + 1 game run proving it (5 min). No benchmarks. | smoke | smoke |
| Edit speed | typecheck + 1 edit_bench run on NYC, new build only, against master's stored numbers. | 1 edit_bench NYC | edit_bench NYC |
| Trip figures, sim | Rust tests + 1 figure diff on the NYC save (total + lines that move). If it's also a speedup: 1 perf_bench NYC run. | smoke + batch checker (25 min max, risky changes only, playing, no benchmarks) | all three saves |
| Loading, memory | typecheck/tests + 1 perf_bench NYC run, judged on bytes and memory first. | 1 perf_bench NYC | all three saves |
| Release | | | pause new agents; smoke + perf_bench nyc, tokyo, belgrade, 1 run each |

- At most 2 benchmark runs per tree, ever. Never re-run to average out noise.
- A big save (asianita, tokyo, belgrade) in a tree only when the brief names it and says why (an investigation of that save, like your Asia load). 1 run.
- A timing taken while more than 4 other cores are busy is marked noisy. The agent reports it as noisy with the exact numbers (bytes, memory, riders) and stops.

## Enforced by tools, not by judgement
- perf_bench and edit_bench inside a tree: refuse a third run, refuse more than 1 run per call, refuse big saves unless the brief's flag is given, compare against master's stored numbers instead of building the old version, stamp noisy runs.
- Every brief starts with two lines: the kind of change, and its measuring budget from the table. No extra measuring clauses.
- Every agent report ends with one line: benchmark runs and minutes of game checks used.
- The watch warns me when cores stay at 12+ of 14 for over 20 minutes, or when benchmarks take more than a quarter of the cores over 2 hours.

## Stop doing
- Building the old version in trees to compare.
- 3-run benchmarks, re-runs for noise.
- Big saves in trees by default.
- 45-90 minute batch checkers: 25 minutes, risky changes only.
- 10-25 minute game sweeps in trees: one targeted run.
