# Batch check 2026-11-24, check9 (master d9ffb4707)

Game checks used: about 30 minutes (over the 25 budget, the NYC cache runs are slow: 2 to 8 min each). Benchmarks: 0.
Raw outputs: scratchpad check9/t328-*.json.

1. T-4194 settle: PARTIAL. Rust test counted_expresses_never_run_through_locals passes (agent profile). NYC load total 13,035,408 rail (t328 'rail' figure; a different figure from the 12,672,465 rail trips/day, not compared). Stop add/remove and headway change on a headway line, and NYC edit+undo total: NOT run (out of time).
2. T-4202 stage 1 (keepSet): NOT TESTED (open save B inside running game; out of time).
3. T-4202 stage 2 stand-in: PASS via t328 modes. reload x2: cached view hash 428b6ce3 equals fresh final. standin: stand-in shown at 6.1 s, clock ran, fresh final 428b6ce3 equals unchanged reload, written under new key, next reload shows it as cache. stale: tampered pack gives full load (no stand-in, no cache); fare change gives no cache/stand-in. edit (STANDIN=1): edit while stand-in solving ends at cold solve figures (7348e11e both). No 0-riders loading card seen in any run (no page errors). Not tried: a stand-in from another save with same network.
4. T-4204: NOT TESTED (hover wording, forced restart). Figures on NYC unchanged across the runs above (428b6ce3).
5. t3084 on head: FAIL (exit 1, reproduced). All bus steps pass; the ferry step times out at t3084-bus-edit.mjs line 161: after clicking "Edit route" on a ferry line, window.tw.tools.ferry.active never becomes true in 20 s. Base c0d319162^ not run (needs a second tree); so not known whether it predates c0d319162.
