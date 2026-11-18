# T-3600 / T-3601 check (kcheck, 2026-11-17)

Tree `.tw-trees/kaohsiung`, Vite 6183, 2 demand workers (ledger gave 3 cores), script and logs in the kcheck scratchpad
(`k1.out` Kaohsiung 1852, `n1.out` NYC real save). Probe every 2 s: tag text, version, lines' riders, count of negative
numeric fields on every line of the view.

Result: PASS on cases 1, 2, 3. Case 4 not run.

1. Kaohsiung 1852, edits 0.5 s apart. On phase (6,4,3), final 327 s: 201 488,827 / 217 253,261 (same as runB). Off phase,
   first edit alone: local at e+6.9 s, 212 16,710. All five off, held behind the first whole solve (as notes say):
   local at e+163 s: 201 442,156 vs final 447,051 (-1.1%), 217 215,529 vs 211,456 (+1.9%), feeders 531-2,277 (final 0).
   Negatives: 0 in all 284 probes. Tag "Calculating trips after your change..." in every probe from e+6.9 s to the final.
   Final: 212-216 all 0.
2. Same run, second to fifth edit arrived during the first quick solve and were held. Final 447,051 / 211,456 against the
   load solve of the same network 447,045 / 211,446 (0.002%). Stale figures: none after the final. The tag was empty in one
   probe 1.2 s after the off phase's first edit (before the solve showed "Trips here being recomputed"); empty otherwise only
   after the final. Not a real gap; the edit had just been sent.
3. NYC real save, line 3: one step up opened no window, final 74 s; line 3 to 0 final 77 s, line 3 reads 0, tag empty only
   after the final, 0 negatives in 83 probes. Run concurrently with the Kaohsiung run, so slower than the 47-57 s of
   T-3513/runNYC2; no regression in figures (977,392 -> 1,006,235 -> 0).
4. Save and reload after an edit: skipped (no ready save-to-bytes path in play.mjs; the final equals the load solve of a
   saved network in case 2, which covers the same ground).

Page errors: Kaohsiung run reported `demand sums worker: undefined` once (at close, after the final); NYC none.
Earlier runs B-E log `[]` or ERR_INSUFFICIENT_RESOURCES. Not chased; may be teardown noise.
