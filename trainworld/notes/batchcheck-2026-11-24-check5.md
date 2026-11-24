# Batch check 2026-11-24, check5 (T-4055, f3521af85)

Own Vite 6123 (base arm 6127 in a worktree at 96ad90da4, removed afterwards). 3-5 cores from the ledger.

## Result: PASS, with two cases not run

- checks/t4055-stale-final.mjs: PASS (PHL 68,341 = 68,341, PVD 11,728 = 11,728, no page errors).
- Real NYC save: final v3, rail 12,672,465 (play.mjs wait.final). One 2 km line in Philadelphia (new zone): final v5 after 157 s on 3 cores, rail 12,751,307. Undo: final v6, rail back to 12,672,465; total trips stay 918,437,125 (zone stays open; baseline 899,564,070).
- New NYC game, edit 0.5 s after ready (load hold on), line in Philadelphia: final v2. Then a Providence line: final v3, no stall.
- North America save (na-day-491.save), 4 solving workers, same sequence on f3521af85 and on 96ad90da4:
  Managua + Bogota in one edits file, then San Jose edit plus Ctrl+Z, final, export, reload.
  Both builds give identical figures: live total 1,762,614,066 / rail 30,764,554; reload total 1,763,631,646 / rail 30,764,554.
  The 0.058% total-trips gap between live and reload is still there and is the same on the old build: not caused or fixed by T-4055.
  Baseline after load 1,719,111,037 / 30,599,434 on both; after Managua+Bogota 1,754,377,411 / 30,721,238 on both.
  Likely source (not checked): the Ctrl+Z did not remove the San Jose line (177 lines), so the gap may be zone or pack state differing between live and reload.

## Not run
- Pack download failure: aborting /packs/*.bin with a Playwright route only gave "TerrainMissing" at construct (outside the dev terrain set), so no edit reached a zone. A hang from a failed pack is unproven either way.
- Market recomposition, zone asked at another level after open: no probe built; only the code read (covers() checks opened, level, wanted markets still fetching).
- Save written once final: not read directly; the exported file reloaded to equal rail figures.

## Mishap
My first base-arm attempt ran play.mjs edits against port 6124, which belongs to another agent's game (tree stopclick); the edits failed (TerrainMissing, nothing built) so it likely changed nothing. Its game showed lines 219, version 4.
