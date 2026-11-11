# tasksweep, 2026-11-11

Sweep of TASKS.md (about 405 open task lines) for lines already done or dead. Method: every line's id checked
against ARCHIVE.md and ARCHIVE-to-2026-10-19.md (entry for the same id), `git log --grep` on the id, DECISIONS.md
for supersessions, and the line's own "done / landed" wording; code checked for the tram and cost lines.

Counts: closed 9, abandoned 0, left about 396.

Most commits that name an open id only filed it ("T-2598 filed"); those are not evidence of done and were left.

## Closed (moved to the top of ARCHIVE.md)

- T-2996 tram street snap ghost flicker: landed d0601e8d (tramfix tree committed; message names T-2996 to T-3006).
- T-2997 tram curves 18 m, no U-turns: d0601e8d; DECISIONS.md has the 18 m call.
- T-2998 streets a tram can use tinted green: d0601e8d.
- T-3000 street graph gaps at tunnel and bridge ends (`joinRoads`): d0601e8d, notes/T-2921.md Round 3.
- T-3001 tram guide circles 0.5 and 1 km: d0601e8d.
- T-3004 tram pays a quarter of the terrain surcharge: d0601e8d, docs/track-and-service.md matches.
- T-3006 white tram wires in light mode: d0601e8d.
- T-2732 upkeep by terrain / crossings: `ROAD_CROSSING_COST` is 10.0 in params.rs, landed de29a14c (message names T-2732).
- T-2460 boat rated load: ARCHIVE line "T-2919, T-2460 (45b6061c)" says boat 320 of 480; `rated()` follows the per-car figures.

T-3002 and T-3003 (the other tramfix lines) say "researched, not built" and stay open.

## Abandoned

None. No later decision plainly supersedes an open line.

## Work may be lost ("Done in tree X, uncommitted" with the tree gone)

None remain. The nine tramfix lines were the only ones; the tree `.tw-trees/tramfix` is gone, but its work is in
commit d0601e8d. Only `.tw-trees/hotfix` exists now.

Other "uncommitted" mentions checked: T-2845 says shonei-server's `now` field (T-2842) is uncommitted; it is
committed there (af818b1, "playing-now and playtime histograms"), the line is a check that waits on a deploy, so left.
T-2854 (wiki rounds 1-2 "uncommitted") not checked: not a tree, the wiki dir is in the main tree.

## Probably done, but lacked evidence (for the orchestrator to check)

- T-3079 Exact mode fixes: line says landed ee29c42c; left because the toggle is hidden until Anita tests it (waits on her).
- T-2417 diagram stretched extension: "fix landed 67468b83, Anita to say if it recurs"; waits on her.
- T-2472 size summary: "done 2026-11-04, sizes", report written; open only for her disk deletions.
- T-2912 security review: reviewed, fixes landed (24c3d6dd, T-2914 hover script, T-2917 archived); notes/T-2912.md still lists the production box unchecked and a bigger origin move.
- T-2663 attractions answer: "archive once Anita has read it".
- T-1882 `checks/t1270-terrain.mjs`: "run it and fix or archive", not run by me.
- T-2589 sheet speed: stages 1 and 2 done and committed; remainder is the aim not met, so correct to leave open.
- T-2866, T-2843, T-658, T-3012 (stage 2 done, T-3033 left): stages landed, remaining parts named; fine as they are.
- T-592 (someday: read the next region's extract while this one gates and joins) overlaps T-1980 (done, ARCHIVE line 698) and T-2070 (confirming it); T-967 says its first half is T-1980's. Possible duplicate.
- T-060 (compressed `.bin` for R2) and T-2486 (same ask, later, with numbers): T-060 is likely folded into T-2486.
- T-3135 and T-3137: C1 and C2 committed in ed4f66fa; C3 is "taking" tonight, so left.
- Duplicate ids in TASKS.md: T-2914 appears as the census-totals line (world section) and in T-2912's text as the hover script fix; not a task line twice, but confusing.
