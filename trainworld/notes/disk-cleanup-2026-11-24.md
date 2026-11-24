# Disk cleanup, 2026-11-24 (afternoon)

C: was 29.6 GB free at the start, 64.3 GB at the end (others were writing meanwhile, so the figures drift).
Anita's ask: when space runs low, clear a decent chunk (aim 60+ GB), not just back over 32 GB.

## Removed

| Path | GB | Why |
|---|---|---|
| Temp\claude\dce1fefcc1d43aef\427323a8-53ce-483c-9c28-8f1452e36f9f | 10.2 | finished session scratch, nothing newer than 2 days, no junctions |
| Temp\claude\dce1fefcc1d43aef\d15915ac-6dd5-4586-bd98-843cbc616824 | 4.7 | same |
| Temp\claude\dce1fefcc1d43aef\4f09b9f6-62ab-4330-bda8-3c96eb7a5896 | 2.3 | same |
| uv cache (uv cache clean) | 6.0 | download cache, refetched on demand |
| pip cache (pip cache purge) | 3.3 | same |
| npm cache (npm cache clean --force) | 2.5 | same |
| Temp\* top-level items untouched 3+ days (374 files and folders, not claude) | 4.5 | orphaned temp files |
| artifacts/trainworld/perfbench/dist/busland-base | about 0 | its files are hard links, so it frees nothing |

## Left on purpose

- perfbench/dist (diskmap shows 414 GB, but the dists are hard links to shared files: deleting one freed 0.1 GB). Not worth deleting.
- Temp\claude scratch f51a9e36 (24 GB, written today), 360c2881 (5.1 GB, passes 2 days 11-25 about 07:55 local), and the sessions from 11-23 and 11-24. Delete 360c2881 tomorrow morning.
- .tw-trees: all live trees. editseq (4.6 GB) is a worktree not on the live list: check with its owner. .tw-trees/modecolors2 is not a worktree any more (no .git, small leftover with trainworld/ and worldcities/), left alone; remove by hand with care.
- .tw-snap (live 8800 build, 9.6 GB), data/packs, data/frozen, data/work, data/raw, saves, Documents, Downloads.
- Not mine to decide: hiberfil.sys 27.5 GB (powercfg /h off frees it, changes her system), Docker wsl 9.2 GB, Unity Library folders in shonei and shonei-test (about 11 GB, regenerable), religiondots/data 42 GB, languagedots/data 19.5 GB, isochrone/data 15.7 GB.

## Note on scanning

diskmap/scan.py took 282 s this time. Its tree counts hard links once per link (1388 GB scanned on a 968 GB used disk), so artifacts/trainworld looks huge but is not.
