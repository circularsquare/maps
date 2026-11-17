# Night queue after T-3399 (prepared 2026-11-17 by nightprep; not started)

Anita, 2026-11-17: "overnight if we finish with anything we should keep going with further rebuilds". A queue that
starts when T-3399's tonight.ps1 ends and runs the ready rebuilds until 10:00 local. Kit (scratchpad):
C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\00d74e61-e552-47a6-ba40-ee320d38eba1\scratchpad\nightprep\. Nothing in the main tree is edited until a stage runs; every edit is made by the stage itself, after its
backup, so a T-3399 restore cannot undo it half-way.

## Launch, stop, logs

    powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\00d74e61-e552-47a6-ba40-ee320d38eba1\scratchpad\nightprep\nightqueue.cmd

Log artifacts/trainworld/night-queue.log (one line a step), state C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\00d74e61-e552-47a6-ba40-ee320d38eba1\scratchpad\nightprep\nightqueue.state (pid first), step logs
artifacts/trainworld/nq-*.log, scan log artifacts/trainworld/t3455-scan.log. Stop: trainworld\data\work\world_run\STOP,
then the pid. Options: -Until 10:00, -NoWait (run without T-3399), -NoScan, -MaxScanGroups 2, -Rehearse (logs every
step to the kit's rehearse.log and runs nothing; rehearsed 2026-11-16 21:46, all stages in order).
Cores: "nightqueue" wants 10 (6 at least), "nightscan" 3, both claimed with the script's pid.

## Order

0. Wait for T-3399 (its tonight.state, up to 30 min to appear; then its pid). Go on only if its log ends "all done",
   "stage 2 skipped", or a stop before any join (a failed join wants restore.py T3399 prejoin, which would undo
   anything after it: the queue then runs nothing).
- scan (T-3455), beside T-3399 once T-3399's log says its cores are granted: scan_coast.py, read-only, 3 cores. For
  each registered pack: osm_units.reclaimed on its own outline and the OSM sea (what a rebuild would add), minus other
  packs' land, pieces touching the sea only, GHSL people and built surface. Flag: 1,000 people or 150,000 m2 built.
  Test on 12 packs (kit scantest/): 0-190 s a pack (Hong Kong 190 s), outlines read in 70 s; Hong Kong 2 km2 and
  Singapore's Changi Bay 2.1 km2 (118,000 m2 built) have nobody and stay below the flag; Tianjin's pack does not
  reach the sea. Guess 3-4 h. Output kit scan/flagged.json, copied to artifacts/trainworld/t3455-flagged.json.
- A, T-3488 (about 2 h): backup T3488 pre (TR and neighbours, 1.1 GB); plan_edit.py t3488.json --go: arv's unit
  (Arnavutkoy, 176,193) into ist, Karamursel (31,464) ist -> kce, Altinova (34,811) ist -> yae; arv's row removed;
  zone-names.json drops arv and gives ist the aliases Arnavutköy, Arnavutkoy. Retire arv (unregister with market heal,
  its 5 data/packs files to data/work/pre-T3488/retired/). Run ist, then kce and yae (a taker is built after its
  giver, since a build cuts every outline on disk out of its own). Join TR, zone_successors.py (arv -> ist), stale.py.
- B, T-3499 + T-2725 (about 2-2.5 h): backup T3499 pre (CO VE EC PA PE BR, 2.5 GB). Bucaramanga (bug, 4 units,
  wrapped by Barrancabermeja's brc on all sides) takes the rest of Soto province from brc: Piedecuesta 201,291, El Playon,
  Rionegro, Los Santos, Tona, Matanza, Charta, Surata, Santa Barbara, California, Vetas (281,418 in all; bug 1.51M, brc
  0.80M). bug then borders Ocana's and Cucuta's zones; brc stays one piece. T-2725: the 28 rows admin_totals.py dry CO
  moves over 10% (mee crs yop sux tuj ccu brc viv vll ccs mnr miz sbo scl slg apr plm snv fle jmu qub roe bna tma oca zip
  gio fus) and bug take module admin_totals, archetype C; pop_vs_plan relaxed where DANE / plan people is outside
  0.8-1.4 (yop 0.60, bna 0.71, sux 0.74, snv 0.75, fus 1.41, oca 1.44). Wave 1 those 28, wave 2 bug; join CO.
- C, T-3455 rebuilds: the flagged solve units, most people first, at most 2, each only if its estimate ends before
  10:00: setplanned.py, run, join of the group's countries. Likely skipped tonight if T-3399 runs its 5-6 h; then
  flagged.json is the list for the next night.
- End: build_inputs.py, status.

Each stage: plan_edit dry output is in the kit (rerun plan_edit.py <spec> for it). A row that does not gate stops the
queue before the join: restore.py <T####> pre --go undoes the stage, or relax and rerun (adding-a-country.md 6). A
failed join stops it: restore.py <T####> prejoin --go (keeps the builds) or pre --go.

## Calls made (for DECISIONS.md once shipped)
- Altinova is a Yalova province district, so it goes to Yalova's zone, not Kocaeli as the task line said.
- Sile (Istanbul province, 34,797) stays in Kocaeli's zone: moving it too makes ist and kce trade units both ways,
  and each build would lose its new units to the other's old outline.
- Bucaramanga's zone is its metro and the rest of Soto province, 1.51M (just over the 1.5M zone cap).

## Not tonight
- T-3033 (37 split rows): the per-row unit edits are not computed (stage 2 dry-ran a few countries; Russia never
  finished), and the rows sit in 18 countries, one join each: more than a night. Mansoura alone needs a call on what is
  left of slw (Tamy Al-Amdid, 213,163, named Sinbillawin) once Sinbillawin joins elm.
- T-3245's other items: T-985 (wix, lnq solve regions) and T-3434 (Sabang) need joins of groups nothing else rebuilds
  tonight; the rest need code first or a whole rebuild (T-3501 too).

## For the morning
After a stage: status, stale.py, zone_origins.py, state_gaps.py TUR / COL, split_zones.py, smoke, the real save. A:
Istanbul Airport's place in ist, zone-successors.json arv -> ist, app/src/i18n/place-names-zones.json still holds
arv's names (drop them). A changelog item for A and for B (Bucaramanga, Colombia's people) once shipped. Backups
pre-T3488, pre-T3499, pre-T3455g* go once checked clean.
