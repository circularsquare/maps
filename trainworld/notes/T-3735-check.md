# T-3735 check (weekendchk, 2026-11-20): PASS, with one note

Tree weekenddep (1ab95d5b + uncommitted) against the main tree (master), own Vites 6191 / 6192, play.mjs games.
Fixture (fresh nyc): line A (3 stops T,J,M) and B (T to a branch at J), both headway 360 s; weekday really runs
480 s each (merge pacing, same on master). Trips read with the clock worker's `lineTrips` (departures, 09-14h).

1. Weekdays unchanged: real save (109 lines), hash of every line's trips 06-12h on days 2, 3 (weekdays) and 6
   (Saturday), with weekends at 100% and again after Apply 70% to all: identical tree vs master (6322 trips; Saturday
   5196 after 70%, same hash). After 3 game days at speed 2: demand 895,752,175 trips/day, 12,558,064 rail,
   cash 11697.8139 (tree, t=370877) vs 11697.8147 (master, t=370965): equal within the stop-time difference.
   Skipped: London game (cores over the cap).
2. Shared track, both lines 50% weekends: Saturday 720 s each, merged departures from T alternate A/B, gaps 341/379 s
   (master 221/499). Tuesday identical to the pre-change hash and to master (gaps 221/259). Stands 852-1308 s (master 732-1428).
   No NaN `next`, no lost or doubled train (trip counts 53/52).
3. Weekend edits then undo x3 / redo x3: weekend share 75, add a train, delete station J: no page errors; undo returns
   exactly the first state's Tuesday and Saturday hashes, redo the last's.
4. Save: no weekend rates: gunzipped track bytes equal master's (631 each), only header level 32 vs 31. With 50%/50%:
   LWP1 present (lists no line: phases equal weekdays'), 6 bytes more than master. Mixed A50/B100: LWP1 lists lines;
   reload gives the same Saturday gaps 19/360/341 s.
5. Master's save with 50%/50% opened in the tree: Saturday hash and gaps equal the tree's own (341/379).

Notes (not failures):
- Mixed headways (A 12 min, B 6 min): the tree's Saturday merged minimum gap is 19 s, master's 139 s (a greedy phase
  pick, 19/360/341 against ideal 180/180/360). The weekday placer does the same with non-matching rates (gaps 9-360 s
  with A at 1138 s). Not a regression of the rule, but "evenly spaced" does not hold for unequal headways.
- With A at 75% (480 s, same as weekday) and B at 50% (720 s): gap 19 s at the merge likewise.
