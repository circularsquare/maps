# Load reports, 11-22 to 11-27 (agent loadreports, read-only, 2026-11-27)

Question (Anita 11-27): "a few times it felt almost instant, but some other times it took like 10 minutes ... other
people 30 minutes. is that expected?" Her guess: low memory or resources on the players' side.

Read: 20 threads from the feedback inbox (the last 5 days) that mention loading, slow, memory, crashes, white screens or
carry a `diedLoad:`; one at a time, no runs, no saves loaded. About 14 more 11-25 threads are "roads and water not
showing" (the base-map provider on a block list, her reply of 11-25); I opened 2 of them (below) and counted none.
The "10 minutes" and "30 minutes" Discord reports are not in the inbox, so the inbox has only 4 slow loads that I can
time and none that was fast. Everything about fast loads below comes from the code, not from a report.

## Table of loads

Workers = the report's `workers N`, which counts the aux worker, so 2 = 1 solving worker, 3 = 2 solving, and 1 on a
phone means the one worker does both jobs (demandClient.ts 1151-1156, 2003).

| thread | ver | device (memory, browser) | workers | save | outcome | why, from the diagnostics |
|---|---|---|---|---|---|---|
| 727e1679 | 0.4.5 | Windows, 8 GB, Chrome 158 | 3 (2 solving) | 55 lines | solved, final 207 s after the solve was sent; player: "calculating trips is long" | ordinary load, 2 solving workers, 4 rounds done |
| bb72f6a0 | 0.4.6 | Windows laptop, 16 GB, Chrome 157, Iris Xe | 2 (1 solving) | 121 lines | still loading at 1,219 s after the pick, "same since 3 hours", card said about 50 min | round 0, 3 of 5 periods done; period 4 had run 497 s; this is the T-4343 case (same numbers); one solving worker, plus the old 10-minute restart; player 11-25 14:50 "a lil better" |
| 32d2389e | 0.4.7.1 | iPhone/iPad Safari, memory not reported | 1 (shared) | 129 lines | 507 s after the pick no period of round 0 had ended; player says the map did not load after 15 min | phone: one worker, cache off on mobile; first period alone is over 8 min; not dead, just serial (5 periods x 4 rounds) |
| e426dbaa | 0.4.7.1 | Windows, 4 GB, Chrome 156 (Intel ADL-N, 1366x598), account | 2 | not shown | "status loading" for 6 min, "3.4G/3G" memory, white screen after 5 min at 100x | no demand line, so no workers or lines; memory at the machine's limit; same machine as the 4 below |
| 614ea1aa | 0.4.5 | Windows laptop, 16 GB, Chrome 157, RTX 5050 | not shown | "one of my saves grew too big" | not loaded after 11 min; still not on 11-25 "even with less detail on" | status loading with no demand line; the report gives no worker, lines or step. I cannot explain it |
| cfc94c1f | 0.4.7.1 | iPhone Safari 18.7.8 | 1 | 931 stations, 56 lines, zone tyo | died 181 s into the demand solve, out of memory screen | OOM on one worker; no retry possible (oomRetry.ts 21) |
| 316caf14 | 0.4.7 | iPad/Mac-class Safari 26.6.1 | 1 | 1,042 stations, 84 lines, 4 zones | died 214 s into the demand solve, out of memory | same |
| 1e3064bc | 0.4.7.1 | Android Chrome 158, 4 GB, PowerVR | 1 | zone osa, counts not known | died 11 s after the pick, step "demand game" | OOM before the network was in |
| 4b393c6d, 140ccdb6, f7eba269 | 0.4.7.1 | Windows, 4 GB, Chrome 156, ADL-N, 9 slots (one player, 3 reports) | 1 | zone tyo (Tokyo) | died 2 s after the pick, step "demand game"; says the PC sits at "3.3G/3G" | OOM at the start of the game load, 4 GB machine |
| 9394915c | 0.4.5 | same ADL-N 4 GB machine type, 8 slots | n/a | n/a | out of memory text, status loading, page open 1 min | OOM |
| ed27a582 | 0.4.5 | iPhone Safari | n/a | n/a | "falls at once", out of memory text, status loading | OOM |
| 423e6782 | 0.4.7 | same Safari as 316caf14, 26 min later | n/a | n/a | "crashes and reloads every 2 minutes while editing" | status loading, worker not started; no message. Likely OOM, not confirmed |
| 937f0236 | 0.4.6 | iPhone Safari | n/a | n/a | crashes on zoom or the slider | worker not started, no message; likely OOM, not confirmed |
| 5e283c5f | 0.4.6 | Android Chrome 138, 2 GB, Mali-G72 | 1, replaced 1 | 30 lines | "load failed" after 56 s | NOT memory: worker 0 panicked at sim/src/demand/everyday.rs:655, index 10191 of len 10191, with 17 MB of WASM memory. A code fault, 0.4.6 |
| 21607ace | 0.4.5 | Safari 27, graphics lost | n/a | n/a | page stopped at startup | graphics lost, not memory |
| 01d72055, a85cf2d5 | 0.4.6 | Brave 8 GB; Firefox 160 | 3 / not started | 43 lines / none | "map not loading", white map | base map provider blocked (her reply 11-25); not a demand load |
| 44e13989 | 0.4.6 | same 4 GB ADL-N machine | 2 (1 solving) | 77 lines | slow while laying track (an edit's solve) | not a load; shows 1 solving worker on 4 GB |

Time-to-load that I can state: 207 s (727e1679); 20+ min and counting (bb72f6a0); 8.5+ min and counting (32d2389e);
6+ min and counting (e426dbaa); 11+ min and counting (614ea1aa). Deaths: 2 s, 2 s, 11 s, 181 s, 214 s after the pick.

## 1. What explains the long and dead loads

Short answer to her guess: right about the dead loads, partly right about the long ones.

- **Out of memory, no way to retry (dead loads): 9 threads with the out of memory text or an iOS crash loop, 7 of them
  certain** (ed27a582, 9394915c, 316caf14, cfc94c1f, 1e3064bc, 4b393c6d/140ccdb6/f7eba269 counted as one device;
  423e6782 and 937f0236 are probable). Every one is a phone or a 4 GB machine, every `diedLoad` says "up to 1 solving
  workers". The retry in oomRetry.ts halves the workers and only works with 2 or more (line 21: "it ran 1 solving
  worker" and null), so a 1-worker load goes straight to the out of memory screen and the player reloads into the same
  crash. Three of the five `diedLoad` lines died 2-11 s in, in the game step before any solving: a 4 GB machine that is
  already at 3.3 GB in the browser dies on Tokyo at once.
- **One solving worker on a big save (slow loads): 2 reports** (bb72f6a0, 32d2389e). With "Use more cores" off a
  solving pool may take 17.5% of the 8 GB budget (demandClient.ts 667, 261), and phones and 4 GB machines get 1
  (268-274). So 16 GB laptops also get 1 worker on a 121-line save: this is a memory budget, not the machine's memory.
  A load is 5 periods x 4 rounds in a row on that worker (T-4343), and the card waits for the last round (see 2c).
- **The old 10-minute restart: 1 report** (bb72f6a0, 0.4.6; "solve v2" after "v1" and a period that ran 497 s fit
  T-4343's restart). Fixed in 0.4.7 (the watchdog waits 4x the longest period, loading.ts 456, demandClient.ts 1969).
  No 0.4.7 or 0.4.7.1 report shows a restart.
- **First download of the data: 0.** The reports that were still loading show no `oldest download`/`downloading` part;
  bb72f6a0 and 32d2389e both had a solve running for 500+ s.
- **Worker panic, a code fault: 1** (5e283c5f; everyday.rs 655, the placing pass of `Columns::pack` ran past the count).
  Not memory. Looks like a real bug on a 2 GB phone; needs a person to look.
- **Not demand loads: 3** (base map block x2, graphics lost x1).
- **Cannot explain: 2.** 614ea1aa (16 GB desktop, 11+ min, no demand line, v0.4.5) and e426dbaa (status loading 6 min on
  the 4 GB machine; probably memory, but nothing in the file says).

So 30-minute loads are not in the inbox. The code gives the arithmetic for one (bb72f6a0: about 4 min a period in round 0,
5 periods, then 3 more rounds at 0.3-0.8 of that: 40-50 min on one worker). I can not say what the Discord 30-minute
players had; the likeliest from the above is 1 solving worker on a big saved game with no usable cache.

## 2. Why one save loads at very different speeds on one machine

a. **The saved result is reused or not.** The demand cache (workers/demandCache.ts, docs/architecture.md 967-1017) is
   what makes a load "almost instant": the last final result is stored beside the slot and shown once the network is in.
   It is off on any phone or tablet (demandClient.ts 1128 `!mobileDevice()`), off with intercity trips off (same line,
   `!marketOff()`), and written only when a solve reaches its final (writeCache 3038-3074). The key is the network, the
   packs, the zones and the code version (only the version and worker count get a stand-in, T-4202); one result per slot.
   So a reload of an unchanged game on desktop is instant; the same game after any edit that did not reach its final
   (an edit's re-solve on a big save runs minutes, T-4359: 249 s for 166 zones; closing the tab before it ends), after
   an import or sync from another device, after a full-disk cache wipe, or on a phone at all, is a full solve of
   several minutes to an hour. Same machine, same save: instant or 20 min depending on whether the last session ended
   with a final. A phone never gets the instant one.
b. **The pool shrinks by memory and stays shrunk.** `rememberPoolCap` (284-286) keeps a halved pool for the tab's session
   after a worker's out of memory (memoryShort, 3298-3316) or a died load. A tab that has been through that loads at half
   speed (or 1 worker) while a fresh tab loads at full, with the same save. The cache key holds the worker count, so
   that load also loses its exact match (stand-in only).
c. **The card waits for the last round of the viewed zone.** HOLD_ROUND = ROUNDS = 3 (demandClient.ts 136, 143-151): a
   game load ends at the zone in view's final, not at round 0 (READY_ROUND, which is what an edit or a new city uses and
   what the crowding rounds "refine in the background"). On one worker rounds 1-3 are 0.26-0.8 of round 0 each (T-4343),
   so about 55-60% of a New York-shaped wait is spent after round 0 had figures.
d. **The time left misleads.** It is blind until a period ends and runs 2.5-3x high on one worker early (T-4343), stepped
   up once at a later round on Belgrade (9 to 17 min, T-4347), reads 1.3-2x high in the last third; bar is held under 50%
   until `bounded` (T-4264). A card showing "about 50 min" at 13% and finishing in 25 is a first-load behaviour on a big
   save, and a reload with a stored pace (`loadPace`) is nearer. Players see the same save as "10 min" and "30 min".
e. **A load that is not the first thing** (T-4359): a zone loaded mid-game waits for an edit's whole re-solve; fixed
   in 0.4.7 for the bar, not for the wait itself.
f. **Hidden or throttled tab and other tabs**: the watchdog counts no quiet in a hidden tab (LoadingScreen.tsx 121), but I
   found no code that speeds or slows the solve itself; not checked further (no runs).

## 3. Proposals (sizes in hours of agent work, rough; nothing filed)

1. **A way through for a load that already runs 1 worker and dies (7+ threads; phones and 4 GB PCs).** Today it ends at the
   out of memory screen. Offer or do a second retry that drops the biggest memory use that is not the player's network:
   intercity trips off for the load, or the coarser zone detail, with a card line saying so; plus find what takes memory
   in the first 2 s on Tokyo (3 of the 5 `diedLoad` died in "demand game"). About 6-10 h to do the retry; the memory cut
   itself is open-ended. Both change what the player sees, so Anita's call.
2. **Keep a result on phones, and keep partial progress.** Cache off on mobile means every reload, including the OOM retry
   reload, starts from zero; a load that has run 8 minutes and is killed loses everything. Writing the final result on
   mobile (the write worker is the memory cost, T-2794), or saving each period's round 0 as it ends so a reload resumes,
   would make second loads fast. About 8-12 h for a mobile final-result cache, 20+ h to resume partial rounds.
3. **End the game load at round 0 and refine after.** Let the card go at READY_ROUND instead of HOLD_ROUND (demandClient.ts
   143-151): the clock runs on round-0 figures, the crowding rounds refine in the background as after any edit. About
   55-60% off a one-worker wait (the share of rounds 1-3), more on big saves. About 4-8 h for the card and the "figures
   still settling" status wording; it changes what figures show first, so her call.
   Smaller: look at the everyday.rs:655 panic (5e283c5f), 2-3 h; and tell a 1-worker big save on the card that
   "Use more cores" does nothing below 8 threads (her call, noted in T-4343).

I did not call the advisor.
