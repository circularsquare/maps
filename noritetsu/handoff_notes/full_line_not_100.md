# A whole line ridden but not 100% (2026-10-07)

Anita: "theres a lot of cases where i've ridden the entire line but it says like 99.9% done."
Her screenshot: NYCS A 99.9% (57.5 of 57.5 km), C 99.9% (29.8 of 29.8), 7 99.9% (16.6 of
16.6), F 98.7% (47.4 of 48.0); L and S Franklin Avenue 100%. Later, same day: a Kagayaki ride
Nagano - Tokyo does not give her Hakutaka's Nagano - Tokyo stretch; and a ride on the NYC 6
does not fully credit the <6>.

Investigated in headless Chrome against the served page (localhost:8800/noritetsu/), using the
page's own `rideGids`, `creditRun`, `mapBack`, `mergeSpans`, `lineKm`, `pctLabel` on rides built
with `newRide` (nothing saved). No edits to `dist/index.html` or any Python file; the proposed
change is the diff at the end, tested by loading it into the page over the old functions.

## The sample and the two ways of riding "the whole line"

Every listed, non-named-train line of jp, gb, de, fr, kr, us: **3,971 lines**.

- **A, one ride**: from the first to the last stop of the line's `display` order (what Anita
  does). 3,223 lines have two stops in `display`. Lines with branches cannot reach 100% this
  way, and that is right.
- **B, best case**: every stop-to-stop shortest path on the line, all at once. Whatever B
  leaves uncredited no set of rides on that line can ever credit.

| | lines | 100% today | shows 99.9% today |
|---|---|---|---|
| A, one ride | 3,223 | 1,291 (40%) | 447 |
| B, every stop pair | 3,971 | 1,459 (37%) | 514 |

## Causes, biggest first

### 1. Arithmetic: a fully ridden line comes out a hair under its length (all of the NYC 99.9%s)

With every section ridden, `lineKm` sums `secKm * 1` over the sections and divides by
`line.km`. Two ways that misses 1.0 exactly:

- **Float order**: A Train sections sum to 57.483999999999995 against 57.484; C 29.846999999999998
  against 29.847; 7 16.608999999999998 against 16.609. `pctLabel` shows 100 only for exactly
  100 and caps everything else at 99.9, so these read "99.9%, 57.5 of 57.5 km". This is all of
  A, C, 7, 3, 4, <6> in her screenshot's kind of case: with every stop pair ridden, nothing is
  left uncovered on any NYC line.
- **Rounded section lengths**: `line.km` comes from unrounded lengths, each section's km is
  rounded to the metre, so the sum misses by up to 8 m either way (jp 156 lines short, de
  154, fr 58, gb 55, kr 20; us none). Short by any amount, the line can never show 100.

In B, 306 lines are float-only and 210 rounding-only. In Japan this is 193 of the 199 lines
that fail B; in Korea 26 of 27.

**Fix**: `lineKm` returns `line.km` when every open section is ridden in full (frac 1;
direct rides and `mergeSpans`' end snapping both give exactly 1). A: 1,291 -> **1,731** lines
at 100%. B: 1,459 -> 1,965. Nothing can wrongly reach 100% through this: it fires only
when every section is wholly ridden. (The "finished" counter already used `> 0.999` and
was never affected.)

### 2. Stubs: track of the line with no stop on it, past the last stop (her hypothesis)

Real, but not what her NYC screenshot showed (NYC has none). Common in register countries:
the register ends a line at a junction or switch a few hundred metres past its terminus
station ("Lichtenfels, W 038", 0.78 km past Lichtenfels; ECML's 104 m past Edinburgh
Waverley; "Hamburg Hbf (S-Bahn), W 58", 17 m), at a border point (Bayerisch Eisenstein to
the Czech border, 55 m), or on a stub siding. `isStop` is false there, so no ride can end
there, and a stop-to-stop ride never covers it.

Measured as the stopless trees left when non-stop leaf nodes are pruned off a line's graph,
hanging off a stop: 2,205 such trees in the sample. By size: 203 under 250 m, 328 under
500 m, 310 under 1 km, 318 under 2 km, 418 under 5 km, **628 over 5 km (18,878 km)**. The
long ones are not protrusions: LGV Sud-Est from Bif. Lieusaint to Le Creusot TGV (273 km),
US freight subdivisions with one Amtrak stop, register lines whose stations all lie at one
end. Those are reachable only by a traced journey through the junction, which works today.

**Fix**: credit a stub along with any ride on its line that reaches the stop it hangs off,
when it is at most **1 km and at most 10% of the line**. Only trees hanging off a stop count
(not off a junction mid-line, which would be a connecting curve the rider did not take).
In A, 690 stubs get credited at 680 ride ends; on top of fix 1, about **326 more lines reach
100%** (1,731 -> ~2,057 of 3,223, 64%). The 10% share keeps 49 short lines from reaching 100%
off a stub that is a real part of them (Manchester and Ordsall Lane Junction Branch would go
53% -> 100% off Deansgate's 1.0 km; Larkhall Branch 79.5% -> 100%).
Cap trade-off measured (A, lines at 100% after fix 1, no share cap): 250 m 1,798; 500 m 1,948;
1 km 2,106; 2 km 2,261; 5 km 2,488. Past 1 km it starts crediting things like the 1.7 km
from Schnelldorf to the 4951/5902 line change, where trains run on to Crailsheim.
The stub's own footprint is credited, so country totals rise by the stubs' km (de sample
ride set: +0.78 km; gb: +1.05 km).

Wrongly reaching 100%: a stub under the cap that a rider could have ridden separately and
did not. With the 1 km / 10% caps the largest credited ones are platform-to-switch runs at
termini (Wabern 0.96 km, Königs Wusterhausen 0.96 km, Irrenlohe 0.98, Capitol 0.99, Bingen
Stadt 1.0). I found none that is a separate passenger route.

### 3. Parts of a line no stop-to-stop ride can reach at all (left alone)

Pieces with no stop (curves, border stubs, the Channel Tunnel): gb 71 lines (180 km), de 118
(321 km), fr 21 (731 km), us 119 (2,321 km). They credit only through a traced journey
across a junction, or a named train over them. No change proposed; a line with no stop
cannot be "ridden start to end".

### 4. Alternatives inside one line (left alone; her F Train)

Two routes between the same stops in one line: a stop-to-stop ride takes one. In B only
jp 4 lines (Sobu Main Line Tsudanuma - Makuhari 4.9 km, Chitose Line Shiroishi - Shin-Sapporo
5.2 km, Tokaido Line Shimbashi - Shinagawa and Musashi-Kosugi - Tsurumi, Chuo Main Line Ogaki -
Sekigahara 16.7 km), gb 17, de 31, us 19, fr 1, kr 1. Physically different track, so not
crediting it is correct; the rider uses "the other way round" or a second ride.

**NYCS F** (her 47.4 of 48.0) is this: the F's OSM relation includes a diversion variant
via 53rd Street, so the line has 5th Avenue-53rd St - 57th St (0.387 km, a link no regular F
takes) and only 35% of 57th St - 47th-50th Sts is shared with it. 0.387 + 0.217 = 0.60 km,
her exact gap. Not fixable by a general rule; the build could drop diversion variants, but
that is a decision for her (see the end).

### 5. Slivers between two services drawn over the same track (Kagayaki / Hakutaka)

Both are named trains on the same stations (Nagano g002031, Omiya, Ueno, Tokyo g003766).
Hakutaka's Nagano - Tokyo stretch is three sections. After a Kagayaki ride Nagano - Tokyo,
Ueno - Tokyo and Ueno - Omiya are credited 1.0, **Omiya - Nagano 0.985**: 0.48 km of the
Joetsu Shinkansen at Omiya (owner jp:6610, Kagayaki's footprint ends at 0.987) and 2.38 km at
Takasaki (jp:6607, Kagayaki's relation leaves the Joetsu track at 0.978, Hakutaka's at 0.92,
and joins the Hokuriku track at 0.949 vs 0.82). The two OSM relations are drawn over
different tracks at the Takasaki junction (most likely one per direction), so their
footprints differ by 2.9 km. The strip diagram calls a section ridden only above 0.999
(`ridden: gid => frac > 0.999`), so the whole 194 km section looks unridden. Same pattern
everywhere two services' relations are drawn a little apart: Nozomi / Mizuho (0.94-0.95),
NEX variants at Tokyo (0.91), Empire Builder / Borealis (0.95), Lyria / TGV at Mulhouse.

**Fix ("siblings")**: after a ride, a section of any loaded line counts as ridden when
(a) both its ends are stops the ride passes, (b) its length is within 10% + 0.5 km of the
ride's run between those stops, and (c) the ride already credits at least **80%** of it
through shared owner track. It is marked ridden on its own line only (added to `direct`);
the owner track under it gets nothing more than the ride gave it, so country and operator
totals do not move.

Measured with one end-to-end ride on every listed line and every named train in the six
countries: **2,991 sections on 1,544 lines** go to fully ridden (jp 328 / 199 lines, us 227 /
120, gb 229 / 155, de 1,636 / 752, fr 532 / 290, kr 39 / 28). In-page test: Kagayaki Nagano -
Tokyo now gives Hakutaka's three sections 1.0; Tohoku Shinkansen and the conventional Tohoku
Line stay at 0 for each other.

Why 80%, from the merges I read:
- 80% and over: all the samples I read are the same track (Nozomi/Mizuho, Haruka/Thunderbird
  Kyoto - Shin-Osaka, Munich Ost - Hbf for every EC/NJ/RE, Exeter St Davids - Central, Glacier
  Discovery / Seward Subdivision). Borderline: ECML vs Great Northern local Finsbury Park -
  King's Cross (0.85), which may be a different pair of the six tracks.
- 50-80%: real wrong merges appear: Midland Main Line fast lines vs Thameslink Kentish Town -
  West Hampstead (0.76), Seoul Line 1 vs Gyeongbu Line Guro - Gasan (0.77), TGV on the classic
  line vs the LGV St-Pierre-des-Corps - Poitiers (0.74).

**The "family" test Anita asked about** (same register line by foot.json ownership, or
shared ref / route_master): evaluated on the same sample. Ownership family alone, without
the overlap test, merges different physical tracks that belong to one register line:
Tokaido vs Yokosuka tracks Shimbashi - Shinagawa (N02 has both as 東海道線; overlap 0),
NEX Tokyo - Shinagawa, Chitose Line Shiroishi - Shin-Sapporo (0.06), RE 6 over 6399 vs the
6107 high-speed line Wolfsburg - Oebisfelde (0.09), NYC R vs N local/express 53rd St - 59th
St (0). Requiring the family on top of 80% overlap changes little (2,637 sections instead of
2,991): what it drops are sections where a small second owner differs between the two (ECML
vs Great Northern Finsbury Park - King's Cross, where a Northern City Line owner appears),
which is no better a guard than the overlap itself. ref and route_master cannot decide Anita's case at all: lines.json has no
route_master, and named trains (Kagayaki, Hakutaka) have no ref or network. So the proposal
uses the overlap test, not a family. It keeps Shinkansen and conventional lines apart
because they share no track (verified: JR Tohoku Line Omiya - Sendai credits 0 km of Tohoku
Shinkansen).

One thing found on the way: crediting a sibling's whole footprint (first version) gave Tohoku
Shinkansen 4.4 km from a conventional ride, because OSM services' Sendai - Nagamachi sections
have part of their footprint on Shinkansen track. Hence siblings mark only their own section.

## The 6 and <6> (case by case, not general)

A ride on NYCS 6 Brooklyn Bridge - Pelham Bay Park credits <6> **68.2%** (16.4 of 24.0 km),
and the reverse ride the same 68.2% of the 6. The <6> runs on the Pelham Line's express track
between 3rd Avenue-138th St and Parkchester: its sections 3rd Av-138th - Hunts Point Av
(3.97 km) and Hunts Point Av - Parkchester (3.15 km) are their own owner track (us:5825,
us:5826) and get 0; 3rd Av-138th - 125th St and Parkchester - Castle Hill get 0.86 and 0.66.
The express sections skip stops, so the sibling rule does not see them either (0% shared).

Proposed: a hand list of line pairs treated as one for crediting, `SAME_TRACK`, used by the
sibling rule in place of the 80% test (the stops and length checks still apply). With
`['m366777', 'm9721630']` a 6 ride gives the <6> 100% and the reverse; country totals do not
move (the express track is not counted as ridden for the US total, only for the <6>'s
percentage). Parallel tracks stay apart everywhere else by default, Shinkansen vs the
conventional JR line included. Other candidates of the same kind if she wants them: <7> / 7,
<F> / F (the same local/express split in the NYC data).

Observation, not changed: the 6's Lexington Avenue local sections lie (foot.json) on owner
sections of the NYCS 4 Train, so a 6 ride credits 10.9 km of the 4. Worth a look in
ownership.py if the 4/6 local and express tracks are meant to be separate.

## Effect of all three general fixes together (A, one ride per line)

1,291 -> about 2,057 of 3,223 lines at 100% (40% -> 64%). What is left is branches (one
ride cannot do them), long stopless runs (cause 2 over the cap, cause 3), and alternatives
(cause 4). Sibling crediting adds the cross-line part (Hakutaka after Kagayaki) on top.

## Proposed change to dist/index.html

Line numbers as of 2026-10-07 afternoon; the hunks are by content. Tested in the page with
the live data (`creditState`/`lineKm` replaced at run time, rides set in memory):
NYCS C 99.9 -> 100, 7 99.9 -> 100, L stays 100; <6> after a 6 ride 68.2 -> 100; Hakutaka's
Omiya - Nagano 0.985 -> 1; ECML end to end 99.8 -> 100; Tohoku Shinkansen after a
conventional Tohoku Line ride stays 0.

```diff
@@ -2372,3 +2372,124 @@ function creditRun(gids, push, direct) {
     }
   }
 }
+
+/* WHAT A RIDE CREDITS BESIDES ITS OWN PATH (2026-10-07; handoff_notes/full_line_not_100.md).
+   For crediting only: the map's white path and a ride's km still come from rideGids.
+   1. STUBS. Track of the ride's line with no stop on it, hanging off a stop the ride reaches,
+      at most TAIL_KM in all and TAIL_SHARE of the line: past a terminus to where the register
+      ends the line, a station throat, a stub siding. No ride from stop to stop can reach it,
+      so a ride from one end of the line to the other stopped at 99.x%. A longer one (an LGV
+      from its junction to its first station) still needs a ride through the junction.
+   2. SIBLINGS. A section of any line whose two ends are both stops this ride passes, about as
+      long as the ride's run between them, of which at least SIB_SHARE the ride already
+      credits through shared track. Two OSM relations drawn a few hundred metres apart over
+      the same line (Kagayaki and Hakutaka at Takasaki: 98.5%) then count as one. A parallel
+      track shares next to nothing with the ride (Shinkansen and the conventional line, the
+      Tokaido and Yokosuka tracks, local and express tracks), so it stays apart. A sibling is
+      ridden on its own line only: the owner track under it gets nothing more.
+      SAME_TRACK names pairs of lines to treat as one anyway, case by case. */
+const TAIL_KM = 1, TAIL_SHARE = 0.1, SIB_SHARE = 0.8;
+const SAME_TRACK = [
+  ['m366777', 'm9721630'],   // NYCS 6 and <6>: Pelham local and express tracks, same stations
+];
+const SAME_TRACK_OF = new Map();
+for (const [p, q] of SAME_TRACK) for (const [x, y] of [[p, q], [q, p]]) {
+  if (!SAME_TRACK_OF.has(x)) SAME_TRACK_OF.set(x, new Set());
+  SAME_TRACK_OF.get(x).add(y);
+}
+
+/* Stop -> the sections of the stopless trees hanging off it, each within the caps: leaves
+   that are not stops are pruned until only stops end the line. */
+function lineTails(line) {
+  if (line._tails && line._tails.gen === DATA_GEN) return line._tails.by;
+  const adj = new Map(), deg = new Map(), gone = new Set();
+  for (const [a, b, km, gid] of line.sections) {
+    if (CLOSED.has(gid) || a === b) continue;
+    for (const [u, v] of [[a, b], [b, a]]) {
+      if (!adj.has(u)) adj.set(u, []);
+      adj.get(u).push([v, km, gid]);
+      deg.set(u, (deg.get(u) || 0) + 1);
+    }
+  }
+  const tree = new Map(), todo = [];
+  for (const [n, d] of deg) if (d === 1 && !isStop(n)) todo.push(n);
+  while (todo.length) {
+    const v = todo.pop();
+    if (deg.get(v) !== 1) continue;
+    const e = adj.get(v).find(x => !gone.has(x[2]));
+    if (!e) continue;
+    const [u, km, gid] = e;
+    gone.add(gid);
+    deg.set(v, 0);
+    deg.set(u, deg.get(u) - 1);
+    const tv = tree.get(v) || { gids: [], km: 0 }, tu = tree.get(u) || { gids: [], km: 0 };
+    tu.gids.push(...tv.gids, gid);
+    tu.km += tv.km + km;
+    tree.set(u, tu);
+    tree.delete(v);
+    if (deg.get(u) === 1 && !isStop(u)) todo.push(u);
+  }
+  const by = new Map();
+  const cap = Math.min(TAIL_KM, TAIL_SHARE * line.km);
+  for (const [u, t] of tree) if (isStop(u) && t.km <= cap) by.set(u, t.gids);
+  line._tails = { gen: DATA_GEN, by };
+  return by;
+}
+
+// Node -> every loaded section ending there.
+let SECS_AT = { gen: -1, by: new Map() };
+function sectionsAt(node) {
+  if (SECS_AT.gen !== DATA_GEN) {
+    const by = new Map();
+    for (const l of LINES) for (const s of l.sections) for (const n of [s[0], s[1]]) {
+      if (!by.has(n)) by.set(n, []);
+      by.get(n).push(s);
+    }
+    SECS_AT = { gen: DATA_GEN, by };
+  }
+  return SECS_AT.by.get(node) || [];
+}
+
+/* {gids, sibs}: the sections a ride credits with their footprints (its own and its stubs),
+   and the siblings it credits on their own lines only. */
+function creditGids(ride) {
+  const sig = DATA_GEN + '#' + GAPS_GEN;
+  if (ride._cg && ride._cg.sig === sig) return ride._cg;
+  const gids = rideGids(ride), line = LINE_BY_ID.get(ride.line);
+  if (!line || ride.whole || !gids.length) return (ride._cg = { sig, gids, sibs: [] });
+  const out = [...gids], have = new Set(gids), sibs = [];
+  const add = g => { if (!have.has(g)) { have.add(g); out.push(g); } };
+  // 1. Stubs off the stops the ride reaches.
+  const tails = lineTails(line);
+  const reached = pathStations(gids);
+  reached.add(ride.from);
+  reached.add(ride.to);
+  for (const n of reached) for (const g of tails.get(n) || []) add(g);
+  // 2. Siblings between two stops of the ride. Not on a ride from a junction mid-section,
+  //    whose parts do not walk as whole sections.
+  if (!ride.cuts) {
+    const seq = [ride.from], cum = [0];
+    for (const g of gids) {
+      const e = SEC_BY_GID.get(g);
+      if (!e) break;
+      const cur = seq[seq.length - 1];
+      seq.push(e.sec[0] === cur ? e.sec[1] : e.sec[0]);
+      cum.push(cum[cum.length - 1] + e.sec[2]);
+    }
+    const at = new Map();
+    seq.forEach((n, i) => { if (isStop(n) && !at.has(n)) at.set(n, i); });
+    const raw = new Map();
+    creditRun(gids, (t, lo, hi) => {
+      if (hi <= lo) return;
+      if (!raw.has(t)) raw.set(t, []);
+      raw.get(t).push([lo, hi]);
+    }, new Set());
+    const R = new Map([...raw].map(([t, iv]) => [t, mergeSpans(iv, secKm(t))]));
+    const pair = SAME_TRACK_OF.get(line.id);
+    for (const n of at.keys()) for (const [a, b, km, gid] of sectionsAt(n)) {
+      if (have.has(gid) || CLOSED.has(gid) || a === b || !at.has(a) || !at.has(b)) continue;
+      const i = Math.min(at.get(a), at.get(b)), j = Math.max(at.get(a), at.get(b));
+      if (Math.abs(cum[j] - cum[i] - km) > 0.1 * km + 0.5) continue;
+      const e = SEC_BY_GID.get(gid);
+      const paired = pair && e && pair.has(e.line.id);
+      if (paired || spansLength(mapBack(gid, R)) >= SIB_SHARE) { have.add(gid); sibs.push(gid); }
+    }
+  }
+  return (ride._cg = { sig, gids: out, sibs });
+}
+
 function creditState() {
   const rides = here();
   const sig = DATA_GEN + '#' + GAPS_GEN + '#' + rides.map(rideSig).join(';');
@@ -2380,7 +2501,12 @@ function creditState() {
     if (!raw.has(gid)) raw.set(gid, []);
     raw.get(gid).push([lo, hi]);
   };
-  for (const ride of rides) creditRun(rideGids(ride), push, direct);
+  for (const ride of rides) {
+    const c = creditGids(ride);
+    creditRun(c.gids, push, direct);
+    for (const g of c.sibs) direct.add(g);
+  }
   const R = new Map();
   for (const [t, iv] of raw) R.set(t, mergeSpans(iv, secKm(t)));
   const touched = new Set(direct);
@@ -2430,7 +2556,9 @@ function rideCredit(ride) {
     if (!raw.has(gid)) raw.set(gid, []);
     raw.get(gid).push([lo, hi]);
   };
-  creditRun(rideGids(ride), push, direct);
+  const c = creditGids(ride);
+  creditRun(c.gids, push, direct);
+  for (const g of c.sibs) direct.add(g);
   const R = new Map();
   for (const [t, iv] of raw) R.set(t, mergeSpans(iv, secKm(t)));
   return (ride._credit = { gen: DATA_GEN, R, direct });
@@ -3496,9 +3624,15 @@ function stats() {
-// Capped at the line's length: its sections can sum a rounding step over it (14.2 of 14.1).
+/* Capped at the line's length: its sections can sum a rounding step over it (14.2 of 14.1).
+   And every section ridden is the whole line, exactly: the rounded section lengths can sum a
+   few metres under line.km, and a float sum lands a last digit short (NYCS A: 57.48399...
+   of 57.484 read "99.9%"). */
 function lineKm(line, frac) {
-  let km = 0;
-  for (const [a, b, secKm, gid] of line.sections)
-    if (!CLOSED.has(gid)) km += secKm * (frac.get(gid) || 0);
-  return Math.min(km, line.km);
+  let km = 0, all = true;
+  for (const [a, b, secKm, gid] of line.sections) {
+    if (CLOSED.has(gid)) continue;
+    const f = frac.get(gid) || 0;
+    if (f < 1) all = false;
+    km += secKm * f;
+  }
+  return all && km > 0 ? line.km : Math.min(km, line.km);
 }
```

Notes on the diff:
- `SAME_TRACK` holds bare line ids, as `LINE_BY_ID` keys them (`m366777` is NYCS 6,
  `m9721630` is <6>).
- `rideKm` and the drawn path still use `rideGids`, so a ride's km and its white line do not
  change. `tripsOver` goes through `rideCredit`, so "which trips rode this" sees the same
  credit.
- `creditGids` is cached per ride on DATA_GEN and GAPS_GEN; `sectionsAt` and `lineTails` on
  DATA_GEN. Siblings depend on which countries are loaded, like the rest of crediting.
- A `whole` ride is left as it is (it already takes every section).

## For Anita to decide

- The NYC F's 0.6 km (5th Av-53rd St - 57th St and a part of 57th St - 47th-50th Sts) belongs
  to a diversion variant in OSM. Keep it in the line (honest about the relation), or drop
  diversion variants in the build? No general rule proposed.
- `SAME_TRACK` entries beyond 6 / <6> (<7> / 7, <F> / F are the same kind).

## Every line 100%-able by clicking its diagram (later on 2026-10-07; landed in index.html)

Anita: "every line should be 100%-able by just clicking dots in its line diagram. if thats not
guaranteed, we haven't gotten this working." Her cases: NYCS 5 Botanic Garden -> President
St left Franklin Av's two sections unridden; the E's Steinway St / 46th St branch; the F at
57th St (the 53rd St diversion stays in the line, her call, but must be fillable).

**What was wrong.** (1) A pick rode the shortest path, which could use a section the diagram
does not draw (a hidden shortcut) while the diagram lit the way beside it: the 5's 0.78 km
Botanic Garden - President St link, the F's 5th Av-53rd St - 47th-50th St link. The ride
showed as through Franklin Av / 57th St and credited neither. 2.5% of all stop pairs in the
sample did this (47,000 of 1.86 million, on 210 lines). (2) Track no pick of two stops can
reach: past the last stop to a junction where the walk finds no stop beyond (US freight
subdivisions, 154 km Hastings - Creston), lines with no stop, a stop alone on its piece of
the line with its stub (Artern), ends where a mapped service's footprint stops short of the
junction (Dortmund 48%). (3) Second routes between junctions that are never shortest
(Stafford - Stone via Norton Bridge, 37 km; East London line past Queens Road Peckham; a tram
loop's one-way block; the Underground Loop's turning loop). (4) Straight-line gaps left out
of riding on named trains (Hakone, Yufu).

**The fix (dist/index.html).**
- `drawnGraph`, `drawnPath`, `lineLayout`'s new `drawn` set: a pick rides the shortest way
  over the sections the diagram draws. `pickedRun` offers the plain shortest way beside it
  where it differs ("via Franklin Avenue" / "not via Franklin Avenue", radio buttons, plus
  "the other way round" as before). A ride stores `drawn: true` only when it took the drawn
  way and that differs; old rides and traced journeys are unchanged (shortest).
  `rideGids`, `rideSig`, `newRide`, `commitRide`, `openRide`, `setRoute`, the saved pick.
- `pickableEnds`: a junction end becomes a pickable row ("junction, end of the track") only
  where nothing else reaches its track: no stop listed past it (contRows) that rides there
  (a `thru` row, or a `past` service whose footprint covers the whole way), and not a stub
  that comes with its stop (lineTails) unless that stop is alone on its piece; on a piece
  with no stop, also where the rows past every end name the same one stop (Basel Schänzli).
- `lineOrphans`, `chainPath`, `chainWays`: sections on no drawn/shortest way between picks
  (and junction ends with rows past them), as chains between two nodes. A pick of the two
  stops nearest a chain (nothing pickable between) offers "by the other track between X and
  Y"; the ride stores `via: 'u|v'`. A chain no two stops can be picked around (Stafford's
  three junctions hang off Stone alone), and the junctions of a stopless balloon loop, make
  their junctions pickable (`pickJunctions`). Cached per line; ≤ 115 ms the first time on the
  largest line (山陰線, 161 nodes).
- `lineKm`: a straight-line gap left out of riding (`line._gaps`, from `lineGraph`) is not
  needed for 100%.
- No false credit added: every new option is a run of the line's own sections, credited as
  any ride (footprints, siblings, `SAME_TRACK` and the guards above are unchanged).

**The checker**: `tools/line_100_probe.js`, evaluated in the page. For each line it makes the
rides a rider could pick (every pick pair by both ways, then where needed the other way
round, the chain ways and the rows past junction ends), keeps those that add a section and
checks `lineKm === line.km`; for a failing line it lists the sections left and why.

| | listed lines | 100%-able before | after |
|---|---|---|---|
| jp, gb, de, fr, us, kr, ch, ru, hk, sg, tw, nl | 6,059 | 5,169 (890 fail, 10,431 km) | 6,059 |
| the other 61 regions | 6,369 | (not run) | 6,369 |
| named trains (13 countries) | 860 | jp 2 fail (straight gaps) | 860 |

Before, by country: gb 101 fail, de 310, fr 59, us 204, ch 65, ru 116, hk 1, sg 1, nl 33
(jp, kr, tw 0). Causes: track past the last stop / no stop at all (most), alternatives
between junctions (gb 12, us 11, ru 2, nl 1), the rest partial footprints. After: none.

Junction rows now pickable (sample): 1,168 on 891 of 6,059 lines (gb 116, de 402, fr 78,
us 321, ch 77, ru 128, nl 45, sg 1); all but Stafford's three are ends of track. Chain ways:
206 on 123 lines.

Left as is: the E's diagram has Steinway St as a spur off 46th St and its 36th St - Steinway
section hidden as a shortcut; it is filled by picking 36th St -> Steinway St "not via 46th
Street". That shape is the OSM relation (the E lists the M/R local stops; its 46th St node
sits on the local track, its 46th St sections on the express track), not the app.

## Round 2 (evening 2026-10-07): one pick fills a line, tracks side by side, no dead track

Anita after testing: the whole 4 train still 99.4%, the 2 not filled; "the B train and Q
train are not identified as the same thing ... the spot fixes for the 6 and 7 train are not
sufficient"; and the junction ride ends of round 1 rejected: "if past a junction there are no
stations, we should not be drawing this track at all". The bar: ONE pick end to end fills an
unbranched line.

**1. Sections alongside (along.py, new; index.html `readAlong`, `alongShare`, `creditGids`).**
The 4's and 2's leftovers were the relation's Franklin Av and Botanic Garden variants: two
sections over one stretch of track, drawn on different OSM ways, so footprints never meet.
`along.py` (run by build_model after ownership; also standalone) measures, for every section,
the stretches lying within 30 m of every other section of the same kind of rail
(kind_family), never high-speed against conventional (register flag, else the footprint's
owners), and **never two sections on different register lines** (their owner register lines,
by footprint, must overlap: Sanyo Electric beside JR Sanyo, Nishitetsu beside JR Kagoshima,
Hankyu beside JR Kobe, the Gyeongbu HSL beside the old line all stay apart). Written as
`along.json` (0.05-2.4 MB a country). The app credits a section ridden, on its own line only
(owner track and country totals unchanged, as siblings), when the ride's sections lie beside
>= 90% of it (`ALONG_SHARE`), or >= 60% when both its ends are stops the ride passes and it
is about as long as the ride's run between them (`ALONG_ENDS_SHARE`; the 3's Franklin Av -
Eastern Pkwy swings 130 m off to the other platform). The footprint 80% test stays. The
`SAME_TRACK` hand list is gone: the rule does 6/<6>, 7/<7> and F/<F> by itself.

NYC, one end-to-end pick each: 4 96.9% -> 100, 2 97.0 -> 100, 3 98.1 -> 100; a B ride gives
the Q 47.6% (DeKalb - Brighton Beach); 6 <-> <6> and 7 <-> <7> 100% each way. Still under
100 from one pick, all real branches or diversions in their relations: A (Lefferts and
Rockaway Park), 5 (Dyre Av), N (Montague tunnel), E (Steinway local), F (53rd St).

What the rule adds on other lines (one end-to-end pick per listed line and named train,
sections the footprint rule did not already give): jp 132 sections / 445 km, kr 11 / 101,
us 665 / 847, gb 155 / 342, de 1,878 / 5,009 (mostly named trains and patterns over register
lines), fr 134 / 6,425 (TGV services over each other), cn 120 / 902. Doubtful ones read:
Yamanote Line -> Saikyo / Shonan-Shinjuku (the Yamanote freight tracks, one N02 line);
Ryomo Line -> Watarase Keikoku; Piccadilly <-> District and Metropolitan <-> Jubilee
(parallel tracks, the NYC local/express pattern); Metra Electric <-> Chicago Subdivision
(4.7 km); cn Jingha <-> Jingcheng, Southern Jiangsu Riverine <-> Nanjing-Hangzhou PDL.

**2. Track leading to no station (build_model.prune_dead_track, new).** A section is kept
only on a way between two anchors of its line: stops, border points, junctions within 3 km of
another country's land, and junctions where other track comes within 300 m (`CONTACT_M`; FRA
subdivisions meet up to ~250 m apart) unless the junction's name says the track ends
(`DEAD_END_NAME`: Gleisende, Depot, Abstell-, Rbf, fin de voie, yard ...), or a station not on
the line lies within 500 m, or an FRA "near X" junction names a station that is not the
line's own. Leaf track is pruned repeatedly, then loops reached from one anchor, then pieces
with fewer than two anchors. Pruned ways stop naming the line in ways.json, so the tiles draw
them as rail with no passenger trains. Logged per country ("track leading to no station").

All 73 countries rebuilt (75.7 min), then the 30 with any pruning again after two anchor
rules were added (52.7 min); build_regions run. Register km, original -> now:

| cc | km | cc | km | cc | km |
|---|---|---|---|---|---|
| am | -18.3 | de | -93.7 | it | -27.5 |
| at | -9.7 (+ OSM -1.1) | es | -36.6 | kz | -289.9 |
| au | -91.4 | fr | -87.7 | nl | -35.9 |
| be | -41.9 | gb | -9.4 | pl | -77.5 |
| bg | -6.1 | hr | -5.6 | ro | -10.3 |
| by | -34.4 | hu | -4.3 | ru | -126.6 |
| ch | -23.8 | id | -14.5 | se, si, sk | -3.0, -14.6, -3.4 |
| cz | -2.9 | ua | -71.4 | tm, us, uz | -14.6, -85.7, -80.5 |

All others unchanged. Total -1,321 km of 679,857 register km. **81 line ids are gone** (whole
lines with no station on them: depots, yards, freight spurs; the list and every country's
km are in `handoff_notes/dead_track_km_2026-10-07.txt`, e.g. de 2419 Düsseldorf Abstellbf, ch Sursee - Triengen, kz Тараз - Жаңатас,
uz Nókis - Shımbay, nl Lage Zwaluwe - Moerdijk); no id changed otherwise, none new.

**3. Junction ride ends removed** (rows, `pickJunctions`, the stuck-chain and balloon picks).
In their place, so a junction end leads to a stop to pick: thru rows to stops a service also
reaches (and the service's `past` row dropped there: the thru ride covers this line's track
in full, and works at both ends of one pick); a far walk (400 km, 14 steps) where the normal
one finds nothing; `nearRows`, the stops of any line's section passing within 300 m (Lincoln
on the California Zephyr for the Hastings Subdivision); rows kept when the hub rule would
leave none (Seaford Branch); the line's own stops on another piece of it (Australia's "near
Robina"); stubs past a line's end stop up to 2.5 km whatever the line's length
(`TAIL_END_KM`). And a bug fixed on the way: a third of all sections' points run from b to a
(us 2,582 of 7,467), and cutGraph, departures and throughEnds' heading assumed a to b, so a
ride from a junction on such a section was credited on the wrong side of it.

**Numbers** (tools/line_100_probe.js; before = round 1's code without junction picks and
without along.json, on the old data; 13-country sample):

| | before | after |
|---|---|---|
| lines 100%-able by picks (jp gb de fr us kr ch ru hk sg tw nl cn) | 5,984 of 6,867 | 6,740 of 6,826 |
| one end-to-end pick fills it, lines whose stops form a path | 4,167 of 5,237 | 5,118 of 5,237 |
| ... the same, strict (no other junction end) | - | 5,094 of 5,163 |
| all 73 countries, 100%-able | - | 12,211 of 12,358 |
| all 73, one pick fills an unbranched line (strict) | - | 9,226 of 9,313 |

What still fails (147 lines, about 4,160 km left unridable; ru 57 lines and 3,164 km of them): lines whose stations the data does not
mark as stops (ru tariff lines such as Tommot - Nizhny Bestyakh, 0 stops), ends whose stops lie
in a neighbour not loaded with them, US/AU register pieces with no stop between two junctions
whose rows both name the same stop, and Stafford - Stone's junction triangle (reachable only
through junctions in mid-line, which have no rows). Junction picks would have fixed all of
them; Anita turned those down.

## How this was measured

Scratch scripts (session scratchpad, not kept): `probe.js` (causes per line, A and B),
`probe2.js` (stub caps), `probe4.js` (sibling candidates, one ride per line incl. named
trains), `proposed.js` + `test.js` (the diff loaded over the page's functions, old vs new on
the same rides). Driven by a copy of `neighborhoods/tools/screenshot.js` with a configurable
port, headless Chrome with its own profile.
