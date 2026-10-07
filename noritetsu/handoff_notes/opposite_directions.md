# A line's two directions as one track, and the "abroad" outline bug (2026-10-05)

Status at handoff: **code is in the working tree and has been trialled, but nothing has been
rebuilt.** Nothing in dist/data was written by this work. A final `ab.py --all` trial was
still running into the scratchpad when this was written (see "Trials").

## Anita's report

"ive ridden path hoboken-33rd, but have not input journal sq - 33rd, and they seem to get
mapped to opposite tracks on the same line. i feel like ideally these should be merged. i
think it might make sense to try merging opposite direction rail in general."

The screenshot showed JSQ-33rd via Hoboken at 100%, JSQ-33rd at 73% and 33rd-Hoboken at 59%.
`opposite_directions_tools/ride_sim.py` (a Python copy of the app's creditRun, mapBack and
mergeSpans) gives exactly those numbers for a whole ride on "PATH Journal Square-33rd via
Hoboken" (m11100813) over today's dist/data/us.

## Cause: mostly not the opposite tracks

1. **The main cause is ownership's "abroad" rule.** ownership.py tested each free way's
   midpoint against the country outline in dist/regions.json, with a 150 m buffer. That
   outline is simplified to 0.02° (about 2 km) by tools/build_regions.py and leaves out the
   sea. So the PATH ways under Greenwich Village (188 m past the simplified Manhattan shore)
   and under the Hudson counted as abroad. A way that is abroad is owned by nobody, so
   footprints on it credit nothing. In PATH, 9th St - Christopher St kept only 16% of its
   footprint, and Christopher St - Hoboken only the part on land.

   The same bug hits every metro tunnel or bridge over water, and coastal track where the
   simplified coast cuts inland: New York's East River tunnels (the A, 2, 1, N, 7 and F all
   gain 3-11 km), BART's Transbay Tube (+7.7 km to each line through it), Chinese metros
   under rivers and estuaries (Shenzhen 11, 18, 20, Xiamen 1, Wenzhou S1/S2, about 180 km
   in all), Athens' coastal trams, Lisbon, Rio and Santos, Copenhagen, Helsinki, Tunis.

   Measured on today's builds, OSM-line track that sat outside the coarse outline but was
   really this country's own land or water came to about 790 km (`m_all.txt`, columns
   "shore" and "water"): cn 374, us 166, gr 71, br 45, pt 47, it 32, dk 36, za 31, tn 32,
   ca 33, se 23, gb 24, fi 16, tr 17 and so on.

2. **The opposite-track effect is real but smaller.** An OSM line's sections come from
   whichever route relation produced each station pair first (build(): `sections` is keyed
   by the unordered pair). So JSQ-33rd's sections lie on one tunnel track and 33rd-Hoboken's
   on the other. A section's footprint credits whoever owns the ways it lies on. When both
   tracks have the same owner (PATH uptown: both owned by 33rd-Hoboken, since all three
   lines run both directions), the pieces of either track project onto that owner's
   section, and crediting is already right.

   The problem appears only when the two tracks of one corridor have different owners:
   - a line mapped in one direction only, or the two directions mapped as two separate OSM
     lines (Russia's 7509/7510 suburban trains, Austria's IC 610/IC 611, Kazan's
     ring trams 5 and 5a);
   - one track inside a register line's 40 m buffer and the other outside it;
   - the other track abroad, which is case 1 again.

   Measured on today's builds, with the same pairing rule as the code below: way pairs with
   different owners came to about 500 km where both are OSM-owned (ru 191, de 71, pl 43, at
   39, us 44 of which much is junction throats), and about 400 km where a free way's pair is
   register-owned (ru 159, in 99, mostly long-distance trains). Register/register pairs with
   different owners came to 2,058 km (in 538, de 515, ru 201). Those are left alone: the
   register decides which line owns them. Full table: `opposite_directions_tools/m_all.txt`,
   summarised by `summ.py`.

## The fix (ownership.py; build_model.py hooks)

### A. Abroad needs another country's land (`ownership.abroad_mask`)

A free way whose midpoint lies outside the coarse outline (+150 m) is now abroad only when
both of these hold:

- it is more than ABROAD_M (150 m) from the country's own full-resolution outline:
  `borders.outline(cc)`, which honours the OUTLINE overrides, unioned with
  build_regions.EXTRA_AREAS (Russia's annex), loaded from tools/build_regions.py by file
  path;
- it is on another country's land in Natural Earth 1:10m (borders.NAMES). This country's
  own Natural Earth land also counts as foreign where a register OUTLINE cuts it out:
  Crimea for ua, Abkhazia for ge. Over water, the way is abroad only if the nearest land
  within WATER_REACH_DEG (0.3°) is foreign and nearer than this country's outline.

Water near nobody's land stays home, and so does this country's own land. The old outline
test is still the first filter, so this can only move a way from abroad to home, never the
other way.

Unit tests in tests/test_ownership.py `TestAbroad`:
- Greenwich Village, the Hudson and the East River are home for us;
- Toronto and Ciudad Juárez are abroad for us;
- Simferopol is home for ru and abroad for ua.

The tests are skipped if religiondots' outline files are missing.

### B. A line's two directions are one track (`ownership.directional_pairs`, plus a step in run())

**Pairs.** Ways a and b are a pair when all of these hold:
- one route relation of a line runs over a and not b, and another relation of the same
  line runs over b and not a;
- the two relations travel them in opposite directions: travel tangents taken where the
  ways lie side by side, cosine below -PAIR_COS (0.7);
- b lies within PAIR_M (25 m) of at least PAIR_SHARE (50%) of a;
- a and b have the same kind family and the same `layer` tag;
- no other drawn way crosses the line from a's middle to b, so the two tracks are
  neighbours.

Only ways no register line owns are looked for (status none, osm or svc), and never abroad
ones.

**Owner.** Each such free way takes its pair's owner, from the owners as they stood before
this step and only one step out, so a corridor's owner never spreads along a network:
- if the pair is register-owned (status reg, reg-throat or reg-twin), the register line
  owning most of the pair;
- otherwise the fixed rule over the non-named-train lines on both tracks.

In both cases the chosen line must have a running section within OWNER_SEC_M of the way, so
its pieces project; otherwise the next candidate is tried. Without that check, Romania lost
4 km of credit to pieces that found no section. These ways get status 8, "dir-pair", which
appears in the log's "OSM section km by the track under it" line.

**Data.** build_model.build() now records `build.variant_dirs`: for each line id, one
`{way id: +1/-1}` per route relation, saying whether the relation travels the way along or
against its node order (new function `travel_dirs`). main() passes it as
`ownership.run(..., variants=...)`. Nothing else in build_model changed. lines.json and
stations.json are unaffected: every trial matched dist on lines, km, sections and station
ids, apart from other agents' changes listed under Trials.

**Why it cannot merge genuinely different lines.** The evidence is a single line's own
relations, never closeness alone:
- Tokyo's Yamanote and Keihin-Tohoku tracks, the Chuo Rapid and Chuo-Sobu local, and the
  Lötschberg base and old lines are never paired. No one line runs one way on one and back
  on the other, and in jp, ch and Tokyo they are register-owned anyway: register and
  register pairs are never touched. jp moved only 5.3 km of way (Takarazuka Line track to
  the JR Kyoto Line between Osaka and Amagasaki, みずほ track to the Kyushu Shinkansen, and
  similar).
- Four-track lines whose express and local differ by time of day (New York's E and 4):
  - the same direction never counts as opposite;
  - the other direction's track across the express pair is not a neighbour (the adjacency
    test);
  - stacked tunnels are caught by the `layer` test. The first trial, before the layer
    rule, gave 6 km of the 4's Eastern Parkway express track (layer -3, under the local
    tracks at layer -2) to the 2. The final trial no longer does.
- Three-track lines with a peak-direction express (the 5 on White Plains Road) do pair the
  centre track with the local track in the other direction. That follows the stated rule
  (that line's two directions), and the 5 already credits the 2's local track in that
  direction.

## Results

PATH, the ride Anita made (whole JSQ-33rd via Hoboken), with today's data and then with the
final-code trial:

| line | today | after |
|---|---|---|
| JSQ-33rd via Hoboken | 100% | 100% |
| JSQ-33rd | 73.2% | 94.9% |
| 33rd-Hoboken | 58.9% | 96.8% |
| WTC-Hoboken | 50.7% | 50.7% |

What remains:
- **33rd-Hoboken, ~3%:** at the 33rd Street terminal each line's relation uses its own
  stub or platform track. JSQ-33rd via Hoboken's departure track 552266811 is not
  33rd-Hoboken's 552266810. The gap is about 185 m on a 0.78 km section, which is more
  than the app's 150 m / 10% gap closing. Not fixed: these are platform roads, not a
  directional pair.
- **JSQ-33rd:** the rest is track that ride really did not cover, between the Newport
  junction and the Hoboken - Christopher Street tunnel.

Trial summary per country (final code; `compare_foot.py` summary line):
- "credited" is the km of OSM-line sections whose footprint names a running owner;
- "owned" is the app's ownTrack total;
- "new owner" means a line's sections now credit a line they never credited before.

Owned km falls slightly where the two tracks used to be counted for two different lines.
That was double counting, and it is now one owner per corridor.

- us: credited 22,397 -> 22,554 (+157), owned 43,149 -> 43,224; 798 ways no longer abroad;
  125 ways (36.8 km of way) paired, 5 to register lines.
- cn: credited +183, owned +182 (Chinese metros under water); pairing 0.7 km.
- ru: credited 197,234 -> 197,357, owned 80,271 -> 80,194; pairing 471 ways (248.5 km of
  way, 36 to register lines); 294 "new owner" rows (1,060 km).

  Most of ru's are per-direction suburban trains and ring trams now sharing an owner:
  7509/7510 30 km, Kazan's trams 5/5a 21 km, Magnitogorsk's 9/10 13 km. Also the Kerch -
  Anapa diesel taking 34 km of the Crimean bridge from the 183A/184S named train's
  unowned track. **Not reviewed row by row: look before landing ru.**
- de: credited +11, owned -27; pairing 751 ways (81 km); 23 new owners (18 km).
- in: pairing 82 km, nearly all to register lines (Rajdhani tracks beside the register
  line); no change in credit.
- it +15, es +6, au ~0 (Melbourne trams re-owned among 35/57/58/59, Sydney L2/L3).

From the first full trial: the logic is the same apart from the layer and reach checks. The
numbers would shift slightly with the final code; re-read them from aball3 when it finishes.
- pl: credited unchanged, owned -24; 14 new owners (17.5 km), mostly the tourist tram "0"
  taking both tracks of streets it runs one way along, which is the fixed rule.
- at: IC 610/IC 611 (one OSM line per direction) now share an owner; R1 owns R5/S5's other
  track at Lindau - Bludenz.
- cz -10 owned, ua -19 owned (named trains' paired track to register lines), gr +42
  credited (Athens coast), pt +32, br +29, dk +19, tn +16, fi +13, se +14, hk +5, tr +10.
- Unchanged (foot.json identical): al, am, ar, cl, dz, eg, ge, ie, ir, kg, ma, md, me, mk,
  mx, my, nz, th, tj, tm, tw, uz, vn, xk.
- ways.json only: az, lu.
- Changed: everything else.

Lines that differ in trials but are NOT from this work:
- au: "Melbourne - Adelaide Rail Corridor" is renamed "Sydney–Melbourne rail corridor" by
  the working tree. That looks wrong; it is au_register or a colour/name source, not
  ownership.
- xa: three OSM lines +7.4 km, from someone's ru/xa border work.
- gb, fr: the Channel Tunnel agent's register edits.

## Files changed (working tree, uncommitted)

- `ownership.py`:
  - docstring rules 5 and the pairing paragraph;
  - constants WATER_REACH_DEG and PAIR_M/PAIR_SHARE/PAIR_COS;
  - run() takes `variants=None`;
  - STATUS gains "dir-pair";
  - W_layer;
  - the abroad block now calls abroad_mask and logs the ways kept;
  - the new pairing step after "3, 4. twins, then OSM lines";
  - new functions directional_pairs, _land, _home_shape, abroad_mask.
- `build_model.py`:
  - new function `travel_dirs`;
  - build() fills `build.variant_dirs` (one `variant_dirs[lid].append(...)` per relation);
  - main() passes `variants=getattr(build, "variant_dirs", None)` to ownership.run.
- `tests/test_ownership.py`: class TestAbroad (3 tests). `python -m unittest discover -s
  tests -p test_ownership.py` passes; run the full `discover -s tests` before landing.
- `handoff_notes/opposite_directions_tools/`: the measuring and trial scripts. They are
  read-only on data and take paths as arguments.
  - `measure.py`: set MEASURE_DIST to read a trial folder.
  - `compare_foot.py <trial dir> cc...`
  - `owner_changes.py <trial dir> cc`
  - `ride_sim.py <data dir> cc <line id> "<from>" "<to>" <name filter>`
  - `m_all.txt`: the measurements on today's dist.

No app change is needed: foot.json and ways.json keep their format.

## Trials

- Final code, all countries: NORITETSU_AB_DIR = the session scratchpad `.../scratchpad/aball3`,
  summary in `.../scratchpad/aball3.txt` (scratchpad =
  `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\93aeae86-72e5-40ca-8897-eb6a862284f7\scratchpad`).
  It was still running at handoff: us, au, cn, ru, de, fr, gb, in, it and es were done.
  The scratchpad is temporary; if it is gone, re-run
  `python tools/slot.py 3 -- python tools/ab.py -j 3 --all` with your own
  NORITETSU_AB_DIR (~75 min).
- `aball` (first full trial) predates the layer and reach checks; ignore it except as the
  rough numbers above.

## Still untested or unreviewed

- ru's 294 new-owner rows (above), de's 23 and pl's 14: skimmed, not reviewed one by one.
- Not yet checked with the final code: check_model.py per country and the full unit test
  suite.
- How the app reads it: only simulated (ride_sim.py), not seen in the browser.

## Risks

- **Totals fall a little** where two OSM lines each owned one track of a corridor (at -30,
  pl -24, de -27, ru -77 km owned). That removes double counting, but country totals and
  some lines' "owned" km drop. An OSM line that loses its owned share may drop out of the
  lists if its off-register share was borderline. listed() uses offRegisterKm, not owned
  km, so this should not happen, but it is unverified.
- **The fixed rule now reaches across both tracks.** A one-way tourist or night line with a
  low ref takes both tracks of streets it runs one way along (pl's tram 0).
- **abroad_mask's distances are in degrees × 111,320**, which overstates east-west
  distances by 1/cos(lat). That errs towards "abroad", as before. It needs
  ../religiondots/data/geo/ne_10m_admin_0_countries.geojson. If the file is missing, it
  falls back to the old rule for the land part.
- **Build time:** pairing adds 1-20 s per country (de 20 s, us 7 s).

## Next steps

1. When `aball3` finishes, run `compare_foot.py` on it for every country with
   `--rows 30`, and `owner_changes.py` on ru, de, pl and at. Review the NEW rows; anything
   naming two genuinely different lines on separate tracks is a bug.
2. The unit tests pass with the final code (`python -m unittest discover -s tests`, 19 tests,
   run at handoff).
3. Real rebuild (model only would do, since tiles do not depend on ownership; rebuild.py
   `--model-only`), of the countries whose foot.json or ways.json changed, **leaving out gb
   and fr** (reserved for the Channel Tunnel agent; give them this change in that agent's
   rebuild). Tell the managing session the list first, since another agent may rebuild
   RINF countries.

   Rebuild list: us, au, cn, ru, de, in, it, es, jp, pl, ca, at, ch, cz, no, se, ua, az,
   ba, be, bg, br, by, dk, ee, fi, gr, hk, hr, hu, id, kr, kz, lt, lu, lv, nl, pt, ro, rs,
   si, sk, tn, tr, xa, za. Confirm this against aball3's "same/DIFFERS".

   Then:
   - `compare_lines.py save <list>`;
   - `rebuild.py --model-only <list>`;
   - `compare_lines.py diff <list>`;
   - `check_model.py --region <cc>` each;
   - `tools/build_regions.py` (owned totals in regions.json change).
4. Tell Anita: the PATH percentages above, the remaining 33rd Street platform gap, and that
   New York's river tunnels, BART's Transbay Tube and Chinese river crossings now count.
5. Optional: add a HISTORY.md entry.
