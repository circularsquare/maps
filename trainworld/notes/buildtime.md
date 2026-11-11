# How long 8800's snapbuild takes, and what would cut it

Measured 2026-11-11 by agent buildtime, in a scratch copy of HEAD (git archive, so other agents' uncommitted edits did
not matter), idle-ish machine, `CARGO_BUILD_JOBS=2`, a copy of `.tw-snap-target` as the Cargo cache. Nothing in the
main tree, `.tw-snap` or 8800's dist was touched. The wasm numbers are for a commit where sim/ changed.

## Stage table

| stage | seconds | note |
|---|---|---|
| git checkout in .tw-snap | not measured | a few seconds |
| wasm-pack track | 72 | cargo compile of the sim crate about 62 (one rustc process, one core), wasm-bindgen under 1, wasm-opt about 7 |
| wasm-pack demand | 108 | same crate with `--features demand`, about 98 compile, wasm-opt about 9 |
| tsc --noEmit | 3.5 | TypeScript 7 |
| vite bundle | 3 | |
| vite writeBundle: packs into dist | 33 | `copyFileSync` of 19.9k files, 5.9 GB, one at a time; 91% of Vite's 38 s |
| keep_assets carry | 0.1 | |
| swap (two `move`s) | about 0 | same volume |
| rmdir /s /q dist-old | 10 | 43.6k files, 6.9 GB |
| wiki_build --html | 0.7 | |
| total | about 232 (3.9 min) | |

Observed runs are 4-6 min: the extra is contention (other agents' builds on the same cores), a cold disk cache for the
6 GB copy, and Windows Defender on 40k files. The shape is the same.

When sim/ has NOT changed, both wasm stages take 7 s and 9.5 s (cargo finds nothing to do; bindgen and wasm-opt still
run), so the snapbuild is then about 2 min.

## Does a stage rerun needlessly?

- WASM: no. `.tw-snap-target` is kept, and a checkout only touches the files that differ, so cargo skips the compile when
  sim/ is unchanged (17 s for both modules, not 180 s). A hash-the-sim-tree skip would save only those 17 s. The sim
  crate also reads `app/src/game/countryIncome.json` and `income.ts` through `include_str!`; cargo tracks those too.
- How often sim changes: 27 of the last 80 trainworld commits touch sim/ or `countryIncome.json`/`registry.json`. A
  snapbuild covers several commits, so most builds recompile.
- Packs: yes. 5.9 GB are copied afresh every build although they rarely change.
- The two wasm modules run one after the other although each compile is single-threaded (one codegen unit, LTO).
  A parallel run used two cores with no slowdown: both done in 107 s wall (track 75 s, demand 107 s) against 180 s.
- The `rmdir` of dist-old sits in the critical path.

## Ideas, ranked by saving

| # | idea | saving | effort | risk |
|---|---|---|---|---|
| 1 | Build track and demand wasm at the same time, each with its own `CARGO_TARGET_DIR` (one dir would serialise on cargo's lock). A `wasm:par` script in package.json or a few lines in snapbuild.cmd; `npm run wasm` stays for agents | 72 s (180 to 108) | 1 h | low. Needs 2 more cores while it runs; a second warm cache of about 0.4 GB. First run after the change is cold |
| 2 | Hard-link packs into dist instead of copying (as terrain-world already is): in the packs plugin `linkSync` with a copy fallback. Linking 19.9k files took 5.6 s against 33 s; deleting dist-old with links is also cheaper (3.4 s against 9.7 s) | about 27 s, plus about 6 s on the delete | 1 h | low to medium. dist/packs then shares bytes with data/packs: anything that rewrites a pack in place (open for write) changes the served game and the freeze. Pipeline writes need to go through temp + replace. deploy.py and `check_dist` only read, so they are fine. `TW_PACKS`/freeze builds are separate files, still fine |
| 3 | Delete dist-old after "swapped" is printed (start it detached) | 10 s of the wait | 0.5 h | low. A tab keeps working, the old files go anyway |
| 4 | Hash the sim inputs and skip both wasm-packs when unchanged | 15 s, only when sim is unchanged | 2 h | medium: a missed input means a stale wasm. Not worth it, cargo already does this |
| 5 | Lighter `wasm` profile for 8800 only (`lto=thin`, 16 codegen units) | about 11 s per module, wasm 6% larger | 0.5 h | medium. 8800 would no longer match the published build for speed checks. The time is mostly rustc front end, not LLVM, so it barely helps. Skip |
| 6 | Skip tsc in snapbuild | 3.5 s | 0.1 h | not worth the loss |
| 7 | `wasm-pack --no-opt` | 16 s | 0.1 h | bigger wasm, and 8800 would differ from the published build. Skip |
| 8 | Wiki only when wiki/ changed | 0.7 s | | no |

Ideas 1 + 2 + 3: 232 s down to about 125 s (2.1 min) when sim changed; about 2 min is the floor from the demand
compile alone (98 s of single-core rustc). An unchanged-sim build would be about 25 s.

## Core cap

`python cores.py list` showed a cap of 14 of 16, nothing claimed at the time. A snapbuild uses 1 core per rustc, so idea 1
asks for 2 cores plus whatever Vite and the copy use; claiming 2-3 for it is enough.

## Not done

No scripts were changed. Measurement scripts were in the agent's scratchpad (timeit.py, par.py, var.py).

## Built (ideas 1 and 3), same day

- `tools/snap_wasm.py` builds track and demand at the same time (same wasm-pack commands as package.json, own target dirs
  `.tw-snap-target` and `.tw-snap-target-demand`, `CARGO_BUILD_JOBS=2` each; the demand dir is seeded from the track one
  when missing). `checks/snapbuild.cmd` now runs it, then `npx tsc --noEmit` and `npx vite build --outDir dist-new`
  instead of `npm run build`. `npm run build` is unchanged.
- After the swap, dist-old is renamed `dist-trash-<n>` and deleted in a separate minimised window; the start of the next
  build deletes any `dist-trash-*` it finds, so a leftover cannot break `move dist dist-old`.
- `.tw-snap-target-demand/` is in `.git/info/exclude`.
- Tested with the real snapbuild.cmd on a scratch layout (a git repo made from HEAD standing in for .tw-snap, a copy of
  `.tw-snap-target`, a fake old dist), sim sources touched so both modules recompiled: 178 s (first run, includes
  seeding the demand dir) and 154 s (second run, wasm 114 s of it). The machine had other agents' checks running (3 cores
  busy) in the first run. Against about 232 s before: about 75 s saved, in line with the estimate (72 s wasm + 10 s delete
  = 82 s). The 154 s still has Vite's 35 s packs copy, which idea 2 (hard links) would take to about 5 s.
- Not covered by the test: the wiki step (scratch layout had no wiki), and the real `git checkout` of .tw-snap.

## Idea 2 built (Anita approved, 2026-11-11)

- `app/vite.config.ts`: with `TW_LINK_PACKS=1` (set by `checks/snapbuild.cmd`, so also snapbuild-frozen) the packs plugin
  hard-links the packs, terrain, drives, outlines, real saves and graphs instead of copying; other builds copy as before.
  It never copies over an existing file (it unlinks first), so a link back to the data file cannot be truncated; a copy
  where a link is refused.
- Guard: every pack writer in pipeline/ and the Rust bins that write packs replaces the file (.part then `os.replace`);
  freeze.py already depended on that and its `check` finds in-place writes. The rule is now in docs/world-run.md's rules.
  One exception, unchanged and already linked in every build: `world_terrain.py` writes `terrain-world/index.bin` in place.
- Scratch test of the real snapbuild.cmd: 174 s, wasm 140 s (a noisy run: fresh seeded target, other agents busy), Vite 24 s
  with writeBundle 20 s (was 35 s and 33 s; the 20 s is mostly the 19k terrain-world and pack links through node, more
  than the 5.6 s a bare link loop took). Link counts in dist/packs were 5 for packs, 2 for world.water and terrain.
- Expect about 2.5-3 min with sim changed on a quiet machine (wasm about 110 s, Vite about 25 s, rest about 15 s), and
  about 1 min when sim is unchanged. Stopping here as asked.
