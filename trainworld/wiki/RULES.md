# How to edit the wiki

Build and check: `python trainworld/tools/wiki_build.py` from the maps root (`--check` writes nothing).
A page with anything wrong stops the whole build, so a bad page never reaches the site.

- **One idea a page.** If a page needs a second heading to stay readable, it is probably two pages.
- **Short and plain.** Written for a curious player: short sentences, plain words, a little cute,
  following "User-facing text" in Anita's global CLAUDE.md. It says what happens, in the present tense.
- **Link with `[[page-name]]`** or `[[page-name|words to show]]`. Page names never change; they are
  what agents search for. Every page links out, and `index.md` links to every page.
- **Figures come from the code.** Write a marker, never the number:
  `{{sim/src/track/params.rs:BASE_COST_PER_KM}}`, `{{sim/src/track/params.rs:LEVEL_MULT[3]}}`,
  `{{sim/src/demand/kernel.rs:Params.road_city_kmh}}` (the value in `impl Default for Params`),
  `{{app/src/game/income.ts:FARE_E}}` (TS works the same). A worked-out figure is arithmetic in the
  marker, over one file's names: `{{sim/src/track/params.rs:UPKEEP_BORED_KM * ECONOMY / 1000}}`.
  Add `|0` to round. A build fails if a name has gone.
- **No figure can be pulled? Leave it out.** A number buried inside a function, or a table in
  data, gets words instead: "slower in dense places", "counts more". Don't dress a tuned number up
  as a word ("twenty") to get past the check. Shapes are fine: "half the gap between trains".
- **Each page ends with `In the code:`** and the file or files that own the mechanic, in backticks.
  That is the only place file paths appear.
- **Stays out:** task ids, build or pipeline steps, the history of why something changed, anything
  better read in code (exact formulas, every edge case, station naming). The spec keeps those.
- **Don't write about what's moving.** A mechanic being changed right now waits until it has landed.
- A changed rule changes its page in the same edit, as with the spec.
