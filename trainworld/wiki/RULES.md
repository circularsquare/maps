# How to edit the wiki

**To read a page, open `wiki/read/<page>.md`**: the same page with its figures filled in. Edit `wiki/<page>.md`,
then run `python trainworld/tools/wiki_build.py` from the maps root and commit `read/` with it.
`--check` writes nothing; smoke runs it, so a changed constant with no rebuild fails there. Any problem stops the build.

- **One idea a page.** Written for a curious player: short sentences, plain words, a little cute, present
  tense, following "User-facing text" in Anita's global CLAUDE.md. Wiki text is not listed for her review.
- **Link with `[[page-name]]`** or `[[page-name|words to show]]`. Page names never change. `index.md` has one
  line per page saying what it answers.
- **Figures come from the code.** Write a marker, never the number: `{{sim/src/track/params.rs:LEVEL_MULT[3]}}`,
  `{{sim/src/demand/kernel.rs:Params.road_city_kmh}}` (from `impl Default for Params`), TS the same. A worked-out
  figure is arithmetic in the marker, over one file's names: `{{sim/src/track/params.rs:UPKEEP_BORED_KM * ECONOMY / 1000}}`.
  Add `|0` to round. A JSON file works too, with a dotted key path: `{{pipeline/longhaul.json:tail_onset_km[0]|0}}`. A build fails if a name or key has gone.
- **No figure can be pulled? Leave it out.** A number buried inside a function gets words: "slower in dense
  places". Don't spell a tuned number out to get past the check. Shapes are fine: "half the gap between trains".
- **Each page ends with two lines**, the only place paths appear, and the build checks them.
  `Exact rule:` the doc and "Heading" with the whole rule, plus which trips the page leaves out where
  commutes, everyday trips and trips between cities differ.
  `In the code:` each file, followed by the functions or constants that own the mechanic.
- **Stays out:** task ids, build steps, the history of why, every edge case. The exact rule line points to those.
- **Don't write about what's moving.** A mechanic being changed right now waits until it has landed.
- A changed rule changes its page in the same edit, as with the spec.
