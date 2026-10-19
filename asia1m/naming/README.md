# asia1m naming

Candidate names for unnamed asia1m regions, a batch at a time, reviewed at
http://localhost:8800/asia1mnames/ (`python serve.py` at the maps root).

## Flow

1. `python asia1m/naming/prepare.py` after drawing: exports a copy of `asia1m.ase`, re-extracts
   regions, writes `regions.json` (one fact sheet per region: admin units by share of people and
   cover, townships for China, cities, neighbours).
2. `python asia1m/naming/batch.py <name> asia1m/naming/text.png --province "<province>" [--prefectures A,B]`
   makes `batches/<name>/`: `facts.json`, pictures of the area with the text layer (`labels_*`) and
   with region numbers (`numbers_*`), and one thumbnail per region. It then runs `overlay.py`,
   which puts helper1m's county outlines (other countries: the levels in its `LEVELS`) over each
   thumbnail as `overlay.json`; the viewer draws them faintly, with a hover, under a Counties button.
3. `python asia1m/naming/sources.py <name>` writes `batches/<name>/sources.md`: historical seats
   per region from CHGIS v6 and trimmed Chinese Wikipedia articles for the main units (China only).
4. An agent (Sonnet) follows `BRIEF.md` and writes `batches/<name>/candidates.json`. It uses
   `lookup.py hits <name> <province>` for the recognisability test instead of web searches.
   Then `python asia1m/naming/pins.py <name>` puts a pin on the thumbnail for every candidate that
   isn't a current county or city: the agent's own lat/lon, else the Wikipedia article it cited,
   CHGIS seats, townships in view, title variants; rivers get their OSM line instead.
5. Review in the viewer: Pick / No per candidate, More ideas, a note. Picks stay in the browser;
   Copy feedback gives text to paste back, which becomes `batches/<name>/feedback.txt` for round two.
6. `python asia1m/naming/usage.py <batch> <round> <agent transcript>` logs the agent's tokens and
   tool calls to `usage.csv` (transcripts: `~/.claude/projects/<project>/<session>/subagents/`).
   The page summaries WebFetch makes, and WebSearch's own processing, aren't in the transcript.
   `finish.py <batch> <round> <transcript>` runs pins, usage and `dupes.py` in one go. `dupes.py`
   lists names suggested for more than one region or already picked elsewhere; the viewer shows
   these as a Clash line on the candidate.
   A batch can also be a list of regions: `batch.py <name> <text.png> --keys "<key>;<key>"`.
   Once named, `merge.py <fix batch> [--drop <batch>:<n>]` moves such a fix-up batch's regions
   into the batches they belong to, numbered after the existing ones, and deletes it.
7. `python asia1m/naming/picks.py` collects every batch's picks into `picks.csv`; later batches
   show them as neighbour names and leave picked regions out.

## Files

- `pop_raster.py`: `pop.npy`, people per map pixel from Kontur's global H3 r6 file (the gzipped
  copy in `religiondots/data/geo/kontur/`). Run once.
- `facts.py`, `batch.py`, `prepare.py`: above.
- `../regionnaming/`: the region and admin rasters this builds on (`build_admin.py` lists the
  admin source per country).

`pop.npy`, `text.png` and `regions.json` are regenerable and gitignored.
