# Brief for the noritetsu agent: two tile properties for anitabuilder's real-network layer

From trainworld (the rail game "anitabuilder", `maps/trainworld/`), task T-682. Anita asked for this
to go to whoever is working in noritetsu; nothing in noritetsu has been changed from trainworld.

## What trainworld does with your tiles now
Since 2026-10-19 the game has a "Real network" switch (`trainworld/app/src/map/realNetwork.ts`) that
draws noritetsu's `dist/data/<region>.pmtiles` faintly under the player's network, reading
`regions.json` to pick the regions in view. It reads the `track` layer and only these properties:
`k` (kind), `r` (rank), `p` (pax), `n` (no line runs over it), as built in `build_tiles.py`
(the properties block around lines 455-470, the low-zoom chain merge around lines 396-445). It
draws three classes: metro (`k` subway), main line (`k` rail, `r` 0), other passenger rail (`k` rail,
`r` 1, `p` 1); everything with `n` 1 is hidden. It changes nothing on noritetsu's side and works as
your tiles are.

## What the game would like, if it fits noritetsu
1. **A line type on every track feature, at every zoom**: say `t`, a short code for the type of the
   line(s) running over the way, from the classes you already keep per line (`types.json`,
   `line_types.md`): at least high speed, conventional main line, metro, tram or light rail. Where
   several lines of different types share a way, the "highest" one (high speed over main over
   metro) is enough. The game wants this mainly to draw high-speed lines as their own class.
2. **The line's English (Latin-script) name from about zoom 6**: say `ne`, for a hover that names
   the line ("Tokaido Shinkansen"). Today `nm` is the OSM way's name in native script and only
   from zoom 10, with merged chains (no way id, no name) below that.

## Where the cost is
- The low-zoom merge groups by `(kind, rank, pax, c, n)` (line ~419). Adding `t` to that key splits
  chains where the type changes: small, since type mostly follows rank. Adding a name to the key
  splits far more (every line its own chain): measure the feature count and file size at z5-z9 for
  a busy region (jp, de, gb) before and after, and drop the name below some zoom if it costs too
  much. Names could also live only from z8 or z9; the game can live with that.
- `vector_layers` in the metadata (~line 605) should list the new fields.
- File sizes the game sees now (read by range, a view fetches a few hundred KB): jp 10.5 MB, us 39 MB,
  cn 37 MB, gb 8.9 MB, ca 8 MB.

## Unrelated, for whoever publishes noritetsu's R2 copy
The game would read the R2 copy (`https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/noritetsu/`)
when published elsewhere; that needs the bucket's CORS to allow the game's origin and range
requests (`Range` header, `Content-Range` and `Accept-Ranges` exposed). Not urgent: the game falls
back to the local copy on 8800 in development.

## When it is done
Tell Anita which property names and zooms you chose (trainworld's files are not yours to edit). The
trainworld side then switches its style on `t` and adds the hover, a small change in `realNetwork.ts`.
