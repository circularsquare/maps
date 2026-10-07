# maps — notes for agents

## Serving maps locally

Anita views the maps through one server, `python serve.py` at the root of this tree
(http://localhost:8800/, a front page with a link to each map). Which maps it serves is the
register `served.json`, and keeping it current is the agents' job:

    python serve.py list
    python serve.py add <name> <folder with index.html> "a few words"
    python serve.py remove <name>

- **This is THE way to serve a map, for her and for you.** Do not start a server of your own
  (`npx serve`, `python -m http.server`, a new per-map serve.py) for a page she will look at;
  register it here and give her the `localhost:8800/<name>/` link. Point your own headless
  screenshots at 8800 too. It speaks HTTP Range, so .pmtiles work, and it opens files per
  request rather than holding them, so a rebuild can replace an archive while it is being
  served (`npx serve` holds files open on Windows and the rebuild's rename fails).
- **Register a map when you start real work on one that is not listed**, so she can open it
  at `localhost:8800/<name>/`. Remove one only when she says it is no longer being worked on.
- Use these commands, not a hand edit; they lock the file, since sessions run at once.
- The running server re-reads the register on every request: no restart after add or remove.
- A map's own older server (`noritetsu/serve.py` on 8767 and the like) still works, and some
  existing screenshot tooling points at it; leave those alone, but do not add new ones.
- A map's page must use relative URLs (`data/x.pmtiles`, never `/data/x.pmtiles`), because
  it is served under `/<name>/`.
