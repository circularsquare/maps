# religiondots — notes for agents

`spec.md` is the load-bearing record and later sections reverse earlier ones; read the one
that applies before changing behaviour it describes. `COMMANDS.txt` has the new-country
checklist and every command in build order.

## Before writing any user-facing text

**Read the field docstring at the top of `countries.py`.** It documents `how`, `fill`,
`grain`, `gap` and `note_public`, and carries Anita's rules on voice. The short version, all
of it hers:

- **No em dashes** in `how`, `fill`, `grain` or `gap`. Comma, semicolon or bracket instead.
  Asserted at the foot of `countries.py`, so a slip fails the import.
- **No markup in those four fields either.** The viewer escapes them, so a `<b>` or a
  backtick reaches the reader as itself.
- **In a `note_public`, bold only a figure inside a sentence.** A `**bold sentence.**` that
  starts one is rendered as a paragraph break and loses its bold (spec §7d) — that is how a
  long note is meant to be structured. Reaching for bold to make a point loud produces the
  listicle voice, which is the specific thing that reads as machine-written.
- **Say the specific thing.** "from the 2010 census", not "from an earlier source". A
  phrasing general enough to fit every country describes none of them.
- The existing notes predate these rules and are full of em dashes. They are **Anita's to
  clean up** — don't convert one as a side effect of editing near it, and don't add new ones.

`python tools/check_md.py` after editing any `note_public`: it applies the viewer's own
markdown regexes to every note and fails on a marker that would reach the reader as text.

## Two things that bite

- **`tiles.py --refresh-meta`** pushes edited display fields into `counts.json` in about a
  second. Editing a name, source, note or view box does **not** need a retile.
- **Other sessions edit this tree at the same time.** Check a data file's mtime and size
  before blaming the code, and prefer a script that matches on text over one that matches on
  line position.
