One fragment per country: taxonomy/tree.d/<cc>.txt, holding the nodes that country adds.
Same format as ../tree.txt (id | label), plus an optional colour for a NEW GROUP OR FAMILY:

    afroasiatic.berber | Berber | 0.74 0.12 95
    afroasiatic.berber.tachelhit | Tachelhit

Colours are OKLCH `L C h` (see ../build.py's docstring for which families sit where on the
wheel). A language under an existing group needs no colour; build.py generates one near its
group. Hand-picked colours for big languages are Anita's to tune, in build.py's HAND.

A node already in tree.txt or another fragment may be repeated only with the identical label.
Run `python taxonomy/build.py` after editing; it fails on a clash or a dangling mapping.
