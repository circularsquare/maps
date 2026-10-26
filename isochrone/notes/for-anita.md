# Open questions for Anita

Questions that outlive a session. The orchestrator keeps this list; answered ones move to
DECISIONS.md and leave here.

- **Review the 100 m design** (T-030): `docs/fine.md`. Test tiles for New York and Cluj (T-038)
  wait on it.
- **OSM roads** (ODbL share-alike on published road tiles): yes? Recommended: yes, and T-010 uses
  OSM too, so there is one road source.
- **Where 100 m goes after the test regions**: the world is 15-30 GB, which does not fit on C:.
- **Zoom 9-10**: stay with crisp 1 km cells, smooth the band edges by colouring an interpolated
  time (no new data, every level), or a 100 m patch round the view centre.
