# languagedots — notes for agents

**If you were spun up to add a country, read `AGENT_BRIEF.md` in full.** It is the standing brief
for an unsupervised session. `spec.md` is the design record (numbered, DECIDED/PROPOSED);
`COMMANDS.txt` is the build order; `coverage/COVERAGE.md` is what the world's censuses hold.

- Sister to `../religiondots`, sharing its geography READ-ONLY (`rdlink.py`). Never write into
  `religiondots/`. Its `playbooks/geography.md` is required reading before building geography.
- One file per country (`countries/<cc>.py`), one tree fragment per country
  (`taxonomy/tree.d/<cc>.txt`), one mapping per census (`taxonomy/<cc><year>.py`). Shared files are
  few on purpose; see AGENT_BRIEF.md §6 for the ones you may still meet.
- Viewer: `index.html`, served at http://localhost:8800/languagedots/ by `maps/serve.py`. Agents do
  not edit it, `scatter.py`, `tiles.py` or the colours in `taxonomy/build.py`; those are Anita's.
- Do not commit or push.
