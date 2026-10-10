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
- **Run the rebuild yourself.** After any change that touches dots or the tree, run the build tail
  (`python tools/build_tail.py --id <sid>`, backgrounded; it waits for the lock) so the map at
  localhost:8800 shows it. Anita does not run it (2026-10-09: "please do the rebuilds ... i dont
  need to run myself"); never hand it to her. A session running several agents runs one tail
  after they finish, as AGENT_BRIEF says for supervised agents.
- **Judgement calls on sources are ours** (Anita, 2026-10-09): choose what reads closest to the
  language spoken at home, and lean towards splitting a lumped answer into named languages (by
  place from Glottolog or knowledge is fine, recorded as such). Write the call and why into
  `sources/<cc>.md` instead of asking her.
