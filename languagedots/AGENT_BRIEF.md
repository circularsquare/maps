# languagedots — standing brief for a country agent

You were spun up to add **one country** to a world dot map of first languages, and nobody is
watching turn by turn. Anita's standing instruction on her maps: trust yourself to make a
reasonable decision and write down why; things marked for her should stay rare. This brief is a
slimmer copy of `../religiondots/AGENT_BRIEF.md`, which ran about a hundred countries; where they
differ, this one wins.

**Do not commit or push.** Anita reviews and commits herself.

## 0. The one-paragraph version

Claim a country. Find its census language table, normalise it, map its labels onto the language
tree, give it a placement layer, scatter it. Decide the arguable calls yourself and record the
reasoning in `sources/<cc>.md`. Stop at a checkpoint (§5) while you still have context to write
the record. Report in the fixed shape (§1).

## 1. Orient, claim, report

```
python tools/claim.py                          # claimed, parked, waiting for the build tail, free
python tools/ask.py                            # what already waits on Anita
python tools/claim.py take <cc> --id <sid> --note "..."
```

- **`<sid>` is the id you were handed.** Use exactly that string. Only if you were given none,
  use the last path component of your scratchpad directory. Subagents share their parent's
  scratchpad, so ids derived independently collide.
- **Scratch files go in `<scratchpad>/<sid>/`**, never loose in the shared scratchpad.
- **Take a parked country first** (`handoff/<cc>.md` says where it stopped), else the top of the
  free list unless you have a reason (another agent is in the neighbourhood; it needs an account).
- Read `queue.csv`'s row and `coverage/<region>.csv`'s row for your country: the sweep's agents
  already found the table's URL, level and categories. Treat that as a lead, not evidence.

**Your final report, under about 300 words, fixed shape:** (1) one outcome line: drawn / parked
at A-B-C / ruling needed / closed, `cc`, grain, people; (2) at most three findings another session
needs; (3) calls someone might reverse, one line each; (4) asks filed, by number; (5) files
touched. Everything else lives in `sources/<cc>.md`.

**Never end your turn to wait for a background job.** Run long steps in the foreground (a
600000 ms timeout covers ten minutes) or poll in the foreground. Report only when done or parked.

## 2. Which countries, and which questions, you may build

`queue.csv` says `free` for tiers A and B of the coverage sweep: a census or register that asks a
language question. Tiers C (no question, near-monolingual), D (ethnicity only) and E (nothing)
said `ruling` until Anita's 2026-10-05 rulings below; rows a supervisor has since set `free` may
be built under them. A proxy of a kind the rulings below don't cover is still hers to allow.

By the question the census asked:

- **mother tongue, home language, main language**: build. Say which in `how`.
- **ex-USSR "native language"**: build, and say in `note_public` that it leans towards identity
  rather than use (Belarus 2019: 54% Belarusian native, 26% at home).
- **indigenous languages only** (Mexico, Colombia, Argentina, Chile, Brazil...): build the
  indigenous languages as measured. Draw everyone else on the country's main language node with
  `tier="derived"` in counts(), and say so in `how` ("census, 2020, indigenous languages; everyone
  else drawn as Spanish"). Anita's ruling (spec §3.5). Look for a second source that speaks to
  the remainder (a survey asking home language, an older census that asked everyone) and say in
  `sources/<cc>.md` whether it corroborates; wanted, not required.
- **multi-answer only** ("languages spoken", Morocco's shares summing to 117%): build, sharing
  each person across the languages named. Within each unit scale the mentions to the unit's
  population, `count = mentions * population / sum(mentions)`, every row `tier="derived"`, and
  `how` says "languages used, several allowed; each person shared across the languages they
  named" (spec §3.6). A single-answer table from the same office beats this, so search for one
  first; microdata giving the combinations beats the scaling.
  **Learned second languages are not home languages** (Anita, 2026-10-05, El Salvador and
  Ecuador): where the question is about ability ("speaks well enough to converse", "speaks a
  language besides Spanish"), a foreign or school language most people named only as a second
  language (English, French in Latin America) is drawn on the country's main language, not
  shared in, and the record tables the folded counts. If that language is also a real first
  language for a sizeable group in the country, say so in the record and flag it to the
  supervisor rather than deciding alone.
- **ethnicity only** (tier D; Anita, 2026-10-05, general after Indonesia ask 009 and
  Bangladesh): build, reading each ethnic group as its language, rows `tier="derived"`, said in
  `how` and `note_public`. **Check retention first:** for each sizeable group, find how many of
  that ethnicity actually speak the language (a census cross-table of ethnicity x language from
  any year, a survey, a published study). Many minorities mostly speak the national language;
  where a source gives a retention share, move the rest onto the national language and cite it.
  Where none exists, say so in the record. Groups with no language of their own (a trade
  community, "other") go on the national language or a remainder node, with the reason. Add a
  "room for improvement" note to `sources/<cc>.md` naming what a real language table would fix.
- **no language question, but a survey or microdata asks one** (Anita, 2026-10-05): build from
  it, rows `modelled` (survey shares x a population base, as `sources/bq.md`). Prefer the source
  with the most regions; pool waves if it helps; say the sample and year in `how`.
- **no language question at all, rich countries** (France, Spain, Italy, Belgium...; Anita,
  2026-10-05): build the national language plus (a) regional languages from regional language
  surveys, as Wales was done, and (b) immigrant languages proxied by citizenship or country of
  birth, the way Latin America's remainders will be. All proxy rows `derived`, every source
  named in `how`.
- **lingua francas in surveys** (Anita, 2026-10-05, ask 018): where a survey's home-language
  answers inflate a lingua franca (Swahili, English, French, Lingala, Hausa...), draw it at
  Afrobarometer R7's separate **mother tongue** question (Q2A; `sources/wafr_afro.py <cc>`).
- **no measured source at all** (Anita, 2026-10-05, ask 019): published speaker estimates placed
  by homeland are allowed, rows `modelled`, the estimate's source and the placement rule in `how`
  and `note_public` (`sources/eg.md` is the model).
- **diaspora languages** (Anita, 2026-10-05): `sources/origin_mix.py` defaults to the home
  country's mix; an obvious override without a citation (Belgians in France on French) is
  allowed, marked "uncited" in `sources/origin_mix.md`.
- **coarse grain** (province or national only): build at the grain published. There is no unit
  floor (Anita: coarse geography never excuses skipping a place; state the grain).
- **old vintage**: fine. Say the year.

## 3. Decide it yourself, and write it down

Yours, with the reasoning in `sources/<cc>.md`: which release and vintage; which table; how to join
names to boundaries (assert the join, never eyeball it); every label → node mapping; new tree
nodes; whether to give up; every word of `how`, `grain`, `gap`, `note_public`.

**The category rules (spec §3), which are the part most likely to go wrong:**

- **Every label the census prints as a language gets a node of its own**, even a small one, even a
  dialect. Spelling variants of one answer may merge, with a comment.
- **An unnamed remainder** ("Others under X", "Other") sits on the narrowest node containing
  everything the census filed there, never guessed into a member. A NAMED label must never sit on
  a group node: the viewer draws group nodes washed out as "language not named" (that bug hit
  India's Rajasthani, 26M people).
- **A label whose meaning depends on place** may be split by the census's own geography, with the
  reason in the mapping (India's Pahari).
- **Your fragment must list every node your mapping uses that `tree.txt` lacks**, even when
  another country's fragment already defines it: the build tail builds only drawn countries, so a
  node borrowed from an unfinished country breaks it (2026-10-05, fr borrowing es's Kriol). Repeat
  the line (`id | label`, no colour) in your own fragment.
- **New nodes go in `taxonomy/tree.d/<cc>.txt`**, never in `tree.txt`. Middle levels are the
  conventional ones; check the family and branch against Glottolog
  (`data/raw/glottolog/languages.csv` and `values.csv`, CC BY). A NEW family or group needs a colour
  in the fragment (`id | label | L C h`, OKLCH; `taxonomy/build.py`'s docstring says which part of
  the wheel each family owns); a language under an existing group gets one generated.
- **Regrouped ids** (2026-10-06): `taxonomy/regroup.txt` moves some nodes after they are written
  (Bantu into Guthrie zones, Austronesian branches, Arabic varieties, languages over their
  dialects; `taxonomy/GROUPING.md`). Keep writing ids as the fragments do; a new Bantu or
  Austronesian language also gets a line in regroup.txt putting it in its zone or branch.
- **Families as most readers know them** (Anita, 2026-10-04, asks 001-003). Nilo-Saharan is one
  root and Omotic sits inside Afroasiatic, though Glottolog splits both; a root per family is fine
  (the Americas' many families each get one); language isolates share one root `isolate`. Where
  a country has very many small families, group them weighing readability against accuracy, and
  say how in the record. A new root under this rule needs no ask, only a colour.
- **Keep indigenous remainders apart from other remainders** (Anita, 2026-10-04). "Other
  indigenous language" goes on `americas_other` (or the region's equivalent), never on `other`
  with "other foreign language", wherever the census lets the two be told apart.
- **Colour for local differentiability within the country.** Languages that border each other on
  the ground must be easy to tell apart, while a family still reads as one region of the wheel.
  South Asia is the model (Anita, 2026-10-04): the Dravidian languages are all blueish-green yet
  clearly distinct. After the build, look at your country's big languages and their neighbours;
  where two generated colours sit too close, hand-pick one in your fragment (the same `L C h`
  third field works on a language line). Do not recolour a node another country already uses
  without checking it there. Good enough beats perfect: colours get tweaked later, so do not
  spend long on it.
- **Sign language, "not stated", "other"**: sign languages on `signlanguage`; "not stated" is not
  drawn (it goes in `gap`); a bare "other" on `other`.
- **No glottocodes from memory.** Leave them out rather than guess.

## 4. Geography: reuse before you build

1. **Is the country in religiondots?** (`queue.csv` `rd_geo`.) Read
   `../religiondots/countries/<cc>.py`: its `place=` is a placement layer already joined, checked and
   cap-reviewed. If your language table is on the same units, map your unit ids onto its `unit`
   column and set `place=RD_GEO / ...` exactly as `countries/np.py` does. Read-only.
2. **Different units, or not in religiondots:** build units in `sources/<cc>_geo.py` (COD-AB,
   geoBoundaries, the office's own layer; assert the unit count and the join both ways), then a
   placement layer with `sources/_grid.py`'s `hex_layer(cc, units, census=...)`, which writes
   `data/geo/<cc>/<cc>_hexes.gpkg` and prints the checks. Where religiondots has a hex layer for
   the country on other units, its hexes can be re-keyed instead of fetching Kontur again.
3. **Read `../religiondots/playbooks/geography.md` before building any geography.** It is a hundred
   countries' worth of traps (wrong twins in name joins, boundary vintages, Kontur false cities).
4. **Be willing to place people** (Anita, 2026-10-05). A proxy that only moves people *within*
   the unit the census counted them in (where in a Land, a province, a district) is welcome and
   needs no ask: the counts stay the census's, only the placement inside the unit is borrowed.
   Prefer the most specific one: immigrant languages by the grid or small-area count of people
   of that specific foreign origin or citizenship (Turkish by Turkish citizens, Polish by Polish
   citizens); failing that, by all foreign citizens or foreign-born; failing that, plain
   population. Say which in `how`. A proxy that changes the *counts* is still Anita's to allow.
5. **Kontur cap blocks**: the scatter stops on an unregistered block at Kontur's cap. Register it in
   **`languagedots/kontur_cap.csv`** (religiondots' columns; `python ../religiondots/kontur_cap.py`
   explains the statuses), never in religiondots' file.

## 5. Build steps and checkpoints

```
1. sources/<cc>_<tag>.py --fetch     -> data/raw/<cc>/, data/normalized/<cc>.csv, with its own checks
                                        (units sum to the national table; a second table of the same
                                        census agreeing per unit is the best check there is)
2. taxonomy/<cc><year>.py            NAMES or CODES dict + resolve(); EXTRA_NODES for anything
                                        resolve() returns that is not a table value
   taxonomy/tree.d/<cc>.txt          new nodes
3. python taxonomy/build.py          fails on a dangling mapping or a clashing node
4. geography (§4)
5. countries/<cc>.py                 ENTRY (fields in countries.py's docstring); copy countries/np.py
                                        Fill `parts` too: one entry per source or rule and the
                                        people it draws (census languages, immigrants, "everyone
                                        else as X"); countries/mx.py and ar.py are examples
6. python tools/check_country.py <cc>     must say ok
7. python scatter.py --country <cc>
8. python tools/claim.py done <cc> --id <sid>
9. sources/<cc>.md                   the record: table, vintage, checks and their numbers, calls
```

Then the build tail, **unless a supervisor spawned you** (your prompt says so): `python
tools/build_tail.py --id <sid>`, backgrounded. It waits for the lock; waiting is correct.

**Context.** At about 50% stop taking on new scope; at about 75% park wherever you are. Checkpoints:
- **A**: you know the table (URL, level, categories), nothing downloaded. Park here if you are past
  50%: record in `sources/<cc>.md` and the queue note, `claim.py park`.
- **B**: step 1 done, normalized CSV reconciles. The designed handoff line.
- **C**: steps 2-6 done. Push through to the end even past 75%; 7-9 are cheap.

`python tools/claim.py park <cc> --id <sid>` writes `handoff/<cc>.md`: **fill it in** (last step
done, what is on disk, what you were about to do, the one thing that will bite). A country that
turns out miserable may be parked with that said plainly; that is useful, not a failure.

## 6. Running alongside other agents

Shared files you may meet: `queue.csv` and `claims.json` (only through `claim.py`),
`kontur_cap.csv`, `ask/` (only through `ask.py`), `spec.md`, `COMMANDS.txt`. **Never rewrite a
shared file wholesale**; edit against a unique anchor. Before any Write to `sources/<cc>*`,
`taxonomy/<cc>*`, `countries/<cc>.py` or `data/`, check whether the path exists. Not yours to edit:
`index.html`, `scatter.py`, `tiles.py`, `rdlink.py`, `taxonomy/build.py`, `taxonomy/tree.txt`.
If you need one changed, say so in your report.

Set `OMP_NUM_THREADS=2` before anything that imports numpy: several of you share a machine Anita
also uses.

## 7. What goes to Anita, and the bar is high

`python tools/ask.py new <cc> --title "..." --summary "<= 40 words"`, then fill in the file: the
decision you already took, what reversing it costs, what you need. **An ask never blocks**; aim for
zero per country. **If the ask needs Anita to download something, put the exact download URL(s)
and the folder to drop the file in at the top of the ask file** (Anita, 2026-10-06: she should
not have to dig through `sources/` for the link). It is for:

- **Not local safety.** Anita, 2026-10-05: language maps of warzones and persecuted minorities
  (Uyghur, Tibetan, Myanmar, Afghanistan) are widely published, so draw at the grain the source
  gives and do not hold anything back or file an ask on safety grounds. A contested census is
  said in `note_public`, not escalated.
- **A source whose terms are unclear or forbid this use.**
- **A grouping §3's "families as most readers know them" does not settle**, or a change to a rule every country shares (the category rules, the
  indigenous-only remainder, the colour plan).
- **Anything needing money or her identity** (a free self-serve account with no identity is fine). Never put her name or email in a
  request; generic browser User-Agent only. Do not fetch citypopulation.de.
  **Never offer "email the office for the table"** as an option (Anita, 2026-10-05: emails from an
  informal project go unanswered). A free self-serve account is acceptable.

Not for her: a mapping call, a join, a vintage, a wording question, abandoning a country.

## 8. Sources and access

**Afrobarometer** (open microdata, home-language question, region and often district codes, ~35
African countries) is already downloaded under religiondots' raw data as .sav files (read-only);
`sources/ng_afro.py` and `sources/ng.md` show the pooled-rounds method and its traps (2026-10-05).

**World Values Survey** ("language at home", region codes, ~100 countries): its online analysis
tool answers plain HTTP form posts, no login; `sources/ir_wvs.py::fetch` is reusable, and
`sources/ir.md` lists its relabelling traps (2026-10-05).

A search result is a lead, not evidence: open the table, cite the release. Offices often hide
their real data behind an SPA's API or a REDATAM server (`../religiondots/sources.md` and the
memory notes in Claude's project memory describe the routes). A bot wall: retry once with a
browser User-Agent, try the Wayback Machine, then record the URL in the queue note and park; Anita
fetches walled files herself. Gated data (registration, application): skip and note it.
