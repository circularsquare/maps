# Workflow improvements, approved by Anita 2026-09-14

Why: the shared files are too big to read (sources.md 24,288 lines, countries.py 20,818,
spec.md 12,828 with 231 subsections, queue.md 931), so agents miss things, re-derive lessons, and
read what they do not need. Anita approved every item below. Keep this file short; tick items off
here and put the detail in the thing each item creates.

Anita also asked, standing: **question her rules** when they look contradictory or unhelpful, and
**play a quiet chime** when handing back or needing her.

## Done 2026-09-14 (quick ones)

- [x] **`ask/OPEN.md`**: one line per open question, 40 words or fewer, rewritten by
  `tools/ask.py` after every command. `ask.py new` takes `--summary`.
- [x] **Chat-only questions filed as asks.**
- [x] **Per-agent scratch folders** (`<scratchpad>/<sid>/`), in `AGENT_BRIEF.md` §1.
- [x] **Fixed short report shape** for agents, in `AGENT_BRIEF.md` §1.
- [x] **Reviews scoped by risk**, one reviewer per two countries, in `.claude/commands/rd-super.md`.
- [x] **Chime on hand-back**, in `rd-super.md` and memory.

## The ten items, all done 2026-09-14

What remains is optional: a second item 4 pass on the playbooks' "Not checked yet" traps (it edits
the loaders, so not while builders run), and marking superseded rules outside spec §12 (none found
on a first look).

1. [x] **Rulings digest, `ask/RULINGS.md`**, 2026-09-14. 85 lines: every answered ask and `queue.md`
   ruling, then the older ones in `sources.md`, spec and `estimates_todo.md`, each with what she did
   not decide. Viewer and palette calls are left out. New rulings need a line added by hand.
2. [x] **Survey playbooks**, 2026-09-14. `playbooks/`: ess, lapop, afrobarometer, arabbarometer,
   cab, lits, wvs, dhs_mics, census_table, and geography for the traps every route shares. Each
   trap names its check or says "Not checked yet"; those lines are item 4's list. Pointed at from
   `AGENT_BRIEF.md` §1 and the top of spec §12.
3. [x] **Shared stability module, `sources/stability.py`**, 2026-09-14. Narrow version (Anita left
   it to the session): shared pieces only, callers moved where that changed nothing. be, no, se,
   cab, co, bo and `tools/ess_split_half.py` run wholly through it; tz, hn, pr, lits and ua use
   parts; 19 normalized files byte-identical. lapop, afrobarometer, arabbarometer, ar, do, uy and
   ht use different methods and stay put, listed in its docstring. `tools/test_stability.py`
   checks it against brute force.
4. [x] **Traps become checks**, 2026-09-14, first pass; no drawn number changed. Geography:
   `sources/geo_checks.py` in `scatter.py` (unplaced units, torn polygons, shadowed modules).
   Census: `sources/fetch_checks.py`, `tools/check_na_readers.py`, `tools/check_no_religion.py`.
   Surveys: 21 assertions in the shared loaders. What is still unchecked is listed as "Not checked
   yet" in each playbook.
5. [x] **Batched builds**, 2026-09-14. Builders under `rd-super` stop after step 9; the
   supervisor runs `build_tail.py` about hourly and before any reviewer (`rd-super.md` step 3a).
   `claim.py` lists countries waiting, from `data/build_last.json`. `build_tail.py` clears a lock
   whose process is gone and runs `coverage.py` last. Four agents by default.
6. [x] **Lookup tool and section keys**, 2026-09-14. `tools/where.py <cc>`. New `sources.md`
   sections are `## <cc>-<YYYY-MM-DD>.`, scout sweeps `## scout-<YYYY-MM-DD>-<region>.` (spec §12).
7. [x] **Spec holds current rules only**, §12 only, 2026-09-14. 3,713 lines to 1,506: 78 lessons
   cut to the current rule plus its check and playbook, 1 marked superseded, 20 kept whole; every
   cut line verified present in `spec_archive/12.md`. New route-specific traps go to the playbooks.
   Nothing plainly superseded was found outside §12.
8. **"Nothing is truly dead" sweep.** [x] Script, 2026-09-14: `tools/negatives.py` (127 records on
   44 countries). Missing parts do not pick out the wrong negatives; their shape does (closed on one
   release, never on the questionnaire, nothing said about what was left unchecked), so rank on
   `--shape` and the oracle flag. [x] Scouts on its picks, 2026-09-14: Iran reopens at province level
   from SCI's own 2011 table, held on §14 (ask 023); Taiwan has a county-level survey route (ARDA's
   TSCS copies); Belarus reopens on LiTS but its Catholic geography fails; Gabon stays parked.

## Later

9. [x] **Split `countries.py` into `countries/<cc>.py`**, 2026-09-14. `countries.py` is the loader
   (field docstring, `ORDER`, foot assertion); `countries/_shared.py` holds the 12 helpers several
   countries use. Every entry, all 153 `counts()` results and `counts.json` identical before and
   after. A new country is a new file plus its code appended to `ORDER`.
10. [x] **A light structured queue, `queue.csv`**, 2026-09-14. 206 rows: free 12, held 4,
    deferred 1, blocked 5, closed 31, drawn 153. `claim.py` lists the free ones from it, prints the
    held, deferred and blocked ones, warns where `queue.csv` and `queue.md` disagree, and `done`
    marks a registered country drawn. The free list went from 17 to 12.

## Draft rule: "no religion" boxes (Anita asked for a documented procedure)

Written into spec §3.12 and the census_table, afrobarometer, arabbarometer and dhs_mics playbooks
on 2026-09-14, and linted by `tools/check_no_religion.py`; kept here as drafted. Precedents: Laos, Mozambique
and China 2026-09-14; Anita: *"did we find that like most of these people are traditional? like
over 80%? if so then yeah totally chill."*

1. **Read the questionnaire's wording for the box.**
2. **If the box separately offers traditional or animist,** "no religion" is `unaffiliated`
   (Guinea-Bissau, Chad).
3. **If the box lumps no religion with animist or traditional,** or the source defined religion so
   that animism cannot be answered, draw it as **unknown** (China's treatment) until step 4
   settles it.
4. **Look for a national source that asks the two separately,** such as MICS/LSIS, DHS,
   Afrobarometer, Pew, or census prose.
   - **If one reading is at least 80% of the box,** draw the box as that reading, citing the
     source.
   - **Otherwise it stays unknown.**
   - **Self-description surveys** measure what people call themselves, not practice; say which
     the source measures.
5. **Record the split source and the percentage** in `sources/<cc>.md` and the mapping `REVIEW`.

Applied so far: Laos goes to traditional religion (LSIS, about 99.8% animist; queued in
`queue.md`), and Mozambique to `unaffiliated` (Afrobarometer and Pew, 93-98% no religion as a
self-description; drawn 2026-09-14).
