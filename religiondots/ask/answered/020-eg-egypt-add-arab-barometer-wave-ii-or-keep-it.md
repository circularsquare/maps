# 020 — eg: Egypt: add Arab Barometer wave II, or keep it out as built

Summary: Egypt is drawn without Arab Barometer wave II (1,219 respondents). Adding it moves Christians 6.02% to 5.83% and puts Asyut above Minya on one wave's 70 Asyut interviews. Keep it out (recommended), or rebuild with it?

*Filed 2026-09-14 by session `afaeb3fc-asks`. Anita's call; nothing is waiting on it.*

## What I did

Nothing new: Egypt stays as built on 2026-09-08, from waves III, IV, V and VII (`sources/eg.py::WAVES`).
Wave II is held out by `sources/arabbarometer.py::OMITTED`, added 2026-09-09 when the loader
started seeing wave II. That entry has said since then that the call is yours, but no ask was filed.

## What it costs to reverse

Several edits, not just adding the wave. First, map three wave II governorate labels in `sources/eg.py`'s `NORM`. One of them,
`2007. East` (89 respondents), is not a governorate name and needs a decision. Then re-run
`sources/eg.py`, reword `note_public` (it names Minya) and retile.

## Why it is yours rather than mine

AGENT_BRIEF.md §3: it changes an already-drawn country's numbers, including the one superlative a
reader is likely to take from the Egypt map.

## The detail

Measured by the 2026-09-09 review, which built Egypt both ways and wrote nothing (`sources/jo.md` §9.3):

| | as drawn (III, IV, V, VII) | with wave II |
|---|---:|---:|
| respondents | 6,778 | 7,767 |
| national Christian share | **6.024%** | **5.831%** |
| Christians drawn | 6.48M | 6.28M |
| split-half | +0.518 | +0.541 |

- **Nearly all the movement is Asyut**: 13.58% to 17.45%, which puts it above Minya (16.41%, the
  governorate `note_public` names). No other governorate moves more than 1.6 points, and the order
  of the rest barely changes (rank correlation +0.987).
- **The Asyut change comes from wave II's 70 Asyut interviews**, 28 of them Christian (40.0%, against
  13.6% in the other four waves). Arab Barometer samples in clusters, so one or two Coptic
  neighbourhoods in 2011 would produce this, and no test here can tell that from a real difference.
- **For adding it**: 5.83% is closer to the 1986 census's 5.7-5.8%, and wave II is 18% more sample.
- **The table understates the change**: 229 of wave II's 1,219 respondents have labels `eg.py` cannot map yet
  (`2008. Qaliubiya` 90, `2009. Kafr el-Sheikh` 50, `2007. East` 89). The vintage would become
  2010-2022.

Options:

1. **Keep wave II out, as built.** The reviewer recommended this and so do I: adding the wave changes the one
   headline fact about the Egypt map on a single clustered cell.
2. **Add wave II and rebuild**, with `note_public` naming Minya, Asyut and Sohag as a group, which
   the current note already asks the reader to do.

Once ruled, `OMITTED`'s entry moves into `sources/eg.py` or is deleted, as its comment says.


---

## Ruled 2026-09-14 by Anita: keep wave II out

*"this seems like a very marginal improvement, we can skip it."*

Option 1. Egypt stays on waves III, IV, V and VII, and its numbers do not move.
