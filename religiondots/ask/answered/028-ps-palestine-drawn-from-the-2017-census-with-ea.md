# 028 — ps: Palestine drawn from the 2017 census, with East Jerusalem, Gaza as in 2017, and no settlers

Summary: Palestine drawn at 16 governorates from PCBS 2017, East Jerusalem included and Israeli settlers drawn by neither entry; Gaza shown as in 2017 with its 1,138 Christians. Built; flip if Palestine or Gaza should not be drawn.

*Filed 2026-09-15 by session `d743fc47-ps`. Anita's call; nothing is waiting on it.*

## What I did

Drew Palestine at its 16 governorates from PCBS's 2017 census (Table 3: Islam, Christian, other,
in counts), on the same OCHA line Israel's entry was cut on. So East Jerusalem's Palestinians are
drawn here, the Israelis living in the settlements are drawn on neither entry, and the Gaza Strip is
drawn as counted in 2017, with `note_public` saying the war has displaced most of its people since
October 2023.

## What it costs to reverse

Taking Palestine off: remove `ps` from `ORDER` and the next build tail drops it. Leaving only the
Gaza Strip undrawn: a filter in `countries/ps.py::_ps_counts`, a `gap` sentence and a rescatter,
about 10 minutes.

## Why it is yours rather than mine

AGENT_BRIEF §3's first bar, §14: whether a place may be drawn at all. It also sits beside your
2026-09-07 decision on Israel's territorial cut (`sources/il.md` §7), which it completes rather
than changes.

## The detail

- **The count.** 4,665,426 Palestinians counted; the form asks religion of Palestinians only.
  Christians 46,850: Bethlehem 23,165 (10.9%), Ramallah and Al-Bireh 10,255, Jerusalem 8,558, the
  Gaza Strip 1,138 (1,082 in Gaza governorate, about one dot at 1:1,000 placed across a
  governorate of 640,314). Governorate is the finest PCBS publishes, so §14 rule 2 is met, and
  your Egypt, Chad, Burkina Faso and Mali rulings (asks 001, 017, 018) drew minorities at units of
  this size.
- **The settlers.** Your Israel decision cut East Jerusalem in both directions so as not to draw
  the settlements while leaving out the Palestinians they were built among. With Palestine drawn,
  the Palestinians of East Jerusalem and the West Bank are on the map and the settlements are not:
  about 720,000 people (Jews and the register's Others in CBS's 2022 census units beyond the Green
  Line, from `data/normalized/il.csv`). Both `gap` and `note_public` say so. The other ways to go:
  draw them on Israel's entry, which reverses the 2026-09-07 cut; or draw them inside Palestine's,
  which mixes a register count with a census self-identification (§3.1). I did neither. Their
  population is taken out of the placement grid, so Palestinian dots mostly do not land in them
  (`sources/ps.md` §6).
- **Gaza.** The census describes 2017. Drawing it with a sentence is how every older census here
  is drawn (Congo's is 2007); the difference is that the change since is a war.


---

## Ruled 2026-09-15 by Anita: Palestine stays as drawn

*"palestine good. it seems like we can draw the settlers? i think it might make most sense to show
settlers as part of neither country yeah."*

Palestine keeps East Jerusalem, and Gaza as counted in 2017. The settlers belong to neither Israel's
entry nor Palestine's. Still to confirm: whether that means drawing them as their own entry (the
counts are in `data/normalized/il.csv`) or leaving them undrawn as now. Nothing changes until then.
