# 021 — jo: Jordan and Iraq: does Egypt's governorate ruling (ask 001) cover them

Summary: Jordan (Christians) and Iraq (Sunni and Shia) shipped at governorate citing Egypt's ask 001, with no ask of their own. Iraq's Christians and other minorities are spread at the national rate. Confirm as built (recommended), or coarsen either?

*Filed 2026-09-14 by session `afaeb3fc-asks`. Anita's call; nothing is waiting on it.*

## What I did

Nothing new: Jordan and Iraq (both built 2026-09-09) stay as drawn, at governorate. Both builders
treated ask 001 as covering them and filed nothing (`sources/jo.md` §7, `sources/iq.md` §8).

## What it costs to reverse

Jordan as one national unit: a two-line change to `sources/jo.py`'s `units` (per `jo.md` §7), a
re-run and the build tail. Iraq without sect geography: an edit to `sources/iq.py` (`CARRIES` or the
units), a re-run and the build tail. Nothing else on the map depends on either.

## Why it is yours rather than mine

AGENT_BRIEF.md §3, the §14 bar: at what resolution a group is drawn where publishing where it lives
may matter. On 2026-09-08 you said of Jordan, Lebanon, Iraq and Yemen: *"build them, and decide what
to show once there is something to show"* (`queue.md`, From the lead triage, §A). Two were then
shown without coming back to you. `ask/RULINGS.md` lists this as ambiguous.

## The detail

**Jordan**: 12 governorates, 11.9M people, Arab Barometer waves II-VIII. Muslim and Christian are
both drawn on their own governorate shares. Christians are 1.40% nationally, highest in Balqa (4.33%),
Ajloun (3.67%) and Karak (2.32%). Amman is 1.57% but holds 47% of the Christians drawn; Jerash and
Tafilah draw none. The split-half passes (+0.617), but not with Balqa or Ajloun left out (+0.509).
The builder's case: Jordan's Christians have reserved parliamentary seats and churches that publish
their own counts, so their geography is not hidden, and the survey has nothing below governorate.

**Iraq**: 18 governorates, 46.1M people, waves V, VI-3, VII and VIII. **The first Sunni/Shia
geography on the map**: Shia 45.2%, Sunni 30.6% and "just a Muslim" 21.9%, each on its own
governorate shares. **The minorities are not placed.** Christians (0.31%, 25 respondents) and other
religions (0.29%, 27 respondents: Yazidi, Sabean-Mandaean, Kaka'i and others) fall under the 1% floor
and are spread at the national rate over all 18. The builder says that is sample size, not design.
The case for the sect map: the Sunni/Shia split is one of the most published facts about Iraq.

**What ask 001 settled and did not.** It ruled Egypt's Christians at governorate because
"governorates are pretty big". It did not rule on sect geography (`RULINGS.md` lists Egypt's
Sunni/Shia split as undecided). By population Jordan's governorates average about 1.0M and Iraq's
2.56M, against about 4.0M for Egypt's 27. `iq.md` §8 says Iraq's are more than twice Egypt's, which
is wrong by population.

Options:

1. **Confirm both as built** (my recommendation: Jordan's Christians are not in the Copts' position,
   and Iraq places no minority).
2. **Jordan as one national unit**, Iraq as built.
3. **Iraq's Muslim answers at the national rate** (no Sunni/Shia geography), Jordan as built.

Whatever is ruled here also applies to Yemen, which is still queued. Lebanon is closed on the data.
