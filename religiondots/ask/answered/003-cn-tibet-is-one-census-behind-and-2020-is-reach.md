# 003 — cn: Tibet is one census behind, and 2020 is reachable in HTML at prefecture tier

*Filed 2026-09-08 by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-cn-tibet`. Anita's call; nothing is waiting on it.*

## What I did

**Nothing — Tibet stays on 2010 magnitudes, as §4 decided on 2026-09-05, and the audit that
raised this concluded the map is correct as drawn** (`sources/cn.md` §9). I am not proposing a
change; I am reporting that one of §4's two stated reasons for choosing 2010 has stopped being
true for this one region, and asking whether that is worth acting on.

## What it costs to reverse

A new reconciliation branch in `sources/cn.py` for the 54xxxx block plus a re-scatter and retile
of China, which is 1.26 million dots and much the largest country on the map. Call it an hour of
build, most of it waiting. Reversing it again afterwards is free — it is one code path.

## Why it is yours rather than mine

AGENT_BRIEF §3, third bullet: *something that changes an already-drawn country's numbers*. It also
sits directly on top of a call you already made and recorded (§4, 2010 over 2020, 2026-09-05), so
it is a request to look again at your own decision with one fact added, not a new question.

## The detail

**The gap.** The map draws the Tibet Autonomous Region at **8.17% Han** (245,261 people), which is
the 2010 census exactly. The 2020 census puts it at **12.15%** (443,370). That is 198,107 people
and about 198 further grey dots. Tibetan goes 90.48% → 86.01%.

**Why §4's reasoning is weaker here than nationally.** §4 chose 2010 partly because 2010 → 2020 is
close to a uniform per-group rescale, so the vintage *"moves magnitude and not geography"*. Across
China that holds. In the TAR it does not: over the decade **Han grew 80.8% and Tibetan 15.5%**, so
the vintage moves the one ratio a reader of this region is most likely to be looking at. Xinjiang
is presumably the same case and was not checked here.

**And why the input is easier than §4 assumed.** §4's other reason was that the NBS publishes the
national province × 56-nationality table only as a 3 MB JPEG scan. Still true of that table. But
**each of the TAR's seven prefectures publishes a `民族构成` section in HTML in its own 2020
communiqué.** Three were read and all three carry it:

| | total | Tibetan | Han |
|---|---|---|---|
| Lhasa | 867,891 | 608,856 | 233,082 (26.9%) |
| Nyingchi | 238,936 | 159,783 | 58,983 (24.7%) |
| Ngari | 123,281 | 107,199 | 14,695 (11.9%) |

So no OCR is needed, and the reconciliation would be to **seven prefecture × 3-category margins**
rather than the current one province × 56-nationality vector.

**What it buys, and it is level rather than shape.** The audit checked the drawn geography against
these same 2020 figures and it already reproduces them: the drawn share of the region's Han is
50.8% in Lhasa against a published 52.6%, 15.0% in Nyingchi against 13.3%, 2.2% in Ngari against
3.3%. Every prefecture is uniformly 4 to 6 points low on the Han share of its own population.
**So this would correct a level that is one decade stale, not a placement that is wrong.**

**What it costs besides the build.** A mixed vintage inside one country: Tibet on 2020, the other
thirty provinces on 2010. That is the specific thing §4 was avoiding, and Hainan (§8) is already a
second reconciliation rule, so this would be the third. It also drops Tibet from 56 nationality
categories to 3 — which loses nothing that draws, since the region's only other religio-ethnic
group is 12,632 Hui and the communiqués do not name them.

**My own read, for what it is worth and it is not much:** leave it. The region is 86% Tibetan on
either vintage, the map's error is 4 points on a category that draws grey, and a second mixed
vintage costs more in explicability than 198 dots are worth. But it is a real known error with a
known open fix, and §3 says that class of thing is yours.

---

## Ruled 2026-09-08 by Anita

**Leave it, as the filing agent recommended.** She added a standing view worth keeping: a mixed
vintage inside China is acceptable in principle, so long as we are confident in the method for
integrating the vintages. She judged this particular case a minor impact and not worth the
re-scatter now. So the objection to acting is the size of the prize, not the mixed vintage
itself, and a future case with a larger prize should not treat this as precedent against it.
