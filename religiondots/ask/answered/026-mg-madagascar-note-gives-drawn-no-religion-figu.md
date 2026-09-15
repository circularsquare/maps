# 026 — mg: Madagascar note gives drawn no-religion figures as survey answers

Summary: Madagascar's note calls no religion the answer of a fifth or more in most regions (Sofia 39.4%), but that includes 90.5% of traditional answers; Sofia answered 27.4% none. Reword to say drawn? Recommended: yes.

*Filed 2026-09-15 by session `d743fc47`. Anita's call; nothing is waiting on it.*

## What I did

Left Madagascar's `note_public` as the builder wrote it. The dots are built and do not change
whichever way this goes; only the note's wording does.

## What it costs to reverse

An edit to `note_public` in `countries/mg.py`, `python tools/check_md.py`, then `tiles.py
--refresh-meta` (about a second, no retile).

## Why it is yours rather than mine

The voice and content of a `note_public` are yours (`CLAUDE.md`), and this sentence tells readers
what people in each region said about their religion.

## The detail

From the review of `mg` (`sources/mg.md` §11). Madagascar's None and traditional answers swap between
Afrobarometer rounds, so the build places them as one box and splits it 90.5/9.5 at the recent
rounds' ratio, inside both DHS surveys' range.

- **The note says** no religion is "the answer of a fifth or more of the people interviewed in most
  regions", citing 39.4% of Sofia.
- **What people answered:** in Sofia, 27.4% said none and 16.2% said traditional. The 39.4% is the
  drawn figure, which includes 90.5% of the traditional answers.
- **Count of regions:** outside the central highlands, 6 of 16 regions reach 20% "none" by answers;
  11 do as drawn.
- **Suggested wording:** "No religion is drawn at 12.7%, and at a fifth or more of the people in most
  regions outside the central highlands: 39.4% of Sofia and 27.3% of Androy."
- **A second, smaller point:** the note calls Anglicans and "other religions" too few to place, but
  `Other` (3.34%) and Anglicans (1.15%) are both larger than Muslims (1.30%), who are placed. The real
  reason is that their pattern does not repeat between the two halves of the survey.

Options:

1. **Use the suggested wording**, and say the Anglican and other answers are not placed because their
   pattern does not repeat. The reviewer recommends this, and so do I.
2. **Leave the note as built.**


---

## Ruled 2026-09-15 by Anita: use the rewording

*"okay"*

Option 1: the no-religion sentence says drawn, and the Anglican and other answers are described as
not placed because their pattern does not repeat.
