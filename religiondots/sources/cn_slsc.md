# cn_slsc — Spiritual Life Study of Chinese Residents, 2007

Obtained 2026-09-08 by Anita from **ARDA**, `thearda.com`, free and with no application.
`data/raw/cn/slsc2007.dta`, 3.98 MB, 7,021 respondents, 397 variables. `data/` is gitignored,
so this file is the record.

**Nothing is drawn from this survey and nothing should be.** It exists here for one number,
and that number is the most important caveat on the `chinesefolk` layer (spec §14.22).

| | |
|---|---|
| who | Dr Anna Sun's team, fielded May 2007; distributed by the Association of Religion Data Archives |
| design | multi-stage, **56 locales**: 3 municipalities, 6 provincial capitals, 11 regional cities, 16 towns, 20 administrative villages, KISH grid within household |
| licence | ARDA's standard terms: acknowledge ARDA **and the original collectors**, data "as is", governed by Indiana law. No commercial restriction |
| why not drawn | 56 sampling points is the design §14.16 showed to be a lottery for anything spatially clustered; there is no usable provincial cut |

## THE NUMBER IT EXISTS FOR

It is the only survey this project holds that asks the naming question and the practice
question **of the same people**, which is the gap §14.14 and §14.16 have been asserting from
Pew's summary rather than measuring:

| | | |
|---|---|---|
| `religblf` | *Do you have any religious belief?* | **Yes 15.8%** (1,109 of 7,021) |
| `WORSHIP1` | *Have you worshipped God or gods/spirits in the following settings in the past year?* | **"I never worship" 37.6%** |

So about **62% worshipped** somewhere in the past year — 2,857 at a family grave, 1,279 in a
temple or church, 136 at home — against **15.8% who will say they have a religious belief.**

***Four times as many people practise as name it, and the folk layer draws the naming.*** That
is the sentence `note_public` now carries, and it is measured here rather than borrowed.

`BELIEVE1`, asked as *regardless of whether you have been to churches or temples, do you
believe in...*, returns Buddhism 1,168 (16.6%) against 5,481 (78%) believing in nothing —
which is the same gap from the other side.

## AND IT SETTLES THE CONFUCIANISM QUESTION FROM THE RESPONDENTS' OWN SIDE

`isconfrl`, *Do you think Confucianism is a religion?*

| No | Hard to say | Yes | Refused |
|---|---|---|---|
| **4,068 (58%)** | 1,683 (24%) | 1,207 (17%) | 63 |

§14.22 refused Confucianism because neither CGSS nor CLDS offers it as an answer. This says the
answer sets are not the whole reason: **most Chinese people do not consider Confucianism a
religion either.** A Confucian layer on this map would be asserting a category over the heads
of the people in it, which is exactly what §14.5 exists to prevent.

## What else is in it, unused

Rich on practice and untouched: `frqvener` (venerate ancestors), `frqburn` (burn incense),
`frqbdpry` / `frqwrshp` / `frqread` / `frqvegbd` (Buddhist prayer, worship, texts, vegetarian
diet), `frqrdcon` (read the Confucian Classics), `everpray`, `donated`, `convbudd` / `convconf`
(formal conversion ceremony), `RELIG15A-C` (religion at fifteen) and both parents' and
spouse's religion. `SPC_CITY` and `SPC_CLCY` are the sampling-point codes.

**A practice-basis layer is not buildable from it and not wanted anyway** — §3.1 forbids mixing
bases, and every other country here is `self_id`. The value of the practice questions is as the
denominator the naming question is a fraction of, which is how they are used.
