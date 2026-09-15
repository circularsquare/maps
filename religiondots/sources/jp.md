# Japan — `sources/jp.py`, `sources/jp_checks.py`, `taxonomy/jp2024.py`

Drawn 2026-09-14, twice. **Now 47 prefectures, 20 nodes, 124 million people, 96.4% of
respondents drawn, every row `modelled`.** The first build that day drew one national unit, and
§1-6 are its record and still true. Anita's second reply in ask 014 accepted NHK's 1996 survey for
allocating and a block level from another survey; §7 is that build. §8 is the second source
search, the ISSP walk-through, the Global Flourishing Study check and the Christian question.
Anita's first ruling (`ask/answered/011`) set the route; §11q (the Agency roll) and §11an (the NHK
survey) are the history.

| | |
|---|---|
| counting geography | 47 prefectures; national figures spread over them (`sources/jp_alloc.py`) |
| placement | Kontur 400 m hexagons (2023-11-01), 229,211 populated, keyed to COD-AB ADM1 2019 by `sources/jp_grid.py` |
| basis | self-identification, sample survey |
| population base | 人口推計 2024-10-01 by prefecture, 123,803,000; the 3.6% who did not answer stay undrawn |
| tier | `modelled` throughout |
| national figures | JGSS-2021H, 2022H, 2023D, 2024N pooled, 10,612 respondents |
| block level | JGSS-2015, believe or family religion, six sampling blocks (Iwai 2017) |
| prefecture pattern | NHK 全国県民意識調査 1996, measured off 図録7770 (`sources/jp_checks.py`) |
| check | Global Flourishing Study 2023, open, prefecture-coded (`sources/jp_gfs.py`) |
| ask | **014**, closed 2026-09-14: Christians stay on NHK 1996, and no GESIS request for ISSP |

---

## 1. The baseline: JGSS's published national table

`jgss.daishodai.ac.jp/surveys/table/XXRL.html` prints variable XXRL, *信仰する宗教/家の宗教（本人）*,
for all eighteen JGSS waves as unweighted counts. It was recoded in 2025 from two digits to **164
four-digit codes**; the recoded microdata is only in JGSSDDS, but the marginal table is public.

**The question.** DORL: *do you believe in a religion?* yes / *I do not believe personally, but my
family has a religion* / no. The first two are asked XXRL, *what is it?* Code 8888 非該当 is the
no. A family-religion answer is drawn under the religion named: JGSS-2015 splits 9.2% yes, 21.2%
family, 68.6% no (Iwai, Pew 2017), and yes plus family is what lines up with ISM's "personal
religious faith" (28%) and NHK 1996's "which religion do you believe in" (31.2%). Yes alone would
put Japan at 9%. `taxonomy/jp2024.py` REVIEW has it.

**Code 6000 is a subtotal.** 上記のうち、回答が5ケース未満の宗教カテゴリ. Every wave's rows close on the
page's two 計 rows (all respondents; those asked XXRL) only with it left out. `jp.py` asserts it
per wave.

**The pool.** The four most recent waves, 10,612 respondents, 20-89, stratified two-stage random
samples on six blocks and four city sizes, self-administered. The no-religion share drifts: 64.7%
pooled 2000-2012, 69.0% 2015-2018G, 71.7% here.

| drawn as | share | n |
|---|---:|---:|
| unaffiliated (8888) | 71.68% | 7,607 |
| buddhism.mahayana | 18.80% | 1,995 |
| eastasiannew.japanese | 3.08% | 327 |
| Christian nodes and `unification` | 1.15% | 122 |
| shinto | 0.85% | 90 |
| other.jp | 0.83% | 88 |
| islam | 0.01% | 1 |
| **not drawn** (9999 no answer 361, 8700 don't know 21) | 3.60% | 382 |

Inside Buddhism: school not given 7.01%, Jodo Shinshu 5.52%, Zen 1.84%, Shingon 1.48%, Nichiren
schools 1.43%, Jodo 1.24%, Tendai 0.24%. Soka Gakkai is 1.80% of all respondents and Tenrikyo 0.35%.

**Pew 2023 is not comparable.** Its Japan figures are 46% Buddhist and 42% no religion, from a
"what is your present religion" question with no middle answer. §3.1a: two self-identification
sources are comparable only where they offered the same choices.

## 2. The six blocks: every route is academic

JGSS samples on six blocks (Hokkaido/Tohoku, Kanto, Chubu, Kinki, Chugoku/Shikoku, Kyushu), so a
block variable is in every file. **It is not in any published table of religion.**

| route | checked 2026-09-14 |
|---|---|
| XXRL page | national only |
| SSJDA microdata | application, university researchers or supervised students |
| JGSSDDS | application, approved, advisor or guarantor for some user types; prefecture data needs the head of your institution |
| ICPSR (JGSS 2012, cumulative files) | ICPSR member institutions; the site also 403s a scripted fetch |
| GESIS | personal account, academic terms |
| SSJDA Data Analysis (`online-data-analysis.iss.u-tokyo.ac.jp`, anonymous cross-tabs) | `/api/stats/survey-list` returns **247 surveys and no JGSS**. Its consent screen limits use to academic secondary analysis and asks for outputs to be reported |
| JGSS Research Series papers | the religion paper checked (jgssm13_12) has no regional table |

**One block table is published**, in Iwai's 2017 Pew slides: JGSS-2015, yes / family religion / no
by block. It is one wave, percentages without counts, and not split by religion. Set against NHK
1996 aggregated to the same blocks (2024 population weights, Mie in Kinki):

| block | JGSS-2015 yes + family | NHK 1996 has faith | NHK 1996 Pure Land | NHK 1996 Tendai/Shingon |
|---|---:|---:|---:|---:|
| Hokkaido/Tohoku | 25.9 | 27.7 | 8.8 | 2.2 |
| Kanto | 24.7 | 23.2 | 5.9 | 4.1 |
| Chubu | 34.3 | 36.2 | 17.1 | 2.1 |
| Kinki | 31.8 | 35.7 | 16.3 | 5.9 |
| Chugoku/Shikoku | 36.0 | 39.5 | 17.5 | 8.7 |
| Kyushu | 36.9 | 36.3 | 21.4 | 1.8 |

The two instruments, nineteen years apart, order the blocks the same way except the top two
(Spearman +0.94 on six units). **That is real evidence the religious share differs by block, and
it is still not the spec's test**, which asks whether a category's ranking replicates across waves
of one instrument with counts behind it. Using the 2015 table is option 2 in ask 014.

## 3. The 1996 chart, measured

`sources/jp_checks.py` reads 図録7770's 815×655 GIF. Gridlines every 65 px for 10 points; a
segment is the distance between the centres of the black border runs around it, over 6.5; a label
wide enough to look like a border is merged back into its segment because each colour appears
once per bar.

**Precision.** All 48 bars (nation plus 47) within **0.08** points of their printed totals, mean
0.04. Segments read against their printed labels on a 3× crop agree to **0.14** or better (Tokyo
3.4 / 8.3 / 4.0 / 3.4, Toyama 41.3, Shizuoka 9.4). One pixel is 0.154 points; a segment under about
0.3 cannot be told from zero. **The survey, not the image, limits it**: about 600 answers per
prefecture puts roughly 1.2 points of sampling error on a 10% category and 0.6 on a 2% one.

Written to `data/raw/jp/nhk1996_7770_measured.csv`. Nothing is drawn from it.

**What it shows**, as shares of respondents 16+:

- **Pure Land and Jodo Shinshu is the geography.** Toyama 41.4, Fukui 41.4, Ishikawa 36.2,
  Hiroshima 35.2; Okinawa about 0, Tokyo 8.3, Ibaraki 4.2, Tochigi 3.1. A 40-point range.
- **Tendai and Shingon is Shikoku and Okayama**: Tokushima 19.9, Okayama 16.7, Kagawa 14.1; about 1
  in Akita, Aomori, Shizuoka, Yamanashi.
- **Zen is the Tohoku Pacific side**: Iwate 13.1, Miyagi and Akita 9.5, Shizuoka 9.5; about 0 in
  Kochi and Okinawa.
- **Nichiren is Yamanashi (8.9) and Shizuoka (7.2)**, and Okayama 5.9 ("Bizen Hokke").
- **Soka Gakkai is flat**, 1 to 5 everywhere: Osaka 5.2, Hiroshima 4.9, Okinawa 4.1, Tokyo 4.0.
  At n≈600 most of that range is noise.
- **Christian**: Nagasaki 5.0, Tokyo 3.3, Okinawa 2.5, Kanagawa 2.2; national 1.46.
- **Has any faith**: Fukui 58.0, Hiroshima 53.7 against Okinawa 7.8, Chiba 18.1.

So the schools carry most of Japan's religious geography, and it is exactly what a one-unit map
cannot show and a six-block map would mostly flatten: the Pure Land range inside Chubu alone runs
from Toyama's 41 to Shizuoka's 6.

## 4. The Agency roll's Christian line

Anita's lead: nippon.com's *Proportion of Christians by Japanese Prefecture*, from the fiscal 2017
宗教統計調査, Tokyo first and Nagasaki second. Checked on the 2024-12-31 edition (table 2(2),
`statInfId=000040387906`).

**How the roll assigns people to a prefecture.** 宗教年鑑 令和5年版 Q4-Q5 says the figures are what
the umbrella corporations, independent corporations and prefectural registrars report, that
"believer" is whatever each body calls its members, and that each body counts its own way. It does
not say where a member is placed. The numbers do:

- **Tokyo holds 46.9% of the roll's 1,872,320 Christians on 11.5% of the population.**
- Tokyo reports **886 believers per Christian body**; the rest of Japan 133. At the rest's rate
  Tokyo would report 131,721; the excess is **746,776, 39.9% of the national roll**.
- Table 5: **968,496 of the Christian roll are independent corporations** (単立宗教法人), which
  report at their registered address. The excess is most of that.
- Outside Tokyo the roll runs at a **median 0.60** of NHK 1996's share. Spearman against NHK: +0.61
  over 47, +0.55 without Tokyo.

So the roll assigns members to **where their church or head office is registered**, which is
residence for an ordinary parish and Tokyo for a national body.

**The national total against self-identification.** Roll 1.51% of everyone (2024); NHK 1.46% of
16+ (1996); JGSS 1.15% of 20-89 (2021-2024). The match is between totals only.

**Catholics, from the Catholic Bishops' Conference** (カトリック教会現勢 2023, 15 dioceses, 418,101):
correlate +0.886 with the roll's Christians and +0.318 with NHK. Catholics are 96.5% of the roll in
Nagasaki and 10.4% in the Tokyo archdiocese. **Nagasaki is real on every source**: roll 4.72%, NHK
5.00%, Catholic 4.37% of the diocese.

**Verdict: not an allocator.** The Tokyo excess belongs to people whose prefecture nothing
publishes, which is the case the proxy rule refuses.

## 5. Other allocating candidates, and what each measures

| candidate | measures | verdict |
|---|---|---|
| Roll, Shinto and Buddhist lines | shrine parish residents and temple households (§11q) | catchments, no |
| Roll, Christian line | members at the registered address (§4) | head-office map, no |
| CBCJ Catholics by diocese | registered parishioners, 15 units; foreign Catholics largely unregistered | a roll for one denomination; the unregistered share per diocese is unpublished |
| Prefectural 宗教法人名簿 (§11q) | temples and churches by sect, with addresses | buildings, §8.3 refused |
| JGSS-2015 block table | believe or family religion, six blocks, one wave | ask 014 option 2 |
| NHK 1996 chart | every school, 47 prefectures | Anita: a check, not the basis |
| Komeito vote by prefecture, for Soka Gakkai | votes; JGSS-2015 says 30% of Komeito supporters are not Soka Gakkai, nationally only | the non-matching share per prefecture is unpublished; **not fetched** |
| ISSP Japan (fielded by NHK) and WVS Japan | religion every round; region variables not checked | GESIS account / WVS download form; **not checked** |

## 6. Numbers to check a rebuild against

```
XXRL page 更新日 2025-05-25: 164 codes, 18 waves, every wave closes on its 計 rows without 6000
pool 2021H 3,522 + 2022H 3,145 + 2023D 1,277 + 2024N 2,668 = 10,612; 8888 = 7,607
drawn 10,230 on 15 nodes; not drawn 382 (3.60%)
Kontur JP 2023-11-01: 229,211 hexes, 123,294,123 people
chart: 48 bars, |measured - printed| mean 0.040, max 0.080
roll 2024-12-31: Christian 1,872,320; Tokyo 878,497 on 991 bodies; 単立 Christian 968,496
```

## 7. The prefecture build (second build, 2026-09-14)

Anita, ask 014: *"if nothing is more recent, 1996 is better than nothing. we can use it for
allocating"*; a block pattern *"can come from different survey"*; the Buddhist split left to the
agent.

**The construction** is in `sources/jp_alloc.py`'s docstring. National totals per JGSS code do not
move. Non-response is laid on every prefecture at the national rate. Each prefecture's share naming
a religion starts at NHK 1996's and moves by one log-odds shift per JGSS block until the block
matches JGSS-2015's share relative to its own national figure, rescaled to 2021-2024's 25.64% of
answers. The religious people are then split by iterative proportional fitting: rows are the
prefecture religious totals, columns the JGSS national totals, seeds NHK 1996's share for the
answer each code falls under. It converges in 11 passes; every prefecture sums to its 2024
population and every code's national total is its JGSS share to within 5e-7.

| JGSS codes | NHK 1996 answer whose pattern they take |
|---|---|
| Pure Land: Jodo-shu, Jodo Shinshu, Ji, Yuzu Nembutsu | 浄土宗・浄土真宗系 |
| Tendai; Shingon | 天台宗・真言宗系, shared |
| Zen | 禅宗（臨済宗・曹洞宗）系 |
| Nichiren schools | 日蓮宗系 |
| 2000 Buddhist, no school | the five Buddhist answers summed |
| 2905 Soka Gakkai | 創価学会 |
| 2817 Rissho Kosei-kai | 立正佼成会 |
| Shinto and sect Shinto | 神道系 |
| every Christian code, and Unification | キリスト教系 |
| other new religions, Islam, combinations, ancestor veneration, unclassifiable | none: flat on each prefecture's religious population |

**Every NHK answer passes the noise test**, a chi-square over 47 prefectures on the measured share
times the achieved sample (600, or §11an's seven known values): p runs from 1.8e-211 (Tendai and
Shingon) to 1.4e-5 (Soka Gakkai), so nothing went flat. A measured 0.0 is read as 0.15, the middle
of what the chart cannot resolve; left at zero, Fukushima and Saga would have no Christians.

**Blocks.** JGSS-2015 against NHK 1996 ranks the six blocks at +0.943. Drawn religious share of
answers: Hokkaido/Tohoku 22.1%, Kanto 21.1%, Chubu 29.3%, Kinki 27.2%, Chugoku/Shikoku 30.8%,
Kyushu 31.5%.

**What it draws**, as shares of everyone:

| node | national | highest | lowest |
|---|---:|---|---|
| unaffiliated | 71.68 | Okinawa 90.2, Chiba 80.6, Fukushima 80.3 | Fukui 48.1, Hiroshima 54.3 |
| buddhism.mahayana (no school) | 7.01 | Fukui 15.1, Toyama 12.5, Hiroshima 12.2 | Okinawa 0.38, Kochi 3.79 |
| buddhism.mahayana.pureland | 6.80 | Fukui 22.2, Toyama 21.2, Ishikawa 18.5 | Okinawa 0.09, Tochigi 1.92 |
| eastasiannew.japanese | 3.08 | Hiroshima 4.4, Tokyo 3.9, Osaka 3.8 | Miyagi 1.69, Ishikawa 1.78 |
| buddhism.mahayana.zen | 1.84 | Iwate 6.0, Shizuoka 4.4, Miyagi 4.2 | Kochi 0.06, Okinawa 0.07 |
| buddhism.mahayana.shingon | 1.48 | Tokushima 6.9, Okayama 5.6, Kagawa 4.7 | Akita 0.05, Yamanashi 0.30 |
| buddhism.mahayana.nichiren | 1.43 | Yamanashi 4.5, Shizuoka 3.8, Okayama 2.8 | Iwate 0.08, Fukushima 0.08 |
| shinto | 0.85 | Kochi 3.6, Miyazaki 2.9, Kagoshima 1.9 | Iwate 0.11, Okinawa 0.11 |
| every Christian node | 1.15 | Nagasaki 3.84, Tokyo 2.73, Okinawa 2.01 | Fukushima and Saga, about 0.1 |

**Okinawa is 90.2% unaffiliated**, because 7.8% there named a religion in 1996 against 31.2%
nationally. The Global Flourishing Study independently puts Okinawa lowest too (15.6% against 38.1%),
so the direction is not the 1996 card's alone. Carried over as a log-odds shift, GFS would put
Okinawa's religious share near 9% of answers where this build has about 6.4%. GFS offers
`Primal, Animist, or Folk religion` and 15 respondents in all of Japan chose it; whether Okinawa's
low figure is people whose ancestral practice no answer fits is not established.

**Taxonomy.** Five children of `buddhism.mahayana`: `pureland`, `shingon`, `tendai`, `zen`,
`nichiren`. Pure Land is one node because NHK asked Jodo-shu and Jodo Shinshu as one answer, and two
nodes would draw Jodo-shu (1.2%) with Jodo Shinshu's (5.5%) Hokuriku map. Tendai and Shingon are two
nodes on one map, which Shingon mostly sets. Buddhism has no `ROOT_BAND`, so the five take
Mahayana's colour at the top level and separate when Buddhism is opened, the position Anita accepted
for Mauritius's Hindu rows on 2026-09-07. **Nichiren is pinned yellow** (`index.html` PIN, [52, 90,
55], Anita after seeing the map): the allocator's #1228ba measured contrast 1.8 against the map and
sat 23 degrees from unspecified Mahayana's #1268ba. The Zen node now also holds New Zealand's `Zen
Buddhism` (1,401). Poland's seven Zen, Chan and Seon unions were moved and put back the same day:
2,166 nationally, but only 935 placed at gmina level, which draws no dot.

**Placement.** COD-AB ADM1 2019, joined on pcode with the Japanese name asserted on all 47. 7,392
hexes (1.11M people) sit outside the simplified coastline and are snapped to the nearest
prefecture, all but 2 people within 2 km. Kontur against 人口推計 2024 per prefecture runs 0.951
(Okinawa) to 1.137 (Tottori); with the labels shuffled, 35 of 47 fall outside 1.5x.
`sources/jp.py` no longer writes the hex layer: its geometry() wrote the one-unit version to the
same path.

## 8. The second search, and the calls it leaves (2026-09-14)

### Christians: the Agency roll or NHK 1996

Anita asked whether the roll is that unrealistic, since it is official and is what everyone cites
for Tokyo. Every Christian node scaled to JGSS's national 1.15%, before fitting (a scratch
comparison; `python sources/jp_alloc.py --christians roll` builds the roll version for real):

| allocator | Tokyo | Nagasaki | Kanagawa | Saitama | Chiba | Aichi |
|---|---:|---:|---:|---:|---:|---:|
| NHK 1996 (built) | 2.54 | 3.84 | 1.71 | 1.25 | 0.77 | 1.25 |
| the roll as published | 4.71 | 3.59 | 2.46 | 0.37 | 0.44 | 0.47 |
| the roll with Tokyo at the rest of Japan's per-body rate | 1.17 | 5.97 | 4.09 | 0.62 | 0.73 | 0.78 |

Believers per Christian body: Tokyo 886, Kanagawa 578, Nagasaki 300, Fukuoka 151, median 76.
Nagasaki's is Catholic and real on every source (§4). Tokyo's and Kanagawa's are national bodies
filing their whole membership at their address, and **correcting Tokyo alone moves the capture to
Kanagawa**. The roll's national total is official; its prefecture split records where bodies are
registered, which is why it draws Saitama and Chiba at a third to a half of NHK's share. NHK 1996
against the roll across 47 prefectures is +0.61, so the two agree outside the head-office effect.
GFS cannot referee: its Christian share has no prefecture pattern that replicates (split-half
+0.03).

**A per-church count sides with NHK on the head-office question.** The United Church of Christ in
Japan publishes members and Sunday attendance for its 17 districts, counted at each church
(2024-03-31, 154,787 members). With West Tokyo merged into the Tokyo district (Tokyo plus Chiba),
each district over its source's national rate:

| | Tokyo + Chiba | Kanagawa | Saitama and four more (Kanto district) | Kyushu | Okinawa |
|---|---:|---:|---:|---:|---:|
| Kyodan members | 1.68 | 1.11 | 0.51 | 0.62 | 0.43 |
| Kyodan Sunday attendance | 1.58 | 1.25 | 0.63 | 0.55 | 0.72 |
| NHK 1996 | 1.74 | 1.49 | 0.88 | 0.91 | 1.70 |
| built map | 1.87 | 1.60 | 0.93 | 0.90 | 1.75 |
| Agency roll | 2.96 | 2.14 | 0.29 | 0.78 | 1.16 |

Over all 16 districts one mainline Protestant church tracks nobody well (Spearman +0.12 to +0.31;
it is strong in Kyoto, Hyogo and Shikoku and weak where Christians are Catholic), so it is a witness
on one question only: counted at the church, Tokyo is about 1.6x the national rate, which is NHK's
figure and half the roll's. **Built on NHK 1996. Anita's call.**

### ISSP, and what Anita would have to do

From the research pass of 2026-09-14; the fetch tool could not open www.gesis.org itself (403), so
the GESIS terms below are from the CESSDA guide, an R package documenting the form, and search
snippets of the usage regulations.

- **What it would give.** ISSP Japan is fielded by NHK, about 1,100 to 1,500 a year, with a
  9-region variable from the Basic Resident Register (`J_REG` in 2009, `JP_REG` in 2011) and five
  religion codes in every Japan file checked, including the 1998, 2008 and 2018 religion modules:
  Buddhism, Shinto, Christianity, other, none. No schools, no Soka Gakkai, no prefectures.
- **What stands in the way.** GESIS access category A is *"released for academic research and
  teaching"*; the usage regulations say individuals outside academic research and teaching can apply
  in writing.
- **The steps.** (1) Email `isspservice@gesis.org`: an unaffiliated individual making a
  non-commercial website, publishing only regional aggregates, never redistributing microdata, and
  the ZA numbers wanted. (2) Register at `login.gesis.org`: first and last name, country, discipline
  (`Others`), email, password, and the usage-regulations checkbox; there is no institution field.
  (3) Download each study while logged in; the purpose menu includes *for non-scientific purposes*,
  which is the honest choice.
- **Worth it?** Recency at 9 regions for broad categories, which GFS already gives more cheaply for
  the religious share (below). Low priority.

WVS: free for non-commercial publication through a download form; Japan waves of about 1,000 to
1,400; 2019 has prefecture codes (n 1,353) and no Shinto code; 2005 is a quota panel. Pew's 2023 East
Asia dataset is released behind a Pew account (Japan n 1,742 by phone, region variable not
confirmed). Asian Barometer is an academic application with an affiliation field. NHK's own 2018
ISSP report has no regional breakdown.

### The Global Flourishing Study: open, current, prefecture-coded, and a check rather than an input

`sources/jp_gfs.py`. Wave 1 Japan, 20,543 respondents, December 2022 to June 2023, CC BY 4.0 on OSF
since 2026-04-08 with no account; `REGION1` is the 47 prefectures, `REL2` *"What is your current
religion?"* with Buddhism and Shinto listed; an opt-in web panel, weighted. Its list reads Japan as
38.1% religious and 33.2% Buddhist against JGSS's 25.6% of answers, so only its pattern could be
used (§3.1a).

| 47 prefectures | split-half (bar 0.29) | against NHK 1996 |
|---|---:|---:|
| names any religion | +0.686 | +0.670 |
| Buddhist | +0.698 | +0.718 |
| Shinto | +0.230, fails | +0.141 |
| Christian | +0.033, fails | +0.015 |

**This is what licenses using 1996 at all**: the share naming a religion and the Buddhist share
still fall across prefectures the way they did twenty-seven years earlier, in a survey that shares
nothing with NHK's.

**Why it is not the block level instead of JGSS-2015.** At the six blocks it ranks with JGSS-2015
(+0.886), JGSS-2000/01 (+0.829) and NHK 1996 (+0.943), but it puts Hokkaido/Tohoku at 1.10x its
national figure where JGSS-2015 has 0.86, JGSS-2000/01 1.00 and NHK 0.90. Three probability samples
agreeing against one opt-in panel decides it the spec's usual way. Using GFS for prefecture
religiosity outright would also bring its thinnest cells in: Fukui 78 respondents, Shimane and
Kochi 90.

### Everything else the second search turned up

| find | what | use |
|---|---|---|
| Kimura 2003, JGSS Research Series 2, fig. 3 | JGSS-2000/01 believe / family religion / none by six blocks and city size; bars without values, measured to about a point | a second JGSS wave at blocks, in `jp_gfs.py`'s comparison |
| United Church of Christ in Japan (Nihon Kirisuto Kyodan), 2024-03-31 | 17 districts: churches, 154,787 members, communicants, Sunday attendance, counted where each church is | one denomination; compared in the Christians section above, where it sides with NHK on Tokyo |
| JMR Data Book 2023 | Protestant worship attendance as a share of population by prefecture, from yearbooks | paid, 1,800 yen; not seen |
| church-info.jp | about 12,000 Protestant churches by municipality | buildings |
| Komeito PR votes by municipality (MIC, 2025 upper house) | Soka Gakkai proxy | still refused on the proxy rule; not fetched |
| Survey on Time Use and Leisure Activities | no religion item in any year | closed |
| JGSS Research Series vols 1-21 | only vol. 2 has a regional religion table | closed |
| SSP-2015, SSM 2015 | no religion item | closed |
| ISM Asia-Pacific Values 2010, Dai-ichi Life 2007, Ishii's 2011 compilation, Niwano Peace Foundation 2019 | no regional tables (Niwano by city size only) | closed |
| Global East Survey of Religion and Spirituality, Japan 2024 | national postal survey, N 3,947 | no regional release |
