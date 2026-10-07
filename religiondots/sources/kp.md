# North Korea (`kp`)

Drawn 2026-10-03 by `fafd1067-kp`: Pew Research Center's 2020 national mix (the World Religion
Database's figure) at COD-AB's 11 provinces, on the 2008 census, placed on Kontur. Every row
`modelled`. Built on Anita's rulings of 2026-09-15 (a §14 case files an ask and carries on) and
2026-09-16 (a compiler's figure where nothing asks). Ask 052 holds whether North Korea, and its
Christians in particular, are drawn at all. Code: `sources/kp_geo.py`, `sources/kp_grid.py`,
`sources/kp.py`, `taxonomy/kp2020.py`, `countries/kp.py`; node `indigenous.korean` in
`taxonomy/branches.py`. Pew's "other" split onto Cheondogyo, Korean folk religion and Chinese folk
religion the same day by `fafd1067-kp2`, on Anita's ruling on ask 052 (§6, §7).

## 1. Nothing asks

- **Census.** The 2008 form (national report Annex 2, CPF-2) and the 1993 form have no religion item;
  nor does MICS 2009 (`sources.md` §11, North Korea, closed at the questionnaire). MICS 2017's
  household form is still unread (every copy 403 or Cloudflare); a state-run interview is not free
  self-identification in any case.
- **No independent survey of residents exists.** The only surveys that ask North Koreans about
  religion are of defectors in South Korea (below), which is a different population.

## 2. The level: Pew 2020, which is the World Religion Database

`data/raw/estimates/pew.zip`, unrounded counts, North Korea 2020: 26,136,312 people; unaffiliated
19,044,889 (72.868%), other religions 6,591,569 (25.220%), Buddhists 396,455 (1.517%), Christians
100,372 (0.384%), Muslims 2,614, Hindus 414, Jews 0. 2010 is the same mix to two decimals.

**Checked what it rests on** (the Syria lesson): Pew's Appendix A
(`PR_2025.06.09_global-religious-change_appendix-a.pdf`, read 2026-10-03) sources North Korea's 2010
and 2020 composition to the **World Religion Database**, population UN WPP 2024, switching "Data
unavailable"; its first page names North Korea as one of "about two dozen countries and territories"
where "the only source available is the World Religion Database". So Pew's row is a compiler's
ascription, with no survey behind it.

What the WRD's 25.2% "other" is: ARDA's free view of the WRD
(`thearda.com/world-religion/national-profiles?u=123c`, 2025, read 2026-10-03): agnostics 57.29%,
atheists 15.58%, **new religionists 12.88%, ethnic religionists 12.28%**, Buddhists (Mahayana)
1.52%, Christians 0.38% (Independents 0.35, Protestants 0.03, Catholics 0.01), Chinese folk 0.06%,
Muslims 0.01%. New + ethnic + Chinese folk = 25.22 = Pew's other. In Korea "new religionists" is
Cheondogyo and "ethnic religionists" is Korean shamanism (musok). These three are what the map draws
in place of Pew's "other" (§6).

Other figures, all national, none a count of residents (State Department IRF report 2022, read
2026-10-03 from `2021-2025.state.gov/wp-content/uploads/2023/05/441219-KOREA-DEM-REP-2022-...pdf`):

| who | figure | kind |
|---|---|---|
| DPRK government to the UN Human Rights Committee, 2002 | 12,000 Protestants, 10,000 Buddhists, 800 Catholics, 15,000 Cheondoists | the state; an interested party |
| Center for the Study of Global Christianity (= the WRD) | 100,000 Christians | compiler |
| Open Doors USA | 400,000 Christians | advocacy; an interested party |
| "UN estimates" | 200,000-400,000 Christians | unspecified |
| Religious Characteristics of States, 2015 | 70.9% atheist, 11% Buddhist, 1.7% other, 16.5% unknown | compiler |
| NKDB defector surveys (15,169 cumulative responses to 2024) | 99.6% say no religious freedom; among defectors who practise, most are Protestant | defectors in the South, not residents; NKDB is an advocacy NGO |
| Korea Future | shamanism "the most widespread religious practice ... with practitioners in every province" | advocacy; no figure |

## 3. Nothing places anyone

Asked, of every source above: does anything give religion by province, from anyone other than an
interested party? **No.** The census never asked; the WRD says it carries provincial figures only
from censuses and surveys, and there are none; the IRF reports, KINU and NKDB white papers name
provinces only for individual persecution cases (an execution in North Hamgyong in 2011, South
Hwanghae in 2015, Pyongsong in South Pyongan in 2018), which are incidents, not counts, and come from
interested parties. NKDB's database records defectors' province of origin, and defectors come
overwhelmingly from the northern border provinces, so even a tabulation of it (not found published)
would measure who escapes, not who believes. Not checked: whether NKDB's 2024 white paper prints
religious-activity experience by province of origin (a lead only; it would still be defectors, and
§14 applies to it directly).

## 4. Population base: the 2008 census, Table 2, re-cut

- **Source.** *DPR Korea 2008 Population Census, National Report*, Central Bureau of Statistics,
  Pyongyang 2009, carried out with UNFPA support
  (`unstats.un.org/unsd/demographic/sources/census/wphc/North_Korea/Final%20national%20census%20report.pdf`,
  in `data/raw/kp/`). Table 2 (pp.18-22) gives 209 cities, districts and counties in 10 provinces,
  23,349,859 people; every printed province total equals its rows (asserted).
- **The 702,372.** Table 1's national total is 24,052,231, footnoted "Includes all individuals living
  in private households, institutional living quarters and military camps"; Table 2 carries no such
  note and no province holds the difference. Commonly read as the military; the report does not say
  so in words. Not drawn; it is most of the `gap` (2.92%).
- **The re-cut to COD-AB's 11 units** (`kp_geo.py`, pinned and asserted): Nampo (KP11) is Nampho City
  plus Kangso, Onchon, Ryonggang, Taean and Chollima out of South Phyongan (983,660); North Hwanghae
  takes Kangnam, Junghwa and Sangwon from Pyongyang (239,477, the 2010 transfer).
- **COD-PS was not used, because it is wrong by two counties.** OCHA's `cod-ps-global` admin 1 for
  PRK (WFP, reference year 2008) is this same table re-cut the same way, but at admin 2 Samchon county
  (South Hwanghae, 86,042) is missing and Sindo county (11,810) is listed on its own *and* added into
  Ryongchon (135,634 in Table 2, 147,444 in COD-PS). So its South Hwanghae is 86,042 short, North
  Pyongan 11,810 long, total 23,275,627. Found by matching every COD-PS admin 2 value to a Table 2 row.
- **Vintage.** 2008, unscaled: 23.35 million in provinces against Pew's 26.1 million for 2020. The note
  says the dots stand where people lived in 2008. No later census was found on UNSD's census page;
  not checked further.

## 5. Geography and placement

- **COD-AB** `cod-ab-prk` v01 (valid 2019-06-24), `prk_admin1.geojson`, KP01-KP11. Join on folded
  English names (`Phyongan` = `Pyongan`).
- **Kontur KP 20231101**: 45,338 hexes, 26.33 million. 1,025 border hexes snapped within 2 km, 106
  dropped (12,998 people, 0.05%). Ratio to the census 1.127; rank witness +0.982, no shuffle in
  20,000 reaches it. Per province 0.80 (Nampo) to 1.09 (Jagang). Seat check: no hole.
- **Seventy cap blocks, every one at a named town** (all within 1.7 km of a GeoNames place, mostly the
  county seat, `-up`). Judged per county against Table 2's urban population for that county, at
  Kontur's national ratio (`kontur_cap.csv`, the why column carries each figure): 64 `real`, 6
  `capped`, the blocks of Sonchon (2.8x), Sinchon (2.7x) and Koksan (2.5x). **The bar is 2x, on
  purpose**: capping lowers a block to its 3 km ring's median, which erases the town, so it misplaces
  about the census urban population; keeping it misplaces the excess above that. The two are equal at
  2x. At 1.5x it would also have capped Paechon, Chongdan, Kwaksan and Cholwon, losing four towns to
  correct a smaller overstatement. Samsu's block is Hyesan's second (0.3 km from the city; COD-AB's
  county line cuts it) and is counted with Hyesan.

## 6. As drawn

23,347,154 people: no religion 17,014,470; Cheondogyo (`eastasiannew.korean.cheondogyo`, the WRD's
new religionists) 3,007,458; Korean folk religion (`indigenous.korean`, its ethnic religionists)
2,867,358; Chinese folk religion 14,010; Buddhist 354,188; Christian 89,670. The same
72.88 / 12.88 / 12.28 / 0.06 / 1.52 / 0.38 in every province: Pew's 25.22% "other" is split in the
WRD's proportions (`WRD_OTHER` in `sources/kp.py`, asserted to sum to Pew's share). Not drawn:
702,372 in no province and 2,705 (Pew's Muslims and Hindus at 0.0116%); `gap_share` 0.029314 of
24,052,231. 23,345 dots at 1:1,000, 2,331 at 1:10,000, no rings.

From the first build until Anita's ruling on ask 052 (both 2026-10-03), the 25.22% was one residual
node, `other.kp` (5,888,827 people), now retired.

## 7. Calls someone might reverse

- **Pew's "other" split, on Anita's ruling (ask 052, "yeah i think we should split").** The first
  build kept it on one residual, `other.kp`, because both halves are WRD ascription and the
  Cheondogyo half draws about 3.0 million, 45 times South Korea's counted 65,964 and 200 times the
  state's own 15,000. Both points stand and the note gives them; the split is drawn anyway.
  Cheondogyo goes on South Korea's `eastasiannew.korean.cheondogyo` (the WRD's new religionists may
  include a few Jeungsanists or Daejonggyo; nothing splits them). Shamanism goes on a new
  `indigenous.korean`, "Korean folk religion", under `indigenous` on the Philippine and Myanmar
  pattern, not on `indigenous.northeurasian` (Mongolia's shamanists), whose label is Russia's.
  Used by `kp` only. The 0.06% Chinese folk goes to `chinesefolk` (about 14 dots).
- **Unaffiliated drawn as such.** The WRD's agnostics and atheists, in a state that punishes practice.
  The note says it is an estimate of what people would say, not a measurement.
- **The military 702,372 left out** rather than spread pro rata: they belong to no province in the
  census, and spreading them would put them in proportion to civilians, which they are not.
- **Buddhists on bare `buddhism`**, not Mahayana (Taiwan ruling, ask 025; South Korea's are bare too).

## 8. What would make it better

For the religion figures, nothing in view: a census or survey of residents that asks religion would,
and none is planned that anyone has published. For placement only (it moves no religion figure, since
the mix is national): Table 2's 209 county totals could calibrate Kontur county by county, as Iran's
was (`playbooks/geography.md`), instead of the cap-block review; COD-AB admin 2 has 179 units, so the
join needs the 2010 transfers and Pyongyang's districts handled. MICS 2017's household form (`HC1A` in the MICS6 template) is worth one more
try for whether the question was even fielded.

## 9. Review, 2026-10-03 (`fafd1067-rev12`, light pass on the other split)

- Note wording: "new religions, which in Korea means Cheondogyo" changed to "which in North Korea
  means chiefly Cheondogyo". In South Korea the WRD's new religionists include Won Buddhism and the
  Jeungsanist bodies (kr2015 counts Daesun Jinrihoe separately), so the general claim was wrong;
  `kp2020.py`'s REVIEW already concedes a few Jeungsanists. `tiles.py --refresh-meta` run.
- Palette, the first country drawing `eastasiannew` and `indigenous` together, in near-equal halves
  (3,007 and 2,867 dots). `check_palette.py` and `check_overview.py` both pass kp (blue #5172f6
  against violet #7d4bf1, CIE76 dE 29.1 over the bar of 25), but CIEDE2000 puts the pair at 13.4,
  below Catholic against Orthodox (20.8). On screen they read as one periwinkle wash at the country
  view and separate cleanly at city zoom (Pyongyang, z10.5). Not changed; recorded in case Anita
  sees the wash and wants one of the two moved.
- Mapping, gap, numbers: checked against `kp.csv` and the note; nothing else found.

## 10. Top text before the 75-word cut, 2026-10-03 (`fafd1067-top75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/kp.py`; `note_public` was not changed.

- `how`: no source asks; a compiler's national estimate, one mix everywhere
- `grain`: provinces, 2.1 million people on average
- `gap`: the 702,372 people the 2008 census counts nationally but in no province (its national tables include military camps), and about 2,700 Muslims and Hindus in Pew's estimate; 2.93% in all
