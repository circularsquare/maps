# Nauru — built 2026-09-11 from the 2021 census, table G-7

**State: DRAWN.** Nineteen categories, 11,680 people, one unit, eight dots and eleven rings.
Built by session `f95259a4-nr2`, resuming the checkpoint-A scouting record `f95259a4-pg` left
on 2026-09-09.

    sources/nr.py         2021 census workbook + Kontur -> data/normalized/nr.csv, nr_hexes.gpkg
    taxonomy/nr2021.py    the mapping, one EXCLUDED cell and ten REVIEW notes
    countries.py "nr"     the entry
    sources.md §9cx       the part that generalises

---

## 1. The handoff was wrong about one thing, and it is the thing that unblocked the country

`handoff/nr.md` and this file's previous version both recorded **"no WP File Download plugin
is present"** on `stats.gov.nr`, which sent the next step towards `/documents/` page bodies
and then towards the `.Stat` instance. **The plugin is present.** The page source carries it
twice:

```
wpfdajaxurl = "https://stats.gov.nr/wp-admin/admin-ajax.php?juwpfisadmin=false&action=wpfd&"
```

So Fiji's route (§9bd) and PNG's (§11ab) apply here unchanged, and `[[reference_wpfd_sweep]]`
is exactly right about all three of its traps:

```
/wp-admin/admin-ajax.php?juwpfisadmin=false&action=wpfd&task=files.getFiles&id=0&page=<n>
```

**`id=0` returns the whole library: 131 files in 37 categories, ten to a page, fourteen
pages.** No key, no login, no rate limit. The previous session's `wp/v2/media` sweep was
correct and correctly reported (48 items, one non-image file) and simply does not see these
files, which is the corollary trap the reference note already names.

**The generalisable lesson is narrower than "the note was wrong".** A WP File Download
install is invisible in `wp-json/` — the plugin registers no REST namespace, so the namespace
list shows only `wp/v2`, `divi/v1`, `oembed/1.0` and the rest, and looks like a site with no
document plugin at all. The tell is in the rendered page, not in the API index: grep the HTML
for `action=wpfd` or `wpfdajaxurl`. That is one `curl` and one `grep`.

## 2. The source

**Nauru Bureau of Statistics, 2021 Population and Housing Census, Tables Vol 1, sheet G-7**,
*Total population by religious affiliation and sex*.

```
https://stats.gov.nr/download/49/2021/182/population-housing-census-2021-tables-vol1.xlsx
```

75 KB, 38 sheets, of which G-7 is religion. Also read and worth keeping the ids for:

```
  49 /  359   Nauru 2021 PHC Analytical Report          10.2 MB  142 pp   table 25, 2011 vs 2021
  49 /  358   Person Tables 1-36                         618 KB   78 pp   tables 18 and 19
  60 /  497   2021 Census Questionnaire                  626 KB   56 pp   question 307, page 10
  43 /  149   Nauru 2011 Census Report FINAL             7.0 MB  217 pp   table 23, 2002 vs 2011
```

The download route is `/download/<catid>/<catslug>/<fileid>/<anything>.<ext>`; the trailing
slug is free, the rest is not.

**Nauru is not in UNSD table 28.** `tools/oracle.py --list` carries 117 countries on
2026-09-11 and Nauru is not one of them, so unlike §9bf's nine there was no Yearbook fallback
and no independent cross-check on the office. The checks below are all internal to the
office's own publications, and there are enough of them that this is not a weakness.

## 3. The table, and the partition

Nineteen categories, summing to 11,680, which is the census's own total population, with a
difference of **zero**.

```
    Nauruan Congregational        4,001  34.26%        Fishers of Men Church       57   0.49%
    Catholic                      3,959  33.90%        Christ Embassy              48   0.41%
    Assemblies of God (AOG)       1,365  11.69%        Brethren Church             47   0.40%
    Pacific Light House             706   6.04%        Methodist Church            18   0.15%
    Nauru Independent               410   3.51%        Fundamental Christian       15   0.13%
    Shalosh Pentecostal Church      186   1.59%        Hinduism                     6   0.05%
    Baptist                         175   1.50%
    Seven Day Adventist             168   1.44%        No Religion                157   1.34%
    Protestant                      126   1.08%        Other religion              98   0.84%
    FOM Pentecostal Church           81   0.69%        Do not wish to answer       57   0.49%
```

`sources/nr.py` pins the total and the category count, so a re-publication that revises a
figure fails the build rather than moving the map quietly.

## 4. The questionnaire is what makes the table readable, and it is not optional here

**Question 307 offers ten pre-coded answers and the table prints nineteen.** From the 2021
questionnaire, page 10, the `religion` single-select:

```
  00 No Religion                 05 Pacific Light House
  01 Nauruan Congregational      06 Seven Day Adventist
  02 Catholic                    07 Baptist
  03 Assemblies of God (AOG)     08 Do not wish to answer
  04 Nauru Independent           97 Other religion   -> 307_oth, free text, reached only from 97
```

So **nine of G-7's nineteen rows are the office's back-coding of the `307_oth` write-ins**:
Protestant, Shalosh Pentecostal, Fishers of Men, Brethren, FOM Pentecostal, Christ Embassy,
Hinduism, Fundamental Christian and Methodist. Two consequences that change how the table is
mapped, both of them written into `taxonomy/nr2021.py`:

* **Those nine names are respondents' own words**, not an office classification. `FOM
  Pentecostal Church` is somebody's abbreviation, which is why it sits beside a separate
  `Fishers of Men Church` row for what is almost certainly the same body.
* **`Other religion` at 98 people is a coding tail, not a sampling tail.** It is what survived
  nine rounds of classification, which is why it is 0.84% where the same cell is several
  percent in most censuses of this size.

This is `[[reference_census_questionnaire]]`'s case exactly: nothing in the table itself says
any of this, and read without the form it looks like a nineteen-way pre-coded list.

## 5. Three checks, and all three close

**The 2021 table against the 2011 table.** The Analytical Report's table 25 prints both, and
the office's own rollup of the 2021 write-in rows into `Other religion` gives 485, which is
`186 + 57 + 81 + 48 + 15 + 98` from G-7 exactly. Its `Not stated` of 57 is G-7's `Do not wish
to answer`, which is what settles that cell as a refusal rather than a religion.

**The citizen table against the whole-population table.** Person Tables table 19 covers the
11,215 Nauruan citizens and dual citizens. Subtracting it from G-7 category by category gives
465 non-citizens, which is 11,680 - 11,215 to the person, with no negative cell:

```
  Catholic      +162      Nauruan Congregational  +112      AOG  +64      SDA  +22
  Do not wish    +20      Methodist                +18      the five write-in bodies  +17
  No Religion    +15      Brethren                 +13      Nauru Independent  +7
  Baptist         +7      Hinduism                  +6      Pacific Light House  +2
  Protestant       0
```

Which also confirms the Analytical Report's sentence that **all 6 Hindus and all 18 Methodists
were non-citizens**, and that the Brethren and the Adventists lean foreign.

**Kontur against the census.** 44 populated hexes, 12,374 people against 11,680 counted, a
ratio of 1.06 for a grid two years newer than the census.

## 6. The district table exists, is real, and is deliberately not drawn

**Person Tables table 19, `Nauruan Population (citizen/dual) by Sex by District by Religion`,
pages 37-38.** Fifteen districts, twelve non-empty categories. It is not used, for two
reasons, and neither is that it is bad:

1. **Twelve dots cannot express fifteen districts.** At 1 dot = 1,000 people the country draws
   eight dots and eleven rings; §9bf's permission is precisely that placement inside a country
   this size asserts nothing.
2. **It would cost the other 465 people and seven of the categories.** It covers citizens
   only, and folds Shalosh, Fishers of Men, FOM, Christ Embassy and Fundamental Christian back
   into `Other religion` (468 against G-7's 98). Trading 19 categories over 11,680 people for
   12 over 11,215 is the wrong direction.

**But it should be recorded, because the mission partition of a 21 km² island is visible in
it**, and it is the Gilberts' pattern (§9bi, `ki2015.py`) at a thousandth of the scale. Shares
of each district's citizen population:

```
                     Congregational   Catholic        the district's own oddity
    4-Buada              58.3%          10.1%
    2-Boe                51.7%          14.6%
   14-Meneng             42.6%          21.7%         111 of the 403 Nauru Independent
   11-Anabar             43.1%          34.4%
    6-Nibok              39.8%          34.8%
   15-Location           27.5%          36.6%         23 of the 34 Brethren
   10-Anetan             25.4%          45.8%         71 of the 168 Baptists
    9-Ewa                26.5%          59.3%
    1-Yaren              14.1%          58.0%
    8-Baitsi             14.0%          79.5%
    7-Uaboe               4.7%          41.5%         77 of the 126 Protestants
```

**Buada is 58.3% Congregational and Baitsi, four kilometres away, is 79.5% Catholic.** Uaboe
is 4.7% Congregational. And the whole national `Protestant` cell is essentially one district:
77 of 126 people are in Uaboe, where the Congregational church is weakest anywhere on the
island, which is circumstantial support for the US State Department's line that the Nauru
Congregational Church "includes the Nauru Protestant Church" being about a real second body
rather than a loose phrase.

## 7. The mapping, and what is uncertain in it

`taxonomy/nr2021.py` carries ten REVIEW notes; the three worth naming here:

**`Nauruan Congregational` -> a new node**, `christianity.reformed.congregational.ncc`, the
sixth of the Pacific Congregational set. The evidence that puts it beside `.cccs`, `.cicc`,
`.ekt`, `.kpc` and `.niue` rather than at the parent is the **Council for World Mission's own
member list**, which carries the Nauru Congregational Church (NCC) in its Pacific region
alongside CCCS, CICC, EKT and the Kiribati Uniting Church; CWM is the LMS's successor body.
`.kpc`'s note called itself "the fifth and last of the Pacific Congregational set" and it was
the fifth.

**`Nauru Independent` -> `christianity.nondenominational` is the weakest call in the file.**
410 people, 3.5%, one of the five churches the government has registered, and **nothing
published says what family it belongs to.** The two characterisations that exist disagree:
Operation World calls it the largest evangelical group in the country; the Gale encyclopedia's
Nauru entry says a breakaway Protestant church was formed in 1977 "under the American
Pentecostal church" without naming it. It is filed the way Australia and New Zealand file
`Independent Evangelical Churches`. A founding history would move it, in either direction.

**`Pacific Light House` -> `christianity.pentecostal.charismatic`.** 706 people, 6.0%, the
fourth-largest body, and no cell at all in 2011. The only published description found is
"Pacific Light House Church (Born Again Christian Church)", opened in Boe on 5 September 2019.
It is not on the government's registered list, which fits a body too young for the 750-member
rule and does not fit it being a branch of the separately-printed Assemblies of God.

## 8. What changed between 2011 and 2021, which is the country's own story

```
                            2011     2021
    Nauruan Congregational  3,552    4,001     35.7% -> 34.3%
    Roman Catholic          3,278    3,959     33.0% -> 33.9%
    Assemblies of God       1,291    1,365     13.0% -> 11.7%
    Nauru Independent         945      410      9.5% ->  3.5%
    Pacific Light House         -      706         -  ->  6.0%
    Jehovah's Witness          89        -      0.9% ->    -
    Not stated                109       57
    Total                   9,945   11,680
```

Two movements dominate and everything else is within a point. The Nauru Independent Church
loses 535 people and Pacific Light House arrives with 706. **Nothing published connects them**
and the census asks nobody where they came from, so this file does not claim it; the
arithmetic is recorded because the next person will notice it. The Analytical Report's own
comment on the third one is that the Jehovah's Witnesses "seem to have disappeared by 2021";
Person Tables table 19 keeps a Jehovah's Witness column and it is empty in all fifteen
districts.

## 9. Ethics, checked and not escalated

Nauru publishes this itself, in full, at 6 people; the map draws no geography inside the
country; and the US State Department's 2023 religious freedom report records **no societal
actions affecting religious freedom** and no discrimination in the registration process. The
registration rule is restrictive for new groups (750 members, land, a building, resident
Nauruan clergy, which is why the Latter-day Saints remain unregistered) but that is a burden
on institutions, not a risk to people whose census answer is drawn as one dot in twelve on a
national partition. No §14 ask filed.

## 10. Retired, so nobody re-derives them

* **`nauru.popgis.spc.int` is open and has no religion.** 168 datasets, all 2011 Census, 4,437
  indicators, and the census question ids `p1`-`p43` contain no religion question. The
  previous session's note stands and its parameter trap with it: `GC_init.php` needs `lang`
  and NOT `obs`; `GC_listIndics.php` needs `obs=main`; the wrong one returns a 426-byte
  *Unavailable service* page that looks exactly like the Solomons' disabled endpoint (§9bh)
  and is not one.
* **`wp/v2/media` on `stats.gov.nr`**: 48 items, page 2 is a 400, one non-image file. Correct
  and useless. See §1.
* **The regional SPC `.Stat` agency has no religion dataflow for any Pacific country** (§11ab,
  checked 2026-09-09). `stats-nr.pacificdata.org`, `naurufinance.info` and `sdd.spc.int` were
  never needed and are still unprobed.
* **The widely-quoted CIA World Factbook shares** (Protestant 60.4% etc., "2011 est.") are
  superseded and were never traceable to a Nauruan release. The real 2011 figures are in §8.

---

## 11. Review, 2026-09-11, session `f95259a4-nrrev`

Second pass under `.claude/commands/rd-review.md`. `check_md.py`, `built_countries.py --check`,
`check_rollup.py nr`, `check_mapping.py nr` and `gap_share.py --check` all clean. Nothing here
was rebuilt and nothing in `countries.py` was touched, because another session was writing that
file at the time. **Three of the four findings are in `note_public` and all three contradict
this file or `sources.md` §9cx, so the note drifted from the record rather than the record being
wrong.**

**`note_public` says the questionnaire listed "six named churches". It listed seven.** §4 above
has the codes: 01 Nauruan Congregational, 02 Catholic, 03 AOG, 04 Nauru Independent, 05 Pacific
Light House, 06 Seven Day Adventist, 07 Baptist. The sentence's own arithmetic gives it away,
since no religion plus six churches plus the refusal box plus other religion is nine and the
sentence claims ten. **`taxonomy/nr2021.py`'s docstring carries the same slip a different way**:
it says the ten pre-coded answers are *"the first eight above plus `Do not wish to answer` and
`Other religion`"*, but the first eight rows of that count-sorted table include `Shalosh
Pentecostal Church`, which the very next sentence correctly calls a write-in, and exclude
`No Religion`, which is code 00. It should read the first nine other than Shalosh.

**`note_public` dates the Catholic mission to 1899, and 1899 is the Protestant arrival.** This
file's §7 and `sources.md` §9cx both give 1899 to Philip Delaporte of the American Board, and
neither gives the Catholic mission a date at all. The usual date for the Catholic mission is
**1902**, when Father Friedrich Grundl of the Missionaries of the Sacred Heart was sent from
Germany; that is the date the accounts of Delaporte's rivalry with Father Alois Kayser carry.
One local page (`lifefmnauru.org`) does say 1899, so it is not certain, but the coincidence with
Delaporte's own year and the fact that no other file here makes the claim both point at a
conflation. **Checked but not changed**, because a date is a judgement rather than a typo.

**`note_public` says the country draws twelve dots. It draws eight.**
`data/processed/dots_nr.geojson` has 8 features (4 NCC, 3 Catholic, 1 AOG) and
`rings_nr.geojson` has 11. §2 of this
file and §9cx both say eight and eleven. Twelve is 11,623/1,000 rounded up, but with one unit
there is nothing for §4.1a's carry to accumulate across, so every fraction floors away and the
sub-dot nodes become rings instead. The sentence's point survives either number; the number
does not. It is said twice, in the last paragraph.

**The §9bs pointer in the header block was wrong and is fixed to §9cx.** §9bs is Cyprus.

### What was checked and is right

* **`gap_share=0.00488` is exact.** 57 / 11,680 = 0.004880, the nineteen categories sum to
  11,680, and `gap_share.py --check` returns *agrees (rows only)* against the authored 0.49%.
  The `-31.2%` in its `vs dots` column is the flooring above, not a data problem: `bm` is
  -31.6% and `ck` -24.9% on the same tier.
* **`christianity.reformed.congregational.ncc` earns its row.** The test that matters is not
  the CWM roll on its own but that this map already draws the same mission two ways on purpose,
  and Nauru honours the split: where a census **names** the body it gets a named node
  (`.cccs`, `.cicc`, `.ekt`, `.kpc`, `.niue`), and where a census prints a generic cell it goes
  to `christianity.protestant`, which is what `mh1999.py`, `fm2023.py` and `pw2005.py` do for
  the same LMS and American Board congregations. Nauru's table has both kinds and files both
  the standard way. No ask.
* **`Nauru Independent` -> `christianity.nondenominational` should stand.** Searched
  independently and found nothing the mapping note does not already have. Keep it because the
  church's own name says Independent, because `au2021.py`, `nz2023.py` and about twenty named
  Filipino bodies in `ph2020.py` are already filed there, and because `christianity.evangelical`
  holds an answer a source collected rather than a church. The Gale sentence about a 1977
  breakaway *"under the American Pentecostal church"* does not name the church and is ambiguous
  between doctrine and sponsorship, which is too little to move 410 people on.
  `christianity.other` is the other candidate and is worse, since it would throw away the one
  thing the name does say.
* **All 19 marks land inside the island's bbox**, and the 5 that fall outside
  `country_shapes.geojson` are the shapes layer being generalised for a 21 km² island rather
  than dots in the sea: every microstate does it and at a higher rate (`tv` 94%, `mh` 90%,
  `ki` 71%, `nr` 26%).
* **§14**: agreed with §9 above, no escalation.
