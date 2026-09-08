# Nepal — NSO, National Population and Housing Census 2021 (NPHC 2078), religion Table 1

Wired 2026-09-07. 29,164,578 people, **753 local levels**, 10 categories, **99.18% drawn**.

| | |
|---|---|
| source | National Statistics Office, **NPHC 2021**, *Population by religion and sex* — the caste/ethnicity release's **Table 5**, published as `Religion_NPHC_2021.xlsx` |
| basis | `self_id`, whole census population (**not** the household population — see §5) |
| geography | **753 local levels** (palikas) — ~38,400 people each; also published by district and province |
| categories | **10** plus the universe total; all 10 drawn |
| drawn | **28,925,480 people, 99.18%** — the balance is the institutional population (§5) |
| licence | NSO publication, free to download and cite; one GET, no wall, no registration |

**Ten named religions, an exact partition, and no residual of any kind** — no `Other`, no
`Not stated`, no suppression. Three of the ten are drawn on no other map here.

---

## 1. The host every earlier note names is dead, and the live one is inside a JS bundle

`sources.md` §11c recorded `censusnepal.cbs.gov.np` as answering "with **18 bytes** at the
root and 404s the documented dataset path", and §11r left Nepal as *"answers but thinly —
needs a proper look"*. Both are still true of that host and neither is true of Nepal.

```
  https://censusnepal.cbs.gov.np/               200, 18 bytes: <p>...working</p>
  https://censusnepal.cbs.gov.np/results        404
  https://cbs.gov.np/                           200, 0 bytes
```

**The office moved.** CBS became the **National Statistics Office** and `nsonepal.gov.np` is
alive (341 KB at the root). Its census results live on a third host again,
**`censusresults.nsonepal.gov.np`**, which is a **Next.js app**: 3 MB of HTML at the root,
`/downloads/caste-ethnicity?type=data` renders, and nothing on the page is a link to a file.

This is `sources.md`'s **"SPAs hide real APIs"** rule with a twist, because there is no API:

* `/_next/data/<buildId>/<route>.json` — the usual Next data route — **404s on every
  route**, including `index`. The pages are SSG and fetch nothing.
* `__NEXT_DATA__` on the rendered page carries **only the i18n dictionary**. It has the
  table TITLES (`religions_in_nepal_data` → *"Table XIII: Religions in Nepal"*) and no
  filenames, which is enough to know the file exists and not enough to fetch it.
* **The file list is a `JSON.parse('[…]')` literal inside the page's own chunk.** Every
  download page has one:
  `/_next/static/chunks/pages/downloads/caste-ethnicity-*.js` holds ten rows, and the row
  wanted is

  ```json
  {"group":"caste","key":"Table -5: Population by religion and sex",
   "xls_filename":"Religion_NPHC_2021.xlsx"}
  ```

* **And the folder is in the code, not in the row.** The same chunk builds the href as
  `"/files/caste/" + xls_filename`, while the *report* rows in the same array carry a
  full `href` ending `/files/result-folder`. Guessing `result-folder` for the data file —
  which is what the rows themselves suggest — **404s**. The thematic release is a third
  folder again, `/files/thematic/`.

```
  https://censusresults.nsonepal.gov.np/files/caste/Religion_NPHC_2021.xlsx   269,517 B  OK
  https://censusresults.nsonepal.gov.np/files/result-folder/Religion_...xlsx        404
```

**The generalising bit, for §12:** when an SPA has no API, the file list may still be a
literal in the page chunk — and *the path prefix will be in the JSX beside it, not in the
data*. Reading the row and guessing the folder is the failure mode; both halves are in the
same file and both have to be read.

## 2. Two workbooks, and the small one is the right one

| | |
|---|---|
| `caste/Religion_NPHC_2021.xlsx` | 270 KB, **2 sheets**, and the whole country at three tiers — **this is what is used** |
| `thematic/Excel_Tables_Series_XIII_Religions in Nepal.xlsx` | 160 KB, **68 sheets**, the *Religion in Nepal* monograph's table series |

The monograph's 68 sheets are national and provincial cross-tabulations — religion by age,
by sex, by literacy, by province — and **none of them goes below province**. The unglamorous
two-sheet file is the one with 753 local levels in it. Checked rather than assumed; the
monograph PDF (`thematic/Religion in Nepal.pdf`, 6.0 MB) is worth reading for context and
carries no finer geography either.

## 3. The sheet is indented, not coded, and that decides the parse

`Prov_District_local level` is 4,579 rows × 17 columns and **carries no geographic code of
any kind**. The hierarchy is columns B, C and D, one label per tier, blank otherwise:

```
  A          B         C            D                          E        G        H…
  NEPAL
                                                               Total    29,164,578  23,677,744 …
             KOSHI
                                                               Total     4,961,412   3,343,183 …
                       Taplejung
                                                               Total       120,590      36,717 …
                                    Phaktanlung Gaunpalika
                                                               Total        11,791       1,057 …
                                                               Male          6,088         556 …
                                                               Female        5,703         501 …
```

So `np.py` is a state machine, and **all four tiers are read even though only one is
written**, because the parent tiers are the only check that catches the failure this parse is
actually exposed to: a local level attached to the wrong district. Every national identity
survives that; the district identity does not.

Five checks, all passing on the first run:

1. the ten religions sum to the row total on **all 918 rows**;
2. each district equals its local levels **plus its institutional row**, on all 11 columns;
3. each province equals its districts, on all 11 columns;
4. Nepal equals its provinces, on all 11 columns;
5. the separate **`Nepal` sheet** agrees on all 11 national figures.

**Check 5 shares no cell with checks 1–4** and is what would catch a transposed column, which
all four nested identities would survive. It has its own trap: that sheet prints `Bahai` as
**`0`**, which is the 1-dp *percentage* and not the count of 537 — so the assertion is against
its column B, and reading its column C would fail on a correct parse.

## 4. Ten categories, and three of them are drawn on no other map here

```
  Hindu       23,677,744  81.19%      Prakriti      102,048   0.35%   <- new node
  Bouddha      2,393,549   8.21%      Bon            67,223   0.23%   <- new node
  Islam        1,483,066   5.09%      Jain            2,398   0.01%
  Kirat          924,204   3.17%   <- new node
  Christian      512,313   1.76%      Sikha           1,496   0.01%
                                      Bahai             537   0.00%
```

**They sum to 29,164,578 exactly.** There is no `Other` box and no `Not stated`, which puts
Nepal in a very small group here — Zimbabwe, Malawi and Guyana — and makes it much the
largest source of which that is true.

**Kirat, Prakriti and Bon are boxes on the form, not write-ins.** That is the difference from
India, which is the only comparable thing on this map: India's 83 named Adivasi religions all
sit inside `Other religions and persuasions` and had to be recovered from an appendix, so
`indigenous.indian` is a floor. Nepal asks. See `taxonomy/np2021.py` for each call and
`taxonomy/branches.py` for the nodes.

**The one that surprised the build is Bon.** The expectation was Mustang and Dolpa — the
trans-Himalayan districts with Yungdrung Bon monasteries. The census puts most of it in
**Gandaki's middle hills** (Manang 6.1%, Gorkha 5.7%, Lamjung 4.7%), which is Gurung country,
and the likeliest reading is that the cell is mostly the **Tamu shamanic tradition** rather
than monastic Bon. Written up in the node's note. *This was caught only because the
descriptive statistics were computed before the prose was written*, and the first draft of
that note asserted the opposite — §12's point about reading the whole list rather than the
label, one level up: **check the geography of a new category before describing it.**

## 5. §3.7's institutional population, and why it is dropped rather than spread

NSO prints an **`INSTITUTIONAL`** row inside each district, beside that district's local
levels and **not inside them**: 77 rows, **239,098 people, 0.82%**. Barracks, prisons,
hospitals, hostels — and Nepal's **monasteries and gompas**, which spec §3.7 names as exactly
the population this map most wants to see and structurally cannot.

Nepal lands on the *good* side of §3.7's distinction — the universe here is the whole census
population, so these people are counted and published — and on the bad side of a different
one: **there is no geography for them finer than the district.**

Three options, and the third was taken:

* **Spread them across the district's local levels by population.** Rejected. The
  institutional population is concentrated by its nature — a prison is in one palika, not
  smeared over twelve — so a population-weighted spread would be *actively wrong* rather than
  merely uncertain, and it would invent a location the source does not publish (§14.4 rule 1).
* **Draw them at district level, beside the palikas.** Rejected on the pipeline rather than
  the principle: the placement layer keys each hex to exactly one unit, so a country cannot
  have units at two tiers at once.
* **Drop them and say so.** Taken. §3.5 — undercounting is marked, not filled — and the
  `gap=` line in `countries.py` is where a reader learns it while the blank is on screen.

The largest institutional shares are where you would expect: Kathmandu, and the districts
with large monasteries.

## 6. What Nepal adds, beyond the three nodes

* **The second-largest Hindu population on earth**, at 23.7M — nearly twice Bangladesh's
  12.3M. §9v's Bangladesh bullet asserted the opposite and has been corrected; see sources.md.
* **The Muslim Terai, continuous with India across the border.** Rautahat 22.6%, Banke 18.7%,
  Kapilbastu 18.2%, against 0.00% in Bajhang. Both sides of that border are now drawn.
* **Three Buddhisms in one box** — Tibetan Vajrayana, the Newar Vajrayana of the Kathmandu
  valley that exists nowhere else on earth, and a 20th-century Theravada revival. The census
  separates none of them and neither does this map (§2.4).
* **A Christian population that is a floor for a stated legal reason.** Nepal's 2017 penal
  code criminalises conversion and "hurting religious sentiment" and prosecutions are
  documented; 1.76% is what people said in that setting.

## 7. Vintage and licence

NPHC 2021 was enumerated 11–25 November 2021 (Kartik–Mangsir 2078). The results release ran
2023–2024 and the caste/ethnicity/language/religion volume is part of it. **Census 2021 is
the current census**; the next is due 2031.

Nepal has no formal open-data licence on these files. They are official statistical
publications offered for public download with no registration, terms page or robots
restriction on the file paths used. Same footing as NIS Cambodia and ZIMSTAT — fine to draw
and cite, and worth re-checking before any commercial use
([[reference_poster_commercial_licences]]).
