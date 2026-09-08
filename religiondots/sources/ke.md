# Kenya — KNBS, 2019 Kenya Population and Housing Census, Volume IV

Wired 2026-09-05. 47,213,282 people, 47 counties, 11 drawn categories.

| | |
|---|---|
| source | Kenya National Bureau of Statistics, KPHC 2019 **Volume IV, Table 2.30** |
| basis | `self_id`, conventional household population |
| geography | **47 counties** — the coarsest counting geography on this map |
| categories | **13** plus the universe total; 11 drawn |
| drawn | **47,133,120 people, 99.83%** of the table's universe |
| licence | KNBS publication, free to download and cite |

**The deepest religion question in Africa, on the coarsest geography this project draws.**
Both halves of that sentence are the country. Two of the thirteen categories —
`Evangelical Churches` and `African Instituted Churches` — are counted by no other census
anywhere on this map, and each earned a new node in `branches.py`. And 47 units for 47.2M
people is about a million each, coarser than the Philippines' 929,000 (§9m).

---

## 1. One page, and the parse is exact

Table 2.30 is a single page of a 498-page PDF (page index 434, printed page 422). 47 county
rows plus a KENYA row, fourteen figures each, no suppression, no rounding, no blank cells.

Parsed **by order rather than by x-position**: every row prints its figures left to right in
the header's order and never omits one. That is checked twice and both checks are
equalities, because there is nothing in the table for them to be inequalities about —

* the 13 categories sum to each row's own `Total`, on all 48 rows;
* the 47 counties sum to the KENYA row, on all fourteen columns.

A column-boundary reading would have been more fragile, not less: the figures are
right-aligned, so a wide number's `x0` crosses into its neighbour's band. `sources/ke.py`
also asserts the page's TITLE before parsing, so a re-paginated re-issue fails loudly rather
than parsing the ethnicity table on the next page as though it were religion.

## 2. 47 counties is KNBS's ceiling, and the volume says so on its own face

This is the §3.9 trade taken to the category end, and it was made by the office rather than
here. **Volume IV has forty-odd tables and every other one of them is titled "…by County and
Sub-County"** — activity status, education, disability, albinism, crops, livestock, mobile
phones, internet use, births. Religion is Table 2.30, "…by Religious Affiliation and
County", and there is no sub-county twin anywhere in the volume. Religion is the one
variable KNBS stopped at county for.

It is not a collection limit. The questionnaire annex at the back of the same volume lists
the geography captured for every household: County / Sub-County / Division / Location /
Sub-Location / E.A. The data exists at enumeration-area grain and is not published.

**The upgrade path is IPUMS**, whose Kenya 2019 sample is 10% (4.72M persons) and identifies
**division**, below county — sources.md §10a, and it needs the account that is still
outstanding. Nothing else found in a day of looking gets below county: openAFRICA carries
the county table as CSV and nothing finer, and is behind Cloudflare anyway.

## 3. The category list, and why the Kenyan split of Christianity is not the usual one

```
Total                          47,213,282   universe
  Catholic                      9,726,169   20.60%
  Protestant                   15,777,473   33.42%
  Evangelical Churches          9,648,690   20.44%
  African Instituted Churches   3,292,573    6.97%
  Orthodox                        201,263    0.43%
  Other Christian               1,732,911    3.67%
  Islam                         5,152,194   10.91%
  Hindu                            60,287    0.13%
  Traditionists                   318,727    0.68%
  Other Religion                  467,083    0.99%
  No religion /Atheists           755,750    1.60%
  Don't Know                       73,253    0.16%   off the tree
  Not Stated                        6,909    0.01%   off the tree
```

**`Protestant` and `Evangelical Churches` are PEERS, not parent and child**, and reading
them the usual way inverts the map. In KNBS's usage `Protestant` is the mainline mission
inheritance — the Anglican Church of Kenya, the Presbyterian Church of East Africa, the
Methodists, the Lutherans, the Salvation Army — and `Evangelical Churches` is the
faith-mission and evangelical-alliance stream: the Africa Inland Church, the Baptists, the
Pentecostal Assemblies of God, Deliverance Church, the Redeemed Gospel Church. **An
Anglican here is a Protestant; a Baptist here is an Evangelical.** Together they are 53.9%
of Kenya.

That is why `Evangelical Churches` maps to a new answer-node, `christianity.evangelical`,
and not to `christianity.pentecostal` or `christianity.baptist`: the category cuts across
families the tree keeps apart, and choosing one of them for 9.6M people would assert a
division the census did not make. `mk2021.py` wanted this node for 678 Macedonian
`Евангелисти` and was right not to add it then — spec §2 says a node earns its place by
being countable, and Kenya makes it countable fourteen thousand times over.

## 4. African Instituted Churches — the reason to draw Kenya

3,292,573 people, 7.0%, and **the single most valuable category in the file.** These are the
churches founded in Africa by Africans outside the mission denominations: in Kenya the Legio
Maria (a Luo Catholic-derived church with its own pope), the Nomiya Luo Church, the African
Israel Nineveh Church, the Akorino, the Roho churches.

Almost nowhere else on earth counts them. Ghana has exactly the same kind of church — the
Musama Disco Christo Church, the Twelve Apostles, the African Faith Tabernacle — and no cell
for any of it, so `gh2021.py` had to send all of them to `christianity.other` and wrote that
this node was what it wanted and where its people would be found until someone supplied it.
Kenya supplies it, one day later. That is spec §2.4's "deepening later costs nothing" working
as designed: Ghana's rows still carry their `source_category` and can be moved the moment a
Ghanaian source separates them.

**And they have a homeland, which is the part a map can show.** Siaya is 23.9% AIC, Kisumu
18.2%, Homa Bay 17.8%, Vihiga 15.1%, Migori 13.8% — the Luo counties of Nyanza and western
Kenya — against a national 7.0%. That concentration is the Roho and Legio Maria heartland
and it is tight enough to read at a glance.

## 5. The universe, and the footnote that explains it

47,213,282 against a census 47,564,296. The 351,014 difference — 0.74% — is named by the
table's own footnote: *"The question was not asked to those who are in hotels/lodges,
Hospital, Prison/Police Cell, Children's Home, Travellers and Outdoor Sleepers."* So the
universe is the conventional household population, which is the same shape as the
Philippines' (§9m) and is stated rather than inferred. Not scaled up (§14.4).

`Don't Know` (73,253) and `Not Stated` (6,909) are non-answers and go off the tree per §3.5.
`Don't Know` is the larger of the two and is worth a thought: Kenya enumerates by household,
so it mostly means one person did not know a relative's religion, not that anyone was
unsure of their own.

## 6. What the map shows

- **The Muslim north-east is a block, not a gradient.** Mandera 99.4%, Wajir 99.0%, Garissa
  97.6% — as near-total as anything on this map outside Sulu — with Tana River 81.5% and
  Isiolo 72.3% behind it and the coast strip running high into Mombasa and Lamu.
- **Catholic Kenya is the Rift Valley missions.** Samburu 57.3%, Elgeyo/Marakwet 51.1%,
  Turkana 44.1%, against a national 20.6%.
- **AIC Kenya is Luo Kenya** — see §4.
- **Traditional religion is a pastoralist survival**: Marsabit 15.5%, Samburu 9.9%, Turkana
  4.7%, against 0.68% nationally and near zero everywhere south of it. Read it as a floor
  for the same reason as Ghana: the box is exclusive of the Christian and Muslim ones.
- **Kilifi is 10.2% no-religion**, two and a half times the next county (Narok, 4.0%) and a
  real outlier on the Mijikenda coast. Nothing else in the table looks like it, and it is
  the one figure here worth being curious about rather than confident.
- **Kenya's Hindus are two cities.** 60,287 nationally; Nairobi holds 38,141 and Mombasa
  6,136. A tiny national share that is a visible local one — the East African Asian
  community.
- **The Orthodox are western, which is unexpected and correct.** Nandi 1.69%, Kisumu 0.93%,
  Vihiga 0.84% — higher than Nairobi. The Orthodox Church in Kenya grew from Reuben Spartas'
  African Orthodox movement in the 1930s and was received by the Patriarchate of Alexandria;
  its heartland has been western Kenya ever since. A reader who expects Orthodoxy to mean a
  diaspora in the capital will read this county pattern wrongly.

## 7. Ethics (§14)

Nothing here approaches §14's line, for the same reasons as Ghana (`gh.md` §7). KNBS
publishes religion itself, in a free national report; the map draws exactly the tier KNBS
publishes and nothing finer, so §14.4's rule about persecuted groups is satisfied trivially.
The north-eastern counties are Muslim, Somali and have been subject to security operations,
which is a reason to be careful about resolution — and the resolution here is the county,
which is the state's own.

## 8. Not done

- **Nothing below county**, and IPUMS is the only route — see §2.
- **The 2009 census** also asked religion, with a shorter list. Not ingested; spec §13 says
  no time slider.
- **`Don't Know` is not modelled away.** It would be easy to distribute it across the named
  categories pro rata and it would change nothing visible, and it would also be inventing
  71,000 people's answers.
