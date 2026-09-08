# Sri Lanka — DCS, Census of Population and Housing 2024

Wired 2026-09-04. 21,781,800 people, 14,003 GN divisions, 6 categories.

| | |
|---|---|
| source | Department of Census and Statistics, CPH **2024**, `GN_Level_Population_by_Religion.xlsx` |
| basis | `self_id` |
| geography | **14,003 Grama Niladhari divisions** (~1,555 people each) |
| categories | 6 plus the universe total |
| drawn | **21,781,800 people, 100.0%** |
| licence | DCS publishes it openly; attribution |

**The finest geography in the project outside the US tracts and the German grid, and the
shallowest category list anywhere in it.** Sri Lanka is §3.9's trade made about as hard as
it goes in one direction: 14,003 units and six answers. It is also the first
Buddhist-majority country on the map, and the first where four of the world's large
traditions are present in numbers and in sharply separated places.

It cost one download. **All of the difficulty was in the boundaries** — see `lk_geo.md`,
which is the interesting half of this country.

---

## 1. Getting it

`https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024` links a block of
GN-level workbooks — population by ethnic group, by five-year age group, by local government
area, **by religion**, by sector. No API, no auth, no bot protection; the links have no file
extension and serve the xlsx as an attachment, which is the only thing worth noting.

The 2024 census results were published during 2025–26, which is why this is a 2024 source
and not the 2012 one §11 had listed. **The 2012 census also published religion at GN level**
and is not used: §13 rules out a time slider, and the newer count is simply better.

## 2. The trade, and which end of it Sri Lanka is on

Six categories: Buddhist, Hindu, Islam, Roman Catholic, Other Christian, Other.

That is a question asked at the level of world religions. **Nothing in it needed a new node**
except the residual, which for a country of 21.8M is unusual and says what kind of source
this is. Compare Croatia, where 3.9M people arrive in 30-odd named bodies, or Poland's 216.

What the geography buys instead is that the four traditions are *separated on the ground*
rather than averaged into districts:

- **Buddhists 15,196,960 (69.8%)** — the south, the centre, the whole dry zone.
- **Hindus 2,718,154 (12.5%)** — in **two disconnected places**, which is the thing the fine
  grain shows and a district map does not: the Jaffna peninsula and the Vanni in the north
  (Sri Lankan Tamils), and the tea districts of the central highlands around Hatton,
  Nuwara Eliya and Maskeliya (Indian-origin Tamils, brought in under British rule). The
  religion question cannot tell the two populations apart; the map can, because they are
  200 km apart.
- **Muslims 2,327,605 (10.7%)** — the east coast in a near-continuous strip from
  Trincomalee through Batticaloa, Kalmunai and Akkaraipattu, plus inland pockets and the
  Puttalam resettlement area on the west coast.
- **Roman Catholics 1,209,072 (5.6%)** — a coastal band running north from Negombo through
  Chilaw to Kalpitiya, plus Mannar. **This is the sharpest religious boundary in the
  country** and it is a few kilometres wide, so it exists on this map and would not exist
  on a district one.
- **Other Christians 266,515 (1.2%)**, **Other 63,494 (0.29%)**.

### Measured off the drawn map, rather than asserted

**The two Hindu populations**, which is the claim the fine grain exists to support:

```
                 Hindu    Buddhist        pop
  Jaffna         82.3%       0.4%      594,751     the north
  Kilinochchi    80.6%       1.0%      136,710
  Mullaitivu     72.4%       8.3%      122,619
  Nuwara Eliya   52.0%      38.4%      725,280     the tea country, 200 km away
  Badulla        19.0%      73.0%      872,307
```

Nuwara Eliya is the interesting one: **52% Hindu and 38% Buddhist in the same district**, so
a district-level map shows a purple average and this map shows the estates and the villages
separately.

**The east coast**: Trincomalee 46.4% Muslim, Ampara 45.6%, Batticaloa 27.1%, Puttalam 21.6%.

**The Catholic strip.** Twelve DS divisions are over 40% Roman Catholic against a national
5.55%, and they are almost all coastal:

```
  77.6% Wennappuwa (Puttalam)      63.5% Negombo (Gampaha)     50.4% Mahawewa (Puttalam)
  66.4% Nanattan (Mannar)          60.9% Delft (Jaffna)        49.0% Ja-Ela (Gampaha)
  50.6% Mannar Town                50.6% Jaffna                44.4% Chilaw (Puttalam)
```

And it really is a boundary rather than a gradient. Within Gampaha and Puttalam alone,
**116 GN divisions are over 80% Roman Catholic and 875 are under 5%**, out of 1,725 — a
bimodal split a few kilometres wide, which is the single best argument for drawing this
country at GN level rather than DS.

**No school, branch or jurisdiction anywhere.** 15.2M Buddhists arrive with no vehicle
attached even though the island is one of Theravada's historic centres; 2.3M Muslims with no
madhhab; 266k non-Catholic Christians in one cell that runs from the Church of Ceylon to
independent Pentecostals. `taxonomy/lk2024.py` argues each of those calls; the short version
is that the source does not say, so neither does the map.

## 3. The whole population is drawn, and that is a fact about the question

The six categories account for **all 21,781,800 people**. There is no `not stated`, no
`no religion` and no refusal line anywhere in this census.

No other country on this map does that, and it is not a quality signal — it is the opposite
of one. **Irreligious Sri Lankans are not missing from this map; they are counted inside one
of the six.** So a Buddhist share here and a Buddhist share from a country with a
no-religion option are not the same quantity, and §3.1's rule about never mixing bases has a
subtler form for Sri Lanka: the basis is self-identification, but the *universe* has no exit.

`note_public` says this in as many words. It is the most likely way for a reader to be
misled by this country.

## 4. The disclosure rule is in the sheet name

The sheet is called `By Religon(<10 add to Other)` (DCS's spelling). Where a religion has
fewer than ten people in a GN division, its count is moved into `Other`.

This is spec §3.8 — disclosure control biasing the rare categories downward — declared by
the source rather than discovered, which is the good version. Two consequences:

- **`-` appears in 39,241 cells and means zero**, but a zero that may be "zero" or "up to
  nine people who are now in Other". `sources/lk.py` classifies every cell and raises on
  anything it does not recognise; `errors="coerce"` here would have deleted the sentinel
  silently and broken the row sums.
- **`Other` is not a clean residual.** It holds genuine other-religion answers mixed with
  the suppressed tail of the five named categories, in unknown proportion. The ceiling on
  the damage is exact and small in the direction that matters: everything the rule moves
  lands in `Other`, so at most **63,494 people, 0.29%**, are misfiled. What is lost for
  good is Sri Lanka's small-group tail — the Bahá'ís, the Parsis of Colombo, the Malay
  Muslims, Vedda practice — none of which can be recovered at this geography.

## 5. The arithmetic

Every check in `sources/lk.py` is an equality, and all of them hold:

- each of the 14,003 GN divisions: its six categories sum to its own published total
  (14,003 separate checks, 0 failures);
- the 14,003 units sum to the workbook's national row on all seven columns exactly;
- the national row is 21,781,800, the published CPH 2024 figure;
- 25 districts, 340 DS divisions, 14,003 GN divisions, no duplicates.

The workbook holds **one aggregate row only** (the national line, row 4, with empty code
cells) and no district or DS subtotals mixed into the body. That is worth stating because it
is where Hungary and India both went wrong — §12's rule is to sum one category and compare
with the published figure, which is done here and passes.

## 6. Not done

- The Buddhist school split and the Muslim madhhab split. No table carries either.
- The 2012 census, which would allow a comparison; §13 rules out a time slider.
- DCS also publishes GN-level **ethnicity**, which would separate Sri Lankan Tamils from
  Indian-origin Tamils and Moors from Malays. Using it to subdivide religion is exactly
  what §14.4 forbids, and here the correlation is high enough that it would be tempting.
  Not done, and should not be.
