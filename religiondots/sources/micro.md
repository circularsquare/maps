# The microstate tier — nine countries from UNSD table 28

Wired 2026-09-08. 311,888 people across nine countries, none of which required opening a
statistical office.

| | |
|---|---|
| source | UNSD Demographic Yearbook table 28, `tools/oracle.py` |
| basis | `self_id` |
| geography | one unit per country, the country itself |
| placement | Kontur population hexagons, 2023-11-01 vintage |
| drawn | **306,393 of 311,888, 98.2%** |

| code | country | census | cats | people | dots 1:1k | what it brings |
|---|---|---:|---:|---:|---:|---|
| `pw` | Palau | 2005 | 9 | 19,907 | 17 | **Modekngei**, a new root |
| `ck` | Cook Islands | 2011 | 9 | 14,974 | 11 | Cook Islands Christian Church |
| `tv` | Tuvalu | 2017 | 11 | 10,507 | 9 | Ekalesia Kelisiano Tuvalu, 85.9% |
| `nu` | Niue | 2017 | 8 | 1,591 | **0** | Ekalesia Niue; rings only |
| `ms` | Montserrat | 2001 | 11 | 4,303 | **0** | rings only |
| `bm` | Bermuda | 2010 | 23 | 64,237 | 43 | highest AME share on the map |
| `ag` | Antigua and Barbuda | 2001 | 21 | 76,886 | 69 | highest Moravian share |
| `dm` | Dominica | 2001 | 14 | 68,635 | 61 | 61.4% Catholic in a Leeward group |
| `mh` | Marshall Islands | 1999 | 4 | 50,848 | 49 | shallowest table on the map |

---

## 1. Why these are buildable, which is a decision and not a discovery

queue.md had eleven microstates parked as *"a decision rather than a build"*. Anita made it on
2026-09-08: *"for the really small island countries we might not even need any divisions. like
for instance if we do palau, it has 17000 people so itll just be 17 dots."*

**At 1 dot = 1,000 people, placement inside a country of 20,000 carries no claim.** Seventeen
dots spread over Palau assert nothing about where Palau's Catholics live, so a national-only
table is a *complete* source rather than a coarse one. That is what retires spec §3.9b's
unit-count floor and §3.9c's variety floor for this tier, and only for this tier.

The count Palau actually draws is **seventeen**, which is what Anita's sentence predicted to
within a rounding error, and is a good sanity check on the whole premise.

## 2. The oracle was already holding the data

The route is `tools/oracle.py`, and the finding that made it possible is written up in
sources.md §11r: **the documented column set returns an index with the values stripped out.**
`c=0,2,3,6,8,10,15,16` returns the counts. Applied to the queue, **33 of 37 countries** turned
out to be present with figures; these nine are the ones where the oracle is a *source* rather
than a check.

No office was contacted for any of them. Several of queue.md's blockers evaporated rather than
being solved: `mfem.gov.ck/statistics` still 404s, `rmi-data.sprep.org` is still a 403, and
neither mattered.

## 3. The partitions, and the one that does not close

Eight of nine sum to their own stated total **to the person**. `sources/micro.py` pins the
expected total and category count per country, so a re-publication that revises a figure fails
the build rather than quietly changing the map.

**Antigua and Barbuda's 21 categories sum to 76,889 against a stated 76,886**, three people
over. It is 0.004%, it is in the Yearbook rather than in this code, and the country draws 77
dots either way. `EXPECTED` allows exactly 3 for `ag` and 0 everywhere else.

## 4. Two countries draw no dots at all, and that surfaced two real bugs

**Niue's largest religion is 981 people and Montserrat's is 937.** One dot is a thousand
people, so every category in both countries is sub-dot and both draw entirely as §4.3 rings.
That is the correct rendering, and nothing in the pipeline had ever seen it:

* **`buffers.py` crashed.** `nb` is floored at 1, so an empty country still entered the
  bucket loop once with `cnt == 0`, and `.min()` on the empty slice raised *zero-size array
  to reduction operation minimum*. Fixed by emitting no bucket; the manifest then carries
  `dots: 0, buckets: []`.
* **`tiles.py` dropped them from `counts.json`.** The per-country loop was
  `for cc in sorted(set(dots["c"]))`, so a rings-only country never got an entry — which is
  not "no dots" but the country vanishing from the picker, the about panel and `covers`
  entirely, undoing the ring one step later. Fixed by iterating dots ∪ rings and taking the
  bbox from the rings when there are no dots.

Both were latent for every country ever added; the tier just made them reachable.

## 5. What this tier may not be used for

**Every row is national.** Nothing here says where inside a country anyone lives. Bermuda is
the only one where the Yearbook also publishes urban and rural, and even that is not read: a
two-way split is not a geography.

**Placement is population only**, from Kontur's 2023-11-01 model, so a Catholic dot and an
Adventist dot spread identically. Ratios against the census run 0.85 (Marshall Islands) to
1.56 (Niue); the censuses are 1999 to 2017 and Kontur models the present, so a wide band is
expected and only a wild one would mean the wrong country's file.

## 6. Staleness, declared per country

Census years run **1999 to 2017** and four are 2001 or older. Each country's `how=` says its
year on its own panel. **The Marshall Islands' 1999 round is the oldest census drawn anywhere
on this map**, and its four categories the shallowest table; it is drawn because the variety
floor was retired, not because it is good.

## 7. New nodes

* **`modekngei`** — a ROOT, on caodaism's precedent (spec §3.3): a named, founded, organised
  syncretic religion, not a diffuse body of traditional practice. 1,733 people, 8.7% of
  Palau, a larger share of its country than any other indigenous religion here.
* **`christianity.reformed.congregational.cicc` / `.ekt` / `.niue`** — the three London
  Missionary Society daughter churches, each the national church of its country. Samoa's
  CCCS would be a fourth if Samoa is ever drawn, and the Marshall Islands' UCC is a fifth
  that **cannot** be filed there, because its census says only `Protestant`.
* Nine `other.<cc>` residuals, grouped in `branches.py` with one shared caveat: the
  Yearbook's category names are the office's, but the classification into them is UNSD's.
