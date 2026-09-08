# England's churches — two registers, one of which validates the other

`sources/uk_churches.py` → `data/normalized/uk_churches.csv` (447 rows).

This is the placement half of England's denominational split. `sources/uk_ecc.md` is the
congregation-size half, `sources/uk_bes.md` is the anchor, and `uk_split.py` multiplies the
three together. Read this one first, because it exists as a **correction** and the mistake
it corrects is instructive.

---

## 1. The bug that made this file necessary

The first version of the split placed denominations using the English Church Census 2005's
**attendance totals** by county, response-corrected by the national per-denomination rate its
user guide publishes. It drew a Merseyside that was **72.2% Anglican and 16.0% Catholic**.

That is not Liverpool, and the reason is one sentence in the user guide:

> We were also able, however, to input data on total attendance, sometimes with additional
> information, that was very kindly supplied by ten Church of England and eight Roman
> Catholic Dioceses, the Baptist Union of Great Britain, the Fellowship of Independent
> Evangelical Churches, the Salvation Army and 91 Methodist Circuits for the remainder.

The Baptist Union, the FIEC and the Salvation Army supplied **nationally**, so their coverage
is even across England. The Anglican, Catholic and Methodist supplies came **diocese by
diocese and circuit by circuit**, so theirs is not. A county whose bishop's office sent a
spreadsheet has near-total coverage; a county next door has a postal response rate.

**Measured against the Church of England's own complete register**, the church census's
Anglican response rate is:

| | |
|---|---:|
| England | **56.2%** (the user guide says 55%, so the measurement is sound) |
| Norfolk | 25.9% |
| Durham | 35.5% |
| Tyne & Wear | 37.3% |
| … | |
| Bedfordshire | 85.2% |
| Merseyside | **91.4%** |
| Gloucestershire | 92.3% |
| Greater Manchester | 92.5% |

A factor of three and a half between counties, and the high ones are the ten dioceses. So
Merseyside's Anglicans were counted almost completely and its Catholics were not, because
Liverpool did not send a spreadsheet and Hexham & Newcastle did. **A single national
correction cannot see that**, and applying one produced a confident, precise, wrong map.

**What survives and what does not.** A *total* carries the response rate. A *mean
congregation size* does not — losing half a county's Methodist chapels changes how many you
saw, not how big they were. So the split was rebuilt as

    placement weight  =  churches now  ×  mean congregation in 2005

with this file supplying the first factor and `uk_ecc.py` the second.

**A first instinct that did not work, recorded so nobody repeats it.** Bulk-supplied churches
ought to be identifiable — they should have a total and none of the questionnaire detail. They
are not. Restricting to churches that answered the age grid, the ethnicity question or the
churchmanship question made the county spread *worse* (a factor of 105), and the "form return"
rate came out **highest** for Catholics at 79.6% and Orthodox at 93.9%. Whatever those fields
measure, it is question completion and not data provenance.

---

## 2. The two registers

### Church of England, `Churches_July2026`

The CofE's own operational list of its churches, published on its ArcGIS organisation. **15,784
rows, 15,492 with coordinates inside England.** Authoritative in the strict sense: it is the
register itself, not a survey of it. Anglican placement comes from here alone.

**OSM's Anglican churches are discarded rather than merged.** Two registers of the same thing
added together give about 32,000 Anglican churches in a country with roughly 16,000. This is
the kind of double count that produces a plausible-looking map, so `register()` never reads
`denomination=anglican` from OSM except to compute the validation in §3.

### OpenStreetMap

Everything else, because no other free register covers all denominations at once. Overpass,
`amenity=place_of_worship` + `religion=christian` across England and Wales: **33,780 places of
worship, 84.5% carrying a `denomination` tag.**

```
[out:json][timeout:600];
area["ISO3166-2"="GB-ENG"]->.e;
area["ISO3166-2"="GB-WLS"]->.w;
(
  nwr["amenity"="place_of_worship"]["religion"="christian"](area.e);
  nwr["amenity"="place_of_worship"]["religion"="christian"](area.w);
);
out tags center;
```

Overpass **requires a User-Agent** or returns a bare 406.

---

## 3. The validation, and it is the reason this file is trustworthy

OSM's reliability is not assumed. It is measured on the one denomination where a complete
list exists — **OSM's Anglican churches against the Church of England's own, across all 47
counties**:

| | |
|---|---:|
| correlation `r` | **0.9734** |
| national coverage | 105.4% |
| per-county median | 103% |
| interquartile range | 100–105% |
| full range | 92–149% |

`check()` recomputes this on every build and **fails the run if `r` drops below 0.90**, because
every non-Anglican leg rests on the assumption that OSM tracks reality the way it does here.

The over-count is consistent and small: OSM includes some churches the CofE's current list does
not (closed, converted, chapels-of-ease). Merseyside at 149% and Devon at 130% are the worst,
and Merseyside is flagged in §5.

---

## 4. What this file will and will not place

| placed here | anglican · catholic · methodist · baptist · reformed |
|---|---|
| **deferred** | **pentecostal · newchurch · orthodox · other** |

OSM maps buildings, so it finds denominations that own buildings and misses denominations that
rent halls. In the whole of England it has **355 Pentecostal and 129 Orthodox** churches, both
large undercounts — Pentecostal congregations meet in industrial units and hired schools,
Orthodox parishes are mostly post-2004 and often share Anglican buildings, and `newchurch`
(Vineyard, Newfrontiers, FIEC) has **no OSM denomination tag at all**.

Using those counts anyway put **6.0% of Norfolk's Christians on Orthodoxy and 7.6% of
Durham's on Pentecostalism**, neither of which is true. The English Church Census is blind in
exactly the same places and for the same reason: it is twenty years old and those are the legs
that grew afterwards.

So those four legs — 8.0% of England's Christians — are deferred to a census-proxy placement
(country of birth for Orthodoxy, ethnic group for Pentecostalism). Anita, 2026-09-07: *"lets do
pentecostal and orthodox later, get the real placeables first."* Until then `uk_split.py` leaves
those people on the census's own `Christian` category rather than inventing a county for them.

**A shared building counts as a fraction of each denomination named on it.** OSM writes Local
Ecumenical Partnerships as `anglican;methodist`, `methodist;united_reformed` and so on, 100+ of
them. Each named denomination gets 1/n of the building; the Anglican share is then dropped
because the CofE register already counts that building.

---

## 5. Open, and worth checking before this ships

**The `reformed` leg looks concentrated where former chapels are thickest.** Merseyside has
**81.5 Reformed churches per 197 Anglican — 41 per 100, against a national rate of 6.2.**
Shropshire is 20 per 100, Avon 12.5. Liverpool genuinely was a Welsh Presbyterian and
Congregational stronghold and the Welsh borders genuinely were Congregational, so some of this
is real. But those are also the places with the most disused chapels, and OSM does not reliably
retag a chapel that has become a house.

It does not affect the national level, which the British Election Study anchors. It affects
where 568,000 dots land. The test would be to compare against the United Reformed Church's own
list of congregations, which it publishes but not as data.

---

## 6. Licences

- **Church of England church locations** — published open on the CofE's ArcGIS organisation;
  attribute as *Church of England, Data Services*.
- **OpenStreetMap** — © OpenStreetMap contributors, **ODbL 1.0**. Share-alike, and it is the
  only share-alike source in this map's England. Attribution is required on anything drawn
  from it; see `reference_poster_commercial_licences` before any of this reaches a print.
- The ONS lookups (`oa_ruc.csv`, `oa_lad.csv`, `oa_centroids.csv`) are OGL v3.0, Crown
  copyright, from the Open Geography Portal.
