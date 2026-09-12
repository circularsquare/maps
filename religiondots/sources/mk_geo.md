# North Macedonia — boundaries

`sources/mk_geo.py` writes `data/geo/mk/mk_opstini.gpkg` (80 polygons) and
`data/geo/mk/mk_lookup.csv`. **It downloads nothing.**

## 1. GISCO LAU is not the EU27, and that is worth knowing before hunting

The 98 MB Eurostat file `sources/pl_geo.py` already pulled covers **34 countries**, and its
`CNTR_CODE` list includes `MK`, `RS` and `AL` — candidate countries, not members. North
Macedonia's 80 opštini were already on disk before this country was started.

That makes GISCO LAU 2021 the answer for Serbia and Albania too, if either ever gets counts.
The whole list, for anyone checking: AL AT BE BG CH CY CZ DE DK EE EL ES FI FR HR HU IE IS
IT LI LT LU LV MK MT NL NO PL PT RO RS SE SI SK.

## 2. The join is by name, and it had to be

SSO's PxWeb municipality codes are four digits (`1066` = Bitola). GISCO's `LAU_ID` is
`MK00101`-style. They share no substring and joining as delivered matches **zero of 80** —
Poland's trap exactly (§12), except that Poland could be rescued by slicing a derived key and
North Macedonia cannot, because there is nothing to slice and the Eurostat LAU–NUTS
correspondence workbook is EU27 and excludes it.

What makes a name join safe here is a piece of luck worth naming: **GISCO carries North
Macedonia's names in Cyrillic, and so does SSO.** The English PxWeb edition says `Veles`;
the Macedonian edition says `Велес`; GISCO says `Велес`. So `sources/mk.py` fetches *both*
language editions, for two different consumers — English for the category labels that
`taxonomy/mk2021.py` keys on, Macedonian for the names that the geometry keys on. Neither
edition alone builds this country.

Folded (NFKD, combining marks stripped, non-alphanumerics dropped) the match is **80 of 80,
zero unmatched on either side, zero ambiguous**.

## 3. A 100% name join is exactly what a subtly wrong one looks like

So it is verified against something independent, per §12. GISCO carries `POP_2021`; the
census carries its own municipal totals; they are different collections of the same people
and a correct pairing has to put them in a *systematic* ratio.

```
census / GISCO POP_2021 over the 73 units GISCO populates
  min 0.521   median 0.895   max 1.047
```

Tight, one-sided, and every unit inside a factor of two. A scrambled join pairs villages with
cities and scatters that ratio over orders of magnitude, which is the thing this check can
actually detect — so the assertion is on the *shape* of the distribution, not on equality.
An earlier version demanded near-equality, failed at a median 11.7% disagreement, and was
wrong to: the gap is real and it is emigration. The 2021 census counts **residents**, and the
lowest ratios are precisely the western emigration municipalities — Центар Жупа 0.52,
Маврово и Ростуше 0.57, Желино 0.67.

**Two quantities that measure the same population differently must not be asserted equal.
Assert the relationship instead.**

## 4. GISCO's population is ZERO for most of Skopje

`POP_2021` is `0` on seven of the ten Skopje municipalities — **Аеродром, Бутел, Кисела
Вода, Сопиште, Центар, Чаир, Шуто Оризари** — which between them are a large part of the
capital. It is a hole in GISCO, not a join failure, and it is why the MK column of that file
sums to 1,746,833 against a census 1,836,713.

Nothing in this pipeline uses `POP_2021` for anything except the check above (§8.2: placement
needs no population data), so it costs nothing here. But anyone who later reaches for it —
as a placement weight, or as a denominator — will silently lose central Skopje. `mk_geo.py`
prints the list on every run rather than filtering it away.

## 5. Placement

The count layer **is** the placement layer, as in Poland, Romania and India: 80 polygons, one
per unit, dots spread uniformly inside each. The median municipality is about 12,000 people
and 300 km², so at city zoom North Macedonia is blocky.

The upgrade exists in principle — SSO publishes by settlement for other variables — but a
settlement polygon layer for North Macedonia is not in GISCO and was not looked for, because
§8.2's usual justification does not apply: settlements are natural places of wildly differing
size, not units engineered to a population target, so an equal share of dots per settlement
would weight a hamlet like Tetovo. It would need real settlement populations to be worth
doing.
