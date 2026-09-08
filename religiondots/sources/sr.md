# Suriname — ABS, Census 7 (2004), census profile at ressort level

Wired 2026-09-07. 492,829 people, 62 ressorten, 5 drawn categories, **84.33% drawn**.

| | |
|---|---|
| source | Algemeen Bureau voor de Statistiek, **Census 7 (2004)**, `census-profile-on-ressort-level.xls`, block 5 `Religion`; district grouping from `district-profiel-census.xls` |
| basis | `self_id`, whole census population |
| geography | **62 ressorten** — ~7,900 people each, **the finest counting tier in the Americas on this map** after Saint Vincent's enumeration districts |
| categories | **6** plus the universe total; 5 drawn |
| drawn | **415,625 of 492,829 people, 84.33%** — the gap is `Don't know/No answer` |
| licence | ABS publication, free to download and cite |

**Suriname is 13.45% Muslim — the highest share in the Americas by a wide margin — and
19.93% Hindu**, and the grain is fine enough to show that neither is spread but sorted:
Hinduism reaches **65% in Jarikaba**, Islam **48% in Nieuw Amsterdam**. With Guyana (§9r) and
Trinidad (§9ak) this completes the Indo-Caribbean geography.

---

## 1. The vintage is 2004 and §11t was wrong to call it a fork

§11t framed Suriname as a choice between fine geography (2004) and deep categories (2012).
**It is not a choice. 2012 has no usable geography at all.** Checked three ways:

* **Census 8 Volume 1 publishes religion nationally only.** It carries a full denominational
  list — Rooms Katholiek, Luthers, Volle Evangelie, E.B.G., Javanisme, Islam Soenniet,
  Hindoe Sanatan, Overig Hindoe (Incl. Aryah) and more — crossed with **ethnicity** and with
  **nationality**, and cut by no geography whatsoever.
* **The Districtsresultaten presentations reach 3 of 10 districts.** Volume III (Marowijne,
  Brokopondo, Sipaliwini) has `Bevolking naar Godsdienst (denominatie) en geslacht`;
  Volumes I and II have substantial text tables on other subjects and **no religion pages**.
  That was verified against the text layer — all three volumes carry real text (22k, 25k and
  14k characters) — so the absence is the source's and not a scanning artefact.
* **Census 9 has published nothing.** Fielded to July 2025 ("CENSUS LAATSTE RONDE 290725").
  ABS's WordPress media library holds **3,662 items**, enumerated in full; everything
  Census 9 is promotional, legal or recruitment. When its results land they supersede this
  file entirely.

So 2004 is the only whole-country sub-national religion table Suriname has. Drawn, with the
vintage in `gap=` and stated plainly in `note_public`. Precedent for the age: China 2000
(§14.6) and Ethiopia 2007 are both drawn and both older relative to their own countries'
change.

## 2. Access

```
https://www.statistics-suriname.org/wp-content/uploads/2019/03/census-profile-on-ressort-level.xls   67,072 B
https://www.statistics-suriname.org/wp-content/uploads/2019/03/district-profiel-census.xls           24,064 B
```

Both are OLE2 `.xls`, read with `xlrd`, no wall and no key. §11t's note that ABS's
publication route "was not found" is resolved: the **WordPress REST API** reaches it
(`/wp-json/wp/v2/media`, and `/wp-json/wp/v2/search` for the pages), while the
`/en/census-statistieken/` path 404s. §12's WordPress rule, for the fourth time.

`fetch()` asserts the OLE2 magic `d0 cf 11 e0` rather than the byte count (§5a).

## 3. The file has no district column, and the names are not unique

The ressort workbook is **62 unlabelled columns** in print order. That matters because
**ressort names collide**:

* `Welgelegen` exists in **Paramaribo** and in **Coronie** — ABS disambiguates in the header
  as `Welgelegen (Par'bo)` and `Welgelegen (Coronie)`, but COD carries neither qualifier;
* `Centrum` exists in **Paramaribo** and in **Brokopondo** — ABS writes the second as
  `Brokopondo Centrum`.

A name-only join collides on four units and would put Coronie's people in Paramaribo.

**The sibling district file supplies the grouping.** `district-profiel-census.xls` row 2 is
`Aantal Ressorten` per district: **12, 7, 5, 3, 6, 6, 6, 5, 6, 6** — Paramaribo, Wanica,
Nickerie, Coronie, Saramacca, Commewijne, Marowijne, Para, Brokopondo, Sipaliwini. That sums
to 62 and **consumes the ressort columns exactly, in order**, which is what makes the
district assignment a read rather than a guess. `check()` asserts both the list and the sum.

## 4. An exact partition, in both directions

* the six categories sum to each ressort's own population total, on all 63 columns;
* the 62 ressorten sum to the national column, on all 7 rows;
* gap of **zero** on every cell.

Integers, no rounding, no suppression — the third source here of which that is true, after
Malawi and Zimbabwe. Unusually clean for a file this old.

## 5. What is drawn, and it is a striking map

| | national |
|---|---|
| Christianity | 40.73% |
| Hinduism | **19.93%** |
| *Don't know / No answer* | *15.67%* |
| Islam | **13.45%** |
| Traditional religion + other | 5.79% |
| No religion | 4.42% |

**The Indo-Caribbean belt is the agricultural coastal plain**, and it separates sharply from
the Afro-Surinamese interior and from Para:

| | Hinduism | Islam | Christianity |
|---|---|---|---|
| Jarikaba (Saramacca) | **65%** | 12% | 15% |
| Westelijke Polders (Nickerie) | 60% | 26% | 5% |
| Groot Henar (Nickerie) | 59% | 29% | 8% |
| Nieuw Amsterdam (Commewijne) | 27% | **48%** | 14% |
| Lelydorp (Wanica) | 28% | **47%** | 15% |
| Para Zuid | 1% | 2% | **71%** |
| Brownsweg (Brokopondo) | 0% | 0% | 60% |

Suriname's Muslims are mostly **Javanese**, from the Dutch East Indies indenture, and
Indo-Surinamese; its Hindus are Indo-Surinamese. That two neighbouring polders can be 60%
Hindu and 48% Muslim is what 7,900 people per unit buys.

## 6. What is lost, and it is a lot

**Christianity is one undivided colour.** 200,744 people in a single cell, for a country
whose Christianity is Moravian, Catholic and Pentecostal in different places — the
Evangelische Broedergemeente has been in Suriname since **1735** and is one of the oldest
Protestant missions in the Americas. The 2012 census names all of them separately and this
one does not.

**It is deliberately NOT subdivided using the 2012 national split.** Applying a national
denominational mix to 62 ressorten would invent the entire spatial structure of the result
while the only published number is a national one — §14 rule 1, and the identical refusal
§11r made for Saudi Arabia. A Census 9 build fixes this properly.

**Winti has no cell.** Suriname's Afro-Surinamese religion — sibling to Vodou, Candomblé and
Trinidad's Orisha — sits inside `Traditional Religion +Others` together with indigenous
Amerindian religion, Judaism and the Jehovah's Witnesses. Census 8's equivalent district
label spells the pooling out: **`Andere godsdienst Jodendom Winti Jehova's Getuigen`**.

The cell's geography is at least consistent with Winti being most of it in the interior:
**31% in Brokopondo Centrum, Boven-Suriname and Tapanahony**, which are the Maroon districts,
against 0% in the Nickerie polders. It is still not split (§14.4). See `other.sr` in
`taxonomy/branches.py`.

## 7. The largest non-answer on this map, with two peaks

`Don't know/No answer` is **77,204 people, 15.67%** — ahead of Trinidad's 11.10%. Every
share above is a share of everybody, so a religion's share among people who answered is
about a fifth higher than what is drawn. Never redistributed (§3.5).

**Its geography has two peaks that contradict each other**, by district:

```
  Sipaliwini 22.9   Paramaribo 21.1   Brokopondo 18.5   Marowijne 17.4
  Para 12.5   Nickerie 12.4   Saramacca 8.6   Coronie 8.4   Commewijne 7.3   Wanica 2.6
```

The worst single unit is `Welgelegen (Par'bo)` at 33%; Albina and Tapanahony are at 29%.
A remote-enumeration explanation covers Sipaliwini and not the capital; a refusal explanation
covers the capital and not the interior. ABS publishes no analysis and none is invented.

## 8. Boundaries and placement

See `sources/sr_geo.md`. Short version: COD-AB ADM2 is the ressort tier exactly (62 for 62),
the join is on **(district, ressort)** with four spellings resolved **by elimination rather
than by an alias table**, and placement is Kontur's 400 m grid because Suriname is 163,820
km² with ~90% of its people on the coastal strip.

## 9. Ethics (§14)

Suriname's religion question is voluntary and ordinary and no group in it is persecuted by
the state today. Two things are still worth having written down:

* **Winti was criminalised in Suriname until 1971.** Its absence from this map as a named
  religion is a data limitation, not a judgement, and §6 says so where a reader will see it.
* **7,900 people per unit is fine, and Galibi has 43 modelled people.** The Kalina (Carib)
  village area is a small, identifiable indigenous community. What is drawn there is the
  census's own published composition at the census's own published tier — nothing is
  modelled finer than ABS published (§14.3) — and the map says the grain in the legend.

## 10. Not done

* **Subdividing Christianity or the traditional/other cell** from 2012 national figures.
  §14.4, §6 above. This is the single biggest improvement available and it needs a source,
  not a technique.
* **A Census 9 build.** The moment ABS publishes religion by district or ressort from the
  2024–25 census, this file should be replaced rather than supplemented — the categories
  will be better and the vintage twenty years newer.
* **§3.4 rebasing** 2004 counts onto a newer population. Suriname's ressort boundaries look
  stable (COD's 2017 ADM2 is 62 units and joins cleanly), so a rebase is feasible if a
  ressort-level population for 2012 or later ever appears; it would fix the level without
  fixing the categories.
