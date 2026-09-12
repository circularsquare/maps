# North Macedonia — SSO, Попис 2021

Wired 2026-09-04. 1,836,713 people, 80 municipalities, 13 categories.

| | |
|---|---|
| source | Државен завод за статистика, Census 2021, PxWeb table **T1012P21** |
| basis | `self_id` |
| geography | 80 општини (municipalities) |
| categories | 13 plus the universe total |
| drawn | **1,701,595 people, 92.6%** |
| licence | SSO open data, free reuse with attribution |

**The cheapest country in the project so far.** One PxWeb table, one API call per language,
boundaries already on disk, and nothing needed allocating. §12's "try a PxWeb API before
anything else" is the whole story: it was found by walking
`/pxweb/api/v1/en/MakStat` and took minutes, against Slovakia's counts which are still
not found after three sessions.

---

## 1. Finding it — and why the first walk missed it

A keyword walk of the MakStat tree found **nothing**, and the reason is worth carrying: the
walk was depth-limited and the religion table sits five levels down, under

```
Popisi > Popis2021 > NaselenieVkupno > NaseleniePopis2021 > EtnoKulturniKarakteristiki > T1012P21
```

Two tables in that folder are `T1008P21` (ethnicity) and `T1015P21` (mother tongue); religion
is the third. A separate `NaselenieSet` folder carries a **national-only** religion table,
`T1010P21`, and stopping at that one would have produced a country with no geography at all.
**In a PxWeb tree the same variable often appears at several geographies in different
folders; find them all before choosing.**

## 2. The ceiling is 80 units, and going below it is forbidden rather than hard

SSO publishes ethnicity **by settlement** — `T1503P21`, roughly 1,700 units — and religion
only by municipality. That asymmetry is a live temptation, because religion and ethnicity
here are near-collinear and a settlement-level religion map could be produced by pushing the
municipal religion totals down the settlement ethnicity table.

**That is precisely what spec §14.4 forbids**: never model at a finer resolution, or a
stronger claim, than the source publishes its magnitude at, and never estimate religion from
ethnicity. It would also destroy the only interesting thing the map has to say — a map built
that way could not show a Macedonian-speaking Muslim village or an Albanian Catholic one,
because it would have assumed them away. 80 municipalities, median about 12,000 people,
comparable to Estonia's 79. That is the map.

## 3. One category in nine is not a religion, and it is 7.2%

`Лица за кои податоците се превземени од административни извори` — persons whose data were
taken from administrative sources — is **132,260 people, 7.20%**.

The 2021 census enumerated part of the resident population from registers rather than in
person, and those records carry no religion. It is a **coverage residual wearing a
category's clothes**. Reading it as irreligion would multiply North Macedonia's atheists
fifteen-fold; reading it as a refusal would be wrong too, because refusals are their own
category (`Не се изјаснил`, 1,964) and so is `Непознато` (894). All three are excluded and
named in `taxonomy/mk2021.py`.

It is not evenly spread — it is a property of how each municipality was enumerated — so it
is the one number to check before believing any municipal share on this map.

## 4. `Orthodox` and `Christians` are one thing split by geography, not two things

The census offers both `Православни` (847,390) and `Христијани` (242,579), and 13.2% is a
very large "Christian, unspecified" by the standards of any other country here.

Measured off the drawn map, **the choice between the two labels is regional**:

```
                     west of 21.2E     east of 22.0E
  Orthodox                26.5%            65.5%
  Christian (unspec.)      7.7%            24.7%
```

and in individual eastern municipalities the unspecified answer is the majority one —
Росоман 76.4%, Македонска Каменица 70.4%, Ранковце 67.8%. Orthodox + Christians is 59.3%
nationally, close to the combined Macedonian, Serb and Vlach share of the population.

**So the two nodes must be read together.** Drawn apart — and they are drawn apart, because
inventing a merged category would be worse — they put a hard colour boundary through eastern
Macedonia that reflects how people filled in a form. `note_public` says so in as many words;
this is the most likely way for a reader to be misled by this country.

## 5. What the map shows

- **The northwest is Muslim and the east is Orthodox**, and the line is sharp. Measured:
  67.2% Muslim west of 21.2°E against 8.1% east of 22.0°E. Aračinovo, Bogovinje, Želino,
  Lipkovo, Plasnica are 99.8–100.0% Muslim; Centar Župa and Plasnica are the Torbeš
  (Macedonian-speaking Muslim) municipalities and look identical to the Albanian ones on
  this map, which is the clearest illustration of why §14.4's rule matters.
- **No jurisdictions anywhere.** The census asks for a religion, not a church. 847,390
  Orthodox arrive with no patriarchate attached — Macedonian, Serbian and Vlach in one cell
  — and 590,878 Muslims with no school, so the Bektashi of Tetovo are invisible. This is a
  shallower table than Croatia's and there is no second table to deepen it.
- **Irreligion is 0.48%**, among the lowest on the map.

## 6. A canonicity question that the calendar decides

`Православни` is mapped to `christianity.orthodox`, the parent, and **not** to
`christianity.orthodox.canonical`, where `hr2021.py` files Croatia's identically-bare
`Pravoslavci`. The Macedonian Orthodox Church – Ohrid Archbishopric had been in schism since
1967 and was recognised by the Serbian Patriarchate and Constantinople in **May 2022** —
months *after* the September 2021 census. Filing 847,390 people as canonical asserts
something untrue on the date they were asked; filing them as `.other` asserts the reverse
about a church that is canonical now. The parent says what the source says. Full note in
`taxonomy/mk2021.py`.

## 7. Not done

- The Bektashi/Sunni split, which no table carries.
- 2002 census religion by municipality exists in the same PxWeb tree
  (`PopisNaNaselenie/PopisOpstini`) and is not read; spec §13 rules out a time slider.
