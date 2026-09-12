# Paraguay — DGEEC, Censo Nacional de Población y Viviendas 2002, variable P17

**Drawn 2026-09-08.** 3,892,603 people aged 10 and over, 229 census districts drawn on 224
polygons, **54 religion categories** — the longest list on this map, ahead of Austria's 31,
which `queue.md` called "the deepest undrawn religion list in the world".

Files: `sources/py.py`, `sources/py_geo.py`, `sources/py_grid.py`, `taxonomy/py2002.py`,
the `py` entry in `countries.py`, and `other.py` in `taxonomy/branches.py`.

---

## 1. Why four earlier passes closed it, and what they were actually closing

`sources.md` §9k left Paraguay **unresolved**; §11t resolved the oracle row (2002, twelve
categories, an exact partition) and never opened the office; `queue.md` carried that forward
as *"§11t resolved the row and never opened the office"*.

Every one of those was **true of the published output**. INE's 2002 census library serves 52
national tables and 20 district ones, and religion is in exactly one of them:

    CUADRO P11   Población de 10 años y más por grupos de edad, según área
                 urbana-rural, sexo y religión, 2002
                 -> national, urban/rural, FOUR named categories:
                    Católica / Evangélica o protestante / Religión indígena / Otras

Nine district tables (`P01`–`P09 distrital.pdf`, downloaded and scanned) carry **no religion
at all**. So on the printed evidence Paraguay is a four-category country with no subnational
religion, which is roughly what the queue said.

**The microdata tabulator is a different thing and it is open.** It has the variable itself,
at district level, with all 54 codes.

## 2. The route

    https://prod.redatam.org/redpry/                     index; names its OWN cgi dir
    /binpry/RpWebEngine.exe/Portal?BASE=CPV2002          mints a session, returns 4 iframes
    /binpry/RpWebStats.exe/Frequency?...&ITEM=FREQPOB    the form
    POST it with MODE=RUN, ROW=PERSONA.religion, AREABREAK=DISTRITO, SELECTION=ALL

Two bases are mounted, `CPV2002` and `CPV2022`. Only 2002 has religion; §3 below.

**§11x's shared-host trap applies and was avoided**: `BASE=` is not namespaced on
`prod.redatam.org`, so the cgi path must be read from *Paraguay's own* index page. It is
`/binpry/`. Reading `CPV2002` off one country's page and requesting it under another's cgi
dir returns a different country's census with a 200 and no warning.

### 2a. The one character that hid it, and the browser launched before the regex was re-read

The portal response is **4,611 bytes and looks empty** — no table, no content, a shell with a
sidebar and a footer. A frame-walker written for the older R+SP servers (Venezuela's, probed
the same day, §4) finds nothing in it and reports the deployment dead.

It is not dead. It emits **`<iframe>`** where the R+SP servers emit **`<frame>`**, and the
regex was `<frame[^>]+src=`. One character.

A headless Chrome was launched over CDP to prove the page was broken, and what it showed was
four iframes loading `/redpry/tempo/<session>/~tmp_*.htm` perfectly well. **The browser was
the right instinct and the wrong conclusion**: it proved the server worked, and the bug was in
the reader. `<i?frame` is what `sources/py.py` uses now.

### 2b. Two failures that read like a dead host and are not

* `FORMAT=HTML` and `Submit=Ejecutar` are **not optional**. Without them the engine answers
  **500** with `Número de Error 1 en función : setOutputFormats — Opción inválida o no
  seleccionada`. A 500 from a census server reads as "it is down".
* The `Portal` call must come first. It mints the per-session scratch files the form and its
  output live in; posting the form cold also 500s.

A nonsense `BASE` gives a 500 too, which is how `CPV2002` and `CPV2022` were confirmed real.

## 3. The 2022 census has no religion question, confirmed from its own dictionary

INE runs a **second, newer instance** at `redatam.ine.gov.py` — "Redatam Online", a SPA whose
`config.js` names an unauthenticated REST backend:

    /redxrest/v1/databases          -> cen22 = /usr/redatam/db/cen22/cpv-pry-2022.rxdb
    /redxrest/v1/dictionary/cen22   -> 632 KB of JSON, 155 variables

**Zero of the 155 are religion.** The only hit for `religi` anywhere in it is the value label
`Comunidad religiosa`, which is a *collective dwelling type*. That is the same false positive
Venezuela's census produces (§4), and it is worth knowing by sight.

`prod.redatam.org` does **not** run the redxrest backend; nor do the Venezuelan, Colombian,
Peruvian, Nicaraguan or Dominican hosts. Paraguay's own host is the only one found with it.

## 4. The rest of South America, checked the same day

Anita asked for Colombia, Venezuela or Ecuador. All three are dead, at different depths, and
the evidence is now on the record rather than assumed:

| | finding |
|---|---|
| **Ecuador** | Already closed correctly. `prod.redatam.org/redecu/` serves six bases (CPV1990, CPV2001, CPV2010, CPV2022 and two recodes) with 98–100 person variables and **no religion**. |
| **Venezuela** | Was marked *unchecked* in `sources.md`. It is now checked and dead. `redatam.ine.gob.ve` is live (the root serves a XAMPP default page, the app is at `/redatam/` → `/vencgibin/`), base `CPV2011`, and its person entity has **67 variables, none religion**. INE's own 24-page `Meta_Persona.pdf` defines every person question and has none either. One base only; the "eval" instance is the same `CPV2011`. |
| **Colombia** | The census negative is confirmed from DANE's own microdata dictionary rather than from reputation: **CNPV 2018 has 116 variables and Censo 1985 has 71, none religion-shaped**, and GEIH 2023–2026 has 760 variables with none. Fourteen DANE studies match the word `religion`; in every one it is a question about receiving help from religious organisations, book-trade subsectors, or similar. The **only** religion variable in the whole catalogue is in ELCA (Universidad de los Andes' longitudinal panel, ~10,000 households, five regions), which is not an official source and is far coarser than anything drawn here. |

NADA's variable API keys on **IDNO, not the numeric study id**; the numeric one returns
`{"status":"failed","message":"IDNO-NOT-FOUND"}`, which reads exactly like "this study has no
variables" and is how the first Colombian pass produced fourteen false negatives.

## 5. Geography — the vintage gap goes both ways

COD-AB Paraguay (`cod-ab-pry`) ships **DGEEC's own 2020 shapefile**, and DGEEC is INE's former
name and the office that ran the 2002 census, so `ADM2_PCODE` is `PY` + the census's own
four-digit `DDdd` code. **223 codes match unchanged.** The two differences:

* **Asunción is 6 census districts and 1 polygon.** The capital is tabulated as La
  Encarnación, Catedral, San Roque, Lambaré, Recoleta and Santísima Trinidad; every boundary
  source treats it as one unit. Aggregated onto `0000`. A real loss of resolution over
  512,000 people, stated in `grain`.
* **26 districts were created after 2002** and are dissolved back into a 2002 parent by
  longest shared boundary within the department (`PARENT` in `sources/py_geo.py`, each with
  its margin).

Two of those are not judgements at all: **Boquerón had exactly one district in 2002**
(`1602`), so Filadelfia and Loma Plata can only have come from it; and Alto Paraguay's two
children share 322 km and 308 km with one parent and **zero** with the other.

**Eleven are close calls** (runner-up above 60% of the winner): `0108`, `0109`, `0110`,
`0111`, `0220`, `0418`, `0522`, `0611`, `1020`, `1021`, `1413`, `1509`. **No count depends on
any of them.** The counts are per 2002 district whatever polygon they are drawn in, so a wrong
call moves dots between two adjacent districts of the same department.

**`1503` Pto. Pinasco is the one to look at.** It counted 2,702 people aged 10+ and its
reconstructed unit holds 40,056 in Kontur 2023, a ratio of **14.8x against a national 1.80**.
Either the central Chaco filled up that fast, which it partly did, or `1507` Tte. Irala
Fernández was carved from somewhere else. Drawn as assigned; `sources/py_grid.py` prints the
ratio every run so it stays visible.

## 6. The checks

**The partition is exact three ways** (`sources/py.py check()`):

    every district's categories sum to its own printed Total ....  229 / 229
    national total .............................................  3,892,603
      = UNSD Demographic Yearbook table 28, Paraguay 2002, to the person
    religion universe + No Aplica ..............................  5,163,198
      = the 2002 census population, to the person

**The department run is an independent witness on the same engine**: re-running the frequency
with `AREABREAK=DEPTO` gives 18 departments whose totals the district sums reproduce exactly.

**Kontur is the witness that shares no lineage with either source.** r = **0.959** on 224
districts against a best of 0.488 over 500 shuffles. The vintage gap is **21 years**, the
widest on this map.

**And the tail lands where it should.** The four districts with the highest Buddhist shares in
Paraguay are La Paz (4.6%), Pirapó (2.9%), Yguazú (2.0%) and La Colmena (1.5%) — which are the
four Japanese agricultural colonies, planted from 1936. `Reyukai` (72 people nationally) and
`Sintoismo` (30) peak in the same four. Nothing in the join knows what a Japanese colony is.

## 7. What the map shows

* **89.6% Catholic**, one of the highest shares here; Paraguarí 96.2%, Cordillera 95.7%, and
  four districts above 99%.
* **Boquerón is unlike any other unit on this map.** One district in 2002, 91,000 km², 30,896
  people aged 10+, and **12.2% Mennonite, 39.5% other evangelical, 10.3% no religion**. The
  Fernheim and Menno colonies, settled from Russia and Canada from 1927.
* **Indigenous religion** is 0.6% nationally, 7.3% of Amambay, 19.7% in Itanará, 13.2% in
  Ypehú.
* **Five syncretic categories** (`Indígena + católica`, `+ anglicana`, `+ evangélica`,
  `+ mennonita`, `+ otras`), 1,478 people, an answer no other source here offers. All drawn as
  `indigenous`; `taxonomy/py2002.py` REVIEW argues it and it is the call most worth a second
  opinion.
* **The largest uncertainty is `Otras - Evangélica`**, 186,107 people (4.78%), sitting at the
  end of nineteen named Protestant bodies rather than in place of them. On
  `christianity.evangelical`; §14.4 rule 1 forbids inventing the split.
* **And in Boquerón that cell is almost certainly Mennonite.** 39.5% against a national
  4.8%, in the one district holding Fernheim, Menno and Neuland, beside only 12.2%
  `Mennonita`. The colonies carry two church bodies and only one is called Mennonite in
  Spanish; the *Mennonitische Brüdergemeinde* self-describes as *evangélica*. Crediting the
  excess would take the national Mennonite figure from 8,445 to about 19,000, still 0.49%
  of Paraguay. Not done, but it is what the cell hides.
* **Germans in Paraguay are mostly not Mennonites, and they are a different map.**
  `Luterana` (8,849) slightly OUTNUMBERS `Mennonita` (8,445) and shares almost no geography
  with it: San Alberto 7.7%, Naranjal 6.5%, Nueva Esperanza 5.6%, Obligado 4.4%, Hohenau
  3.9%, which are the Itapúa and Alto Paraná colonies settled from Germany and southern
  Brazil from 1900, and which hold essentially zero Mennonites. Nueva Germania, the 1887
  Förster colony, is 3,106 people with 81 Lutherans.
* **Not drawn: `No especificado`**, 37,206 people, 0.96%. Paraguay is 99.04% drawn.

## 8. Still on the table

* **Asunción's six barrios** would come back if a barrio-level boundary layer turned up.
* **`1507`'s true 2002 parent** is a documentable fact (a district-creation law) and would
  settle §5's one bad ratio.
* The engine also serves `CrossTab`, so **religion × ethnicity or × department of birth** is
  one POST away if a question ever wants it. `PERSONA.ETNIA` and `PERSONA.PUEBLO` are in the
  same dictionary.
