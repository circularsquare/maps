# Unspecified splits: Germany's Protestants, India's Christians (slug `de-in`)

**Status 2026-10-16: cases 1 (Germany) and 2a (Kerala) approved and BUILT, waiting for the retile**
(`sources/de.md` §12, `sources/in.md` §16, spec §2.7b). Meghalaya, Nagaland and Austria not approved.

Research only, 2026-10-16. Nothing in taxonomy/, sources/, countries or index.html was changed.
Read first: `taxonomy/de2022.py` (REVIEW, `Evangelische Kirche`), `sources/de.md` §8, `taxonomy/in2011.py`,
`sources/in.md` §9 to §14. Both cases were looked at before and turned down. This file says what changes under
the new rule (a modelled split beats a flat colour that stops at a border) and what new evidence there is.

Ranked by visual payoff:

| # | case | grey now | proposal | confidence |
|---|---|---:|---|---|
| 1 | Germany, `christianity.protestant` | 19.1M, 23% of Germany, every Gemeinde | by Landeskirche territory: Lutheran / United / Reformed | high on counts, the node for "United" is her call |
| 2 | Kerala, `christianity` remainder | 2.21M of 6.14M Christians | KMS district rows: Oriental, Mar Thoma, CSI, Pentecostal | medium-low inside the state, high on the state totals |
| 3 | Meghalaya, `christianity` | 2.21M, the whole state | church rolls, two regions (Garo Hills, Khasi-Jaintia) | medium; the roll year needs choosing |
| 4 | Nagaland | 1.74M | Catholic share from the Kohima roll; Baptist only as a floor | low; probably leave |

---

## 1. Germany: the EKD's 19.1M

### What the bucket holds

`Evangelische Kirche (öffentlich-rechtlich)` -> `christianity.protestant`, Zensus 2022 at 10,786 Gemeinden, about
19.1M (23.1% of Germany). Destatis defines it as the EKD's twenty Landeskirchen and nothing else. **It does not
include the Freikirchen.** They sit in `unrecorded`, and ESS's `Eine evangelische Freikirche` already draws about
1.17M of them on `christianity.evangelical` by Bundesland (`sources/de.md` §9). SELK (Lutheran, 31,584) is also outside it.

### What the neighbours use

| country | file | Protestant majority body -> node |
|---|---|---|
| Denmark, Norway, Sweden, Finland | dk/no/se/fi2024 | national church -> `christianity.lutheran` |
| Netherlands | nl2014 | PKN (a 2004 union incl. the Lutherans) -> `christianity.reformed.continental.pkn`, plus `.hervormd` and `.gereformeerd` |
| Switzerland | ch2000 | Evangelisch-reformierte Landeskirchen -> `christianity.reformed` |
| Czechia | cz2021 | ČCE (a 1918 Reformed + Lutheran union) -> `christianity.reformed`; the Lutheran churches -> `christianity.lutheran` |
| Poland | pl2021 | Lutheran -> `christianity.lutheran`; Reformed -> `christianity.reformed.continental` |
| **Austria** | at2001 | Evangelische Kirche A.B. und H.B. -> **`christianity.protestant`, also unsplit** |

So Germany and Austria are the two grey patches in a ring of Lutheran and Reformed. Austria is not my case, but
it has the same problem and a similar fix (the church reports A.B. and H.B. separately, and its superintendencies
mostly match Bundesländer). Whoever takes Austria should look at that.

Nodes already on the tree: `christianity.lutheran`, `christianity.reformed`, `christianity.reformed.continental`
("Dutch, German and related Reformed churches"), and `christianity.united` ("churches formed by union across the
Protestant families"; it names Australia's Uniting Church, the United Church of Canada and the CSI).

### New evidence: membership follows the main residence by church law

The 2026-09-07 refusal (`sources/de.md` §8) said a territorial join "would draw which church body governs a place,
not what confession its people hold". The first half of that is true and is the point: **the register counts
members of a body, and by church law that body is fixed by where they live.**

- Kirchengesetz über die Kirchenmitgliedschaft (EKD, 1976, as of 2022) §1 Abs. 2: *"Die Kirchenmitgliedschaft
  besteht zur Kirchengemeinde und zur Gliedkirche des Wohnsitzes des Kirchenmitgliedes."* Wohnsitz is the
  Hauptwohnung under state Melderecht, which is the same register the Zensus read.
- EKD, *Kirchenmitgliederzahlen Stand 31.12.2022* (corrected edition, July 2024), Table 1 footnote 3: membership
  across Landeskirche borders in special cases *"würde saldiert weniger als 0,14 ‰ der Mitglieder ausmachen"*.

So a Zensus `Evangelisch` count in a Gemeinde is, to within 0.014%, members of the Landeskirche whose territory
holds that Gemeinde. Except for one church (below), this is a lookup, not a model. The node then says what that body
is (Lutheran, United or Reformed), the same claim `nl2014` makes for the PKN and `cz2021` for the ČCE. The finer
point, that a Rhineland Union congregation may still be Lutheran or Reformed in its own confession, is true and is
the same caveat the PKN carries.

### Source: EKD members per Landeskirche, 31.12.2022

`https://www.ekd.de/ekd_de/ds_doc/Ber_Kirchenmitglieder_2022_r.pdf` (open, 2 MB; copy in the scratchpad at
`.../scratchpad/de-in/dl/ekd2022.pdf`). Table 1 is per Gliedkirche, with the population of each territory; Table 2
is per Bundesland. National total 19,152,827. Downloads for other years: `https://www.ekd.de/kirchenmitgliederzahlen-downloads-44413.htm`
(2023 to 2025 also as .xlsx).

| confession (by body) | Landeskirchen, members 31.12.2022 | total | share |
|---|---|---:|---:|
| Lutheran | Bayern 2,143,233; Hannover 2,302,547; Württemberg 1,821,266; Nordkirche 1,772,953; Sachsen 610,503; Oldenburg 371,688; Braunschweig 293,979; Schaumburg-Lippe 45,764 | 9,361,933 | 48.9% |
| United (uniert) | Rheinland 2,266,794; Westfalen 2,001,009; Hessen und Nassau 1,361,322; Baden 1,035,524; Berlin-Brandenburg-schlesische Oberlausitz 833,846; Kurhessen-Waldeck 730,478; Pfalz 456,498; Bremen 163,710; Anhalt 26,236 | 8,875,417 | 46.3% |
| Reformed | Evangelisch-reformierte Kirche 159,309; Lippe 140,313 | 299,622 | 1.6% |
| Mitteldeutschland (EKM), see below | 615,855 | 615,855 | 3.2% |

### Proposal

1. **Build a Gemeinde -> Landeskirche table** (10,786 rows; most Länder resolve by Kreis). No open boundary file
   exists (OSM has two Protestant `religious_administration` relations, `sources/de.md` §8; no geodata found).
   Gemeinde-level work is needed where Landeskirche borders follow pre-1866 state lines through modern Kreise:
   Baden/Württemberg (with Hohenzollern), Hessen (EKHN / EKKW / the Rhineland's Wetzlar area), Rheinland-Pfalz
   (Rheinland / EKHN / Pfalz), Saarland (Rheinland / Pfalz in the Saarpfalz), Niedersachsen (Hannover / Braunschweig
   / Oldenburg / Schaumburg-Lippe), Bremen (Bremerhaven is Hannover's), Sachsen-Anhalt (EKM / Anhalt), Sachsen (Sachsen /
   EKBO's Görlitz area), Thüringen (Schmalkalden is EKKW's). Grafik 2 in the EKD report maps the territories, including
   the small exclaves.
2. **Check it twenty times.** Sum the Zensus count by territory and compare with Table 1 (Zensus is May 2022, EKD
   December 2022, so expect every ratio near 1.01 to 1.02). A misassigned Kreis shows up as a miss in two
   neighbouring churches. Table 2 by Bundesland is a second check. This is the "can something contradict it" test,
   and it is strong.
3. **Nodes:** Lutheran bodies -> `christianity.lutheran`. ERK and Lippe -> `christianity.reformed.continental`.
   United bodies -> **her call** (below). Rows are `derived` (spec §2.7a: split inside a column the Zensus counted),
   so "inferred dots: not shown" rolls them back to `christianity.protestant`.
4. **The Reformed church inside Hannover.** EKD: *"Beiden Kirchen können Bevölkerungszahlen nicht direkt zugeordnet
   werden."* The ERK's own 2010 report (`esklum.reformiert.de/.../april_2010_-_bericht_des_mdg_-1_.pdf`) gives about
   70,000 members in Ostfriesland, about 60,000 in Grafschaft Bentheim and Emsland, and about 50,000 scattered. Scaled
   to 2022's 159,309, that is roughly 62k, 53k and 44k. Put the first two inside those Kreise in proportion to their
   Protestants, taken out of Hannover's count there. Leave the scattered third on whichever body's territory it sits
   in. This is the only modelled step. Leaving the ERK out entirely files 0.8% of the bucket as Lutheran, but it would
   paint the Reformed heartland of Leer, Emden and Bentheim Lutheran, which a German reader would notice.
5. **EKM** (615,855) belongs to both the Lutheran VELKD and the Union UEK. It was formed in 2009 from Lutheran
   Thuringia and the Union Kirchenprovinz Sachsen, and its congregations keep their own confession. Options: (a) split
   by former church (Thüringen -> Lutheran, Sachsen-Anhalt part -> United), which follows a real line and is a lookup
   like the rest; (b) all Lutheran; (c) all United. I'd pick (a).

### Decisions only she can make

- **Where the United churches go (46% of the bucket).** (a) `christianity.united`: fits the node's definition
  literally (Prussian Union 1817, Pfalz 1818, Baden 1821). The 2026-09-07 rejection was about the EKD as a whole,
  a federation, which is a different thing from a single Union church. (b) `christianity.reformed`, following
  Czechia's ČCE and the Dutch PKN. That overstates the Reformed side, since the Prussian Union congregations were
  mostly Lutheran. (c) Leave the United half on `christianity.protestant` and split out only Lutheran and Reformed:
  half the grey goes, and the line between the two parts is the Prussian Union border. I'd choose (a). It also
  makes the Prussian Union border visible, which is a real historical line.
- **Reversing her own 2026-09-07 call.** The new argument is the membership law. The old counter-argument
  (governing body versus personal confession) still applies to Union congregations and to Lippe's Lutheran
  minority. This is the same claim `nl2014` and `cz2021` already make.

---

## 2. India: Christians outside the drawn churches

### What is drawn now (`in_split_christian.py`, `sources/in.md` §9 to §14)

9.71M of 27.82M Christians (34.9%) are on a church node: Kerala's Catholics by district (KMS, Zachariah WP468
Table 6), Mizoram's ten churches from state rolls, and Pew's Catholic (39.0%) and Baptist (South 13.0%) shares in
the East and South. The rest are on `christianity`. The Northeast outside Mizoram was drawn and then withdrawn on
2026-09-15, because one Pew share fitted no state. Kerala's St Thomas churches were searched and left undrawn on
2026-10-03.

Nodes available: `christianity.catholic(.latin/.eastern)`, `christianity.oriental` with `.syriac` (Jacobite)
and US leaves `.malankara-orthodox` and `.marthoma` (to be promoted, spec §2.7), `christianity.united` (names the
CSI), `christianity.pentecostal`, `christianity.plymouth` (Brethren), `christianity.baptist`,
`christianity.reformed.presbyterian`. The US legend already shows Oriental Orthodox.

### 2a. Kerala (2.21M grey of 6.14M Christians)

**Size of each church, two survey years, state level** (shares of Christians):

| church | KMS 2008-14 (WP468 Table 6, Kerala row) | KMS 2023 (IIMAD draft, Table 2.10) | x 6.14M |
|---|---:|---:|---:|
| Jacobite | 7.9 | 7.7 | ~485k |
| Orthodox | 8.0 | 7.7 | ~491k |
| Mar Thoma | 6.6 | 5.3 | ~405k |
| CSI | 4.5 | 3.6 | ~276k |
| Pentecost / Church of God / Brethren | 3.5 | 5.3 | ~215k |
| Dalit Christian, Others | 8.5 | 7.1 | stays grey |

The state totals agree across the two rounds, which is the second-year test.

**New finding from re-running `sources/in_osm_churches.py`** (read-only; it prints a grouping §14 did not report):
Jacobite + Orthodox + Mar Thoma together rho +0.37 (p 0.19), against +0.06 for Jacobite + Orthodox. Almost all
the disagreement is in the six Malabar districts, where Christians are few: Malappuram, Kozhikode, Kasaragod and
Palakkad give the worst cells (Jacobite + Orthodox 0.0% of places against 16.0% in Kasaragod; Mar Thoma 0.7%
against 22.3% in Kozhikode). Thiruvananthapuram is the one bad southern cell (1.4% against 13.7%). In the eight
Travancore and Cochin districts, the order mostly holds.

**Why one Kerala-wide share does not work:** it does not fit next to the district Catholic shares that are
already drawn. Thrissur is drawn 88.5% Catholic, which leaves 11.5%, while the state's named non-Catholic churches
add up to 30.5%. Any non-Catholic split has to divide each district's own non-Catholic remainder, and the only source
that gives the mix inside a district is Table 6.

**Proposal:**
- For each district, split the census Christians not already drawn Catholic using Table 6's non-Catholic mix,
  rescaled to that remainder:
  - Jacobite + Orthodox -> `christianity.oriental`. One node absorbs the Jacobite/Orthodox label swap the witness
    finds (both are Oriental Orthodox, two factions of one church).
  - Mar Thoma -> `christianity.oriental.marthoma` (promote the ASARB leaf), or onto `christianity.oriental`.
  - CSI -> `christianity.united`.
  - Pentecost/Church of God/Brethren -> `christianity.pentecostal`. It passes best (+0.89), but the row includes the
    Brethren, who are not Pentecostal (they would belong on `christianity.plymouth`).
  - Dalit Christian and Others stay on `christianity`.
- Malabar (Palakkad, Malappuram, Kozhikode, Wayanad, Kannur, Kasaragod): use **one pooled non-Catholic mix**,
  weighted by census Christians, instead of each district's row. The historical reason is that Malabar's
  Christians are mostly 20th-century settlers from Travancore. Be honest that this rule was picked after seeing
  the witness; §14 refused a south-only rule on that ground.
- Result: Kerala's grey goes from 36% of its Christians to about 8%.
- What could contradict it: KMS 2023's state shares (they agree), and OSM place shares by district (printed in
  the build, as now).

### 2b. Meghalaya (2,213,027 Christians in 2011, all grey)

The churches follow the two hill regions, and their rolls sum to roughly the census count:

| region (2011 districts) | church | figure | year | source |
|---|---|---:|---|---|
| Garo Hills (West, East, South Garo Hills) | Garo Baptist Convention (baptised and unbaptised) | 500,560 | undated | Wikipedia, *Christianity in Meghalaya* |
| | Catholic Diocese of Tura | 239,511 (2010); 317,645 (2020) | 2010 | catholic-hierarchy.org/diocese/dtura.html |
| Khasi-Jaintia (West Khasi, East Khasi, Ri Bhoi, Jaintia Hills) | Khasi Jaintia Presbyterian synods (Mihngi 294,320, Sepngi 370,764, Ri Bhoi 70,510) | 735,594 | 2018 (750,989 in 2015) | Wikipedia; Sepngi's own site kjpss.org gives 370,764 = 174,406 communicants + 196,358 others |
| | Catholic: Shillong 342,169, Nongstoin 164,334, Jowai 111,930 | 618,433 | 2020 | Wikipedia (unsourced there; check catholic-hierarchy) |
| | Church of God (Mylliem) | ~100,000 | undated | Wikipedia |
| | Church of North India, Diocese of North East India | ~50,000 | undated | Wikipedia |

**Proposal:** one share per region, each church's roll over the region's summed rolls, applied to census Christians.
Nodes: Baptist -> `christianity.baptist`; Presbyterian -> `christianity.reformed.presbyterian`; Catholic ->
`christianity.catholic.latin` (Latin dioceses, as Mizoram's roll is filed); CNI -> `christianity.united`; Church of
God -> check its lineage (the Meghalaya church comes from the Anderson, Indiana holiness movement) or leave it grey.
Rows `derived`, as Mizoram's are.

**Caveat, and the reason to pick roll years carefully:** Meghalaya's Catholic rolls grew from 588,465 (2004, the
figure `in.md` §11 used) to 945,145 (2020), while the Presbyterian figure fell slightly from 2015 to 2018. Read
Catholic rolls for 2010 to 2012 from catholic-hierarchy's per-diocese pages (Tura 2010: 239,511; the other three
were rate-limited, HTTP 429). A region's rolls should sum to 0.9 to 1.2 times its census Christians, and that check
should run as part of the build. The Khasi-Jaintia region is about half Presbyterian and 40% Catholic; the Garo
Hills about 60% Baptist and 40% Catholic.

### 2c. Nagaland (1,739,651 Christians)

The Nagaland Baptist Church Council reports **baptised members**: 454,349 (2007), 610,825 (about 2016, BWA),
716,495 (current Wikipedia infobox). Baptists baptise adults, so children are left out and these are floors, not
community counts. Turning them into a community count needs a ratio nobody publishes for the NBCC. (Meghalaya's
Sepngi synod is 47% communicants, but that is a different church.) The Catholic Diocese of Kohima's roll (50,873 in
2004, about 3% of Christians) can be drawn. **Proposal:** draw Catholic only, or Catholic plus Baptist at the
baptised floor (about 35 to 40%), leaving the rest grey. Her call; I'd leave Nagaland until a community figure turns up.

### 2d. Not pursued here
- Manipur (1.18M), Arunachal, Assam, Tripura: no church figures by state found in this pass. The Wikipedia pages
  give none. CBCNEI reports 1.2 to 1.56M members across six conventions with no per-convention split found.
- Tamil Nadu's CSI: no diocesan membership found. Pew's public file recodes CSI and CNI into "other" (`in.md` §10).
- Pew's file has nothing below region, so it cannot give a Nagaland or Meghalaya share.

---

## Sources

- EKD, *Kirchenmitgliederzahlen Stand 31.12.2022*, https://www.ekd.de/ekd_de/ds_doc/Ber_Kirchenmitglieder_2022_r.pdf ; index of editions https://www.ekd.de/kirchenmitgliederzahlen-downloads-44413.htm ; 2025 figures https://www.ekd.de/statistik-kirchenmitglieder-17279.htm
- Kirchenmitgliedschaftsgesetz, e.g. https://www.kirchenrecht-ekd.de/pdf/18038.pdf
- Evangelisch-reformierte Kirche, 2010 report, https://esklum.reformiert.de/files/reformiert.de/Bilder/artikelbilder/pdf/april_2010_-_bericht_des_mdg_-1_.pdf
- REMID, Protestant bodies in Germany, https://remid.de/info_zahlen/protestantismus/ (BEFG 73,956; BFP 64,800; EmK 42,565; FeG 40,904; SELK 31,584; NAK 307,291; Russian-German Baptists and Mennonites about 290,000 in 2007; VEF 297,000 in 2020)
- K.C. Zachariah, CDS WP468 (2016), already at `data/raw/in/zachariah_2016/WP468.pdf`; KMS 2023 draft, `iimad.org/wp-content/uploads/2024/06/KMS-2023-Report.pdf`
- https://en.wikipedia.org/wiki/Christianity_in_Meghalaya ; https://www.kjpss.org ; https://www.catholic-hierarchy.org/diocese/dtura.html
- https://en.wikipedia.org/wiki/Nagaland_Baptist_Church_Council ; https://en.wikipedia.org/wiki/Council_of_Baptist_Churches_in_Northeast_India
