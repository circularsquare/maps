# Poland — GUS, Narodowy Spis Powszechny 2021

Ingested and drawn 2026-09-03. Rebuild with `python sources/pl.py --fetch`.

**The second-best source in the project, after ASARB and beside Czechia.** It is the only
other source so far that publishes named churches at its finest geography, and it goes
one step further than Czechia in a way nothing else does: its tail is *congregations*, not
denominations.

---

## 1. The file

One XLSX, 3,479,174 bytes, published 29 Apr 2024.

```
https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/6536/10/1/1/
przynaleznosc_wyznaniowa_-_dane_nsp_2021_dla_kraju_i_jednostek_podzialu_terytorialnego_1.xlsx
```

Saved as `data/raw/pl/przynaleznosc_wyznaniowa_nsp2021.xlsx`.

### The download fails, and it is not the URL

`stat.gov.pl` **serves an incomplete certificate chain** — it omits the intermediate. Both

```
curl: (60) SSL certificate problem: unable to get local issuer certificate
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

are that, not a bad URL, not a proxy, and not bot protection. `certifi`'s bundle does not
fix it because the missing certificate is the server's to send. `pl.py --fetch` turns
verification off **for this one host** and validates the file structurally instead: at
least 3,000,000 bytes, a real zip, and all eight expected sheets present. That is the
§5a rule ("a download that returns HTTP 200 is not a download") applied to a case where
the usual defence is unavailable.

## 2. Eight sheets, and only four of them are useful

| sheet | what | used |
|---|---|---|
| TABL.1 | national, full 7-level classification | no — the hierarchy is documented in §4 |
| TABL.2 | national, **216 named denominations**, flat | **yes** → `country` |
| TABL.3 | voivodeship × group | no — TABL.5 is the same thing with leaves |
| TABL.4 | powiat × group | no — TABL.6 is the same thing with leaves |
| TABL.5 | voivodeship, full 7-level classification | **yes** → `voivodeship`, leaves only |
| TABL.6 | powiat (380), named denominations, flat | **yes** → `powiat` |
| TABL.7 | **gmina (2,477), named denominations, flat** | **yes** → `gmina`, the drawn level |

### The trap: three of the four are flat and TABL.5 is not

TABL.2, TABL.6 and TABL.7 list a level-1 universe row, a level-2 row, and then named
churches with nothing in between. **TABL.5 carries the whole 7-deep classification**, so
`chrześcijaństwo` (L4), `katolicyzm` (L5) and `Kościół katolicki` (L6) all appear as rows
*above* `Kościół katolicki - obrządek łaciński` (L7).

Summing TABL.5 the way the other three are summed counts the Latin rite four times and
returns roughly four times the country. `pl.py` reads GUS's own `Poziom klasyfikacji`
column and keeps only depth 7, which is where every leaf sits — including under `islam`
and `protestantyzm i tradycja protestancka`, which skip the intervening depths entirely.

### The layout is parsed by position, not by label

Which *column* a label lands in is the only thing that says whether it is a universe row
or a denomination. GUS pads two labels with a trailing `w tym:` ("of which") on some
sheets and not others, and embeds newlines inside cells, so `należący do wyznania` arrives
in two spellings. `pl2021.py` normalises the suffix off before lookup; matching label text
without doing that silently drops a 27.6M-person universe row into the unmapped bucket.

## 3. Reconciliation — exact, at all four levels

```
gmina         2,477 units    38,036,118      affiliated 27,601,000
powiat          380 units    38,036,118      affiliated 27,601,000
voivodeship      16 units    38,036,118      affiliated 27,601,000
country           1 unit     38,036,118      affiliated 27,601,000
```

Both figures are GUS's published nationals, to the person, at every level independently.
No suppression, no rounding, no sentinel values. `Polski Kościół Dialogu` has exactly one
adherent in Poland and GUS prints the 1.

## 4. §3.9, and Poland almost does not do it

Every other source splits category depth from spatial depth. Poland splits it *slightly*:

| level | named categories | reaches this share of the affiliated |
|---|---|---|
| gmina | 139 | 99.787% |
| powiat | 198 | 99.970% |
| voivodeship | 204 | 99.997% |
| country | 216 | 100% |

The 77 categories that never reach gmina are **1,648 people in total** — every one a body
below GUS's per-gmina publication floor. So `allocate.py` is not run for Poland: it would
move a rounding error. Poland and Czechia are the only two countries on the map with no
allocation step, and Poland is the only one where that is a *measured* 99.8% rather than a
structural 100%.

### GUS's own hierarchy, for reference

```
chrześcijaństwo
  katolicyzm
    Kościół katolicki              → 5 rites, incl. greckokatolicki and neounicki
    starokatolicyzm                → 10 bodies, incl. both Mariavite churches
  chrześcijaństwo wschodnie (ortodoksyjne)
    prawosławie                    → 3, incl. two Old Believer churches
    chrześcijańskie kościoły orientalne → 4
  protestantyzm i tradycja protestancka → 94, and this is where the congregations are
  nurt badaczy Pisma Świętego      → 4: the Witnesses and three Bible Student bodies
  inne chrześcijańskie             → 19
islam → 6   judaizm → 5   buddyzm → 20   hinduizm → 9
pogaństwo - rekonstrukcjonizm i neopogaństwo → 6
inne religie i wierzenia → 33
```

Two places where this disagrees with `branches.py` are recorded in `pl2021.py`'s REVIEW:
GUS files the antitrinitarian bodies inside Christianity, and it declines to file
`Wyznawcy Słońca` under paganism despite having the group available.

## 5. What the category list actually contains

**The tail is congregations.** About 60 of the 216 are single congregations that registered
as religious associations and got their own row: `Zbór Ewangeliczny "Betel" w Warszawie`
(2), `Kościół Jezusa Chrystusa w Werbkowicach` (5), `Zbór Wolnych Chrześcijan w Jaworznie`
(18), `Warsaw International Church` (24), `Kościół w Radomiu` (30). No other source in the
project goes below the denomination. **A Polish "category" and an American one are not the
same size of thing**, and any cross-country count of categories should say so.

**Tradition and institution are separate rows over overlapping people**, exactly as in
Czechia: `różne afiliacje islamskie (ogólnie islam, muzułmanizm, sunnizm, szyizm itp.)`
sits beside `Muzułmański Związek Religijny`. They aggregate to one node, which is correct;
what must not happen is treating the pair as a hierarchy.

**Positions rather than religions** — `deizm`, `teizm`, `panteizm`, `gnostycyzm` — are
inside `należący do wyznania`, i.e. the respondents offered them as their affiliation.

**Parody answers are large.** `pastafarianizm` is 2,312 people in 29 gminy, bigger than 190
of the 216 categories; `jediizm (religia Jedi)` is 687. Both go to `parody`, per the
precedent Czechia set.

**Poland's own churches are the interesting part.** The Mariavites (`Kościół Starokatolicki
Mariawitów` 12,248 + `Kościół Katolicki Mariawitów` 200) are a 1906 movement around
Feliksa Kozłowska's revelations and the only Old Catholic body anywhere with a Polish
origin; it split again in 1935, which is why there are two. The Slavic native-faith bodies
(`Rodzima Wiara`, `Rodzimy Kościół Polski`, `Polski Kościół Słowiański`) have a real 1930s
lineage rather than being a generic residual.

## 6. The dominant fact: 20.53% refused

| | people | share |
|---|---|---|
| total | 38,036,118 | 100% |
| answered | 30,212,506 | 79.4% |
| — named a religion | 27,601,000 | 72.6% |
| — no religion | 2,611,506 | 6.9% |
| **refused** | **7,807,553** | **20.5%** |
| not established | 16,059 | 0.04% |

The question was voluntary. The refusers are **excluded** rather than drawn (spec §3.5),
so the Polish map shows 30.2M of 38.0M people. This is the same problem as Czechia's 30%,
one size smaller, and it belongs in `note_public` rather than a footnote.

`Nie ustalono` is a **fourth level-1 category** beside Ogółem / Udzielający / Odmawiający,
and it is easy to miss — it appears only on the subnational sheets, never nationally.

## 7. What is drawn

139 categories → 42 taxonomy nodes → **30,138 dots and 24 rings** at 1:1,000.

Poland is 98.3% Latin Catholic among those who named a church, so the map's interest is
entirely in the other 1.7%: Orthodoxy along the Belarusian border, Lutherans in Cieszyn
Silesia, Greek Catholics where the Ukrainian minority was resettled in 1947, the Mariavites
around Płock, and Old Believers in Masuria.

Three branches were added to `branches.py` for Poland:

- `christianity.orthodox.oldbeliever` — the 1653-66 Nikonian split, three centuries before
  and unrelated to the Old Calendarists. spec §R2 names Old Believers as a test case.
- `christianity.biblestudent` — the part of Russell's movement that did not follow
  Rutherford. GUS's `nurt badaczy Pisma Świętego` group holds four bodies, of which one is
  the Witnesses and three are these; filing them together would invert the descent.
- `other.pl` — GUS's own residual leaves.

## 8. Not done

- **Below branch level.** Like Czechia and Brazil, Poland is mapped at branch level per
  spec §2.4, so the viewer shows one Pentecostal node rather than Kościół Zielonoświątkowy
  and Kościół Boży w Chrystusie apart. The source category name is on every row, so
  deepening costs nothing and redoes nothing — and Poland is the best candidate for it,
  because its categories are already named institutions.
- **Warsaw is one polygon.** See `pl_geo.md` §4.
- The `powiat` and `voivodeship` levels are normalised and unused. They exist so that the
  77 sub-floor categories *could* be allocated down; nothing needs it yet.
