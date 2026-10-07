# Austria: Volkszählung 2001, Umgangssprache

Built 2026-10-05 (session edd42a8c-at). Rebuild:

```
python sources/at_vz2001.py --fetch    (reads religiondots/data/raw/at/*.pdf if present, else
                                        downloads the ten volumes to data/raw/at/)
python sources/at_vz2001.py            -> data/normalized/at.csv, at_place.csv
python taxonomy/build.py
python tools/check_country.py at
python scatter.py --country at
```

Drawn: 8,032,926 people (everyone; the census imputed non-response) on 405 units, 45 labels on
42 nodes, 8,010 dots and 5 rings. Nothing is left out.

## 1. Source and vintage

The 2001 census is the last count of language in Austria. The 2011 and 2021 censuses are
register-based and carry no language; the Mikrozensus has no language item (coverage sweep).
Source: *Volkszählung 2001, Hauptergebnisse I*, one PDF volume per Land plus the Österreich
volume, `https://www.statistik.at/fileadmin/publications/Volkszaehlung_2001__Hauptergebnisse_I_-_<Land>.pdf`,
the same files religiondots read for religion (Tabelle 4). statistik.at omits a TLS intermediate,
so the fetch disables verification, as religiondots' does.

**The question** (Erläuterungen, section 8): the language or languages usually spoken in private
life (family, relatives, friends). Foreign-language knowledge was not to be given; Statistik
Austria says some did anyway. A double answer is folded onto its non-German half ("Slowenisch"
includes "Deutsch und Slowenisch"), so the columns partition the population: **"Deutsch" is
German alone**. Nationally 88.6% gave German alone, 8.6% German and another, 2.8% only another
(Übersicht 5). `how` says "everyday language (Umgangssprache)": it is closer to home language
than to mother tongue.

## 2. The tables and the grain

| table | what | grain |
|---|---|---|
| 5 | Insgesamt, Deutsch, Burgenland-Kroatisch, Kroatisch, Romanes, Slowakisch, Slowenisch, Tschechisch, Ungarisch, Windisch, Sonstige; whole population, then Austrian citizens | **Gemeinde only in Burgenland and Kärnten**; Wien by Gemeindebezirk; the other six Länder by Politischer Bezirk |
| 14 | 45 languages x (total, Austrians, of whom born in Austria, foreigners) | Land |
| 2 | citizenship: Austria, EU-15 (Germany, Italy), ex-Yugoslavia (Bosnia, Yugoslavia, Croatia, Macedonia, Slovenia), Poland, Romania, Switzerland, Slovakia, Czechia, Turkey, Hungary, USA, other | Gemeinde (Wien: Zählbezirk, Gemeindebezirk) |

The coverage sweep's "Umgangssprache by municipality" holds only for Burgenland and Kärnten. A
Gemeinde table for the other Länder was looked for and not found: the archived "Ein Blick auf die
Gemeinde" profiles (Wayback CDX of statistik.at/blickgem/: series blick1-8, ae, az, fa; the
2006 ones checked carry no language) and data.gv.at (search "Umgangssprache", "Volkszählung":
nothing on language). **Not searched: STATcube** (may hold VZ 2001 by Gemeinde; worth a look if
someone has an hour), and the Länder statistics offices' own 2001 releases (Vorarlberg republished
its religion table as .xls; a language one may exist).

So the units are 303 Gemeinden (Burgenland 171, Kärnten 132, including the four Statutarstädte,
whose Gemeinde row is minted from the Bezirk row as religiondots does), 23 Wien districts and 79
Politische Bezirke (NÖ 25, OÖ 18, Steiermark 17, Tirol 9, Salzburg 6, Vorarlberg 4), 19,800
people on average.

## 3. Sonstige: split by citizenship inside each Land

Sonstige is 8.0% of Austria (Wien 20.1%, Vorarlberg 12.1%) and holds Turkish (183,445) and
Serbian (177,320), the second and third languages of the country. Drawing it as `other` would
have hidden them, so it is split. Tabelle 14 gives each of its 36 member languages per Land,
measured. Per Land and per citizenship block (Austrians, foreigners), an IPF fits the unit x
language table to both printed margins: each unit's Sonstige (Tabelle 5) and each language's Land
total (Tabelle 14). The seed is, 90%, the unit's Tabelle 2 count of the citizenship the language
follows, and 10% the unit's Sonstige (`PROXY` in `sources/at_vz2001.py`):

| language | citizenship |
|---|---|
| Turkish, Kurdish | Turkey |
| Serbian | Yugoslavia (Serbia and Montenegro) |
| Albanian | Yugoslavia + Macedonia (Kosovo Albanians held Yugoslav passports) |
| Bosnian, Macedonian, Polish, Romanian, Italian | their country |
| English | EU-15 less Germany and Italy, plus USA |
| French, Spanish, Portuguese, Dutch, Danish, Swedish, Finnish, Greek | EU-15 less Germany and Italy |
| everything else | "anderer Staat; unbekannt" |

Rows are `derived`. Every Land total of every language is the census's, and so is every unit's
Sonstige; only the split of a unit's Sonstige among languages is borrowed. I treated this as
AGENT_BRIEF §4.4 (people moved within the Land the census counted them in; Germany's Mikrozensus
is the same move at Land grain) and did not file an ask. **If it is read as a count-changing
proxy instead**, the fallback is small: write Tabelle 5's Sonstige per unit as one `other` row
instead of the IPF rows in `sources/at_vz2001.py`; the named columns are unaffected.

**Placement inside a Bezirk** (`countries/at.py`, `_AtWeighter`): each language's dots go to the
Bezirk's Gemeinden by the same citizenship counts (German by Austrian citizens, Croatian by
Croatian, Hungarian by Hungarian, Slovak, Czech, Slovene by theirs; Burgenland Croatian, Romani and
Windisch by population), 90%, and population 10%; inside a Gemeinde by Kontur population. Where
the unit is one Gemeinde or a Wien district it is plain population. Spot check: Dornbirn Bezirk's
7 Turkish dots fall 4 in Dornbirn, 2 in Lustenau, 1 in Hohenems (Turkish citizens 2,528, 2,000,
1,069).

## 4. Checks (all equalities; the script stops on any failure)

- Tabelle 5: the ten columns sum to Insgesamt on every row of both blocks in all nine volumes;
  Gemeinden sum to their Bezirk (Burgenland, Kärnten), Bezirke to their Land; Austrians never
  exceed everyone in a cell.
- Tabelle 14: members sum to the printed group rows ("Sprachen der anerkannten österr.
  Volksgruppen", "Sprachen des ehem. Jugoslawien und der Türkei", ...) and all to Insgesamt;
  Austrians + foreigners = total for every language.
- Tabelle 5's nine named columns equal Tabelle 14's at Land level, both blocks; Tabelle 14's
  other 36 languages sum exactly to Tabelle 5's Sonstige, both blocks, in all nine Länder.
- The nine Länder's Tabelle 14 equal the Österreich volume's, all 45 labels x 4 columns, and sum
  to the census total 8,032,926.
- Tabelle 2: total and Austrians equal Tabelle 5's, unit by unit (405 units); citizenship groups
  sum to foreigners; ex-Yugoslav and Czechoslovak members sum to their groups; foreigners by
  Umgangssprache (cols 24-26) sum to foreigners; Gemeinden sum to Bezirk, Bezirke to Land.
- IPF margins reproduced within 0.01 of a person.
- Vorarlberg prints "Andere Sprachen" (2 people) for every other volume's "Andere Sprachen,
  unbekannt"; aliased.

## 5. Mapping calls (taxonomy/at2001.py has them all)

- **Burgenland Croatian** (19,412): new leaf beside Croatian, Glottolog burg1244 (a Chakavian
  dialect under Serbian-Croatian-Bosnian). The census prints it apart from "Kroatisch" (131,307,
  mostly immigrants from Croatia and Bosnia, 105,487 of them foreign citizens). Orange.
- **Windisch** (568): new leaf beside Slovenian, the census's own box for Carinthian Slovene
  speakers who did not call their language Slovene. No Glottolog entry. Khaki.
- Romanes on `romani.romani`; Statistik Austria suspects Romanian speakers ticked it by mistake.
  It also notes that some people named a foreign language they had learned, English especially
  (said in note_public until the 2026-10-06 text sweep moved it here).
- "Russisch, Ukrainisch, Weißrussisch" (8,446) on Russian, as ch2000 draws the same merge.
- "Indisch" (3,582) on `other` (a country, not a language; cy2021, zm2022). "Philippinisch" on
  Filipino (lu2021). "Chinesisch" on Sinitic (de2023). "Holländisch/Flämisch" on Dutch.
- Cross-family remainders on `other`; "sonstige afrikanische Sprachen" on `africa_other`.
- **English (58,582) is drawn as measured.** The question asked for the private-life language
  and told people not to give foreign languages; Statistik Austria says some of the 33,000
  Austrians naming English probably did anyway. 25,155 foreign citizens named it too. This is not
  the ability-question case of AGENT_BRIEF §2, so it is not folded into German; said in
  `note_public`.

## 6. Colour

Burgenland Croatian orange (0.72 0.16 55) against German blue, Hungarian light blue and Croatian
yellow; Windisch khaki (0.66 0.12 95) against Slovenian mint. Hungarian (#6cd3fa) and German
(#359bd9) are both blues in Burgenland's Hungarian villages (Oberwart, Unterwart); distinguishable,
left alone since both are other countries' colours.

## 7. Not done

- STATcube and Länder offices for a Gemeinde-level 2001 table outside Burgenland and Kärnten.
- Wien at Zählbezirk: Tabelle 2 has citizenship by Zählbezirk, but religiondots' hexes are keyed
  to Gemeindebezirk only, so inside a Wien district placement is by population.
