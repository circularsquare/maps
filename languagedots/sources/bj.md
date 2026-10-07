# Benin: RGPH-4 2013 household language (IPUMS sample), read as first language through the census's ethnic clusters

Drawn 2026-10-05 (session edd42a8c-wafr). 10,008,749 people (RGPH-4), 77 communes, 55
answers, every row `modelled`. 9,982 dots.

```
python sources/bj_census.py              # -> data/normalized/bj.csv (first-language reading)
python sources/bj_census.py --as-given   # the answers as given, for comparison (re-run plain after)
python taxonomy/build.py
python tools/check_country.py bj
python scatter.py --country bj
```

| | |
|---|---|
| language | RGPH-4 Q17 "langue principale parlée dans le ménage", persons 3+, write-in (IPUMS `BJ2013A_LANG` / `LANGBJ`, ~75 codes; question text and code list from the World Bank microdata library's DDI of the IPUMS subset, catalog 6820) |
| tabulation | CLEAR Global, HDX "Benin - Languages" (CC BY-SA): `clearglobal_language_use_ben_admin2.csv`, IPUMS 10% sample by commune, proportions with glottocodes; `data/raw/bj/` |
| population, ethnicity | INStaD *Principaux indicateurs* (12 booklets), Tableau 2 (commune populations) and Tableau 8 (nine ethnic clusters + "autres" + "étrangères", full count), read with religiondots' `sources/bj.py` parser (imported read-only) |
| geography | religiondots' `bj_communes.gpkg` / `bj_hexes.gpkg` (COD pcodes = CLEAR's codes), read-only |

## 1. Not ethnicity-only after all

The queue had Benin as tier D. RGPH-4 asked a language question; INStaD never tabulated it, but
the IPUMS sample carries it and CLEAR Global's HDX layer is that variable by commune (not an
ethnicity relabel). So this is a census language table at commune grain, from a 10% sample
(~1M records), and the rows are `modelled` (sample shares x full-count population).

## 2. CLEAR's codes, checked against where the answers fall

- "Central Malay" (mala1479) is IPUMS's **Lekpa** (Lokpa): Ouaké 54%, Djougou 15%, Bassila,
  Copargo. Relabelled Lokpa (lukp1238).
- "Tagwana Senoufo" (an Ivorian language) is IPUMS's **Agouna** (code 129, a Gbe language of
  Djidja): Djidja 8%, Zogbodomey, Savalou. Relabelled.
- Bare "Gbe" (gbee1241) holds IPUMS's **Toligbe, Setogbe, Kogbe**, which have no glottocode
  (Avrankou 55%, Akpro-Missérété 46%, Tori-Bossito 36%). One leaf "Toli, Seto and Kogbe".
- IPUMS has two "Defi" codes (116 among Gbe, 193 after the northern languages). In the south
  (Sèmè-Kpodji 4%) it is Defi Gbe; in the four northern departments (Ouaké 21%, where Tableau
  8 has 0% Adja) it is drawn on the Gur group node as "language not identified".
- "Agu (Ewe)" is IPUMS's "Ewe": Ewe's leaf. "Unknown" (0.37%) is not drawn.

## 3. The check against Tableau 8, and the first-language reading (ask 018)

Every answer was filed under Tableau 8's cluster (`CLUSTER`) and compared per commune.
Nationally they agree closely (language answers / ethnicity, %): Fon 40.0 / 38.4, Adja 14.2 /
15.1, Yoruba 11.1 / 12.0, Bariba 9.7 / 9.6, Peulh 8.6 / 8.6, Ottamari 6.1 / 6.1, Yoa-Lokpa 4.4 /
4.3, **Dendi 4.8 / 2.9**; French/English 1.0. Kotafon is filed under Fon (INStaD puts Lokossa
and Athiémé at 61-66% "Fon et apparentés"; they are 55-57% Kotafon). The gaps are the lingua
francas: Dendi in Malanville 81/60, Kandi 21/12, Parakou 19/9, Djougou 24/15, Ségbana 10/2;
French 6.4% of Cotonou, 3.4% of Porto-Novo. Fon in Cotonou is not one: the Fon cluster's
share there matches the answers (Fon drawn 42.5% of Cotonou either way).

**Drawn (`FIRST_LANGUAGE = True`):** each commune's eight named clusters at Tableau 8's shares,
each split by the answers of that cluster's languages in the commune (department, then
nation, where none). "Ethnies étrangères" keeps the foreign-language answers at their given
share (capped at the cluster); the rest of it, and "Autres ethnies du Bénin" (0.9%, 15% of
Malanville, no language named for it), go on the commune's other clusters pro rata. French and
English get nothing. National effect: Dendi 3.3% -> 2.4%, French 0.9% -> 0, Fon 20.5% ->
19.7%, Ede Nago 5.2% -> 5.6%.

One oddity carried in: Tableau 8 puts Kandi at 18.7% Yoruba against 1% Yoruba answers; drawn,
Kandi's Yoruba share comes from the commune's few Yoruba answers scaled up.

## 4. Calls someone might reverse

- The first-language reading at all (one line, `FIRST_LANGUAGE`; ask 018 open).
- The four relabellings in section 2 (inferred from geography, not from IPUMS labels per commune).
- Toli, Seto and Kogbe on one leaf; Gbe languages as siblings under Kwa (not under `kwa.gbe`,
  us.txt's leaf).
- Children under 3 (not asked) drawn on their commune's mix.

## 5. Room for improvement

An IPUMS extract (account blocked) would give the full ~75 codes, with Toli, Seto, Kogbe,
Holli, Gando, Lamba-like northern codes separated, and the language x ethnicity cross-table
that would replace the cluster rescale with a real one.

## Terms

CLEAR Global data CC BY-SA (derived from IPUMS International; IPUMS citation applies). INStaD
publications quoted. Glottolog CC BY; Kontur CC BY 4.0.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Every Gbe language (Fon, Aja, Gun, Gen, Ayizo, Maxi, Weme, Tofin, Saxwe, Xwela, Kotafon, Ci, Defi, Agouna, Toli, Ewe) is in a Gbe group with Togo's and Ghana's Ewe, Mina and Ouatchi; Yoruba and the Ede languages (Nago, Idaasha, Ifè, Ije, Cabe, Manigri) in a 'Yoruba and Ede' group. Benin's small Ewe count is right: its Gbe speakers name other Gbe languages. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
