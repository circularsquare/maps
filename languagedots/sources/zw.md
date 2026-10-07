# Zimbabwe: 2022 census mother tongue by province, placed by district from Afrobarometer

Drawn 2026-10-05 (session edd42a8c-zw). 13,913,253 people (aged 3 and over), 10 provinces, 17
answers, every row `measured`. 13,906 dots at 1:1000, 1 ring.

```
python sources/zw_census.py          # Table 2.17 -> data/normalized/zw.csv, with its checks
python sources/zw_afro.py --fetch    # Zimbabwe's Afrobarometer rows from religiondots' .sav (read-only)
python sources/zw_place.py           # religiondots' hexes + each hex's district
python taxonomy/build.py
python tools/check_country.py zw
python scatter.py --country zw
```

## 1. The table

The coverage sweep had Zimbabwe at tier E ("has never conducted a census that enumerated
people by language"; IPUMS 2012 has no language variable). **The 2022 census did ask.** ZIMSTAT,
*2022 Population and Housing Census Report* (27 Jan 2023), religiondots'
`data/raw/zw/zw_phc2022_report.pdf` (read-only;
zimstat.co.zw/wp-content/uploads/Census/2022_PHC_Report_27012023_Final.pdf):

- p. 16: "Mother tongue is the language usually spoken in the individual's home in his/her
  early childhood." Shona 80.9%, Ndebele 11.5%.
- **Table 2.17** (PDF p. 148, printed 123), "Distribution of Population by Mother Tongue and
  Province": 16 languages + Other, 10 provinces, counts. A first-language question, so ask 018
  does not arise: English is drawn as counted (43,909, 0.3%, half in Harare).

Province is ZIMSTAT's ceiling: the other 2022 releases (district/ward population, projections,
and the Fertility, Disability and Youth thematic reports) carry no finer language table. The
Youth report (downloaded to `data/raw/zw/zw_phc2022_youth_report.pdf`) repeats mother tongue for
ages 15-35 by province, in percentages only. No microdata is published.

**Universe.** The table sums to 13,913,253 of 15,178,957 enumerated; each province's table is
0.907-0.937 of its census population. That fits persons aged 3 and over (the education module's
universe; 13,102,643 are 5+). The under-3s, 1,265,704, are in `gap`, as Burkina Faso and Côte
d'Ivoire do. No not-stated row is printed.

## 2. Checks (`sources/zw_census.py`)

| check | result |
|---|---|
| each language's provinces sum to its Total column | 17 of 17 exact |
| each province's languages sum to the Total row | 10 of 10, and 13,913,253 nationally |
| report text | Shona 80.87% (80.9), Ndebele 11.49% (11.5) |
| table / census population per province | 0.907-0.937 (the 3+ universe) |
| Afrobarometer, all six rounds pooled, national | Shona 77.1 / 80.9, Ndebele 14.5 / 11.5, Ndau 2.9 / 2.7, Tonga 1.7 / 1.7, Shangani 0.7 / 0.8, Venda 0.7 / 0.5 |
| Afrobarometer, the language's main province, survey / census % | Ndebele Mat. North 65.7 / 64.1; Ndau Manicaland 19.9 / 18.0; Tonga Mat. North 20.5 / 20.6; Venda Mat. South 10.1 / 9.3; Shangani Masvingo 6.1 / 6.7; Kalanga Mat. South 9.9 / 4.7 |

## 3. The survey, for placement only (`sources/zw_afro.py`)

Ten units of 1.4M are too coarse to show Ndau, Tonga or Venda where they are, so inside each
province the dots follow the Afrobarometer's own respondents by district (AGENT_BRIEF §4.4:
the counts stay the census's). Rounds 4, 6, 7, 9 carry a district (5,999 respondents; all join a
COD-AB district of their province, urban councils folded into the district around them: 65
placement districts, all sampled). Each district's share of a language = (its respondents'
weighted answers + 8 x the census's province share) / (its respondents + 8), so a language the
survey barely sees in a district falls back to the census share.

**The rounds and the interview language (Malawi's and Niger's traps), checked.** Unlike Malawi
the rounds agree: Shona plus its dialects 71-81% in every round, Ndebele 11-15% except R5's
19.9% (an oversample of Matabeleland, not a wording effect). The R7-R9 wording change ("language
spoken in home") moves English from 0.3-0.4% to 0.8-0.9% and nothing else. Interviews were only
in Shona, Ndebele and English (10 in Ndau in R5), so a Tonga or Nambya speaker answered in
another language; even so the survey's Tonga share in Matabeleland North equals the census's,
and 101 of 125 Tonga answers there came from Ndebele-language interviews. Since the survey only
places, a bias would move dots within a province, not change any count.

As placed: Ndau 190 of 372 dots in Chipinge, 74 Chimanimani; Tonga 96 of 231 Binga, 41 Hwange;
Venda 55 of 71 Beitbridge; Shangani 75 of 115 Chiredzi, 18 Mwenezi; Nambya 23 of 38 Hwange;
Kalanga 26 of 52 Bulilima and Mangwe; Sotho 26 of 39 Beitbridge and Gwanda.

Free-text answers are mapped onto the census's categories in `VERB` (Shona varieties such as
Buja, Bocha, Hwesa, Shangwe, Budya to Shona; Dombe to Nambya, the nearest category in Hwange;
Lozwi to Kalanga; Nyanja and "Malawian" to Chewa). Placement only.

## 4. Mapping and tree (`taxonomy/zw2022.py`, `taxonomy/tree.d/zw.txt`)

- **Shona is one node.** The census prints one Shona. The Afrobarometer does name the dialects
  from R5 (Karanga, Zezuru, Manyika, Korekore: 10-30% of answers, the rest plain "Shona"), but it
  cannot split the census's Shona: most respondents never name one, and making Shona a group
  would wash out 11M dots and the leaf other countries draw on.
- Ndau on mz.txt's Cindau, Kalanga on bw.txt's, both leaves beside Shona; **Nambya** a new leaf
  beside them (namb1291; Glottolog puts all three inside Shona S.10).
- **Tonga on zm.txt's "Tonga (Zambia)"** (tong1318, listed for ZW): the same language across the
  Zambezi. The label reads oddly on Zimbabwe; zm.txt's owner could rename it "Tonga (Zambezi)".
- **Sotho** on a new leaf "Sotho (Zimbabwe)" under Sotho-Tswana: 84% are in Matabeleland South
  (Gwanda, Beitbridge), whose Sotho is the Limpopo valley's Birwa / Northern Sotho speech, not
  Lesotho's; the census does not say which, so neither existing node.
- Shangani on Xitsonga; Chibarwe on mz.txt's Barwe; Chewa, Venda, Tswana, Xhosa, English as
  named. **Koisan** (305) on the Khoisan root, as bw2011.py's Sesarwa (Tshwao, Kalahari Khoe;
  the census names no language). Other on `other`; Sign Language on `signlanguage`.
- **Colours.** Shona and Ndebele (Zimbabwe) were never hand-coloured and came out two near
  browns although they meet across Midlands and Bulawayo; set here: Shona the same brick
  (0.58 0.14 38), Ndebele a light sand (0.84 0.12 82). Both also draw in bw, uk, mw and others.
  Nambya dark red-brown, Sotho (Zimbabwe) yellow-green.

## 5. Calls someone might reverse

- Drawing the 3+ universe and leaving the under-3s out (rather than scaling up).
- District placement from the survey (switch: drop `place_weight` for plain population).
- Shona undivided; Sotho on its own leaf; Tonga on Zambia's node.
- Shona and Ndebele recoloured in this fragment.

## 6. Room for improvement

A district table of Table 2.17 (ZIMSTAT has the data; not published) would replace the survey
placement. The Shona dialects would need a source asking for them; the Afrobarometer's dialect
answers (Karanga in Masvingo and Midlands, Zezuru around Harare, Manyika in Manicaland, Korekore
in Mashonaland Central) are too partial to split counts.

## Terms

ZIMSTAT 2022 PHC report: free publication, quoted. Afrobarometer: free download, citation
requested ("Afrobarometer Data, Zimbabwe, Rounds 4-9, 2009-2022"). COD-AB Zimbabwe (OCHA);
Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Shona is now a group holding Shona (its own leaf), Ndau, Manyika and Tewe, so Mozambique's Ndau, Manyika and Tewe read as Shona varieties. Kalanga and Nambya stay beside it. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
