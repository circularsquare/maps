# Bhutan (bt): record

Drawn 2026-10-05 (session edd42a8c-mono4). 727,145 people (2017 census), 20 dzongkhags, 71
nodes, every row `modelled`. 709 dots at 1:1000, 44 rings.

```
python sources/bt_gnh.py
python sources/origin_mix.py --fragment bt
python taxonomy/build.py
python tools/check_country.py bt
python scatter.py --country bt
```

Files: `sources/bt_gnh.py`, `taxonomy/bt2015.py`, `taxonomy/tree.d/bt.txt`, `countries/bt.py`,
`data/normalized/bt.csv`. Inputs read from religiondots, read-only: the GNH 2015 report PDF
(`data/raw/bt/gnh_2015_compass_report.pdf`), `data/geo/bt/bt_lookup.csv` (2017 census Bhutanese
and non-Bhutanese per dzongkhag, Tables 2.6 and 2.8), the UN DESA migrant-stock workbook
(`data/raw/mr/`), and `data/geo/bt/bt_hexes.gpkg` for placement.

## 1. What exists

- **Census**: PHCB 2005 and 2017 ask literacy in Dzongkha, English, Lhotshamkha, other; no
  spoken or mother-tongue question (coverage sweep, 2026-10-03).
- **MICS 2010 (BMIS)**: World Bank catalog 1315's DDI checked 2026-10-05: no language variable
  (interviews were in Dzongkha, Lhotshamkha and Sharchopkha, but that is not recorded). The
  report PDFs sit behind Cloudflare.
- **Gross National Happiness survey 2015** (Centre for Bhutan & GNH Studies, *A Compass Towards
  a Just and Harmonious Society*, 2016): asks mother tongue; Table A1.5 (pdf pp.300-301) gives
  21 categories per dzongkhag, sample weighted (p.51), about 7,150 Bhutanese aged 15+.
  religiondots already uses this table to place Hindus. This is the source, under the
  2026-10-05 ruling (no language question, a survey asks one: rows `modelled`).

## 2. How the counts are made

- Bhutanese per dzongkhag (2017 census) x A1.5's shares for that dzongkhag (rows rescaled to
  exactly 100; printed rows sum to 100 +-0.1). Largest-remainder rounding within each dzongkhag.
- Non-Bhutanese (45,425, by dzongkhag; no nationality tabulated) x UN DESA 2020's origins for
  Bhutan (53,612; India 46,974; the 24 named origins are the mix, `Others` assumed alike). Each
  origin through `origin_mix.mix(iso, "bt")`, except **India**: Indians in Bhutan are mostly
  labourers from West Bengal, Assam, Bihar and Jharkhand (BhutanWiki, "Foreign Workers in
  Bhutan", a weak source but the only one found), so their mix is those four states' drawn counts
  on this map, pooled, 1% cut: Bengali 37%, Hindi 16%, Bhojpuri 11%, Assamese 6%, Maithili 5%,
  Magahi 5%, Urdu 5%, Khortha, Santali, Sadri, and 6% on India's unnamed Indo-Aryan remainder.
  All-India's home mix would have put 3,400 Marathi and 3,300 Telugu in Bhutan.
- Children drawn at the 15+ shares.

**Checks.** A1.5's Bhutan row is pinned (Dzongkha 21.13, Tshangla 33.72, Nepali 18.69, Khengkha
8.05, Others 4.28). Drawn among Bhutanese: Tshangla 32.8%, Dzongkha 22.4%, Nepali 18.3%: the
survey's weighting and the 2017 census's dzongkhag sizes nearly agree. Every dzongkhag sums to
its census population (asserted). For comparison only, van Driem (1993, government-commissioned)
had Dzongkha 24.5%, Lhotshamkha 24%, Tshangla 21.2%; nothing here depends on it.

## 3. Calls someone might reverse

- **"Monpakha" drawn as Olekha** (Black Mountain Monpa, olek1239): its answers are in Trongsa,
  Dagana, Wangdue, around the Black Mountains. 1,296 people.
- **"Others" on `other`**: 4.3% nationally, 24% of Tsirang, 20% of Dagana, 15% of Sarpang;
  very likely Tamang, Gurung, Rai, Limbu among the Lhotshampa, but unnamed, so not guessed.
- **"Bokha (Tibetan)"** on `sinotibetan.tibetic.tibetan`.
- **Indians on four neighbouring states' mix**, from a weak source (above).
- **English** kept as answered (0.02%): this is a mother-tongue question, not an ability one.
- Olekha, Gongduk and Lhokpu sit directly under Sino-Tibetan (Glottolog's top-level branches);
  the East Bodish group is the conventional one (Bumthang, Kheng, Kurtöp, Nyenkha, Chali, Dzala,
  Dakpa); Chocangacakha, Brokpa, Lakha, Layakha are Tibetic per Glottolog.
- Hand colours for six East Bodish and Tibetic leaves (fragment) so they part from Dzongkha's
  cyan and Tshangla's lilac.

## 4. Room for improvement

A census mother-tongue question would replace a 7,000-person survey. The GNH 2022 report
(religiondots' `data/raw/bt/gnh_2022_report.pdf`) prints no mother-tongue distribution, only the
fluency indicator (checked 2026-10-05), so 2015 is the latest. A nationality table for the 45,425 foreign residents would replace the DESA mix.
