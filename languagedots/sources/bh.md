# Bahrain (bh): record

Drawn 2026-10-05 (session edd42a8c-gulf). Method shared with the other Gulf states:
`sources/gulf.md`. 2020 census, 4 governorates, 1,501,635 people, 91 nodes, every row
`derived`, on religiondots' Kontur 400 m hexes. 1,462 dots at 1:1,000; 37 rings.

```
python sources/bh_build.py --fetch     # --fetch re-runs bh_extract.py
python taxonomy/build.py
python tools/check_country.py bh
python scatter.py --country bh
```

Files: `sources/bh_extract.py`, `sources/bh_build.py`, `sources/gulf_mix.py`,
`taxonomy/bh2020.py`, `taxonomy/tree.d/bh.txt`, `countries/bh.py`, `data/raw/bh/bh_groups.csv`,
`data/normalized/bh.csv`.

## Tables

- **Census 2020** (data.gov.bh), governorate x nationality group (Bahraini, GCC, Other Arabs,
  Asian, African, European, North American, Others) x sex, through religiondots'
  `sources/bh.py` (`load_census`, which checks it against the governorate and religion tables),
  loaded by path by `bh_extract.py`, read-only.
- **Non-Bahrainis**: each group and sex at UN DESA 2020's named origins in that group (religiondots'
  group assignment), each on its language or home mix. The census `Others` group (Oceania, Latin
  America, the rest; no DESA origin) on `other`. Indians: Keralites 79,708 (KMS 2023, 3.7%) =
  26.0% of DESA 2020's Indians.

## Bahrainis: Baharna Arabic and Gulf Arabic, by sect

Bahrain's Arabic splits along sect: the Shia Baharna speak Baharna (Bahrani) Arabic, baha1259;
Sunni Bahrainis (the 'Arab, of Najdi and tribal origin) speak a Gulf Arabic dialect (Holes 1987;
Glottolog lists both). **No source counts either dialect, nor sect.** The Shia counts per
governorate are religiondots' modelled split (`../religiondots/sources/bh.md` §7, Anita's ruling
on ask 055 there): Arab Barometer I (2009), 57.2% Shia of all Bahraini answers, placed by
governorate in proportion to the Ja'fari and Sunni endowments' mosques, one logit shift. Read from
religiondots' `data/normalized/bh.csv` and asserted against the census's Bahrainis per governorate.

| governorate | Bahrainis | Baharna Arabic | share |
|---|---:|---:|---:|
| Capital | 169,192 | 132,671 | 78% |
| Muharraq | 136,124 | 29,195 | 21% |
| Northern | 272,093 | 216,344 | 80% |
| Southern | 134,953 | 28,242 | 21% |

Everyone else Bahraini (Sunni, "Muslim" with no sect, the 2,295 non-Muslim citizens) on Gulf
Arabic.

**The Ajam** (Shia of Persian descent, Persian or Achomi-speaking in older generations) are inside
the Shia count and so drawn on Baharna Arabic; no source counts them or their home language. **The
Hawala** (Sunni families from the Iranian coast) are on Gulf Arabic. Both named in the record,
neither on the map.

## Results

Baharna Arabic 27.1%, Gulf Arabic 21.2%, Bengali 8.6%, Hindi 6.0%, Malayalam 5.8%, Punjabi 3.6%,
Egyptian Arabic 2.8%, Tamil 2.7%, Pashto 2.0%, Telugu 1.6%, English 1.5%.

## Calls someone might reverse

- **Sect as dialect.** Reversing it draws all Bahrainis on Gulf Arabic: one constant in
  `bh_build.py`. The split itself rests on a modelled sect count, so the Baharna figure is twice
  removed from a measurement.
- Census `Others` group on `other` (religiondots pooled it with Europeans and North Americans).

## Placement inside units (2026-10-06)

Citizens and foreign residents are now placed apart inside each unit (session 5d7dac7e-gulf): foreign dots lean to dense hexes and fill OSM industrial land and labour camps, citizens take the rest; one rule for the six Gulf states, fitted on Kuwait's areas and Oman's wilayat. Method, fit, data searched: `sources/gulf_place.md`. People in hexes 90%+ non-Bahraini: 0 -> 6%; governorate shares unchanged.
