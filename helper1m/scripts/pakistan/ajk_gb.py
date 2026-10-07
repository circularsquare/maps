"""Azad Kashmir and Gilgit-Baltistan: 2017 and 2023 census counts.

Neither is in PBS's 2023 district tables (which cover the four provinces and Islamabad only),
but both governments republish PBS's census counts for their own units:

AZAD KASHMIR, 32 tehsils. AJ&K Statistical Year Book 2025, Table 15.15 "Tehsil-wise Area,
Population and Nos. of Households of AJ&K (Census 2023)", p.184 of the book (PDF page 220):
2023 census count and the 2017 count re-tabulated on the 2023 tehsils (so Mirpur and Dudyal's
2017 figures differ from the 2024 book's, which used the old line between them). Source line:
"Population & Housing Census Report 2023, Pakistan Bureau of Statistics; AJ&KBoS P&DD".
    https://www.pndajk.gov.pk/uploadfiles/downloads/Statistical%20Year%20Book%202025.pdf
District totals cross-checked against AJK At a Glance 2025, p.3 (2017 and 2023 census by district).

GILGIT-BALTISTAN, 10 districts, no tehsil figures published. GB at a Glance 2025 (Statistical &
Research Cell, P&DD GB), p.3 "District Wise Population and Area of GB": census 2017 and 2023.
Source line: "Pakistan Bureau of Statistics; SRC P&DD GB".
    https://pnd.gog.pk/storage/downloads/AiRIlDEcscWPC1s58oXIgpjlVAS7jd-metaR0IgQVQgR2xhbmNlIDIwMjUuMS5wZGY=-.pdf

Both transcribed by hand from the PDFs' text layer; the sums below are asserted in check().
"""

# tehsil (as printed) -> (COD adm2 name, COD adm3 name, pop 2023, pop 2017)
AJK_TEHSILS = [
    ("Muzaffarabad", "Muzaffarabad", "Muzaffarabad", 526_926, 498_390),
    ("Naseerabad", "Muzaffarabad", "Naseerabad", 176_735, 152_764),
    ("Athmuqam", "Neelum", "Athmuqam", 153_909, 129_215),
    ("Sharda", "Neelum", "Sharda", 67_603, 60_570),
    ("Chikar", "Jhelum Valley", "Chikar", 40_596, 36_573),
    ("Hattian", "Jhelum Valley", "Hattian", 180_975, 161_424),
    ("Karnah", "Jhelum Valley", "Leepa", 35_488, 28_499),          # Leepa renamed
    ("Bagh", "Bagh", "Bagh", 189_844, 165_316),
    ("Dhir Kot", "Bagh", "Dhir Kot", 161_467, 130_035),
    ("Hari Gail", "Bagh", "Harighel", 85_484, 76_550),
    ("Haveli", "Haveli", "Haveli", 94_911, 81_037),
    ("Khurshid Abad", "Haveli", "Khurshid Abad", 32_700, 27_264),
    ("Mumtazabad", "Haveli", "Mumtazabad", 43_217, 37_919),
    ("Abbaspur", "Poonch", "Abbaspur", 70_661, 62_213),
    ("Hajira", "Poonch", "Hajira", 188_080, 181_079),
    ("Rawalakot", "Poonch", "Rawalakot", 256_942, 231_576),
    ("Thorar", "Poonch", "Thorar", 29_215, 24_958),
    ("Baluch", "Sudhnoti", "Baluch", 101_507, 96_724),
    ("Mang", "Sudhnoti", "Mong", 26_545, 25_976),
    ("Pallandari", "Sudhnoti", "Pallandari", 133_164, 124_179),
    ("Tarar Khal", "Sudhnoti", "Tarar Khal", 52_455, 51_001),
    ("Charhoi", "Kotli", "Charhoi", 107_417, 104_252),
    ("Darliah Jattan", "Kotli", "Dulliya Jattan", 16_404, 16_802),
    ("Fatehpur Thakiala", "Kotli", "Fatehpur Thakiala", 117_000, 107_211),
    ("Khui Ratta", "Kotli", "Khui Ratta", 159_160, 149_926),
    ("Kotli", "Kotli", "Kotli", 309_545, 298_610),
    ("Sehnsa", "Kotli", "Sehnsa", 94_739, 97_216),
    ("Dudyal", "Mirpur", "Dadyal", 88_993, 92_262),
    ("Mirpur", "Mirpur", "Mirpur", 352_791, 364_279),
    ("Barnala", "Bhimber", "Barnala", 144_952, 137_596),
    ("Bhimber", "Bhimber", "Bhimber", 169_944, 159_001),
    ("Samahni", "Bhimber", "Samahni", 124_098, 121_946),
]
AJK_TOTAL = {2023: 4_333_467, 2017: 4_032_363}
# AJK At a Glance 2025 p.3, district totals (2023) -- the tehsils above must add up to these
AJK_DISTRICTS_2023 = {"Muzaffarabad": 703_661, "Neelum": 221_512, "Jhelum Valley": 257_059,
                      "Bagh": 436_795, "Haveli": 170_828, "Poonch": 544_898, "Sudhnoti": 313_671,
                      "Kotli": 804_265, "Mirpur": 441_784, "Bhimber": 438_994}

# GB district (as printed) -> (COD adm2 names merged into it, pop 2023, pop 2017)
# COD v01 already splits off Darel, Tangir (from Diamer), Gupis-Yasin (from Ghizer) and Rondu
# (from Skardu); the census still counts the ten older districts, so those are folded back.
GB_DISTRICTS = [
    ("Astore", ["Astore"], 111_573, 95_416),
    ("Diamer", ["Diamir", "Darel", "Tangir"], 337_329, 269_772),
    ("Ghanche", ["Ghanche"], 157_822, 156_697),
    ("Ghizer", ["Ghizer", "Gupis-Yasin"], 200_069, 172_696),
    ("Gilgit", ["Gilgit"], 324_552, 285_236),
    ("Hunza", ["Hunza"], 65_497, 51_372),
    ("Kharmang", ["Kharmang"], 61_304, 54_613),
    ("Nagar", ["Nagar"], 87_410, 71_746),
    ("Shigar", ["Shigar"], 84_608, 74_540),
    ("Skardu", ["Skardu", "Rondu"], 278_885, 260_836),
]
GB_TOTAL = {2023: 1_709_049, 2017: 1_492_924}


def check():
    bad = []
    for yi, y in ((3, 2023), (4, 2017)):
        s = sum(t[yi] for t in AJK_TEHSILS)
        if s != AJK_TOTAL[y]:
            bad.append(f"AJK tehsils {y} sum {s:,} != {AJK_TOTAL[y]:,}")
    for d, want in AJK_DISTRICTS_2023.items():
        s = sum(t[3] for t in AJK_TEHSILS if t[1] == d)
        if s != want:
            bad.append(f"AJK {d} 2023 tehsils {s:,} != district {want:,}")
    for yi, y in ((2, 2023), (3, 2017)):
        s = sum(t[yi] for t in GB_DISTRICTS)
        if s != GB_TOTAL[y]:
            bad.append(f"GB districts {y} sum {s:,} != {GB_TOTAL[y]:,}")
    return bad
