# South and Central Asia — language coverage

South Asia's three big mother-tongue censuses are all fine-grained and open. India 2011 C-16 gives ~270 mother tongues down to sub-district (~5,900 units) as one xlsx per state; using the mother-tongue rows rather than the language rows undoes the "Hindi umbrella" (Bhojpuri, Rajasthani, Magahi etc. are their own rows). Nepal 2021 gives ~125 mother tongues for all 753 local levels in one xlsx, plus second-language and ancestral-language tables. Pakistan 2023 Table 11 gives ~15 categories by tehsil. PBS's own PDFs moved and 404, but the CRAN package PakPC2023 carries it as data.

Sri Lanka 2024 and Bangladesh 2022 are ethnicity-only for mapping purposes. Sri Lanka publishes ethnicity as Excel at Grama Niladhari level (~14k units), which is very fine, and the language-ability item is tabulated nationally only. Bangladesh has ethnic groups down to union in PDF district reports. Both crosswalk cleanly except Sri Lankan Moors (mostly Tamil-speaking).

Central Asia asks Soviet-style native language (identity-flavoured) but publishes it coarsely. Kyrgyzstan 2022 is the best: ethnicity x native language by rayon in the Book III oblast PDFs. Turkmenistan 2022 was a surprise: a public English volume gives nationality x mother tongue by velayat (7 units). Kazakhstan 2021 publishes native language only nationally, and only as "own nationality's language vs another", but its ethnicity-by-settlement xlsx (~6k settlements) is a strong crosswalk base. Tajikistan 2020 lists a language volume with no tables online. Uzbekistan finally held a census (Jan-Feb 2026), but only national preliminary figures are out so far.

Gaps: Afghanistan (no census; Asia Foundation Survey of the Afghan People by province is the fallback), Iran (no census item; one 2015 ministry survey across 31 provinces exists only as reported percentages), Bhutan (literacy-by-language only) and Maldives (guess, not checked).

Access quirks: censusindia.gov.in and censusnepal.cbs.gov.np have broken TLS (curl -k works). The live Nepal portal is censusresults.nsonepal.gov.np, with files under /files/caste/.
