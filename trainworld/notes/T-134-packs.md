# T-134: Boston and Washington city packs

Anita requested a lighter city-addition process: use a sensible broad metro extent and
the established sources, without a separate extent study or independent calibration.

`pipeline/build_city.py` now has `bos` and `wdc` configs and the additional state abbreviations
MA, NH, DC, VA and WV. Origins are central Boston (-71.0589, 42.3601) and central Washington
(-77.0365, 38.8977). Packs use the existing 2020 Census blocks, 2023 LODES WAC/RAC (2022
fallback), ACS 2019–2023 work-from-home adjustment, 25 m TIGER water and 2 km zone gravity.
There are no new city-specific job relocations or calibration rules.

Boston covers Essex, Middlesex, Norfolk, Plymouth and Suffolk in Massachusetts, and
Rockingham and Strafford in New Hampshire: seven counties. Washington covers DC; Calvert,
Charles, Frederick, Montgomery and Prince George's in Maryland; Arlington, Clarke, Culpeper,
Fairfax County, Fauquier, Loudoun, Prince William, Rappahannock, Spotsylvania, Stafford and
Warren in Virginia; Alexandria, Fairfax city, Falls Church, Fredericksburg, Manassas and
Manassas Park; and Jefferson in West Virginia: 24 counties or equivalents.

These are broad conventional commuter metro footprints, deliberately retaining Calvert
rather than chasing a recent statistical-area revision. They are gameplay demand areas,
not municipal boundaries. The existing [BLS county list](https://www.bls.gov/oes/2024/may/msa_def.htm)
provided a quick check on Washington's counties and independent cities; no new geographic
research project was needed.

Both builds and independent `read_pack.py` validation passed:

| Pack | Counties | Cells | Population | Jobs | Commutes/day | Build time | Binary raw/gzip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Boston | 7 | 86,131 | 4,941,632 | 2,811,395 | 2,274,483 | 94.2 s | 3.72/2.22 MB |
| Washington | 24 | 156,617 | 6,371,325 | 3,161,530 | 2,446,057 | 107.4 s | 6.18/3.42 MB |

Read-back checks cover array bounds/alignment, cell coordinates/totals, commute arrays,
water-mask structure, and independent gravity row/column margins. Boston has 2,613 zones
and 16.92 km mean commute; Washington has 4,364 zones and 18.36 km. Both margin errors
are below 0.007%, and neither lost any LODES jobs to missing 2020 block centroids.
Population totals exactly match the pipeline's cached Census base check. The automatic
QCEW check reports Boston jobs +4.0% and Washington -0.3%; these are ordinary source
differences and no independent job relocation/calibration work was added. No new
named-location water checks were invented (the read-back report explicitly lists zero).

Run with the maps venv, `CARGO_BUILD_JOBS=2`; no pipeline parallelism was added. The first
attempt hit sandbox network restrictions and the download/build commands succeeded with
the standard network escalation. Build logs are
`T-134-bos-build.log` and `T-134-wdc-build.log`; inputs are cached in `data/raw/` and generated
JSON/binary/boundary products are in `data/packs/`, all following the existing pipeline.
