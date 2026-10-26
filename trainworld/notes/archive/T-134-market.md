Archived: task done, see ARCHIVE.md. Kept as background; SPEC.md and docs/ say what is true now.

# T-134 regional market expansion

The regional endpoint pack now reads New York, Philadelphia, Boston and Washington DC city-pack cells, plus the existing Mercer County Census/LODES points. Market IDs remain New York 0, Philadelphia 1, Mercer 2, Boston 3, Washington DC 4. The source weights, 600 m sites, 6 km zones, trip rate, distance decay, fares and mode-choice model are unchanged.

The Rust reader takes its market count from the short `markets` metadata list. It still deserializes numeric sites directly into flat `[f64; 5]` rows rather than retaining a generic JSON tree. Old packs without that metadata keep three slots. Corridor rows cover all unordered market pairs including within-market trips; five markets produce 15 rows. `market_count()` exposes the count. Selection headers remain `1 + 8 * markets`, station headers remain `6 * markets`, followed by the existing zone data.

Validation: the full native suite passes 101 tests (six manual tests ignored). Nine intercity tests pass (one manual real-pack test ignored). The new five-market regression routes service between Boston and Washington, checks that selections in both new markets match their endpoint sites, and checks station market totals against zone totals and all rail trips. Existing three-market compatibility, fares, feeders, crowding, conservation and malformed-input tests pass. Fractional or out-of-range market IDs and empty market lists are rejected. `python -m py_compile pipeline/intercity.py` passes.

Generated `data/packs/intercity.json`: 167,594 endpoint sites, 5,965,010 bytes. Native loading yields 2,101 zones, 1,398,656 directed pairs and 590,774.074 daily all-mode trips. The existing manual real-pack acceptance passes: all output summary values finite and a New York/Philadelphia/Mercer service still attracts positive rail trips. No runtime calibration was changed to offset the larger destination set.
