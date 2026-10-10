# Middle East build (sa, ae, qa, iq, jo), 2026-10-08: shared-file changes for the managing session

All five are built in `dist/data/` (model, tiles, check_model passing). Nothing below has been
applied; the country builds do not need it except for `tools/rebuild.py` batch rebuilds.

## 1. tools/rebuild.py: REGISTER entries, and Qatar with no register

Qatar has no main-line railway, so it builds OSM-only (`build_model.py --region qa`, no
`--register`). rebuild.py always passes `--register`, defaulting to RINF, so it needs a way
to say "none".

```diff
     "al": "balkans_register:data/raw/rinf/al", "xk": "balkans_register:data/raw/rinf/xk",
+    # The Middle East: hand lists through rinf.py (mideast_register.py, nafrica's code).
+    # After an extract: `mideast_register.py --clip <cc>`; then `--join iq` for Iraq and
+    # `--fill jo` for Jordan. Qatar is all metro and tram: no register (None).
+    "sa": "mideast_register:data/raw/rinf/sa", "ae": "mideast_register:data/raw/rinf/ae",
+    "iq": "mideast_register:data/raw/rinf/iq", "jo": "mideast_register:data/raw/rinf/jo",
+    "qa": None,
 }
```

```diff
 def run_country(cc, model_only):
     reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
-    steps = [["build_model.py", "--region", cc, "--register", reg]]
+    steps = [["build_model.py", "--region", cc] + (["--register", reg] if reg else [])]
```

(Gated: no other country maps to None, so no other build changes; no ab.py run needed.)

## 2. tools/build_regions.py

Run it to put sa, ae, qa, iq, jo in `regions.json`. Neighbours need no rebuild: no passenger
train crosses any of these borders (sa-jo, sa-iq, ae-om, sa-ae, sa-qa), so there are no
border points.

## 3. Proposed, not needed now: a "drawn, not counted" flag for OSM lines

The managing session's call was that Doha's Education City tram and the Msheireb loop be
drawn but not counted. build_model has no mechanism: `looks_like_service` is called for
route=train only (a tram is never a named train), and the only precedent (rules/mx.py
making a route a named train) works for trains only. They are built as ordinary OSM lines and
count (8.2 km). A rules hook such as `UNCOUNTED_ROUTES = {relation id, ...}` setting the
line's `service` flag (or a new `uncounted` flag the app leaves out of totals) whatever the
route kind would do it; qa would set {10563564, 13475082, 13475083, 10567246}.

## Files the agent wrote (for the record)

New: `mideast_register.py`, `mideast_lines.py`, `rinf_countries/{sa,ae,iq,jo}.py`,
`rules/{sa,ae,iq}.py`, `colours/{sa,ae,iq,jo}.csv`. Edited: `check_model.py` (REGISTER sa,
iq, jo; KNOWN sa, ae, qa), `{sa,ae,qa,iq,jo}_sources.md` ("Build (2026-10-08)").
mideast_register imports nafrica_register and extends its LINES / NOT_SERVICE / LANGS / ISO3
tables and its `abroad()` in its own process only; nafrica_register.py is unchanged, but a
change to its function signatures (convert, clip, fill, trace_cmd, fork_cmd, build,
country_conf) would break these five.
