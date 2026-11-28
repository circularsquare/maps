# Country adds, slow plan (2026-11-28)
Agent countryrun. Kit: C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\bae761ca-c85c-416e-b4e2-0a931ae6b8a6\scratchpad\countryrun\
Anita 11-27: cores are low, start a country add plan, keep it slow. Rules: docs/world-run.md "Rules for any data run", docs/adding-a-country.md.
Task context: T-3764 (fill the gaps; its notes hold the batch list this follows). No task taken by this agent.

## Order (about 8 batches; one country at a time, 3 cores)
I found no stored list of player region requests (no entry in data-register.md, TASKS.md or handoff.md beyond "11-25: requests are leads, not orders"), so the order follows T-3764's gap list: built neighbours first (each batch makes the next touch a built country), then size of people. Nigeria, Ethiopia, Kenya come up when she wants "popular first"; say so and they move up.
1. **Uzbekistan** (33.6M, 36 rows: Tashkent plus 35 zones, default rule, 10 dead-space units). Running now. Catch: none found; zones and the preflight are clean (0 flags). Tashkent's extract also reads Kazakhstan's. Tertiary share 0.011 (2019), so no university_rate park. No census hook (UZ is not in admin_totals), so GHSL people.
2. **Kazakhstan** (18.5M; planned alm, ast, shm). Catch: huge empty land, needs a `reach` rule as Canada's (dry run, read the merge log); touches Russia and China (border factors).
3. **Tajikistan, Kyrgyzstan, Turkmenistan, Mongolia** (6 + 9 + 6 + 3M; one planned each): small, run one after the other. Catch: Mongolia on a reach rule; Turkmenistan touches Iran (border factor); possible GHSL piles in mountain villages (preflight FLAG phantom). Then one Central Asia join.
4. **Nepal, Sri Lanka, Bhutan** (52M). Catch: Nepal touches India; Sri Lanka is an island, `contiguous` not needed; check phantom pixels.
5. **Levant and Yemen** (Syria, Jordan, Lebanon, Yemen, 63M). Catch: lines of control (Syria, Yemen) are `DE_FACTO` / `LEFT_OUT` calls, hers to overturn; GHSL is poor in war-damaged areas, so census totals may be wanted.
6. **Africa beside built countries** (Sudan, Libya, Niger, Mali, Mauritania, Chad, Cameroon, Benin, about 150M). Catch: thin tertiary figures (parks on university_rate until NATIONAL_TERTIARY); few census hooks.
7. **Nigeria** (13 planned rows, the biggest unbuilt country) then Ghana, Ivory Coast, Burkina, Senegal and the rest of West Africa; Ethiopia, Kenya, Tanzania, Uganda. Catch: GHSL piles in dense African towns, held rows.
8. **Southern Africa** (Mozambique, Zimbabwe, Namibia, Botswana), then Papua New Guinea and the Pacific.
AWS: not used in this first run (country adds build on this machine; docs/aws.md allows timing runs only through bench.py). If a later batch uses it, say why here first.

## Pace
- Claim 3 cores at most (cores.py claim 3 countryrun), one country at a time, detached. The gate chunks and the sim build stay within it (CARGO_BUILD_JOBS=3, --jobs 3).
- The join needs at least 6 cores (docs/adding-a-country.md 7). Under a 3-core cap it waits for her word on a higher claim, or the batches queue and share one join.
- Uzbekistan's run builds with --no-join, so its 36 rows end gated and registered; the join follows later (one for UZ, or UZ with Kazakhstan).

## Hold needed (for the orchestrator)
- **A run owns data/packs.** From the first gate chunk's register (rows are registered, gated and either kept or taken out again) until the join ends, nothing else may write data/packs. 8800 rebuilds need the freeze while it runs (docs/orchestrator.md).
- **The gate serves a frozen copy of app/** made at each chunk's start (T-2298), so app/ and WASM landings need no hold per world-run.md; the orchestrator can still hold app/ writes during a gate if it wants no risk. Tell it before the first gate (about an hour in, when the first chunk of 10 rows is built).
- The sim copy is HEAD's sim/ at C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\bae761ca-c85c-416e-b4e2-0a931ae6b8a6\scratchpad\countryrun\simroot\sim; do not commit to sim/ without `setup_sim.py refresh` if a later batch should see it.

## Run 1: Uzbekistan
- Plan: dry run C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\bae761ca-c85c-416e-b4e2-0a931ae6b8a6\scratchpad\countryrun\uz.json and uz.png (preflight 0 flags, 1 note: Afghanistan within 50 km, default border factor), live plan written 2026-11-28 03:41 (36 planned rows: tas srv adj dba ure ynk nmn sjk sjl kby sxa kge kci ckc gij ksp kbz trm bva ttk frg bdt pyk nuk atx bdv kca psd kku urt kcb sjm akn cbv pse kcc). Status before: planned 79; after plan: 114 (79 + 35 new; tas was already planned).
- Backup: trainworld\data\work\pre-Tcountryrun\pre (UZ, AF packs, 956 MB). Delete after a clean join.
- Command: C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\bae761ca-c85c-416e-b4e2-0a931ae6b8a6\scratchpad\countryrun\run_uz.cmd (run --countries UZ --limit 300 --jobs 3 --port 6381 --no-join).
- Batch 20261127-224249, log artifacts\trainworld\countryrun-uz-run.log (runner) and artifacts\trainworld\world_run\20261127-224249\.
- PIDs: cmd 13416, launcher python 30372, child python 6984. Claim: countryrun (3 cores, pid 30372).
- Check: `python cores.py list`; `Get-Process -Id 30372`; `venv\Scripts\python.exe -I trainworld\pipeline\world_run.py status`.
- Stop: create trainworld\data\work\world_run\STOP (the row under way finishes).
- Restart: the same .cmd (built rows are taken up; after parks use `--rows`).
- After: parks read per docs/adding-a-country.md 6 (one relax run), then the join.
