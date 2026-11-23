# Jordan / Jojo save-load investigation, 2026-11-15

Read-only production investigation requested by Anita. No player saves, account records, or server code changed. No private save payloads downloaded. Production files were inspected in memory over SSH, returning diagnostic summaries only. A proposed persistent local download was rejected by automatic approval review; in-place inspection completed the diagnosis instead.

## Confirmed evidence

Jordan: account `ue83342ed78f8fc4c`, feedback `1794764329-b77884c3`, build `mwhxbd5k`, browser reported as `Safari 16`, French. Feedback has no underlying load exception. Anita subsequently relayed that Chrome has the same problem.

| Cloud game | Revision checked | Compressed bytes | Strict gzip validation | Recovered network structure |
| --- | ---: | ---: | --- | --- |
| Loire-Atlantique sandbox | 1 | 16,394 | Incomplete/truncated stream | 167 nodes, 45 stations, 167 edges; cuts off in base line data, at track byte 4,537 |
| Paris sandbox | 1 | 119,356 | Incomplete/truncated stream | 1,838 nodes, 1,282 stations, 1,938 edges, all 98 base line records readable; cuts off in LPH1 after 44 of 67 complete timing entries, at track byte 142,456 |
| Loire-Atlantique sandbox 2 | 41 | 25,604 | Pass | Day 4, 6 base line records, 145 stations |
| Same game's recent backup | 40 | 25,401 | Pass | Day 4 header |
| Same game's day backup | 1 | 10,725 | Pass | Day 2 header |

Both damaged games have only revision 1 in their cloud index and no listed backups. Validity means gzip completion/checksum and game-header checks; no full simulation/browser load was attempted. The base-network scanner followed `sim/src/track/save.rs`'s node, edge and line encoding, with strict byte bounds and UTF-8 reads; it is a structural diagnostic, not full model validation.

All five streams yield TWG1 / game anitabuilder / format 1 / region northeast / TWT6. The two bad streams do not reach zlib's end-of-stream marker. Python's strict zlib decompressor rejects both with error -5 (incomplete or truncated stream). Therefore this is more than a misleading card or a browser failing to download a valid cloud copy. Chrome cannot restore compressed bytes missing from the persisted save.

## Cause and code paths

Confirmed mechanism: saved gzip payloads are incomplete. Original source of truncation remains an inference.

Likely source: Safari's known CompressionStream flush bug, WebKit issue 254021, fixed upstream in May 2023: https://bugs.webkit.org/show_bug.cgi?id=254021 . It prematurely exits when flush output fills its buffer. The first bad file is exactly a 10-byte gzip header plus 16,384 bytes. This fits the failure, but does not prove Jordan used an affected Safari point release when saving. `browserOf` in `app/src/ui/FeedbackPanel.tsx` deliberately retains only the major browser version, so Safari 16.4 versus later 16.x cannot be distinguished from the feedback.

- `app/src/workers/saveFile.ts`: encodeSave trusts CompressionStream output without reading it back; decodeSave maps every decompression failure to "This file is not a trainsworld save."
- `app/src/workers/clockClient.ts`: fileChoice also maps any decode failure to the generic not-save text.
- `../shonei-server/trainworld_games.go`: twCheckSave decompresses only four bytes and checks TWG1. It never consumes the stream to EOF, so missing endings/checksums are accepted. This allows already-damaged saves to sync; it is not evidence the cloud truncated the upload.
- `app/src/workers/clock.worker.ts`: a failed stored-game load preserves the unreadable slot and falls back to a fresh game. Day 1 is January 27. This can explain an empty January 27 world after a failed load; it does not establish Jojo's cause.

## Jojo and other reports

No account matched jojo, jojo_nnd or KitaOut in password usernames or leaderboard names. An in-game account name or feedback thread is needed to identify their cloud saves. No emails, password hashes or session tokens were returned. The feedback summaries available from the inbox during the investigation yielded only Jordan's report when filtering for English/French/Japanese/Chinese save/load terms; this does not rule out other reports elsewhere.

## Follow-up choices

1. Prevention: verify newly compressed bytes round-trip to the original before replacing a local save or rotating its backups; provide a working compressor fallback for affected engines. On the server, consume gzip to EOF with an uncompressed-size cap before accepting a new revision. A server rejection alone does not protect browser-local saves.
2. Recovery: Paris may permit an explicit salvage copy using its intact base network and complete extensions before the cut-off timing block. This must be tested in isolation and must not be described as lossless: later extensions and data never emitted by the compressor are unavailable. The old Loire save needs deeper partial reconstruction because its base line section is truncated. Do not silently append a footer and claim repair.
3. Diagnostics: retain Safari's point version and compression/decompression error details in feedback, distinguish damaged save from wrong file, and make the blank-world fallback clearer.

The current Loire-Atlantique sandbox 2 and its backups pass the file-integrity checks, but Jordan's ability to open that particular game in either browser remains unverified.

## Local fix requested by Anita, 2026-11-15

Implemented in the app; published as v0.3.5 on Anita's later explicit go (release details below):

- Pin pako 3.0.3 as the bundled gzip implementation. `workers/gzipCore.ts` compresses and strictly inflates, checking end-of-stream, checksum and the output cap where supplied. Compression round-trips and compares every original byte before returning, so the existing autosave failure path keeps the previous slot and backups if verification fails.
- `workers/gzip.ts` runs the codec directly in the clock worker. Calls from the UI use a reusable `gzip.worker.ts`, keeping compression/inflation off the UI thread. Request failures reject without stranding other requests; input buffers are copied before transfer so a failed operation cannot detach the caller's only save bytes.
- `workers/saveFile.ts` uses this codec for saves and imports. `game/sharedSave.ts` uses the same strict decoder with its existing Community output cap. `diagram/shared.ts` uses checked gzip for the Community drawing, which otherwise has the same native compression bug. TWG1/TWT6 and normal gzip remain unchanged.
- Feedback retains Safari's full Version value, including point release. No in-game strings added or changed.

Checks:

- `npm run check --prefix trainworld/app`: pass.
- `node trainworld/checks/safari-save.mjs`: pass. Independent Node zlib checks new files; bundled decoding checks Node-created files. Cases cover 35,578/35,579 bytes around WebKit's published threshold, 48,891 bytes, 64 KiB, 2 MiB, empty input, deterministic output, missing/truncated streams, CRC and length-trailer corruption, non-gzip input and decompression output limits. Original NYC header and track bytes round-trip unchanged. Native compression/decompression are replaced by faulty implementations and never called.
- `node trainworld/checks/safari-save-browser.mjs http://localhost:6186/`: pass. Temporary Chrome profile, native compression/decompression disabled in the page and every module worker: import NYC's 109 lines; write a slot and backup and decode both; export a file independently accepted by zlib; create a Community save copy without changing its network; reject truncated input and a capped inflate, then successfully reuse the codec worker; keep a UI timer running during a large compression; reload and reopen all 109 lines. Safari 16.4.1 is retained by feedback diagnostics.
- `node trainworld/checks/t116-save-check.mjs`: pass, legacy NYC and Northeast save round-trips.
- Existing `t1826-saves.mjs` was also run. Its newer-save preservation, unchanged export, backup-before-cloud-replacement, newer-copy, closing-warning and old-header behavior pass. The suite exits failing because it expects the old newer-save notice wording and four separately listed backup rows (the storage tiers themselves all exist), and because external network calls are blocked in this environment. These are not treated as a green full-suite result. Its source was left unchanged.
- `git diff --check` on the changed app/docs: pass.

Test limitation: no actual Safari 16 binary/device was available; the regression demonstrates independence from faulty native browser codecs in both the real clock worker and main-page flows. Jordan's precise Safari point version when the damaged saves were written is still unknown.

No production saves changed and no recovery was attempted. The server's four-byte-only upload validation remains a separate hardening follow-up; it does not prevent the client fix from protecting future writes once published. Existing truncated files still need explicit recovery, not permissive normal loading. Jojo's identification awaits their feedback, as Anita confirmed.

## Incident archive requested by Anita, 2026-11-15

Anita explicitly authorized finding affected cloud saves and pre-corruption backups, archiving them for future player restoration, and filing an automatic detection/preservation task. Filed **T-3395 [P1], Detect corrupted saves and preserve recovery copies**, in TASKS.md with requirements and acceptance checks in notes/T-3395.md. It remains open; no recurring scan or server validation change has been deployed.

Ran `tools/ops/archive_bad_saves.py` on the server at reduced CPU priority. It reads live files and writes only a new private archive outside the live store and normal backup rotation. The scan completed in 190 seconds, covering 5,841 account directories, 14,101 indexed/orphan game groups, and 37,499 save revisions (2,377,155,447 compressed bytes). Each account's index is read as that account is visited, so this is not a simultaneous global snapshot. No index failures, missing referenced files or observed rotation races occurred.

- 37,493 revisions pass full gzip/checksum, JSON game header and supported track-magic checks. This is not full track-model/simulation validation.
- **3 current saves are confirmed truncated, across 2 accounts**: Jordan's two previously diagnosed saves, and one additional player's current revision 37.
- The additional affected game has **3 intact pre-corruption copies**, revisions 1, 18 and 19. Revision 19 is the newest surviving intact predecessor. All three are archived along with damaged revision 37. The manifest maps the account/game IDs to the precise original bytes; no account/email lookup was needed.
- Jordan's two damaged revision-1 saves have no older surviving cloud revisions. Both damaged originals are archived for later explicit salvage.
- One additional game's 3 revisions (1, 65, 66) exceed the scan's 64 MiB unpacking cap and were conservatively archived as **unverified, not corrupt**. A subsequent streaming check validates all three gzip checksums and endings: total raw sizes 68,333,795 / 76,213,443 / 76,659,455 bytes, almost entirely JSON headers. Their oversized JSON headers were not fully parsed in the bounded scan, and no simulation load was attempted.

Archive: **`/opt/shonei-recovery/save-integrity-2026-11-15/`** on the game server, owner-only directory permissions (0700). It holds **9 original save files**, 8,686,238 bytes total, plus original account-index snapshots and per-game recovery manifests. Every archived save was reread and verified against its SHA-256. All live files were left untouched; nothing was restored or replaced.

Independent package on that server: **`/opt/shonei-recovery/save-integrity-2026-11-15.tar.gz`**, 5,816,863 bytes, file permissions 0600, SHA-256 `a4c818c050371911909b54ab9703da50695536bd20ee6f8248fc669366a4ce4f`. The root manifest and each game's `.recovery.json` identify all revisions, validation results, the newest intact predecessor where present, sizes and checksums. Archive originals are not silently repaired.

Follow-up oversized-file checksum results are saved beside the archive in `/opt/shonei-recovery/save-integrity-2026-11-15-additional-validation.json` (0600). This supplement is separate from the original tar package; the original scan manifest intentionally retains its initial bounded-validation results.

A second local copy was initially rejected by automatic approval review pending explicit destination approval. Anita then answered **"Yes, make the local copy"** for `C:\Users\anita\projects\maps\artifacts\trainworld\save-recovery\2026-11-15`. The package and supplemental validation manifest were downloaded there after that approval. Its package SHA-256 matches the server, and all nine contained save files were independently checked against their manifest sizes and SHA-256 hashes without extracting player payloads. The folder also holds `manifest.json` and `local-verification.json` for recovery lookup and verification. Git confirms the destination is ignored by `/artifacts/trainworld/`; no private payloads were added to source control. The server archive remains in place.

Archive-tool synthetic checks passed: damaged gzip detection, newest intact predecessor selection, inclusion of orphaned older files, exact-byte archive/hash verification, all live source hashes unchanged, and rejection of an archive destination inside the live save store. Python compilation and diff whitespace checks pass.

Before restoring: find the account/game in the private manifest; verify the chosen file's SHA-256; validate/load it in isolation; restore as a separate player-controlled copy. Do not overwrite damaged originals or assume a gzip-valid candidate has passed simulation validation. No message has been sent to any player.

## v0.3.5 published, 2026-11-15

Anita: "ok, can you please publish?" Publishing is complete, with no account-server deploy or data upload.

- Maps source commit `bfe58c54f2aa016f0ce6ca632135466259f67751`; isolated release branch/tree `release/safari-save-035` / `.tw-trees/safari-save-035`.
- Website commit `c9b74a8239462a64a15b027525dd101fb9bebfc2` pushed to master. Pages run `43772084654` completed successfully.
- Public source sync `6b8ef91` pushed to circularsquare/trainsworld main. Private notes, operation tooling and recovery archives are excluded.
- Source smoke: all 28 checks passed. Built package smoke: all 28 in light and all 28 in dark passed, each with 109 lines, 1,756 stations, 4,498.6 km and 12,581,681 rail trips/day.
- Additional compiled-package check passed: file picker rejects a truncated file; valid file loads the new gzip worker; all 109 lines save/reopen; exported gzip independently validates with zlib. Browser-native save compression was disabled. No app-code changes were needed during these release checks.
- Built on freeze `20261113-013559`, unchanged simulation binaries from the matching 6feec4b4 snapshot (no sim source changes), market drives from the served snapshot, output `artifacts/trainworld/publish/v0.3.5/dist`. Reused published data `2026-11-13` and terrain v2; 8800/board was not swapped.
- Live https://anita.garden/trainsworld/ references `index-ur4F201n.js`. That entry, `clock.worker-Cbxq7smW.js`, and `gzip.worker-Ba-jtRGe.js` each match the tested package SHA-256. HTML cache status DYNAMIC. Range, data/terrain CORS, JSON encoding, script MIME and old-path redirect verification all pass. Result in `artifacts/trainworld/publish/v0.3.5/live-verification.json`.
- New release text: "update v0.3.5: fixed saving games in older Safari versions" and "fixed older Safari versions creating damaged saves; checks new saves before replacing the previous copy". Other languages currently fall back to English for these two new keys; i18n check passes.

Jordan's backup answer: their two corrupted games have no surviving cloud predecessors in the scan. They can check the **original browser/device**, open Saves, expand **other versions**, and choose **Open** on an older local entry if one exists. This creates a restored copy rather than requiring an operator restore. We cannot see or verify their browser-local backups remotely, so do not promise recovery or confuse their separate intact newer Loire game with a backup of Paris/the old Loire game. No message was sent to Jordan.
