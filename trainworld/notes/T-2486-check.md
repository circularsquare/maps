# T-2486 check (binzipchk)

Method: 62 zones (nyc phl bos tyo lon par sha sao lag tor, the 3 smallest and 2 largest packs, one pack of each odd array
count (34, 36, 39), 30 random under 3 MB) hard-linked into a scratch folder and packed with packzip.py; a scratch packs
folder (hard links of all of data/packs, the 62 packed + market-drives.bin gzipped) served by this tree's Vite on 6163 via
TW_PACKS and TW_MARKET_DRIVES. Raw = the default Vite. Scratch deleted afterwards.

## 1 Packed vs raw games: PASS
T-007-real-nyc.save, packed vs raw: tripsPerDay 895752175 both, railTripsPerDay 12558064 both, 109 lines, 1756 stations,
cash 6000 both, no console errors, no failed requests. New nyc game opens on packed. Schematic with water on: the pixel
diff of the diagram area is empty (packed vs raw, same viewport, same time after opening). market-drives.bin served gzipped
and the markets loaded. (Fresh nyc game fetches no zone pack until there are lines, so only the save exercises it.)

## 2 Range reads: PASS
In the page: fetchRange (packed server) against a plain Range fetch (raw server) over every array of 16 zones plus odd
spans (3-7, last 5 bytes, a 100 kB span from a third in): 706 ranges, 0 differ, 0 errors. Pan to 5 areas and the
schematic with water: no console errors, no 4xx/failed requests.

## 3 Corrupt data: FAIL (one case, low severity)
Real save, nyc pack swapped, load watched 25 s:
- truncated packed .bin (60%): "New York could not load" (unpackBin throws), nyc not opened. Trips fall to 827M / 203k rail
  with the alert shown (same as any failed pack).
- bin_gz index off by 3 bytes: could not load. Last member length +50000: could not load.
- raw .bin with a bin_gz header: could not load.
- raw header (stale, no bin_gz) with raw .bin: loads as before. Raw truncated: nyc opens and still alerts (as before).
- packed .bin with a header that has no bin_gz: the game never finishes loading (still "starting" after 150 s, no alert).
  That mismatch needs a header and bin from different uploads, which versioned prefixes prevent, but it is a hang. A guard in
  downloadPack (header without bin_gz and bin starting 1f 8b: throw) would turn it into the alert.

## 4 packzip round trip: PASS
62 zones: pack + check OK (0.52 of raw, drives 17.0 -> 10.0 MB); independent whole-gzip inflate of each output equals the raw
file and the parts are contiguous. members() (the refusal test) run over all 4765 registry zones plus the 5 other .bin in
data/packs (ase bmo lel scd tsx): none refused. No .bin without a .json, none in subfolders.

## 5 deploy.py: read only (dry-run not run: the permission check denied it, it would read the main tree's 8800 snapshot and list R2)
- Layout: stage_bins runs packzip pack then check on the whole build (--all), exit non-zero stops before any upload. Packed
  .bin go to the version prefix with `+ /*.bin`; the raw copy excludes `/*.bin` (top level only; no other .bin exist below the
  top level). Headers: stage_gz swaps in the packed headers for top-level .json whose packed copy exists, then gzips them with
  Content-Encoding as before. Stage takes about 2.3 GB under artifacts/trainworld/publish/<version>/bin (raw size 4.3 GB, 0.52).
- Order: bins up first, headers last, all under a new version prefix; a --resume into an old version that held raw files would
  briefly pair packed bins with raw headers (the hang above).
- Old clients: the old page names the old version prefix, so it keeps reading raw files there; new files only appear under the
  new prefix. The terrain blocks (fixed shared prefix) and .rgr/.save/world.water are untouched by this change.
- --verify: adds "a .bin is packed" (first two bytes of nyc.bin are 1f 8b). Gaps: it does not check that nyc.json has
  bin_gz, nor market-drives.bin, nor one inflated range. The existing "Range request raw" check still passes on a packed file.
- Dry run prints the summary's .bin group as "raw" (cosmetic).

Other notes: rangeFetch caches each pack's index per worker for its life (fine published, immutable; on 8800 a pack
rebuilt mid-session would keep the old index). On raw packs each worker makes one extra .json fetch.
Port 6164 answered from another tree's Vite during part of case 2 (mine did not stay up); it serves the same data/packs raw.

## Re-check (binzipchk2)

Method: scratch packs (hard links of data/packs, nyc.json/nyc.bin swapped per case, nyc packed with packzip.py, drives gzipped) on this tree's Vite, port 6173; T-007-real-nyc.save, 2 cores.
1. Packed .bin under a header without bin_gz: PASS. Game finishes loading (about 20 s), "New York could not load" shown, trips 827,223,527 / rail 203,136 (the failed-pack numbers). No hang. Other corrupt cases fail the same way within 15-25 s: truncated packed bin, bin_gz index +3 bytes, last member length +50000, raw bin with a bin_gz header; raw truncated also alerts. Raw header + raw bin: loads, 895,752,175 / 12,558,064, 109 lines.
2. Packed nyc on the real save: 895,752,175 trips/day, 12,558,064 rail, 109 lines, no "could not load", no errors in the play log. Matches raw. (Stations not re-read; same save and same numbers.)
3. deploy.py read only (not run): see report.
4. Master since base 1f61c1a5 (58 commits): demandClient.ts and docs/code-map.md changed on master, in other hunks (no overlap); marketRule.ts, diagram.worker.ts, rangeFetch.ts, deploy.py not changed on master.
