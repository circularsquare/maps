# Demand model: cost and accuracy measurements

Moved out of docs/demand.md 4.4 in the T-588 tidy (2026-10-17): the numbers behind the kernel's cost, taken on the real New York pack with the hand-made network and on the combined save. The rules they led to are in docs/demand-kernel.md and docs/demand-runtime.md.

**Cost.** One period on the 202,337-cell New York pack and the hand-made network, one stage (the
free-flow estimate or one crowding round), one thread: 0.25-0.3 s native, 0.27 s in WASM (0.23-0.24
before T-090's wider reach: the extra origin subzones each build their station row and visit the
destination zones, and the drive costs a little more per pair; a day is 15% longer in WASM). An
edit reruns everything except gravity. Periods are independent, so the game runs them in a pool
of workers ([architecture.md](architecture.md) 7): with three workers in Chrome the mode split updated 0.49-0.55 s after a
new network reached them and the third crowding round landed by 1.9-2.0 s at T-078; T-090 adds
about 15% (not measured in Chrome). Each worker holds its own copy of the city (one for its commutes and everyday layer together): 18 MB of WASM
memory open, about 23 MB after solving (notes/T-026.md). New York's real 867-station network is
13.8M pairs, 1.0-1.7 s a stage in WASM, 24-25 s for a whole day on one worker (20 s before T-090).

**Accuracy of subzones** against one subzone per cell with the same model (T-090, hand-made
network, free flow): rail total within 0.2%, station entries and exits within 5.0% and 6.3% (sum
of absolute differences), segment loads within 3.0%. The logit roughly halved the station error
of all-or-nothing choice at any grouping, which is why the 5-minute walk bands are gone; what is
left is the grouping (a subzone's walk is one figure for all its cells). A plain mean of the
cells' perceived walks put rail 3.3% under, since the walk's cost is convex.

**Cost on the real combined network** (T-119: the 109-line New York save plus one Philadelphia
line, 875 demand stations, 5,039 subzones, 13.9M subzone pairs a period; native, one period, free
flow or one round, three populations): 1.75 s before. Strategy searches took 0.28 s for each of
the three populations (Philadelphia's has 35 subzones but searched towards all 875 stations); New
York's mode choice 0.81 s; the market's pair loop 0.06 s. Now 1.25 s: one search (0.29 s), the
mode choice, and Philadelphia's 2 destinations. Results are identical to floating point noise
(max difference 2e-16 of each array's largest value over line loads, boardings, entries and the
summaries of all five periods). Since T-172 (fixing T-121) the market's free-flow estimate also
routes with the distance fare, as the local populations' does, so round 0 shares their search
too; before, its own round-0 search towards every station a far zone can reach took 1.8 s natively
a period on the 109-line save (against 0.1 s for a crowding round), and in five Chrome workers
at once made round 0 four times as long as a crowding round (10 s against 2.3 s).

In Chrome on this machine (16 threads), the real network plus the Philadelphia line:

| | before (3 workers) | 3 workers | 5 workers |
|---|---|---|---|
| free-flow estimate published | 4.5 s | 3.2 s | 1.7 s |
| round three published | 17.2 s | 11.0 s | 6.1 s |
| bubbles drawn | 17.4 s (after round three) | 3.4 s | 1.9 s |
| WASM memory per worker | 129-132 MB | 120-134 MB | 120 MB |

Each period's stage in a worker takes about 1.5 s (2.1-2.3 s before). Historical single-city
measurements above retain their context.

**Memory follow-up** (T-123). The regional JSON pack now parses into flat numeric rows, skipping
metadata; temporary rows are consumed before constructing pairs. Immutable pair and lookup
vectors release their unused capacity. This preserves all trip parameters and results. On the
current full-network fixture, opening the five solving workers plus the auxiliary uses
360 MB against 596 MB before; each solving worker after the solve uses 127–128 MB against
137 MB. Estimated pool memory falls by 46 MB (6.4%) to about 668 MB, including the auxiliary's
opening allocation. These are WASM linear-memory measurements, not browser RSS. Live network,
strategy and period arrays still set the full-network peak. Comparison: notes/T-123.md.
