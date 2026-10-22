# docs/architecture.md: reasons, comparisons and measurements

Moved out of docs/architecture.md in the doc tidy (T-940); the doc keeps the rules.

## 2. Why passengers, not trains, are the cost

Subway Builder's changelog points at passenger work: a timetable search per 200-person group,
"15 minute lag spikes" from every group re-deciding at once, freezes when drawing track on big
networks. It is Electron + React + MapLibre, so the map library is not the problem; bursty
per-group passenger work is.

NIMBY Rails hands the renderer a small motion snapshot each frame and routes passengers over a
graph of transfer stations only; it simulates individual riders, and overfull queues slow its
whole sim. The flow model is meant to go far past that.

## 7. Map: what the overlay is not

Track, stations and trains are not GeoJSON sources (per-frame `setData` pins a core) and not a
MapLibre custom layer that draws (that repaints the whole basemap on every train frame); the
custom layer is empty and only hands over the camera matrix (T-075).

## 7. Loading: how the hold came to wait for the final

T-172 released the loaded game's clock at the first estimate and T-328 at the cached result, but
the game then lagged through the crowding rounds, so Anita had it wait for the fresh solve's final
(T-347). A new game's card went about 2.7 s after navigation instead of about 7 s once its viewed
zone opened bare (T-351). Deriving the track of Japan's 709 lines took about 5 s, which is why the
save's stations are read and sent ahead (T-889).

## 11. Saves: sizes

- Money saved to the dollar and rides to a tenth gzip to about 60% of full precision (T-567).
- The shipped real networks (`packs/real/<id>.save`) are 7-80 KB each (T-731).
- The demand cache was 7.3 MB for the real New York save, 4.9 MB of it the bubbles; with a matching
  key the result showed about 0.1 s after the network was drawn, the bubbles at about 3.7 s
  (notes/T-328.md).
