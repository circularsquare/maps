// T-3491: the coarse outlines' shared borders (map/coarseEdges.ts), worked out once off the main thread; the
// worker ends with its answer, so the tables it passes through go with it.
import { coarseTopology } from "../map/coarseEdges";
import type { PackedOutlines } from "../map/coarseOutlines";

self.onmessage = (e: MessageEvent<PackedOutlines>) => {
  const t = coarseTopology(e.data);
  (self as unknown as Worker).postMessage(t, [t.partner.buffer]);
};
