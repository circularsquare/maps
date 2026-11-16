// T-1764 rule 2: a clipped line of at most 3 stops in a view does not touch it on trains an hour, run
// times or span; a longer one does. Run: node --experimental-strip-types trainworld/checks/touch-few-stops.mjs
import { sameView } from "../app/src/workers/demandView.ts";

// one line of n stops; pads at both ends when clipped
function view(n, { clipped, tph = 4, run = 60, span = 1 } = {}) {
  const net = {
    lineN: Uint32Array.of(n), stops: Uint32Array.from({ length: n }, (_, i) => i),
    times: new Float32Array(6 * n).fill(run), tph: Float32Array.of(tph, tph, tph),
    span: new Float32Array(2 * n * 6).fill(span), cars: Float32Array.of(5), hopKm: new Float32Array(2 * n).fill(1),
  };
  return { net, stLocal: new Int32Array(0), nodeGlobal: new Uint32Array(2 * n), pads: clipped ? Uint32Array.of(0, n) : new Uint32Array(0) };
}
let bad = 0;
const is = (name, got, want) => { if (got !== want) { bad++; console.log("FAIL", name, got, "want", want); } else console.log("ok  ", name); };
for (const [what, edit] of [["trains an hour", { tph: 8 }], ["run times", { run: 90 }], ["span", { span: 0.5 }]]) {
  is(`3 stops clipped, ${what}: same view`, sameView(view(3, { clipped: true }), view(3, { clipped: true, ...edit })), true);
  is(`4 stops clipped, ${what}: touched`, sameView(view(4, { clipped: true }), view(4, { clipped: true, ...edit })), false);
  is(`3 stops whole (no pads), ${what}: touched`, sameView(view(3, {}), view(3, { ...edit })), false);
}
process.exit(bad ? 1 : 0);
