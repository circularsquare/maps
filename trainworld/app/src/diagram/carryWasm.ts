// The carry's solver in WASM (sim/src/diagram.rs, T-2589) for sheetLines.ts, given the module's class: the same bits as
// the JavaScript one. The matrix is copied in once; a reweighting sends only the entries it changed.

import type { CarrySolver as WasmSolver } from "../wasm/track/sim";
import type { CarrySolverMaker } from "./sheetLines";

type SolverClass = new (nx: number, ny: number, a: Float64Array) => WasmSolver;

export const wasmCarrySolver = (Solver: SolverClass): CarrySolverMaker => (gw, gh, A) => {
  const s = new Solver(gw, gh, A);
  s.multigrid();
  return {
    changed: (at) => s.set(Uint32Array.from(at), Float64Array.from(at, (i) => A[i])),
    remake: () => s.multigrid(),
    solve: (b, x, tol, maxIt, on) => {
      const it = s.solve(b, x, tol, maxIt, on[0], on[1]);
      return [it[0], it[1]];
    },
    free: () => s.free(),
  };
};
