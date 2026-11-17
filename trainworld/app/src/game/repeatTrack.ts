// T-3562: a line whose run goes over the same piece of track twice in the same direction. Going over a
// piece out and back (opposite directions) is how every line works and is not flagged.

/** The edge where a run's path first repeats a piece in the same direction (a path entry is
 * `edge << 1 | direction`), or null. A line with a separate return path is checked run by run. */
export function repeatedEdge(l: { path: readonly number[]; back?: readonly number[] }): number | null {
  for (const run of [l.path, l.back]) {
    if (!run) continue;
    const seen = new Set<number>();
    for (const p of run) {
      if (seen.has(p)) return p >> 1;
      seen.add(p);
    }
  }
  return null;
}
