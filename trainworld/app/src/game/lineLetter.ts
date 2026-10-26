/** The chip of a line with no symbol set (T-330): the number of "Line 7" or "Bus 10" (T-1475), a
 * short code as it is ("6X", "SIR", "HBLR", "c2c"), else the name's first letter. */
export function letterOf(name: string): string {
  const m = /^(?:Line|Bus) (\S{1,2})$/.exec(name);
  if (m) return m[1];
  const t = name.trim();
  if (/^[A-Z0-9]{1,5}$/.test(t) || /^(?=.*\d)[A-Za-z0-9]{1,5}$/.test(t)) return t;
  return (t[0] ?? "?").toUpperCase();
}
