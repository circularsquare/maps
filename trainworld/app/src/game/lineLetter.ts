/** The chip of a line with no symbol set (T-330): the number of "Line 7" or "Bus 10" (T-1475), a
 * short code as it is ("6X", "SIR", "HBLR", "c2c"), else the name's first letter. T-4226: also the number
 * of the default name in the player's language ("Linie 7", "路線7", "7호선"), from the same message that
 * makes it. */
let own: RegExp | null = null;

/** The message that makes a new line's name ("Line {n}") in the language shown, set by i18n when the
 * language changes; a message of another shape is ignored. */
export function setDefaultNameMessage(msg: string | undefined) {
  const parts = msg?.split("{n}");
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  own = parts?.length === 2 && !/[{}']/.test(parts.join("")) ? new RegExp(`^${esc(parts[0])}(\\S{1,2}?)${esc(parts[1])}$`) : null;
}

export function letterOf(name: string): string {
  const m = /^(?:Line|Bus) (\S{1,2})$/.exec(name) ?? own?.exec(name);
  if (m) return m[1];
  const t = name.trim();
  if (/^[A-Z0-9]{1,5}$/.test(t) || /^(?=.*\d)[A-Za-z0-9]{1,5}$/.test(t)) return t;
  return (t[0] ?? "?").toUpperCase();
}
