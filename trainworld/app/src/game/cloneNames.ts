// T-1872: the name and symbol a copied line starts with (DECISIONS 2026-10-30).

/** A name or symbol ending in a digit, or a digit and one letter: "Line 1", "Line 1a", "1", "1a". */
const NUMBERED = /^(.*\d)([A-Za-z]?)$/;
/** The longest symbol a clone is given; the same five characters `letterOf` shows as it is. */
const SYMBOL_MAX = 5;

/** "Line 1" -> "Line 1a", "Line 1a" -> "Line 1b": the next letter whose result is free, else null. */
function nextLetter(text: string, taken: (s: string) => boolean): string | null {
  const m = NUMBERED.exec(text);
  if (!m) return null;
  const from = m[2] ? m[2].toLowerCase().charCodeAt(0) + 1 : 97;
  for (let c = from; c <= 122; c++) {
    const s = m[1] + String.fromCharCode(m[2] && m[2] === m[2].toUpperCase() ? c - 32 : c);
    if (!taken(s)) return s;
  }
  return null;
}

/** The copy's name: "<base> clone N" with the lowest free N when it is itself a clone (Anita 2026-10-30:
 * "X clone 1" -> "X clone 2"), else the next letter after a trailing number, else "<name> clone N". */
export function cloneName(name: string, names: ReadonlySet<string>): string {
  const t = name.trim().replace(/ clone \d+$/, "");
  const lettered = t === name.trim() ? nextLetter(t, (s) => names.has(s)) : null;
  if (lettered) return lettered;
  let n = 1;
  while (names.has(`${t} clone ${n}`)) n++;
  return `${t} clone ${n}`; // i18n-ignore: a line's saved name
}

/** The copy's symbol: the next letter after a trailing number, else the old symbol's first letter and the
 * lowest number no other line's symbol uses ("t" or "tm" -> "t1", then "t2"). Taken symbols compare in lower case. */
export function cloneSymbol(symbol: string, symbols: ReadonlySet<string>): string {
  const taken = (s: string) => symbols.has(s.toLowerCase());
  const lettered = nextLetter(symbol, taken);
  if (lettered && lettered.length <= SYMBOL_MAX) return lettered;
  const first = [...symbol][0] ?? "?";
  let n = 1;
  while (taken(first + n)) n++;
  return first + n;
}
