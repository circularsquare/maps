// A line colour can come from a save and is written into SVG markup (busDraw, ferryDraw): only a
// plain CSS colour goes through (T-2914).
const PLAIN = /^(#[0-9a-f]{3,8}|[a-z]{3,20}|(rgb|hsl)a?\([\d\s.,%/-]{1,40}\)|var\(--[\w-]{1,30}\))$/i;
export const safeColour = (c: string, fallback = "#888"): string => (PLAIN.test(c) ? c : fallback);
