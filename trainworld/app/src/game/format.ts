// Number and time formatting for the UI. "−" (U+2212) for negatives, as in the mock.

/** 312000 -> "312k", 1.84e6 -> "1.84M" */
export function count(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 1 : 2).replace(/\.?0+$/, "") + "M";
  if (n >= 1e3) return Math.round(n / 1000) + "k";
  return String(Math.round(n));
}

export function money(n: number): string {
  const a = Math.abs(n), s = n < 0 ? "−" : "";
  if (a >= 1e9) return s + "$" + (a / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return s + "$" + (a / 1e6).toFixed(2) + "M";
  if (a < 500) return "$0";
  return s + "$" + Math.round(a / 1000) + "k";
}

/** Three significant figures, for the drawing tag (T-085): "$25.5M", "$133M", "$1.33B". */
export function moneyShort(n: number): string {
  const a = Math.abs(n), s = n < 0 ? "−" : "";
  const sig = (v: number) => (v >= 100 ? Math.round(v).toString() : v >= 10 ? v.toFixed(1) : v.toFixed(2));
  if (a >= 1e9) return s + "$" + sig(a / 1e9) + "B";
  if (a >= 1e6) return s + "$" + sig(a / 1e6) + "M";
  return s + "$" + Math.round(a / 1000) + "k";
}

/** 1520 -> "1.5 km", 350 -> "0.35 km" */
export function km(m: number): string {
  return (m >= 1000 ? (m / 1000).toFixed(1) : (m / 1000).toFixed(2)) + " km";
}

/** headway in minutes from trains an hour */
export function headway(tph: number): string {
  if (!tph) return "none";
  const m = 60 / tph;
  return (Number.isInteger(m) ? m : m.toFixed(1)) + " min";
}

/** trains needed to run `tph` on a round trip of `rtS` seconds */
export function trainsNeeded(rtS: number, tph: number): number {
  return Math.ceil((rtS * tph) / 3600);
}

/** 83 -> "1m23s", 4330 -> "1h12m" */
export function duration(s: number): string {
  s = Math.round(s);
  if (s >= 3600) return `${Math.floor(s / 3600)}h${String(Math.floor(s / 60) % 60).padStart(2, "0")}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

export function clockTime(minute: number): string {
  const m = minute % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export function dayOf(minute: number): number {
  return Math.floor(minute / 1440);
}

export function levelLabel(l: number): string {
  return l > 0 ? "+" + l : l < 0 ? "−" + -l : "0";
}
