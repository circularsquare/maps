// T-2122: the device as the demand pool sizes itself by it (notes/T-2122.md). A phone's browser tab has
// far less memory than the phone: iOS kills an iPhone's tab near 2 GB however much the phone holds, and
// Chrome on Android below 8 GB is 32-bit, its page and workers sharing one 4 GB address space. Chrome and
// Edge report the device's memory (`navigator.deviceMemory`, a power of two, at most 8); Safari and
// Firefox do not.

let mobile: boolean | null = null;
/** A phone or tablet, by its user agent (iPadOS gives a desktop Mac's, with touch) */
export function mobileDevice(): boolean {
  if (mobile === null) {
    const ua = navigator.userAgent;
    mobile = /Android|iPhone|iPad|iPod|Mobi/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  }
  return mobile;
}

let bits32: boolean | null = null;
/** T-2867: a 32-bit browser, whose page and workers share about 2-4 GB of address space however much the
 * machine holds (notes/T-2867.md). The user agent cannot tell: 32-bit Firefox on 64-bit Windows says "Win64; x64",
 * as a 64-bit one does. A resizable ArrayBuffer is the tell: a 32-bit engine refuses a 4 GB maximum at once,
 * and a 64-bit one only reserves it (3 ms). A browser too old to know the option is taken as 64-bit.
 * `?bits32=1` or `0` says, for checks. */
export function addressLimited(): boolean {
  if (bits32 === null) {
    const q = new URLSearchParams(location.search).get("bits32");
    if (q === "0" || q === "1") bits32 = q === "1";
    else try {
      new (ArrayBuffer as unknown as new (n: number, o: { maxByteLength: number }) => ArrayBuffer)(0, { maxByteLength: 2 ** 32 });
      bits32 = false;
    } catch { bits32 = true; }
  }
  return bits32;
}

/** GB, as `navigator.deviceMemory` reports it; 4 for a phone or tablet that does not report it;
 * undefined for any other device that does not (desktop Safari and Firefox) */
export function deviceGB(): number | undefined {
  const dm = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return dm ?? (mobileDevice() ? 4 : undefined);
}

/** A phone or tablet, or a device of 4 GB or less: one solving worker and no focus pool, and the map's lean
 * ways (T-2794: its pixel ratio capped, the outlines kept compact) */
export function smallDevice(): boolean {
  const gb = deviceGB();
  return mobileDevice() || (gb !== undefined && gb <= 4);
}

/** T-2794: the most a small device draws the map and overlays at (DECISIONS.md 2026-11-08, notes/T-2794.md) */
const SMALL_MAX_RATIO = 2;
/** The pixel ratio the map is capped to here, null where the device's own is used. */
export function cappedPixelRatio(): number | null {
  return smallDevice() ? Math.min(SMALL_MAX_RATIO, window.devicePixelRatio || 1) : null;
}
