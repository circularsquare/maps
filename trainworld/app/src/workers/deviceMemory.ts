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

/** GB, as `navigator.deviceMemory` reports it; 4 for a phone or tablet that does not report it;
 * undefined for any other device that does not (desktop Safari and Firefox) */
export function deviceGB(): number | undefined {
  const dm = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return dm ?? (mobileDevice() ? 4 : undefined);
}

/** A phone or tablet, or a device of 4 GB or less: one solving worker and no focus pool */
export function smallDevice(): boolean {
  const gb = deviceGB();
  return mobileDevice() || (gb !== undefined && gb <= 4);
}
