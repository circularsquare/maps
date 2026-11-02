/** The nearest ancestor that scrolls. */
export function scrollerOf(el: Element | null): HTMLElement | null {
  for (let e = el?.parentElement ?? null; e; e = e.parentElement) if (e.scrollHeight > e.clientHeight + 1 && /auto|scroll/.test(getComputedStyle(e).overflowY)) return e;
  return null;
}
