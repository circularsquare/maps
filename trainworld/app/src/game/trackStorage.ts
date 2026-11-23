// T-4126: in the track build (game/trackBuild.ts) localStorage and sessionStorage are views of the page's own under
// "track:", so its settings, last slot and sign-in never mix with the main build's on the same origin. Imported first by
// main.ts, before any module reads storage.

import { TRACK_BUILD } from "./trackBuild";

const PREFIX = "track:";

function prefixed(s: Storage): Storage {
  const own = (): string[] => {
    const keys: string[] = [];
    for (let i = 0; i < s.length; i++) { const k = s.key(i); if (k?.startsWith(PREFIX)) keys.push(k.slice(PREFIX.length)); }
    return keys;
  };
  return {
    get length() { return own().length; },
    key: (i: number) => own()[i] ?? null,
    getItem: (k: string) => s.getItem(PREFIX + k),
    setItem: (k: string, v: string) => s.setItem(PREFIX + k, v),
    removeItem: (k: string) => s.removeItem(PREFIX + k),
    clear: () => { for (const k of own()) s.removeItem(PREFIX + k); },
  };
}

if (TRACK_BUILD && typeof window !== "undefined") {
  for (const name of ["localStorage", "sessionStorage"] as const) {
    try {
      const real = window[name];
      Object.defineProperty(window, name, { value: prefixed(real), configurable: true });
    } catch { /* storage blocked: the page runs without it either way */ }
  }
  document.title = "track: " + document.title; // i18n-ignore: 8800 only
}
