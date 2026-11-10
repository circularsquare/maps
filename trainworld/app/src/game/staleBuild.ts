// T-2962: a tab left open across a publish asks for hashed files (a language's messages, a worker) that the
// server no longer has. Whatever fails that way calls `noteStaleBuild`, and ui/NewVersion.tsx offers a reload.

import { signal } from "@preact/signals";

/** a lazy file did not load, so this tab is probably older than the files on the server */
export const staleBuild = signal(false);

export function noteStaleBuild(e?: unknown) {
  if (e !== undefined) console.warn("a file of this page did not load:", e);
  staleBuild.value = true;
}

/** Vite fires this on window when a dynamic import (or the files it depends on) fails to load; the error
 * is then thrown to the caller as usual. */
if (typeof window !== "undefined") {
  addEventListener("vite:preloadError", (e) => noteStaleBuild((e as Event & { payload?: unknown }).payload));
}
