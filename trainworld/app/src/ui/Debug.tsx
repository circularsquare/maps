// Performance readout, only with ?debug=1. Updated four times a second from main.ts.

import { signal } from "@preact/signals";

export const perfText = signal("");

export function Debug() {
  return <pre id="debug">{perfText.value}</pre>;
}
