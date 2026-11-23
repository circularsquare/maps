// T-4109: the track model stopped for good (too many panics, workers/clockPanic.ts): the notice at the top centre
// (ui/ModelStopped.tsx) says to reload. Set once by the clock client; an edit after it is refused with nothing more said.

import { signal } from "@preact/signals";

export const modelStopped = signal(false);
