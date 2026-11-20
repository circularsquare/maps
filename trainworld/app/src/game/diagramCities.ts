// T-3409: the cities (zones) the network diagram is of, none for the whole network. Saved with the game (the header's
// `schematic`, kept by the clock worker); diagram/view.ts tells the clock worker which stations are in them.

import { signal } from "@preact/signals";

/** The chosen zone ids; empty for the whole network. */
export const diagramCities = signal<readonly string[]>([]);
/** Counts the loads, each of which brings the game's own choice and wants the stations in it sent again. */
export const diagramCitiesLoads = signal(0);

/** The game's choice as the clock worker has it, on a load. */
export function takeDiagramCities(cities: readonly string[]) {
  diagramCities.value = [...cities];
  diagramCitiesLoads.value++;
}
