// T-3759: whether new stations start bound to their complex's name, the player's setting (Settings, Game),
// remembered in this browser with the other settings (`DisplaySettings.bindNew`) and used in every game. The
// clock worker hears it after each load and on each change; changing it never changes a station already placed.

import { computed } from "@preact/signals";
import { display, setDisplay } from "./ui";

export const bindNewStations = computed(() => display.value.bindNew !== false);

/** The player's change. */
export function setBindNew(on: boolean) {
  setDisplay("bindNew", on);
}
