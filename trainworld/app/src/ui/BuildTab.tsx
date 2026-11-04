// The Build tab: the tools and the blueprint quote. The track and station tools' options are on the
// map while the tool is on (ToolBar.tsx, T-2282).

import { tool } from "../game/ui";
import { Icon } from "./widgets";
import { BlueprintControls, BlueprintPick } from "./Blueprint";
import { BUS_KEY, FERRY_KEY, pickTool, TOOLS } from "./toolKeys";
import { newBusLine, newFerryLine } from "../game/lineActions";
import { live } from "../map/live";

/** The building tools, then the blueprint's quote and construction. */
export function BuildTab() {
  const t = tool.value;
  return (
    <>
      <div class="build-options">
        <div class="group tool-pick">
          {TOOLS.map(([id, label, I, key, title]) => (
            <button class={"btn text" + (t === id ? " on" : "")} data-guide={id === "track" ? "track" : id === "station" ? "stations" : undefined} data-tip={title.replace(/\n|$/, ` (${key})$&`)} onClick={() => pickTool(id)}>
              <I />
              <span class="tool-label">{label}</span>
              <span class="key muted">{key}</span>
            </button>
          ))}
          {/* T-1514: the bus tool, also on the Lines tab; it starts a new line, so it is no TOOLS entry */}
          <button class={"btn text tool-bus" + (t === "bus" ? " on" : "")} data-tip={`Draw a bus line (${BUS_KEY.toUpperCase()})`}
            onClick={() => (t === "bus" ? live.tools?.exit() : newBusLine())}>
            <Icon.bus />
            <span class="tool-label">Bus</span>
            <span class="key muted">{BUS_KEY.toUpperCase()}</span>
          </button>
          {/* T-2194: the ferry tool beside it, placing ports */}
          <button class={"btn text tool-ferry" + (t === "ferry" ? " on" : "")} data-tip={`Draw a ferry line (${FERRY_KEY.toUpperCase()})`}
            onClick={() => (t === "ferry" ? live.tools?.exit() : newFerryLine())}>
            <Icon.ferry />
            <span class="tool-label">Ferry</span>
            <span class="key muted">{FERRY_KEY.toUpperCase()}</span>
          </button>
        </div>
        <BlueprintPick />
      </div>
      <BlueprintControls />
    </>
  );
}
