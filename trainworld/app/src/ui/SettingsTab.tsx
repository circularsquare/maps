// Settings (T-037): display settings (SPEC 8) and the theme, remembered per browser: the commuter
// bubbles (T-097: off, where commuters live, where they work), track colour, trains, station names,
// capacity markers (T-031), map labels, the riders drawn on the network (T-084: line width by the
// track display, train fill; the station circles are in the Stations tab since T-213), the UI
// scale (T-247), and last the trip explorer's button (T-220). The saves are in their own panel
// (T-251, ui/SavesPanel.tsx). T-253: no explanatory notes, only controls and their labels.

import { levelLabel, speed } from "../game/format";
import { LEVEL_COLOURS, SPEED_STOPS } from "../game/palette";
import { LEVELS } from "../game/types";
import { display, setDisplay, theme, THEMES, tool, UI_SCALES, uiScale, type Theme } from "../game/ui";
import { pickTool, TRIP_KEY } from "./tabs";
import { Check, Segmented, Stepper } from "./widgets";

const THEME_NAME: Record<Theme, string> = { pink: "Pink", blue: "Blue", green: "Green" };

export function SettingsTab() {
  const d = display.value;
  return (
    <>
      {/* T-169: the slider is the section's only control, so the heading goes with it */}
      <div class="h">Demand bubbles</div>
      {/* T-470: the opacity applies to every bubble, so the section is always there */}
      <label class="slider">
        <span>Size</span>
        <input type="range" min="0.5" max="2" step="0.05" value={d.bubbleSize} data-tip={`Bubbles at ${Math.round(d.bubbleSize * 100)}% of their usual size`}
          onInput={(e) => setDisplay("bubbleSize", Number((e.target as HTMLInputElement).value))} />
      </label>
      <label class="slider">
        <span>Opacity</span>
        <input type="range" min="0.2" max="1" step="0.05" value={d.bubbleOpacity} data-tip={`Bubbles at ${Math.round(d.bubbleOpacity * 100)}% of their full opacity`}
          onInput={(e) => setDisplay("bubbleOpacity", Number((e.target as HTMLInputElement).value))} />
      </label>
      {/* T-222: longer-trip bubbles are always drawn; their setting is gone */}
      {/* T-464: the day the demand views show is chosen in the demand bar (ui/DemandSelector.tsx) */}
      <div class="h">Track display</div>
      <div class="track-modes"><Segmented<"line" | "traffic" | "height" | "speed">
        options={[
          ["line", "Line"],
          ["traffic", "Line + traffic thickness"],
          ["height", "Height"],
          ["speed", "Max speed"],
        ]}
        value={d.trackColour}
        onChange={(v) => setDisplay("trackColour", v)}
      /></div>
      {d.trackColour === "speed" && <ul class="speed-key">
        {SPEED_STOPS.map(([v, colour]) => <li><i style={{background: colour}} />{speed(v, 5)}</li>)}
      </ul>}
      {d.trackColour === "height" && (
        <ul class="levels-key">
          {LEVELS.map((l) => (
            <li>
              <i style={{ background: LEVEL_COLOURS[l] }} />
              {levelLabel(l)}
            </li>
          ))}
        </ul>
      )}
      <div class="h">Show</div>
      <div class="checks">
        <Check checked={d.trains} onChange={(v) => setDisplay("trains", v)}>
          Trains
        </Check>
        <Check checked={d.stationNames} onChange={(v) => setDisplay("stationNames", v)}>
          Station names
        </Check>
        <Check checked={d.capacity} onChange={(v) => setDisplay("capacity", v)}>
          Capacity markers
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          Map labels
        </Check>
      </div>
      {/* T-213: station demand circles and one circle per complex are in the Stations tab */}
      {/* T-164: each train's own load and each platform's crowd, from the boarding account;
          T-253: the note on how the counts are made is now the platform option's hover */}
      <div class="h">Riders on trains and platforms</div>
      <div class="checks">
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          How full each train is
        </Check>
        {/* T-219: the platform fills up as riders wait; the bands and dots options are gone */}
        <Check checked={d.platformCrowds !== "off"} onChange={(v) => setDisplay("platformCrowds", v ? "patch" : "off")}>
          <span data-tip="Grey fills a platform as riders wait, red when a full train leaves some behind. From street zooms.">How full each platform is</span>
        </Check>
      </div>
      <div class="h">Theme</div>
      <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      <UiScale />
      <div class="h">Units and performance</div>
      <div class="checks">
        <Check checked={d.usUnits} onChange={(v) => setDisplay("usUnits", v)}>
          <span data-tip="Miles, mph and feet. Off shows kilometres, km/h and metres.">US units</span>
        </Check>
        <Check checked={d.moreCores} onChange={(v) => setDisplay("moreCores", v)}>
          <span data-tip="Up to 5 demand workers, using more of the processor and about 400 MB of memory each. Counts from the next time the page loads.">Use more cores</span>
        </Check>
      </div>
      <TripStart />
    </>
  );
}

/** T-247: the size of the panels and their text (and of the map's name tags), remembered per browser. */
function UiScale() {
  const s = uiScale.value;
  const i = UI_SCALES.findIndex((v) => Math.abs(v - s) < 0.001);
  const step = (d: number) => setDisplay("uiScale", UI_SCALES[Math.max(0, Math.min(UI_SCALES.length - 1, (i < 0 ? UI_SCALES.indexOf(1) : i) + d))]);
  const pct = Math.round(s * 100);
  return (
    <>
      <div class="h">Panel and text size</div>
      <Stepper wide label="panel and text size" value={<span data-tip={`Panels and text at ${pct}% of their usual size`}>{pct}%</span>} onStep={step} />
    </>
  );
}

/** The trip explorer's button (T-220: here since the City tab went; the T key from anywhere). */
function TripStart() {
  const on = tool.value === "trip";
  return (
    <div class="pane-foot trip-start">
      <button class={"btn text" + (on ? " on" : "")} data-tip={`Click two places or stations on the map to see how people would travel between them (${TRIP_KEY.toUpperCase()})`} onClick={() => pickTool("trip")}>
        Explore a trip<span class="key muted">{TRIP_KEY.toUpperCase()}</span>
      </button>
    </div>
  );
}
