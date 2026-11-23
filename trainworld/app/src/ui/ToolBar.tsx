// T-2282: the track and station tools' options on the map, in the selection bar's place (top centre,
// the map's bottom edge on a phone), shown only while that tool is on; T-2284: the click mode's
// colour as an aura round the map.

import { batch, computed, type Signal } from "@preact/signals";
import { useState } from "preact/hooks";
import type { ComponentChildren, JSX } from "preact";
import { distUnit, kmhFromShown, kmhToShown, levelLabel, levelName, millions, speed } from "../game/format";
import { linking } from "../game/links";
import { moneyState } from "../game/money";
import { CLICK_MODE_COLOURS, inkOn, LEVEL_COLOURS, legibleOn, TRACK_TYPE_COLOURS } from "../game/palette";
import { buildMode } from "../game/buildMode";
import { linesByNum, world } from "../game/world";
import { CAR_M, stepPlatform } from "../game/boarding";
import { carsText } from "./inspector/schedule";
import { autoRamp, buildAdvanced, buildLevel, buildPlatform, buildTramPlatform, clampDesign, DESIGN_SPEEDS, designSpeed, flyoverSpeed, keepPlacing, lineDraft, quadTrack, setDoubleTrack, setQuadTrack, setSingleTrack, setTrackType, singleTrack, shade, besideOn, snapBeside, snapReal, snapRealTram, snapRoads, snapStreets, stationChoice, stationKind, stationOnTram, stationOneTrack, TRAM_PLATFORM, tool, trackType, type StationChoice, type TrackType } from "../game/ui";
import { costBackground } from "../map/trackTagRows";
import { levelTip, multText, trackMult, unitPrice, upkeepAt } from "./trackPrice";
import { Icon, LevelButtons, Segmented, Stepper, TrackCountButtons } from "./widgets";
import { t, tRich } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";
import { FareBar } from "./FareZoneParts";

export type ClickMode = keyof typeof CLICK_MODE_COLOURS | "select";
/** What a click on the map does now: the tool, or picking a passage's other station (T-197). */
export const clickMode = computed<ClickMode>(() => (linking.value !== null ? "passage" : tool.value));

/** T-2687: the colour the click mode shows (the aura, the descriptor's square): the line's own while its
 * stops are edited, kept legible on the shade (`legibleOn`); the mode's fixed colour otherwise. */
export function clickColour(mode: ClickMode): string {
  if (mode === "select") return "transparent";
  if (mode === "track") return TRACK_TYPE_COLOURS[buildMode().mode]; // T-3693: the track type's colour
  const draft = lineDraft.value, w = world.value;
  if (mode === "line" && draft.line !== null && w) {
    const l = linesByNum(w).get(draft.line);
    if (l) return legibleOn(l.colour, shade.value === "dark");
  }
  return CLICK_MODE_COLOURS[mode];
}

const trackTypes = (): [TrackType, string][] => [[2, t("toolBar.regular")], ["hsr", t("toolBar.highSpeed")], ["tram", t("toolBar.tram")], ["busway", t("toolBar.busway")]];
const typeLabel = (tt: TrackType) => (tt === "busway" ? t("toolBar.busway") : tt === "tram" ? t("toolBar.tram") : tt === "hsr" ? t("toolBar.highSpeed") : t("toolBar.regular"));
const speedTip = () => t("toolBar.speedTip");

/** T-3121: the next step up or down from the speed set, which may be a typed one between steps. */
function stepSpeed(sig: Signal<number>, s: number) {
  const now = sig.value;
  const to = s > 0 ? DESIGN_SPEEDS.find((v) => v > now + 1e-6) : [...DESIGN_SPEEDS].reverse().find((v) => v < now - 1e-6);
  if (to === undefined) return false;
  sig.value = to;
  return undefined;
}

/** A target speed: a step's to the nearest 5, as a round km/h figure reads in mph; a typed one as typed. */
const speedShown = (v: number) => speed(v, (DESIGN_SPEEDS as readonly number[]).includes(v) ? 5 : 1);

/** A target speed's stepper, its value typed in the shown unit (T-3121): the design speed, or the
 * flyover target max speed (T-3169). */
function SpeedStepper({ sig = designSpeed, label = t("toolBar.targetSpeedLabel") }: { sig?: Signal<number>; label?: string }) {
  const v = sig.value;
  return <Stepper value={speedShown(v)} wide label={label} onStep={(s) => stepSpeed(sig, s)}
    type={{ text: String(Math.round(kmhToShown(v))), set: (n) => { sig.value = clampDesign(kmhFromShown(n)); } }} />;
}

/** T-3169: the track bar's Advanced settings, one label and stepper each. */
function AdvancedSettings() {
  return (
    <>
      <span class="tb-speed">
        <span data-tip={speedTip()}>{t("toolBar.targetSpeed")}</span>
        <SpeedStepper />
      </span>
      <span class="tb-speed">
        <span data-tip={t("toolBar.flyoverSpeedTip")}>{t("toolBar.flyoverSpeed")}</span>
        <SpeedStepper sig={flyoverSpeed} label={t("toolBar.flyoverSpeedLabel")} />
      </span>
    </>
  );
}

/** Two rows: the level, the ramp switch and the price; the track type, the three placing switches and
 * the Advanced expander, which adds a third, the two target speeds (T-3169). On a phone one row, the
 * level, type and speeds as one button each that opens its choices above the bar. */
function TrackBar() {
  const lv = buildLevel.value;
  const tt = trackType.value;
  // T-3378: a busway is drawn as off-street tram track, with no street snaps and always double
  const busway = tt === "busway";
  const tram = tt === "tram" || busway;
  const q = quadTrack.value;
  const one = singleTrack.value;
  const m = moneyState.value;
  const mult = trackMult(lv, tt, q, one);
  const upkeep = upkeepAt(lv, tt, m, q, one);
  const adv = buildAdvanced.value;
  const [open, setOpen] = useState<"level" | "type" | "speed" | null>(null);
  const toggle = (k: "level" | "type" | "speed") => setOpen(open === k ? null : k);
  return (
    <>
      <div class="tool-bar track-bar" role="group" aria-label={t("toolBar.track")} style={{ "--mode": CLICK_MODE_COLOURS.track }}>
        <div class="tb-row">
          <span class="muted tb-wide">{t("toolBar.level")}</span>
          <div class="tb-wide"><LevelButtons value={lv} tip={(l) => levelTip(l, tt, m, q, one)} onPick={(l) => (buildLevel.value = l)} /></div>
          <button class="btn lv on tb-narrow tb-level" style={{ "--lc": LEVEL_COLOURS[lv], "--lk": inkOn(LEVEL_COLOURS[lv]) }}
            data-tip={levelTip(lv, tt, m, q, one)} aria-label={t("toolBar.level")} aria-expanded={open === "level"} onClick={() => toggle("level")}>{levelLabel(lv)}</button>
          {!tram && <button class={"btn icon tb-ramp" + (autoRamp.value ? " on" : "")} aria-label={t("toolBar.ramp")} aria-pressed={autoRamp.value}
            data-tip={t("toolBar.rampTip")}
            onClick={() => (autoRamp.value = !autoRamp.peek())}><Icon.autoRamp /></button>}
          <p class="muted small price tb-wide" data-tip={t(busway ? "toolBar.priceTipBusway" : tram ? "toolBar.priceTipTram" : "toolBar.priceTip", { level: levelName(lv), hsr: tt === "hsr", street: tram && snapStreets.value }) + (upkeep ? "\n" + upkeep : "")}>
            {tRich("toolBar.price", { price: millions(unitPrice(lv, tt, q, one)), unit: distUnit(), cost: <span class={costBackground(mult) ? "cost-mult tinted" : "cost-mult"} style={{ background: costBackground(mult) }}>{t("toolBar.cost", { mult: multText(lv, tt, q, one) })}</span> })}
          </p>
        </div>
        <div class="tb-row">
          <div class="tb-wide"><Segmented<TrackType> options={trackTypes()} value={tt} onChange={setTrackType} /></div>
          <button class="btn tb-narrow tb-type" aria-label={t("toolBar.trackType")} aria-expanded={open === "type"} onClick={() => toggle("type")}>{typeLabel(tt)}</button>
          {!busway && <TrackCountButtons value={q ? null : one ? 1 : 2}
            onPick={(n) => n === 1 ? setSingleTrack(true) : setDoubleTrack()}>
            <button class={"btn icon" + (q ? " on" : "")} aria-label={t("toolBar.quadTrack")} aria-pressed={q} data-tip={t("toolBar.quadTrackTip")}
              onClick={() => setQuadTrack(true)}><Icon.tracks4 /></button>
          </TrackCountButtons>}
          <span class="build-toggles">
            <button class={"btn icon" + (keepPlacing.value ? " on" : "")} aria-label={t("toolBar.keepPlacing")} aria-pressed={keepPlacing.value}
              data-tip={t(mobileDevice() ? "toolBar.keepPlacingTipTouch" : "toolBar.keepPlacingTip")}
              onClick={() => (keepPlacing.value = !keepPlacing.peek())}><Icon.keepPlacing /></button>
            {busway ? null : tram ? (
              <>
                {/* T-3069: tram track's own real network snap (tram and light rail), beside Snap to streets; the two exclude each other */}
                <button class={"btn icon" + (snapRealTram.value ? " on" : "")} aria-label={t("toolBar.snapReal")} aria-pressed={snapRealTram.value}
                  data-tip={t("toolBar.snapReal")}
                  onClick={() => { const on = !snapRealTram.peek(); batch(() => { snapRealTram.value = on; if (on) snapStreets.value = snapBeside.value = false; }); }}><Icon.snapReal /></button>
                <button class={"btn icon tb-streets" + (snapStreets.value ? " on" : "")} aria-label={t("toolBar.snapStreets")} aria-pressed={snapStreets.value}
                  data-tip={t("toolBar.snapStreets") + (mobileDevice() || snapRealTram.value ? "" : "\n" + t("toolBar.snapStreetsCtrl", { on: snapStreets.value }))}
                  onClick={() => { const on = !snapStreets.peek(); batch(() => { snapStreets.value = on; if (on) { snapRealTram.value = false; setQuadTrack(false); } }); }}><Icon.snapStreets /></button>
              </>
            ) : (
              <>
                <button class={"btn icon" + (snapReal.value ? " on" : "")} aria-label={t("toolBar.snapReal")} aria-pressed={snapReal.value}
                  data-tip={t("toolBar.snapReal")}
                  onClick={() => { const on = !snapReal.peek(); snapReal.value = on; if (on) snapBeside.value = snapRoads.value = false; }}><Icon.snapReal /></button>
                <button class={"btn icon" + (snapRoads.value ? " on" : "")} aria-label={t("toolBar.snapRoads")} aria-pressed={snapRoads.value}
                  data-tip={t("toolBar.snapRoads") + (mobileDevice() ? "" : "\n" + t("toolBar.snapRoadsCtrl"))}
                  onClick={() => { const on = !snapRoads.peek(); snapRoads.value = on; if (on) snapBeside.value = snapReal.value = false; }}><Icon.snapRoads /></button>
              </>
            )}
            {/* beside is shared by both track types, so it turns off both real snaps (and roads), whichever type is picked later */}
            {!(tram && snapStreets.value) && <button class={"btn icon" + (besideOn.value ? " on" : "")} aria-label={t("toolBar.snapBeside")} aria-pressed={besideOn.value}
              data-tip={t("toolBar.snapBesideTip")}
              onClick={() => { const on = !besideOn.peek(); batch(() => { snapBeside.value = on; if (on) snapReal.value = snapRoads.value = snapRealTram.value = false; }); }}><Icon.snapBeside /></button>}
          </span>
          {!tram && <button class={"btn text tb-wide tb-advanced" + (adv ? " open" : "")} aria-expanded={adv}
            onClick={() => (buildAdvanced.value = !buildAdvanced.peek())}><Icon.chevron />{t("toolBar.advanced")}</button>}
          {!tram && <button class="btn tb-narrow tb-speed-btn" data-tip={`${t("toolBar.targetSpeed")}\n${speedTip()}`} aria-label={t("toolBar.targetSpeed")} aria-expanded={open === "speed"}
            onClick={() => toggle("speed")}>{speedShown(designSpeed.value).split(" ")[0]}</button>}
        </div>
        {!tram && adv && <div class="tb-row tb-wide tb-adv-row">
          <AdvancedSettings />
        </div>}
      </div>
      {open === "level" && (
        <div class="tool-bar-pop touch-levels">
          <LevelButtons value={lv} tip={(l) => levelTip(l, tt, m, q, one)} onPick={(l) => { buildLevel.value = l; setOpen(null); }} />
        </div>
      )}
      {open === "type" && (
        <div class="tool-bar-pop">
          <Segmented<TrackType> options={trackTypes()} value={tt} onChange={(ty) => { setTrackType(ty); setOpen(null); }} />
        </div>
      )}
      {open === "speed" && (
        <div class="tool-bar-pop tb-pop-speeds">
          <AdvancedSettings />
        </div>
      )}
    </>
  );
}

/** T-3999, T-4010: what the station tool places, a train station, a tram stop, a BRT station or a port. */
const stationKinds = (): [StationChoice, string, () => JSX.Element][] => [["auto", t("toolBar.kindAuto"), () => <svg viewBox="0 0 16 16"><path d="M3.2 13.5 8 2.5l4.8 11M5 9.6h6" /></svg>], ["rail", t("toolBar.kindRail"), Icon.train], ["tram", t("toolBar.kindTram"), Icon.tram], ["bus", t("stationInspector.brtStation"), Icon.brt], ["port", t("stationInspector.port"), Icon.anchor]];

function StationBar() {
  const tram = stationOnTram.value;
  const len = tram ? buildTramPlatform : buildPlatform;
  const kind = stationChoice.value;
  const shown = stationKind.value;
  const [open, setOpen] = useState(false);
  const kinds = (pick: (k: StationChoice) => void, cls = "") => (
    <div class={"group tb-kind" + cls} role="group" aria-label={t("toolBar.stationKind")}>
      {stationKinds().map(([k, label, I]) => (
        <button class={"btn icon" + (kind === k ? " on" : "")} aria-label={label} aria-pressed={kind === k} data-tip={label} onClick={() => pick(k)}><I /></button>
      ))}
    </div>
  );
  const [, kindLabel, KindIcon] = stationKinds().find(([k]) => k === kind)!;
  return (
    <>
    <div class="tool-bar station-bar" role="group" aria-label={t("toolBar.station")} style={{ "--mode": CLICK_MODE_COLOURS.station }}>
      <div class="tb-row">
        {kinds((k) => (stationChoice.value = k), " tb-wide")}
        {/* on a phone the kind is one button that opens the four above the bar, as the track type */}
        <button class="btn icon tb-narrow" aria-label={kindLabel} aria-expanded={open} onClick={() => setOpen(!open)}><KindIcon /></button>
        {(shown === "rail" || shown === "tram") && <>
          <span class="tb-wide">{t("toolBar.platformLength")}</span>
          <Stepper value={carsText(Math.round(len.value / CAR_M))} wide label={t("toolBar.platformLengthLabel")} onStep={(s) => {
            const to = tram ? Math.max(TRAM_PLATFORM.min, Math.min(TRAM_PLATFORM.max, len.value + s * TRAM_PLATFORM.step)) : stepPlatform(len.value, s);
            if (to === len.value) return false;
            len.value = to;
            return undefined;
          }} />
          <PlatformTracks one={stationOneTrack.value} onPick={(one) => (stationOneTrack.value = one)} />
        </>}
      </div>
    </div>
    {open && <div class="tool-bar-pop">{kinds((k) => { stationChoice.value = k; setOpen(false); })}</div>}
    </>
  );
}

/** T-3332: a station's platforms on one track or on both, the station bar's and the station panel's. */
export function PlatformTracks({ one, onPick, oneTip, bothTip, children }: { one: boolean; onPick: (one: boolean) => void; oneTip?: string; bothTip?: string; children?: ComponentChildren }) {
  return (
    <div class="group tb-count" role="group" aria-label={t("toolBar.platformTracks")}>
      <button class={"btn icon" + (one ? " on" : "")} aria-label={oneTip ?? t("toolBar.oneTrackStation")} aria-pressed={one} data-tip={oneTip ?? t("toolBar.oneTrackStation")}
        onClick={() => onPick(true)}><Icon.tracks1 /></button>
      <button class={"btn icon" + (!one ? " on" : "")} aria-label={bothTip ?? t("toolBar.bothTracksStation")} aria-pressed={!one} data-tip={bothTip ?? t("toolBar.bothTracksStation")}
        onClick={() => onPick(false)}><Icon.tracks2 /></button>
      {children}
    </div>
  );
}

/** The map's tool bar for the track and station tools, and the aura while a click does anything
 * but select (the selection bar is `BlueprintBar`). */
export function MapToolBar() {
  const mode = clickMode.value;
  const tl = tool.value;
  return (
    <>
      {mode !== "select" && <div class="mode-aura" style={{ "--mode": clickColour(mode) }} />}
      {tl === "track" ? <TrackBar /> : tl === "station" ? <StationBar /> : tl === "fares" ? <FareBar /> : null}
    </>
  );
}
