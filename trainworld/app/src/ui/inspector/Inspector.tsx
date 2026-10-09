// The inspector shows whatever is selected, and is blank when nothing is (SPEC 8).

import { selection } from "../../game/ui";
import { CommutersInspector } from "./CommutersInspector";
import { LineInspector } from "./LineInspector";
import { AreaInspector, JunctionInspector, StationInspector, TrackInspector, TrainInspector } from "./others";

export function Inspector() {
  const s = selection.value;
  let body = null;
  if (!s) body = null; // nothing selected: the area stays, empty
  else if (s.kind === "line") body = <LineInspector key={s.line} id={s.line} />;
  else if (s.kind === "station") body = <StationInspector id={s.station} />;
  else if (s.kind === "train") body = <TrainInspector line={s.line} profile={s.profile} dep={s.dep} />;
  else if (s.kind === "track") body = <TrackInspector edge={s.edge} />;
  else if (s.kind === "junction") body = <JunctionInspector node={s.node} />;
  else if (s.kind === "commuters") body = <CommutersInspector />;
  else body = <AreaInspector sel={s} />;
  return <section id="inspector">{body}</section>;
}
