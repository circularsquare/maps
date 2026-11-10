// The inspector shows whatever is selected, and is blank when nothing is (SPEC 8).

import { selection } from "../../game/ui";
import { AchievementsView } from "./AchievementsView";
import { AllCitiesView } from "./AllCitiesView";
import { CityInspector } from "./CityInspector";
import { CommutersInspector } from "./CommutersInspector";
import { LineInspector } from "./LineInspector";
import { GroupInspector } from "./GroupInspector";
import { BusStopInspector } from "./BusStopInspector";
import { TripInspector } from "./TripInspector";
import { JunctionInspector } from "./JunctionInspector";
import { StationInspector } from "./StationInspector";
import { TrackInspector } from "./TrackInspector";
import { TrainInspector } from "./TrainInspector";

export function Inspector() {
  const s = selection.value;
  let body = null;
  if (!s) body = null; // nothing selected: the area stays, empty
  else if (s.kind === "line") body = <LineInspector key={s.line} id={s.line} />;
  else if (s.kind === "group") body = <GroupInspector key={s.group} group={s.group} />;
  else if (s.kind === "busStop") body = <BusStopInspector key={s.lines.join()} lines={s.lines} stop={s.stop} />;
  else if (s.kind === "station") body = <StationInspector id={s.station} />;
  else if (s.kind === "train") body = <TrainInspector line={s.line} profile={s.profile} dep={s.dep} />;
  else if (s.kind === "track") body = <TrackInspector edge={s.edge} />;
  else if (s.kind === "junction") body = <JunctionInspector node={s.node} />;
  else if (s.kind === "commuters") body = <CommutersInspector />;
  else if (s.kind === "city") body = <CityInspector key={s.city} id={s.city} />;
  else if (s.kind === "world") body = <AllCitiesView />;
  else if (s.kind === "achievements") body = <AchievementsView />;
  else body = <TripInspector />;
  return <section id="inspector">{body}</section>;
}
