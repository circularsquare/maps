// The network diagram's water as drawn (T-2169, T-2580; docs/ui-diagram.md "Water"): the sea and the rivers as path
// commands, made on the rubber sheet's carried map (sheetLines.ts `sheetMap`) from the water on the ground (waterReal.ts).

export interface DiagramWater {
  /** the rivers' centre lines as path commands (draw.ts `toPath`), one path per width, stroked with square-cut (butt)
   * ends, so a band's width steps square where it changes */
  rivers: Float64Array[];
  /** each path's width, layout units (T-2528) */
  bands: number[];
  /** per width, a stub (path commands, a hundredth of a pitch long) at each end of a band, stroked with round ends: the
   * band's own ends and where rivers meet stay round */
  riverEnds?: Float64Array[];
  /** the sea's outline as path commands, filled even-odd */
  sea?: Float64Array;
  /** where a band meets the sea or another band, the corner of land between them filled with water along an arc
   * (T-3833): path commands, filled nonzero over the sea and the bands */
  fillets?: Float64Array;
  /** checks only: the sea's rings as fitted */
  seaRings?: Float64Array[];
  /** checks only: counts of the river finding and the map's time, ms */
  stats?: Record<string, number>;
}
