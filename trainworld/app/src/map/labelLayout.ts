// Shared visibility and breathing room for network annotations.
export const MIN_LABEL_ZOOM = 12;
export function labelPadding(zoom: number): [number, number] {
  return zoom < 14 ? [24, 14] : [10, 6];
}
