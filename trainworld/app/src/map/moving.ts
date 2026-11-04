// While the camera moves, work done for each MapLibre frame that need not be (DOM placers, platform
// crowds) runs at most every MOVING_PLACE_MS, and in full on the frame after the move ends (T-1207: a
// full placement of station names or capacity tags took 80-120 ms a frame at 6x CPU throttling).

export const MOVING_PLACE_MS = 100;

/** A placer's clock: `due()` is true with the camera still, and at most every MOVING_PLACE_MS while
 * it moves; `done()` after a full placement. */
export class PlaceThrottle {
  private at = 0;
  constructor(private moving: () => boolean) {}
  due(): boolean {
    return !this.moving() || performance.now() - this.at >= MOVING_PLACE_MS;
  }
  done() {
    this.at = performance.now();
  }
}
