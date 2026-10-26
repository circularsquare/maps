// While the camera moves, work done for each MapLibre frame that need not be (DOM placers, platform
// crowds) runs at most every MOVING_PLACE_MS, and in full on the frame after the move ends (T-1207: a
// full placement of station names or capacity tags took 80-120 ms a frame at 6x CPU throttling).
// Between full placements a flat pan moves a placer's marks as one, by a transform on their parent.

import type { Map as MlMap } from "maplibre-gl";

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

/** The camera as a pan plane sees it: whether it moves, what a flat pan leaves alone (null: not
 * flat), and one fixed point's place on screen, CSS px. */
export interface PanCamera {
  moving(): boolean;
  still(): number[] | null;
  origin(): [number, number];
  /** a full placement happened: the fixed point may be chosen again */
  placed?(): void;
}

/** The overlay's camera of the last MapLibre frame: for placers run after each frame. */
export function overlayCamera(o: { moving: boolean; globe: boolean; camMatrix: ArrayLike<number> | null; toScreen(x: number, y: number): [number, number] }): PanCamera {
  return {
    moving: () => o.moving,
    // a flat Mercator pan changes only the translation (12-14)
    still: () => { const m = o.camMatrix; return m && !o.globe && m[3] === 0 && m[7] === 0 ? Array.from(m).filter((_, i) => i < 12 || i > 14) : null; },
    origin: () => o.toScreen(0, 0),
  };
}

/** MapLibre's camera as it is now: for placers run on its move events. */
export function mapCamera(map: MlMap, moving: () => boolean): PanCamera {
  let ref: [number, number] = [0, 0];
  return {
    moving,
    still: () => (map.getProjection()?.type === "globe" && map.getZoom() < 12 ? null : [map.getZoom(), map.getBearing(), map.getPitch()]),
    origin: () => { const p = map.project(ref); return [p.x, p.y]; },
    placed: () => { const c = map.getCenter(); ref = [c.lng, c.lat]; },
  };
}

/** `plane`, the parent of a placer's marks (at the container's top left, no clipping of its own),
 * moved with a flat pan since the last full placement. */
export class PanPlane {
  private key: number[] | null = null;
  private anchor: [number, number] = [0, 0];
  private shift = "";
  readonly throttle: PlaceThrottle;
  constructor(private cam: PanCamera, private plane: HTMLElement) {
    this.throttle = new PlaceThrottle(cam.moving);
  }

  /** True when the placer should place in full now (then call `placed`); otherwise the plane has
   * followed the pan, or kept still through a zoom, until the next. */
  due(): boolean {
    if (this.throttle.due()) return true;
    const k = this.cam.still(), p = this.key;
    if (k && p && k.length === p.length && k.every((v, i) => v === p[i])) {
      const [x, y] = this.cam.origin();
      this.set(`translate(${x - this.anchor[0]}px, ${y - this.anchor[1]}px)`);
    }
    return false;
  }

  /** After a full placement with the camera as it is. */
  placed() {
    this.throttle.done();
    this.set("");
    this.cam.placed?.();
    this.key = this.cam.still();
    this.anchor = this.cam.origin();
  }

  private set(t: string) {
    if (t !== this.shift) this.plane.style.transform = this.shift = t;
  }
}
