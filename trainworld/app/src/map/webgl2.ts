// T-3307: MapLibre needs WebGL2 and throws its own English message without it. Asked before the map starts, so
// the game can say what to do instead (game/loading.ts `loadNoWebgl2`, `loadGraphicsGone`).
// T-3582: a browser that has WebGL2 can still refuse a context: Chrome can block a page's WebGL after its context
// was lost (a GPU crash or reset) until the browser restarts, and a reload does not lift it.

/** "ok"; "missing" when the browser has no WebGL2 at all (too old); "refused" when it has WebGL2 but gave no
 * context (blocked after a lost context, or hardware acceleration off with no software fallback), with the
 * browser's own reason when it gave one. */
export type Webgl2State = { state: "ok" } | { state: "missing" } | { state: "refused"; reason: string };

/** Whether this browser gives a WebGL2 context now. The probe context is let go at once. */
export function webgl2State(): Webgl2State {
  if (typeof WebGL2RenderingContext === "undefined") return { state: "missing" };
  let reason = "";
  try {
    const canvas = document.createElement("canvas");
    canvas.addEventListener("webglcontextcreationerror", (e) => { reason = (e as WebGLContextEvent).statusMessage || reason; });
    const gl = canvas.getContext("webgl2");
    if (!gl) return { state: "refused", reason };
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { state: "ok" };
  } catch (e) {
    return { state: "refused", reason: reason || String(e) };
  }
}
