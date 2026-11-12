// T-3307: MapLibre needs WebGL2 and throws its own English message without it. Asked before the map starts, so
// the game can say what to do instead (game/loading.ts `loadNoWebgl2`).

/** Whether this browser gives a WebGL2 context (it can be missing for a browser too old, or with hardware
 * acceleration off and no software fallback). The probe context is let go at once. */
export function hasWebgl2(): boolean {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return false;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}
