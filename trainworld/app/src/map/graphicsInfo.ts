// T-3589: what a feedback report says about the graphics ("graphics:" line), so a WebGL crash like T-3582's can
// be placed. Lost and restored contexts are counted for every canvas through one capture listener on the document
// (these events do not bubble, but capture sees every attached canvas: MapLibre's, the overlays', the schematic's).

let lost = 0;
let restored = 0;
let creationError = "";

if (typeof document !== "undefined") {
  document.addEventListener("webglcontextlost", () => { lost++; }, true);
  document.addEventListener("webglcontextrestored", () => { restored++; }, true);
  document.addEventListener("webglcontextcreationerror", (e) => { noteCreationError((e as WebGLContextEvent).statusMessage); }, true);
}

/** T-4073: a context of the page's is lost and not back (as far as the events have come) */
export const contextsLost = () => lost > restored;

/** T-4073: whether a fresh WebGL2 context works now: made, not lost, and answering (a browser whose GPU process went
 * down hands out contexts that are lost from the start, whose every answer is null). Let go at once. */
export function freshContextWorks(): boolean {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return false;
    const ok = !gl.isContextLost() && gl.getParameter(gl.RENDERER) != null;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return ok;
  } catch {
    return false;
  }
}

/** The last context creation error (also fed by the start probe, whose canvas is never in the page). */
export function noteCreationError(message: string | undefined) {
  if (message) creationError = message;
}

/** The unmasked renderer and vendor where the browser gives them, else the plain RENDERER. Asked of a short-lived
 * context of its own, so it works when the page's canvases are lost. */
function rendererText(): string {
  try {
    const gl = document.createElement("canvas").getContext("webgl2") ?? document.createElement("canvas").getContext("webgl");
    if (!gl) return "no context";
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const text = info
      ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) + " / " + gl.getParameter(info.UNMASKED_VENDOR_WEBGL)
      : String(gl.getParameter(gl.RENDERER));
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return text;
  } catch (e) {
    return "error " + String(e).slice(0, 80);
  }
}

/** One line: renderer / vendor, lost and restored counts, last creation error, the map canvas's buffer size. */
export function graphicsReport(): string {
  const canvas = document.querySelector<HTMLCanvasElement>("canvas.maplibregl-canvas");
  const size = canvas ? canvas.width + "x" + canvas.height : "no map canvas";
  return `${rendererText()}; lost ${lost}, restored ${restored}; creation error: ${creationError || "none"}; buffer ${size}`.slice(0, 500);
}
