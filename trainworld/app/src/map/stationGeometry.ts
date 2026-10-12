// Platform boxes share geometry between GPU rendering, picking and placement previews.
export type Project = (x: number, y: number) => [number, number];
export interface StationBox { x: number; y: number; ux: number; uy: number; halfLength: number; halfWidth: number }

/** Half-platform vector along the nearest nonzero segment of a track polyline. */
export function platformAxis(points: ArrayLike<number> | undefined, x: number, y: number, halfLength: number): [number, number] {
  let nearest = Infinity, axis: [number, number] = [halfLength, 0];
  if (!points) return axis;
  for (let k = 0; k + 5 < points.length; k += 3) {
    const dx = points[k+3]-points[k], dy = points[k+4]-points[k+1], length = Math.hypot(dx,dy);
    if (length < 1e-6) continue;
    const t = Math.max(0,Math.min(1,((x-points[k])*dx+(y-points[k+1])*dy)/(length*length)));
    const d = (x-points[k]-t*dx)**2+(y-points[k+1]-t*dy)**2;
    if (d < nearest) { nearest=d; axis=[dx/length*halfLength,dy/length*halfLength]; }
  }
  return axis;
}

function interpolate(zoom: number, stops: [number,number][]) {
  for (let i=1;i<stops.length;i++) if (zoom < stops[i][0]) {
    const [z0,a]=stops[i-1], [z1,b]=stops[i];
    return a+(b-a)*Math.max(0,(zoom-z0)/(z1-z0));
  }
  return stops[stops.length-1][1];
}
/** CSS pixels: platform min length is 3.6r and min width 1.3r; physical length can grow beyond it. */
export function stationRadii(zoom: number): [number,number] {
  return [interpolate(zoom,[[9,2.6],[12,4.8],[15,7]]),interpolate(zoom,[[9,3.5],[12,6.2],[15,9]])];
}
export function stationHaloRadius(zoom: number) { return interpolate(zoom,[[9,18],[12,40],[15,56]]); }

/** CPU twin of STATION_VS's box dimensions; demand circles never enlarge the hit target. */
export function stationBox(x: number,y: number,ax: number,ay: number,project: Project,radius: number,bundle: number,slot: number,selected=false): StationBox {
  const [sx,sy]=project(x,y), [ex,ey]=project(x+ax,y+ay);
  const length=Math.hypot(ex-sx,ey-sy), scale=selected?1.35:1;
  return {x:sx,y:sy,ux:length>1e-6?(ex-sx)/length:1,uy:length>1e-6?(ey-sy)/length:0,
    halfLength:Math.max(radius*1.8,length)*scale,
    halfWidth:Math.max(radius*.65,bundle>1.5?.5*bundle*slot+1:0)*scale};
}
export function insideStation(box: StationBox,x: number,y: number,padding=0) {
  const dx=x-box.x,dy=y-box.y;
  return Math.abs(dx*box.ux+dy*box.uy)<=box.halfLength+padding
    && Math.abs(-dx*box.uy+dy*box.ux)<=box.halfWidth+padding;
}
