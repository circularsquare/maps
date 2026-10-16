// Station geometry (T-144, redone for T-189): platforms along their track at true length, curved
// as the track is, dots zoomed out, no growth when selected; picking on the drawn shape.
// Run from maps: node trainworld/checks/t144-check.mjs
import assert from 'node:assert/strict';
import {build} from '../app/node_modules/vite/dist/node/index.js';
import {resolve} from 'node:path';
const root=resolve('trainworld/app');
// game/ui.ts touches the DOM when it loads; the check itself uses none
globalThis.location={search:''};
globalThis.document={visibilityState:'visible',documentElement:{dataset:{},style:{setProperty(){}}},addEventListener(){},fonts:{ready:Promise.resolve()}};
const bundle=await build({configFile:false,logLevel:'silent',plugins:[{
 name:'station-check',resolveId(id){if(id.endsWith('station-check'))return '\0station-check'},
 load(id){if(id==='\0station-check')return `
 export * from ${JSON.stringify(root+'/src/map/stationGeometry.ts')};
 export { networkFromState } from ${JSON.stringify(root+'/src/map/network.ts')};
 export { NetworkRenderer } from ${JSON.stringify(root+'/src/map/renderer.ts')};
 export { buildLayers, stationScale, widthFor } from ${JSON.stringify(root+'/src/map/demandLayers.ts')};
 export { topSeg } from ${JSON.stringify(root+'/src/map/pick.ts')};`}
}],build:{write:false,minify:false,lib:{entry:'station-check',formats:['es']}}});
const output=(Array.isArray(bundle)?bundle[0]:bundle).output;
// game/ui.ts (through demandLayers -> game/demand, since T-260) sets the theme on the page at import
globalThis.location??={search:''};
globalThis.document??={documentElement:{dataset:{},style:{setProperty(){}}},addEventListener(){}};
const mod=await import('data:text/javascript;base64,'+Buffer.from(output.find(o=>o.type==='chunk').code).toString('base64'));
const {stationDistance,boxHalfPxAtBoxZoom,platformShape,waysFrom,stationShape,stationCoords,insideStation,stationPath,stationRadii,stationHaloRadius,STATION_BOX_ZOOM,STATIONS_ZOOM,plainBoxHalfPx,networkFromState,NetworkRenderer,buildLayers,stationScale}=await import('data:text/javascript;base64,'+Buffer.from(output.find(o=>o.type==='chunk').code).toString('base64'));
const F=a=>new Float32Array(a),near=(a,b,msg,eps=1e-4)=>assert.ok(Math.abs(a-b)<eps,`${msg??''} ${a} != ${b}`);

// --- where a platform lies on its track
let sh=platformShape([[0,0,0,100,0,0],[0,0,0,-100,0,0]],0,40);
near(sh.mx,0);near(sh.my,0);near(sh.ax,40);near(sh.ay,0);near(sh.phi,0,'straight');
// a dead end: the platform runs from the node along the track, the train stopping at its middle
sh=platformShape([[0,0,0,100,0,0]],0,40);
near(sh.mx,40,'dead end middle');near(sh.ax,40);near(sh.phi,0);
// a curve of radius 200 bending left of the way out: half the turn is half / radius
const arcWay=(sign)=>{const p=[];for(let i=0;i<=60;i++){const t=sign*i*2/200;p.push(200*Math.sin(t)*sign*sign,200*(1-Math.cos(t)),0)}return p};
sh=platformShape([arcWay(1),arcWay(-1)],0,40);
near(sh.phi,0.2,'curved half angle',2e-3);near(sh.mx,0);near(sh.my,0);
assert.ok(sh.ax>39.9&&Math.abs(sh.ay)<0.1,'tangent at the middle');
// followed the other way round, the same bend is to the right
sh=platformShape([arcWay(-1),arcWay(1)],0,40);
near(sh.phi,-0.2,'same bend seen from the other way',2e-3);assert.ok(sh.ax<-39.9);
const [fwd,back]=waysFrom([0,0,0,100,0,0],30,1);
assert.deepEqual([fwd[0],fwd[1],fwd[fwd.length-3]],[30,0,100]);assert.deepEqual([back[0],back[back.length-3]],[30,0]);

// --- on screen (T-204: a box is fixed on the ground, so it has one shape at every zoom; T-239: as
// wide as a plain line and its rim at STATION_BOX_ZOOM less a quarter, whatever width the lines
// are drawn at in traffic mode)
const U=40075016.686/(512*2**STATION_BOX_ZOOM); // local units per CSS px at STATION_BOX_ZOOM
const zoomed=(k)=>(x,y)=>[k*x/U+10,k*y/U+20]; // the map k times as large as at STATION_BOX_ZOOM
const project=zoomed(1);
const style={boxes:true,radius:stationRadii(15),linePx:4,ringPx:1.5,boxLinePx:6.6,boxRingPx:2};
const straight={mx:0,my:0,ax:200,ay:0,phi:0};
const HL=200/U; // half the platform on screen at STATION_BOX_ZOOM, px
let s=stationShape(0,0,straight,1,project,style,false);
const BOXW=0.75*(0.5*0.6*6.6+2+1); // T-239: a quarter narrower than a plain line and its rim, 3.735 px
near(s.halfLength,HL,'true length');near(s.halfWidth,BOXW,'a quarter narrower than a plain line and its rim at the box zoom');
assert.ok(insideStation(s,10+HL-1,20),'the end of a long platform can be picked');
assert.ok(!insideStation(s,10,20+s.halfWidth+2),'beside the box is not the box');
for (const k of [2,4,16]) {
 const z=stationShape(0,0,straight,1,zoomed(k),style,false);
 near(z.halfLength,k*HL,`length scales with the map x${k}`,1e-3);near(z.halfWidth,k*s.halfWidth,`width scales with the map x${k}`,1e-3);
}
near(stationShape(0,0,straight,1,project,{...style,linePx:20,ringPx:3},false,1).halfWidth,s.halfWidth,'the lines as drawn now do not change a box');
near(stationShape(0,0,straight,20*U,project,style,false,1).halfWidth,20,'true width when wider');
near(stationShape(0,0,{...straight,ax:0,ay:1},1,project,style,false,1).halfLength,s.halfWidth,'never shorter than wide');
near(boxHalfPxAtBoxZoom(style),BOXW,'the shader\'s width at the box zoom');
near(plainBoxHalfPx(style,STATION_BOX_ZOOM),BOXW,'plain box for the labels');near(plainBoxHalfPx(style,STATION_BOX_ZOOM+2),4*BOXW);near(plainBoxHalfPx({...style,boxes:false},12),0);
// rotated
s=stationShape(0,0,{mx:0,my:0,ax:0,ay:200,phi:0},1,project,style,false);
assert.ok(insideStation(s,10,20+HL-1));assert.ok(!insideStation(s,10+HL-1,20));
// curved: the ends lie off the chord, on the arc
s=stationShape(0,0,{mx:0,my:0,ax:200,ay:0,phi:0.5},1,project,style,false);
const R=HL/0.5,endX=10+R*Math.sin(0.5),endY=20+R*(1-Math.cos(0.5));
assert.ok(insideStation(s,endX-1,endY-0.1),'the bent end is inside');
assert.ok(!insideStation(s,endX-1,20),'the straight end is not');
near(stationCoords(s,endX,endY)[0],HL,'arc length to the end (T-254: the arc sampled as a polyline, chords a few hundredths of a px short)',0.1);near(stationCoords(s,endX,endY)[1],0,'on the centre line',1e-3);
assert.ok(stationPath(s).startsWith('M'));
// T-254: the box follows the track's own line, so an S-curve is an S, not the arc through its ends
{
 const {PLATFORM_POINTS,polyFrame}=mod;
 // an S: y = 30 sin(pi x / 200) through the node at the origin, x from -250 to 250
 const S=(dir)=>{const p=[];for(let i=0;i<=100;i++){const x=dir*i*2.5;p.push(x,30*Math.sin(Math.PI*x/200),0)}return p};
 const ss=platformShape([S(1),S(-1)],0,100);
 assert.equal(ss.pts.length,2*PLATFORM_POINTS,'PLATFORM_POINTS points');const M=PLATFORM_POINTS-1;
 near(ss.pts[M],0,'middle point at the node');near(ss.pts[M+1],0);
 const at=(x)=>[x,30*Math.sin(Math.PI*x/200)];
 const box=stationShape(0,0,ss,1,project,style,false);
 // on screen at the box zoom: x, y in local / U; the track at x = 50 local is 30 local up
 const scr=(x,y)=>project(x,y);
 for (const x of [-80,-50,0,50,80]) {const [px,py]=scr(...at(x));assert.ok(insideStation(box,px,py),`on the track at ${x}`)}
 const [cx,cy]=scr(50,0);assert.ok(!insideStation(box,cx,cy),'off the track where the arc would have been');
 // the placement ghost's outline lies on the drawn shape's edge (side vertices; the four corners are a box corner)
 const verts=stationPath(box).slice(1,-1).split('L').map(v=>v.split(',').map(Number));
 assert.equal(verts.length,2*PLATFORM_POINTS,'one vertex a side a point');
 for (const [vx,vy] of verts) near(Math.abs(stationDistance(box,box.x+vx,box.y+vy)),0,'ghost on the edge',0.15);
 const [ex,ey]=scr(...at(70));near(stationCoords(box,ex,ey)[1],0,'on the centre line',0.3);
 // straight track: the same box as before (T-189's straight platform)
 const flat=platformShape([[0,0,0,100,0,0],[0,0,0,-100,0,0]],0,40);
 const a=stationShape(0,0,flat,1,project,style,false),b=stationShape(0,0,{...flat,pts:undefined},1,project,style,false);
 near(a.halfLength,b.halfLength,'straight: same length');near(a.halfWidth,b.halfWidth);
 for (const [px,py] of [[10+1,20+2],[10-5,20-3.5],[10+40/U*0.99,20]]) {const p=stationCoords(a,px,py),q=stationCoords(b,px,py);near(p[0],q[0],'straight: same frame',1e-3);near(p[1],q[1],'',1e-3)}
 // a dead end starts at the node and runs along the track
 const de=platformShape([[0,0,0,300,0,0]],0,40);near(de.pts[0],0,'dead end from the node');near(de.pts[2*M],80,'to the platform\'s length');
 // past the end of short track the line carries straight on
 const sh2=platformShape([[0,0,0,20,0,0],[0,0,0,-20,0,0]],0,40);near(sh2.pts[2*M],40,'carried on past a short edge');
 near(polyFrame(Array.from({length:PLATFORM_POINTS},(_,i)=>[i,0]).flat(),M+1,1)[0],1+M/2,'past the end, along runs on');
}
// dots zoomed out, covering the line as drawn
s=stationShape(0,0,straight,1,project,{...style,boxes:false},true);
near(s.dot,stationRadii(15)[1],'transfer dot');assert.ok(insideStation(s,10+s.dot-0.1,20));assert.ok(!insideStation(s,10+HL-1,20),'a dot is not the platform');
near(stationShape(0,0,straight,1,project,{...style,boxes:false,linePx:40},false).dot,0.5*0.6*40+2.5,'a dot covers a plain line at its width now');
assert.ok(STATION_BOX_ZOOM>=14.5&&STATION_BOX_ZOOM<=15.5,'boxes from about zoom 15');
assert.ok(STATIONS_ZOOM>=10&&STATIONS_ZOOM<STATION_BOX_ZOOM,'stations hidden further out than the dots');
near(stationRadii(9)[0],2.2*1.32,'T-313, T-322: 32% wider than T-189\'s');near(stationRadii(15)[0],5.6*1.32);near(stationHaloRadius(15),56);

// --- the network's stations
function state(reverse=false){
 const edges=[{id:8,a:0,b:1,built:true,lengthM:100,vminKmh:100},{id:3,a:0,b:2,built:true,lengthM:100,vminKmh:100}];
 const points=[[0,0,0,100,0,0],[0,0,0,-100,0,0]];
 if(reverse){edges.reverse();points.reverse()}
 return {edges,lines:[],nodes:[{id:0,x:0,y:0,lng:-73.98,lat:40.73,platform:200,stationBuilt:false,ports:2}],
 render:{edgePts:F(points.flat()),edgeOff:new Uint32Array([0,2,4]),phases:F([]),meta:F([]),samples:F([]),lineTable:F([])}};
}
const net=networkFromState(state()),again=networkFromState(state(true));
assert.deepEqual(net.stationAxis,again.stationAxis,'edge insertion order cannot flip the platform');
const half=100/Math.cos(40.73*Math.PI/180);
near(Math.abs(net.stationAxis[0]),half,'',1e-2);near(net.stationAxis[1],0);near(net.stationShape[0],0);near(net.stationShape[2],0,'straight through');
near(net.stationShape[3],6/Math.cos(40.73*Math.PI/180),'true half width',1e-3);
assert.equal(net.stations[2]&2,2,'blueprint flag retained');
const end=networkFromState({...state(),edges:[state().edges[0]],render:{...state().render,edgePts:F([0,0,0,300,0,0]),edgeOff:new Uint32Array([0,2])}});
near(end.stationShape[0],half,'dead end: middle along the track',1e-2);

// --- picking on the drawn shape
const renderer=new NetworkRenderer();renderer.net=net;
renderer.stationStyle={boxes:true,radius:stationRadii(15),linePx:4,ringPx:1.5,boxLinePx:6.6,boxRingPx:2};
const id=(x,y)=>[x,y];
assert.equal(renderer.stationAt(half-1,0,id,0),0,'pick actual platform length');
assert.equal(renderer.stationAt(0,30,id,0),-1,'do not pick the analytical circle area');
renderer.selectedStation=0;
assert.equal(renderer.stationAt(0,30,id,0),-1,'selected: no larger target, the glow is not the station');
assert.equal(renderer.selectedShapes(id).length,1);
renderer.selectedStation=-1;
renderer.stationStyle={...renderer.stationStyle,boxes:false};
assert.equal(renderer.stationAt(half-1,0,id,0),-1,'zoomed out: a dot, not the platform');
assert.equal(renderer.stationAt(1,0,id,0),0);
// T-205: zoomed far out stations are hidden and not picked, except a selected one
renderer.stationStyle={...renderer.stationStyle,hidden:true};
assert.equal(renderer.stationAt(1,0,id,0),-1,'hidden: not picked');
renderer.selectedStation=0;
assert.equal(renderer.stationAt(1,0,id,0),0,'hidden: the selected one is still drawn and picked');
renderer.selectedStation=-1;
renderer.stationStyle={...renderer.stationStyle,boxes:true,hidden:false};
net.stations=F([0,0,0,0,0,0]);net.stationAxis=F([0,100,0,100]);net.stationShape=F([0,0,0,0,0,0,0,0]);net.stationCount=2;net.stationIds=['under','over'];net.stationLine=undefined;
assert.equal(renderer.stationAt(0,60,id,0),1,'topmost platform wins overlap');
renderer.selectedStation=0;
assert.equal(renderer.stationAt(0,60,id,0),0,'the selected one is drawn last, on top');
renderer.selectedStation=-1;

// --- T-217: a link's bridge, and a complex as one dot zoomed out
{
 const {bridgeEnds,bridgeShape,BRIDGE_HALF_PX,spanningPairs,connectorHalfPx}=mod;
 // T-313: the shortest joins linking a complex's dots
 let sp=spanningPairs([[0,0],[10,0],[20,0],[10,1]]);assert.equal(sp.length,3,'n-1 joins');
 assert.ok(!sp.some(([a,b])=>(a===0&&b===2)||(a===2&&b===0)),'no long join past a nearer dot');
 assert.equal(spanningPairs([[0,0]]).length,0,'one dot, no join');
 const flat={mx:0,my:0,ax:100,ay:0,phi:0},inset=6*U;
 let e=bridgeEnds(0,0,flat,300,0,flat);
 near(e[0],100-inset,'end to end: from inside the first platform\'s end',1e-3);near(e[2],200+inset,'to inside the second\'s',1e-3);near(e[1],0);near(e[3],0);
 e=bridgeEnds(0,0,flat,30,80,flat);
 near(e[1],0,'side by side: across',1e-3);near(e[3],80,'',1e-3);near(e[0],e[2],'straight across',1e-3);
 const b=bridgeShape(0,0,10*U,0,project,style);
 near(b.halfLength,5,'bridge on screen');near(b.halfWidth,BRIDGE_HALF_PX+style.ringPx,'thin: its white and the rim');
 assert.equal(bridgeShape(1,1,1,1,project,style),null,'no bridge where its ends meet');
 const r2=new NetworkRenderer();
 r2.net={...net,stations:F([0,0,0,0,100,0]),stationAxis:F([50,0,50,0]),stationShape:F([0,0,0,0,0,0,0,0]),stationCount:2,stationIds:['a','b'],stationLinkIdx:new Int32Array([0,1]),stationLinks:F([0,0,0,100])};
 r2.stationStyle={boxes:false,radius:stationRadii(13),linePx:4,ringPx:2,boxLinePx:6.6,boxRingPx:2};
 // T-313: zoomed out every station is its own dot, a group's dots joined by a connector
 r2.syncGroups(r2.net,{members:[['a','b']]});
 const d0=r2.drawnShapeOf(0,id),d1=r2.drawnShapeOf(1,id);
 assert.ok(d0.dot>0&&d1.dot>0&&d1.dot<20,'a dot each, not one covering both');near(d0.y,0);near(d1.y,100,'each at its own platform');
 assert.equal(r2.connPairs.length,2,'one connector in the group');assert.equal(r2.connEnds.length,4,'the link inside the group adds none');
 near(r2.connEnds[1],0,'connector from one dot');near(r2.connEnds[3],100,'to the other');
 assert.equal(r2.stationAt(0,0,id,0),0,'each dot picks its own station');assert.equal(r2.stationAt(0,100,id,0),1);
 r2.selectedStation=0;assert.equal(r2.selectedShapes(id).length,1,'only the selected dot');
 r2.selectedAlso=new Set([1]);const sel=r2.selectedShapes(id);assert.equal(sel.length,3,'two dots and their connector share one outline');
 near(sel[2].halfWidth,connectorHalfPx(r2.stationStyle),'the connector as drawn: its white and the dots\' rim');
 r2.selectedAlso=new Set();
 r2.stationStyle={...r2.stationStyle,boxes:true};
 assert.equal(r2.stationAt(0,0,id,0),0,'platforms are picked one by one');
 r2.selectedAlso=new Set([1]);assert.equal(r2.selectedShapes(id).length,3,'two platforms and their bridge share one outline');
 // the player's link between two groups: one connector between the linked stations' dots
 const r3=new NetworkRenderer();
 r3.net={...r2.net};r3.stationStyle={...r2.stationStyle,boxes:false};
 r3.syncGroups(r3.net,{members:[['a'],['b']]});
 assert.equal(r3.connPairs.length,2,'the link\'s connector');near(r3.connEnds[1],0,'connector from one dot');near(r3.connEnds[3],100,'to the other');
 r3.selectedStation=0;r3.selectedAlso=new Set([1]);assert.equal(r3.selectedShapes(id).length,3,'two dots and their connector share one outline');
 r3.selectedAlso=new Set();assert.equal(r3.selectedShapes(id).length,1,'no connector when only one end is selected');
}

// --- T-227: line width linear in riders, 0.5 to 6.6 line widths against the day's busiest stretch;
// picking reaches across a wide line
{
 const {widthFor,topSeg}=mod;
 near(widthFor(0,1000),0.5,'least used');near(widthFor(1000,1000),6.6,'busiest');near(widthFor(500,1000),3.55,'linear');near(widthFor(2000,1000),6.6,'capped');near(widthFor(10,0),0.5,'no riders anywhere');
 const ov={camMatrix:[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],toScreenInto:(x,y,o,a=0)=>{o[a]=x;o[a+1]=y},toScreen:(x,y)=>[x,y]};
 const seg=F([0,0,100,0]),slop=7*7+1;
 assert.equal(topSeg(ov,seg,1,50,15,slop)[0],-1,'a plain line is not hit 15 px off');
 assert.equal(topSeg(ov,seg,1,50,15,slop,undefined,0,undefined,F([6.6]),6.6)[0],0,'a line 44 px wide is hit 15 px off its centre');
 assert.equal(topSeg(ov,seg,1,50,25,slop,undefined,0,undefined,F([6.6]),6.6)[0],-1,'but not beyond its edge');
 assert.equal(topSeg(ov,seg,1,50,5,slop,undefined,0,undefined,F([0.5]),6.6)[0],0,'a thin line keeps the 7 px reach');
}

// --- station use circles (T-482)
const world={save:{lines:[]},edges:[]};
const demand={lines:new Map(),stations:new Map()};
const use=new Map([['under',75000],['over',18750]]);
const switches={lineLoad:false,stationFigure:(id)=>use.get(id),trainLoad:false};
const layers=buildLayers(net,world,demand,0,switches);
near(layers.stationSize[1],1);near(layers.stationSize[3],.5);
near(layers.stationSize[1]**2,4*layers.stationSize[3]**2,'circle area proportional to riders a day');
assert.ok(buildLayers(net,world,demand,0,{...switches,stationFigure:null}).stationSize.every(v=>v===0));
near(stationScale(0),0);near(stationScale(300000),1);
console.log('T-144/T-189/T-204/T-205 checks passed: platform along its track (straight, curved, dead end), true length, one shape at every zoom, covers its lines at the box zoom, dots zoomed out, hidden further out but for the selected, curved picking, no growth when selected, selected on top, deterministic platform, blueprint flag, demand circles; T-217 bridges between linked platforms, one outline round selected platforms and their bridge; T-313 a dot per station zoomed out, picked one by one, joined by connectors within a group and along links; T-227 linear line widths and picking across a wide line; T-254 boxes follow the track (S-curve, straight unchanged, dead end, short edge).');
