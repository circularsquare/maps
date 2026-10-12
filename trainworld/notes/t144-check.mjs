// Run from maps: node trainworld/notes/t144-check.mjs
import assert from 'node:assert/strict';
import {build} from '../app/node_modules/vite/dist/node/index.js';
import {resolve} from 'node:path';
const root=resolve('trainworld/app');
const bundle=await build({configFile:false,logLevel:'silent',plugins:[{
 name:'station-check',resolveId(id){if(id.endsWith('station-check'))return '\0station-check'},
 load(id){if(id==='\0station-check')return `
 export * from ${JSON.stringify(root+'/src/map/stationGeometry.ts')};
 export { networkFromState } from ${JSON.stringify(root+'/src/map/network.ts')};
 export { NetworkRenderer } from ${JSON.stringify(root+'/src/map/renderer.ts')};
 export { buildLayers, stationScale } from ${JSON.stringify(root+'/src/map/demandLayers.ts')};`}
}],build:{write:false,minify:false,lib:{entry:'station-check',formats:['es']}}});
const output=(Array.isArray(bundle)?bundle[0]:bundle).output;
const {platformAxis,stationBox,insideStation,stationRadii,stationHaloRadius,networkFromState,NetworkRenderer,buildLayers,stationScale}=await import('data:text/javascript;base64,'+Buffer.from(output.find(o=>o.type==='chunk').code).toString('base64'));
const F=a=>new Float32Array(a),near=(a,b)=>assert.ok(Math.abs(a-b)<1e-4,`${a} != ${b}`);
assert.deepEqual(platformAxis([0,0,0,100,0,0,100,100,0],100,50,80),[0,80]);
assert.deepEqual(platformAxis([0,0,0,0,0,0,20,0,0],5,0,80),[80,0],'skip degenerate polyline segment');
const project=(x,y)=>[2*x+10,2*y+20];
const box=stationBox(0,0,40,0,project,5,1,4);
near(box.halfLength,80);near(box.halfWidth,3.25);
assert.ok(insideStation(box,88,20),'the end of a long platform can be picked');
assert.ok(!insideStation(box,10,34),'halo area outside box does not become selectable');
const rotated=stationBox(0,0,0,40,project,5,1,4);
assert.ok(insideStation(rotated,10,99));assert.ok(!insideStation(rotated,89,20));
const selected=stationBox(0,0,40,0,project,5,1,4,true);
near(selected.halfLength,108);
const minimum=stationBox(0,0,0,0,project,5,1,4);
near(minimum.halfLength,9);near(minimum.halfWidth,3.25);
near(stationBox(0,0,10,0,project,5,6,4).halfWidth,13,'wide bundles remain covered');
near(stationRadii(9)[0],2.6);near(stationRadii(15)[0],7);near(stationHaloRadius(15),56);

function state(reverse=false){
 const edges=[{id:8,a:0,b:1,built:true,lengthM:100,vminKmh:100},{id:3,a:0,b:2,built:true,lengthM:100,vminKmh:100}];
 const points=[[0,0,0,100,0,0],[0,0,0,0,100,0]];
 if(reverse){edges.reverse();points.reverse()}
 return {edges,lines:[],nodes:[{id:0,x:0,y:0,lng:-73.98,lat:40.73,platform:200,stationBuilt:false,ports:2}],
 render:{edgePts:F(points.flat()),edgeOff:new Uint32Array([0,2,4]),phases:F([]),meta:F([]),samples:F([]),lineTable:F([])}};
}
const net=networkFromState(state()),again=networkFromState(state(true));
assert.deepEqual(net.stationAxis,again.stationAxis,'edge insertion order cannot flip junction platform');
near(net.stationAxis[0],0);near(net.stationAxis[1],100/Math.cos(40.73*Math.PI/180));
assert.equal(net.stations[2]&2,2,'blueprint flag retained');
const renderer=new NetworkRenderer();renderer.net=net;
assert.equal(renderer.stationAt(0,100,(x,y)=>[x,y],0),0,'pick actual platform length');
assert.equal(renderer.stationAt(30,0,(x,y)=>[x,y],0),-1,'do not pick analytical circle area');
net.stations=F([0,0,0,0,0,0]);net.stationAxis=F([0,100,0,100]);net.stationCount=2;net.stationIds=['under','over'];
assert.equal(renderer.stationAt(0,60,(x,y)=>[x,y],0),1,'topmost platform wins overlap');
const world={save:{lines:[]},edges:[]};
const demand={lines:new Map(),stations:new Map([['under',{boardings:150000,alightings:0}],['over',{boardings:37500,alightings:0}]])};
const switches={lineLoad:false,stationRiders:true,trainLoad:false};
const layers=buildLayers(net,world,demand,0,switches);
near(layers.stationSize[1],1);near(layers.stationSize[3],.5);
near(layers.stationSize[1]**2,4*layers.stationSize[3]**2,'circle area proportional to daily demand');
assert.ok(buildLayers(net,world,demand,0,{...switches,stationRiders:false}).stationSize.every(v=>v===0));
near(stationScale(0),0);near(stationScale(300000),1);
console.log('T-144 checks passed: physical length, rotation, minimum size, wide bundles, deterministic junction tangent, blueprint flag, end/overlap picking, non-pickable halo and demand area.');
