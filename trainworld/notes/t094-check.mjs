import {writeFile} from 'node:fs/promises';
import {build} from '../app/node_modules/vite/dist/node/index.js';
const bundle=await build({configFile:false,logLevel:'silent',build:{write:false,minify:false,lib:{entry:'trainworld/app/src/map/demandLayers.ts',formats:['es']}}});
const output=(Array.isArray(bundle)?bundle[0]:bundle).output;
const {buildLayers}=await import('data:text/javascript;base64,'+Buffer.from(output.find(o=>o.type==='chunk').code).toString('base64'));
const K=1/Math.cos(40.73*Math.PI/180);
function fixture(xs){
 const count=xs.length,stationIds=xs.map((_,i)=>String(i)),stations=new Float32Array(count*3);
 xs.forEach((x,i)=>stations.set([x*K,0,0],i*3));
 const net={lines:{count:0,edge:[],colour:[]},samples:[],stations,stationIds,stationCount:count};
 const world={save:{lines:[]},edges:[]};
 const demand={lines:new Map(),stations:new Map(stationIds.map(id=>[id,{boardings:150000,alightings:0}]))};
 return {net,world,demand};
}
const cases=[];
for(const [name,xs,expected] of [
 ['isolated',[0,101,202],[1,1,1]],
 ['near',[0,99,300],[1,1,1]],
 ['same place',[0,0,0],[1,1,1]],
 ['cell boundary',[-1,98,400],[1,1,1]],
 ['chain',[0,90,180],[1,1,1]],
]){
 const {net,world,demand}=fixture(xs),switches={lineLoad:false,stationRiders:true,trainLoad:false};
 const size=buildLayers(net,world,demand,0,switches).stationSize,actual=expected.map((_,i)=>size[i*2+1]);
 if(actual.some((v,i)=>Math.abs(v-expected[i])>1e-6))throw Error(name+': '+actual);
 const off=buildLayers(net,world,demand,0,{...switches,stationRiders:false}).stationSize;
 if(off.some(v=>v!==0))throw Error('Disabled demand circles should have zero halo scale');
 cases.push({name,actual,haloScale:actual});
}
const {net,world,demand}=fixture(Array.from({length:10000},(_,i)=>i*200));
const start=performance.now();buildLayers(net,world,demand,0,{lineLoad:false,stationRiders:true,trainLoad:false});
const firstMs=performance.now()-start,again=performance.now();buildLayers(net,world,demand,0,{lineLoad:false,stationRiders:true,trainLoad:false});
const cachedMs=performance.now()-again;
const result={cases,firstMs,cachedMs};await writeFile('artifacts/trainworld/notes-outputs/T-094-check.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
