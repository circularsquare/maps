import {build} from '../app/node_modules/vite/dist/node/index.js';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const entry='trainworld/notes/t116-coordinator-entry.ts';
const bundle=await build({configFile:false,logLevel:'silent',plugins:[{
 name:'coordinator-test',
 transform(code,id){if(id.endsWith('/demandClient.ts'))return code.replace('new URL("./demand.worker.ts", import.meta.url)','"test-worker"');}
}],build:{write:false,minify:false,lib:{entry,formats:['es']}}});
const output=(Array.isArray(bundle)?bundle[0]:bundle).output;
globalThis.location={search:'?demandWorkers=2'};
Object.defineProperty(globalThis,'navigator',{value:{hardwareConcurrency:8},configurable:true});
globalThis.document={baseURI:'http://test/'};
globalThis.fetch=async()=>({ok:true,text:async()=>'{}',arrayBuffer:async()=>new ArrayBuffer(0)});
const table=[6,10,0,0,0,1.4,10,16,1,0,0,1.15,16,20,0,0,0,1.4,20,24,1,0,0,1.3,0,6,2,0,0,1.5];
const workers=[];
class Worker {
 constructor(){this.posts=[];workers.push(this);}
 postMessage(msg){this.posts.push(msg);if(msg.kind==='init')queueMicrotask(()=>this.emit({kind:'ready',cityInfo:{nyc:[2,1,100,.2],phl:[3,1,80,.3]},info:[2,1,100,.2],periods:table,wasmMs:1,openMs:1,warmMs:1,memBytes:1}));}
 emit(data){this.onmessage({data});}
}
globalThis.Worker=Worker;
const m=await import('data:text/javascript;base64,'+Buffer.from(output.find(o=>o.type==='chunk').code).toString('base64'));
const client=m.startDemand();await new Promise(r=>setTimeout(r,10));
assert.equal(workers.length,3,'two that solve periods and one with none (T-119)');assert.equal(workers[2].posts[0].aux,true);assert.deepEqual(workers[2].posts[0].periods,[]);assert.deepEqual(workers[0].posts[0].packs.map(p=>p.id),['nyc','phl']);
client.submitNetwork({stations:[{id:1,x:0,y:0},{id:2,x:1000,y:0}],lines:[{id:1,stops:[1,2],tph:[10,10,10],hopS:[[[60],[60],[60]],[[60],[60],[60]]],dwellS:[],cars:10}]});
const result=(q,round,version=1)=>({kind:'result',version,period:q,summary:[round,100,30+round,10],citySummary:{nyc:[round,60,20+round,5],phl:[round,30,5,5]},intercitySummary:[round,10,5,0],corridors:[0,1,10,5],seg:new Float32Array([q+1+round,0,q+2+round,0]),board:new Float32Array([2,0,3,0]),intercityBoard:new Float32Array([1,0,1,0]),alight:new Float32Array([0,2,0,3]),loadOfCrush:new Float32Array([.1,0,.2,0]),ms:1,setupMs:0,memBytes:1});
// A fast worker can finish every refinement before the slow worker's initial result.
for(let round=0;round<=3;round++)for(const q of [0,2,4])workers[0].emit(result(q,round));
assert.equal(m.demandView.peek(),null);
for(const q of [1,3])workers[1].emit(result(q,0));
assert.equal(m.demandView.peek().round,0);
assert.equal(m.demandView.peek().railTripsPerDay,150,'round zero never mixes advanced periods');
for(let round=1;round<=3;round++)for(const q of [1,3])workers[1].emit(result(q,round));
assert.equal(m.demandView.peek().round,3);
assert.equal(m.demandView.peek().railTripsPerDay,165);
assert.equal(m.demandView.peek().lines.get('1').ridersPerDay,25);
assert.equal(m.demandView.peek().lines.get('1').intercityRidersPerDay,10);
assert.equal(m.intercityDemand.peek().railTripsPerDay,25);
assert.equal(m.cityDemandViews.peek().get('nyc').railTripsPerDay,115);
// T-119: every worker's sums arrive unasked with the round, and the cells' modes are asked of the worker with no periods
for(const [w,mask] of [[workers[0],0b10101],[workers[1],0b01010]])w.emit({kind:'sums',version:1,round:3,city:'nyc',mask,sums:new Float32Array([1,2,3,4])});
const modes=client.cellModes();await new Promise(r=>setTimeout(r,0));
const ask=workers[2].posts.at(-1);assert.equal(ask.kind,'cellModes');assert.deepEqual([...ask.sums],[2,4,6,8]);
assert.equal(workers.slice(0,2).flatMap(w=>w.posts).filter(p=>p.kind==='subSums').length,0,'no question to a worker busy with the next round');
workers[2].emit({kind:'cellModes',id:ask.id,city:'nyc',version:1,modes:new Float32Array(6)});assert.ok(await modes);
m.activeCity.value='phl';assert.ok(Math.abs(m.modeSplit.peek().train-100/6)<1e-8);
// Focus switches cancel city-indexed answers, including switching away and back.
const pending=client.cityCells();await Promise.resolve();
const request=workers[2].posts.at(-1);assert.equal(request.city,'phl');
m.activeCity.value='nyc';m.activeCity.value='phl';
workers[2].emit({kind:'cells',id:request.id,city:'phl',xy:new Float32Array([0,0]),h3:new BigUint64Array([1n]),zone:new Uint32Array([0]),zoneXY:new Float32Array([2000,0,0])});
assert.equal(await pending,null);
// Old network results cannot replace the current publication.
client.submitNetwork({stations:[],lines:[]});workers[0].emit(result(0,3,1));assert.equal(m.demandView.peek().version,1);
const evidence={workers:workers.length,sumsPushedNotAsked:true,localCities:workers[0].posts[0].packs.map(p=>p.id),coherentRounds:true,totalRail:165,lineBoardings:25,regionalLineBoardings:10,focusCancellation:true,staleNetworkDiscard:true};
await writeFile('trainworld/notes/T-116-coordinator-check.json',JSON.stringify(evidence,null,2));console.log(evidence);
