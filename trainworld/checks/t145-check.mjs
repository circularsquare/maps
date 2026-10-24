// Run from maps: node trainworld/checks/t145-check.mjs
import assert from 'node:assert/strict';
import {loadApp} from './lib.mjs';
const {strokesFromPreview,F_BLUEPRINT,F_BAD,F_PLACEMENT}=await loadApp('src/map/network.ts');
const points=new Float32Array([0,0,0,50,0,8,100,50,16]);
const speeds=new Float32Array([160,75]);
const original=points.slice(),ghost=strokesFromPreview(points,false,speeds,true);
assert.equal(ghost.count,2);
assert.deepEqual(ghost.speed,speeds,'speed values remain in the authoritative segment order');
assert.deepEqual([...ghost.level],[0,1],'height changes remain aligned with segment starts');
assert.ok(ghost.flags.every(f=>(f&F_BLUEPRINT)!==0&&(f&F_PLACEMENT)!==0&&(f&F_BAD)===0));
const bad=strokesFromPreview(points,true,speeds,true);
assert.ok(bad.flags.every(f=>(f&F_BAD)!==0),'invalid colour override survives solid placement mode');
const ordinary=strokesFromPreview(points,false);
assert.ok(ordinary.flags.every(f=>f===F_BLUEPRINT),'ordinary blueprint preview semantics remain unchanged');
assert.throws(()=>strokesFromPreview(points,false,new Float32Array([160]),true),/length mismatch/);
assert.deepEqual(points,original,'previews do not mutate source geometry');
// T-418: a tangent route's circle, `arc` = centre x, y (local), ground radius m, 0
const {circleOnScreen}=await loadApp('src/map/curveCircle.ts');
// a plain frame: 1 local unit = 1 screen px, lng/lat in metres scaled about lat 0
const perDeg=6371008.8*Math.PI/180;
const scr=(x,y)=>[x,y],toLL=(x,y)=>[x/perDeg,y/perDeg],toLoc=(lng,lat)=>[lng*perDeg,lat*perDeg];
const c=circleOnScreen(new Float64Array([100,50,300,0]),scr,toLL,toLoc);
assert.ok(c&&c.x===100&&c.y===50&&Math.abs(c.r-300)<0.01,'the circle is centred on the arc centre with its ground radius: '+JSON.stringify(c));
assert.equal(circleOnScreen(undefined,scr,toLL,toLoc),null,'no arc, no circle');
assert.equal(circleOnScreen(new Float64Array([1,2,0,0]),scr,toLL,toLoc),null,'a straight has no circle');
console.log('T-145 preview contract checks passed: exact speed/level alignment, solid-only placement flag, invalid override, blueprint preservation, malformed-array refusal and the T-418 curve circle.');
