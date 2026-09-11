// State integration test with a minimal DOM/MapLibre double; not browser visual QA.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../dist');
class Element{
  constructor(){this.children=[];this.style={};this.value='';this.hidden=false;this.disabled=false;}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;}
  addEventListener(){}setAttribute(k,v){this[k]=v;}
  get options(){return this.children;}get selectedOptions(){return this.children.filter(c=>c.value===this.value);}
  showModal(){}close(){}
}
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
elements['period-select'].value='all';
const errors=[],events={},sources={},layers={};
class MapDouble{
  addControl(){}on(event,...args){events[[event,...args.slice(0,-1)].join(':')]=args.at(-1);}
  addSource(id,s){assert(!sources[id],'duplicate source');sources[id]={data:s.data,setData(data){this.data=data;}};}
  getSource(id){return sources[id];}addLayer(layer){assert(!layers[layer.id],'duplicate layer');layers[layer.id]=layer;}
  getLayer(id){return layers[id];}setFilter(id,filter){layers[id].filter=filter;}
  fitBounds(bounds){this.bounds=bounds;}getCanvas(){return {style:{}};}
}
let fetchHook=null;
async function read(url){return {ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(root,url),'utf8'))};}
const context=vm.createContext({document:{createElement:()=>new Element(),getElementById:id=>{assert(elements[id],'missing element '+id);return elements[id];}},window:{innerWidth:1200},maplibregl:{Map:MapDouble,NavigationControl:class{},ScaleControl:class{}},URL,console:{error:e=>errors.push(e)},fetch:url=>fetchHook?fetchHook(url):read(url)});
vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),context);
const run=code=>vm.runInContext(code,context);
(async()=>{
  await events.load();
  assert.equal(run('activeDataset.id'),'fi');assert.equal(sources.trains.data.features.length,454);
  elements['period-select'].value='weekend';elements['period-select'].onchange();
  assert.equal(run('average(meta.daily_trains)'),950.5);
  run('detail(network.features[0].id)');assert.equal(elements.detail.hidden,false);
  await run("loadDataset('ie')");
  assert.equal(run('activeDataset.id'),'ie');assert.equal(elements.detail.hidden,true);
  assert.equal(elements['period-select'].value,'weekend');assert.equal(run('average(meta.daily_trains)'),530);
  assert.equal(elements.timezone.textContent,'Europe/London');
  assert.equal(elements['period-select'].options.length,10);assert.equal(elements['source-link'].href,'https://data.gov.ie/dataset/nta-gtfs');
  fetchHook=async url=>url.startsWith('data/fi/')?{ok:false,status:503}:read(url);
  await run("loadDataset('fi')");
  assert.equal(run('activeDataset.id'),'ie');assert.equal(elements['dataset-select'].value,'ie');assert.equal(errors.length,1);
  const pending=[];
  fetchHook=url=>url.startsWith('data/ie/')?new Promise(resolve=>pending.push(()=>resolve(read(url)))):read(url);
  const stale=run("loadDataset('ie')");
  await run("loadDataset('fi')");
  pending.forEach(release=>release());await stale;
  assert.equal(run('activeDataset.id'),'fi');assert.equal(elements['dataset-select'].value,'fi');assert.equal(elements.status.hidden,true);
  assert.equal(sources.trains.data.features.length,454);assert.equal(elements.timezone.textContent,'Europe/Helsinki');
  console.log('PASS: dataset switching, weekend averages, source/timezone updates, detail reset, failed-load preservation and stale-request protection.');
})().catch(error=>{console.error(error);process.exitCode=1;});
