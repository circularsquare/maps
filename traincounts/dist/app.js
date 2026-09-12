'use strict';
const $=id=>document.getElementById(id), fmt=n=>n.toLocaleString('en',{maximumFractionDigits:1});
const palette={intercity:'#67d9b0',commuter:'#b59bff',other:'#ffc474'};
let meta,network,activeDataset,selected=null,scale=1,ready=false,indices=[],catalog=[],requestNumber=0;
const width=value=>Math.max(.65,Math.sqrt(value)*.48)*scale;
function text(tag,content,className){const el=document.createElement(tag);el.textContent=content;if(className)el.className=className;return el;}
function legend(){
  $('legend').replaceChildren(...[2,10,50,200,800].map(value=>{
    const row=document.createElement('div');row.className='legend-row';
    const line=document.createElement('i');line.className='legend-line';line.style.height=width(value)+'px';
    row.append(line,text('span',fmt(value)));return row;
  }));
}
legend();
const dialog=$('about-dialog');
$('about').onclick=()=>dialog.showModal();$('close-about').onclick=()=>dialog.close();
dialog.addEventListener('click',event=>{if(event.target===dialog){const b=dialog.getBoundingClientRect();if(event.clientX<b.left||event.clientX>b.right||event.clientY<b.top||event.clientY>b.bottom)dialog.close();}});
function fail(message){$('status').hidden=false;$('status').textContent=message;}
if(typeof maplibregl==='undefined')throw(fail('The map renderer could not load. Check your connection and reload.'),new Error('MapLibre unavailable'));
const map=new maplibregl.Map({container:'map',style:'https://tiles.openfreemap.org/styles/dark',center:[25.2,64.2],zoom:4.5,maxZoom:15,attributionControl:true});
map.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');
map.addControl(new maplibregl.ScaleControl({maxWidth:100,unit:'metric'}),'bottom-right');
map.on('error',e=>{console.error(e.error);if(!ready)fail('The map could not finish loading. Check your connection and reload.');});
function fit(){if(activeDataset)map.fitBounds(activeDataset.bounds,{padding:{left:window.innerWidth>600?280:40,right:45,top:185,bottom:55},duration:650});}
$('home').onclick=fit;
function average(values){return indices.length?indices.reduce((sum,i)=>sum+values[i],0)/indices.length:0;}
function repaint(){
  if(!ready)return;
  const choice=$('period-select').value;
  indices=meta.dates.map((_,i)=>i).filter(i=>{const day=new Date(meta.dates[i]+'T12:00:00Z').getUTCDay();return choice==='all'||(choice==='weekday'?day!==0&&day!==6:choice==='weekend'?day===0||day===6:i===Number(choice));});
  network.features.forEach(f=>{f.properties.shown=average(f.properties.daily);f.properties.width=width(f.properties.shown);});
  map.getSource('trains').setData(network);
  $('coverage').textContent=`${meta.segments} corridors · ${meta.stations} stations · ${fmt(average(meta.daily_trains))} trains/day`;
  if(selected!==null)detail(selected);
}
function detail(id){
  const feature=network.features.find(f=>f.id===id);if(!feature)return;
  const p=feature.properties;selected=id;$('detail').hidden=false;
  const box=$('detail-content');box.replaceChildren(text('h2',p.from_name+' — '+p.to_name),text('div',fmt(p.shown),'number'),text('div','scheduled trains / day · both directions','unit'));
  box.append(text('p',`${fmt(average(p.forward))} toward ${p.to_name} · ${fmt(average(p.reverse))} toward ${p.from_name}`));
  const bars=document.createElement('div');bars.className='days';const max=Math.max(...p.daily,1);
  p.daily.forEach((n,i)=>{const el=document.createElement('div');el.className='day';el.title=meta.dates[i]+': '+n+' trains';const b=document.createElement('b');b.style.height=(n/max*45)+'px';b.style.opacity=indices.includes(i)?1:.3;el.append(text('em',String(n)),b,text('span',new Date(meta.dates[i]+'T12:00:00Z').toLocaleDateString('en',{weekday:'short',timeZone:'UTC'})));bars.append(el);});
  box.append(bars,text('h3','OPERATORS · SNAPSHOT AVERAGE'));
  const table=document.createElement('table');for(const[name,total]of Object.entries(p.operators)){const row=document.createElement('tr');row.append(text('td',name),text('td',fmt(total/meta.dates.length)+'/day'));table.append(row);}box.append(table);
  box.append(text('p','Geometry: '+p.geometry_quality+'. '+(activeDataset.point_note||'Timetable points may include places where trains do not stop.'),'hint'));
  map.setFilter('selection',['==',['id'],id]);
}
function clearDetail(){selected=null;$('detail').hidden=true;if(map.getLayer('selection'))map.setFilter('selection',['==',['id'],-1]);}
$('close-detail').onclick=clearDetail;$('period-select').onchange=repaint;
$('thickness').oninput=()=>{scale=Number($('thickness').value);$('thickness-value').textContent=scale+'×';legend();repaint();};
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error(url+': '+r.status);return r.json();}
function link(id,label,url){const el=$(id);el.textContent=label;el.href=new URL(url).protocol==='https:'?url:'#';}
function setMetadata(){
  $('dataset-select').value=activeDataset.id;$('country-name').textContent=activeDataset.label.toLowerCase();
  $('dataset-scope').textContent=activeDataset.subtitle;$('map').setAttribute('aria-label','Passenger train frequency: '+activeDataset.label);
  $('home').textContent='Fit map ↗';$('home').title='Fit '+activeDataset.label;
  $('period').textContent=meta.dates[0]+' – '+meta.dates.at(-1);
  $('scope-title').textContent=activeDataset.label+' · coverage';$('scope').textContent=meta.scope;
  $('timezone').textContent=meta.timezone||activeDataset.timezone;
  $('methodology').textContent=meta.methodology_note||activeDataset.methodology_note;
  $('geometry-note').textContent=meta.geometry_note;
  $('source-attribution').textContent=meta.attribution||'';
  $('source-attribution').hidden=!meta.attribution;
  $('fallback-note').textContent=meta.geometry_fallbacks?`${meta.geometry_fallbacks} of ${meta.segments} corridors use approximate geometry, shown dashed. See the geometry notes above for the matching method.`:'All displayed corridors use railway geometry from the feed. This does not independently verify the publisher’s geometry.';
  link('source-link',meta.source_name,meta.documentation_url||activeDataset.documentation_url||meta.source_url);
  link('license-link',meta.license,meta.license_url||activeDataset.license_url);$('feed-version').textContent='Feed version: '+meta.feed_version;
  const oldChoice=$('period-select').value;
  $('period-select').replaceChildren(...[['all',meta.dates.length===7?'Full week':'Whole period'],['weekday','Weekdays'],['weekend','Weekend']].map(([value,label])=>{const option=text('option',label);option.value=value;return option;}));
  meta.dates.forEach((date,i)=>{const option=text('option',new Date(date+'T12:00:00Z').toLocaleDateString('en',{weekday:'short',month:'short',day:'numeric',timeZone:'UTC'}));option.value=String(i);$('period-select').append(option);});
  $('period-select').value=['all','weekday','weekend'].includes(oldChoice)?oldChoice:'all';
  for(const option of $('period-select').options){if(option.value==='weekday'||option.value==='weekend')option.disabled=!meta.dates.some(date=>{const d=new Date(date+'T12:00:00Z').getUTCDay();return option.value==='weekend'?d===0||d===6:d!==0&&d!==6;});}
  if($('period-select').selectedOptions[0]?.disabled)$('period-select').value='all';
}
function setupLayers(stations){
  if(map.getSource('trains')){map.getSource('trains').setData(network);map.getSource('stations').setData(stations);return;}
  const base={type:'line',source:'trains',layout:{'line-cap':'round','line-join':'round'},filter:['>',['get','shown'],0]};
  map.addSource('trains',{type:'geojson',data:network});
  map.addLayer({...base,id:'selection',filter:['==',['id'],-1],paint:{'line-color':'#fff','line-width':['+',['get','width'],6],'line-opacity':.8}});
  map.addLayer({...base,id:'casing',paint:{'line-color':'#e2f1ff','line-width':['+',['get','width'],.7],'line-opacity':.55}});
  map.addLayer({...base,id:'rail',filter:['all',base.filter,['==',['get','geometry_quality'],'feed shape']],paint:{'line-color':['get','color'],'line-width':['get','width'],'line-opacity':.96}});
  map.addLayer({...base,id:'approx',filter:['all',base.filter,['!=',['get','geometry_quality'],'feed shape']],paint:{'line-color':['get','color'],'line-width':['get','width'],'line-dasharray':[2,2]}});
  map.addLayer({...base,id:'hit',paint:{'line-color':'#000','line-width':['max',14,['get','width']],'line-opacity':0}});
  map.addSource('stations',{type:'geojson',data:stations});
  map.addLayer({id:'stations',type:'circle',source:'stations',minzoom:8,paint:{'circle-radius':['interpolate',['linear'],['zoom'],8,1.5,12,3],'circle-color':'#101822','circle-stroke-color':'#e5edf6','circle-stroke-width':1}});
  map.on('click','hit',e=>{if(ready&&e.features.length)detail(Number(e.features[0].id));});
  map.on('mouseenter','hit',()=>map.getCanvas().style.cursor='pointer');map.on('mouseleave','hit',()=>map.getCanvas().style.cursor='');
}
async function loadDataset(id){
  const dataset=catalog.find(d=>d.id===id);if(!dataset)return;
  const request=++requestNumber;fail('Loading '+dataset.label+'…');
  try{
    const[nextMeta,nextNetwork,stations]=await Promise.all(['metadata.json','segments.geojson','stations.geojson'].map(name=>json(dataset.path+'/'+name)));
    if(request!==requestNumber)return;
    if(!nextMeta.dates?.length||!nextNetwork.features?.length)throw new Error('Empty snapshot');
    nextNetwork.features.forEach(f=>{const p=f.properties;const category=Object.entries(p.categories).sort((a,b)=>b[1]-a[1])[0]?.[0];p.color=palette[category]||palette.intercity;p.shown=p.average;p.width=width(p.average);});
    clearDetail();meta=nextMeta;network=nextNetwork;activeDataset=dataset;
    setupLayers(stations);setMetadata();ready=true;repaint();$('status').hidden=true;fit();
  }catch(error){
    if(request!==requestNumber)return;console.error(error);
    if(activeDataset)$('dataset-select').value=activeDataset.id;
    fail('Could not load '+dataset.label+'. '+(activeDataset?'The previous map is still available. Choose a dataset to retry.':'Check your connection and choose a dataset to retry.'));
  }
}
$('dataset-select').onchange=()=>loadDataset($('dataset-select').value);
map.on('load',async()=>{
  try{
    catalog=await json('data/datasets.json');
    $('dataset-select').replaceChildren(...catalog.map(dataset=>{const option=text('option',dataset.label);option.value=dataset.id;return option;}));
    $('dataset-select').disabled=false;await loadDataset(catalog[0].id);
  }catch(error){console.error(error);fail('Could not load the dataset list. Reload to retry.');}
});
