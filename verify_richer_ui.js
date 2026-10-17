// Execute the shipped standalone scripts. No browser rendering is claimed.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
process.chdir(__dirname);
const page=fs.readFileSync('richer.html','utf8'),nodes=new Map(),downloads=[],blobs=new Map();
for(const tag of page.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)){
 const attrs={};for(const m of tag[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g))attrs[m[1]]=m[2]===undefined?true:m[2];
 if(attrs.id){assert(!nodes.has(attrs.id),'Duplicate ID');nodes.set(attrs.id,{value:attrs.value||'',textContent:'',_html:'',listeners:{},children:[],
   get innerHTML(){return this._html;},set innerHTML(v){this._html=v;this.children=[];},
   addEventListener(event,fn){this.listeners[event]=fn;},appendChild(o){this.children.push(o);if(this.value==='')this.value=String(o.value);}});}
}
nodes.get('attenuation').value='1';nodes.get('trace').value='0';
const context=vm.createContext({window:{},Blob,console,document:{getElementById(id){assert(nodes.has(id),'Missing ID '+id);return nodes.get(id);},
 createElement(tag){if(tag==='option')return {value:'',textContent:''};assert.equal(tag,'a');return {click(){downloads.push({name:this.download,url:this.href});}};}},
 URL:{createObjectURL(blob){const url='blob:'+blobs.size;blobs.set(url,blob);return url;},revokeObjectURL(){}}});
const scripts=[...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);assert.equal(scripts.length,2);
for(const script of scripts)vm.runInContext(script,context);
const D=JSON.parse(JSON.stringify(context.window.RICHER_DATA)),saved=JSON.parse(fs.readFileSync('richer_results.json'));
assert.deepEqual(D.models,saved.models);assert.deepEqual(D.norm,saved.provenance.normalization);
assert.equal(D.models.length,9);assert.equal(D.features.length,21);assert.equal(D.capital.length,301);
const near=(a,b)=>assert(Math.abs(a-b)<2e-11,`${a} != ${b}`);
function direct(raw,net,unit=null,fraction=0){
 const x=raw.map((v,j)=>(v-D.norm.input_mean[j])/D.norm.input_std[j]);
 const h1=net.W1.map((row,j)=>Math.max(0,row.reduce((s,w,k)=>s+w*x[k],net.b1[j])));
 if(unit!==null)h1[unit]*=1-fraction;
 const p2=net.W2.map((row,j)=>row.reduce((s,w,k)=>s+w*h1[k],net.b2[j])),h2=p2.map(v=>Math.max(v,0));
 const yn=net.b3.map((b,o)=>h2.reduce((s,v,j)=>s+v*net.W3[j][o],b));
 return {h1,p2,yn,y:yn.map((v,o)=>v*D.norm.output_std[o]+D.norm.output_mean[o])};
}
let predictions=0,interventions=0,updates=0;
for(let mi=0;mi<9;mi++){
 nodes.get('model').value=String(mi);nodes.get('model').listeners.change();const m=D.models[mi];
 assert.equal(nodes.get('unit').children.length,m.width);
 for(let s=0;s<21;s++)for(const index of [0,115,300]){
  const raw=[D.capital[index],...D.features[s]],actual=direct(raw,m.network);
  const shipped=context.window.RICHER_DEMO.forward(raw,m.network);
  actual.y.forEach((v,o)=>near(v,shipped.y[o]));predictions++;
  for(const fraction of [.01,.1,1]){
   const j=(s+index)%m.width,a=direct(raw,m.network,j,fraction),b=context.window.RICHER_DEMO.forward(raw,m.network,j,fraction);
   a.y.forEach((v,o)=>near(v,b.y[o]));interventions++;
  }
 }
 for(const index of [0,115,300]){
  nodes.get('capital').value=String(index);nodes.get('unit').value=String(index%m.width);
  nodes.get('capital').listeners.input();updates++;
  const selected=context.window.RICHER_DEMO.selection(),expected=direct(selected.raw,m.network);
  assert(nodes.get('outputs').innerHTML.includes(expected.y[0].toFixed(5)));
  for(const id of ['fit','gates','label'])assert(!/NaN|undefined|Infinity/.test(nodes.get(id).textContent));
  assert(!/NaN|undefined|Infinity/.test(nodes.get('effects').innerHTML+nodes.get('policy-chart').innerHTML));
  assert.equal((nodes.get('effects').innerHTML.match(/<tr>/g)||[]).length,3);
 }
}
for(const id of ['productivity','effort','unit','attenuation','trace'])assert.equal(typeof nodes.get(id).listeners.change,'function');
assert(!/<script[^>]+src=|<link[^>]+href=/.test(page),'External assets');assert(!page.includes('/*RICHER_'));assert(!/\/(?:Users|home)\//.test(page));
nodes.get('download').listeners.click();assert.equal(downloads.length,1);
blobs.get(downloads[0].url).text().then(text=>{
 const downloaded=JSON.parse(text);assert.deepEqual(downloaded.normalization,D.norm);assert.equal(downloaded.model.width,64);
 console.log(JSON.stringify({status:'PASS',saved_models:9,joint_states:21,live_predictions:predictions,finite_interventions:interventions,render_updates:updates,download:'verified',visual_rendering:'not tested'},null,2));
}).catch(error=>{console.error(error);process.exitCode=1;});
