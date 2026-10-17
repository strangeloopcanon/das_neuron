// Runs the shipped HTML scripts and checks numerical accounting and controls.
// This is not browser layout verification.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
process.chdir(__dirname);
const page=fs.readFileSync('system.html','utf8'),nodes=new Map(),downloads=[],blobs=new Map();
for(const tag of page.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)){
 const attrs={};for(const m of tag[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g))attrs[m[1]]=m[2]===undefined?true:m[2];
 if(attrs.id){assert(!nodes.has(attrs.id),'Duplicate ID');nodes.set(attrs.id,{value:attrs.value||'',hidden:attrs.hidden===true,textContent:'',attrs,_html:'',children:[],listeners:{},
  get innerHTML(){return this._html;},set innerHTML(v){this._html=v;this.children=[];},setAttribute(k,v){this.attrs[k]=v;},
  addEventListener(e,f){this.listeners[e]=f;},appendChild(o){this.children.push(o);if(this.value==='')this.value=String(o.value);}});}
}
const context=vm.createContext({window:{},Blob,console,document:{getElementById(id){assert(nodes.has(id),'Missing '+id);return nodes.get(id);},
 createElement(tag){if(tag==='option')return {value:'',textContent:''};assert.equal(tag,'a');return {click(){downloads.push({name:this.download,url:this.href});}};}},
 URL:{createObjectURL(blob){const url='blob:'+blobs.size;blobs.set(url,blob);return url;},revokeObjectURL(){}}});
const scripts=[...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);assert.equal(scripts.length,2);
for(const script of scripts)vm.runInContext(script,context);
const API=context.window.SYSTEM_DEMO,D=JSON.parse(JSON.stringify(context.window.SYSTEM_DATA)),source=JSON.parse(fs.readFileSync('system_analysis.json'));
const near=(a,b)=>assert(Math.abs(a-b)<2e-11,`${a} != ${b}`);
assert.equal(D.default_model_id,'H16-seed113');assert.equal(D.models.length,9);
assert.equal(D.features.length,21);assert.equal(D.capital.length,301);
assert.deepEqual(D.normalization,source.provenance.normalization);
assert(nodes.get('system-reveal').hidden);nodes.get('trace-system').listeners.click();assert(!nodes.get('system-reveal').hidden);assert.equal(nodes.get('trace-system').attrs['aria-expanded'],'true');
const svgExport=id=>`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${nodes.get(id).attrs.viewBox}"><style>text{font:12px Arial,sans-serif;fill:#25342f}</style><rect width="100%" height="100%" fill="#f2efe7"/>${nodes.get(id).innerHTML}</svg>`;
fs.mkdirSync('tmp/system',{recursive:true});fs.writeFileSync('tmp/system/default-network.svg',svgExport('system-network'));fs.writeFileSync('tmp/system/default-contributions.svg',svgExport('route-bars'));
assert(nodes.get('interaction-detail').textContent.includes('sign wrong'),'Sign discrepancy disclosed at default');
let labels=0,weights=0,cases=0,updates=0;
for(let mi=0;mi<D.models.length;mi++){
 nodes.get('system-model').value=String(mi);nodes.get('system-model').listeners.change();
 const m=D.models[mi],original=source.models[mi];assert.deepEqual(m.network,original.network);assert.deepEqual(m.labels,original.labels);
 const records=JSON.parse(JSON.stringify(API.weightRecords(m)));assert.equal(records.length,m.width*m.width+8*m.width+3);weights+=records.length;
 assert.equal(new Set(records.map(row=>row.slice(0,3).join('/'))).size,records.length,'Unique weight addresses');
 assert.equal((nodes.get('weight-matrices').innerHTML.match(/<td>/g)||[]).length,records.length,'Every weight printed');
 assert.equal((nodes.get('all-labels').innerHTML.match(/<tr>/g)||[]).length,2*m.width);labels+=2*m.width;
 const fixedLabels=nodes.get('all-labels').innerHTML,fixedWeights=nodes.get('weight-matrices').innerHTML;
 for(const example of original.examples){
  nodes.get('system-capital').value=String(example.grid_index);nodes.get('system-capital').listeners.input();updates++;
  const a=API.current.account;
  ['baseline','productivity','cost','both'].forEach((key,index)=>{
   const snapshot=example.snapshots[key];
   ['p1','h1','p2','h2'].forEach(layer=>snapshot[layer].forEach((v,j)=>near(v,a.snapshots[index][layer][j])));
   snapshot.physical_outputs.forEach((v,o)=>near(v,a.snapshots[index].y[o]));
  });
  const expected=example.contrasts.joint;
  for(let o=0;o<3;o++){
   near(a.delta[o],expected.delta[o]);
   near(a.contributions.reduce((s,row)=>s+row[o],0),a.delta[o]);
   near(a.interactions.reduce((s,row)=>s+row[o],0),a.interaction[o]);
   near(a.economicInteraction[o],example.interaction.teacher[o]);
   nodes.get('system-output').value=String(o);nodes.get('system-output').listeners.change();updates++;
   assert.equal((nodes.get('route-table').innerHTML.match(/<tr>/g)||[]).length,m.width);
   assert.equal((nodes.get('system-network').innerHTML.match(/class="neuron"/g)||[]).length,2*m.width);
   assert.equal((nodes.get('system-network').innerHTML.match(/<line /g)||[]).length,m.width*m.width+6*m.width);
   assert(!/NaN|Infinity/.test(nodes.get('route-bars').innerHTML+nodes.get('route-table').innerHTML));
   cases++;
  }
  assert.equal(nodes.get('all-labels').innerHTML,fixedLabels,'Labels unchanged across states');
  assert.equal(nodes.get('weight-matrices').innerHTML,fixedWeights,'Weights unchanged across states');
 }
 for(let layer=0;layer<2;layer++){
  nodes.get('inspect-layer').value=String(layer);nodes.get('inspect-layer').listeners.change();
  nodes.get('inspect-unit').value=String(m.width-1);nodes.get('inspect-unit').listeners.change();
  assert(nodes.get('neuron-inspector').innerHTML.includes(`h${layer+1}[${m.width-1}] = max(0`));
 }
 nodes.get('system-download').listeners.click();nodes.get('weights-csv').listeners.click();
}
// All joint shock states and null-shock cases stay valid in the live controls.
nodes.get('system-model').value='0';nodes.get('system-model').listeners.change();
for(let z=0;z<7;z++)for(let e=0;e<3;e++){
 nodes.get('target-productivity').value=String(z);nodes.get('target-cost').value=String(e);nodes.get('target-cost').listeners.change();updates++;
 const a=API.current.account;
 for(let o=0;o<3;o++){near(a.contributions.reduce((s,row)=>s+row[o],0),a.delta[o]);near(a.interactions.reduce((s,row)=>s+row[o],0),a.interaction[o]);}
 if(z===3||e===1)a.interaction.forEach(v=>near(v,0));
}
nodes.get('system-network').listeners.keydown({key:'Enter',preventDefault(){},target:{getAttribute(k){return k==='data-layer'?'1':'7';}}});assert.equal(nodes.get('inspect-unit').value,'7');
assert(!/<script[^>]+src=|<link[^>]+href=/.test(page));assert(!page.includes('/*SYSTEM_'));assert(!/\/(?:Users|home)\//.test(page));
Promise.all(downloads.map(async(d,i)=>{
 const text=await blobs.get(d.url).text(),m=D.models[Math.floor(i/2)];
 if(i%2===0){const parsed=JSON.parse(text);assert.deepEqual(parsed.model.network,m.network);assert.deepEqual(parsed.normalization,D.normalization);}
 else {const rows=text.trim().split('\n').slice(1).map(row=>row.split(','));assert.equal(rows.length,m.width*m.width+8*m.width+3);const expected=JSON.parse(JSON.stringify(API.weightRecords(m)));rows.forEach((row,j)=>{assert.deepEqual(row.slice(0,3),expected[j].slice(0,3));assert.equal(Number(row[3]),expected[j][3]);});}
})).then(()=>console.log(JSON.stringify({status:'PASS',labels,weights,example_output_cases:cases,live_updates:updates,exports:downloads.length,all_information:'weights and labels match frozen evidence',visual_rendering:'SVG exports available; browser layout not tested'},null,2))).catch(e=>{console.error(e);process.exitCode=1;});
