// Execute the shipped circuit and controls; independently check all 64 binary states.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
process.chdir(__dirname);
const page=fs.readFileSync('response_superposition.html','utf8'),nodes=new Map();
for(const m of page.matchAll(/id="([^"]+)"/g)){
 assert(!nodes.has(m[1]));nodes.set(m[1],{innerHTML:'',textContent:'',attrs:{},listeners:{},setAttribute(k,v){this.attrs[k]=v;},addEventListener(k,f){this.listeners[k]=f;}});
}
const context=vm.createContext({window:{},document:{getElementById(id){assert(nodes.has(id),id);return nodes.get(id);}}});
const scripts=[...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];assert.equal(scripts.length,2);
scripts.forEach(s=>vm.runInContext(s[1],context));
const api=context.window.RESPONSE_DEMO;
let count=0;
for(let mask=0;mask<64;mask++){
 const x=Array.from({length:6},(_,i)=>(mask>>i)&1);api.setState(x);
 const a=api.evaluate(x),expected=x.map((_,i)=>Math.max(0,x.reduce((s,v,j)=>s+v*[2,1,-1,-2,-1,1][(j-i+6)%6],0)-1));
 for(let i=0;i<6;i++)assert(Math.abs(a.predicted[i]-expected[i])<1e-12);
 if(x.reduce((s,v)=>s+v,0)<=1)assert(a.maxError<1e-12);
 assert(!/NaN|Infinity/.test(nodes.get('latent-geometry').innerHTML+nodes.get('response-bars').innerHTML));count++;
}
for(const p of ['single','adjacent','separated','opposite'])nodes.get('preset-'+p).listeners.click();
assert(api.evaluate(api.getState()).predicted.every(v=>Math.abs(v)<1e-12));
nodes.get('sector-3').listeners.click();assert.equal(api.getState()[3],0);
assert.equal(nodes.get('sector-3').attrs['aria-pressed'],'false');
api.setState([1,0,0,0,0,0]);
fs.mkdirSync('output/blog',{recursive:true});
fs.writeFileSync('output/blog/superposition.svg',api.geometry([1,0,0,0,0,0]));
fs.writeFileSync('output/blog/superposition-interference.svg',api.bars([1,0,0,1,0,0]));
assert(!/<script[^>]+src=/.test(page));assert(!/\/(?:Users|home)\//.test(page));
console.log(JSON.stringify({status:'PASS',binary_states:count,presets:4,feature_buttons:6,scope:'Shipped JS and DOM-stub interactions; no browser layout claim'}));
