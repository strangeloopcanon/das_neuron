/* Native Node DOM-stub integration checks. No browser or network required. */
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import cp from 'node:child_process';
const html=fs.readFileSync('dsge_prediction.html','utf8'),app=fs.readFileSync('dsge_prediction.js','utf8');
const payload=html.match(/const STUDY_DATA=(.*?);const PYTHON_SOURCE=/s)[1],data=JSON.parse(payload);
assert.deepEqual(Object.keys(data.banks),['practice','A','B']);
for(const bank of ['A','B'])for(const t of data.banks[bank])for(const k of ['answer_sd','edited_trace','truth','edited_output'])assert(!(k in t));
assert.equal(data.metadata.tasks_sha256,crypto.createHash('sha256').update(fs.readFileSync('dsge_prediction_tasks.json')).digest('hex'));
assert(!/<details id="automated-results"[^>]*\bopen\b/.test(html));
const ids=[...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
function mount(){
 const elements=Object.fromEntries(ids.map(id=>[id,{value:'',hidden:false,disabled:false,innerHTML:'',textContent:'',listeners:{},addEventListener(k,f){this.listeners[k]=f;},click(){this.listeners.click?.();}}]));
 Object.assign(elements.kind,{value:'demo'});elements.slot.value='1';elements.condition.value='complete_model';elements.form.value='A';
 let clock=0;const blobs=[];
 const context={STUDY_DATA:data,PYTHON_SOURCE:'simulator',performance:{now:()=>clock+=7},document:{getElementById:id=>{assert(elements[id],id);return elements[id];},createElement:()=>({click(){}})},Blob:class{constructor(parts){blobs.push(parts.join(''));}},URL:{createObjectURL:()=>'',revokeObjectURL(){}},setTimeout:()=>{}};
 vm.createContext(context);vm.runInContext(app,context);return {e:elements,context,blobs,click:id=>elements[id].click(),change:id=>elements[id].listeners.change()};
}
const exports=[];
for(let slot=0;slot<24;slot++){
 const m=mount(),{e}=m;e.kind.value='human';e.slot.value=String(slot+1);m.change('kind');assert(e['demo-options'].hidden);
 e.participant.value='ui-slot-'+slot;m.click('start');assert.equal(e.guide.hidden,data.allocations[slot].condition!=='mechanism_guide');assert(e['automated-results'].hidden);
 assert.throws(()=>m.context.DSGE_STUDY.exportPayload());
 // Every original parameter occurs verbatim in the exact-value tables.
 let count=0;for(const vals of Object.values(data.model.network))for(const row of vals)for(const v of Array.isArray(row)?row:[row]){assert(e.weights.innerHTML.includes('>'+String(v)+'</td>'));count++;}assert.equal(count,387);
 for(const layer of Object.values(data.labels))for(const l of layer){assert(e.labels.innerHTML.includes(l.label));assert(e.labels.innerHTML.includes(l.definition.replaceAll('>','&gt;')));}
 m.click('begin-practice');
 e.prediction.value='';m.click('submit');assert(!e.prediction.disabled);assert(e.feedback.textContent.includes('finite'));
 for(let index=0;index<14;index++){
  m.click('download-case');const casePacket=JSON.parse(m.blobs.at(-1));assert('baseline' in casePacket);assert('intervention' in casePacket);assert(!('answer_sd' in casePacket));assert(!('edited_trace' in casePacket));
  e.prediction.value=String(index/100);e.tools.value='local verifier';m.click('submit');assert(e.prediction.disabled);assert(e.submit.disabled);
  if(index<2){assert(e.feedback.textContent.includes('Practice answer:'));assert.equal(Boolean(e['practice-trace'].innerHTML),data.allocations[slot].condition==='mechanism_guide');}
  else{assert.equal(e.feedback.textContent,'Prediction locked. No scored feedback.');assert.equal(e['practice-trace'].innerHTML,'');}
  // A second submit cannot replace the locked value.
  e.prediction.value='999';m.click('submit');
  if(index<13)assert.throws(()=>m.context.DSGE_STUDY.exportPayload());
  m.click('next');
 }
 const p=JSON.parse(JSON.stringify(m.context.DSGE_STUDY.exportPayload()));assert.equal(p.slot,slot);assert.equal(p.condition,data.allocations[slot].condition);assert.equal(p.form,data.allocations[slot].form);assert.equal(p.responses.length,12);assert(p.responses.every(r=>r.prediction_sd!==999&&r.response_ms>=0&&r.tool_use==='local verifier'));assert(p.total_ms>=p.materials_ms+p.responses.reduce((s,r)=>s+r.response_ms,0));assert.equal(new Set(p.task_order).size,12);exports.push(p);
}
for(const condition of ['complete_model','mechanism_guide'])for(const form of ['A','B']){const m=mount();m.e.participant.value='demo';m.e.condition.value=condition;m.e.form.value=form;m.click('start');assert.equal(m.e.guide.hidden,condition!=='mechanism_guide');}
const scored=cp.spawnSync('python',['-c','import json,sys; from score_dsge_predictions import score; rows=json.load(sys.stdin); [score(row) for row in rows]; print(len(rows))'],{input:JSON.stringify(exports),encoding:'utf8'});
assert.equal(scored.status,0,scored.stderr);assert.equal(scored.stdout.trim(),'24');
console.log('UI checks passed: 24 human assignments, 4 demo cells, all 387 exact parameters and 32 labels, practice/guide separation, numeric validation, locked answers, 12-case exports, exact hashes, timings, Python scorer acceptance.');
