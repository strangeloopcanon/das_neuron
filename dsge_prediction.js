/* No scored edited forward pass or answer is shipped to the UI. */
(() => {
 'use strict';
 const D=STUDY_DATA, $=id=>document.getElementById(id), now=()=>performance.now();
 const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const table=(heads,rows)=>'<div class="scroll"><table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
 const vectors=obj=>Object.entries(obj).map(([name,values])=>'<h3>'+esc(name)+'</h3>'+table(['index','value'],values.map((v,i)=>[i,v]))).join('');
 const download=(name,text,type='application/json')=>{const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
 let state=null;
 function assignment(){
  const kind=$('kind').value;
  if(kind==='human') {const s=Number($('slot').value)-1;if(!Number.isInteger(s)||s<0||s>23)throw Error('Enter your assigned slot from 1 to 24.');return {...D.allocations[s],participant_kind:kind};}
  if(!['demo','agent_smoke'].includes(kind))throw Error('Invalid session type.');
  const condition=$('condition').value,form=$('form').value;
  if(!['complete_model','mechanism_guide'].includes(condition)||!['A','B'].includes(form))throw Error('Invalid condition or form.');
  return {slot:D.allocations.find(a=>a.condition===condition&&a.form===form).slot,condition,form,participant_kind:kind};
 }
 function updateAssignment(){
  $('demo-options').hidden=$('kind').value==='human';$('slot').disabled=$('kind').value!=='human';
  try{const a=assignment();$('assignment').textContent=(a.participant_kind==='human'?'Assigned':'Selected')+': '+a.condition.replaceAll('_',' ')+' · form '+a.form+'. Export allocation index: '+a.slot+' (zero-based).';}catch(e){$('assignment').textContent=e.message;}
 }
 function renderMaterials(){
  $('weights').innerHTML=Object.entries(D.model.network).map(([k,v])=>'<h3>'+k+'</h3>'+table(['row / unit',...(Array.isArray(v[0])?v[0].map((_,i)=>'column '+i):['value'])],v.map((row,i)=>[i,...(Array.isArray(row)?row:[row])]))).join('');
  $('normalization').innerHTML=table(['input index','name'],D.model.inputs.map((v,i)=>[i,v]))+table(['output index','name'],D.model.outputs.map((v,i)=>[i,v]))+vectors(D.model.normalization);
  $('label-scope').textContent=D.label_scope;
  $('labels').innerHTML=Object.entries(D.labels).map(([layer,ls])=>'<h3>'+layer+'</h3>'+table(['unit','association label','r capital','r productivity','r labor shock','active fraction','definition'],ls.map(l=>[l.unit,l.label,...l.correlations,l.activation_frequency,l.definition]))).join('');
  $('code').textContent='Downloaded model: dsge_prediction_model.json\nCLI: python dsge_prediction_materials.py "[1.3,0,0]" --tasks dsge_prediction_model.json\nAdd --edit with a JSON intervention object copied from a case.\n\n'+PYTHON_SOURCE;
 }
 function renderCase(){
  if(state.index===14){state.finished=now();$('work').hidden=true;$('complete').hidden=false;$('automated-results').hidden=false;return;}
  const practice=state.index<2,t=practice?D.banks.practice[state.index]:D.banks[state.form][state.index-2];
  state.task=t;state.locked=false;state.caseStarted=now();
  $('work').hidden=false;$('progress').textContent=practice?`PRACTICE ${state.index+1} / 2 · FEEDBACK AVAILABLE`:`SCORED ${state.index-1} / 12 · NO FEEDBACK`;
  $('case-title').textContent=t.task_id+' · '+t.family.replaceAll('_',' ');
  $('question').textContent='Predict the change in '+t.output_name+' after this intervention. Enter edited minus baseline in TRAIN output SD.';
  $('edit').textContent=JSON.stringify({raw_input:t.raw_input,intervention:t.intervention,selected_output_index:t.output_index},null,2);
  $('inputs').innerHTML=table(['input','physical value','normalized value'],D.model.inputs.map((name,i)=>[name,t.raw_input[i],t.normalized_input[i]]));
  $('baseline').innerHTML=vectors(t.baseline);
  $('prediction').value='';$('prediction').disabled=false;$('tools').value='';$('tools').disabled=false;$('submit').disabled=false;$('next').hidden=true;$('feedback').textContent='';$('practice-trace').innerHTML='';
 }
 function start(){
  if(state)return;
  try{const a=assignment(),id=$('participant').value.trim();if(!id)throw Error('Enter a pseudonymous participant ID.');state={...a,participant_id:id,started:now(),index:0,responses:[]};$('setup').hidden=true;$('session').hidden=false;$('guide').hidden=a.condition!=='mechanism_guide';$('automated-results').open=false;$('automated-results').hidden=true;$('session-label').textContent=a.participant_kind+' · '+a.condition.replaceAll('_',' ')+' · form '+a.form;$('setup-error').textContent='';}
  catch(e){$('setup-error').textContent=e.message;}
 }
 function begin(){if(!state||state.materials_ms!==undefined)return;state.materials_ms=now()-state.started;$('begin-practice').hidden=true;renderCase();}
 function submit(){
  if(!state||!state.task||state.locked)return;
  const raw=$('prediction').value.trim(),n=Number(raw);if(!raw||!Number.isFinite(n)){$('feedback').textContent='Enter a finite signed number.';return;}
  const elapsed=now()-state.caseStarted;state.locked=true;$('prediction').disabled=true;$('tools').disabled=true;$('submit').disabled=true;
  if(state.index<2){$('feedback').textContent='Practice answer: '+state.task.answer_sd+' TRAIN output SD. Your absolute error: '+Math.abs(n-state.task.answer_sd)+'.';if(state.condition==='mechanism_guide')$('practice-trace').innerHTML='<details><summary>Practice worked trace · edited forward pass</summary><p>Apply the stated edit, then recalculate downstream gates and the selected readout. Subtract the baseline normalized output.</p>'+vectors(state.task.edited_trace)+'</details>';}
  else {state.responses.push({task_id:state.task.task_id,prediction_sd:n,response_ms:elapsed,tool_use:$('tools').value.trim()});$('feedback').textContent='Prediction locked. No scored feedback.';}
  $('next').hidden=false;$('next').textContent=state.index===13?'Finish study':state.index===1?'Begin twelve scored cases':'Continue';
 }
 function advance(){if(!state?.locked)return;state.index++;renderCase();}
 function exportPayload(){
  if(!state||state.responses.length!==12||state.index!==14)throw Error('Complete all twelve scored predictions before export.');
  const summed=state.materials_ms+state.responses.reduce((s,r)=>s+r.response_ms,0);
  return {schema_version:1,participant_kind:state.participant_kind,participant_id:state.participant_id,slot:state.slot,condition:state.condition,form:state.form,task_order:state.responses.map(r=>r.task_id),responses:state.responses.map(r=>({...r})),materials_ms:state.materials_ms,total_ms:Math.max(summed,now()-state.started),metadata:Object.fromEntries(['protocol_sha256','tasks_sha256','model_sha256','labels_sha256'].map(k=>[k,D.metadata[k]]))};
 }
 function diagram(){let s='';const layers=[3,16,16,3],xs=[80,350,650,920],ys=n=>Array.from({length:n},(_,i)=>35+i*145/(n-1));for(let l=0;l<3;l++)for(const y of ys(layers[l]))for(const z of ys(layers[l+1]))s+=`<line x1="${xs[l]}" y1="${y}" x2="${xs[l+1]}" y2="${z}" stroke="#25342f" opacity=".08"/>`;layers.forEach((n,l)=>{ys(n).forEach((y,i)=>s+=`<circle cx="${xs[l]}" cy="${y}" r="${n===3?7:4}" fill="${l===1||l===2?'#bf492e':'#25342f'}"/><text x="${xs[l]+11}" y="${y+3}" font-size="8" fill="#5e665f">${i}</text>`);s+=`<text x="${xs[l]}" y="208" text-anchor="middle" font-size="12" fill="#25342f">${['3 inputs','H1 · 16 ReLU','H2 · 16 ReLU','3 linear outputs'][l]}</text>`;});$('diagram').innerHTML=s;}
 ['kind','slot','condition','form'].forEach(id=>$(id).addEventListener('change',updateAssignment));
 $('start').addEventListener('click',start);$('begin-practice').addEventListener('click',begin);$('submit').addEventListener('click',submit);$('next').addEventListener('click',advance);
 $('download-model').addEventListener('click',()=>download('dsge_prediction_model.json',JSON.stringify({model:D.model,labels:D.labels,label_scope:D.label_scope,metadata:D.metadata},null,2)));
 $('download-code').addEventListener('click',()=>download('dsge_prediction_materials.py',PYTHON_SOURCE,'text/x-python'));
 $('download-case').addEventListener('click',()=>{if(!state?.task)return;const {task_id,raw_input,normalized_input,baseline,intervention,output_index,output_name}=state.task;download(task_id+'.json',JSON.stringify({task_id,raw_input,normalized_input,baseline,intervention,output_index,output_name},null,2));});
 $('export').addEventListener('click',()=>{try{const p=exportPayload();download('dsge-responses-'+p.participant_id.replace(/[^a-zA-Z0-9_-]/g,'_')+'.json',JSON.stringify(p,null,2));$('export-status').textContent='Response JSON exported locally. No data was uploaded.';}catch(e){$('export-status').textContent=e.message;}});
 renderMaterials();diagram();updateAssignment();
 // Explicit interface for the local DOM-stub verifier; no answers or simulator.
 if(typeof globalThis!=='undefined')globalThis.DSGE_STUDY={exportPayload};
})();
