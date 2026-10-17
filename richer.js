(() => {
  'use strict';
  const D=window.RICHER_DATA, $=id=>document.getElementById(id);
  const names=['Consumption','Labour','Investment'];
  const fmt=x=>Number(x).toFixed(5), pct=x=>(100*x).toFixed(2)+'%';
  const vector=(input,W,b)=>W.map((row,i)=>row.reduce((s,w,j)=>s+w*input[j],b[i]));
  function forward(raw,net,unit=null,fraction=0){
    const x=raw.map((v,i)=>(v-D.norm.input_mean[i])/D.norm.input_std[i]);
    const p1=vector(x,net.W1,net.b1),h1=p1.map(v=>Math.max(0,v));
    if(unit!==null)h1[unit]*=1-fraction;
    const p2=vector(h1,net.W2,net.b2),h2=p2.map(v=>Math.max(0,v));
    const yn=net.b3.map((b,o)=>h2.reduce((s,h,j)=>s+h*net.W3[j][o],b));
    return {p1,h1,p2,h2,yn,y:yn.map((v,o)=>v*D.norm.output_std[o]+D.norm.output_mean[o])};
  }
  function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o);}
  D.models.forEach((m,i)=>option($('model'),i,`${m.training.parameter_count.toLocaleString()} parameters · H=${m.width} · seed ${m.seed}`));
  const zs=[...new Set(D.features.map(v=>v[0]))], es=[...new Set(D.features.map(v=>v[1]))];
  zs.forEach((z,i)=>option($('productivity'),i,`z = ${Math.exp(z).toFixed(3)}`));
  es.forEach((e,i)=>option($('effort'),i,`cost multiplier = ${Math.exp(e).toFixed(3)}`));
  $('model').value=String(D.models.findIndex(m=>m.width===32&&m.seed===113));
  $('productivity').value=String(Math.floor(zs.length/2));$('effort').value=String(Math.floor(es.length/2));
  function selection(){
    const m=D.models[+$('model').value],z=zs[+$('productivity').value],e=es[+$('effort').value];
    const state=D.features.findIndex(f=>f[0]===z&&f[1]===e),index=+$('capital').value;
    return {m,z,e,state,index,raw:[D.capital[index],z,e]};
  }
  function drawChart(s,current){
    const o=+$('trace').value, teacher=D.teacher[s.state].map(y=>y[o]);
    const neural=D.capital.map(k=>forward([k,s.z,s.e],s.m.network).y[o]);
    let low=Math.min(...teacher,...neural),high=Math.max(...teacher,...neural),pad=(high-low)*.08||.01;
    low-=pad;high+=pad;
    const X=i=>70+i/(D.capital.length-1)*900,Y=v=>300-(v-low)/(high-low)*265;
    const path=values=>values.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join(' ');
    let svg='';for(let t=0;t<=4;t++){const v=low+(high-low)*t/4,y=Y(v);svg+=`<line x1="70" y1="${y}" x2="970" y2="${y}" stroke="#d5d1c8"/><text x="58" y="${y+4}" text-anchor="end">${v.toFixed(3)}</text>`;}
    [0,100,200,300].forEach(i=>svg+=`<text x="${X(i)}" y="325" text-anchor="middle">${(D.capital[i]/D.kss).toFixed(2)}</text>`);
    svg+=`<text x="70" y="18">${names[o]} / model units</text><text x="970" y="345" text-anchor="end">Capital / reference capital</text>`;
    svg+=`<path d="${path(teacher)}" fill="none" stroke="#242822" stroke-width="2.5"/><path d="${path(neural)}" fill="none" stroke="#aa452d" stroke-width="2" stroke-dasharray="7 5"/>`;
    svg+=`<line x1="${X(s.index)}" y1="35" x2="${X(s.index)}" y2="300" stroke="#888" stroke-dasharray="3 4"/><circle cx="${X(s.index)}" cy="${Y(current.y[o])}" r="4" fill="#aa452d"/>`;
    $('policy-chart').innerHTML=svg;
  }
  function update(){
    const s=selection(),m=s.m,net=m.network;
    const current=forward(s.raw,net),unit=+$('unit').value||0,fraction=+$('attenuation').value;
    const edited=forward(s.raw,net,unit,fraction);
    const predicted=names.map((_,o)=>net.W2.reduce((sum,row,l)=>sum+(current.p2[l]>0?row[unit]*net.W3[l][o]:0),0)*(-fraction*current.h1[unit])*D.norm.output_std[o]);
    const actual=edited.y.map((v,o)=>v-current.y[o]);
    $('capital-label').textContent=(s.raw[0]/D.kss).toFixed(3);
    $('outputs').innerHTML=names.map((name,o)=>`<div><h3>${name}</h3><p>Economic solution <strong>${fmt(D.teacher[s.state][s.index][o])}</strong></p><p>Neural policy <strong>${fmt(current.y[o])}</strong></p></div>`).join('');
    $('fit').textContent=`${m.training.eligible?'Passed':'FAILED'} the three-output VALIDATION gate. TEST error: ${m.fit.TEST.normalized_rmse.map((v,o)=>names[o]+' '+pct(v)).join(' · ')}. Dense 21-state grid maximum errors: ${m.fit.dense_21_state_grid.normalized_max_error.map(pct).join(' / ')} of TRAIN output SD.`;
    $('effects').innerHTML=names.map((name,o)=>`<tr><td>${name}</td><td>${fmt(predicted[o])}</td><td>${fmt(actual[o])}</td><td>${fmt(actual[o]-predicted[o])}</td></tr>`).join('');
    const switches=current.p2.filter((v,j)=>(v>0)!==(edited.p2[j]>0)).length;
    $('gates').textContent=`${switches} of ${m.width} downstream gates changed. Selected neuron activation: ${fmt(current.h1[unit])}. Units are physical output units. Displayed rounding can hide small differences.`;
    const label=m.labels;
    $('label').textContent=`Final-layer neuron ${label.unit} has DISCOVERY correlation ${label.selected_correlations[1].toFixed(3)} with productivity. Its capital and cost-of-work correlations are ${label.selected_correlations[0].toFixed(3)} and ${label.selected_correlations[2].toFixed(3)}. This is a different neuron from the first-layer intervention selector above.`;
    $('label-responses').innerHTML=label.responses.map(r=>`<tr><td>${r.input.replaceAll('_',' ')}</td><td>${pct(r.retained_response_norm_ratio)}</td><td>${pct(r.response_distortion_ratio)}</td></tr>`).join('');
    drawChart(s,current);
  }
  function modelChanged(){const old=+$('unit').value||0;$('unit').innerHTML='';const m=D.models[+$('model').value];for(let j=0;j<m.width;j++)option($('unit'),j,`Neuron ${j}`);$('unit').value=String(Math.min(old,m.width-1));update();}
  $('sweep').innerHTML=D.models.map(m=>`<tr><td>${m.width} / ${m.seed}</td><td>${m.training.parameter_count.toLocaleString()}</td><td>${m.training.eligible?'Pass':'FAIL'}</td>${m.fit.TEST.normalized_rmse.map(v=>`<td>${pct(v)}</td>`).join('')}<td>${pct(m.interventions.first_layer[2].normalized_response_rmse)}</td></tr>`).join('');
  $('study-status').textContent=`${D.models.length} trained networks · 21 joint shock states · three inputs and three outputs · local CPU`;
  $('baselines').textContent=`TEST baselines, consumption / labour / investment: affine ${D.baselines.affine.normalized_rmse.map(pct).join(' / ')}; quadratic ${D.baselines.quadratic.normalized_rmse.map(pct).join(' / ')}. All use the same TRAIN data and output scaling.`;
  $('scope').textContent=D.conclusion;
  $('teacher-check').textContent=D.teacher_summary;
  $('membership').textContent=Object.entries(D.membership).map(([name,v])=>`${name}: ${v.n.toLocaleString()} states (seed ${v.seed})`).join('; ')+'.';
  $('model').addEventListener('change',modelChanged);
  ['productivity','effort','unit','attenuation','trace'].forEach(id=>$(id).addEventListener('change',update));$('capital').addEventListener('input',update);
  $('download').addEventListener('click',()=>{const m=selection().m,blob=new Blob([JSON.stringify({model:m,normalization:D.norm,protocol:D.protocol},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`richer-H${m.width}-seed${m.seed}.json`;a.click();URL.revokeObjectURL(url);});
  window.RICHER_DEMO={forward,selection,update};modelChanged();
})();
