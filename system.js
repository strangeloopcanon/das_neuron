(() => {
  'use strict';
  const D=window.SYSTEM_DATA,$=id=>document.getElementById(id);
  const outputNames=['Consumption','Labour','Investment'],inputNames=['Capital','Log productivity','Work-cost shock'];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const shortLabel=l=>l.correlation_defined?`${['Capital','Productivity','Work cost'][l.strongest_input_index]} ${l.correlations[l.strongest_input_index]>=0?'↑':'↓'} r=${l.correlations[l.strongest_input_index].toFixed(2)}`:(l.activation_frequency===0?'Inactive (DISCOVERY)':'Constant (DISCOVERY)');
  const f=x=>Number(x).toFixed(5),signed=x=>(x>=0?'+':'')+f(x),pct=x=>(100*x).toFixed(1)+'%';
  const option=(select,value,label)=>{const node=document.createElement('option');node.value=String(value);node.textContent=label;select.appendChild(node);};
  const norm=D.normalization;
  function forward(raw,net){
    const x=raw.map((v,i)=>(v-norm.input_mean[i])/norm.input_std[i]);
    const affine=(a,W,b)=>W.map((row,i)=>row.reduce((s,w,j)=>s+w*a[j],b[i]));
    const p1=affine(x,net.W1,net.b1),h1=p1.map(v=>Math.max(v,0));
    const p2=affine(h1,net.W2,net.b2),h2=p2.map(v=>Math.max(v,0));
    const yn=net.b3.map((b,o)=>h2.reduce((s,h,j)=>s+h*net.W3[j][o],b));
    return {raw,x,p1,h1,p2,h2,yn,y:yn.map((v,o)=>v*norm.output_std[o]+norm.output_mean[o])};
  }
  const zs=[...new Set(D.features.map(v=>v[0]))],es=[...new Set(D.features.map(v=>v[1]))];
  D.models.forEach((m,i)=>option($('system-model'),i,`${m.width*2} hidden neurons · ${m.width*m.width+8*m.width+3} weights and biases · seed ${m.seed}`));
  $('system-model').value=String(D.models.findIndex(m=>m.id===D.default_model_id));
  zs.forEach((v,i)=>option($('target-productivity'),i,`z = ${Math.exp(v).toFixed(3)}`));
  es.forEach((v,i)=>option($('target-cost'),i,`cost × ${Math.exp(v).toFixed(3)}`));
  $('target-productivity').value=String(zs.length-1);$('target-cost').value=String(es.length-1);
  $('system-output').value='0';$('inspect-layer').value='0';
  function selection(){
    const m=D.models[+$('system-model').value],index=+$('system-capital').value;
    const z0=zs[Math.floor(zs.length/2)],e0=es[Math.floor(es.length/2)],z=zs[+$('target-productivity').value],e=es[+$('target-cost').value];
    const k=D.capital[index],raws=[[k,z0,e0],[k,z,e0],[k,z0,e],[k,z,e]];
    const states=raws.map(raw=>D.features.findIndex(v=>v[0]===raw[1]&&v[1]===raw[2]));
    return {m,index,k,raws,states,o:+$('system-output').value};
  }
  function account(s){
    const snapshots=s.raws.map(raw=>forward(raw,s.m.network));
    const contributions=s.m.network.W3.map((row,j)=>row.map((v,o)=>norm.output_std[o]*v*(snapshots[3].h2[j]-snapshots[0].h2[j])));
    const interactions=s.m.network.W3.map((row,j)=>row.map((v,o)=>norm.output_std[o]*v*(snapshots[3].h2[j]-snapshots[1].h2[j]-snapshots[2].h2[j]+snapshots[0].h2[j])));
    const delta=snapshots[3].y.map((v,o)=>v-snapshots[0].y[o]);
    const additive=delta.map((_,o)=>snapshots[1].y[o]+snapshots[2].y[o]-2*snapshots[0].y[o]);
    const interaction=delta.map((v,o)=>v-additive[o]);
    const teacher=s.states.map(state=>D.teacher[state][s.index]);
    const economicInteraction=delta.map((_,o)=>teacher[3][o]-teacher[1][o]-teacher[2][o]+teacher[0][o]);
    return {snapshots,contributions,interactions,delta,additive,interaction,teacher,economicInteraction};
  }
  function weightRecords(m){
    const n=m.network,records=[];
    n.W1.forEach((row,j)=>{row.forEach((v,i)=>records.push(['W1',`H1.${j}`,`x.${i}`,v]));records.push(['b1',`H1.${j}`,'bias',n.b1[j]]);});
    n.W2.forEach((row,j)=>{row.forEach((v,i)=>records.push(['W2',`H2.${j}`,`H1.${i}`,v]));records.push(['b2',`H2.${j}`,'bias',n.b2[j]]);});
    n.b3.forEach((b,o)=>{n.W3.forEach((row,j)=>records.push(['W3',`y.${o}`,`H2.${j}`,row[o]]));records.push(['b3',`y.${o}`,'bias',b]);});
    return records;
  }
  function drawMatrices(m){
    const n=m.network;
    function matrix(title,W,b,columns,rows,open=false){
      return `<details${open?' open':''}><summary>${title} · ${W.length*(columns.length+1)} stored values</summary><div class="weight-grid"><table><thead><tr><th>Receiving unit</th>${columns.map(v=>`<th>${esc(v)}</th>`).join('')}<th>Bias</th></tr></thead><tbody>${W.map((row,j)=>`<tr><th>${esc(rows[j])}</th>${row.map(v=>`<td>${String(v)}</td>`).join('')}<td>${String(b[j])}</td></tr>`).join('')}</tbody></table></div></details>`;
    }
    const units=prefix=>Array.from({length:m.width},(_,j)=>`${prefix}.${j}`);
    $('weight-matrices').innerHTML=matrix('Inputs → first hidden layer',n.W1,n.b1,inputNames,units('H1'),true)+
      matrix('First → second hidden layer',n.W2,n.b2,units('H1'),units('H2'))+
      matrix('Second hidden layer → outputs',outputNames.map((_,o)=>n.W3.map(row=>row[o])),n.b3,units('H2'),outputNames);
    $('normalization').innerHTML='<table><thead><tr><th>Variable</th><th>TRAIN mean</th><th>TRAIN standard deviation</th></tr></thead><tbody>'+inputNames.map((v,j)=>`<tr><td>${v}</td><td>${norm.input_mean[j]}</td><td>${norm.input_std[j]}</td></tr>`).join('')+outputNames.map((v,j)=>`<tr><td>${v}</td><td>${norm.output_mean[j]}</td><td>${norm.output_std[j]}</td></tr>`).join('')+'</tbody></table>';
    $('all-labels').innerHTML=['H1','H2'].flatMap(layer=>m.labels[layer].map((l,j)=>`<tr><td>${layer}.${j}</td><td class="wrap">${esc(l.label)}</td>${l.correlations.map(v=>`<td>${l.correlation_defined?v.toFixed(3):'undefined'}</td>`).join('')}<td>${pct(l.activation_frequency)}</td></tr>`)).join('');
    $('system-inventory').textContent=`${2*m.width} labelled hidden neurons. ${weightRecords(m).length.toLocaleString()} weights and biases. Three known economic inputs and three predicted decisions. No weights are withheld.`;
  }
  function populateUnits(){const m=selection().m,old=+$('inspect-unit').value||0;$('inspect-unit').innerHTML='';for(let j=0;j<m.width;j++)option($('inspect-unit'),j,`${+$('inspect-layer').value===0?'H1':'H2'}.${j}`);$('inspect-unit').value=String(Math.min(old,m.width-1));}
  function inspector(s,a){
    const layer=+$('inspect-layer').value,j=+$('inspect-unit').value||0,key=layer===0?'H1':'H2',l=s.m.labels[key][j],n=s.m.network;
    const weights=layer===0?n.W1[j]:n.W2[j],bias=layer===0?n.b1[j]:n.b2[j];
    const terms=weights.map((w,i)=>`${w} × ${layer===0?'x':'h1'}[${i}]`).join(' + ');
    const base=a.snapshots[0][layer===0?'h1':'h2'][j],target=a.snapshots[3][layer===0?'h1':'h2'][j];
    const corrText=l.correlation_defined?`capital ${l.correlations[0].toFixed(3)}, productivity ${l.correlations[1].toFixed(3)}, work cost ${l.correlations[2].toFixed(3)}`:'undefined because this unit is constant on DISCOVERY';
    $('neuron-inspector').innerHTML=`<p><strong>${key}.${j}: ${esc(l.label)}</strong></p><p>DISCOVERY correlations: ${corrText}. Active on ${pct(l.activation_frequency)} of DISCOVERY states.</p><code>${key.toLowerCase()}[${j}] = max(0, ${esc(terms)} + ${bias})</code><p>Current baseline activation ${f(base)} → both-shock activation ${f(target)}.</p>`;
  }
  function drawNetwork(s,a){
    const m=s.m,n=m.network,height=Math.max(570,m.width*20+100),y=j=>70+j*(height-130)/Math.max(1,m.width-1);
    const iy=i=>height*(.3+.2*i),ox=915,oy=o=>height*(.3+.2*o),selectedLayer=+$('inspect-layer').value,selectedUnit=+$('inspect-unit').value||0;
    let svg=`<text x="60" y="25">Inputs</text><text x="315" y="25" text-anchor="middle">First hidden layer</text><text x="650" y="25" text-anchor="middle">Second hidden layer</text><text x="915" y="25" text-anchor="middle">Decisions</text>`;
    function edge(x1,y1,x2,y2,w,selected){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${selected?'#bf492e':'#80877d'}" stroke-opacity="${selected?.6:.09}" stroke-width="${selected?1.1:.65}"${w<0?' stroke-dasharray="3 3"':''}/>`;}
    n.W1.forEach((row,j)=>row.forEach((w,i)=>svg+=edge(85,iy(i),315,y(j),w,selectedLayer===0&&selectedUnit===j)));
    n.W2.forEach((row,j)=>row.forEach((w,i)=>svg+=edge(315,y(i),650,y(j),w,(selectedLayer===0&&selectedUnit===i)||(selectedLayer===1&&selectedUnit===j))));
    n.W3.forEach((row,j)=>row.forEach((w,o)=>svg+=edge(650,y(j),ox,oy(o),w,selectedLayer===1&&selectedUnit===j)));
    inputNames.forEach((v,i)=>svg+=`<circle cx="85" cy="${iy(i)}" r="6" fill="#25342f"/><text x="12" y="${iy(i)-12}">${v}</text>`);
    outputNames.forEach((v,o)=>svg+=`<circle cx="${ox}" cy="${oy(o)}" r="6" fill="#25342f"/><text x="${ox-20}" y="${oy(o)-15}">${v}</text>`);
    ['H1','H2'].forEach((key,layer)=>m.labels[key].forEach((l,j)=>{
      const active=a.snapshots[0][layer===0?'h1':'h2'][j]>0,after=a.snapshots[3][layer===0?'h1':'h2'][j]>0,x=layer===0?315:650,selected=layer===selectedLayer&&j===selectedUnit;
      svg+=`<circle class="neuron" data-layer="${layer}" data-unit="${j}" tabindex="0" role="button" aria-label="Inspect ${key}.${j}, ${esc(l.label)}" cx="${x}" cy="${y(j)}" r="${selected?6:4.5}" fill="${active?'#25342f':'#f2efe7'}" stroke="${active!==after?'#bf492e':'#25342f'}" stroke-width="${active!==after?2:1}"><title>${key}.${j}: ${esc(l.label)}</title></circle><text x="${layer===0?302:663}" y="${y(j)+4}" text-anchor="${layer===0?'end':'start'}">${key}.${j} ${esc(shortLabel(l))}</text>`;
    }));
    $('system-network').setAttribute('viewBox',`0 0 1000 ${height}`);$('system-network').innerHTML=svg;
    $('network-caption').textContent=`All ${m.width*3+m.width*m.width+m.width*3} inter-unit weights are drawn. Biases are in the tables. Dashed edges have negative weights. Filled nodes are active at baseline; outlined nodes are inactive. Accent outlines mark gates that differ with both shocks. Selected-neuron connections are highlighted. Labels remain DISCOVERY associations.`;
  }
  function drawRoutes(s,a){
    const o=s.o,values=a.contributions.map(row=>row[o]),interaction=a.interactions.map(row=>row[o]);
    const max=Math.max(...values.map(Math.abs),1e-14),height=70+s.m.width*24,X=v=>620+v/max*270,Y=j=>55+24*j;
    let svg=`<text x="15" y="20">Fixed neuron label</text><text x="620" y="20" text-anchor="middle">Contribution to ${outputNames[o].toLowerCase()} change</text><line x1="620" y1="35" x2="620" y2="${height-15}" stroke="#888"/>`;
    values.forEach((v,j)=>{const x=X(v),l=s.m.labels.H2[j];svg+=`<text x="15" y="${Y(j)+4}">H2.${j} ${esc(l.label)}</text><rect x="${Math.min(x,620)}" y="${Y(j)-7}" width="${Math.abs(x-620)}" height="13" fill="${v>=0?'#bf492e':'none'}" stroke="${v>=0?'#bf492e':'#25342f'}"/><text x="${v>=0?x+6:x-6}" y="${Y(j)+4}" text-anchor="${v>=0?'start':'end'}">${signed(v)}</text>`;});
    $('route-bars').setAttribute('viewBox',`0 0 1000 ${height}`);$('route-bars').innerHTML=svg;
    $('route-table').innerHTML=values.map((v,j)=>`<tr><td class="wrap">H2.${j} / ${esc(s.m.labels.H2[j].label)}</td><td>${signed(a.snapshots[3].h2[j]-a.snapshots[0].h2[j])}</td><td>${s.m.network.W3[j][o]}</td><td>${signed(v)}</td><td>${signed(interaction[j])}</td></tr>`).join('');
    const sum=values.reduce((s,v)=>s+v,0),isum=interaction.reduce((s,v)=>s+v,0);
    $('reconstruction').textContent=`All ${values.length} routes sum to ${signed(sum)}. Neural joint response: ${signed(a.delta[o])}. Accounting residual: ${(sum-a.delta[o]).toExponential(2)}. Interaction-accounting residual: ${(isum-a.interaction[o]).toExponential(2)}. Numerical values use model units.`;
  }
  function update(){
    const s=selection(),a=account(s),o=s.o;
    $('system-capital-label').textContent=(s.k/D.kss).toFixed(3);
    $('baseline-description').textContent=`Baseline: productivity ${Math.exp(s.raws[0][1]).toFixed(3)}, work-cost multiplier ${Math.exp(s.raws[0][2]).toFixed(3)}. Shock states are actual states of the teacher's 21-state Markov benchmark.`;
    $('four-states').innerHTML=outputNames.map((name,o)=>`<tr><td>${name}</td>${a.snapshots.map((v,i)=>`<td>${f(v.y[o])}<br><span class="teacher-value">DSGE ${f(a.teacher[i][o])}</span></td>`).join('')}</tr>`).join('');
    $('joint-comparison').textContent=`${outputNames[o]}: separate responses sum to ${signed(a.additive[o])}. The joint response is ${signed(a.delta[o])}. Their difference is ${signed(a.interaction[o])}.`;
    const diff=a.interaction[o]-a.economicInteraction[o];
    const signWarning=a.interaction[o]*a.economicInteraction[o]<0?' The network gets the economic interaction’s sign wrong in this comparison.':'';
    $('interaction-detail').textContent=`Economic interaction: ${signed(a.economicInteraction[o])}. Neural interaction: ${signed(a.interaction[o])}. Approximation discrepancy: ${signed(diff)}.${signWarning} Neural interaction includes approximation artifacts; its entire value cannot be treated as an economic effect.`;
    const fidelity=s.m.interaction_fidelity;
    $('interaction-fidelity').textContent=`Interaction fidelity check: for the fixed highest-productivity/highest-work-cost contrast across 301 capital states, this network’s ${outputNames[o].toLowerCase()} interaction error is ${fidelity.error_rms_over_teacher_rms[o].toFixed(2)}× the economic interaction’s RMS size. Sign agreement is ${pct(fidelity.sign_agreement[o])}. Exact neural accounting succeeds; the small economic interaction is not faithfully recovered when these errors are large.`;
    const values=a.contributions.map(row=>row[o]),positive=values.filter(v=>v>0).reduce((s,v)=>s+v,0),negative=values.filter(v=>v<0).reduce((s,v)=>s+v,0);
    $('cancellation-explanation').textContent=`Positive routes contribute ${signed(positive)}. Negative routes contribute ${signed(negative)}. Their net is ${signed(a.delta[o])}. The label alone specifies none of these state-dependent amounts.`;
    const switchCount=key=>a.snapshots[0][key].filter((v,j)=>(v>0)!==(a.snapshots[3][key][j]>0)).length;
    const changed=values.filter(v=>Math.abs(v)>1e-12).length;
    $('gate-explanation').textContent=`${switchCount('p1')} first-layer gates and ${switchCount('p2')} second-layer gates differ between baseline and both shocks. ${changed}/${s.m.width} final-layer neurons contribute a nonzero change to this decision. Compare capital states using the same labels and weights.`;
    const overlap=s.m.top5_route_overlap_low_high.cost[o];
    $('route-change-example').textContent=`For a work-cost-only shock, the five largest ${outputNames[o].toLowerCase()} contributions come from H2 units ${overlap.low_units.join(', ')} near capital .6, and ${overlap.high_units.join(', ')} near capital 1.6. ${overlap.intersection_count}/5 appear in both lists. These are fixed low/high comparisons with the highest work-cost state and baseline productivity. The labels and weights are identical. Ranking uses absolute contribution, not a percentage of behavior explained.`;
    inspector(s,a);drawNetwork(s,a);drawRoutes(s,a);
    window.SYSTEM_DEMO.current={selection:s,account:a};
  }
  function modelChanged(){populateUnits();drawMatrices(selection().m);update();}
  $('system-model').addEventListener('change',modelChanged);
  $('inspect-layer').addEventListener('change',()=>{populateUnits();update();});$('inspect-unit').addEventListener('change',update);
  ['target-productivity','target-cost','system-output'].forEach(id=>$(id).addEventListener('change',update));$('system-capital').addEventListener('input',update);
  function selectNode(event){const node=event.target;if(!node||!node.getAttribute)return;const layer=node.getAttribute('data-layer'),unit=node.getAttribute('data-unit');if(layer===null||unit===null)return;$('inspect-layer').value=layer;populateUnits();$('inspect-unit').value=unit;update();}
  $('system-network').addEventListener('click',selectNode);$('system-network').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectNode(event);}});
  $('trace-system').addEventListener('click',()=>{const reveal=$('system-reveal');reveal.hidden=!reveal.hidden;$('trace-system').setAttribute('aria-expanded',String(!reveal.hidden));$('trace-system').textContent=reveal.hidden?'Trace the combined computation':'Hide the system trace';});
  function download(text,type,name){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);}
  $('system-download').addEventListener('click',()=>{const m=selection().m;download(JSON.stringify({model:m,normalization:norm,provenance:D.provenance},null,2),'application/json',`system-${m.id}.json`);});
  $('weights-csv').addEventListener('click',()=>{const m=selection().m;download('tensor,receiving_unit,sending_unit,value\n'+weightRecords(m).map(row=>row.join(',')).join('\n')+'\n','text/csv',`all-weights-${m.id}.csv`);});
  $('system-provenance').textContent='Labels use 2,048 DISCOVERY states (seed112003). TRAIN, VALIDATION, TEST, and INTERVENTION_TEST remain separate in the original study. This walkthrough is an illustrative analysis of frozen checkpoints, not a new blinded test.';
  window.SYSTEM_DEMO={forward,selection,account,weightRecords,update,current:null};modelChanged();
})();
