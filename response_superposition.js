(() => {
  'use strict';
  const D = window.RESPONSE_DATA;
  const $ = id => document.getElementById(id);
  const palette = ['#087f78','#315c9b','#9a5236','#744a86','#6d742c','#a04162'];
  let active = [1,0,0,0,0,0];
  const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
  function evaluate(x) {
    const h = D.encoder.map(row => dot(row,x));
    const predicted = D.decoder.map((row,i) => Math.max(0,dot(row,h)+D.bias[i]));
    return {h,predicted,actual:[...x],maxError:Math.max(...x.map((v,i)=>Math.abs(v-predicted[i])))};
  }
  const fmt = v => Math.abs(v)<1e-12 ? '0' : Number(v.toFixed(4)).toString();
  function geometry(x) {
    const {h} = evaluate(x),cx=238,cy=163,scale=70;
    let s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 330" font-family="Arial, sans-serif" role="img" aria-label="Six sector directions share a two-dimensional latent space. Black marks the sum of the selected feature vectors."><rect width="480" height="330" fill="#fff"/>';
    for(let v=-2;v<=2;v++){
      s+=`<line x1="73" y1="${cy-v*scale}" x2="403" y2="${cy-v*scale}" stroke="#e5ebed"/><line x1="${cx+v*scale}" y1="13" x2="${cx+v*scale}" y2="313" stroke="#e5ebed"/>`;
    }
    s+=`<circle cx="${cx}" cy="${cy}" r="${scale}" fill="none" stroke="#b9c8ce" stroke-dasharray="3 4"/><text x="465" y="${cy-7}" text-anchor="end" fill="#536974" font-size="12">coordinate 1</text><text x="${cx+8}" y="17" fill="#536974" font-size="12">coordinate 2</text>`;
    for(let i=0;i<6;i++){
      const vx=D.encoder[0][i],vy=D.encoder[1][i],ex=cx+scale*vx,ey=cy-scale*vy;
      s+=`<line x1="${cx}" y1="${cy}" x2="${ex}" y2="${ey}" stroke="${palette[i]}" stroke-width="${x[i]?4:1.4}" opacity="${x[i]?1:.55}"/><circle cx="${ex}" cy="${ey}" r="${x[i]?5:3}" fill="${palette[i]}"/><text x="${cx+(scale+23)*vx}" y="${cy-(scale+23)*vy+4}" text-anchor="middle" font-size="13" font-weight="${x[i]?700:400}" fill="${palette[i]}">S${i+1}</text>`;
    }
    const hx=cx+scale*h[0],hy=cy-scale*h[1];
    s+=`<line x1="${cx}" y1="${cy}" x2="${hx}" y2="${hy}" stroke="#152d38" stroke-width="2.2" stroke-dasharray="5 3"/><circle cx="${hx}" cy="${hy}" r="7" fill="#152d38" stroke="white" stroke-width="2"/><text x="${hx+10}" y="${hy-10}" font-size="13" fill="#152d38">h = (${fmt(h[0])}, ${fmt(h[1])})</text></svg>`;
    return s;
  }
  function bars(x) {
    const a=evaluate(x),width=360,scale=97;
    let s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 330" font-family="Arial, sans-serif" role="img" aria-label="True economic responses and reconstructed neural responses for all six sectors, relative to baseline consumption."><rect width="480" height="330" fill="#fff"/>';
    for(let tick=0;tick<=3;tick++){
      s+=`<line x1="${85+tick*scale}" y1="23" x2="${85+tick*scale}" y2="296" stroke="#e5ebed"/><text x="${85+tick*scale}" y="318" text-anchor="middle" font-size="12" fill="#536974">${tick}</text>`;
    }
    for(let i=0;i<6;i++){
      const y=35+i*44;
      s+=`<text x="60" y="${y+10}" text-anchor="end" font-size="13" fill="#152d38">S${i+1}</text><rect x="85" y="${y-6}" width="${x[i]*scale}" height="11" fill="#087f78"/><rect x="85" y="${y+8}" width="${a.predicted[i]*scale}" height="11" fill="#315c9b"/><text x="${Math.max(85+x[i]*scale,85+a.predicted[i]*scale)+8}" y="${y+10}" font-size="12" fill="#536974">${fmt(x[i])} / ${fmt(a.predicted[i])}</text>`;
    }
    return s+'</svg>';
  }
  function draw() {
    const a=evaluate(active),n=active.reduce((s,v)=>s+v,0);
    $('latent-geometry').innerHTML=geometry(active);
    $('response-bars').innerHTML=bars(active);
    $('support-status').textContent=n<=1 ? 'Guaranteed support: zero or one active sector. Reconstruction is exact.' : 'Combined-shock stress test: outside the guaranteed support. The economic responses remain additive.';
    $('response-error').textContent='Largest reconstruction error: '+fmt(a.maxError)+' times baseline sector consumption.';
    $('latent-values').textContent='Two latent coordinates: '+a.h.map(fmt).join(', ');
    for(let i=0;i<6;i++)$('sector-'+i).setAttribute('aria-pressed',String(Boolean(active[i])));
    $('feature-count').textContent=n+' active sector'+(n===1?'':'s');
  }
  for(let i=0;i<6;i++)$('sector-'+i).addEventListener('click',()=>{active[i]=1-active[i];draw();});
  const presets={single:[1,0,0,0,0,0],adjacent:[1,1,0,0,0,0],separated:[1,0,1,0,0,0],opposite:[1,0,0,1,0,0]};
  for(const [key,x] of Object.entries(presets))$('preset-'+key).addEventListener('click',()=>{active=[...x];draw();});
  const table=(rows,headers)=>'<table><thead><tr>'+headers.map(t=>'<th>'+t+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  $('encoder-weights').innerHTML=table(D.encoder.map((r,i)=>['h'+(i+1),...r.map(fmt)]),['coordinate','S1','S2','S3','S4','S5','S6']);
  $('decoder-weights').innerHTML=table(D.decoder.map((r,i)=>['S'+(i+1),...r.map(fmt),fmt(D.bias[i])]),['output','h1','h2','bias']);
  window.RESPONSE_DEMO={evaluate,geometry,bars,setState(x){active=[...x];draw();},getState(){return [...active];}};
  draw();
})();
