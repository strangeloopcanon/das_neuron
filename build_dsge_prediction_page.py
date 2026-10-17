"""Build portable answer-blind study page from frozen source bytes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def build():
 raw=(ROOT/'dsge_prediction_tasks.json').read_bytes(); data=json.loads(raw)
 fields=('task_id','bank','family','output_index','output_name','raw_input','normalized_input','state','baseline','intervention')
 data['banks']={k:[dict(t) if k=='practice' else {f:t[f] for f in fields} for t in data['banks'][k]] for k in ('practice','A','B')}
 data['metadata']['tasks_sha256']=hashlib.sha256(raw).hexdigest()
 payload=json.dumps(data,separators=(',',':')).replace('</','<\\/')
 html=(ROOT/'dsge_prediction_template.html').read_text().replace('/*__STYLE__*/',(ROOT/'style.css').read_text()).replace('/*__DATA__*/',payload).replace('/*__PYTHON__*/',json.dumps((ROOT/'dsge_prediction_materials.py').read_text())).replace('/*__APP__*/',(ROOT/'dsge_prediction.js').read_text())
 (ROOT/'dsge_prediction.html').write_text(html)
 return data
if __name__=='__main__':
 d=build(); print('Built answer-blind page; tasks SHA256 '+d['metadata']['tasks_sha256'])
