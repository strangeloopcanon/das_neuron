"""Package the constructed response circuit, retaining the separate learned trial."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def build():
    evidence=json.loads((ROOT/'constructive_superposition.json').read_text())
    # Keep browser data minimal. The evidence JSON remains downloadable in full.
    network=evidence.get('network',evidence)
    data={key:network[key] for key in ('encoder','decoder','bias')}
    page=(ROOT/'response_superposition_template.html').read_text()
    page=page.replace('/*RESPONSE_DATA*/','window.RESPONSE_DATA='+json.dumps(data,separators=(',',':'))+';')
    page=page.replace('/*RESPONSE_SCRIPT*/',(ROOT/'response_superposition.js').read_text())
    assert '/*RESPONSE_' not in page
    (ROOT/'response_superposition.html').write_text(page)
    print('Built response_superposition.html: constructed circuit; no training.')

if __name__=='__main__':build()
