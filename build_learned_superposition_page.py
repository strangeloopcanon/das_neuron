"""Build the offline viewer from frozen, inspectable JSON; no training."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def payload():
    results = json.loads((ROOT/'learned_superposition_v2_results.json').read_text())
    confirmation = json.loads((ROOT/'learned_superposition_v2_confirmation_results.json').read_text())
    keys = ['seed','stage','W','bias','initial_W','initial_bias','parameters','evaluation','distribution_shifts','test_pass','gates']
    return {'models':[{k:m[k] for k in keys if k in m} for m in results['models']], 'confirmation':confirmation, 'summary':results['summary'], 'default_seed':results['default_seed'], 'confirmation_criteria':json.loads((ROOT/'learned_superposition_v2_confirmation_protocol.json').read_text())['criteria']}

if __name__ == '__main__':
    template = (ROOT/'learned_superposition_template.html').read_text()
    page = template.replace('/* DATA */', 'window.LEARNED_DATA='+json.dumps(payload(), separators=(',',':'))+';').replace('/* CODE */',(ROOT/'learned_superposition.js').read_text())
    (ROOT/'learned_superposition.html').write_text(page)
    print('Built learned_superposition.html')
