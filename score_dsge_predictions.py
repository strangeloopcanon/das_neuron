"""Score one participant JSON; --summarize aggregates exports by participant.
Use protocol.response_schema. Test truth is recomputed; no answer list is read.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from dsge_prediction_materials import forward
ROOT=Path(__file__).resolve().parent

def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def require(ok,msg):
    if not ok: raise ValueError(msg)
def finite(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def direction(v): return 0 if abs(v)<=.002 else (1 if v>0 else -1)
def score(export,tasks_path=ROOT/'dsge_prediction_tasks.json',protocol_path=ROOT/'dsge_prediction_protocol.json'):
    tb=Path(tasks_path).read_bytes();pb=Path(protocol_path).read_bytes();tasks=json.loads(tb);protocol=json.loads(pb);model=tasks['model']
    hashes=dict(protocol_sha256=sha(pb),tasks_sha256=sha(tb),model_sha256=sha(canonical(model)),labels_sha256=sha(canonical(tasks['labels'])))
    require(tasks['metadata']['protocol_sha256']==hashes['protocol_sha256'],'task/protocol hash mismatch')
    for key in ['model_sha256','labels_sha256']: require(tasks['metadata'][key]==hashes[key],key+' mismatch')
    require(export.get('metadata')==hashes,'export metadata hashes mismatch')
    require(export.get('schema_version')==1,'schema_version must be 1')
    require(export.get('participant_kind') in ['human','demo','agent_smoke'],'invalid participant_kind')
    require(isinstance(export.get('participant_id'),str) and export['participant_id'].strip(),'missing participant_id')
    slot=export.get('slot'); require(type(slot) is int and 0<=slot<24,'invalid slot')
    allocation=tasks['allocations'][slot]
    require(all(export.get(k)==allocation[k] for k in ['condition','form']),'condition/form mismatch with assigned slot')
    rows=tasks['banks'][export['form']];ids=[t['task_id'] for t in rows]
    allids=[t['task_id'] for bank in tasks['banks'].values() for t in bank]
    require(len(allids)==len(set(allids)),'task bank duplicate IDs')
    order=export.get('task_order');require(isinstance(order,list) and len(order)==len(ids) and set(order)==set(ids),'task_order must contain every assigned ID exactly once')
    responses=export.get('responses');require(isinstance(responses,list) and len(responses)==len(ids),'incomplete responses')
    require([r.get('task_id') for r in responses]==order,'responses must match task_order')
    for k in ['materials_ms','total_ms']: require(finite(export.get(k)) and export[k]>=0,'invalid '+k)
    mapped={t['task_id']:t for t in rows}; scored=[]
    for response in responses:
        require(finite(response.get('prediction_sd')),'prediction_sd must be finite TRAIN output SD delta')
        require(finite(response.get('response_ms')) and response['response_ms']>=0,'invalid response_ms')
        require(isinstance(response.get('tool_use'),str),'tool_use must be string')
        t=mapped[response['task_id']];o=t['output_index']
        y=forward(model,t['raw_input'],t['intervention'])['normalized_output'][o]-forward(model,t['raw_input'])['normalized_output'][o]
        error=abs(response['prediction_sd']-y)
        scored.append(dict(family=t['family'],absolute_error_sd=error,numeric_correct=error<=protocol['metrics']['numeric_absolute_tolerance'],direction_correct=direction(response['prediction_sd'])==direction(y),response_ms=response['response_ms']))
    require(export['total_ms']+1e-6>=export['materials_ms']+sum(r['response_ms'] for r in responses),'total_ms smaller than recorded component times')
    def aggregate(rs): return dict(n=len(rs),mae_sd=statistics.mean(r['absolute_error_sd'] for r in rs),numeric_accuracy=statistics.mean(r['numeric_correct'] for r in rs),direction_accuracy=statistics.mean(r['direction_correct'] for r in rs),mean_response_ms=statistics.mean(r['response_ms'] for r in rs))
    return {**{k:export[k] for k in ['participant_kind','participant_id','slot','condition','form','materials_ms','total_ms']},'metadata':hashes,'overall':aggregate(scored),'families':{f:aggregate([r for r in scored if r['family']==f]) for f in protocol['families']}}

def summarize(paths,tasks_path=ROOT/'dsge_prediction_tasks.json',protocol_path=ROOT/'dsge_prediction_protocol.json'):
    included=[];excluded=[];seen=set();slots=set()
    for path in paths:
        try:
            result=score(json.loads(Path(path).read_text()),tasks_path,protocol_path)
            require(result['participant_id'] not in seen,'duplicate participant_id')
            require(result['participant_kind']!='human' or result['slot'] not in slots,'duplicate human allocation slot')
            seen.add(result['participant_id'])
            if result['participant_kind']=='human': slots.add(result['slot'])
            included.append(result)
        except (ValueError,KeyError,TypeError,OSError) as error: excluded.append(dict(path=str(path),reason=str(error)))
    groups={}
    for kind in ['human','demo','agent_smoke']:
        groups[kind]={}
        for condition in ['complete_model','mechanism_guide']:
            for form in ['A','B']:
                rs=[r for r in included if (r['participant_kind'],r['condition'],r['form'])==(kind,condition,form)]
                groups[kind][condition+'/'+form]=dict(n_participants=len(rs),participant_mean_mae_sd=statistics.mean(r['overall']['mae_sd'] for r in rs) if rs else None,participant_mean_direction_accuracy=statistics.mean(r['overall']['direction_accuracy'] for r in rs) if rs else None,participant_mean_total_ms=statistics.mean(r['total_ms'] for r in rs) if rs else None,families={f:{metric:statistics.mean(r['families'][f][metric] for r in rs) if rs else None for metric in ['mae_sd','direction_accuracy','numeric_accuracy']} for f in ['h1_attenuation','incoming_weight','h2_attenuation','sham']})
    return dict(unit='participant; no pooled-trial inference',groups=groups,included_participants=[dict(participant_id=r['participant_id'],participant_kind=r['participant_kind']) for r in included],exclusions=excluded)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('exports',nargs='+');p.add_argument('--summarize',action='store_true');p.add_argument('--tasks',type=Path,default=ROOT/'dsge_prediction_tasks.json');p.add_argument('--protocol',type=Path,default=ROOT/'dsge_prediction_protocol.json');p.add_argument('--output',type=Path)
    a=p.parse_args()
    try:
        require(a.summarize or len(a.exports)==1,'single export required without --summarize')
        result=summarize(a.exports,a.tasks,a.protocol) if a.summarize else score(json.loads(Path(a.exports[0]).read_text()),a.tasks,a.protocol)
        output=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if a.output: a.output.write_text(output)
        else: print(output,end='')
    except (ValueError,KeyError,TypeError,OSError) as error: p.error(str(error))
