"""Feasible SA, score-weight sensitivity, panel expansion, and Grover-mixer QAOA."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import time
import numpy as np
from quality_analysis import assignment_valid, assignment_score, greedy_local
from reanalysis import evaluate, formula_probs
from sensitivity import train, adam_step
from larger_noiseless import chunked_state

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis/revision'
WEIGHTS=(.3,.45,.6,.75,.9)
KEYS=('nv','n','D','drugs','min_size','k','m_classes','moa','h','tox','Jscore','lam')


def problem_key(r):
    return hashlib.sha256(json.dumps({k:r[k] for k in KEYS},sort_keys=True).encode()).hexdigest()


def assignments(r):
    result=[]
    for m in range(r['min_size'],r['k']+1):
        for chosen in itertools.combinations(range(r['n']),m):
            for doses in itertools.product(range(r['D']),repeat=m):
                a=[-1]*r['n']
                for i,d in zip(chosen,doses):a[i]=d
                if assignment_valid(a,r):result.append(tuple(a))
    return result


def bits_for(aa,r):
    bits=np.zeros((len(aa),r['nv']),dtype=int)
    for j,a in enumerate(aa):
        for i,d in enumerate(a):
            if d>=0:bits[j,i*r['D']+d]=1
    return bits


def neighbors(a,r):
    found=set()
    for i in range(r['n']):
        for d in range(-1,r['D']):
            b=a[:i]+(d,)+a[i+1:]
            if b!=a and assignment_valid(b,r):found.add(b)
    for i in range(r['n']):
        if a[i]<0:continue
        for j in range(r['n']):
            if a[j]>=0:continue
            for d in range(r['D']):
                b=list(a);b[i],b[j]=-1,d
                if assignment_valid(b,r):found.add(tuple(b))
    return sorted(found)


def random_feasible(r,rng):
    # Rejection of whole uniform assignment vectors makes the feasible start uniform.
    while True:
        a=tuple(int(v) for v in rng.integers(-1,r['D'],size=r['n']))
        if assignment_valid(a,r):return a


def anneal(r,seed,budget=200):
    rng=np.random.default_rng(seed)
    current=random_feasible(r,rng)
    cache={current:assignment_score(current,r)}
    trace=[cache[current]]
    linear=np.asarray(r['h'])-r['lam']*np.asarray(r['tox'])
    scale=max(float(np.max(np.abs(linear))),max((abs(v) for a,b,v in r['Jscore']),default=0),1e-6)
    neighborhood_cache={}
    # Fixed schedule: 2,000 proposals, or 200 unique evaluations, whichever comes first.
    for step in range(2000):
        if len(cache)>=budget:break
        if current not in neighborhood_cache:neighborhood_cache[current]=neighbors(current,r)
        hood=neighborhood_cache[current]
        if not hood:break
        candidate=hood[int(rng.integers(len(hood)))]
        if candidate not in cache:
            cache[candidate]=assignment_score(candidate,r)
            trace.append(cache[candidate])
        delta=cache[candidate]-cache[current]
        temperature=scale*(.001**(step/1999))
        if delta>=0 or rng.random()<math.exp(delta/temperature):current=candidate
    return {'seed':seed,'evaluations':len(cache),'best_score':max(trace),'evaluation_scores':trace}


def classical(r):
    aa=assignments(r)
    values=np.array([assignment_score(a,r) for a in aa])
    optimum=float(values.max())
    greedy=greedy_local(r)
    greedy['found_optimum']=abs(greedy['score']-optimum)<1e-10
    sa=[anneal(r,seed) for seed in range(100)]
    hits=[]
    for row in sa:
        match=[i+1 for i,s in enumerate(row.pop('evaluation_scores')) if abs(s-optimum)<1e-10]
        row['first_hit_evaluations']=match[0] if match else None
        row['found_optimum']=bool(match)
        if match:hits.append(match[0])
    return {'name':r['name'],'problem_key':problem_key(dict(r,lam=.6)),'qubits':r['nv'],'lambda':r['lam'],
            'feasible_states':len(aa),'optimum':optimum,'greedy':greedy,'sa':sa,
            'sa_success_fraction':len(hits)/100,'sa_median_first_hit':float(np.median(hits)) if hits else None,
            'sa_median_evaluations':float(np.median([x['evaluations'] for x in sa]))}


def grover_state_gradient(params,values):
    gamma,beta=params
    uniform=np.full(len(values),1/np.sqrt(len(values)))
    # Cost minimizes minus the same feasible score, with no penalty terms.
    phase=uniform*np.exp(1j*gamma*values)
    dg=1j*values*phase
    factor=np.exp(-1j*beta)-1
    overlap=np.dot(uniform,phase)
    state=phase+factor*uniform*overlap
    dg=dg+factor*uniform*np.dot(uniform,dg)
    db=-1j*np.exp(-1j*beta)*uniform*overlap
    cost=-float(np.dot(abs(state)**2,values))
    gradient=-2*np.real([np.vdot(dg,values*state),np.vdot(db,values*state)])
    return state,cost,gradient


def grover(r,seed):
    values=np.array([assignment_score(a,r) for a in assignments(r)])
    params=np.random.default_rng(seed).standard_normal(2)*.1
    first,second=np.zeros(2),np.zeros(2)
    for step in range(1,61):
        _,_,gradient=grover_state_gradient(params,values)
        params,first,second=adam_step(params,gradient,first,second,step,.1)
    state,cost,_=grover_state_gradient(params,values)
    optimum=np.isclose(values,values.max(),rtol=0,atol=1e-10)
    assert abs(float(np.vdot(state,state).real)-1)<1e-12
    return {'name':r['name'],'seed':seed,'lambda':r['lam'],'qubits':r['nv'],
            'feasible_states':len(values),'feasibility':1.0,'optimum_probability':float(np.sum(abs(state[optimum])**2)),
            'uniform_feasible':float(optimum.mean()),'final_parameters':params.tolist(),'final_cost':cost}


def hardware_instances():
    frozen=json.loads((ROOT/'evidence/recovered_hardware/frozen_instances.json').read_text())
    originals=json.loads((ROOT/'evidence/frozen_ladder.json').read_text())
    recovered=json.loads((ROOT/'analysis/recovered_hardware.json').read_text())['rows']
    wanted={r['problem_sha256'] for r in recovered}
    unique={}
    for r in originals+[x['instance'] for x in frozen]:
        key=problem_key(r)
        if key in wanted:unique.setdefault(key,r)
    assert len(unique)==14,len(unique)
    return list(unique.values()),originals


def train_job(job):
    r,seed=job
    result=train(r,seed,60,.1)
    result['lambda']=r['lam']
    result['cell_line']=r.get('cell_line','CAMA-1')
    result.pop('training_history')
    return result


def save(name,rows):
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/name).write_text(json.dumps(rows,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['classical','panels','grover','rescore'],required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    design=json.loads((ROOT/'evidence/revision_design.json').read_text())
    panels=design['panels']
    original_panels=json.loads((ROOT/'evidence/sensitivity_design.json').read_text())['panels']
    hardware,originals=hardware_instances()
    if args.mode=='classical':
        jobs=[dict(r,lam=lam) for r in hardware for lam in WEIGHTS]
        jobs += [dict(r,lam=.6) for r in panels]
        jobs += [dict(r,lam=lam) for r in original_panels for lam in WEIGHTS if lam!=.6]
        with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
            rows=[]
            for row in pool.map(classical,jobs):
                rows.append(row);save('classical.json',rows)
                print('classical',row['name'],row['lambda'],row['sa_success_fraction'],flush=True)
        try:
            import ortools
            save('ortools.json',{'installed':True,'version':ortools.__version__})
        except ImportError:save('ortools.json',{'installed':False})
    elif args.mode=='panels':
        jobs=[(dict(r,lam=.6),seed) for r in panels for seed in range(3)]
        jobs += [(dict(r,lam=lam),seed) for r in original_panels for lam in WEIGHTS if lam!=.6 for seed in range(3)]
        with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
            rows=[]
            for row in pool.map(train_job,jobs):
                rows.append(row);save('panels.json',rows)
                print('panel',row['panel'],row['lambda'],row['seed'],flush=True)
    elif args.mode=='grover':
        rows=[grover(r,seed) for r in originals+original_panels for seed in range(3)]
        save('grover.json',rows)
    else:
        # Separate fixed archived distributions from redefined optimum sets.
        from analyze_recovered_hardware import expected_gates, gate_match
        from reanalysis import parse_gates
        from larger_noiseless import circuit_hash
        frozen=json.loads((ROOT/'evidence/recovered_hardware/frozen_instances.json').read_text())
        circuits={}
        for path in sorted((ROOT/'evidence/recovered_hardware/tasks').glob('*.json')):
            raw=json.loads(path.read_text());gates=parse_gates(raw['additionalMetadata']['action']['source'])
            r=next(f['instance'] for f in frozen if gate_match(gates,expected_gates(f['instance'])))
            key=circuit_hash(gates)
            circuits.setdefault(key,{'r':r,'tasks':[]})['tasks'].append(raw)
        rows=[]
        for key,entry in circuits.items():
            r=entry['r'];aa=assignments(r);bits=bits_for(aa,r)
            ids=bits@(1<<np.arange(r['nv']-1,-1,-1,dtype=np.int64))
            prior=np.sin(np.array(r['theta'])/2)**2
            initial=np.prod(np.where(bits,prior,1-prior),axis=1)
            if r['nv']<=25:
                if r['nv']<=15:
                    fullbits=((np.arange(1<<r['nv'])[:,None]>>np.arange(r['nv']-1,-1,-1))&1)
                    ideal=formula_probs(r,fullbits)[ids]
                else:
                    state=chunked_state(r);ideal=abs(state[ids])**2;del state
            else:
                p=OUT/f"noiseless_{r['nv']}.json"
                if p.exists():
                    result=json.loads(p.read_text());assert result['circuit_sha256']==key
                    lookup=dict(zip(result['feasible_ids'],result['feasible_probabilities']))
                    ideal=np.array([lookup[int(i)] for i in ids])
                else:ideal=None
            for lam in WEIGHTS:
                model=dict(r,lam=lam)
                valid,values=evaluate(bits,model);assert valid.all()
                optimal=np.isclose(values,values.max(),rtol=0,atol=1e-10)
                hardware_rows=[]
                for task in sorted(entry['tasks'],key=lambda t:t['taskMetadata']['createdAt']):
                    shots=task['taskMetadata']['shots'];probs=task['measurementProbabilities']
                    count=sum(round(probs.get(format(int(i),f"0{r['nv']}b"),0)*shots) for i in ids[optimal])
                    hardware_rows.append({'task_uuid':task['taskMetadata']['id'].split('/')[-1],
                                          'shots':shots,'optimum_count':count,'optimum_fraction':count/shots})
                rows.append({'name':r['name'],'qubits':r['nv'],'problem_key':problem_key(r),'circuit_sha256':key,
                             'lambda':lam,'feasible_states':len(aa),'optimum':float(values.max()),
                             'optimal_bitstrings':[format(int(i),f"0{r['nv']}b") for i in ids[optimal]],
                             'initial_product':float(initial[optimal].sum()),'uniform_feasible':float(optimal.mean()),
                             'ideal_qaoa':float(ideal[optimal].sum()) if ideal is not None else None,'hardware':hardware_rows})
            save('rescoring.json',rows)
            print('rescored',r['name'],r['nv'],flush=True)


if __name__=='__main__':main()
