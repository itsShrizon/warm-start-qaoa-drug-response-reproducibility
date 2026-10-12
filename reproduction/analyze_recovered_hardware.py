"""Verify downloaded Braket tasks; enumerate feasible assignments, never 2**35 states.

No AWS access or task submission. All inputs are bundled and hash checked.
Distinct task IDs are retained; duplicate downloads are not experimental repeats.
"""
from pathlib import Path
import hashlib
import itertools
import json
import csv
import numpy as np
from reanalysis import evaluate, parse_gates, wilson, qasm_probs
from quality_analysis import assignment_score, greedy_local

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence/recovered_hardware'


def expected_gates(r):
    gates = [('ry', t, i) for i, t in enumerate(r['theta'])]
    gates += [('rz', 2*r['gamma']*h, i) for i,h in enumerate(r['Hising']) if abs(h)>1e-12]
    for a,b,j in r['Jising']:
        gates += [('cnot', a,b), ('rz', 2*r['gamma']*j,b), ('cnot', a,b)]
    for i,t in enumerate(r['theta']):
        gates += [('ry', -t,i), ('rz', 2*r['beta'],i), ('ry', t,i)]
    return gates


def gate_match(a,b):
    return len(a)==len(b) and all(x[0]==y[0] and x[2]==y[2] and
        (x[1]==y[1] if x[0]=='cnot' else abs(x[1]-y[1])<1e-10)
        for x,y in zip(a,b))


def feasible_assignments(r):
    assignments=[]
    for m in range(r['min_size'],r['k']+1):
        for chosen in itertools.combinations(range(r['n']),m):
            if len({r['moa'][i] for i in chosen})<r['m_classes']:
                continue
            for doses in itertools.product(range(r['D']),repeat=m):
                a=[-1]*r['n']
                for i,d in zip(chosen,doses):a[i]=d
                assignments.append(tuple(a))
    bits=np.zeros((len(assignments),r['nv']),dtype=int)
    for j,a in enumerate(assignments):
        for i,d in enumerate(a):
            if d>=0:bits[j,i*r['D']+d]=1
    valid,values=evaluate(bits,r)
    assert valid.all() and len(bits)==r['n_feasible']
    independent=np.array([assignment_score(a,r) for a in assignments])
    np.testing.assert_allclose(values,independent,rtol=0,atol=1e-12)
    return bits,values


def analyze_task(raw,r,ideal_cache):
    meta=raw['taskMetadata'];nv=r['nv'];shots=meta['shots']
    assert meta['status']=='COMPLETED'
    assert raw['measuredQubits']==list(range(nv))
    assert meta['deviceParameters']['paradigmParameters']['qubitCount']==nv
    assert meta['deviceId'].endswith('/qpu/ionq/Forte-1')
    source=raw['additionalMetadata']['action']['source']
    assert not raw['additionalMetadata']['action'].get('inputs')
    gates=parse_gates(source)
    assert gate_match(gates,expected_gates(r))
    histogram={}
    for s,p in raw['measurementProbabilities'].items():
        assert len(s)==nv and set(s)<={'0','1'} and 0<=p<=1
        count=round(p*shots)
        assert abs(count-p*shots)<1e-7
        histogram[s]=count
    assert sum(histogram.values())==shots
    bits,values=feasible_assignments(r)
    high,low=float(values.max()),float(values.min())
    optimal=np.isclose(values,high,rtol=0,atol=1e-10)
    feasible_strings=[''.join(map(str,row)) for row in bits]
    assert r['opt_bitstring'] in {s for s,isopt in zip(feasible_strings,optimal) if isopt}
    assert abs(high-r['ideal_best'][2])<1e-5
    counts=np.array([histogram.get(s,0) for s in feasible_strings])
    # A separate decoder checks every observed output, including invalid outcomes.
    observed=np.array([[int(c) for c in s] for s in histogram])
    valid_observed,observed_values=evaluate(observed,r)
    assert int(counts.sum())==sum(c for c,ok in zip(histogram.values(),valid_observed) if ok)
    valid_count=int(counts.sum());optimum_count=int(counts[optimal].sum())
    prior=np.sin(np.asarray(r['theta'])/2)**2
    prior_mass=np.prod(np.where(bits==1,prior,1-prior),axis=1)
    greedy=greedy_local(r)
    greedy.pop('runtime_seconds')
    greedy['normalized_gap']=(high-greedy['score'])/(high-low)
    greedy['found_optimum']=abs(high-greedy['score'])<1e-10
    circuit_sha=hashlib.sha256(json.dumps(gates,separators=(',',':')).encode()).hexdigest()
    ideal=None
    if nv<=15:
        if circuit_sha not in ideal_cache:
            probs=qasm_probs(gates,nv)
            ids=np.array([int(s,2) for s in feasible_strings])
            ideal_cache[circuit_sha]={'feasibility':float(probs[ids].sum()),
                                    'optimum_probability':float(probs[ids[optimal]].sum())}
        ideal=ideal_cache[circuit_sha]
    problem_keys=('nv','n','D','drugs','min_size','k','m_classes','moa','h','tox','Jscore','lam')
    problem_sha=hashlib.sha256(json.dumps({k:r[k] for k in problem_keys},sort_keys=True).encode()).hexdigest()
    return {'task_uuid':meta['id'].split('/')[-1], 'created_utc':meta['createdAt'],
        'qubits':nv,'drugs':r['n'],'doses':r['D'],'name':r['name'],'shots':shots,
        'circuit_sha256':circuit_sha,'problem_sha256':problem_sha,
        'qasm_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'gate_count':len(gates),'cnot_count':sum(g[0]=='cnot' for g in gates),
        'valid_count':valid_count,'feasibility':valid_count/shots,'valid_ci95':wilson(valid_count,shots),
        'optimum_count':optimum_count,'optimum_ci95':wilson(optimum_count,shots),
        'feasible_states':len(bits),'optimum_score':high,'minimum_feasible_score':low,
        'optimal_bitstrings':[s for s,o in zip(feasible_strings,optimal) if o],
        'mean_normalized_quality':float(counts @ ((values-low)/(high-low))/valid_count) if valid_count else None,
        'best_normalized_gap':float((high-values[counts>0].max())/(high-low)) if valid_count else None,
        'initial_product_feasibility':float(prior_mass.sum()),
        'initial_product_optimum_probability':float(prior_mass[optimal].sum()),
        'uniform_feasible_optimum_probability':float(optimal.sum()/len(bits)),
        'uniform_bits_feasibility':len(bits)/(2**nv),'ideal_qaoa':ideal,
        'greedy':greedy,'histogram':histogram}


def main():
    manifest=json.loads((EVIDENCE/'manifest.json').read_text())
    assert hashlib.sha256((EVIDENCE/'frozen_instances.json').read_bytes()).hexdigest()==manifest['frozen_instances_sha256']
    for entry in manifest['files']+manifest['runners']:
        assert hashlib.sha256((EVIDENCE/entry['archived_path']).read_bytes()).hexdigest()==entry['sha256']
    frozen=json.loads((EVIDENCE/'frozen_instances.json').read_text())
    original={json.loads(p.read_text())['taskMetadata']['id'].split('/')[-1] for p in (ROOT/'evidence/tasks').glob('*.json')}
    rows=[];cache={}
    for path in (EVIDENCE/'tasks').glob('*.json'):
        raw=json.loads(path.read_text());gates=parse_gates(raw['additionalMetadata']['action']['source'])
        matches=[f for f in frozen if gate_match(gates,expected_gates(f['instance']))]
        assert matches, path.name
        # Multiple runners can contain the very same instance. Verify all matches.
        problems={json.dumps({k:f['instance'][k] for k in ('n','D','drugs','moa','h','tox','Jscore','lam','min_size','k','m_classes')},sort_keys=True) for f in matches}
        assert len(problems)==1
        row=analyze_task(raw,matches[0]['instance'],cache)
        row['matching_runners']=[f['runner'] for f in matches]
        row['original_six']=row['task_uuid'] in original
        rows.append(row)
    rows.sort(key=lambda r:(r['qubits'],r['doses'],r['created_utc']))
    groups=[]
    for sha in dict.fromkeys(r['circuit_sha256'] for r in rows):
        same=[r for r in rows if r['circuit_sha256']==sha]
        if len(same)>1:groups.append({'qubits':same[0]['qubits'],'circuit_sha256':sha,
            'task_uuids':[r['task_uuid'] for r in same],
            'dates':[r['created_utc'] for r in same],
            'shots':[r['shots'] for r in same],
            'feasibility':[r['feasibility'] for r in same],
            'optimum_counts':[r['optimum_count'] for r in same]})
    report={'input_downloads':len(manifest['files']),'unique_tasks':len(rows),
        'total_shots':sum(r['shots'] for r in rows),'distinct_circuits':len({r['circuit_sha256'] for r in rows}),
        'distinct_problems':len({r['problem_sha256'] for r in rows}),
        'ideal_simulation_max_qubits':15,'rows':rows,'identical_circuit_repeats':groups}
    (ROOT/'analysis/recovered_hardware.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['task_uuid','created_utc','qubits','drugs','doses','shots','valid_count','feasibility',
             'optimum_count','feasible_states','mean_normalized_quality','best_normalized_gap','original_six']
    with (ROOT/'analysis/recovered_hardware.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    for r in rows:print(r['name'],r['created_utc'][:16],r['shots'],r['valid_count'],r['optimum_count'],r['greedy']['objective_evaluations'],r['greedy']['found_optimum'])


if __name__=='__main__':main()
