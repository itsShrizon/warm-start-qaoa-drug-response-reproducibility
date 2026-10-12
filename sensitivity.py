"""Fixed panel sensitivity experiment: analytic one-layer QAOA, no hardware calls."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import platform
import time
import numpy as np
from reanalysis import evaluate, formula_probs, ry, rz, single
from quality_analysis import greedy_local

ROOT = Path(__file__).resolve().parent


def circuit_definition(panel):
    r = dict(panel)
    nv = r['nv']
    c = np.asarray(r['h']).ravel()-r['lam']*np.asarray(r['tox']).ravel()
    j = np.zeros((nv,nv))
    for a,b,value in r['Jscore']:
        j[a,b] = j[b,a] = value
    penalty = 3*(float(abs(c).max())+float(abs(j).sum(axis=1).max())+1e-6)
    q = -j/2
    np.fill_diagonal(q,-c-4*penalty)
    q += penalty*(np.ones((nv,nv))-np.eye(nv))
    for i in range(r['n']):
        for d in range(r['D']):
            for e in range(d+1,r['D']):
                a,b=i*r['D']+d,i*r['D']+e
                q[a,b] += penalty
                q[b,a] += penalty
    h = -q.sum(axis=1)/2
    jj = q/2
    g = c.copy()
    if g.std() > 1e-9:
        g = (g-g.mean())/g.std()
    prior = 1/(1+np.exp(-g/0.5))
    prior = np.clip(prior*2.5/prior.sum(),1e-3,1-1e-3)
    r.update({'Hising':h.tolist(), 'Jising':[[a,b,float(jj[a,b])] for a in range(nv) for b in range(a+1,nv)],
              'theta':(2*np.arcsin(np.sqrt(prior))).tolist(), 'penalty':penalty})
    return r


def initial_and_energy(r):
    nv=r['nv']
    bits=((np.arange(1<<nv)[:,None] >> np.arange(nv-1,-1,-1))&1)
    z=1-2*bits
    energy=z@np.asarray(r['Hising'])
    for a,b,j in r['Jising']:
        energy += j*z[:,a]*z[:,b]
    initial=np.array([1+0j])
    for theta in r['theta']:
        initial=np.kron(initial,[math.cos(theta/2),math.sin(theta/2)])
    return bits,initial,energy


def cost_gradient(params,r,initial,energy):
    gamma,beta=params
    state=initial*np.exp(-1j*gamma*energy)
    dg=-1j*energy*state
    db=np.zeros_like(state)
    for wire,theta in enumerate(r['theta']):
        left,right=ry(theta),ry(-theta)
        mix=left@rz(2*beta)@right
        derivative=left@np.diag([-1j*np.exp(-1j*beta),1j*np.exp(1j*beta)])@right
        db=single(db,mix,wire,r['nv'])+single(state,derivative,wire,r['nv'])
        dg=single(dg,mix,wire,r['nv'])
        state=single(state,mix,wire,r['nv'])
    cost=float(np.dot(abs(state)**2,energy))
    gradient=2*np.real([np.vdot(dg,energy*state),np.vdot(db,energy*state)])
    return cost,gradient


def adam_step(params,grad,first,second,step,lr):
    first=.9*first+.1*grad
    second=.99*second+.01*grad*grad
    rate=lr*np.sqrt(1-.99**step)/(1-.9**step)
    return params-rate*first/(np.sqrt(second)+1e-8),first,second


def train(panel,seed,steps,lr):
    r=circuit_definition(panel)
    bits,initial,energy=initial_and_energy(r)
    params=np.random.default_rng(seed).standard_normal(2)*.1
    initialization=params.copy()
    first,second=np.zeros(2),np.zeros(2)
    history=[]
    start=time.perf_counter()
    for step in range(1,steps+1):
        cost,gradient=cost_gradient(params,r,initial,energy)
        history.append({'step':step-1,'cost':cost,'params':params.tolist(),'gradient':gradient.tolist()})
        params,first,second=adam_step(params,gradient,first,second,step,lr)
    final_cost,_=cost_gradient(params,r,initial,energy)
    runtime=time.perf_counter()-start
    r.update({'gamma':float(params[0]),'beta':float(params[1])})
    probabilities=formula_probs(r,bits)
    valid,scores=evaluate(bits,r)
    optimum=float(scores[valid].max())
    low=float(scores[valid].min())
    optimal=valid & np.isclose(scores,optimum,rtol=0,atol=1e-10)
    quality=(scores-low)/(optimum-low)
    methods={}
    for name,p in [('ideal_qaoa',probabilities),('warm_start_only',abs(initial)**2),
                   ('uniform_feasible',valid/valid.sum())]:
        mass=float(p[valid].sum())
        opt=float(p[optimal].sum())
        methods[name]={'feasibility':mass,'optimum_probability':opt,
            'conditional_mean_normalized':float(np.dot(p[valid],quality[valid])/mass),
            'hit_200':float(-np.expm1(200*np.log1p(-opt))) if opt<1 else 1.0}
    search=greedy_local(r)
    search.update({'found_optimum':abs(search['score']-optimum)<1e-10,
                   'normalized_gap':(optimum-search['score'])/(optimum-low)})
    return {'panel':r['name'],'drugs':r['drugs'],'seed':seed,'steps':steps,
            'initialization':initialization.tolist(),'final_parameters':params.tolist(),
            'final_cost':final_cost,'training_runtime_seconds':runtime,
            'training_gradient_evaluations':steps,'final_diagnostic_evaluations':1,
            'optimum':optimum,'optimal_bitstrings':[format(i,'012b') for i in np.flatnonzero(optimal)],
            'feasible_states':int(valid.sum()),'methods':methods,'greedy_local':search,
            'training_history':history,'circuit_definition':r}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/sensitivity_results.json')
    args=parser.parse_args()
    design_path=ROOT/'evidence/sensitivity_design.json'
    design=json.loads(design_path.read_text())
    rows=[]
    result={'design_sha256':hashlib.sha256(design_path.read_bytes()).hexdigest(),
        'kind':'simulation sensitivity; no additional hardware observations',
        'software':{'python':platform.python_version(),'numpy':np.__version__},
        'execution':{'hostname':platform.node(),'architecture':platform.machine(),
                     'slurm_job_id':os.environ.get('SLURM_JOB_ID')},'rows':rows}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    for panel in design['panels'][:1] if args.smoke else design['panels']:
        for seed in design['training_seeds'][:1] if args.smoke else design['training_seeds']:
            row=train(panel,seed,2 if args.smoke else design['steps'],design['learning_rate'])
            rows.append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(panel['name'],seed,{n:round(m['optimum_probability'],5) for n,m in row['methods'].items()},flush=True)


if __name__=='__main__':
    main()
