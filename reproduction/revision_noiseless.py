"""Archived statevectors through 35 qubits; Gray-code initialization and local mixing.

Run on a scheduled full-memory node. No circuit parameters are changed.
Feasible probabilities are saved individually to allow rescoring at other lambdas.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path
import numpy as np
from numba import njit, prange
from larger_noiseless import load_circuit, chunked_state, metrics, circuit_hash
from reanalysis import ry, rz
from analyze_recovered_hardware import feasible_assignments

ROOT = Path(__file__).resolve().parent


@njit(parallel=True, cache=True)
def initialize(state, h, j, theta, gamma, chunk_bits=16):
    n = len(h)
    chunk_bits = min(n, chunk_bits)
    size = 1 << chunk_bits
    for block in prange(len(state)//size):
        base = block*size
        z = np.empty(n, np.float64)
        amplitude = 1.0
        energy = 0.0
        for a in range(n):
            bit = (base >> (n-1-a)) & 1
            z[a] = 1-2*bit
            amplitude *= np.sin(theta[a]/2) if bit else np.cos(theta[a]/2)
            energy += h[a]*z[a]
        for a in range(n):
            for b in range(a+1,n):
                energy += j[a,b]*z[a]*z[b]
        state[base] = amplitude*np.exp(-1j*gamma*energy)
        previous = 0
        for step in range(1,size):
            gray = step ^ (step >> 1)
            changed = gray ^ previous
            bitpos = 0
            while (changed >> bitpos) != 1:
                bitpos += 1
            wire = n-1-bitpos
            field = h[wire]
            for other in range(n):
                field += j[wire,other]*z[other]
            energy -= 2*z[wire]*field
            ratio = np.tan(theta[wire]/2)
            amplitude *= ratio if z[wire] == 1 else 1/ratio
            z[wire] = -z[wire]
            state[base+gray] = amplitude*np.exp(-1j*gamma*energy)
            previous = gray


@njit(parallel=True, cache=True)
def mix(state, matrix, stride):
    for block in prange((len(state)//2+65535)//65536):
        for pair in range(block*65536,min((block+1)*65536,len(state)//2)):
            zero = (pair//stride)*(2*stride)+pair%stride
            one = zero+stride
            a,b = state[zero],state[one]
            state[zero] = matrix[0,0]*a+matrix[0,1]*b
            state[one] = matrix[1,0]*a+matrix[1,1]*b


def statevector(r, dtype):
    n = r['nv']
    state = np.empty(1<<n,dtype=dtype)
    j = np.zeros((n,n))
    for a,b,v in r['Jising']:
        j[a,b] = j[b,a] = v
    initialize(state,np.array(r['Hising']),j,np.array(r['theta']),r['gamma'])
    print('initialized',n,flush=True)
    for wire,theta in enumerate(r['theta']):
        matrix = ry(theta)@rz(2*r['beta'])@ry(-theta)
        mix(state,matrix,1<<(n-1-wire))
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qubits',type=int,required=True)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    tasks=load_circuit(args.qubits)
    r=tasks[0][3]
    state=statevector(r,np.complex64)
    precision=None
    if args.validate:
        reference=chunked_state(r)
        max_amp=max(float(np.max(abs(state[i:i+(1<<18)]-reference[i:i+(1<<18)]))) for i in range(0,len(state),1<<18))
        bits,values=feasible_assignments(r)
        ids=bits@(1<<np.arange(r['nv']-1,-1,-1,dtype=np.int64))
        p=np.abs(state[ids].astype(np.complex128))**2
        refp=np.abs(reference[ids])**2
        precision={'maximum_amplitude_error':max_amp,'feasible_mass_absolute_error':float(abs(p.sum()-refp.sum())),
                   'maximum_feasible_probability_error':float(np.max(abs(p-refp))),
                   'maximum_feasible_relative_error':float(np.max(abs(p-refp)/np.maximum(refp,1e-30)))}
        assert max_amp<1e-7 and precision['feasible_mass_absolute_error']<1e-7
        del reference
        gc.collect()
    norm=sum(float(np.sum(np.abs(state[i:i+(1<<20)].astype(np.complex128))**2)) for i in range(0,len(state),1<<20))
    assert abs(norm-1)<1e-6,norm
    bits,values=feasible_assignments(r)
    ids=bits@(1<<np.arange(r['nv']-1,-1,-1,dtype=np.int64))
    p=np.abs(state[ids].astype(np.complex128))**2
    optimal=np.isclose(values,values.max(),rtol=0,atol=1e-10)
    prior=np.sin(np.asarray(r['theta'])/2)**2
    result={'qubits':r['nv'],'feasible_states':len(ids),'circuit_sha256':circuit_hash(tasks[0][2]),'norm':norm,
            'precision_comparison':precision,'feasible_ids':ids.tolist(),'feasible_probabilities':p.tolist(),
            'methods':{'ideal_qaoa':metrics(p,values,optimal),
                       'initial_product':metrics(np.prod(np.where(bits,prior,1-prior),axis=1),values,optimal),
                       'uniform_feasible':metrics(np.full(len(ids),1/len(ids)),values,optimal)},
            'execution':{'seconds':time.perf_counter()-start,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                         'state_dtype':'complex64','state_bytes':state.nbytes}}
    out=ROOT/'analysis/revision'
    out.mkdir(parents=True,exist_ok=True)
    (out/f'noiseless_{args.qubits}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('feasible_ids','feasible_probabilities')},indent=2),flush=True)


if __name__=='__main__':
    main()
