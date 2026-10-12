import itertools
import json
from pathlib import Path
import unittest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from quality_analysis import assignment_score, assignment_valid, greedy_local
from reanalysis import evaluate
from sensitivity import circuit_definition, initial_and_energy, cost_gradient, adam_step

ROOT=Path(__file__).resolve().parent


class ImprovementTests(unittest.TestCase):
    def test_reported_sensitivity_against_independent_gate_simulator(self):
        from reanalysis import qasm_probs
        path=ROOT/'analysis/sensitivity_results.json'
        if not path.exists(): self.skipTest('Sensitivity output not yet available')
        result=json.loads(path.read_text())
        self.assertEqual(len(result['rows']),12)
        for row in result['rows']:
            r=row['circuit_definition']
            gates=[('ry',t,i) for i,t in enumerate(r['theta'])]
            gates += [('rz',2*r['gamma']*a,i) for i,a in enumerate(r['Hising'])]
            for a,b,j in r['Jising']:
                gates += [('cnot',a,b),('rz',2*r['gamma']*j,b),('cnot',a,b)]
            for i,t in enumerate(r['theta']):
                gates += [('ry',-t,i),('rz',2*r['beta'],i),('ry',t,i)]
            probability=qasm_probs(gates,r['nv'])
            bits,_,_=initial_and_energy(r)
            valid,scores=evaluate(bits,r)
            optimal=valid & np.isclose(scores,scores[valid].max(),rtol=0,atol=1e-10)
            self.assertAlmostEqual(probability[valid].sum(),row['methods']['ideal_qaoa']['feasibility'],places=10)
            self.assertAlmostEqual(probability[optimal].sum(),row['methods']['ideal_qaoa']['optimum_probability'],places=10)

    def test_assignment_score_and_search_budget(self):
        for r in json.loads((ROOT/'evidence/frozen_ladder.json').read_text()):
            for a in itertools.product(range(-1,r['D']),repeat=r['n']):
                if not assignment_valid(a,r):
                    continue
                bits=np.zeros((1,r['nv']),dtype=int)
                for i,d in enumerate(a):
                    if d>=0: bits[0,i*r['D']+d]=1
                valid,value=evaluate(bits,r)
                self.assertTrue(valid[0])
                self.assertAlmostEqual(assignment_score(a,r),value[0],places=12)
            for budget in [10,40,200]:
                result=greedy_local(r,budget)
                self.assertLessEqual(result['objective_evaluations'],budget)
                if result['assignment'] is not None:
                    self.assertTrue(assignment_valid(result['assignment'],r))

    def test_energy_gradient_and_adam_against_pennylane(self):
        panel=json.loads((ROOT/'evidence/sensitivity_design.json').read_text())['panels'][0]
        r=circuit_definition(panel)
        bits,initial,energy=initial_and_energy(r)
        # Verify the actual modeled polynomial, up to its irrelevant constant.
        _,score=evaluate(bits,r)
        number=bits.sum(axis=1)
        blocks=bits.reshape(-1,r['n'],r['D']).sum(axis=2)
        polynomial=-score+r['penalty']*(number-2)*(number-3)
        polynomial+=r['penalty']*(blocks*(blocks-1)).sum(axis=1)
        np.testing.assert_allclose(polynomial-energy,np.full(len(bits),(polynomial-energy)[0]),atol=1e-10)
        coefficients=list(r['Hising'])+[j for a,b,j in r['Jising']]
        ops=[qml.PauliZ(i) for i in range(r['nv'])]+[qml.PauliZ(a)@qml.PauliZ(b) for a,b,j in r['Jising']]
        h=qml.Hamiltonian(coefficients,ops)
        @qml.qnode(qml.device('default.qubit',wires=r['nv']),interface='autograd',diff_method='backprop')
        def cost(p):
            for i,t in enumerate(r['theta']): qml.RY(t,wires=i)
            for i,a in enumerate(r['Hising']): qml.RZ(2*p[0]*a,wires=i)
            for a,b,j in r['Jising']:
                qml.CNOT(wires=[a,b]); qml.RZ(2*p[0]*j,wires=b); qml.CNOT(wires=[a,b])
            for i,t in enumerate(r['theta']):
                qml.RY(-t,wires=i); qml.RZ(2*p[1],wires=i); qml.RY(t,wires=i)
            return qml.expval(h)
        params=pnp.array([.031,-.12],requires_grad=True)
        value,grad=cost_gradient(np.array(params),r,initial,energy)
        self.assertAlmostEqual(value,float(cost(params)),places=9)
        np.testing.assert_allclose(grad,qml.grad(cost)(params),rtol=1e-9,atol=1e-8)
        optimizer=qml.AdamOptimizer(.1)
        ours=np.array(params)
        first,second=np.zeros(2),np.zeros(2)
        for step in range(1,4):
            _,grad=cost_gradient(ours,r,initial,energy)
            ours,first,second=adam_step(ours,grad,first,second,step,.1)
            params=optimizer.step(cost,params)
            np.testing.assert_allclose(ours,params,atol=1e-8,rtol=1e-8)


if __name__=='__main__':
    unittest.main()
