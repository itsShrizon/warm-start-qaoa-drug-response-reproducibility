import unittest
import numpy as np
from revision_experiments import grover_state_gradient, grover, assignments, bits_for, anneal, neighbors, hardware_instances
from reanalysis import evaluate, formula_probs


class RevisionChecks(unittest.TestCase):
    def test_grover_gradient_and_unitary(self):
        scores=np.array([-.7,.1,.5,1.3])
        params=np.array([.37,-.8])
        state,cost,grad=grover_state_gradient(params,scores)
        u=np.ones(4)/2
        matrix=np.eye(4)+(np.exp(-1j*params[1])-1)*np.outer(u,u)
        np.testing.assert_allclose(state,matrix@(u*np.exp(1j*params[0]*scores)),atol=1e-13)
        finite=[]
        for axis in range(2):
            delta=np.zeros(2);delta[axis]=1e-6
            finite.append((grover_state_gradient(params+delta,scores)[1]-grover_state_gradient(params-delta,scores)[1])/2e-6)
        np.testing.assert_allclose(grad,finite,atol=1e-9)

    def test_feasible_neighborhood_and_budget(self):
        _,originals=hardware_instances()
        for r in originals:
            aa=assignments(r);domain=set(aa)
            bits=bits_for(aa,r)
            self.assertTrue(evaluate(bits,r)[0].all())
            for a in aa[:10]:self.assertTrue(set(neighbors(a,r))<=domain)
            result=anneal(r,0)
            self.assertLessEqual(result['evaluations'],200)
            self.assertLessEqual(result['best_score'],evaluate(bits,r)[1].max()+1e-10)
            self.assertEqual(result,anneal(r,0))

    def test_gray_statevector(self):
        from revision_noiseless import statevector
        _,originals=hardware_instances()
        for r in originals[:3]:
            bits=((np.arange(1<<r['nv'])[:,None]>>np.arange(r['nv']-1,-1,-1))&1)
            reference=formula_probs(r,bits)
            state=statevector(r,np.complex128)
            np.testing.assert_allclose(abs(state)**2,reference,rtol=1e-9,atol=1e-11)


if __name__=='__main__':unittest.main()
