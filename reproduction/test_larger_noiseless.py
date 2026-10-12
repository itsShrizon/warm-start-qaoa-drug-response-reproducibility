"""Numerical equivalence and indexing checks for the bounded-memory simulator."""
import json
import unittest
import numpy as np
from larger_noiseless import ROOT, chunked_state, apply_single_inplace
from reanalysis import formula_probs, qasm_probs, parse_gates, ry


class LargerNoiselessTests(unittest.TestCase):
    def test_six_full_distributions_and_saved_metrics(self):
        frozen = json.loads((ROOT/'evidence/frozen_ladder.json').read_text())
        saved = json.loads((ROOT/'analysis/verified_results.json').read_text())['rows']
        for r, old in zip(frozen, saved):
            nv = r['nv']
            ids = np.arange(1 << nv)
            bits = (ids[:, None] >> np.arange(nv-1, -1, -1)) & 1
            # Small irregular chunks exercise the final partial chunk, too.
            state = chunked_state(r, chunk=173)
            probabilities = abs(state)**2
            np.testing.assert_allclose(probabilities, formula_probs(r,bits), atol=1e-12, rtol=0)
            np.testing.assert_allclose(probabilities, qasm_probs(parse_gates(old['qasm']),nv), atol=1e-12, rtol=0)
            self.assertAlmostEqual(float(probabilities.sum()),1,places=12)
            from reanalysis import evaluate
            valid, scores = evaluate(bits,r)
            ideal = old['methods']['ideal_qaoa']
            self.assertAlmostEqual(float(probabilities[valid].sum()),ideal['feasibility'],places=12)
            quality = (scores[valid]-scores[valid].min())/(scores[valid].max()-scores[valid].min())
            self.assertAlmostEqual(float(probabilities[valid] @ quality / probabilities[valid].sum()),
                                   ideal['conditional_mean_normalized_score'], places=12)
            optimal = valid & np.isclose(scores,scores[valid].max(),rtol=0,atol=1e-10)
            self.assertAlmostEqual(float(probabilities[optimal].sum()),ideal['optimum_probability'],places=12)

    def test_wire_order_and_general_complex_pair_update(self):
        from reanalysis import single
        rng = np.random.default_rng(137)
        for nv in (2,5,9):
            for wire in range(nv):
                state = np.zeros(1 << nv, complex); state[0] = 1
                apply_single_inplace(state, ry(np.pi), wire, nv, chunk=13)
                self.assertEqual(int(np.argmax(abs(state))),1 << (nv-1-wire))
                state = rng.normal(size=1 << nv) + 1j*rng.normal(size=1 << nv)
                matrix = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
                reference = single(state,matrix,wire,nv)
                apply_single_inplace(state,matrix,wire,nv,chunk=13)
                np.testing.assert_allclose(state,reference,atol=1e-12,rtol=0)

    def test_18q_independent_gates_and_archived_pawsey_result(self):
        from larger_noiseless import load_circuit, metrics
        from analyze_recovered_hardware import feasible_assignments
        tasks=load_circuit(18)
        r=tasks[0][3]
        direct=qasm_probs(tasks[0][2],18)
        np.testing.assert_allclose(direct,abs(chunked_state(r))**2,atol=1e-12,rtol=0)
        bits,values=feasible_assignments(r)
        ids=bits @ (1 << np.arange(17,-1,-1))
        optimal=np.isclose(values,values.max(),atol=1e-10,rtol=0)
        independent=metrics(direct[ids],values,optimal)
        archived=json.loads((ROOT/'analysis/larger_noiseless/18q.json').read_text())
        for key,value in independent.items():
            self.assertAlmostEqual(value,archived['methods']['ideal_qaoa'][key],places=12)


if __name__ == '__main__':
    unittest.main()
