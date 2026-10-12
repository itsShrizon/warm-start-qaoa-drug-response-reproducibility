"""Check completed results against independently scored feasible assignments."""
import json
import unittest
from pathlib import Path
import numpy as np
from revision_experiments import hardware_instances, assignments, bits_for, problem_key
from quality_analysis import assignment_score
from reanalysis import evaluate

ROOT=Path(__file__).resolve().parent


class CompletedChecks(unittest.TestCase):
    def test_weighted_enumeration(self):
        hardware,_=hardware_instances()
        result=json.loads((ROOT/'analysis/revision/rescoring.json').read_text())
        for r in hardware:
            aa=assignments(r);bits=bits_for(aa,r)
            ids=bits@(1<<np.arange(r['nv']-1,-1,-1,dtype=np.int64))
            for lam in (.3,.45,.6,.75,.9):
                model=dict(r,lam=lam)
                direct=np.array([assignment_score(a,model) for a in aa])
                valid,vector=evaluate(bits,model)
                self.assertTrue(valid.all())
                np.testing.assert_allclose(direct,vector,rtol=0,atol=1e-12)
                for row in [x for x in result if x['problem_key']==problem_key(r) and x['lambda']==lam]:
                    self.assertAlmostEqual(row['optimum'],direct.max(),places=11)
                    optimum=np.isclose(direct,direct.max(),rtol=0,atol=1e-10)
                    self.assertEqual(set(row['optimal_bitstrings']),{format(int(i),f"0{r['nv']}b") for i in ids[optimum]})

    def test_panel_reproduction(self):
        old=json.loads((ROOT/'analysis/sensitivity_results.json').read_text())['rows']
        new=json.loads((ROOT/'analysis/revision/panels.json').read_text())
        for r in old:
            match=next(x for x in new if x['drugs']==r['drugs'] and x['seed']==r['seed'] and x['lambda']==.6)
            for method in r['methods']:
                self.assertAlmostEqual(match['methods'][method]['optimum_probability'],r['methods'][method]['optimum_probability'],places=9)

    def test_result_completeness(self):
        classical=json.loads((ROOT/'analysis/revision/classical.json').read_text())
        panels=json.loads((ROOT/'analysis/revision/panels.json').read_text())
        grover=json.loads((ROOT/'analysis/revision/grover.json').read_text())
        self.assertEqual(len(classical),125)
        self.assertEqual(len(panels),165)
        self.assertEqual(len(grover),30)
        for row in classical:
            self.assertEqual({x['seed'] for x in row['sa']},set(range(100)))
            self.assertTrue(all(x['evaluations']<=200 for x in row['sa']))
            hits=[x['first_hit_evaluations'] for x in row['sa'] if x['found_optimum']]
            self.assertEqual(row['sa_success_fraction'],len(hits)/100)
            self.assertEqual(row['sa_median_first_hit'],float(np.median(hits)) if hits else None)


if __name__=='__main__':unittest.main()
