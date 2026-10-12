"""Independent safeguards for expanded data, grouping and sparse enumeration."""
import itertools
import json
import unittest
from pathlib import Path
import numpy as np
from analyze_recovered_hardware import feasible_assignments, expected_gates, gate_match
from reanalysis import evaluate, parse_gates

ROOT=Path(__file__).resolve().parent


class RecoveredHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'analysis/recovered_hardware.json').read_text())
        cls.frozen=json.loads((ROOT/'evidence/recovered_hardware/frozen_instances.json').read_text())

    def test_sparse_enumeration_against_cartesian_assignments(self):
        # Independent assignment-space enumeration through the largest 35q model.
        for f in self.frozen:
            r=f['instance'];bits,values=feasible_assignments(r)
            expected=set()
            for a in itertools.product(range(-1,r['D']),repeat=r['n']):
                chosen=[i for i,d in enumerate(a) if d>=0]
                if not r['min_size']<=len(chosen)<=r['k']:continue
                if len({r['moa'][i] for i in chosen})<r['m_classes']:continue
                expected.add(sum(1<<(r['nv']-1-i*r['D']-a[i]) for i in chosen))
            actual={int(''.join(map(str,row)),2) for row in bits}
            self.assertEqual(actual,expected)

    def test_original_six_reconcile_and_duplicates_not_repeats(self):
        old=json.loads((ROOT/'analysis/verified_results.json').read_text())['rows']
        new={r['task_uuid']:r for r in self.data['rows']}
        self.assertEqual(len(new),25)
        self.assertEqual(sum(r['shots'] for r in new.values()),5300)
        for r in old:
            found=new[r['task_id'].split('/')[-1]]
            self.assertEqual(found['histogram'],r['histogram'])
            self.assertEqual(found['valid_count'],r['methods']['hardware']['valid_count'])
            self.assertEqual(found['optimum_count'],r['methods']['hardware']['optimum_count'])
        self.assertEqual(sorted(g['qubits'] for g in self.data['identical_circuit_repeats']),[6,18,20,21,25])
        for g in self.data['identical_circuit_repeats']:
            sources=[]
            for uid in g['task_uuids']:
                raw=json.loads((ROOT/f'evidence/recovered_hardware/tasks/{uid}.json').read_text())
                sources.append(raw['additionalMetadata']['action']['source'])
            self.assertEqual(len(set(sources)),1)

    def test_changed_parameters_rejected_as_identical_circuit(self):
        a=next(f['instance'] for f in self.frozen if f['runner']=='braket_noise_ladder.py' and f['instance']['nv']==6)
        b=next(f['instance'] for f in self.frozen if f['runner']=='braket_noise_ladder_full.py' and f['instance']['nv']==6)
        self.assertFalse(gate_match(expected_gates(a),expected_gates(b)))
        changed=expected_gates(a).copy();changed[0]=('ry',changed[0][1]+1e-5,0)
        self.assertFalse(gate_match(expected_gates(a),changed))


if __name__=='__main__':unittest.main()
