"""Checks using an independent circuit library and original scorer definitions."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import pennylane as qml
from reanalysis import analyze, extract_ladder, qasm_probs, parse_gates, wilson

HERE = Path(__file__).resolve().parent


class ReanalysisTests(unittest.TestCase):
    def test_bell_and_rotation_wire_order(self):
        gates = [('ry', np.pi/2,0),('cnot',0,1)]
        np.testing.assert_allclose(qasm_probs(gates,2),[.5,0,0,.5],atol=1e-14)
        np.testing.assert_allclose(qasm_probs([('ry',np.pi,0)],2),[0,0,1,0],atol=1e-14)

    def test_wilson_boundary(self):
        self.assertAlmostEqual(wilson(0,200)[0],0)
        self.assertAlmostEqual(wilson(0,200)[1],0.0188453263772666)
        self.assertAlmostEqual(wilson(200,200)[1],1)

    def test_archived_tasks_against_pennylane(self):
        ladder = json.loads((HERE/'evidence/frozen_ladder.json').read_text())
        for rung in ladder:
            with self.subTest(qubits=rung['nv']):
                path = next((HERE/'evidence/tasks').glob(f"*_{rung['nv']:02d}q_*.json"))
                raw = json.loads(path.read_text())
                gates = parse_gates(raw['additionalMetadata']['action']['source'])
                device = qml.device('default.qubit',wires=rung['nv'])
                @qml.qnode(device)
                def circuit():
                    for name,a,b in gates:
                        if name == 'cnot': qml.CNOT(wires=[a,b])
                        elif name == 'ry': qml.RY(a,wires=b)
                        else: qml.RZ(a,wires=b)
                    return qml.probs(wires=range(rung['nv']))
                np.testing.assert_allclose(qasm_probs(gates,rung['nv']),circuit(),atol=1e-12,rtol=0)
                result = analyze(raw,rung)
                self.assertEqual(result['cnot_count'],rung['n_twoqubit_gates'])

    def test_original_decoder_and_score(self):
        # Execute only the three pure function definitions, never the job runner.
        source = Path(r'C:\Users\Shrizon\Desktop\Pawsey\projects\drug-search\notebooks\braket_noise_ladder.py')
        if not source.exists(): self.skipTest('Original runner unavailable; bundled tests still apply')
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ['decode','feasible','score']]
        namespace = {}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        from reanalysis import evaluate
        for rung in extract_ladder(source):
            nv = rung['nv']
            bits = ((np.arange(1<<nv)[:,None] >> np.arange(nv-1,-1,-1)) & 1)
            mask, values = evaluate(bits,rung)
            for i,b in enumerate(bits):
                assignment = namespace['decode'](''.join(map(str,b)),rung)
                feasible = assignment is not None and namespace['feasible'](assignment,rung)
                self.assertEqual(bool(mask[i]),feasible)
                if feasible:
                    e,t = namespace['score'](assignment,rung)
                    self.assertAlmostEqual(values[i],e-rung['lam']*t,places=12)


if __name__ == '__main__':
    unittest.main()
