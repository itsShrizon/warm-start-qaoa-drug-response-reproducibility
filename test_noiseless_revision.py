"""Independent probability check before the large archived-circuit calculations."""
import json
import unittest
import numpy as np
from revision_noiseless import statevector, ROOT
from reanalysis import formula_probs


class NoiselessChecks(unittest.TestCase):
    def test_gray_against_original_formula(self):
        records=json.loads((ROOT/'evidence/frozen_ladder.json').read_text())
        for r in records:
            bits=((np.arange(1<<r['nv'])[:,None]>>np.arange(r['nv']-1,-1,-1))&1)
            reference=formula_probs(r,bits)
            for dtype in (np.complex64,np.complex128):
                state=statevector(r,dtype)
                tolerance=1e-7 if dtype==np.complex64 else 1e-10
                np.testing.assert_allclose(abs(state.astype(np.complex128))**2,reference,rtol=tolerance*10,atol=tolerance)


if __name__=='__main__':unittest.main()
