"""Freeze a separate, uploadable calculation package and evidence hash inventory."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
out=ROOT/'analysis/larger_noiseless'
out.mkdir(parents=True,exist_ok=True)
sources=['larger_noiseless.py','test_larger_noiseless.py','larger_noiseless.sbatch',
         'reanalysis.py','analyze_recovered_hardware.py','quality_analysis.py']
files=[ROOT/p for p in sources]
files += [p for p in (ROOT/'evidence/recovered_hardware').rglob('*') if p.is_file()]
files += [ROOT/'evidence/frozen_ladder.json',ROOT/'analysis/verified_results.json']
with zipfile.ZipFile(out/'submitted_package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(ROOT))
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'evidence').rglob('*') if p.is_file()}
(out/'original_evidence_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(out/'submitted_package.zip')
