"""Validate separate Pawsey results against original evidence and generate tables."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
folder=ROOT/'analysis/larger_noiseless'
original=json.loads((ROOT/'analysis/recovered_hardware.json').read_text())['rows']
hashes=json.loads((folder/'original_evidence_hashes.json').read_text())
for name,digest in hashes.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
rows=[]
flat=[]
control_tex=[]
hardware_tex=[]
for width in (18,20,21,25):
    with zipfile.ZipFile(folder/'completed_run.zip') as archive:
        assert (folder/f'{width}q.json').read_bytes()==archive.read(f'analysis/larger_noiseless/{width}q.json')
    row=json.loads((folder/f'{width}q.json').read_text())
    assert row['qubits']==width and abs(row['norm']-1)<1e-10
    assert row['execution']['slurm_job_id']
    for name,digest in row['execution']['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    refs=[r for r in original if r['qubits']==width]
    assert {r['circuit_sha256'] for r in refs}=={row['circuit_sha256']}
    assert {r['task_uuid'] for r in refs}=={r['task_uuid'] for r in row['hardware']}
    for hw in sorted(row['hardware'],key=lambda r:r['created_utc']):
        ref=next(r for r in refs if r['task_uuid']==hw['task_uuid'])
        assert hw['qasm_sha256']==ref['qasm_sha256']
        assert hw['source_sha256']==hashlib.sha256((ROOT/f"evidence/recovered_hardware/tasks/{hw['task_uuid']}.json").read_bytes()).hexdigest()
        np.testing.assert_allclose([hw['feasibility'],hw['optimum_probability'],hw['conditional_mean_normalized_score']],
                                   [ref['feasibility'],ref['optimum_count']/ref['shots'],ref['mean_normalized_quality']],rtol=0,atol=1e-12)
        hardware_tex.append(f"{width} & {hw['created_utc'][5:16].replace('T',' ')} & "
            + ' & '.join(f"{100*hw[k]:.1f}" for k in ('feasibility','optimum_probability','conditional_mean_normalized_score'))+r' \\')
        flat.append({'qubits':width,'method':hw['task_uuid'],**{k:hw[k] for k in row['methods']['ideal_qaoa']}})
    prior=row['methods']['initial_product']
    np.testing.assert_allclose([prior['feasibility'],prior['optimum_probability'],row['methods']['uniform_feasible']['optimum_probability']],
                               [refs[0]['initial_product_feasibility'],refs[0]['initial_product_optimum_probability'],refs[0]['uniform_feasible_optimum_probability']],rtol=0,atol=1e-12)
    for method,label in [('ideal_qaoa','Ideal QAOA'),('initial_product','Initial product'),('uniform_feasible','Uniform feasible')]:
        m=row['methods'][method]
        control_tex.append(f"{width} & {row['feasible_states']} & {label} & "
            + f"{100*m['feasibility']:.2f} & {100*m['optimum_probability']:.4f} & {100*m['conditional_mean_normalized_score']:.1f}"+r' \\')
        flat.append({'qubits':width,'method':method,**m})
    rows.append(row)
for width in (30,35):
    result_path=ROOT/f'analysis/revision/noiseless_{width}.json'
    row=json.loads(result_path.read_text())
    assert row['qubits']==width and abs(row['norm']-1)<1e-6
    assert row['execution']['slurm_job_id']
    refs=[r for r in original if r['qubits']==width]
    assert {r['circuit_sha256'] for r in refs}=={row['circuit_sha256']}
    assert len(row['feasible_ids'])==row['feasible_states']==refs[0]['feasible_states']
    assert len(set(row['feasible_ids']))==row['feasible_states']
    mass=np.asarray(row['feasible_probabilities'])
    assert (mass>=0).all()
    assert abs(mass.sum()-row['methods']['ideal_qaoa']['feasibility'])<1e-12
    lookup=dict(zip(row['feasible_ids'],mass))
    optimum=sum(lookup[int(s,2)] for s in refs[0]['optimal_bitstrings'])
    assert abs(optimum-row['methods']['ideal_qaoa']['optimum_probability'])<1e-12
    np.testing.assert_allclose([row['methods']['initial_product']['feasibility'],row['methods']['initial_product']['optimum_probability']],
                               [refs[0]['initial_product_feasibility'],refs[0]['initial_product_optimum_probability']],rtol=0,atol=1e-12)
    row['hardware']=[]
    for hw in refs:
        entry={'task_uuid':hw['task_uuid'],'created_utc':hw['created_utc'],'shots':hw['shots'],
               'feasibility':hw['feasibility'],'optimum_probability':hw['optimum_count']/hw['shots'],
               'conditional_mean_normalized_score':hw['mean_normalized_quality']}
        row['hardware'].append(entry)
        hardware_tex.append(f"{width} & {hw['created_utc'][5:16].replace('T',' ')} & "
                            + ' & '.join(f"{100*entry[k]:.1f}" for k in ('feasibility','optimum_probability','conditional_mean_normalized_score'))+r' \\')
        flat.append({'qubits':width,'method':hw['task_uuid'],'feasibility':entry['feasibility'],
                     'optimum_probability':entry['optimum_probability'],'conditional_mean_score':None,
                     'conditional_mean_normalized_score':entry['conditional_mean_normalized_score']})
    for method,label in [('ideal_qaoa','Ideal QAOA'),('initial_product','Initial product'),('uniform_feasible','Uniform feasible')]:
        m=row['methods'][method]
        control_tex.append(f"{width} & {row['feasible_states']} & {label} & "
            + f"{100*m['feasibility']:.2f} & {100*m['optimum_probability']:.4f} & {100*m['conditional_mean_normalized_score']:.1f}"+r' \\')
        flat.append({'qubits':width,'method':method,**m})
    rows.append(row)
for name,macro,lines in [('supp_larger_controls.tex','LargerControlRows',control_tex),('supp_larger_hardware.tex','LargerHardwareRows',hardware_tex)]:
    (ROOT/'overleaf'/name).write_text('\\newcommand{\\'+macro+'}{%\n'+'\n'.join(lines)+'\n}\n')
with (folder/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
report={'rows':rows,'original_evidence_files_unchanged':len(hashes),
        'not_simulated_qubits':[],'reason':'The revision adds validated complex64 results at 30 and 35 qubits.'}
(folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['qubits'],row['feasible_states'],row['methods'],row['execution'].get('elapsed_seconds',row['execution'].get('seconds')))
