"""Freeze panel membership using measurement coverage, before optimization."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import itertools
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    source=ROOT/'evidence/revision_source'
    source.mkdir(parents=True,exist_ok=True)
    dataset=source
    for name in ('OI_combinations_synergy_scores_final.txt','Drug_info_release.csv'):
        if not (source/name).is_file():
            raise FileNotFoundError(f'Obtain {name} under the source data access conditions; see DATA_ACCESS.md')
    spec=importlib.util.spec_from_file_location('source_builder',ROOT/'evidence/model_source/_azdream_build_instance.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    graphs={}
    with (source/'OI_combinations_synergy_scores_final.txt').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['QA'].strip()!='1' or builder.f(row['SYNERGY_SCORE']) is None:continue
            a,b=row['COMPOUND_A'].strip(),row['COMPOUND_B'].strip()
            if a==b:continue
            graph=graphs.setdefault(row['CELL_LINE'].strip(),{})
            graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    def first_clique(graph,chosen,candidates):
        if len(chosen)==5:return chosen
        for i,candidate in enumerate(candidates):
            if builder.base_class(candidate) in {builder.base_class(x) for x in chosen}:continue
            # Require measured finite Hill parameters for every selected compound below.
            answer=first_clique(graph,chosen+[candidate],[x for x in candidates[i+1:] if x in graph[candidate]])
            if answer:return answer
        return None
    selected=None
    for cell in sorted(graphs):
        if cell=='CAMA-1':continue
        clique=first_clique(graphs[cell],[],sorted(graphs[cell]))
        if not clique:continue
        builder.CELL=cell;builder.CLIQUE=clique
        med,synergy=builder.load_oi_clique(str(source))
        if all(all(v is not None and np.isfinite(v) for v in p.values()) and p['IC50']>0 for p in med.values()):
            selected=(cell,clique,med,synergy);break
    assert selected
    cell,clique,med,synergy=selected
    panels=[]
    old=ROOT/'evidence/model_source'
    drugs=(old/'CAMA1c7_drugs.txt').read_text().splitlines()
    meta=json.loads((old/'CAMA1c7_D3_meta.json').read_text())
    h=np.loadtxt(old/'CAMA1c7_D3_h.csv',delimiter=',')
    tox=np.loadtxt(old/'CAMA1c7_D3_tox.csv',delimiter=',')
    jj=np.loadtxt(old/'CAMA1c7_D3_Jij.csv',delimiter=',')
    def add_panel(name,cell,choice,names,h,tox,jj,moa):
        wires=[i*3+d for i in choice for d in range(3)]
        j=jj[np.ix_(wires,wires)]
        panels.append({'name':name,'cell_line':cell,'source_indices':list(choice),'drugs':[names[i] for i in choice],
                       'n':4,'D':3,'nv':12,'min_size':2,'k':3,'m_classes':2,'lam':.6,
                       'moa':[moa[i] for i in choice],'h':h[list(choice)].tolist(),'tox':tox[list(choice)].tolist(),
                       'Jscore':[[a,b,float(j[a,b])] for a in range(12) for b in range(a+1,12) if j[a,b]!=0]})
    for i,choice in enumerate(itertools.combinations(range(7),4)):
        add_panel(f'CAMA1_all_{i+1:02}', 'CAMA-1',choice,drugs,h,tox,jj,[meta['moa_class'][d] for d in drugs])
    levels=builder.dose_levels(3)
    h=np.array([[builder.hill_inhibition(d,med[name]['IC50'],med[name]['H'],med[name]['Einf']) for d in levels] for name in clique])
    tox=np.array([[builder.TOX_WEIGHT.get(builder.base_class(name),builder.DEFAULT_TOX_WEIGHT)*d for d in levels] for name in clique])
    jj=np.zeros((15,15))
    for i,j in itertools.combinations(range(5),2):
        for d,e in itertools.product(range(3),repeat=2):
            jj[i*3+d,j*3+e]=jj[j*3+e,i*3+d]=synergy[frozenset((clique[i],clique[j]))]/100*levels[d]*levels[e]
    choices=list(itertools.combinations(range(5),4))
    selection=np.random.default_rng(20260905).choice(len(choices),4,replace=False)
    for i,k in enumerate(selection):
        add_panel(f'second_{i+1}',cell,choices[k],clique,h,tox,jj,[builder.base_class(d) for d in clique])
    result={'selection_rule':'First alphabetically ordered non-CAMA-1 cell with a lexicographically first fully measured five-compound clique, distinct mechanism labels and finite fitted parameters; four of its five four-compound subsets sampled without replacement.',
            'panel_selection_seed':20260905,'training_seeds':[0,1,2],'steps':60,'learning_rate':.1,
            'second_cell_line':cell,'second_clique':clique,'panels':panels,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}}
    out=ROOT/'evidence/revision_design.json'
    if out.exists():assert json.loads(out.read_text())==result
    else:out.write_text(json.dumps(result,indent=2)+'\n')
    print(cell,clique,len(panels))


if __name__=='__main__':main()
