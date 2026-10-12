"""Publication figures and TeX table generated only from verified results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'overleaf'
OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
PREVIEWS = HERE/'qa/figure_previews'
PREVIEWS.mkdir(parents=True, exist_ok=True)
rows = json.loads((HERE/'analysis/verified_results.json').read_text())['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
styles = {
 'hardware':('#0072B2','o','IonQ hardware'),
 'ideal_qaoa':('#009E73','s','Ideal QAOA'),
 'warm_start_only':('#D55E00','^','Warm-start only'),
 'uniform_bits':('#777777','x','Uniform bitstrings'),
 'uniform_feasible':('#7B3294','D','Uniform feasible'),
}
x = np.arange(len(rows))
labels = [str(r['qubits']) for r in rows]

def draw(ax,method,key,offset=0,ci=None):
    color,marker,label = styles[method]
    y = np.array([r['methods'][method][key] for r in rows])*100
    if ci:
        ranges = np.array([r['methods'][method][ci] for r in rows])*100
        ax.errorbar(x+offset,y,yerr=[y-ranges[:,0],ranges[:,1]-y],color=color,
           fmt=marker,capsize=2.5,markersize=5,label=label,lw=1.1)
    else:
        ax.plot(x+offset,y,marker=marker,color=color,lw=1.1,markersize=4,label=label)
    ax.set_xticks(x,labels)
    ax.set_xlabel('Qubits')
    ax.grid(axis='y',alpha=.2)

fig,axes = plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
ax=axes[0]
for m in ['uniform_feasible','uniform_bits','warm_start_only','ideal_qaoa']:
    draw(ax,m,'feasibility')
draw(ax,'hardware','feasibility',ci='valid_ci95')
ax.set_ylabel('Feasible outputs (%)')
ax.set_ylim(-3,106)
ax.set_title('(A) Original six circuits',loc='left',fontweight='bold',fontsize=10)
larger=json.loads((HERE/'analysis/larger_noiseless/comparison.json').read_text())['rows']
expanded=json.loads((HERE/'analysis/recovered_hardware.json').read_text())['rows']
widths=[18,20,21,25,30,35]
big=axes[1]
refs=[next(r for r in expanded if r['qubits']==q) for q in widths]
for method,key in [('warm_start_only','initial_product_feasibility'),('uniform_bits','uniform_bits_feasibility')]:
    color,marker,label=styles[method]
    big.plot(np.arange(6),[100*r[key] for r in refs],color=color,marker=marker,lw=1.1,markersize=4)
big.plot(np.arange(6),np.full(6,100),color=styles['uniform_feasible'][0],marker='D',lw=1.1,markersize=4)
big.plot(np.arange(len(larger)),[100*r['methods']['ideal_qaoa']['feasibility'] for r in larger],
         color=styles['ideal_qaoa'][0],marker='s',lw=1.1,markersize=4)
for i,q in enumerate(widths):
    tasks=[r for r in expanded if r['qubits']==q]
    for shift,r in zip(np.linspace(-.16,.16,len(tasks)) if len(tasks)>1 else [0],tasks):
        y=r['feasibility']*100;low,high=np.array(r['valid_ci95'])*100
        big.errorbar(i+shift,y,yerr=[[y-low],[high-y]],fmt='o',color=styles['hardware'][0],
                     capsize=2,markersize=4,lw=1)
big.set_xticks(np.arange(6),[f"{q}\n{r['feasible_states']:,}" for q,r in zip(widths,refs)])
big.set_xlabel('Qubits / feasible assignments')
big.set_ylim(-3,106)
big.grid(axis='y',alpha=.2)
big.set_title('(B) Larger circuits and executions',loc='left',fontweight='bold',fontsize=10)
handles,legend_labels=ax.get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
for ext in ['pdf','png','tiff']:
    destination = OUT/'figures' if ext == 'pdf' else PREVIEWS
    fig.savefig(destination/f'figure1.{ext}',dpi=350,bbox_inches='tight')
plt.close(fig)

fig,axes = plt.subplots(1,2,figsize=(7,3.4),layout='constrained')
for offset,m in zip([-.16,-.08,.08,.16],['uniform_feasible','warm_start_only','ideal_qaoa','hardware']):
    draw(axes[0],m,'optimum_probability',offset,ci='optimum_ci95' if m=='hardware' else None)
axes[0].set_ylabel('Exact-optimum probability (%)')
axes[0].set_ylim(-.15,6.8)
axes[0].set_title('(A) One sample',loc='left',fontweight='bold',fontsize=10)
for m in ['uniform_feasible','warm_start_only','ideal_qaoa']:
    draw(axes[1],m,'probability_find_optimum_200')
axes[1].set_ylabel('Chance of at least one optimum (%)')
axes[1].set_ylim(-3,105)
axes[1].set_title('(B) 200 independent samples',loc='left',fontweight='bold',fontsize=10)
handles,legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='outside upper center',ncol=4,frameon=False,fontsize=8)
for ext in ['pdf','png','tiff']:
    destination = OUT/'figures' if ext == 'pdf' else PREVIEWS
    fig.savefig(destination/f'figure2.{ext}',dpi=350,bbox_inches='tight')
plt.close(fig)

table = []
flat = []
annealing=json.loads((HERE/'analysis/revision/classical.json').read_text())
for r in rows:
    hw = r['methods']['hardware']
    low,high = hw['valid_ci95']
    table.append(f"{r['qubits']} & {r['drugs']} & {r['doses']} & {r['cnot_count']} & {r['feasible_states']} & {hw['valid_count']}/200 & {100*hw['feasibility']:.1f} ({100*low:.1f}--{100*high:.1f}) & {hw['optimum_count']}/200 \\\\")
    sa=next(a for a in annealing if a['name']==r['name'] and a['lambda']==.6)
    table[-1]=table[-1][:-2]+f" & {100*sa['sa_success_fraction']:.0f}"+r' \\'
    flat.append({'qubits':r['qubits'],'feasible_states':r['feasible_states'],
      'valid_count':hw['valid_count'],'shots':r['shots'],'valid_ci_low':low,'valid_ci_high':high,
      'optimum_count':hw['optimum_count'],
      **{f'{m}_{metric}':v[metric] for m,v in r['methods'].items() for metric in ['feasibility','optimum_probability']}})
(OUT/'table1_rows.tex').write_text('\\newcommand{\\TableOneRows}{%\n'+'\n'.join(table)+'\n}\n',encoding='utf-8')
for filename, metric, methods in [
    ('supp_feas_rows.tex','feasibility',['uniform_bits','warm_start_only','ideal_qaoa','hardware','uniform_feasible']),
    ('supp_hit_rows.tex','probability_find_optimum_200',['uniform_bits','warm_start_only','ideal_qaoa','uniform_feasible'])]:
    lines = [str(r['qubits'])+' & '+' & '.join(f"{100*r['methods'][m][metric]:.2f}" for m in methods)+r' \\' for r in rows]
    macro = 'SupplementFeasRows' if metric == 'feasibility' else 'SupplementHitRows'
    (OUT/filename).write_text('\\newcommand{\\'+macro+'}{%\n'+'\n'.join(lines)+'\n}\n',encoding='utf-8')
with (HERE/'analysis/summary.csv').open('w',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f,fieldnames=list(flat[0]))
    writer.writeheader();writer.writerows(flat)
print('Generated two figures and table rows from verified_results.json')
