"""Generate supplemental results directly from completed revision experiments."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
from revision_experiments import hardware_instances, problem_key

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'overleaf'
DATA=ROOT/'analysis/revision'


def load(name):return json.loads((DATA/name).read_text())
def esc(s):return str(s).replace('_',r'\_').replace('&',r'\&')
def pct(v):return '--' if v is None else f'{100*v:.4f}'
def number(v):return '--' if v is None else f'{v:g}'
def table(caption,columns,header,rows):
    return ('\\begingroup\n\\small\n\\setlength{\\tabcolsep}{4pt}\n'
            '\\begin{longtable}{'+columns+'}\n\\caption{'+caption+'}\\\\\n'
            '\\toprule\n'+header+r'\\'+'\n\\midrule\n\\endfirsthead\n'
            '\\toprule\n'+header+r'\\'+'\n\\midrule\n\\endhead\n'
            + '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{longtable}\n\\endgroup\n')


def main():
    classical=load('classical.json');trained=load('panels.json');grover=load('grover.json');rescored=load('rescoring.json')
    hardware,originals=hardware_instances()
    design=json.loads((ROOT/'evidence/revision_design.json').read_text())
    olddesign=json.loads((ROOT/'evidence/sensitivity_design.json').read_text())['panels']
    oldtraining=json.loads((ROOT/'analysis/sensitivity_results.json').read_text())['rows']
    original_results=json.loads((ROOT/'analysis/verified_results.json').read_text())['rows']
    bykey={(r['problem_key'],r['lambda']):r for r in classical}
    labels={problem_key(r):f'H{i+1:02}' for i,r in enumerate(hardware)}
    panel_labels={r['name']:(f'C{i+1:02}' if i<35 else f'V{i-34}') for i,r in enumerate(design['panels'])}
    circuit_order=sorted({r['circuit_sha256'] for r in rescored},key=lambda k:(next(r['qubits'] for r in rescored if r['circuit_sha256']==k),k))
    circuit_labels={key:f'Q{i+1:02}' for i,key in enumerate(circuit_order)}
    out=[]
    out.append(r'''\clearpage
\section{Additional classical search and panel definitions}
We use simulated annealing (SA) with the same feasible dose-change, addition, deletion and compound-swap neighborhood as greedy local search. We draw the initial assignment uniformly from the feasible space by rejection sampling complete compound assignments. At each step we propose a uniformly chosen neighbor and accept an improving move, or accept a worse move with probability $\exp(\Delta f/T)$. We use 100 seeds, numbered 0 through 99, for each model and weight.

We set the initial temperature to the maximum absolute linear or pair coefficient, with a lower bound of $10^{-6}$. Over 2,000 proposals we cool geometrically to one thousandth of that temperature. We stop after 200 unique objective evaluations or 2,000 proposals, whichever comes first. Cached evaluations do not count again. The initial state's evaluation counts toward the cap. For small feasible sets the proposal limit prevents an endless attempt to obtain 200 distinct evaluations. We record the best score evaluated, and compare its trace with the enumerated optimum only after the run. Median first-hit evaluations condition on successful seeds; the success fraction reports failures separately.

Exhaustive enumeration is the exact solver: it evaluates each feasible assignment once. It supplies no search information to SA or greedy. Table~S8 reports all hardware models and the expanded panels at the archived weight. OR-Tools is not installed in the execution environment, so CP-SAT is not included.
''')
    rows=[]
    models=hardware+design['panels']
    for r in models:
        x=bykey[(problem_key(r),.6)]
        rows.append([panel_labels.get(r['name'],labels.get(problem_key(r))),r['nv'],x['feasible_states'],x['greedy']['objective_evaluations'],
                     'yes' if x['greedy']['found_optimum'] else 'no',f"{100*x['sa_success_fraction']:.0f}",number(x['sa_median_first_hit'])])
    out.append(table('Classical search at $\\lambda=0.6$. H01--H14 are hardware models, C01--C35 are all CAMA-1 panels, and V1--V4 are 647-V panels. Exact: enumeration evaluations, equal to $|\\mathcal F|$. Greedy: total unique evaluations, including partial construction states, and optimum recovery. SA: percentage of 100 seeds reaching the optimum and median evaluations to first hit among successful seeds.','lrrrrrr',r'Model & $q$ & Exact & Greedy eval. & Hit & SA (\%) & First hit',rows))
    out.append(r'Hardware model labels follow the original six instances first, then the additional distinct models. Their compound counts, dose counts and feasible-set sizes identify the encoding; Table~S9 additionally identifies every distinct circuit. Panel membership is given with Table~S11.'+'\n')
    out.append(r'''\clearpage
\section{Score-weight sensitivity}
We retain $\lambda=0.6$ as the archived model value and rescore each feasible assignment at $\lambda\in\{0.3,0.45,0.6,0.75,0.9\}$. All archived circuit coefficients, initialization angles, variational angles and hardware histograms remain fixed. Only the score and the set of globally optimal feasible assignments change. We sum each fixed distribution over that new optimum set, including ties. For the original four 12-qubit panels we instead rebuild the penalty cost and product prior at each weight and retrain all three seeds with 60 Adam steps and learning rate 0.1. Table~S9 distinguishes those retrained panels (P1--P4) from the fixed circuits (Q01--Q18).

Each entry below is a per-draw optimum probability, in percent. Hardware values are observed fractions, with repeated tasks separated by slashes in chronological order. A dash indicates an unavailable noiseless reference. Greedy reports whether it reaches the re-enumerated optimum; SA gives the percentage of successful seeds. For retrained panels the ideal values follow seed order 0, 1 and 2. The machine-readable results include exact optimum scores, all tied optimum bitstrings, individual hardware task identifiers and every SA seed.
''')
    rows=[]
    for key in circuit_order:
        group=sorted([r for r in rescored if r['circuit_sha256']==key],key=lambda r:r['lambda'])
        model=labels[group[0]['problem_key']]
        rows.append([r'\multicolumn{8}{l}{'+f"{circuit_labels[key]}: {model}, {group[0]['qubits']} qubits, {group[0]['feasible_states']:,} feasible assignments"+'}'])
        for r in group:
            c=bykey[(r['problem_key'],r['lambda'])]
            rows.append([f"{r['lambda']:g}",pct(r['initial_product']),pct(r['ideal_qaoa']),pct(r['uniform_feasible']),
                         '/'.join(f"{100*h['optimum_fraction']:.2f}" for h in r['hardware']),
                         'yes' if c['greedy']['found_optimum'] else 'no',f"{100*c['sa_success_fraction']:.0f}",number(c['sa_median_first_hit'])])
    for i,panel in enumerate(olddesign):
        rows.append([r'\multicolumn{8}{l}{'+f'P{i+1}: retrained 12-qubit panel, 162 feasible assignments'+'}'])
        for lam in (.3,.45,.6,.75,.9):
            rr=sorted([r for r in (oldtraining if lam==.6 else trained) if r['panel']==panel['name'] and (lam==.6 or r['lambda']==lam)],key=lambda r:r['seed'])
            assert len(rr)==3
            key=problem_key(dict(panel,lam=.6))
            c=bykey[(key,lam)]
            rows.append([f'{lam:g}',pct(rr[0]['methods']['warm_start_only']['optimum_probability']),
                         '/'.join(f"{100*r['methods']['ideal_qaoa']['optimum_probability']:.2f}" for r in rr),
                         pct(rr[0]['methods']['uniform_feasible']['optimum_probability']),'--','yes' if c['greedy']['found_optimum'] else 'no',
                         f"{100*c['sa_success_fraction']:.0f}",number(c['sa_median_first_hit'])])
    out.append(table('Weight sensitivity for every archived circuit and the original four retrained panels. Prior, Ideal, Feasible and HW are optimum probabilities (percent). Hit: greedy optimum recovery. SA: successful seeds (percent); Eval.: median first-hit evaluations among those seeds.','rrrrlrrr',r'$\lambda$ & Prior & Ideal & Feasible & HW & Hit & SA & Eval.',rows))
    out.append(r'Circuit identifiers map to the archived task inventory as follows. Repeated tasks share a circuit identifier.'+'\n\\begin{itemize}\n')
    for key in circuit_order:
        r=next(r for r in rescored if r['circuit_sha256']==key)
        out.append(r'\item '+circuit_labels[key]+': '+r'\newline '.join(r'\texttt{'+x['task_uuid']+'}' for x in r['hardware'])+'.\n')
    out.append('\\end{itemize}\n')
    out.append(r'''\clearpage
\section{Feasibility-preserving Grover mixer}
We construct $|\mathcal F\rangle$ directly as the uniform vector over enumerated feasible assignments. One layer applies $\exp(i\gamma f)$ followed by $I-(1-e^{-i\beta})|\mathcal F\rangle\langle\mathcal F|$. Thus the optimized energy is $-f$, with no penalty term. We train both angles from the same three random seeds, using 60 Adam steps with learning rate 0.1, and retain every final iterate. Analytic derivatives agree with centered finite differences, and a dense projector calculation independently checks the mixer. Every state remains normalized and entirely feasible.

The construction follows B{\"a}rtschi and Eidenbenz (2020), cited in the main paper. Direct feasible-vector preparation is a simulation device and does not establish a low-cost hardware preparation circuit. The comparison changes initialization and training as well as the mixer: the original penalty circuits retain their archived parameters, whereas both simulated panel methods are trained with the stated settings.
''')
    rows=[]
    for r in grover:
        if r['name'].startswith('panel_'):
            old=next(x for x in oldtraining if x['panel']==r['name'] and x['seed']==r['seed'])
            penalty=old['methods']['ideal_qaoa']['optimum_probability'];greedy=old['greedy_local']['objective_evaluations']
            label='P'+r['name'].split('_')[-1]
        else:
            old=next(x for x in original_results if x['qubits']==r['qubits'])
            penalty=old['methods']['ideal_qaoa']['optimum_probability']
            model=next(x for x in originals if x['name']==r['name'])
            greedy=bykey[(problem_key(model),.6)]['greedy']['objective_evaluations'];label=labels[problem_key(model)]
        rows.append([label,r['seed'],pct(r['optimum_probability']),pct(penalty),pct(r['uniform_feasible']),greedy])
    out.append(table('One-layer Grover-mixer comparison for the six reference instances and four original panels. All probabilities are percentages. Grover feasibility is exactly 100 percent. Penalty: warm-start penalty-QAOA optimum probability. Greedy reaches every optimum, with the listed evaluation count.','lrrrrr',r'Model & Seed & Grover & Penalty & Feasible & Greedy eval.',rows))
    out.append(r'''\clearpage
\section{All CAMA-1 panels and a second cell line}
We enumerate all 35 four-compound subsets of the seven-compound CAMA-1 model. For a second cell line, we inspect QA-passing DREAM pair measurements in alphabetical cell-line order, excluding CAMA-1. We choose the first line with a fully measured five-compound clique having distinct mechanism labels and finite fitted Hill parameters. Within a cell line, compounds are searched lexicographically. This gives 647-V and the compounds AKT, ATM, ATR\_4, CHEK1 and Cisplatin. We sample four of its five four-compound subsets without replacement using seed 20260905, before optimization.

We use the original coefficient-construction rule: median Hill parameters from within-clique observations, mean recorded pair synergy divided by 100 and multiplied by dose fractions, and the same class-weighted dose proxy. Mechanism classes absent from the archived weight dictionary use its existing default weight of 1.0. Every panel uses dose fractions $(0.1,0.32,1)$, 12 qubits and 162 feasible assignments. Table~S11 reports all three training seeds. Figure~S3 shows their optimum probabilities without selecting the best seed.
''')
    rows=[]
    for p in design['panels']:
        rr=sorted([r for r in trained if r['panel']==p['name'] and r['lambda']==.6],key=lambda r:r['seed'])
        c=bykey[(problem_key(p),.6)]
        rows.append([panel_labels[p['name']],*[pct(r['methods']['ideal_qaoa']['optimum_probability']) for r in rr],
                     pct(rr[0]['methods']['warm_start_only']['optimum_probability']),pct(rr[0]['methods']['uniform_feasible']['optimum_probability']),
                     'yes' if c['greedy']['found_optimum'] else 'no'])
    out.append(table('Expanded panel results at $\\lambda=0.6$. Q0--Q2: ideal penalty-QAOA optimum probabilities for training seeds 0--2; Prior and Feasible: corresponding sampling controls. All probabilities are percentages. Hit: greedy optimum recovery. Every SA seed reaches each panel optimum (Table~S8).','lrrrrrr',r'Panel & Q0 & Q1 & Q2 & Prior & Feasible & Hit',rows))
    out.append('\\begin{itemize}\n')
    for p in design['panels']:
        out.append(r'\item '+panel_labels[p['name']]+': '+', '.join(esc(x) for x in p['drugs'])+'.\n')
    out.append('\\end{itemize}\n')
    out.append(r'''\begin{figure}[h!]
\centering
\includegraphics[width=\textwidth]{figures/figureS3.pdf}
\caption{Ideal penalty-QAOA optimum probability for all expanded panels and all three training seeds. The dashed line is uniform feasible sampling, at $100/162$ percent. Panel labels match Table~S11. Left: all CAMA-1 panels. Right: the four 647-V panels selected by measurement coverage.}
\end{figure}
''')
    out.append(r'''\clearpage
\section{Reproducing the added experiments}
The added code and results are included in the reproducibility package. Run \texttt{prepare\_revision.py} to reproduce the frozen panel design from the supplied source tables. Run \texttt{revision\_experiments.py} with \texttt{--mode classical}, \texttt{--mode panels}, \texttt{--mode grover} and \texttt{--mode rescore} for the four computations. The first three modes produce independent result files. Rescoring uses the saved large-circuit feasible probabilities when available. Run \texttt{python -m unittest test\_revision -v} for numerical and search checks, then \texttt{build\_revision\_outputs.py} to regenerate the supplemental tables and panel figure. Script SHA-256 hashes are listed below; source-table hashes and panel membership are preserved in \texttt{evidence/revision\_design.json}.
''')
    names=['prepare_revision.py','revision_experiments.py','revision_noiseless.py','test_revision.py','test_completed_revision.py','test_noiseless_revision.py','build_revision_outputs.py','rewrite_prose.py','integrate_revision.py','integrate_noiseless.py','revision.sbatch','noiseless_revision.sbatch']
    out.append('\\begin{itemize}\n')
    for name in names:
        digest=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        out.append(r'\item \texttt{'+esc(name)+r'}\newline {\footnotesize\texttt{'+digest[:32]+r'\allowbreak '+digest[32:]+'}}.\n')
    out.append('\\end{itemize}\n')
    (OUT/'results.tex').write_text('\n'.join(out),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),gridspec_kw={'width_ratios':[4,1]},layout='constrained')
    for ax,cell in zip(axes,['CAMA-1','647-V']):
        pp=[p for p in design['panels'] if p['cell_line']==cell]
        for seed,color in enumerate(['#0072B2','#D55E00','#009E73']):
            y=[100*next(r for r in trained if r['panel']==p['name'] and r['seed']==seed and r['lambda']==.6)['methods']['ideal_qaoa']['optimum_probability'] for p in pp]
            ax.scatter(np.arange(len(pp))+(seed-1)*.14,y,s=12,color=color,label=f'Seed {seed}')
        ax.axhline(100/162,color='#555555',ls='--',lw=1)
        ticks=list(range(0,len(pp),3)) if len(pp)>4 else list(range(len(pp)))
        ax.set_xticks(ticks,[panel_labels[pp[i]['name']] for i in ticks],rotation=90)
        ax.set_title(cell);ax.set_ylim(bottom=0);ax.set_xlabel('Panel');ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Optimum probability (%)');axes[0].legend(fontsize=7,frameon=False)
    previews = ROOT/'qa/figure_previews'
    previews.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'figures/figureS3.pdf');fig.savefig(previews/'figureS3.png',dpi=180);plt.close(fig)
    print('Generated four supplemental tables and Figure S3.')


if __name__=='__main__':main()
