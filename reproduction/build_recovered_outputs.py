"""Tables, repeat-execution plot and readable inventory from verified task analysis."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    data=json.loads((ROOT/'analysis/recovered_hardware.json').read_text())
    rows=data['rows'];out=ROOT/'overleaf'
    extra=[r for r in rows if not r['original_six']]
    lines=[]
    for r in extra:
        lo,hi=r['valid_ci95'];date=r['created_utc'][5:16].replace('T',' ')
        lines.append(f"{r['qubits']} & {r['drugs']} & {r['doses']} & {date} & {r['shots']} & {100*r['feasibility']:.1f} ({100*lo:.1f}--{100*hi:.1f}) & {r['optimum_count']}"+r'\\')
    (out/'recovered_rows.tex').write_text('\\newcommand{\\RecoveredRows}{%\n'+'\n'.join(lines)+'\n}\n')
    lines=[]
    for r in rows:
        date=r['created_utc'][5:16].replace('T',' ')
        lines.append(f"{r['qubits']} & {r['doses']} & {date} & {r['valid_count']} & {100*r['mean_normalized_quality']:.1f} & {100*r['best_normalized_gap']:.1f} & {r['greedy']['objective_evaluations']}"+r'\\')
    (out/'supp_recovered_quality.tex').write_text('\\newcommand{\\RecoveredQualityRows}{%\n'+'\n'.join(lines)+'\n}\n')
    lines=[r'\begin{itemize}']
    for r in rows:
        date=r['created_utc'][5:16].replace('T',' ')
        lines.append(f"\\item {r['qubits']}q, $D={r['doses']}$, {date}: \\texttt{{{r['task_uuid']}}}.")
    lines.append(r'\end{itemize}')
    (out/'supp_task_inventory.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(9.3,5.2),layout='constrained')
    lookup={r['task_uuid']:r for r in rows}
    for ax,g in zip(axes.flat,data['identical_circuit_repeats']):
        tasks=[lookup[uid] for uid in g['task_uuids']];x=np.arange(len(tasks))
        y=np.array([r['feasibility'] for r in tasks])*100
        ci=np.array([r['valid_ci95'] for r in tasks])*100
        ax.errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',capsize=4,color='#24688c')
        ax.axhline(tasks[0]['initial_product_feasibility']*100,color='#a35d22',linestyle='--',label='Initial product (analytic)')
        ax.set_xticks(x,[r['created_utc'][5:16].replace('T','\n') for r in tasks],fontsize=8)
        for xx,yy,r in zip(x,y,tasks):ax.annotate(f"{r['valid_count']}/{r['shots']}",(xx,yy),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(f"{g['qubits']} qubits")
        ax.set_ylim(0,105);ax.set_xlim(-.5,len(tasks)-.5)
        ax.set_ylabel('Feasible shots (%)');ax.grid(axis='y',alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.04,.88,'Separate completed task IDs\nDates/times in UTC, July 2026\n\nBars: pointwise 95% Wilson intervals\nDashed: analytic initial distribution\n\nTwo or three runs per setting;\nno calibration or equivalence claim.',va='top',linespacing=1.6)
    for suffix in ['pdf','png']:
        destination = out/'figures' if suffix == 'pdf' else ROOT/'qa/figure_previews'
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination/f'figureS2.{suffix}',dpi=180)
    plt.close(fig)
    md=['# Recovered hardware records: verified audit','',
        'The author supplied 26 original Braket JSON downloads. They contain 25 unique completed IonQ Forte-1 tasks (5,300 shots), 18 distinct submitted circuits and 14 distinct frozen optimization problems. All 25 tasks match archived instance definitions gate by gate. One duplicate download of the original 10-qubit task is counted once.',
        '', 'Verified widths are 6, 8, 9, 10, 12, 14, 15, 18, 20, 21, 25, 30 and 35 qubits. No 23-qubit task occurs in this download set.', '',
        '| Qubits | Doses | UTC date/time (2026) | Shots | Valid | Optimal |',
        '|---:|---:|---|---:|---:|---:|']
    for r in rows:md.append(f"| {r['qubits']} | {r['doses']} | {r['created_utc'][:16]} | {r['shots']} | {r['valid_count']} | {r['optimum_count']} |")
    md+=['','## Repeated executions','',
        'Exact submitted-circuit repeats are available at 6, 18, 20, 21 and 25 qubits. The 6-qubit pair is on July 3; the larger repeats span July 11-12. They provide descriptive repeat-execution evidence, not a powered equivalence study. No new paid tasks are needed to report these existing observations. The July 11 circuits at 6, 8, 10 and 15 qubits have changed gamma/beta values relative to the original ladder and are not identical-circuit repetitions. The two 12-qubit records encode different drug/dose instances.',
        '', '## Scientific implications','',
        'The four July 12 summaries at 18/20/21/25 qubits are now verified directly: feasible fractions are 46%, 27%, 35% and 15%. The 30/35-qubit observations are 13/200 and 7/200 feasible, with no optimum. All 14 hardware-associated problems are solved by the existing greedy baseline in 21-200 unique objective evaluations; the 35-qubit search reaches its optimum before the cap ends further exploration. The largest feasible space has only 4,900 assignments despite its 35-bit encoding.',
        '', 'All feasible assignments and all observed strings are scored without allocating a full 2^q statevector. Ideal circuit probabilities are independently calculated only through 15 qubits in this recovered analysis. No large-width noiseless or noise attribution claim is made. The runner labels 25/30/35-qubit parameters as transferred; historical training trajectories remain unavailable.',
        '', '## Reproduction and remaining information','',
        'Run `python analyze_recovered_hardware.py`, then `python build_recovered_outputs.py`. Original downloads, four unexecuted runner sources, literal frozen definitions and SHA-256 provenance are in `evidence/recovered_hardware/`. Per-task counts, quality, baselines and repeat membership are in `analysis/recovered_hardware.json` and `.csv`.',
        '', 'AWS authentication is no longer needed to analyze these supplied records. Author affiliations, correspondence, contributions, funding/support, conflicts and an authorized data-access statement still require author input. New matched hardware controls remain a possible extension, not a prerequisite for honestly reporting this retrospective benchmark.']
    (ROOT/'RECOVERED_HARDWARE_AUDIT.md').write_text('\n'.join(md)+'\n')


if __name__=='__main__':main()
