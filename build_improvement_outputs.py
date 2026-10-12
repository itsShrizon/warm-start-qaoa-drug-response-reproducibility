"""Generate added tables and a supplementary score-distribution figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    quality=json.loads((ROOT/'analysis/quality_results.json').read_text())['rows']
    sensitivity=json.loads((ROOT/'analysis/sensitivity_results.json').read_text())['rows']
    tex=[]
    for r in quality:
        tex.append(f"{r['qubits']} & {100*r['hardware_mean_normalized']:.1f} & "
                   f"{100*r['hardware_best_normalized_gap']:.1f} & "
                   f"{100*r['greedy_local']['normalized_gap']:.1f} & {r['greedy_local']['objective_evaluations']}\\\\")
    (ROOT/'overleaf/table2_rows.tex').write_text('\\newcommand{\\QualityRows}{\n'+'\n'.join(tex)+'\n}\n')
    tex=[]
    for r in sensitivity:
        m=r['methods']
        tex.append(f"{r['panel'].split('_')[-1]} & {r['seed']} & "
            f"{100*m['ideal_qaoa']['feasibility']:.2f} & {100*m['ideal_qaoa']['optimum_probability']:.3f} & "
            f"{100*m['warm_start_only']['optimum_probability']:.3f} & "
            f"{100*m['uniform_feasible']['optimum_probability']:.3f}\\\\")
    (ROOT/'overleaf/supp_panel_rows.tex').write_text('\\newcommand{\\PanelRows}{\n'+'\n'.join(tex)+'\n}\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    figure,axes=plt.subplots(2,3,figsize=(8.2,4.9),layout='constrained',sharex=True,sharey=True)
    for ax,r in zip(axes.flat,quality):
        values=np.array([v['score'] for v in r['feasible_score_histogram']])
        counts=np.array([v['count'] for v in r['feasible_score_histogram']])
        x=(values-r['worst_feasible'])/(r['optimum']-r['worst_feasible'])
        ax.step(np.r_[0,x,1],np.r_[0,np.cumsum(counts)/counts.sum(),1],where='post',color='#0072B2',lw=1.5)
        ax.set_title(f"{r['qubits']} qubits; {r['valid_count']}/{r['total_shots']} valid",fontsize=9)
        ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.grid(alpha=.2)
    figure.supxlabel('Feasible score normalized from worst (0) to optimum (1)')
    figure.supylabel('Cumulative fraction of valid hardware shots')
    for ext in ['pdf','png']:
        destination = ROOT/('overleaf/figures' if ext == 'pdf' else 'qa/figure_previews')
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination/f'figureS1.{ext}',dpi=350)
    plt.close(figure)
    print('Generated quality table, all-seed panel table, and six hardware score distributions')


if __name__=='__main__':
    main()
