"""Rebuild selected paper tables and trajectories from the companion run archives.

Object-valued db_history records are loaded only from the hash-checked
companion artifacts. No PDE data, surrogate training or optimization is invoked.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.numbering import LEGACY_NAMESPACE, paper_problem_id

ORDER = [f'f{i:02}' for i in range(1, 12)]
PAPER = {p: f'F{i}' for i, p in enumerate(ORDER, 1)}
METHODS = ['gp_de', 'pigp_de', 'pinn_de', 'rbfn_de', 'mlp_de', 'pino_de', 'ji_sade_grm', 'glosade']
LABELS = ['GP', 'PIGP', 'PINN', 'RBFN', 'MLP', 'PINO', 'SaDE-SA-GRM', 'GLoSADE']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def best(obj, vio):
    obj, vio = np.asarray(obj, float), np.asarray(vio, float)
    assert obj.shape == vio.shape and obj.size > 0
    good = np.isfinite(obj) & np.isfinite(vio) & (vio <= 1e-4)
    return float(obj[good].min()) if good.any() else float('nan')


def load_run(path, expected):
    assert sha(path) == expected['history_sha256'], path
    config = path.with_name('resolved_config.yaml')
    assert sha(config) == expected['config_sha256'], config
    with np.load(path, allow_pickle=True) as saved:
        manifest = json.loads(str(saved['manifest_json']))
        method, artifact_problem = manifest['method'], manifest['problem']
        assert method == expected['method'] and artifact_problem == expected['artifact_problem']
        problem = paper_problem_id(artifact_problem, LEGACY_NAMESPACE)
        assert problem == expected['problem']
        assert int(manifest['seed']) == int(expected['seed'])
        history = saved['db_history']
        explicit = {'archive_real_obj', 'archive_real_vio'} <= set(saved.files)
        if explicit:
            obj, vio = saved['archive_real_obj'], saved['archive_real_vio']
            source = 'explicit_hf_archive'
        elif 'f_data' in history[-1] and 'vio_data' in history[-1]:
            obj, vio = history[-1]['f_data'], history[-1]['vio_data']
            source = 'legacy_hf_pool'
        elif method in ('ji_sade_grm', 'glosade'):
            # Ji returns its consumed-HF archive; GLoSADE returns a true-evaluation
            # set preserving its best. Only these methods support this
            # reconstruction from their returned final sets.
            obj, vio = saved['final_real_obj'], saved['final_real_vio']
            source = 'legacy_self_managed_full_archive' if method == 'ji_sade_grm' else 'legacy_glosade_best_preserving_set'
        else:
            raise ValueError(f'Unrecognized archive semantics: {path}')
        value = best(obj, vio)
        assert np.isclose(value, float(expected['best_f']), rtol=1e-12, atol=1e-14, equal_nan=True), path
        assert np.isfinite(value) == (expected['feasible'] == 'True'), path
        curves = []
        for index, snap in enumerate(history):
            if explicit or source == 'legacy_self_managed_full_archive':
                count = len(snap['f_data']) if 'f_data' in snap else int(snap['n_archive'])
                y = best(obj[:count], vio[:count])
            elif source == 'legacy_hf_pool':
                y = best(snap['f_data'], snap['vio_data'])
            else:
                y = float(snap['real_best'])
            curves.append(dict(group=expected['group'], paper_problem=PAPER[problem],
                               problem=problem, artifact_problem=artifact_problem,
                               method=method, seed=int(expected['seed']),
                               snapshot=index, hf=int(snap['eval_count']), best_feasible=y))
        assert np.isclose(curves[-1]['best_feasible'], value, rtol=1e-10, atol=1e-12, equal_nan=True)
        budget, metrics = manifest['budget'], manifest['metrics']
        hf = budget['n_parameter_queries'] if budget.get('budget_kind') == 'parameter' else budget['n_state_queries']
        row = dict(group=expected['group'], paper_problem=PAPER[problem], problem=problem,
                   artifact_problem=artifact_problem,
                   method=method, seed=int(expected['seed']), feasible=np.isfinite(value),
                   best_f=value, hf=hf, archive_source=source,
                   protocol=manifest['protocol'], dataset_sha256=manifest.get('dataset', {}).get('sha256', ''),
                   runtime=metrics.get('wall_time_total_sec', float('nan')),
                   train_t=budget.get('surrogate_train_time_sec', float('nan')),
                   infer_t=budget.get('surrogate_infer_time_sec', float('nan')),
                   mse=metrics.get('mse200_final', float('nan')))
    return row, curves


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group'], row['paper_problem'], row['method']].append(row)
    summaries = []
    for (group, problem, method), runs in groups.items():
        assert len({r['seed'] for r in runs}) == len(runs)
        assert len({(r['protocol'], r['dataset_sha256']) for r in runs}) == 1
        row = dict(group=group, paper_problem=problem, method=method, n=len(runs),
                   feasible_runs=sum(r['feasible'] for r in runs))
        for metric in ('best_f', 'hf', 'runtime', 'train_t', 'infer_t', 'mse'):
            values = np.asarray([r[metric] for r in runs], float)
            values = values[np.isfinite(values)]
            row.update({metric + '_mean': float(values.mean()) if len(values) else float('nan'),
                        metric + '_std_ddof0': float(values.std(ddof=0)) if len(values) else float('nan'),
                        metric + '_n': len(values)})
        summaries.append(row)
    return summaries


def plot_convergence(curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    selected = read_csv(ROOT / 'reproduction/metadata/convergence_selection.csv')
    seeds = {r['problem']: int(r['seed']) for r in selected}
    fig, axes = plt.subplots(3, 4, figsize=(12, 7))
    colors = ['#4477AA', '#EE6677', '#228833', '#55AACC', '#AA3377', '#332288', '#666666', '#994F00']
    markers = ['o', 's', '^', 'v', 'P', 'X', '<', '*']
    styles = ['-', '--', '-.', '--', '-.', '-', ':', '--']
    for i, problem in enumerate(ORDER):
        ax = axes.flat[i]
        for method, color, marker, style in zip(METHODS, colors, markers, styles):
            rr = [r for r in curves if r['group'] == 'main' and r['problem'] == problem
                  and r['method'] == method and r['seed'] == seeds[problem]]
            assert rr
            ax.plot([r['hf'] for r in rr], [r['best_feasible'] for r in rr],
                    color=color, marker=marker, linestyle=style, markersize=3, linewidth=1)
        ax.set_title(PAPER[problem])
        ax.set_xlabel('Evaluation')
        if i % 4 == 0:
            ax.set_ylabel('Optimization value')
        ax.grid(axis='y', alpha=.2)
    axes.flat[-1].set_axis_off()
    handles = [Line2D([], [], color=c, marker=m, ls=s, label=l)
               for c, m, s, l in zip(colors, markers, styles, LABELS)]
    fig.legend(handles=handles, loc='lower center', ncol=8, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'convergence_F1-F11.{ext}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True, help='Extracted companion artifact root')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/reproduced_paper')
    parser.add_argument('--plot', action='store_true', help='Also render a portable convergence figure')
    parser.add_argument('--include-supplements', action='store_true', help='Also rebuild the additional optimizer/constraint runs')
    args = parser.parse_args()
    index = read_csv(ROOT / 'reproduction/metadata/run_index.csv')
    assert len(index) == 2640 and not any(r['method'].startswith('rbf_') for r in index)
    if args.include_supplements:
        extra = read_csv(ROOT / 'reproduction/metadata/supplementary_index.csv')
        assert len(extra) == 2700 and not any(r['method'].startswith('rbf_') for r in extra)
        index += extra
    rows, curves = [], []
    for entry in index:
        row, trace = load_run(args.artifacts / entry['history_path'], entry)
        rows.append(row)
        curves.extend(trace)
    summaries = summarize(rows)
    assert len(summaries) == (178 if args.include_supplements else 88)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'per_run.csv', rows)
    write_csv(args.out / 'summary.csv', summaries)
    write_csv(args.out / 'convergence_all_runs.csv', curves)
    text = '# Reconstructed optimization results\n\n'
    text += 'Feasible HF archive only; ddof=0. Conditional objective means are not paired performance comparisons.\n\n'
    text += '| Problem | Method | Feasible / n | Objective mean | Objective std |\n|---|---|---:|---:|---:|\n'
    for r in summaries:
        text += f"| {r['paper_problem']} | {r['method']} | {r['feasible_runs']}/{r['n']} | {r['best_f_mean']:.8g} | {r['best_f_std_ddof0']:.8g} |\n"
    (args.out / 'tables.md').write_text(text, encoding='utf-8')
    if args.plot:
        plot_convergence(curves, args.out)
    report = dict(status='PASS', runs=len(rows), groups=len(summaries),
                  all_objectives_match_retained_ledger=True, source_hashes_checked=True,
                  new_optimizations=0, new_pde_solves=0)
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
