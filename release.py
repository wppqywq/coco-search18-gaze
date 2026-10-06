"""Plot saved results and export an allowlisted publication snapshot; never fit."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.unicode_minus': False})
    run = read('reports/hierarchical/full-shared/run.json')
    assert run['status'] == 'complete' and run['diagnostics']['passed']
    result = read('reports/hierarchical/full-shared/evaluation.json')
    post = result['posterior']['mu']
    indices = [0, 1, 2, 3, 4, 10, 11, 12, 13]
    labels = ['Target', 'Saliency', 'Distance', 'Distance squared', 'History',
              'Directional momentum', 'Horizontal preference', 'First distance', 'First target']
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    mid = np.array(post['mean'])[indices]
    low = np.array(post['lower_95'])[indices]
    high = np.array(post['upper_95'])[indices]
    ax.errorbar(mid, np.arange(len(indices)), xerr=[mid-low, high-mid], fmt='o', capsize=3, color='#245580')
    ax.axvline(0, color='0.6', lw=1)
    ax.set(yticks=np.arange(len(indices)), yticklabels=labels,
           xlabel='Standardized conditional weight (mean and 95% credible interval)',
           title='Hierarchical shared-factor model: full fit')
    ax.invert_yaxis()
    fig.savefig(ROOT/'reports/hierarchical/coefficients.png', dpi=180)
    plt.close(fig)
    d = read('reports/shared-ablation.json')
    assert d['status'] == 'complete' and all(m['converged'] for m in d['models'].values())
    terms = list(d['contrasts'])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    means = np.array([d['contrasts'][k]['mean'] for k in terms])
    ci = np.array([d['contrasts'][k]['crossed_bootstrap_95_percentile_interval'] for k in terms])
    ax.errorbar(means, np.arange(len(terms)), xerr=[means-ci[:, 0], ci[:, 1]-means], fmt='o', capsize=3, color='#98602a')
    ax.axvline(0, color='0.6', lw=1)
    ax.set(yticks=np.arange(len(terms)), yticklabels=[k.replace('_', ' ').title() for k in terms],
           xlabel='Full minus reduced dev bits/choice (95% crossed-bootstrap interval)',
           title='Separate experiment: fully shared spline term ablation')
    ax.invert_yaxis(); ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(ROOT/'reports/shared-ablation.png', dpi=180)
    plt.close(fig)
    d = read('reports/test-evaluation.json')
    keys = ['personal_gain_existing', 'personal_gain_spline', 'spline_gain_shared_factors', 'spline_gain_personal']
    fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
    mid = np.array([d['contrasts'][k]['mean'] for k in keys])
    ci = np.array([d['contrasts'][k]['crossed_bootstrap_95_percentile_interval'] for k in keys])
    ax.errorbar(mid, np.arange(4), xerr=[mid-ci[:, 0], ci[:, 1]-mid], fmt='o', capsize=3, color='#245580')
    ax.axvline(0, color='0.6', lw=1)
    ax.axvline(.005, color='0.5', ls='--', label='Predeclared practical margin')
    ax.set(yticks=np.arange(4), yticklabels=['Personal T/D: existing', 'Personal T/D: spline', 'Spline: shared T/D', 'Spline: personal T/D'],
           xlabel='Gain in test bits/choice (95% crossed-bootstrap interval)',
           title='Earlier penalized comparison: reused held-out test images')
    ax.invert_yaxis()
    fig.savefig(ROOT/'reports/test-summary.png', dpi=180)
    plt.close(fig)
    print('Created three figures from saved human-data result summaries; no fitting or scoring.')


def portable(text):
    text = re.sub(r'/Users/[^\s\"\x27`<>]+', '<local-path>', text)
    text = text.replace('coco\u91cd\u6784\u65b9\u6848.pdf', 'refactoring-proposal.pdf')
    for old, new in {'\u2014':'-', '\u2013':'-', '\u2212':'-', '\u00d7':'x', '\u00b2':'^2', '\u2019':"'", '\u201c':'"', '\u201d':'"', '\u00b7':';'}.items():
        text = text.replace(old, new)
    if not text.isascii():
        raise ValueError('Unmapped non-ASCII text in publication file')
    return text


def export(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Destination exists; review or remove it before exporting again')
    files = [*sorted(ROOT.glob('*.py')), ROOT/'README.md', ROOT/'requirements.txt', ROOT/'.gitignore',
             ROOT/'protocol.md', ROOT/'reports/hierarchical-protocol.md', ROOT/'vendor/README.md']
    files += [ROOT/'reports'/n for n in ['comparison.json', 'comparison-audit.json', 'input-checks.json',
              'test-evaluation.json', 'test-report.md', 'shared-ablation.json', 'shared-ablation.png', 'test-summary.png']]
    files += sorted((ROOT/'reports/frozen').glob('*.json'))
    files += [ROOT/'reports/hierarchical/validation.json', ROOT/'reports/hierarchical/coefficients.png']
    for name in ['full-shared', 'A-shared', 'full-personal']:
        folder = ROOT/'reports/hierarchical'/name
        files += [folder/'run.json']
        if (folder/'evaluation.json').exists(): files.append(folder/'evaluation.json')
    payloads = {}
    manifest = {'purpose':'Publication packaging; numerical results are unchanged.',
                'excluded':'Git history, third-party source, raw data, caches, per-choice arrays, traces, logs and recovery archive.',
                'note':'Original and published hashes distinguish historical evidence from ASCII/path-redacted packaging. Historical protected-source audits require the original local snapshot.',
                'files':{}}
    for source in files:
        relative = str(source.relative_to(ROOT))
        original = source.read_bytes()
        if source.suffix == '.json':
            text = json.dumps(json.loads(original), indent=2, ensure_ascii=True, allow_nan=False)+'\n'
            packaged = portable(text).encode('ascii')
        elif source.suffix == '.py':
            original.decode('ascii')
            packaged = original
        elif source.suffix in ('.md', '.txt'):
            packaged = portable(original.decode()).encode('ascii')
        else:
            packaged = original
        payloads[relative] = packaged
        manifest['files'][relative] = {'source_sha256':hashlib.sha256(original).hexdigest(),
                                       'published_sha256':hashlib.sha256(packaged).hexdigest()}
    destination.mkdir(parents=True)
    for relative, payload in payloads.items():
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
    (destination/'release-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Exported {len(files)} allowlisted files to {destination}. No Git history or remote created.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plot', 'export'])
    parser.add_argument('--destination', type=Path, default=ROOT/'public')
    args = parser.parse_args()
    if args.action == 'plot': figures()
    else: export(args.destination)


if __name__ == '__main__':
    main()
