"""Bounded fit/dev proxy and behavior diagnostics; never fit a new model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from coco_data import ROOT, MAP_PATH, load_rows, read_frozen, sha256, verify_inputs
from fit_factors import GRID, crossed_interval, features, inside, transitions
import numpy as np
from scipy.stats import rankdata

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_NAMES = ['target_saliency', 'target_distance', 'target_history',
              'saliency_distance', 'saliency_history', 'distance_history']
METRICS = ['amplitude', 'center_distance', 'revisit']


def blocks(split):
    result = {}
    for label in ('fit', 'dev'):
        groups = sorted({m['group'] for m in split.values() if m['split'] == label},
                        key=lambda g: hashlib.sha256(f'20260924:{g}'.encode()).hexdigest())
        result.update({g: 'AB'[i % 2] for i, g in enumerate(groups)})
    return result


def cue_contrast(scores, matched):
    """Midrank uses the complete choice set, including ties and nonchosen cells."""
    ranks = (rankdata(scores, axis=1, method='average') - .5) / scores.shape[1]
    return ranks[matched] - np.delete(ranks, matched, axis=0).mean(axis=0)


def candidate_correlations(values):
    centered = values - values.mean(axis=0)
    norms = np.sqrt((centered * centered).sum(axis=0))
    return np.array([float(centered[:, i] @ centered[:, j] / (norms[i] * norms[j]))
                     if norms[i] > 1e-12 and norms[j] > 1e-12 else np.nan for i, j in PAIRS])


def gate_summary(records):
    output = {}
    for phase in ('pre_entry', 'target_entry'):
        rows = [r for r in records if r['phase'] == phase]
        if not rows:
            output[phase] = {'trials': 0, 'interval': None, 'categories': {}, 'blocks': {}}
            continue
        values = np.array([r['contrast'] for r in rows])
        interval = crossed_interval({'sid': np.array([r['subject'] for r in rows]),
                                     'group': np.array([r['group'] for r in rows])}, values)
        categories = {task: {'mean': float(np.mean([r['contrast'] for r in rows if r['task'] == task])),
                              'trials': sum(r['task'] == task for r in rows)}
                      for task in sorted({r['task'] for r in rows})}
        output[phase] = {'trials': len(rows), 'choices': sum(r['choices'] for r in rows),
                         'interval': interval, 'categories': categories,
                         'blocks': {b: float(np.mean([r['contrast'] for r in rows if r['block'] == b]))
                                    for b in ('A', 'B') if any(r['block'] == b for r in rows)}}
    pre = output['pre_entry']
    checks = {
        'pre_entry_interval_above_zero': pre['interval'] is not None and
            pre['interval']['crossed_bootstrap_95_percentile_interval'][0] > 0,
        'both_dev_blocks_positive': len(pre['blocks']) == 2 and all(v > 0 for v in pre['blocks'].values()),
        'at_least_12_of_18_categories_positive': len(pre['categories']) == 18 and
            sum(v['mean'] > 0 for v in pre['categories'].values()) >= 12,
    }
    return {'passed': all(checks.values()), 'checks': checks, 'phases': output,
            'decision': 'May implement the locked four-model comparison' if all(checks.values()) else
                        'Stop four-model comparison under the locked rule; do not expand models'}


def behavior_summary(trials, strata, split, block):
    output = {}
    for label in ('fit', 'dev'):
        chosen = [r for r in trials if r['split'] == label]
        raw = {}
        for subject in sorted({r['subject'] for r in chosen}):
            values = np.array([r['values'] for r in chosen if r['subject'] == subject])
            raw[str(subject)] = {'trials': len(values), 'mean': values.mean(axis=0).tolist(),
                                'trial_mean_quantiles_10_50_90': np.quantile(values, [.1, .5, .9], axis=0).tolist()}
        residuals = defaultdict(list)
        eligible = 0
        for (image, task, order), people in strata.items():
            if split[image]['split'] != label or len(people) < 2:
                continue
            eligible += 1
            means = {s: np.mean(v, axis=0) for s, v in people.items()}
            center = np.mean(list(means.values()), axis=0)
            for subject, value in means.items():
                residuals[subject, block[split[image]['group']]].append(value - center)
        adjusted = {str(s): {b: np.mean(residuals[s, b], axis=0).tolist() for b in ('A', 'B')
                            if residuals[s, b]} for s in sorted({r['subject'] for r in chosen})}
        complete = [v for v in adjusted.values() if 'A' in v and 'B' in v]
        correlations = []
        for index in range(3):
            a = np.array([v['A'][index] for v in complete]); b = np.array([v['B'][index] for v in complete])
            correlations.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None)
        output[label] = {'raw_trial_means': raw, 'adjusted_block_means': adjusted,
                         'adjusted_block_pearson_r': correlations, 'paired_subjects': len(complete),
                         'eligible_image_category_order_strata': eligible,
                         'interpretation': 'Descriptive across ten participants; incomplete matching and sample composition remain limitations'}
    return {'metric_order': METRICS, 'splits': output}


def run(output):
    if output.exists():
        raise FileExistsError('Output exists; preserve the recorded diagnostic result')
    started = time.time()
    input_check = verify_inputs()
    protocol_hash = sha256(ROOT / 'protocol.md')
    source_hash = sha256(Path(__file__))
    split = read_frozen('split.json')['images']; block = blocks(split)
    rows = load_rows()
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['name'], row['task']].append(row)
    tasks = sorted({r['task'] for r in rows})
    if len(tasks) != 18:
        raise ValueError('Expected all 18 category templates')
    crops = defaultdict(list)
    cache_hashes = {}

    def cache_path(image, task):
        path = ROOT / 'data/processed' / f"{image[:-4]}_{task.replace(' ', '_')}.npz"
        cache_hashes[str(path.relative_to(ROOT))] = sha256(path)
        return path

    for image, task in sorted(grouped):
        if split[image]['split'] == 'fit':
            with np.load(cache_path(image, task)) as cache:
                crops[task].append(cache['crop'])
    templates = []
    for task in tasks:
        if len(crops[task]) < 2:
            raise ValueError(f'Insufficient fitting crops: {task}')
        vector = np.mean(crops[task], axis=0)
        templates.append(vector / np.linalg.norm(vector))
    del crops
    print(f'Templates ready from fitting images; processing {len(grouped)} fit/dev image-category pairs', flush=True)
    records = []; trial_behavior = []; correlation_rows = []
    strata = defaultdict(lambda: defaultdict(list))
    maximum_map_error = 0.
    with np.load(MAP_PATH) as maps:
        for index, ((image, task), trials) in enumerate(sorted(grouped.items())):
            label = split[image]['split']; group = split[image]['group']
            key = image[:-4] + '_' + task.replace(' ', '_')
            mapped = maps[key]
            contrast = None
            if label == 'dev':
                with np.load(cache_path(image, task)) as cache:
                    scene = cache['scene']
                scores = np.array([np.einsum('chw,c->hw', scene, v).ravel() for v in templates])
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite reconstructed cue maps')
                matched = tasks.index(task)
                error = float(np.max(np.abs(scores[matched] - mapped[0].ravel())))
                maximum_map_error = max(maximum_map_error, error)
                if error > 1e-6:
                    raise ValueError(f'Reconstructed template differs from frozen map: {image}, {task}, {error}')
                contrast = cue_contrast(scores, matched)
            pair_correlations = []
            for row in trials:
                phase_values = defaultdict(list); behaviors = []
                for t, points in transitions(row):
                    position = points[t+1] / 1680
                    cell = int(points[t+1, 1] / 52.5) * 32 + int(points[t+1, 0] / 52.5)
                    if contrast is not None:
                        phase = 'target_entry' if inside(points[t+1], row['bbox']) else 'pre_entry'
                        phase_values[phase].append(float(contrast[cell]))
                    value = [float(np.linalg.norm(position - points[t]/1680)),
                             float(np.linalg.norm(position - [.5, .3125])),
                             float(np.any(np.linalg.norm(points[:t]/1680-position, axis=1) < .06))]
                    behaviors.append(value)
                    order = 'first' if t == 0 else 'early' if t <= 3 else 'late'
                    strata[image, task, order][row['subject']].append(value)
                    values = features(row, t, points, mapped)[:, [0, 1, 2, 4]].astype(float)
                    pair_correlations.append(candidate_correlations(values))
                if behaviors:
                    trial_behavior.append({'subject': row['subject'], 'split': label,
                                           'values': np.mean(behaviors, axis=0)})
                for phase, values in phase_values.items():
                    records.append({'image': image, 'group': group, 'task': task, 'subject': row['subject'],
                                    'block': block[group], 'phase': phase, 'contrast': float(np.mean(values)),
                                    'choices': len(values)})
            if pair_correlations:
                values = np.array(pair_correlations)
                counts = np.isfinite(values).sum(axis=0)
                means = np.divide(np.nansum(values, axis=0), counts, out=np.full(6, np.nan), where=counts > 0)
                correlation_rows.append({'image': image, 'task': task, 'split': label,
                                         'mean_r': [float(v) if np.isfinite(v) else None for v in means],
                                         'valid_choice_counts': counts.tolist()})
            if (index+1) % 100 == 0:
                print(f'Processed {index+1}/{len(grouped)} image-category pairs; elapsed={time.time()-started:.1f}s', flush=True)
    if sha256(ROOT/'protocol.md') != protocol_hash or sha256(Path(__file__)) != source_hash:
        raise ValueError('Protocol or source changed during execution')
    corr_summary = {}
    for label in ('fit', 'dev'):
        # Equal weight to images, even where one image has multiple target categories.
        images = defaultdict(list)
        for row in correlation_rows:
            if row['split'] == label:
                images[row['image']].append([np.nan if v is None else v for v in row['mean_r']])
        means = np.array([np.nanmean(v, axis=0) for v in images.values()])
        corr_summary[label] = {name: {'mean_r': float(np.nanmean(means[:, i])),
                                      'image_quantiles_10_50_90': np.nanquantile(means[:, i], [.1, .5, .9]).tolist()}
                               for i, name in enumerate(PAIR_NAMES)}
    result = {'scope': 'Observed fit/dev diagnostics, not new model results or causal evidence',
              'seconds': time.time()-started, 'protocol_sha256': protocol_hash, 'source_sha256': source_hash,
              'inputs': input_check, 'cache_sha256': cache_hashes,
              'template_reconstruction_max_absolute_error': maximum_map_error,
              'gate': gate_summary(records), 'proxy_trial_phases': records,
              'behavior': behavior_summary(trial_behavior, strata, split, block),
              'candidate_correlations': corr_summary, 'candidate_correlations_by_image_category': correlation_rows}
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(output)
    print(json.dumps({'output':str(output), 'gate': result['gate'], 'seconds': time.time()-started}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/input-checks.json')
    run(parser.parse_args().output)
