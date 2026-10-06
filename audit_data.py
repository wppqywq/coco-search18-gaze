"""Audit frozen inputs or reload saved comparison fits; never optimize a model."""
import collections
import hashlib
import json
import zipfile
from coco_data import ARCHIVE_PATH, verify_inputs


def main():
    archive = ARCHIVE_PATH
    report = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': {}}
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None
        splits = {}
        for split in ('train', 'validation'):
            name = f'coco_search18_fixations_TP_{split}_split1.json'
            rows = json.loads(source.read(name))
            splits[split] = {r['name'] for r in rows}
            keys = collections.Counter((r['name'], r['task'], r['subject']) for r in rows)
            bad_lengths = sum(not (len(r['X']) == len(r['Y']) == len(r['T']) == r['length']) for r in rows)
            outside = sum(not (0 <= x < 1680 and 0 <= y < 1050) for r in rows for x, y in zip(r['X'], r['Y']))
            first_distances = [((r['X'][0]-840)**2+(r['Y'][0]-525)**2)**0.5 for r in rows if r['X']]
            report['files'][name] = {
                'trials': len(rows), 'images': len(splits[split]),
                'participants': dict(sorted(collections.Counter(r['subject'] for r in rows).items())),
                'categories': dict(sorted(collections.Counter(r['task'] for r in rows).items())),
                'fixations_including_initial': sum(len(r['X']) for r in rows),
                'incorrect_trials': sum(r['correct'] != 1 for r in rows),
                'inconsistent_lengths': bad_lengths,
                'duplicate_subject_image_task_keys': sum(n-1 for n in keys.values()),
                'out_of_display_fixations': outside,
                'nonpositive_durations': sum(t <= 0 for r in rows for t in r['T']),
                'initial_fixation_over_100px_from_center': sum(d > 100 for d in first_distances),
            }
        report['train_validation_image_overlap'] = sorted(splits['train'] & splits['validation'])
    report['notes'] = [
        'Use split1 only: split2 is an alternative assignment of the same observations, not additional participants.',
        'Original readme specifies 1680x1050 padded stimuli, initial center fixation included, and bbox as x,y,w,h.',
        'Out-of-display samples require explicit exclusion without bridging invalid history.',
        'Durations are not onset timestamps; cumulative fixation durations omit intervening saccades.',
        'Archive downloaded from official HTTPS URL with certificate verification disabled after certificate-chain failures; SHA256 records local integrity, not authenticated provenance.',
    ]
    report['frozen_inputs'] = verify_inputs()
    print(json.dumps(report, indent=2))


def verify_comparison():
    """Reload every saved fit without optimizing, and audit all selected dev predictions."""
    import time
    import numpy as np
    from coco_data import ROOT, sha256, read_frozen
    from analyze_factors import blocks, gate_summary
    from fit_factors import (build_data, ChoiceDesign, CONTRAST, choice_objective,
                             score_comparison, crossed_interval)
    started = time.time()
    path = ROOT/'reports/comparison.json'
    result = json.loads(path.read_text())
    if result['status'] != 'complete':
        raise ValueError('Comparison has not completed')
    hashes = {p:sha256(ROOT/p) for p in result['protected_sha256']}
    if hashes != result['protected_sha256']:
        raise ValueError('Recorded computation or protocol has changed')
    gate = json.loads((ROOT/'reports/input-checks.json').read_text())
    assert gate_summary(gate['proxy_trial_phases']) == gate['gate']
    assert gate['gate']['passed']
    data = build_data(raw=True)
    assert data['audit'] == result['data_audit']
    dev = np.flatnonzero(data['split'] == 'dev')
    assignment = blocks(read_frozen('split.json')['images'])
    block_labels = np.array([assignment[g] for g in data['group']])
    errors = dict(predictions=0., candidate_means=0., conditional_checks=0.,
                  selected_objectives=0., selected_max_gradients=0.)
    count = 0; selected_count = 0
    predicted = {}
    expected_families = {f'{b}/{f}/history={h}' for b in ('full','A','B')
                         for f in ('existing','spline') for h in (True,False)}
    assert set(result['runs']) == expected_families
    with np.load(ROOT/'reports/comparison-predictions.npz') as saved:
        for key, values in {'image':data['image'][dev], 'subject':data['sid'][dev]+1,
                            'group':data['group'][dev], 'block':block_labels[dev]}.items():
            np.testing.assert_array_equal(saved[key], values)
        assert len(saved.files) == 28
        for block in ('full', 'A', 'B'):
            train = np.flatnonzero((data['split']=='fit') & ((block_labels==block) if block!='full' else True))
            for form in ('existing', 'spline'):
                design = ChoiceDesign(data, form, train)
                for history in (True, False):
                    family = f'{block}/{form}/history={history}'
                    record = result['runs'][family]
                    assert record['training_choices'] == len(train)
                    assert record['design'] == design.specification()
                    assert set(record['models']) == {'shared_factors','personal'}
                    for name, entry in record['models'].items():
                        assert [c['model']['penalty'] for c in entry['candidates']] == [1.,10.,100.]
                        assert entry['selected'] == max(entry['candidates'], key=lambda c:c['mean_dev_log2_probability'])
                        for candidate in entry['candidates']:
                            model = candidate['model']
                            assert model['converged'] and model['history'] == history
                            assert model['personal_factors'] == (name == 'personal')
                            logp, checks = score_comparison(design, dev, model)
                            assert np.isfinite(logp).all()
                            errors['candidate_means'] = max(errors['candidate_means'], abs(logp.mean()-candidate['mean_dev_log2_probability']))
                            count += 1
                            if candidate == entry['selected']:
                                key = family+'/'+name
                                errors['predictions'] = max(errors['predictions'], float(np.max(np.abs(logp-saved[key]))))
                                predicted[key] = saved[key].copy()
                                for person, measures in checks['by_subject'].items():
                                    for kind in ('observed','predicted'):
                                        error = np.max(np.abs(np.array(measures[kind])-entry['conditional_checks']['by_subject'][person][kind]))
                                        errors['conditional_checks'] = max(errors['conditional_checks'], float(error))
                                weights = np.array(model['weights']); mean = weights.mean(axis=0)
                                shared = [i for i in design.shared if history or design.names[i]!='history']
                                personal = design.spatial + (design.factors if name=='personal' else [])
                                theta = np.r_[mean[shared], (CONTRAST.T @ (weights[:,personal]-mean[personal])).ravel()]
                                loss, gradient = choice_objective(theta, design, train, shared, personal, model['penalty'])
                                errors['selected_objectives'] = max(errors['selected_objectives'], abs(loss-model['objective']))
                                errors['selected_max_gradients'] = max(errors['selected_max_gradients'], abs(np.abs(gradient).max()-model['max_abs_gradient']))
                                selected_count += 1
                    print(f'Verified {family}: {count}/72 saved fits', flush=True)
    assert count == 72 and selected_count == 24
    assert max(errors.values()) < 1e-10, errors
    contrast = {}
    def compare(key, difference, dev_block='full'):
        mask = np.ones(len(dev), dtype=bool) if dev_block == 'full' else block_labels[dev] == dev_block
        contrast[key] = crossed_interval({'sid':data['sid'][dev][mask], 'group':data['group'][dev][mask]}, difference[mask])
    for history in (True, False):
        for block in ('full','A','B'):
            for form in ('existing','spline'):
                family = f'{block}/{form}/history={history}'
                difference = predicted[family+'/personal']-predicted[family+'/shared_factors']
                for db in ('full','A','B'):
                    compare(f'{family}/personal_gain/dev={db}', difference, db)
            for name in ('shared_factors','personal'):
                difference = predicted[f'{block}/spline/history={history}/{name}']-predicted[f'{block}/existing/history={history}/{name}']
                for db in ('full','A','B'):
                    key = f'{block}/response_gain/history={history}/{name}' + ('' if db=='full' else f'/dev={db}')
                    compare(key, difference, db)
            difference = (predicted[f'{block}/spline/history={history}/personal']-predicted[f'{block}/spline/history={history}/shared_factors']
                          -predicted[f'{block}/existing/history={history}/personal']+predicted[f'{block}/existing/history={history}/shared_factors'])
            compare(f'{block}/interaction/history={history}', difference)
    assert all(contrast[key] == value for key, value in result['contrasts'].items())
    behavior = {}; model_scores = {}
    counts = np.array([sum(data['sid'][dev] == s) for s in range(10)])
    for form in ('existing','spline'):
        for name in ('shared_factors','personal'):
            key = f'full/{form}/history=True/{name}'
            entry = result['runs'][f'full/{form}/history=True']['models'][name]
            checks = entry['conditional_checks']['by_subject']
            observed = np.array([checks[str(s)]['observed'] for s in range(1,11)])
            expected = np.array([checks[str(s)]['predicted'] for s in range(1,11)])
            behavior[key] = {'observed_choice_weighted':np.average(observed, axis=0, weights=counts).tolist(),
                             'predicted_choice_weighted':np.average(expected, axis=0, weights=counts).tolist(),
                             'equal_subject_mae':np.abs(expected-observed).mean(axis=0).tolist()}
            model_scores[key] = {'mean_log2_probability':float(predicted[key].mean()),
                                 'penalty':entry['selected']['model']['penalty']}
    for form in ('existing','spline'):
        for name in ('shared_factors','personal'):
            compare(f'full/history_gain/{form}/{name}', predicted[f'full/{form}/history=True/{name}']-predicted[f'full/{form}/history=False/{name}'])
    audit = {'status':'verified', 'fit_count':count, 'selected_models':selected_count,
             'rescored_candidate_choices':count*len(dev), 'verified_saved_predictions':selected_count*len(dev),
             'maximum_absolute_errors':errors, 'recorded_contrasts_recomputed':len(result['contrasts']),
             'protocol_and_source_hashes':hashes, 'result_sha256':sha256(path),
             'predictions_sha256':sha256(ROOT/'reports/comparison-predictions.npz'),
             'seconds':time.time()-started, 'new_fits':0, 'test_scored':False,
             'model_scores':model_scores, 'contrasts':contrast, 'conditional_behavior':behavior,
             'limitations':['All scores are development scores used for penalty selection; intervals condition on selected fits and omit selection/training uncertainty.',
                            'A/B models share frozen visual templates and the same dev observations; this is not independent confirmation.',
                            'A passed cue-rank gate is not proof of specificity after controlling for other visual properties.',
                            'The 0.005 bits/choice margin is a project convention, not a known scientific equivalence threshold.']}
    output = ROOT/'reports/comparison-audit.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:audit[k] for k in ('status','fit_count','verified_saved_predictions','maximum_absolute_errors','seconds')},indent=2))


def score_final_test():
    """User-authorized reuse of the original test; frozen dev-selected models only."""
    from pathlib import Path
    import numpy as np
    from scipy.interpolate import CubicSpline
    from coco_data import ROOT, MAP_PATH, read_frozen, sha256
    from fit_factors import features, transitions, ChoiceDesign, predict, batches, crossed_interval
    output = ROOT/'reports/test-evaluation.json'
    if output.exists():
        raise FileExistsError('Test evaluation already exists; inspect it instead of overwriting')
    inputs = verify_inputs()
    comparison_path = ROOT/'reports/comparison.json'
    comparison = json.loads(comparison_path.read_text())
    assert comparison['status'] == 'complete'
    for name, digest in comparison['protected_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    split = read_frozen('split.json')['images']
    scale = np.array([comparison['data_audit']['feature_scale'][name]
                      for name in __import__('fit_factors').NAMES], dtype=np.float32)
    with zipfile.ZipFile(ARCHIVE_PATH) as source:
        records = sum((json.loads(source.read(f'coco_search18_fixations_TP_{s}_split1.json'))
                       for s in ('train','validation')), [])

    def build(label):
        values=[]; ys=[]; subjects=[]; images=[]; groups=[]; trials=0
        with np.load(MAP_PATH) as maps:
            for row in records:
                if split.get(row['name'], {}).get('split') != label:
                    continue
                trials += 1
                key = row['name'][:-4]+'_'+row['task'].replace(' ','_')
                mapped = maps[key]
                for t, points in transitions(row):
                    values.append(features(row,t,points,mapped))
                    ys.append(int(points[t+1,1]/52.5)*32+int(points[t+1,0]/52.5))
                    subjects.append(row['subject']-1);images.append(row['name']);groups.append(split[row['name']]['group'])
        X=np.array(values); raw=X[:,:,[0,2]].copy()
        X-=X.mean(axis=1,keepdims=True);X/=scale
        return {'X':X,'raw':raw,'y':np.array(ys),'sid':np.array(subjects),
                'image':np.array(images),'group':np.array(groups),'trials':trials}

    def restored(data, spec):
        # Restore recorded basis and scales; never recompute knots or scaling on test.
        design=ChoiceDesign.__new__(ChoiceDesign)
        design.data=data;design.form=spec['form'];design.names=spec['names']
        design.keep=[i for i in range(14) if i not in (0,2,3)]
        design.knots=spec['knots'];design.scale=np.array(spec['scale'])
        design.shared=spec['shared_columns'];design.spatial=spec['spatial_columns'];design.factors=spec['factor_columns']
        design.splines=[CubicSpline(nodes,np.eye(4)[:,:3],bc_type='natural') for nodes in design.knots]
        return design

    def score(data):
        out={}
        for form in ('existing','spline'):
            family=comparison['runs'][f'full/{form}/history=True']
            design=restored(data,family['design'])
            for name in ('shared_factors','personal'):
                model=family['models'][name]['selected']['model']
                values=[]
                for indices in batches(np.arange(len(data['y']))):
                    values.extend(predict({'X':design.matrix(indices),'sid':data['sid'][indices],'y':data['y'][indices]},model))
                out[form+'/'+name]=np.array(values)
                assert np.isfinite(out[form+'/'+name]).all()
        return out

    # Validate the evaluation-only path against every saved primary dev prediction first.
    dev=build('dev');dev_scores=score(dev);maximum_error=0.
    with np.load(ROOT/'reports/comparison-predictions.npz') as saved:
        np.testing.assert_array_equal(dev['image'],saved['image'])
        np.testing.assert_array_equal(dev['sid']+1,saved['subject'])
        for name, values in dev_scores.items():
            form, personal=name.split('/')
            maximum_error=max(maximum_error,float(np.max(np.abs(values-saved[f'full/{form}/history=True/{personal}']))))
    assert maximum_error < 1e-12, maximum_error
    dev_count=len(dev['y'])*4
    del dev,dev_scores
    data=build('test');scores=score(data)
    assert all(split[name]['split']=='test' for name in data['image'])
    assert not set(data['image']) & {name for name,meta in split.items() if meta['split'] in ('fit','dev','previous_test')}
    models={name:{'mean_log2_probability':float(values.mean()),
                  'bits_over_uniform':float(values.mean()+np.log2(640)),
                  'by_subject':{str(s+1):float(values[data['sid']==s].mean()) for s in np.unique(data['sid'])}}
            for name,values in scores.items()}
    differences={
        'personal_gain_existing':scores['existing/personal']-scores['existing/shared_factors'],
        'personal_gain_spline':scores['spline/personal']-scores['spline/shared_factors'],
        'spline_gain_shared_factors':scores['spline/shared_factors']-scores['existing/shared_factors'],
        'spline_gain_personal':scores['spline/personal']-scores['existing/personal'],
    }
    differences['interaction']=differences['personal_gain_spline']-differences['personal_gain_existing']
    report={'scope':'Explicitly requested reuse of the original full test; previously used by the older COCO study, not a new independent confirmation',
            'authorization':'User requested test-set scoring on 2026-09-25',
            'new_fits':0,'retuned_parameters':False,'included_training_blocks':['full'],'history':True,
            'trials':data['trials'],'choices':len(data['y']),'images':len(set(data['image'])),
            'participants':len(set(data['sid'])),'models':models,
            'contrasts':{name:crossed_interval(data,values) for name,values in differences.items()},
            'verification':{'primary_dev_predictions_reproduced':dev_count,'max_absolute_error_bits':maximum_error,
                            'fit_dev_previous_test_overlap':0},
            'source_sha256':sha256(Path(__file__)),'comparison_sha256':sha256(comparison_path),
            'inputs':inputs,'test_scored':True,
            'limitations':['Intervals condition on already selected and fitted models; no refitting in bootstrap.',
                           'Ten participants and prior use of this test limit confirmatory interpretation.',
                           'All four models include personal spatial controls; personal here adds only target/distance offsets.']}
    for name,digest in comparison['protected_sha256'].items():
        assert sha256(ROOT/name)==digest,name
    np.savez_compressed(ROOT/'reports/test-predictions.npz',**scores,image=data['image'],subject=data['sid']+1,group=data['group'])
    report['predictions_sha256']=sha256(ROOT/'reports/test-predictions.npz')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='source', choices=['source','comparison','test'])
    args = parser.parse_args()
    {'source':main,'comparison':verify_comparison,'test':score_final_test}[args.stage]()
