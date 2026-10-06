"""Development-only, fully shared leave-one-term-out analysis."""
import json
import time

from coco_data import ROOT, sha256
from fit_factors import (ChoiceDesign, batches, build_data, choice_objective,
                         crossed_interval, minimize, np, predict)

TERMS = ('x', 'y', 'x_squared', 'y_squared', 'xy', 'momentum',
         'horizontal', 'first_distance', 'first_target')


def fit_shared(design, indices, drop=None):
    if not len(indices) or not np.all(design.data['split'][indices] == 'fit'):
        raise ValueError('Fitting requires nonempty fit-only records')
    if drop is not None and drop not in TERMS:
        raise ValueError('Unknown ablation term')
    shared = [i for i in design.shared if design.names[i] != drop]
    started = time.time()
    iteration = 0
    latest = {}

    def objective(theta):
        loss, gradient = choice_objective(theta, design, indices, shared, [], 0.)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite objective or gradient')
        latest['loss'] = float(loss)
        return loss, gradient

    def progress(theta):
        nonlocal iteration
        iteration += 1
        if iteration == 1 or iteration % 10 == 0:
            print(f'{drop or "full"}: iteration={iteration}, objective={latest["loss"]:.9f}, seconds={time.time()-started:.1f}', flush=True)

    opt = minimize(objective, np.zeros(len(shared)), method='L-BFGS-B', jac=True,
                   callback=progress, options={'maxiter':500, 'ftol':1e-10, 'gtol':1e-6, 'maxls':30})
    weights = np.zeros((10, len(design.names)))
    weights[:, shared] = opt.x
    return {'weights':weights.tolist(), 'dropped_term':drop, 'personal_columns':[],
            'parameter_count':len(shared), 'converged':bool(opt.success),
            'message':str(opt.message), 'iterations':int(opt.nit),
            'objective':float(opt.fun), 'max_abs_gradient':float(np.max(np.abs(opt.jac))),
            'seconds':time.time()-started}


def score_dev(design, indices, model):
    if not len(indices) or not np.all(design.data['split'][indices] == 'dev'):
        raise ValueError('Scoring requires nonempty dev-only records')
    return np.concatenate([predict({'X':design.matrix(chunk),
        'sid':design.data['sid'][chunk], 'y':design.data['y'][chunk]}, model)
        for chunk in batches(indices)])


def run():
    output = ROOT/'reports/shared-ablation.json'
    if output.exists():
        raise FileExistsError('Ablation output exists; refusing to overwrite')
    protected = {name:sha256(ROOT/name) for name in
                 ('ablate_shared.py', 'fit_factors.py', 'coco_data.py', 'protocol.md')}
    started = time.time()
    result = {'status':'preparing', 'test_scored':False, 'planned_fits':10,
              'scope':'Fully shared spline model; independently refit after deleting each single term. No personal offsets or tuning.',
              'terms':TERMS, 'protected_sha256':protected, 'models':{}, 'contrasts':{},
              'interpretation':'Positive full-minus-reduced dev bits/choice favors retaining the term conditional on all others. Exploratory 95% intervals are not adjusted for nine comparisons. No individual effects are estimated.'}

    def save():
        for name, digest in protected.items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Run source changed: {name}')
        result['elapsed_seconds'] = time.time()-started
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    save()
    try:
        data = build_data(raw=True)
        train = np.flatnonzero(data['split'] == 'fit')
        dev = np.flatnonzero(data['split'] == 'dev')
        design = ChoiceDesign(data, 'spline', train)
        result.update(data_audit=data['audit'], training_choices=len(train),
                      development_choices=len(dev), design=design.specification())
        predictions = {}
        print(f'Data ready: fit={len(train)}, dev={len(dev)}; no personal parameters', flush=True)
        for drop in (None, *TERMS):
            name = drop or 'full'
            result.update(status='running', current=name)
            save()
            model = fit_shared(design, train, drop)
            result['models'][name] = model
            if not model['converged']:
                raise RuntimeError(f'Fit did not converge: {name}')
            predictions[name] = score_dev(design, dev, model)
            model['mean_dev_log2_probability'] = float(predictions[name].mean())
            if drop:
                result['contrasts'][name] = crossed_interval(
                    {'sid':data['sid'][dev], 'group':data['group'][dev]},
                    predictions['full']-predictions[name])
            save()
            print(f'Completed {name}: {len(result["models"])}/10 fits', flush=True)
        np.savez_compressed(ROOT/'reports/shared-ablation-predictions.npz', **predictions,
                            image=data['image'][dev], subject=data['sid'][dev]+1,
                            group=data['group'][dev])
        result['status'] = 'complete'
        result.pop('current', None)
        save()
        print(f'Complete: {time.time()-started:.1f} seconds', flush=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    run()
