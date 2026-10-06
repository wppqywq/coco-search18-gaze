"""Existing conditional-choice model; new model comparisons are not implemented."""
from coco_data import ROOT, MAP_PATH, load_rows, read_frozen, verify_inputs
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

threadpool_limits(2)
torch.set_num_threads(2)
NAMES = ['target', 'saliency', 'distance', 'distance_squared', 'history',
         'x', 'y', 'x_squared', 'y_squared', 'xy', 'momentum', 'horizontal',
         'first_distance', 'first_target']
SPATIAL = [5,6,7,8,9]
PERSONAL = SPATIAL + [0,2]
CONTRAST = helmert(10, full=False).T
xx, yy = np.meshgrid((np.arange(32)+.5)/32, (np.arange(20)+.5)/32)
GRID = np.stack([xx.ravel(), yy.ravel()], axis=1)



def inside(p, bbox):
    x,y,w,h = bbox
    return x <= p[0] < x+w and y <= p[1] < y+h


def transitions(row):
    """Condition on recorded first point, stop at first target entry or invalid sample."""
    points = np.stack([row['X'],row['Y']],axis=1)
    for t in range(len(points)-1):
        # Never stitch across a missing/out-of-display fixation.
        if not np.isfinite(points[:t+2]).all():
            break
        if not ((points[:t+2] >= 0).all() and (points[:t+2] < [1680,1050]).all()):
            break
        if inside(points[t], row['bbox']):
            break
        yield t, points
        if inside(points[t+1], row['bbox']):
            break


def features(row, t, points, maps, history_tau=.5):
    p = points / 1680
    delta = GRID - p[t]
    distance = np.linalg.norm(delta,axis=1)
    direction = delta / np.maximum(distance[:,None],1e-8)
    momentum = np.zeros(len(GRID))
    if t > 0:
        prev = p[t]-p[t-1]
        momentum = direction @ (prev / max(np.linalg.norm(prev),1e-8))
    history = np.zeros(len(GRID))
    # Dwell-time clock: source durations do not include intervening saccades.
    starts = np.r_[0,np.cumsum(row['T'])[:-1]] / 1000
    for j in range(t):
        history += np.exp(-(starts[t]-starts[j])/history_tau) * np.exp(-np.sum((GRID-p[j])**2,axis=1)/(2*.06**2))
    x,y = GRID[:,0]-.5, GRID[:,1]-.3125
    target,sal = maps.reshape(2,-1)
    return np.stack([target,sal,distance,distance**2,history,x,y,x*x,y*y,x*y,
                     momentum,direction[:,0]**2-direction[:,1]**2,
                     distance*(t==0),target*(t==0)],axis=1).astype('float32')


def build_data(history_tau=.5, raw=False):
    verify_inputs()
    split = read_frozen('split.json')['images']
    maps = np.load(MAP_PATH)
    X=[]; ys=[]; subjects=[]; images=[]; splits=[]; near=[]; times=[]; tasks=[]; groups=[]
    distances=[]; revisits=[]
    audit={'selected_trials':0,'trials_without_transitions':0,'transitions':0,'history_tau_dwell_seconds':history_tau}
    for r in load_rows():
        if r['name'] not in split or split[r['name']]['split'] not in ('fit', 'dev'):
            continue
        audit['selected_trials'] += 1
        events=list(transitions(r))
        audit['trials_without_transitions'] += not events
        key=r['name'][:-4]+'_'+r['task'].replace(' ','_')
        for t,points in events:
            X.append(features(r,t,points,maps[key],history_tau))
            col=int(points[t+1,0]/52.5); row=int(points[t+1,1]/52.5)
            ys.append(row*32+col); subjects.append(r['subject']-1)
            images.append(r['name']);splits.append(split[r['name']]['split']);tasks.append(r['task'])
            near.append(np.linalg.norm(points[0]-[840,525])<=100);times.append(t)
            groups.append(split[r['name']]['group'])
            distances.append(np.linalg.norm(GRID-points[t]/1680,axis=1))
            revisits.append(np.any(np.linalg.norm(GRID[:,None,:]-points[None,:t,:]/1680,axis=2)<.06,axis=1))
    maps.close()
    X=np.array(X); y=np.array(ys); sid=np.array(subjects); image=np.array(images); sp=np.array(splits)
    raw_values = X[:, :, [0, 2]].copy() if raw else None
    # Choice-wise centering removes softmax-invariant offsets; scale uses fit choices only.
    X-=X.mean(axis=1,keepdims=True)
    scale=np.maximum(np.sqrt(np.mean(X[sp=='fit']**2,axis=(0,1))),1e-6)
    X/=scale
    audit.update({'transitions':len(y),'by_split':{s:int(sum(sp==s)) for s in ['fit','dev']},
                  'by_subject':{str(s+1):int(sum(sid==s)) for s in range(10)},
                  'feature_scale':dict(zip(NAMES,scale.tolist()))})
    result = {'X':X,'y':y,'sid':sid,'image':image,'split':sp,'near':np.array(near),'t':np.array(times),'task':np.array(tasks),'group':np.array(groups),
            'distances':np.array(distances,dtype='float32'),'revisits':np.array(revisits), 'scale':scale,'audit':audit}
    if raw:
        result['raw'] = raw_values
    return result


def subset(d, mask):
    return {k:v[mask] for k,v in d.items() if isinstance(v,np.ndarray) and k != 'scale'}


def fit(d, personal, penalty=10., drop=(), maxiter=250):
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Penalty must be finite and nonnegative')
    if 'split' not in d or not np.all(d['split'] == 'fit'):
        raise ValueError('Fit requires fitting-only observations with split labels')
    keep=[k for k in range(len(NAMES)) if k not in drop]
    per=[k for k in personal if k in keep]
    per_local=[keep.index(k) for k in per]
    X=torch.as_tensor(d['X'][:,:,keep],dtype=torch.float64)
    y=torch.as_tensor(d['y'],dtype=torch.long)
    c=torch.as_tensor(CONTRAST[d['sid']],dtype=torch.float64)
    n=len(y); nf=len(keep)
    def unpack(theta):
        base=theta[:nf]
        delta=theta[nf:].reshape(9,len(per))
        weights=base.expand(n,nf).clone()
        if per:
            weights[:,per_local]=weights[:,per_local]+c@delta
        return base,delta,weights
    def objective(a):
        theta=torch.tensor(a,dtype=torch.float64,requires_grad=True)
        base,delta,w=unpack(theta)
        logits=torch.einsum('ncf,nf->nc',X,w)
        loss=(torch.nn.functional.cross_entropy(logits,y,reduction='sum')+.5*penalty*(delta**2).sum()+.005*(base**2).sum())/n
        loss.backward()
        return float(loss.detach()),theta.grad.numpy().copy()
    opt=minimize(objective,np.zeros(nf+9*len(per)),method='L-BFGS-B',jac=True,
                 options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':30})
    weights=np.zeros((10,len(NAMES)));weights[:,keep]=opt.x[:nf]
    deviations=np.zeros((10,len(NAMES)))
    if per:
        deviations[:,per]=CONTRAST@opt.x[nf:].reshape(9,len(per))
    weights+=deviations
    return {'weights':weights.tolist(),'deviations':deviations.tolist(),'penalty':penalty,
            'personal_features':[NAMES[k] for k in per],'dropped':[NAMES[k] for k in drop],
            'converged':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),
            'max_abs_gradient':float(np.abs(opt.jac).max()),'objective':float(opt.fun)}


def predict(d, model, return_prob=False):
    w=np.array(model['weights'])
    scores=np.einsum('ncf,nf->nc',d['X'],w[d['sid']],optimize=True)
    scores-=logsumexp(scores,axis=1)[:,None]
    return np.exp(scores) if return_prob else scores[np.arange(len(scores)),d['y']]/np.log(2)


def crossed_interval(d, difference):
    """Independent subject and image-group resampling; exploratory with ten subjects."""
    subjects=np.unique(d['sid']);groups=np.unique(d['group'])
    sums=np.zeros((len(subjects),len(groups)));counts=sums.copy()
    si=np.searchsorted(subjects,d['sid']);gi=np.searchsorted(groups,d['group'])
    np.add.at(sums,(si,gi),difference);np.add.at(counts,(si,gi),1)
    rng=np.random.default_rng(2030);values=[]
    for _ in range(2000):
        sw=rng.multinomial(len(subjects),np.full(len(subjects),1/len(subjects)))
        iw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))
        denominator=sw@counts@iw
        if denominator:
            values.append(float(sw@sums@iw/denominator))
    return {'mean':float(difference.mean()),'crossed_bootstrap_95_percentile_interval':np.quantile(values,[.025,.975]).tolist(),
            'n_subjects':len(subjects),'n_image_groups':len(groups),'replicates':len(values)}


class ChoiceDesign:
    """Fixed fit-only basis with three nonconstant natural-spline columns per factor."""

    def __init__(self, data, form, indices):
        from scipy.interpolate import CubicSpline
        self.data = data
        self.form = form
        self.splines = []
        self.knots = []
        self.keep = [i for i in range(14) if i not in (0, 2, 3)]
        if form == 'spline':
            for column in range(2):
                nodes = np.quantile(data['raw'][indices, :, column], [.05, .35, .65, .95])
                if np.any(np.diff(nodes) <= 0):
                    raise ValueError('Duplicate spline nodes; refusing adaptive basis search')
                self.knots.append(nodes.tolist())
                self.splines.append(CubicSpline(nodes, np.eye(4)[:, :3], bc_type='natural'))
            self.names = [NAMES[i] for i in self.keep] + [f'{name}_spline_{i}' for name in ('target', 'distance') for i in range(3)] + ['personal_target', 'personal_distance']
            self.shared = list(range(17))
            self.spatial = [self.keep.index(i) for i in SPATIAL]
            self.factors = [17, 18]
        elif form == 'existing':
            self.names = list(NAMES); self.shared = list(range(14))
            self.spatial = SPATIAL; self.factors = [0, 2]
        else:
            raise ValueError('Unknown locked response form')
        self.scale = np.ones(len(self.names))
        total = np.zeros(len(self.names)); count = 0
        for chunk in batches(indices):
            values = self.matrix(chunk)
            total += (values * values).sum(axis=(0, 1)); count += len(values)*640
        self.scale = np.maximum(np.sqrt(total/count), 1e-6)

    @staticmethod
    def natural_values(spline, values):
        lower, upper = spline.x[[0, -1]]
        clipped = np.clip(values, lower, upper)
        result = spline(clipped)
        # Natural boundary conditions alone do not imply linear extrapolation in SciPy.
        result += np.minimum(values-lower, 0)[..., None]*spline(lower, 1)
        result += np.maximum(values-upper, 0)[..., None]*spline(upper, 1)
        return result

    def matrix(self, indices):
        values = self.data['X'][indices].astype(float)
        if self.form == 'spline':
            curves = [self.natural_values(spline, self.data['raw'][indices, :, j])
                      for j, spline in enumerate(self.splines)]
            values = np.concatenate([values[:, :, self.keep], *curves, values[:, :, [0, 2]]], axis=2)
        values -= values.mean(axis=1, keepdims=True)
        return values/self.scale

    def specification(self):
        return {'form':self.form, 'names':self.names, 'knots':self.knots, 'scale':self.scale.tolist(),
                'shared_columns':self.shared, 'spatial_columns':self.spatial, 'factor_columns':self.factors}


def batches(indices, size=128):
    for start in range(0, len(indices), size):
        yield indices[start:start+size]


def choice_objective(theta, design, indices, shared, personal, penalty):
    """The same summed conditional logit loss and L2 penalty, with bounded-memory gradients."""
    base = theta[:len(shared)]; delta = theta[len(shared):].reshape(9, len(personal))
    loss = .005*(base @ base) + .5*penalty*np.sum(delta*delta)
    gbase = .01*base; gdelta = penalty*delta
    for chunk in batches(indices):
        values = design.matrix(chunk)
        contrast = CONTRAST[design.data['sid'][chunk]]
        weights = np.zeros((len(chunk), len(design.names)))
        weights[:, shared] = base
        if personal:
            weights[:, personal] += contrast @ delta
        scores = np.einsum('ncf,nf->nc', values, weights, optimize=True)
        normalizer = logsumexp(scores, axis=1)
        chosen = design.data['y'][chunk]
        loss += np.sum(normalizer-scores[np.arange(len(chunk)), chosen])
        residual = np.exp(scores-normalizer[:, None])
        residual[np.arange(len(chunk)), chosen] -= 1
        gradient = np.einsum('nc,ncf->nf', residual, values, optimize=True)
        gbase += gradient[:, shared].sum(axis=0)
        if personal:
            gdelta += contrast.T @ gradient[:, personal]
    return loss/len(indices), np.r_[gbase, gdelta.ravel()]/len(indices)


def fit_comparison(design, indices, personal_factors, penalty, history, label, maxiter=500):
    import time
    if len(indices) == 0 or not np.all(design.data['split'][indices] == 'fit'):
        raise ValueError('Comparison fitting requires nonempty fitting-only records')
    shared = [i for i in design.shared if history or design.names[i] != 'history']
    personal = list(design.spatial) + (design.factors if personal_factors else [])
    started = time.time(); iteration = 0; latest = {}

    def objective(theta):
        loss, gradient = choice_objective(theta, design, indices, shared, personal, penalty)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite fitting objective or gradient')
        latest['loss'] = float(loss)
        return loss, gradient

    def progress(theta):
        nonlocal iteration
        iteration += 1
        if iteration == 1 or iteration % 10 == 0:
            print(f'{label}: iteration={iteration}, objective={latest["loss"]:.9f}, seconds={time.time()-started:.1f}', flush=True)

    opt = minimize(objective, np.zeros(len(shared)+9*len(personal)), method='L-BFGS-B', jac=True,
                   callback=progress, options={'maxiter':maxiter, 'ftol':1e-10, 'gtol':1e-6, 'maxls':30})
    weights = np.zeros((10, len(design.names))); weights[:, shared] = opt.x[:len(shared)]
    weights[:, personal] += CONTRAST @ opt.x[len(shared):].reshape(9, len(personal))
    return {'weights':weights.tolist(), 'penalty':penalty, 'personal_factors':personal_factors,
            'history':history, 'converged':bool(opt.success), 'message':str(opt.message),
            'iterations':int(opt.nit), 'objective':float(opt.fun), 'max_abs_gradient':float(np.max(np.abs(opt.jac))),
            'seconds':time.time()-started}


def score_comparison(design, indices, model):
    """Dev-only probabilities summarized under true histories, not generated scanpaths."""
    if not np.all(design.data['split'][indices] == 'dev'):
        raise ValueError('Comparison scoring is restricted to development records')
    logp = []; observed = []; expected = []
    distance_bins = np.array([0., .05, .1, .2, .4, np.inf])
    centers = np.linalg.norm(GRID-[.5, .3125], axis=1)
    for chunk in batches(indices):
        values = design.matrix(chunk)
        chosen = design.data['y'][chunk]
        inputs = {'X':values, 'sid':design.data['sid'][chunk], 'y':chosen}
        probabilities = predict(inputs, model, True)
        logp.extend(predict(inputs, model))
        distance = design.data['distances'][chunk]
        revisit = design.data['revisits'][chunk]
        candidate_metrics = np.stack([distance, np.broadcast_to(centers, distance.shape), revisit,
            *[(distance >= lower) & (distance < upper) for lower, upper in zip(distance_bins[:-1], distance_bins[1:])]], axis=2)
        observed.extend(candidate_metrics[np.arange(len(chunk)), chosen])
        expected.extend(np.einsum('nc,nck->nk', probabilities, candidate_metrics))
    observed = np.array(observed); expected = np.array(expected)
    people = {}
    for sid in np.unique(design.data['sid'][indices]):
        mask = design.data['sid'][indices] == sid
        people[str(sid+1)] = {'observed':observed[mask].mean(axis=0).tolist(),
                              'predicted':expected[mask].mean(axis=0).tolist()}
    return np.asarray(logp), {'metric_order':['grid_amplitude', 'grid_center_distance', 'grid_revisit',
        'distance_0_.05', 'distance_.05_.1', 'distance_.1_.2', 'distance_.2_.4', 'distance_.4_inf'],
        'by_subject':people, 'interpretation':'Conditional grid-quantized checks; raw continuous EDA is a different measurement'}


def run_comparison():
    import json
    from pathlib import Path
    import time
    from analyze_factors import blocks
    from coco_data import sha256
    output = ROOT/'reports/comparison.json'
    if output.exists():
        raise FileExistsError('Comparison output exists; refusing to overwrite or repeat a run')
    gate_path = ROOT/'reports/input-checks.json'
    gate = json.loads(gate_path.read_text())
    if (not gate['gate']['passed'] or gate['protocol_sha256'] != sha256(ROOT/'protocol.md')
            or gate['source_sha256'] != sha256(ROOT/'analyze_factors.py')):
        raise ValueError('Locked proxy gate has not passed under the current protocol')
    started = time.time()
    protected = {str(p.relative_to(ROOT)):sha256(p) for p in
                 (ROOT/'protocol.md', Path(__file__), ROOT/'coco_data.py', ROOT/'analyze_factors.py', gate_path)}
    result = {'status':'preparing', 'test_scored':False, 'protected_sha256':protected, 'runs':{},
              'planned_fits':72, 'settings':'Two response forms, two personalization levels, three penalties; full/A/B fits; with/without history'}

    def save():
        for name, digest in protected.items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Protected run input changed: {name}')
        result['elapsed_seconds'] = time.time()-started
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    save()
    try:
        data = build_data(raw=True)
        result['data_audit'] = data['audit']
        block = blocks(read_frozen('split.json')['images'])
        block_labels = np.array([block[g] for g in data['group']])
        dev = np.flatnonzero(data['split'] == 'dev')
        predictions = {}
        print(f'Data ready: {len(data["y"])} fit/dev choices, {data["X"].nbytes/1e9:.2f} GB baseline design', flush=True)
        for training_block in ('full', 'A', 'B'):
            train = np.flatnonzero((data['split'] == 'fit') & ((block_labels == training_block) if training_block != 'full' else True))
            for form in ('existing', 'spline'):
                design = ChoiceDesign(data, form, train)
                for history in (True, False):
                    family = f'{training_block}/{form}/history={history}'
                    result['runs'][family] = {'design':design.specification(), 'training_choices':len(train), 'models':{}}
                    for personal in (False, True):
                        name = 'personal' if personal else 'shared_factors'
                        candidates = []
                        result['status'] = 'running'; result['current'] = f'{family}/{name}'
                        save()
                        for penalty in (1., 10., 100.):
                            label = f'{family}/{name}/penalty={penalty}'
                            model = fit_comparison(design, train, personal, penalty, history, label)
                            if not model['converged']:
                                result['failed_fit'] = {'label':label, 'model':model}
                                raise RuntimeError(f'Fit failed to converge: {label}')
                            logp, checks = score_comparison(design, dev, model)
                            candidates.append({'model':model, 'mean_dev_log2_probability':float(logp.mean())})
                            result['runs'][family]['models'][name] = {'candidates':candidates}
                            save()
                        best = max(candidates, key=lambda m:m['mean_dev_log2_probability'])
                        logp, checks = score_comparison(design, dev, best['model'])
                        result['runs'][family]['models'][name].update(selected=best, conditional_checks=checks)
                        predictions[f'{family}/{name}'] = logp
                        save()
        # Selected prediction arrays are development-only and retain row identities.
        np.savez_compressed(ROOT/'reports/comparison-predictions.npz', **predictions,
                            image=data['image'][dev], subject=data['sid'][dev]+1, group=data['group'][dev], block=block_labels[dev])
        result['contrasts'] = {}
        for history in (True, False):
            for form in ('existing', 'spline'):
                for train_block in ('full', 'A', 'B'):
                    family = f'{train_block}/{form}/history={history}'
                    difference = predictions[family+'/personal']-predictions[family+'/shared_factors']
                    for dev_block in ('full', 'A', 'B'):
                        mask = np.ones(len(dev), dtype=bool) if dev_block == 'full' else block_labels[dev] == dev_block
                        result['contrasts'][family+f'/personal_gain/dev={dev_block}'] = crossed_interval(
                            {'sid':data['sid'][dev][mask], 'group':data['group'][dev][mask]}, difference[mask])
            for personal in ('shared_factors', 'personal'):
                key = f'full/response_gain/history={history}/{personal}'
                difference = predictions[f'full/spline/history={history}/{personal}']-predictions[f'full/existing/history={history}/{personal}']
                result['contrasts'][key] = crossed_interval({'sid':data['sid'][dev], 'group':data['group'][dev]}, difference)
        result['decisions'] = {}
        for form in ('existing', 'spline'):
            interval = result['contrasts'][f'full/{form}/history=True/personal_gain/dev=full']['crossed_bootstrap_95_percentile_interval']
            stable = all(result['contrasts'][f'{b}/{form}/history={h}/personal_gain/dev=full']['mean'] > 0
                         for b in ('A', 'B') for h in (True, False))
            sensitivity = result['contrasts'][f'full/{form}/history=False/personal_gain/dev=full']['crossed_bootstrap_95_percentile_interval']
            decision = 'inconclusive'
            if interval[1] < .005 and sensitivity[1] < .005:
                decision = 'prefer_shared_factors_with_personal_spatial_controls'
            elif interval[0] > .005 and sensitivity[0] > .005 and stable:
                decision = 'retain_personal_offsets_within_these_representations'
            result['decisions'][form] = decision
        result['status'] = 'complete'; result.pop('current', None)
        save()
        print(json.dumps({'status':result['status'], 'decisions':result['decisions'], 'seconds':time.time()-started}), flush=True)
    except BaseException as error:
        result['status'] = 'failed'; result['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the fixed fit/dev comparison after the proxy gate passes.')
    parser.add_argument('stage', choices=['compare'])
    parser.parse_args()
    run_comparison()
