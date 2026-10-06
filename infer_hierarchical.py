"""Exact Bayesian COCO factor hierarchy on frozen fit/development observations."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/coco18-mpl')
os.environ.setdefault('PYTENSOR_FLAGS', 'compiledir=/tmp/coco18-hierarchical-pytensor')

from coco_data import ROOT, read_frozen, sha256, verify_inputs
from fit_factors import NAMES, SPATIAL, GRID, build_data, crossed_interval
from analyze_factors import blocks
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from numba import njit, prange, set_num_threads

FACTORS = [0, 1, 2, 4]
CACHE = ROOT/'data/processed/hierarchical-v2'
OUTPUT = ROOT/'reports/hierarchical'
PROTOCOL = ROOT/'reports/hierarchical-protocol.md'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare():
    """Cache unchanged original features; never expose either test set."""
    if (CACHE/'manifest.json').exists():
        raise FileExistsError('Prepared cache exists; refusing overwrite')
    audit = verify_inputs()
    started = time.time()
    data = build_data()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE/'X.npy', data.pop('X'))
    block = blocks(read_frozen('split.json')['images'])
    data['block'] = np.array([block[g] for g in data['group']])
    build_audit = data.pop('audit')
    np.savez_compressed(CACHE/'metadata.npz', **data)
    manifest = {'status':'complete', 'test_scored':False, 'input_audit':audit,
                'data_audit':build_audit, 'seconds':time.time()-started,
                'source_sha256':{n:sha256(ROOT/n) for n in
                                 ('fit_factors.py', 'coco_data.py', 'analyze_factors.py')},
                'files':{n:sha256(CACHE/n) for n in ('X.npy', 'metadata.npz')}}
    write_json(CACHE/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def load_data():
    manifest = json.loads((CACHE/'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Feature source changed: {name}')
    for name, digest in manifest['files'].items():
        if sha256(CACHE/name) != digest:
            raise ValueError(f'Cache changed: {name}')
    with np.load(CACHE/'metadata.npz', allow_pickle=False) as source:
        data = {k:source[k] for k in source.files}
    data['X'] = np.load(CACHE/'X.npy', mmap_mode='r')
    if set(np.unique(data['split'])) != {'fit', 'dev'}:
        raise ValueError('Only frozen fit/dev observations are permitted')
    return data


@njit(parallel=True, cache=True, fastmath=False)
def compiled_likelihood(values, chosen, segments, weights):
    """Stored float32 inputs, strictly float64 scores, probabilities and gradients."""
    totals = np.zeros(len(segments), dtype=np.float64)
    gradients = np.zeros((len(segments), 14), dtype=np.float64)
    for part in prange(len(segments)):
        subject, begin, end = segments[part]
        scores = np.empty(640, dtype=np.float64)
        for i in range(begin, end):
            maximum = -np.inf
            for a in range(640):
                score = 0.
                for k in range(14):
                    score += values[i, a, k]*weights[subject, k]
                scores[a] = score
                maximum = max(maximum, score)
            observed_score = scores[chosen[i]]
            total = 0.
            for a in range(640):
                scores[a] = np.exp(scores[a]-maximum)
                total += scores[a]
            totals[part] += observed_score-maximum-np.log(total)
            for a in range(640):
                residual = (1. if a == chosen[i] else 0.) - scores[a]/total
                for k in range(14):
                    gradients[part, k] += values[i, a, k]*residual
    gradient = np.zeros((10, 14), dtype=np.float64)
    for part in range(len(segments)):
        gradient[segments[part, 0]] += gradients[part]
    return totals.sum(), gradient


class ChoiceLikelihood:
    """Exact subject-factorized likelihood with fixed input precision."""

    def __init__(self, data, indices):
        if not len(indices) or not np.all(data['split'][indices] == 'fit'):
            raise ValueError('Likelihood requires nonempty fitting-only observations')
        set_num_threads(4)
        order = indices[np.argsort(data['sid'][indices], kind='stable')]
        self.values = np.ascontiguousarray(data['X'][order])
        self.chosen = data['y'][order].astype(np.int64)
        segments = []
        for subject in range(10):
            ix = np.flatnonzero(data['sid'][order] == subject)
            if len(ix):
                for start in range(int(ix[0]), int(ix[-1])+1, 256):
                    segments.append((subject, start, min(start+256, int(ix[-1])+1)))
        self.segments = np.asarray(segments, dtype=np.int64)
        self.last = None
        self.evaluations = 0

    def evaluate(self, weights):
        if self.last is not None and np.array_equal(weights, self.last):
            return self.cached
        total, gradient = compiled_likelihood(self.values, self.chosen, self.segments, weights)
        self.last = weights.copy()
        self.cached = (np.asarray(total), gradient)
        self.evaluations += 1
        return self.cached

    def information(self, weights):
        """Exact likelihood Fisher matrix used only to initialize NUTS geometry."""
        matrices = np.zeros((10, 14, 14))
        for subject, begin, end in self.segments:
            values = self.values[begin:end].astype(float)
            scores = values @ weights[subject]
            probabilities = np.exp(scores-logsumexp(scores, axis=1, keepdims=True))
            expected = np.einsum('nc,ncf->nf', probabilities, values)
            matrix = values.reshape(-1, 14)
            matrices[subject] += matrix.T @ (matrix*probabilities.ravel()[:, None]) - expected.T @ expected
        return matrices


def make_model(kernel, personal=True, prior_scale=.5):
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.op import Op

    class Gradient(Op):
        itypes = [pt.dmatrix]
        otypes = [pt.dmatrix]

        def perform(self, node, inputs, outputs):
            outputs[0][0] = kernel.evaluate(inputs[0])[1].copy()

    class LogLikelihood(Op):
        itypes = [pt.dmatrix]
        otypes = [pt.dscalar]

        def perform(self, node, inputs, outputs):
            outputs[0][0] = kernel.evaluate(inputs[0])[0].copy()

        def grad(self, inputs, gradients):
            return [gradients[0]*Gradient()(inputs[0])]

    random = FACTORS+SPATIAL if personal else SPATIAL
    scales = np.array([prior_scale if k in FACTORS else .5 for k in random])
    # Numeric coordinates avoid string-coordinate serialization differences.
    with pm.Model(coords={'feature':np.arange(14), 'subject':np.arange(1, 11),
                          'random_feature':np.asarray(random)}) as model:
        mu = pm.Normal('mu', 0., np.array(1.5, dtype=np.float64), dims='feature')
        sigma = pm.HalfNormal('sigma', scales, dims='random_feature')
        z = pm.Normal('z', 0., 1., dims=('subject', 'random_feature'))
        deviation = pm.Deterministic('deviation', z*sigma,
                                    dims=('subject', 'random_feature'))
        weights = pt.repeat(mu[None, :], 10, axis=0)
        weights = pt.set_subtensor(weights[:, random], weights[:, random]+deviation)
        weights = pm.Deterministic('weights', weights, dims=('subject', 'feature'))
        pm.Potential('choice_likelihood', LogLikelihood()(weights))
    return model


def validate(data):
    """Numerical checks on real inputs; not a human-data inference result."""
    from scipy.optimize._numdiff import approx_derivative
    from fit_factors import predict
    import pymc as pm
    train = np.flatnonzero(data['split'] == 'fit')
    indices = np.concatenate([train[data['sid'][train] == i][:3] for i in range(10)])
    kernel = ChoiceLikelihood(data, indices)
    weights = np.random.default_rng(2509).normal(0, .2, (10, 14))
    value, gradient = kernel.evaluate(weights)
    inputs = {'X':np.asarray(data['X'][indices], dtype=float),
              'sid':data['sid'][indices], 'y':data['y'][indices]}
    reference = predict(inputs, {'weights':weights}, False).sum()*np.log(2)
    numerical = approx_derivative(lambda x:kernel.evaluate(x.reshape(10, 14))[0], weights.ravel()).ravel()
    np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(gradient.ravel(), numerical, atol=2e-7, rtol=2e-5)
    # Verify the curvature used only for sampler preconditioning, independently.
    information = kernel.information(weights)
    hessian = approx_derivative(lambda x:kernel.evaluate(x.reshape(10, 14))[1].ravel(), weights.ravel())
    from scipy.linalg import block_diag
    np.testing.assert_allclose(-hessian, block_diag(*information), atol=2e-7, rtol=2e-5)
    rejected = False
    try:
        ChoiceLikelihood(data, np.flatnonzero(data['split'] == 'dev')[:1])
    except ValueError:
        rejected = True
    assert rejected
    model = make_model(kernel)
    point = model.initial_point()
    point['mu'] = np.linspace(-.2, .3, 14)
    point['z'] = np.linspace(-.4, .4, 90).reshape(10, 9)
    logp = model.compile_logp()
    dlogp = model.compile_dlogp()
    from pymc.blocking import DictToArrayBijection
    flat = DictToArrayBijection.map(point)
    def unpack(x):
        return DictToArrayBijection.rmap(type(flat)(x, flat.point_map_info))
    num = approx_derivative(lambda x:logp(unpack(x)), flat.data).ravel()
    np.testing.assert_allclose(dlogp(point), num, atol=3e-7, rtol=3e-5)
    from scipy.stats import norm, halfnorm
    sigma = np.exp(point['sigma_log__'])
    manual_weights = np.broadcast_to(point['mu'], (10, 14)).copy()
    manual_weights[:, FACTORS+SPATIAL] += point['z']*sigma
    manual_logp = (norm.logpdf(point['mu'], 0, 1.5).sum() +
                   halfnorm.logpdf(sigma, scale=.5).sum() +
                   norm.logpdf(point['z']).sum() + point['sigma_log__'].sum() +
                   kernel.evaluate(manual_weights)[0])
    np.testing.assert_allclose(logp(point), manual_logp, atol=1e-10, rtol=1e-12)
    full = ChoiceLikelihood(data, train)
    fitted = initial_values(True, 1, 20260925)[0]
    fw = np.broadcast_to(fitted['mu'], (10, 14)).copy()
    fw[:, FACTORS+SPATIAL] += fitted['z']*fitted['sigma']
    full_value, full_gradient = full.evaluate(fw)
    reference_full = 0.
    for start in range(0, len(train), 128):
        ix = train[start:start+128]
        reference_full += predict({'X':np.asarray(data['X'][ix], dtype=float),
            'sid':data['sid'][ix], 'y':data['y'][ix]}, {'weights':fw}).sum()*np.log(2)
    np.testing.assert_allclose(full_value, reference_full, atol=1e-7, rtol=1e-12)
    directions = np.random.default_rng(925).normal(size=(3, 10, 14))
    directional_errors = []
    for direction in directions:
        direction /= np.linalg.norm(direction)
        finite_difference = (full.evaluate(fw+1e-5*direction)[0]-full.evaluate(fw-1e-5*direction)[0])/2e-5
        exact = (full_gradient*direction).sum()
        directional_errors.append(float(abs(exact-finite_difference)))
        np.testing.assert_allclose(exact, finite_difference, atol=2e-5, rtol=2e-6)
    result = {'status':'passed', 'kind':'numerical_checks_on_real_fit_inputs',
              'choices':len(indices), 'likelihood_absolute_error':float(abs(value-reference)),
              'likelihood_gradient_max_error':float(abs(gradient.ravel()-numerical).max()),
              'full_posterior_gradient_max_error':float(abs(dlogp(point)-num).max()),
              'independent_full_log_density_error':float(abs(logp(point)-manual_logp)),
              'fisher_max_error':float(abs(-hessian-block_diag(*information)).max()),
              'full_fit_likelihood_absolute_error':float(abs(full_value-reference_full)),
              'full_fit_gradient_directional_errors':directional_errors,
              'dev_fitting_rejected':rejected, 'test_scored':False, 'pymc':pm.__version__,
              'source_sha256':sha256(Path(__file__))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT/'validation.json', result)
    print(json.dumps(result), flush=True)


def diagnostics(idata):
    import arviz as az
    variables = ['mu', 'sigma', 'weights', 'deviation']
    summary = az.summary(idata, var_names=variables, kind='diagnostics')
    stats = idata.sample_stats
    out = {'max_rhat':float(summary['r_hat'].max()),
           'min_bulk_ess':float(summary['ess_bulk'].min()),
           'min_tail_ess':float(summary['ess_tail'].min()),
           'divergences':int(stats['diverging'].sum()),
           'max_depth_hits':int((stats['tree_depth'] >= 10).sum()),
           'bfmi':az.bfmi(idata).tolist(),
           'finite':all(np.isfinite(idata.posterior[n]).all().item() for n in variables)}
    # Use unrounded diagnostic arrays for the actual gate.
    def flatten(dataset):
        return np.concatenate([v.values.ravel() for v in dataset.data_vars.values()])
    rh = flatten(az.rhat(idata, var_names=variables))
    eb = flatten(az.ess(idata, var_names=variables, method='bulk'))
    et = flatten(az.ess(idata, var_names=variables, method='tail'))
    if not all(np.isfinite(v).all() for v in (rh, eb, et)):
        for k, v in list(out.items()):
            if isinstance(v, float) and not np.isfinite(v):
                out[k] = None
        out['passed'] = False
        out['nonfinite_diagnostics'] = True
        return out
    out.update(max_rhat=float(rh.max()), min_bulk_ess=float(eb.min()),
               min_tail_ess=float(et.min()))
    out['passed'] = bool(out['finite'] and out['max_rhat'] <= 1.01 and
        min(out['min_bulk_ess'], out['min_tail_ess']) >= 400 and
        out['divergences'] == 0 and out['max_depth_hits'] == 0 and min(out['bfmi']) >= .3)
    return out


def initial_values(personal, chains, seed):
    """Use a previous fit only as a numerical starting point, never as a prior."""
    previous = json.loads((ROOT/'reports/comparison.json').read_text())
    run = previous['runs']['full/existing/history=True']
    saved = run['models']['personal' if personal else 'shared_factors']['selected']['model']
    # Convert the comparison's additional rescaling back to the cached feature axes.
    weights = np.asarray(saved['weights'])/np.asarray(run['design']['scale'])
    mu = weights.mean(axis=0)
    random = FACTORS+SPATIAL if personal else SPATIAL
    sigma = np.maximum(weights[:, random].std(axis=0), .05)
    rng = np.random.default_rng(seed)
    return [{'mu':mu+rng.normal(0, .02, 14),
             'sigma':sigma*np.exp(rng.normal(0, .15, len(random))),
             'z':(weights[:, random]-mu[random])/sigma + rng.normal(0, .1, (10, len(random)))}
            for _ in range(chains)]


def sampler_geometry(kernel, initial, personal, prior_scale):
    """A Fisher metric accelerates exact NUTS; it does not approximate the posterior."""
    from scipy.linalg import cho_factor, cho_solve
    random = FACTORS+SPATIAL if personal else SPATIAL
    count = len(random); size = 14+11*count
    sigma = initial['sigma']; z = initial['z']
    weights = np.broadcast_to(initial['mu'], (10, 14)).copy()
    weights[:, random] += z*sigma
    scales = np.array([prior_scale if k in FACTORS else .5 for k in random])
    # The metric floor limits initial proposals; it is not an extra prior penalty.
    diagonal = np.r_[np.full(14, 1/1.5**2), np.maximum(2*(sigma/scales)**2, 1.), np.ones(10*count)]
    precision = np.diag(diagonal)
    for subject, information in enumerate(kernel.information(weights)):
        jacobian = np.zeros((14, size)); jacobian[:, :14] = np.eye(14)
        jacobian[random, 14+np.arange(count)] = z[subject]*sigma
        jacobian[random, 14+count+subject*count+np.arange(count)] = sigma
        precision += jacobian.T @ information @ jacobian
    covariance = cho_solve(cho_factor(precision), np.eye(size))
    covariance = (covariance+covariance.T)/2
    mean = np.r_[initial['mu'], np.log(sigma), z.ravel()]
    return mean, covariance


def sample(data, name, block='full', personal=True, prior_scale=.5,
           tune=1000, draws=1000, chains=4, seed=20260925, max_per_subject=None):
    import pymc as pm
    import arviz as az
    folder = OUTPUT/name
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    indices = np.flatnonzero((data['split'] == 'fit') &
                            ((data['block'] == block) if block != 'full' else True))
    if max_per_subject is not None:
        if not name.startswith('smoke'):
            raise ValueError('Subsampled observations are allowed only in named numerical smoke tests')
        indices = np.concatenate([indices[data['sid'][indices] == s][:max_per_subject] for s in range(10)])
    validation = json.loads((OUTPUT/'validation.json').read_text())
    if validation['status'] != 'passed' or validation.get('source_sha256') != sha256(Path(__file__)):
        raise ValueError('Validate the current source before sampling')
    protected = {str(p.relative_to(ROOT)):sha256(p) for p in
                 (Path(__file__), PROTOCOL, CACHE/'manifest.json', ROOT/'reports/comparison.json')}
    run = {'status':'preparing', 'pid':os.getpid(), 'block':block, 'personal':personal,
           'prior_scale':prior_scale, 'tune':tune, 'draws':draws, 'chains':chains,
           'seed':seed, 'choices':len(indices), 'test_scored':False,
           'purpose':('numerical_smoke_only' if max_per_subject else
                      'numerical_benchmark_only' if name.startswith('benchmark') else 'full_data_posterior'),
           'protected_sha256':protected, 'pymc':pm.__version__, 'chain_progress':{}}
    def save():
        for p, digest in protected.items():
            if sha256(ROOT/p) != digest:
                raise ValueError(f'Protected input changed during sampling: {p}')
        run['seconds'] = time.time()-started
        write_json(folder/'run.json', run)
    import importlib.metadata
    import sys
    run['runtime'] = {'python':sys.executable, 'python_version':sys.version,
                      'packages':{n:importlib.metadata.version(n) for n in
                                  ('pymc', 'pytensor', 'arviz', 'numba', 'numpy', 'scipy')}}
    save()
    try:
        kernel = ChoiceLikelihood(data, indices)
        weights = np.zeros((10, 14))
        times = []
        for j in range(3):
            begin = time.time()
            kernel.evaluate(weights+j*.001)
            times.append(time.time()-begin)
        run['likelihood_seconds'] = times
        print(f'{name}: {len(indices)} choices, likelihood seconds={times}', flush=True)
        model = make_model(kernel, personal, prior_scale)
        initials = initial_values(personal, chains, seed)
        mean, covariance = sampler_geometry(kernel, initials[0], personal, prior_scale)
        run['sampler_geometry'] = 'Exact fit-data Fisher matrix plus prior curvature; adaptive full covariance'
        run['initial_covariance_eigenvalues'] = [float(v) for v in np.linalg.eigvalsh(covariance)[[0, -1]]]
        def checkpoint(trace, draw):
            if draw.draw_idx % 25 != 0 and draw.draw_idx+1 != tune+draws:
                return
            run['chain_progress'][str(draw.chain)] = {
                'iteration':draw.draw_idx+1, 'warmup':draw.tuning,
                'diverging':bool(draw.stats[0].get('diverging', False)),
                'leapfrog_steps':float(draw.stats[0].get('tree_size', 0)),
                'seconds':time.time()-started}
            # Checkpoints are diagnostic partial traces, never final posterior evidence.
            arrays = {v:trace.get_values(v)[:draw.draw_idx+1] for v in
                      ('mu', 'sigma', 'weights', 'deviation')}
            path = folder/f'chain-{draw.chain}.partial.npz'
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, **arrays, warmup=tune)
            path.with_suffix('.tmp').replace(path)
            save()
            print(f'{name}: chain={draw.chain}, iteration={draw.draw_idx+1}, warmup={draw.tuning}, seconds={run["seconds"]:.1f}', flush=True)
        run['status'] = 'sampling'; save()
        with model:
            from pymc.step_methods.hmc.quadpotential import QuadPotentialFullAdapt
            from pymc.blocking import DictToArrayBijection
            order = [v[0] for v in DictToArrayBijection.map(model.initial_point()).point_map_info]
            if order != ['mu', 'sigma_log__', 'z']:
                raise RuntimeError(f'Unexpected sampler variable order: {order}')
            potential = QuadPotentialFullAdapt(len(mean), mean, covariance, initial_weight=100)
            step = pm.NUTS(potential=potential, target_accept=.95, max_treedepth=10)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                random_seed=seed, step=step, initvals=initials,
                callback=checkpoint, progressbar=False,
                compute_convergence_checks=False, blas_cores=2)
        idata.to_netcdf(folder/'posterior.nc')
        loaded = az.from_netcdf(folder/'posterior.nc')
        for v in ('mu', 'sigma', 'weights', 'deviation'):
            np.testing.assert_array_equal(idata.posterior[v], loaded.posterior[v])
        loaded.close()
        run['diagnostics'] = diagnostics(idata)
        run['reload_exact'] = True
        run['status'] = 'complete' if run['diagnostics']['passed'] else 'diagnostic_failed'
        run['likelihood_evaluations'] = kernel.evaluations
        save()
        print(json.dumps(run), flush=True)
        return run
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error=f'{type(error).__name__}: {error}')
        save()
        raise


def posterior_summary(posterior):
    def describe(values):
        return {'mean':np.mean(values, axis=0).tolist(),
                'median':np.median(values, axis=0).tolist(),
                'lower_95':np.quantile(values, .025, axis=0).tolist(),
                'upper_95':np.quantile(values, .975, axis=0).tolist()}
    result = {}
    for name in ('mu', 'sigma', 'weights', 'deviation'):
        values = posterior[name].values
        result[name] = describe(values.reshape(-1, *values.shape[2:]))
    result['feature_names'] = NAMES
    random = posterior.coords['random_feature'].values.astype(int)
    result['random_feature_names'] = [NAMES[k] for k in random]
    result['sigma_probability_above_0.05'] = (
        posterior['sigma'].values > .05).mean(axis=(0, 1)).tolist()
    return result


def evaluate(data, name):
    """Integrate every retained draw; prohibit scoring training observations."""
    import arviz as az
    folder = OUTPUT/name
    run = json.loads((folder/'run.json').read_text())
    if run['status'] != 'complete' or not run['diagnostics']['passed']:
        raise ValueError('Posterior diagnostic gate must pass before interpretation')
    if run['purpose'] != 'full_data_posterior' or run['chains'] != 4 or min(run['tune'], run['draws']) < 1000:
        raise ValueError('Numerical smoke/benchmark traces are not scientific posterior results')
    if (folder/'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists; refusing overwrite')
    idata = az.from_netcdf(folder/'posterior.nc')
    weights = idata.posterior['weights'].values.reshape(-1, 10, 14)
    result = {'status':'running', 'test_scored':False, 'posterior_draws':len(weights),
              'posterior':posterior_summary(idata.posterior), 'evaluations':{}}
    from scipy.special import erf
    random = idata.posterior.coords['random_feature'].values.astype(int)
    scales = np.array([run['prior_scale'] if k in FACTORS else .5 for k in random])
    result['prior_sigma_probability_above_0.05'] = (1-erf(.05/scales/np.sqrt(2))).tolist()
    idata.close()
    masks = {'dev':data['split'] == 'dev'}
    if run['block'] != 'full':
        masks['opposite_fit_block'] = ((data['split'] == 'fit') & (data['block'] != run['block']))
    saved = {}
    centers = np.linalg.norm(GRID-[.5, .3125], axis=1)
    for label, mask in masks.items():
        indices = np.flatnonzero(mask)
        if np.any(data['split'][indices] == 'fit') and (run['block'] == 'full' or
                np.any(data['block'][indices] == run['block'])):
            raise ValueError('Training observations cannot enter predictive evaluation')
        logp = np.zeros(len(indices))
        half_logp = np.zeros(len(indices))
        draw_logp = np.zeros(len(weights))
        expected = np.zeros((len(indices), 3)); observed = expected.copy()
        for subject in range(10):
            positions = np.flatnonzero(data['sid'][indices] == subject)
            for start in range(0, len(positions), 16):
                pos = positions[start:start+16]; ix = indices[pos]
                x = np.asarray(data['X'][ix], dtype=float)
                chosen = data['y'][ix]
                summed = np.zeros((len(ix), 640)); first_half = summed.copy()
                for begin in range(0, len(weights), 128):
                    end = min(begin+128, len(weights))
                    score = (x.reshape(-1, 14) @ weights[begin:end, subject].T).reshape(len(ix), 640, -1)
                    score -= logsumexp(score, axis=1, keepdims=True)
                    draw_logp[begin:end] += score[np.arange(len(ix)), chosen].sum(axis=0)/np.log(2)
                    probability = np.exp(score)
                    summed += probability.sum(axis=2)
                    # Alternating retained samples include every chain in the precision check.
                    first_half += probability[:, :, np.arange(begin, end) % 2 == 0].sum(axis=2)
                mean_probability = summed/len(weights)
                logp[pos] = np.log2(mean_probability[np.arange(len(ix)), chosen])
                half_logp[pos] = np.log2(first_half[np.arange(len(ix)), chosen]/((len(weights)+1)//2))
                metrics = np.stack([data['distances'][ix], np.broadcast_to(centers, (len(ix), 640)),
                                    data['revisits'][ix]], axis=2)
                expected[pos] = np.einsum('nc,nck->nk', mean_probability, metrics)
                observed[pos] = metrics[np.arange(len(ix)), chosen]
        checks = {str(s+1):{'observed':observed[data['sid'][indices] == s].mean(axis=0).tolist(),
                            'posterior_predictive_mean':expected[data['sid'][indices] == s].mean(axis=0).tolist()}
                  for s in range(10)}
        result['evaluations'][label] = {
            'choices':len(indices), 'mean_log2_probability':float(logp.mean()),
            'all_minus_alternating_draws_bits':float((logp-half_logp).mean()),
            'metric_names':['grid_amplitude', 'grid_center_distance', 'grid_revisit'],
            'conditional_checks_by_subject':checks,
            'by_block':{b:float(logp[data['block'][indices] == b].mean())
                        for b in np.unique(data['block'][indices])}}
        for key, value in {'logp':logp, 'draw_mean_logp':draw_logp/len(indices),
                           'indices':indices, 'expected':expected, 'observed':observed}.items():
            saved[f'{label}_{key}'] = value
    np.savez_compressed(folder/'predictions.npz', **saved)
    result['status'] = 'complete'
    write_json(folder/'evaluation.json', result)
    print(f'{name}: posterior predictive evaluation complete', flush=True)
    return result


def run_all(data):
    """Resume at completed gated fits; never restart or extend a failed run silently."""
    import gc
    jobs = [('full-personal', 'full', True, .5), ('full-shared', 'full', False, .5),
            ('A-personal', 'A', True, .5), ('A-shared', 'A', False, .5),
            ('B-personal', 'B', True, .5), ('B-shared', 'B', False, .5),
            ('prior-narrow', 'full', True, .25), ('prior-wide', 'full', True, 1.)]
    for j, (name, block, personal, scale) in enumerate(jobs):
        folder = OUTPUT/name
        if folder.exists():
            run = json.loads((folder/'run.json').read_text())
            if run['status'] != 'complete' or not run['diagnostics']['passed']:
                raise RuntimeError(f'Existing run needs diagnosis before proceeding: {name}')
            for p, digest in run['protected_sha256'].items():
                if sha256(ROOT/p) != digest:
                    raise RuntimeError(f'Completed run provenance differs: {p}')
        else:
            run = sample(data, name, block, personal, scale, seed=20260925+j)
        gc.collect()
        if not run['diagnostics']['passed']:
            raise RuntimeError(f'Diagnostic gate failed; stopping dependent analyses: {name}')
        if not (folder/'evaluation.json').exists():
            evaluate(data, name)
    summarize(data)


def summarize(data):
    """Report actual posterior outputs, paired prediction and block stability."""
    import arviz as az
    import matplotlib.pyplot as plt
    result = {'status':'complete', 'test_scored':False, 'comparisons':{}, 'stability':{}}
    for block in ('full', 'A', 'B'):
        with np.load(OUTPUT/f'{block}-personal/predictions.npz') as p, np.load(OUTPUT/f'{block}-shared/predictions.npz') as q:
            np.testing.assert_array_equal(p['dev_indices'], q['dev_indices'])
            ix = p['dev_indices']; difference = p['dev_logp']-q['dev_logp']
            contrast = crossed_interval({'sid':data['sid'][ix], 'group':data['group'][ix]}, difference)
            # Independent posterior draws from independently seeded model runs.
            draws = p['dev_draw_mean_logp']-q['dev_draw_mean_logp']
            contrast['posterior_difference_in_mean_conditional_logp_95'] = np.quantile(draws, [.025, .975]).tolist()
            contrast['posterior_contrast_note'] = 'Draw-wise log likelihood contrast, distinct from log posterior predictive probability and crossed-bootstrap uncertainty.'
            contrast['dev_blocks'] = {b:float(difference[data['block'][ix] == b].mean()) for b in ('A', 'B')}
            if block != 'full':
                np.testing.assert_array_equal(p['opposite_fit_block_indices'], q['opposite_fit_block_indices'])
                ox = p['opposite_fit_block_indices']
                contrast['opposite_fit_block'] = crossed_interval({'sid':data['sid'][ox], 'group':data['group'][ox]},
                    p['opposite_fit_block_logp']-q['opposite_fit_block_logp'])
            result['comparisons'][block] = contrast
    a = az.from_netcdf(OUTPUT/'A-personal/posterior.nc')
    b = az.from_netcdf(OUTPUT/'B-personal/posterior.nc')
    wa = a.posterior['weights'].values.reshape(-1, 10, 14)
    wb = b.posterior['weights'].values.reshape(-1, 10, 14)
    for k in FACTORS:
        correlations = np.array([np.corrcoef(x[:, k], y[:, k])[0, 1] for x, y in zip(wa, wb)])
        result['stability'][NAMES[k]] = {
            'posterior_mean_weight_correlation':float(np.corrcoef(wa[:, :, k].mean(0), wb[:, :, k].mean(0))[0, 1]),
            'posterior_correlation_95':np.quantile(correlations, [.025, .975]).tolist()}
    a.close(); b.close()
    names = ('full-personal', 'full-shared', 'A-personal', 'A-shared', 'B-personal', 'B-shared', 'prior-narrow', 'prior-wide')
    result['runs'] = {n:json.loads((OUTPUT/n/'run.json').read_text()) for n in names}
    if not all(r['status'] == 'complete' and r['diagnostics']['passed'] for r in result['runs'].values()):
        raise RuntimeError('All posterior diagnostic gates must pass')
    ev = {n:json.loads((OUTPUT/n/'evaluation.json').read_text()) for n in names}
    result['prior_sensitivity'] = {n:{'sigma':ev[n]['posterior']['sigma'],
        'probability_above_0.05':ev[n]['posterior']['sigma_probability_above_0.05'],
        'prior_probability_above_0.05':ev[n]['prior_sigma_probability_above_0.05']}
        for n in ('full-personal', 'prior-narrow', 'prior-wide')}
    write_json(OUTPUT/'summary.json', result)
    post = ev['full-personal']['posterior']
    fig, axes = plt.subplots(1, 4, figsize=(13, 5), constrained_layout=True)
    for ax, k in zip(axes, FACTORS):
        med = np.array(post['weights']['median'])[:, k]
        low = np.array(post['weights']['lower_95'])[:, k]
        high = np.array(post['weights']['upper_95'])[:, k]
        ax.errorbar(med, np.arange(1, 11), xerr=[med-low, high-med], fmt='o', capsize=2)
        ax.axvline(post['mu']['median'][k], ls='--', color='gray')
        ax.set(title=NAMES[k].title(), xlabel='Conditional weight', yticks=np.arange(1, 11))
        ax.invert_yaxis()
    axes[0].set_ylabel('Observer')
    fig.suptitle('Posterior medians and 95% intervals; dashed line: population median')
    fig.savefig(OUTPUT/'weights.png', dpi=180); plt.close(fig)
    lines = ['# Hierarchical COCO factor weighting, v2', '',
        'Exploratory fit/dev analysis of ten known observers. Both test sets remain unused in this extension. All eight posterior fits passed the declared numerical diagnostic gate.', '',
        'Coefficients are conditional standardized sensitivities, not percentages or causal contributions. Each scientific factor has its own inferred population scale; there is no fitted between-factor covariance matrix.', '',
        '| Factor | Population mean, median [95% interval] | Between-observer sigma, median [95% interval] | Pr(sigma > 0.05) |',
        '| --- | --- | --- | --- |']
    def interval(item, index):
        return f'{item["median"][index]:.4f} [{item["lower_95"][index]:.4f}, {item["upper_95"][index]:.4f}]'
    for j, k in enumerate(FACTORS):
        lines.append(f'| {NAMES[k]} | {interval(post["mu"], k)} | {interval(post["sigma"], j)} | {post["sigma_probability_above_0.05"][j]:.3f} |')
    lines += ['', 'A continuous positive scale prior does not test the exact zero-variation hypothesis. The 0.05 threshold is descriptive. See summary.json for narrow/wide prior sensitivity and prior probabilities.', '',
              '![Individual conditional weights](weights.png)', '',
              '| Fit images | Personal-minus-shared dev bits/choice | Crossed-bootstrap 95% interval |', '| --- | --- | --- |']
    for block, c in result['comparisons'].items():
        low, high = c['crossed_bootstrap_95_percentile_interval']
        lines.append(f'| {block} | {c["mean"]:.6f} | [{low:.6f}, {high:.6f}] |')
    lines += ['', 'The inherited 0.005 bits/choice margin is a descriptive project convention. These development images have been used in earlier studies; this is not fresh confirmation.', '',
              '| Factor | A/B correlation of posterior mean weights | Posterior correlation 95% interval |', '| --- | --- | --- |']
    for factor, s in result['stability'].items():
        low, high = s['posterior_correlation_95']
        lines.append(f'| {factor} | {s["posterior_mean_weight_correlation"]:.3f} | [{low:.3f}, {high:.3f}] |')
    lines += ['', 'A/B fits share frozen full-fit templates and feature scaling. This is conditional stability across image groups, not cross-session reliability or independent feature reconstruction. Posterior correlation intervals do not include uncertainty from sampling a new observer population.', '',
        'Per-run evaluation.json files contain all ten observers\' weights, deviations, conditional distance/center/revisit checks and posterior-integration precision checks. Per-run posterior.nc files retain every posterior draw, run.json retains the diagnostic gates, and predictions.npz retains auditable row indices and prediction arrays. Numerical validation is reported separately in validation.json and is not empirical evidence of individual differences.', '']
    (OUTPUT/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'sample', 'evaluate', 'run', 'summarize'])
    parser.add_argument('--name', default='benchmark')
    parser.add_argument('--block', choices=['full', 'A', 'B'], default='full')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--prior-scale', type=float, default=.5)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--max-per-subject', type=int)
    args = parser.parse_args()
    threadpool_limits(2)
    if args.stage == 'prepare':
        prepare()
    else:
        data = load_data()
        if args.stage == 'validate':
            validate(data)
        elif args.stage == 'evaluate':
            evaluate(data, args.name)
        elif args.stage == 'run':
            run_all(data)
        elif args.stage == 'summarize':
            summarize(data)
        else:
            sample(data, args.name, args.block, not args.shared, args.prior_scale,
                   args.tune, args.draws, args.chains, max_per_subject=args.max_per_subject)


if __name__ == '__main__':
    main()
