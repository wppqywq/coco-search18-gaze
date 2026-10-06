"""Regression checks for leakage, stopping boundaries and probability semantics."""
import ast
import json
import tarfile
import unittest
import os
import tempfile
import importlib.util
import contextlib
import io
from unittest.mock import patch
from pathlib import Path
from coco_data import FROZEN_HASHES, load_rows, read_frozen, verify_inputs
import numpy as np
from fit_factors import ROOT, transitions, features, predict, GRID, NAMES


class FactorChecks(unittest.TestCase):


    def small_real_data(self):
        import fit_factors as model
        split = read_frozen('split.json')['images']
        chosen = []
        counts = {'fit':0, 'dev':0}
        for row in load_rows():
            label = split[row['name']]['split']
            if counts[label] < 12 and list(transitions(row)):
                chosen.append(row); counts[label] += 1
            if min(counts.values()) == 12:
                break
        with patch.object(model, 'load_rows', return_value=chosen):
            return model.build_data(raw=True)

    def test_spline_basis_linear_tails_and_fit_only_nodes(self):
        from fit_factors import ChoiceDesign
        data = self.small_real_data()
        train = np.flatnonzero(data['split'] == 'fit')
        design = ChoiceDesign(data, 'spline', train)
        altered = {**data, 'raw':data['raw'].copy()}
        altered['raw'][data['split'] == 'dev'] += 100
        other = ChoiceDesign(altered, 'spline', train)
        np.testing.assert_array_equal(design.knots, other.knots)
        np.testing.assert_array_equal(design.scale, other.scale)
        self.assertEqual(design.matrix(train[:2]).shape[2], 19)
        for spline in design.splines:
            for edge, sign in ((spline.x[0], -1), (spline.x[-1], 1)):
                y = design.natural_values(spline, edge + sign*np.array([1.,2.,3.]))
                np.testing.assert_allclose(y[0]-2*y[1]+y[2], 0, atol=1e-12)

    def test_batched_gradient_and_existing_objective(self):
        from fit_factors import ChoiceDesign, choice_objective, CONTRAST
        from scipy.optimize._numdiff import approx_derivative
        import torch
        data = self.small_real_data()
        train = np.flatnonzero(data['split'] == 'fit')[:4]
        for form in ('existing', 'spline'):
            design = ChoiceDesign(data, form, train)
            shared = design.shared; personal = design.spatial + design.factors
            theta = np.linspace(-.1,.1,len(shared)+9*len(personal))
            loss, gradient = choice_objective(theta, design, train, shared, personal, 10.)
            numerical = approx_derivative(lambda x:choice_objective(x, design, train, shared, personal, 10.)[0], theta).ravel()
            np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)
            # Independent autograd form matches the retained fitting objective.
            x = torch.tensor(design.matrix(train), dtype=torch.float64)
            parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            base = parameter[:len(shared)]; delta = parameter[len(shared):].reshape(9,len(personal))
            weights = torch.zeros((len(train),len(design.names)),dtype=torch.float64)
            weights[:,shared] = base
            weights[:,personal] += torch.tensor(CONTRAST[data['sid'][train]]) @ delta
            logits = torch.einsum('ncf,nf->nc',x,weights)
            objective = (torch.nn.functional.cross_entropy(logits,torch.tensor(data['y'][train]),reduction='sum') + .005*(base**2).sum()+5*(delta**2).sum())/len(train)
            objective.backward()
            self.assertAlmostEqual(loss,float(objective.detach()),places=12)
            np.testing.assert_allclose(gradient,parameter.grad.numpy(),atol=1e-12)

    def test_proxy_midrank_and_candidate_correlation(self):
        from analyze_factors import cue_contrast, candidate_correlations
        scores = np.array([[0., 1., 2., 3.], [3., 2., 1., 0.], [1., 1., 1., 1.]])
        # A constant cue is tied at midrank 0.5 everywhere.
        expected = np.array([.125, .375, .625, .875]) - (np.array([.875, .625, .375, .125]) + .5)/2
        np.testing.assert_array_equal(cue_contrast(scores, 0), expected)
        x = np.arange(4.)
        correlations = candidate_correlations(np.column_stack([x, -x, 2*x, np.ones(4)]))
        np.testing.assert_allclose(correlations[[0,1,3]], [-1,1,-1])
        self.assertTrue(np.isnan(correlations[[2,4,5]]).all())

    def test_proxy_gate_requires_pre_entry_signal_and_all_rules(self):
        from analyze_factors import gate_summary
        # Synthetic fixtures verify the decision rule, not human-data validity.
        rows = [{'subject':s, 'group':f'g{s}', 'task':f't{t}', 'block':'AB'[s%2],
                 'phase':'pre_entry', 'contrast':.1, 'choices':2}
                for t in range(18) for s in range(1,11)]
        self.assertTrue(gate_summary(rows)['passed'])
        terminal = [{**r, 'phase':'target_entry'} for r in rows]
        self.assertFalse(gate_summary(terminal)['passed'])
        weak = [{**r, 'contrast':-.01 if int(r['task'][1:]) >= 11 else .1} for r in rows]
        self.assertFalse(gate_summary(weak)['checks']['at_least_12_of_18_categories_positive'])

    def test_frozen_inputs_and_fit_dev_boundary(self):
        import coco_data
        result = verify_inputs()
        self.assertFalse(result['test_scored'])
        split = read_frozen('split.json')['images']
        rows = load_rows()
        self.assertEqual({split[r['name']]['split'] for r in rows}, {'fit', 'dev'})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'split.json').write_text('{}')
            with patch.object(coco_data, 'FROZEN', folder):
                with self.assertRaisesRegex(ValueError, 'Frozen manifest changed'):
                    coco_data.load_rows()
            modified = folder/'maps.npz'
            modified.write_bytes(b'invalid')
            with patch.object(coco_data, 'MAP_PATH', modified):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    verify_inputs()

    def test_paths_are_independent_of_working_directory(self):
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(verify_inputs()['frozen_manifests_verified'], 3)
                self.assertTrue(load_rows())
                self.assertEqual(list(Path(directory).iterdir()), [])
        finally:
            os.chdir(cwd)

    def test_model_core_and_real_dev_predictions_match_archive(self):
        import fit_factors as current
        archive = ROOT/'archive/pre-refactor-2026-09-24.tar.gz'
        with tarfile.open(archive) as source:
            old_source = source.extractfile('fit_factors.py').read().decode()
            development = json.load(source.extractfile('reports/full/development.json'))
            for name, expected in FROZEN_HASHES.items():
                original = 'reports/pilot-split.json' if name == 'pilot-split.json' else 'reports/full/'+name
                import hashlib
                self.assertEqual(hashlib.sha256(source.extractfile(original).read()).hexdigest(), expected)
        old_tree = ast.parse(old_source)
        new_tree = ast.parse(Path(current.__file__).read_text())
        names = {'inside', 'transitions', 'features', 'subset', 'predict', 'crossed_interval'}
        old_functions = {n.name:n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name:n for n in new_tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]), name)
        # Only the two input guards precede the unchanged fitting implementation.
        self.assertEqual(
            ast.dump(ast.Module(body=old_functions['fit'].body, type_ignores=[])),
            ast.dump(ast.Module(body=new_functions['fit'].body[2:], type_ignores=[])))
        # Execute only archived numerical functions, never historical training/evaluation entry points.
        reference = dict(vars(current))
        module = ast.Module(body=[old_functions[n] for n in names], type_ignores=[])
        exec(compile(module, '<archived-model-core>', 'exec'), reference)
        split = read_frozen('split.json')['images']
        checked = set()
        with np.load(current.MAP_PATH) as maps:
            for row in load_rows():
                if split[row['name']]['split'] != 'dev' or row['task'] in checked:
                    continue
                events = list(transitions(row))
                if not events:
                    continue
                t, points = events[0]
                key = row['name'][:-4]+'_'+row['task'].replace(' ', '_')
                new = features(row, t, points, maps[key])
                old = reference['features'](row, t, points, maps[key])
                np.testing.assert_array_equal(new, old)
                X = (new-new.mean(axis=0))/np.asarray(development['data_scale'])
                col, line = (points[t+1]/52.5).astype(int)
                data = {'X':X[None], 'sid':np.array([row['subject']-1]), 'y':np.array([line*32+col])}
                for model in development['selected_models'].values():
                    np.testing.assert_array_equal(predict(data, model, True), reference['predict'](data, model, True))
                checked.add(row['task'])
        self.assertEqual(len(checked), 18)

    def test_builder_excludes_test_and_scales_using_fit_only(self):
        import fit_factors as model
        split = read_frozen('split.json')['images']
        selected = {}
        for row in load_rows():
            label = split[row['name']]['split']
            if label not in selected and list(transitions(row)):
                selected[label] = row
            if len(selected) == 2:
                break
        # Deliberate invalid fixture: the test-labelled record must be rejected before feature access.
        test_name = next(n for n, m in split.items() if m['split'] == 'test')
        rows = list(selected.values()) + [{'name':test_name}]
        with patch.object(model, 'load_rows', return_value=rows):
            data = model.build_data()
        self.assertEqual(set(data['split']), {'fit', 'dev'})
        self.assertEqual(set(data['audit']['by_split']), {'fit', 'dev'})
        raw_fit = []
        with np.load(model.MAP_PATH) as maps:
            row = selected['fit']
            key = row['name'][:-4]+'_'+row['task'].replace(' ', '_')
            for t, points in transitions(row):
                raw_fit.append(features(row, t, points, maps[key]))
        raw_fit = np.array(raw_fit)
        raw_fit -= raw_fit.mean(axis=1, keepdims=True)
        expected = np.maximum(np.sqrt(np.mean(raw_fit**2, axis=(0,1))), 1e-6)
        np.testing.assert_array_equal(data['scale'], expected)

    def test_fit_rejects_nontraining_data_and_invalid_penalties(self):
        from fit_factors import fit
        for label in ('dev', 'test', 'previous_test'):
            with self.assertRaisesRegex(ValueError, 'fitting-only'):
                fit({'split':np.array([label])}, [])
        with self.assertRaisesRegex(ValueError, 'fitting-only'):
            fit({}, [])
        for penalty in (-1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                fit({'split':np.array(['fit'])}, [], penalty=penalty)

    def test_gabor_matches_reference_rotation_all_angles_and_phases(self):
        from saliency_models import gbvs
        from saliency_models.helpers.gaborKernelCalculator import getGaborKernel
        params=gbvs.setupParams()['gaborparams']
        sigma=params['stddev'];elongated=sigma*params['elongation']
        radius=int(np.ceil(max(sigma,elongated)*np.sqrt(10)))
        row,col=np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1),indexing='ij')
        for angle in [0,30,45,90,135,170]:
            theta=np.deg2rad(angle)
            major=row*np.cos(theta)-col*np.sin(theta)
            minor=-row*np.sin(theta)-col*np.cos(theta)
            np.testing.assert_allclose(major**2+minor**2,row**2+col**2,atol=1e-10)
            for phase in [0,90]:
                expected=np.cos(2*np.pi/params['filterPeriod']*major+np.deg2rad(phase))*np.exp(-major**2/(2*sigma**2)-minor**2/(2*elongated**2))
                expected-=expected.mean();expected/=np.linalg.norm(expected)
                np.testing.assert_allclose(getGaborKernel(params,angle,phase),expected,atol=1e-12)

    def test_gbvs_degenerate_and_nonconstant_channels(self):
        from saliency_models import gbvs
        from saliency_models.helpers import graphBasedActivation
        cwd=os.getcwd()
        try:
            os.chdir(ROOT/'vendor/gbvs-master')
            blank=gbvs.compute_saliency(np.full((64,80,3),128,dtype=np.uint8))
            self.assertTrue(np.isfinite(blank).all())
            self.assertTrue(np.all(blank==0))
            feature=np.arange(28*32,dtype=float).reshape(28,32)
            out=graphBasedActivation.calculate(feature,4.5)
            self.assertTrue(np.isfinite(out).all())
            self.assertAlmostEqual(float(out.sum()),1,places=6)
        finally:
            os.chdir(cwd)

    def row(self, points):
        return {'X':[p[0] for p in points],'Y':[p[1] for p in points],
                'T':[100]*len(points),'bbox':[1000,500,100,100]}

    def test_first_target_entry_is_included_then_stops(self):
        r=self.row([(840,525),(1000,500),(1200,700)])
        self.assertEqual([t for t,_ in transitions(r)],[0])
        self.assertEqual(list(transitions(self.row([(1000,500),(1200,700)]))),[])

    def test_invalid_samples_do_not_create_new_transitions(self):
        r=self.row([(840,525),(-1,500),(900,600),(950,600)])
        self.assertEqual(list(transitions(r)),[])
        r=self.row([(840,525),(900,600),(1680,500),(950,600)])
        self.assertEqual([t for t,_ in transitions(r)],[0])

    def test_history_excludes_current_and_future_fixations(self):
        r=self.row([(840,525),(900,600),(950,600)])
        points=np.stack([r['X'],r['Y']],axis=1)
        maps=np.zeros((2,20,32))
        first=features(r,0,points,maps)
        self.assertTrue(np.all(first[:,4]==0))
        later=features(r,1,points,maps)
        expected=np.exp(-.1/.5)*np.exp(-np.sum((GRID-points[0]/1680)**2,axis=1)/(2*.06**2))
        np.testing.assert_allclose(later[:,4],expected,rtol=1e-6,atol=1e-8)
        altered=points.copy();altered[2]=[50,50]
        np.testing.assert_array_equal(later,features(r,1,altered,maps))

    def test_softmax_normalization_and_shift_invariance(self):
        rng=np.random.default_rng(2)
        X=rng.normal(size=(3,640,len(NAMES)))
        w=rng.normal(size=(10,len(NAMES)))
        d={'X':X,'sid':np.array([0,1,2]),'y':np.array([0,5,639])}
        p=predict(d,{'weights':w},True)
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
        shifted={**d,'X':X+10}
        np.testing.assert_allclose(p,predict(shifted,{'weights':w},True),atol=1e-12)


@unittest.skipUnless(importlib.util.find_spec('pymc') and importlib.util.find_spec('numba'),
                     'The hierarchical checks require the separate installed PyMC runtime')
class HierarchicalPredictionChecks(unittest.TestCase):

    def test_posterior_probability_mixture_and_opposite_block_isolation(self):
        """Artificial posterior fixtures verify computation, never empirical results."""
        import arviz as az
        import infer_hierarchical as hierarchy
        source = hierarchy.load_data()
        chosen = []
        for subject in range(10):
            for split, block in [('fit', 'A'), ('fit', 'B'), ('dev', 'A'), ('dev', 'B')]:
                candidates = np.flatnonzero((source['sid'] == subject) &
                    (source['split'] == split) & (source['block'] == block))
                self.assertGreater(len(candidates), 0)
                chosen.append(candidates[0])
        chosen = np.array(chosen)
        data = {k:np.asarray(v[chosen]) for k, v in source.items() if k != 'scale'}
        del source
        first = np.zeros((10, 14)); first[:, 0] = -.7
        second = np.zeros((10, 14)); second[:, 0] = 1.1
        draws = np.tile(np.stack([first, second]), (2000, 1, 1)).reshape(4, 1000, 10, 14)
        posterior = az.from_dict(posterior={
            'mu':np.zeros((4, 1000, 14)), 'sigma':np.full((4, 1000, 9), .5),
            'weights':draws, 'deviation':draws[:, :, :, hierarchy.FACTORS+hierarchy.SPATIAL]},
            coords={'feature':np.arange(14), 'subject':np.arange(1, 11),
                    'random_feature':hierarchy.FACTORS+hierarchy.SPATIAL},
            dims={'mu':['feature'], 'sigma':['random_feature'],
                  'weights':['subject', 'feature'], 'deviation':['subject', 'random_feature']})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root/'fixture'; folder.mkdir()
            run = {'status':'complete', 'diagnostics':{'passed':True},
                   'purpose':'numerical_smoke_only', 'chains':4, 'tune':1000,
                   'draws':1000, 'block':'A', 'prior_scale':.5}
            path = folder/'run.json'; path.write_text(json.dumps(run))
            with patch.object(hierarchy, 'OUTPUT', root), patch.object(az, 'from_netcdf', return_value=posterior):
                with self.assertRaisesRegex(ValueError, 'not scientific posterior results'):
                    hierarchy.evaluate(data, 'fixture')
                run['purpose'] = 'full_data_posterior'; path.write_text(json.dumps(run))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = hierarchy.evaluate(data, 'fixture')
                self.assertFalse(result['test_scored'])
                with np.load(folder/'predictions.npz') as output:
                    for label, expected_mask in [('dev', data['split'] == 'dev'),
                        ('opposite_fit_block', (data['split'] == 'fit') & (data['block'] == 'B'))]:
                        indices = np.flatnonzero(expected_mask)
                        np.testing.assert_array_equal(output[f'{label}_indices'], indices)
                        inputs = {'X':data['X'][indices].astype(float),
                                  'sid':data['sid'][indices], 'y':data['y'][indices]}
                        p0 = predict(inputs, {'weights':first}, True)
                        p1 = predict(inputs, {'weights':second}, True)
                        mixture = (p0+p1)/2
                        row = np.arange(len(indices)); y = inputs['y']
                        expected = np.log2(mixture[row, y])
                        mean_log = (np.log2(p0[row, y])+np.log2(p1[row, y]))/2
                        self.assertGreater(np.max(abs(expected-mean_log)), .001)
                        np.testing.assert_allclose(output[f'{label}_logp'], expected, atol=2e-12)
                        per_draw = np.tile([np.log2(p0[row, y]).mean(), np.log2(p1[row, y]).mean()], 2000)
                        np.testing.assert_allclose(output[f'{label}_draw_mean_logp'], per_draw, atol=2e-12)
                        metrics = np.stack([data['distances'][indices],
                            np.broadcast_to(np.linalg.norm(GRID-[.5, .3125], axis=1), mixture.shape),
                            data['revisits'][indices]], axis=2)
                        expected_metrics = np.einsum('nc,nck->nk', mixture, metrics)
                        np.testing.assert_allclose(output[f'{label}_expected'], expected_metrics, atol=2e-12)
                with self.assertRaises(FileExistsError):
                    hierarchy.evaluate(data, 'fixture')


if __name__=='__main__':
    unittest.main()
