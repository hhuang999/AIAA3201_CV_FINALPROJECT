"""Optional training-only discovery study; never changes frozen deployment.

Ports the original 18 candidates, objectives and selection protocol into
self-contained depthlab code. Execution is an explicit separate CPU study.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cache import atomic_json, load_predictions, sha256
from .calibration import (prepare_training, scene_folds, validate_training_cache,
                          validate_training_partition)
from .data import open_nyu, nyu_depth, scene_names, splits
from .frozen import depth_features
from .metrics import evaluate_image

LEGACY_REVISION = "91091a1b1d27fc2e19e56c2910dd0b3020921f59"
METRICS = ("abs_rel", "rmse", "d1")
KNOTS = np.log(np.array([.3, .7, 1.5, 3., 6., 12.], dtype=np.float64))
CANDIDATES = [dict(name=f"{kind}_{loss}", kind=kind, loss=loss)
              for kind in ("scalar", "affine", "power", "bounded", "spline")
              for loss in ("absrel", "rmse", "balanced")]
CANDIDATES += [dict(name=f"ridge_{lam:g}", kind="ridge", ridge=lam) for lam in (.1, 1., 10.)]

def ridge_fit(features,ratios,lam):
    x=np.asarray(features,dtype=np.float64);y=np.log(np.asarray(ratios,dtype=np.float64))
    if x.ndim!=2 or x.shape[0]!=len(y) or x.shape[1]!=8 or lam<=0:
        raise ValueError('Invalid ridge fit arrays/regularization')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite fit data')
    center=x.mean(0);spread=np.maximum(x.std(0),1e-8);z=(x-center)/spread
    bias=float(y.mean())
    coef=np.linalg.solve(z.T@z/len(z)+lam*np.eye(z.shape[1]),z.T@(y-bias)/len(z))
    return {'kind':'ridge','feature_mean':center.tolist(),'feature_std':spread.tolist(),
            'coefficients':coef.tolist(),'bias':bias,'ridge_lambda':float(lam),'scale_bounds':[.5,1.5]}

def predict_scale(model,features):
    x=np.asarray(features,dtype=np.float64)
    z=(x-np.asarray(model['feature_mean']))/np.asarray(model['feature_std'])
    return np.clip(np.exp(z@np.asarray(model['coefficients'])+model['bias']),*model['scale_bounds'])

def apply_calibration(depth,model,features=None):
    p=np.asarray(depth,dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p<=0):raise ValueError('Raw depth must be finite and positive')
    kind=model['kind']
    if kind=='scalar':out=p*model['scale']
    elif kind=='affine':out=p*model['slope']+model['offset']
    elif kind=='power':out=np.exp(model['log_scale']+model['exponent']*np.log(p))
    elif kind=='bounded':out=p*np.exp(model['log_scale']+model['amplitude']*np.tanh(np.log(p/2.)))
    elif kind=='spline':
        x=np.asarray(model['x_knots']);y=np.asarray(model['y_knots']);lp=np.log(p)
        i=np.clip(np.searchsorted(x,lp,side='right')-1,0,len(x)-2)
        out=np.exp(y[i]+(lp-x[i])*(y[i+1]-y[i])/(x[i+1]-x[i]))
    elif kind=='ridge':out=p*float(predict_scale(model,depth_features(p) if features is None else features))
    else:raise ValueError(f'Unknown calibration kind {kind}')
    return np.clip(out,.001,10.)

def make_torch_map(kind,initial_scale,device):
    import torch
    class Mapping(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n=6 if kind=='spline' else (1 if kind=='scalar' else 2)
            initial=np.zeros(n)
            initial[0]=np.arctanh(np.log(initial_scale)/np.log(2.))
            if kind=='spline':initial[1:]=np.log(3./4.)
            self.theta=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float64,device=device))
            self.register_buffer('knots',torch.tensor(KNOTS,dtype=torch.float64,device=device))
        def decoded(self):
            a=np.log(2.)*self.theta[0].tanh()
            if kind=='spline':
                slopes=.25+1.75*self.theta[1:].sigmoid()
                y=torch.cat([(self.knots[0]+a).reshape(1),self.knots[0]+a+torch.cumsum(slopes*torch.diff(self.knots),dim=0)])
                return a,y
            b=self.theta[1] if len(self.theta)>1 else None
            return a,b
        def forward(self,p):
            a,b=self.decoded()
            if kind=='scalar':out=p*a.exp()
            elif kind=='affine':out=p*a.exp()+2*b.tanh()
            elif kind=='power':out=(a+(1+.5*b.tanh())*p.log()).exp()
            elif kind=='bounded':out=p*(a+.5*b.tanh()*(p/2.).log().tanh()).exp()
            elif kind=='spline':
                x=p.log();i=(torch.bucketize(x.contiguous(),self.knots,right=True)-1).clamp(0,len(self.knots)-2)
                out=(b[i]+(x-self.knots[i])*(b[i+1]-b[i])/(self.knots[i+1]-self.knots[i])).exp()
            else:raise ValueError(kind)
            return out.clamp(.001,10.)
        def regularizer(self):
            if kind!='spline':return self.theta.sum()*0
            _,y=self.decoded();slopes=torch.diff(y)/torch.diff(self.knots)
            return .01*torch.diff(slopes).square().mean()+.001*(slopes-1).square().mean()
        def export(self):
            a,b=self.decoded();a=float(a.detach().cpu())
            if kind=='scalar':return {'kind':kind,'scale':float(np.exp(a))}
            if kind=='affine':return {'kind':kind,'slope':float(np.exp(a)),'offset':float((2*b.tanh()).detach().cpu())}
            if kind=='power':return {'kind':kind,'log_scale':a,'exponent':float((1+.5*b.tanh()).detach().cpu())}
            if kind=='bounded':return {'kind':kind,'log_scale':a,'amplitude':float((.5*b.tanh()).detach().cpu())}
            return {'kind':kind,'x_knots':KNOTS.tolist(),'y_knots':b.detach().cpu().tolist()}
    return Mapping()

def loss_components(pred,gt):
    rel=(pred-gt)/gt
    ar=(rel.square()+1e-6).sqrt().sub(.001).mean()
    rmse=((pred-gt).square().mean(-1)+1e-12).sqrt().mean()
    delta=torch.nn.functional.softplus(((pred.log()-gt.log()).abs()-np.log(1.25))/.02).mean()*.02
    return torch.stack([ar,rmse,delta])

def fit_map(kind,loss,p,g,initial_scale):
    model=make_torch_map(kind,initial_scale,p.device)
    with torch.no_grad():reference=loss_components((p*initial_scale).clamp(.001,10.),g).clamp_min(1e-9)
    opt=torch.optim.LBFGS(model.parameters(),max_iter=80,max_eval=120,history_size=20,
                         line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-11)
    history=[]
    def closure():
        opt.zero_grad(set_to_none=True)
        terms=loss_components(model(p),g)/reference
        value=(terms[0] if loss=='absrel' else terms[1] if loss=='rmse' else terms.sum())+model.regularizer()
        if not bool(torch.isfinite(value)):raise ArithmeticError('Nonfinite calibration objective')
        value.backward();history.append(float(value.detach()));return value
    opt.step(closure)
    with torch.no_grad():final=loss_components(model(p),g).cpu().numpy()
    return model.export(),{'closure_calls':len(history),'objective_first':history[0],
        'objective_last':history[-1],'reference_components':reference.cpu().tolist(),
        'final_components':final.tolist(),'history':history}

def huber_log_scale(image_ratios, tuning=1.345):
    """Huber location of log per-image median ratios; no pixel/test fitting.

    The residual scale is 1.4826 * MAD, fixed before iteratively solving the
    location equation. A zero MAD falls back to the median. Every image has
    equal initial weight. The result is still ONE positive scalar.
    """
    ratios = np.asarray(image_ratios, dtype=np.float64)
    if ratios.ndim != 1 or not len(ratios) or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError('Expected nonempty finite positive image ratios')
    if not np.isfinite(tuning) or tuning <= 0:
        raise ValueError('Expected positive finite Huber tuning')
    values = np.log(ratios)
    center = float(np.median(values))
    width = tuning * 1.4826 * float(np.median(np.abs(values - center)))
    if width <= np.finfo(float).eps:
        return float(np.median(ratios))
    for _ in range(200):
        residual = np.abs(values - center)
        weights = np.minimum(1., width / np.maximum(residual, np.finfo(float).tiny))
        updated = float(np.average(values, weights=weights))
        if abs(updated - center) < 1e-12:
            return float(np.exp(updated))
        center = updated
    raise ArithmeticError('Huber location failed to converge')

def choose_primary(table):
    positive=[r for r in table if r.get('joint_positive_means')]
    confident=[r for r in positive if r['joint_adjusted_positive']]
    if confident:return min(confident,key=lambda r:(r['abs_rel'],r['rmse'],-r['d1'],r['method']))['method'],'simultaneous_confidence_gate'
    if positive:return max(positive,key=lambda r:(r['minimum_normalized_gain'],-r['abs_rel']))['method'],'provisional_mean_only'
    return None,'no_joint_positive_candidate'

def fresh_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Use a fresh study directory; existing evidence cannot be overwritten")
    path.mkdir(parents=True)
    return path


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty study evidence")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TrainingData:
    """Verified original-ID cache and training-only preparation; GT stays on disk."""
    def __init__(self, data_root, predictions):
        self.data_root, self.prediction_root = Path(data_root), Path(predictions)

    def __enter__(self):
        partition = splits(self.data_root)
        self.ids = validate_training_partition(partition)
        self.predictions = load_predictions(self.prediction_root, required_ids=self.ids,
                                           expected_known_intrinsics=False, require_complete=True)
        validate_training_cache(self.data_root, self.predictions)
        self.handle = open_nyu(self.data_root)
        try:
            all_scenes = scene_names(self.handle)
            validate_training_partition(partition, all_scenes)
            self.scenes = all_scenes[self.ids]
            self.fold_positions = scene_folds(self.scenes)
            self.assignment = np.empty(len(self.ids), dtype=int)
            for fold, positions in enumerate(self.fold_positions):
                self.assignment[positions] = fold
            self.features, self.ratios, self.samples, self.targets = prepare_training(
                self.handle, self.predictions, self.ids)
        except Exception:
            self.handle.close()
            raise
        return self

    def __exit__(self, *exc):
        self.handle.close()

    def fingerprint(self, module):
        return {"created_utc": datetime.now(timezone.utc).isoformat(),
                "legacy_revision": LEGACY_REVISION, "scope": "official795_training_only",
                "historical_test_exposure": True, "test_read": False,
                "ids": self.ids.tolist(), "scenes": self.scenes.tolist(),
                "folds": self.assignment.tolist(), "fold_seed": 20261031,
                "sample_seed": 20261105, "samples_per_image": 4096,
                "dataset_sha256": sha256(self.data_root / "nyu_depth_v2_labeled.mat"),
                "split_sha256": sha256(self.data_root / "splits.mat"),
                "cache_manifest_sha256": sha256(self.prediction_root / "manifest.json"),
                "source_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                  (module, "calibration_study.py", "calibration.py", "frozen.py", "cache.py", "data.py", "metrics.py")},
                "numerical_runtime": {"device": "CPU", "dtype": "float64", "torch": torch.__version__, "numpy": np.__version__}}


def fit_fold_models(data, positions):
    positions = np.asarray(positions, dtype=int)
    scale = float(np.median(data.ratios[positions]))
    models = {"raw": {"kind": "scalar", "scale": 1.},
              "median": {"kind": "scalar", "scale": scale},
              "huber": {"kind": "scalar", "scale": huber_log_scale(data.ratios[positions])}}
    details = {}
    p = torch.tensor(data.samples[positions], dtype=torch.float64)
    g = torch.tensor(data.targets[positions], dtype=torch.float64)
    for spec in CANDIDATES:
        started = time.perf_counter()
        if spec["kind"] == "ridge":
            model = ridge_fit(data.features[positions], data.ratios[positions], spec["ridge"])
            trace = {}
        else:
            model, trace = fit_map(spec["kind"], spec["loss"], p, g, scale)
        models[spec["name"]] = model
        details[spec["name"]] = {"seconds": time.perf_counter() - started, "optimization": trace}
    return models, details


def score_fold(data, positions, models, fold, extra=None):
    rows = []
    for pos in positions:
        raw = np.asarray(data.predictions[int(data.ids[pos])], dtype=np.float64)
        gt = nyu_depth(data.handle, int(data.ids[pos]))
        for name, model in models.items():
            prediction = apply_calibration(raw, model, features=data.features[pos])
            row = {"dataset_index": int(data.ids[pos]), "scene_id": str(data.scenes[pos]),
                   "fold": int(fold), "method": name, **dict(zip(METRICS, evaluate_image(prediction, gt)))}
            if extra is not None:
                row.update(extra)
            rows.append(row)
    return rows


def select_from_oof(values, groups, repeats=20000, seed=20261105):
    """Original conditional scene bootstrap and discovery selection, no test."""
    names = ["raw", "median", "huber"] + [c["name"] for c in CANDIDATES]
    groups = np.asarray(groups)
    if set(values) != set(names) or not len(groups) or repeats < 1:
        raise ValueError("Expected all 21 paired training-only methods")
    values = {name: np.asarray(values[name], dtype=np.float64) for name in names}
    if any(a.shape != (len(groups), 3) or not np.isfinite(a).all() for a in values.values()):
        raise ValueError("Expected complete finite paired OOF scores")
    reference = values["median"]
    candidate_names = [c["name"] for c in CANDIDATES]
    gains = np.stack([reference - values[n] for n in candidate_names], axis=1)
    gains[:, :, 2] *= -1
    flat = gains.reshape(len(groups), -1)
    unique = np.unique(groups)
    sums = np.array([flat[groups == scene].sum(0) for scene in unique])
    counts = np.array([(groups == scene).sum() for scene in unique])
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, flat.shape[1]))
    for start in range(0, repeats, 200):
        draw = rng.integers(0, len(unique), (min(200, repeats - start), len(unique)))
        boot[start:start + len(draw)] = sums[draw].sum(1) / counts[draw].sum(1, keepdims=True)
    percentile = 100 * .05 / (len(candidate_names) * 3)
    lower = np.percentile(boot, percentile, axis=0).reshape(len(candidate_names), 3)
    ci = np.percentile(boot, [2.5, 97.5], axis=0).reshape(2, len(candidate_names), 3)
    ref = reference.mean(0)
    denominator = np.array([ref[0], ref[1], 1 - ref[2]])
    if np.any(denominator <= 0):
        raise ValueError("Normalized selection gains require a nonperfect positive reference")
    table, intervals = [], {}
    tail_count = int(np.ceil(.1 * len(groups)))
    for name in names:
        means = values[name].mean(0)
        row = {"method": name, "images": len(groups), **dict(zip(METRICS, means.tolist())),
               "worst_decile_absrel": float(np.sort(values[name][:, 0])[-tail_count:].mean()),
               "joint_positive_means": False, "joint_adjusted_positive": False, "minimum_normalized_gain": 0.}
        if name in candidate_names:
            k = candidate_names.index(name)
            gain = gains[:, k].mean(0)
            row.update(joint_positive_means=bool(np.all(gain > 1e-6)),
                       joint_adjusted_positive=bool(np.all(lower[k] > 0)),
                       minimum_normalized_gain=float(np.min(gain / denominator)))
            intervals[name] = {"mean_gain": dict(zip(METRICS, gain.tolist())),
                               "ordinary95CI": {n: ci[:, k, j].tolist() for j, n in enumerate(METRICS)},
                               "bonferroni_one_sided_lower": dict(zip(METRICS, lower[k].tolist()))}
        table.append(row)
    primary, certainty = choose_primary(table)
    by_name = {row["method"]: row for row in table}
    ablations = []
    for kind in ("scalar", "affine", "power", "bounded", "spline", "ridge"):
        family = [c["name"] for c in CANDIDATES if c["kind"] == kind and by_name[c["name"]]["joint_positive_means"]]
        if family:
            ablations.append(min(family, key=lambda name: by_name[name]["abs_rel"]))
    if primary is not None and primary not in ablations:
        ablations.append(primary)
    selection = {"primary": primary, "certainty": certainty, "frozen_ablation_methods": ablations,
                 "images": len(groups), "scenes": len(unique), "folds": 5, "candidates": 18,
                 "bootstrap_repeats": repeats, "bootstrap_seed": seed, "tail_percentile": percentile,
                 "test_read": False, "historical_test_exposure": True,
                 "deployment_changed": False,
                 "interpretation": "Optional training-only discovery recommendation; does not replace any deployed frozen model. Conditional intervals do not refit models."}
    return table, intervals, selection


def run(args):
    torch.set_num_threads(2)
    output = fresh_output(args.output)
    with TrainingData(args.data_root, args.predictions) as data:
        protocol = data.fingerprint("calibration_study.py")
        protocol.update(candidates=CANDIDATES, smoke=args.smoke_only,
                        losses={"absrel": "smooth absolute relative error epsilon .001",
                                "rmse": "equal-image RMSE normalized by median reference",
                                "balanced": "sum of normalized smooth AbsRel, RMSE and soft delta1 error"},
                        spline_regularization=".01 adjacent slope difference squared + .001 slope departure from one squared",
                        optimizer={"name": "LBFGS", "max_iter": 80, "max_eval": 120, "history_size": 20,
                                   "tolerance_grad": 1e-9, "tolerance_change": 1e-11, "line_search_fn": "strong_wolfe"},
                        model_selection="Original joint-mean/confidence discovery rule; frozen deployment unchanged")
        atomic_json(output / "protocol.json", protocol)
        all_rows = []
        for fold in ([0] if args.smoke_only else range(5)):
            train, val = np.flatnonzero(data.assignment != fold), np.flatnonzero(data.assignment == fold)
            if args.smoke_only:
                train, val = train[:12], val[:2]
            if set(data.scenes[train]) & set(data.scenes[val]):
                raise ValueError("Scene leakage")
            models, details = fit_fold_models(data, train)
            rows = score_fold(data, val, models, fold)
            destination = output / f"fold{fold}"
            destination.mkdir()
            atomic_json(destination / "models.json", models)
            atomic_json(destination / "fit_details.json", details)
            write_csv(destination / "per_image.csv", rows)
            atomic_json(destination / "completion.json", {"complete": True, "smoke": args.smoke_only,
                        "fold": fold, "fit_ids": data.ids[train].tolist(), "validation_ids": data.ids[val].tolist(),
                        "protocol_sha256": sha256(output / "protocol.json"), "test_read": False})
            all_rows.extend(rows)
            print("DISCOVERY_FOLD_COMPLETE", fold, len(train), len(val), flush=True)
        if args.smoke_only:
            atomic_json(output / "completion.json", {"complete": True, "smoke": True, "test_read": False, "selection_performed": False})
            return
        position = {int(index): pos for pos, index in enumerate(data.ids)}
        values = {name: np.full((len(data.ids), 3), np.nan) for name in models}
        for row in all_rows:
            pos = position[row["dataset_index"]]
            if np.isfinite(values[row["method"]][pos]).any():
                raise ValueError("Duplicate held-out identity")
            values[row["method"]][pos] = [row[k] for k in METRICS]
        table, intervals, selection = select_from_oof(values, data.scenes)
        write_csv(output / "oof_per_image.csv", all_rows)
        write_csv(output / "cv_methods.csv", sorted(table, key=lambda row: row["abs_rel"]))
        atomic_json(output / "comparisons.json", intervals)
        selection.update(complete=True, smoke=False, protocol_sha256=sha256(output / "protocol.json"),
                         fold_evidence=[{name: sha256(output / f"fold{fold}" / name) for name in
                                         ("completion.json", "per_image.csv", "models.json")} for fold in range(5)])
        atomic_json(output / "selection.json", selection)
        atomic_json(output / "completion.json", {"complete": True, "smoke": False, "test_read": False,
                    "selection_sha256": sha256(output / "selection.json"), "deployment_changed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
