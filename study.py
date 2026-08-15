"""Explicit paired simulation protocol for the reduced graphene model.

The protocol compares a nonlinear reduced-order model with a mechanically
linearized ablation that shares the same seeded frequencies, input mask,
damping, coupling, electrostatics and optical readout.  A digital delay line is
reported as a separate baseline.  All preprocessing is fit on the chronological
training prefix; no test sample calibrates the scaler or readout.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler

import tasks
from physical_model import GrapheneOscillatorReservoir, _require_real_array

PROTOCOL_ID = "graphene-reduced-sim-v1"
SPLIT_FRACTION = 0.65
DELAY_ORDER = 12
BOOTSTRAP_BASE_SEED = 20260814
BOOTSTRAP_SUMMARY_SEEDS = {
    "narma_nrmse.nonlinear": 20260814,
    "narma_nrmse.linear_mechanics": 20260815,
    "narma_nrmse.delay_line": 20260816,
    "parity_accuracy.nonlinear": 20260817,
    "parity_accuracy.linear_mechanics": 20260818,
    "parity_accuracy.delay_line": 20260819,
    "paired_effects.narma_nonlinear_minus_linear": 20260820,
    "paired_effects.narma_nonlinear_minus_delay": 20260821,
    "paired_effects.parity_nonlinear_minus_linear": 20260822,
    "paired_effects.parity_nonlinear_minus_delay": 20260823,
}
MODEL_CONFIG = MappingProxyType({
    "f_lo": 10e6,
    "f_hi": 40e6,
    "Q": 20.0,
    "m_eff": 2e-18,
    "area": 1.77e-12,
    "gap": 250e-9,
    "oxide": 285e-9,
    "V_dc": 1.5,
    "V_ac": 0.15,
    "duffing_onset": 5e-9,
    "nonlinear_damping_ratio": 0.2,
    "coupling_fraction": 0.03,
    "wavelength": 632.8e-9,
    "dt": 0.25e-9,
    "steps_per_input": 8,
    "contact_margin_fraction": 0.05,
})


@dataclass(frozen=True)
class TrialSpec:
    index: int
    model_seed: int
    narma_seed: int
    parity_seed: int


def _require_integer(name: str, value, *, minimum: int) -> int:
    """Normalize integral scalars without accepting bools or numeric coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean or coerced value")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _require_real(name: str, value, *, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean or text")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return normalized


def protocol_trials(count: int = 12) -> list[TrialSpec]:
    count = _require_integer("count", count, minimum=1)
    return [
        TrialSpec(
            index=index,
            model_seed=4000 + index,
            narma_seed=5000 + index,
            parity_seed=6000 + index,
        )
        for index in range(count)
    ]


def _validate_trials(trials: list[TrialSpec]) -> list[TrialSpec]:
    if not trials or any(not isinstance(trial, TrialSpec) for trial in trials):
        raise ValueError("trials must be a non-empty list of TrialSpec values")
    normalized = [
        TrialSpec(
            index=_require_integer("trial index", trial.index, minimum=0),
            model_seed=_require_integer("model seed", trial.model_seed, minimum=0),
            narma_seed=_require_integer("NARMA seed", trial.narma_seed, minimum=0),
            parity_seed=_require_integer("parity seed", trial.parity_seed, minimum=0),
        )
        for trial in trials
    ]
    indices = [trial.index for trial in normalized]
    role_seeds = [
        seed
        for trial in normalized
        for seed in (trial.model_seed, trial.narma_seed, trial.parity_seed)
    ]
    if len(set(indices)) != len(normalized) or len(set(role_seeds)) != 3 * len(normalized):
        raise ValueError("trial indices and role seeds must be unique and disjoint")
    return normalized


def make_paired_models(model_seed: int, *, n: int = 16):
    model_seed = _require_integer("model_seed", model_seed, minimum=0)
    n = _require_integer("n", n, minimum=2)
    common = {**MODEL_CONFIG, "n": n, "seed": model_seed}
    return (
        GrapheneOscillatorReservoir(**common, nonlinear=True),
        GrapheneOscillatorReservoir(**common, nonlinear=False),
    )


def _nrmse(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    variance = float(np.var(truth))
    if variance <= 0.0:
        raise ValueError("NRMSE is undefined for a constant target")
    return float(np.sqrt(np.mean((truth - prediction) ** 2) / variance))


def _validate_split(X, y, split: int):
    features = _require_real_array("X", X)
    target = _require_real_array("y", y)
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("X must be 2D and aligned with a 1D target")
    if not 2 <= split <= len(features) - 2:
        raise ValueError("split must leave at least two train and two test rows")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("features and target must be finite")
    return features, target


def fit_regression_audited(X, y, *, split: int, alpha: float = 1e-3) -> dict:
    split = _require_integer("split", split, minimum=2)
    alpha = _require_real("alpha", alpha, minimum=0.0)
    features, target = _validate_split(X, y, split)
    scaler = StandardScaler().fit(features[:split])
    train = scaler.transform(features[:split])
    test = scaler.transform(features[split:])
    model = Ridge(alpha=alpha).fit(train, target[:split])
    prediction = model.predict(test)
    return {
        "nrmse": _nrmse(target[split:], prediction),
        "train_rows": int(split),
        "test_rows": int(len(features) - split),
        "train_feature_mean": [float(value) for value in scaler.mean_],
    }


def fit_classification_audited(X, y, *, split: int, alpha: float = 1.0) -> dict:
    split = _require_integer("split", split, minimum=2)
    alpha = _require_real("alpha", alpha, minimum=0.0)
    features, target = _validate_split(X, y, split)
    scaler = StandardScaler().fit(features[:split])
    train = scaler.transform(features[:split])
    test = scaler.transform(features[split:])
    model = RidgeClassifier(alpha=alpha).fit(train, target[:split])
    prediction = model.predict(test)
    return {
        "accuracy": float(np.mean(prediction == target[split:])),
        "train_rows": int(split),
        "test_rows": int(len(features) - split),
        "train_feature_mean": [float(value) for value in scaler.mean_],
    }


def _evaluate_trial(
    trial: TrialSpec,
    *,
    length: int,
    washout: int,
    n_modes: int,
) -> dict:
    length = _require_integer("length", length, minimum=1)
    washout = _require_integer("washout", washout, minimum=0)
    n_modes = _require_integer("n_modes", n_modes, minimum=2)
    if washout < DELAY_ORDER or length - washout < 40:
        raise ValueError(
            "protocol requires washout >= delay order and enough scored rows"
        )

    narma_input, narma_target = tasks.narma10(length, seed=trial.narma_seed)
    narma_drive = 4.0 * narma_input - 1.0
    nonlinear, linear = make_paired_models(trial.model_seed, n=n_modes)
    model_frequencies_hz = [float(value) for value in nonlinear.f0]
    nonlinear_states = nonlinear.run(narma_drive, washout=washout)
    linear_states = linear.run(narma_drive, washout=washout)
    minimum_air_gap_fraction = {
        "narma_nonlinear": float(nonlinear.minimum_air_gap / nonlinear.gap),
        "narma_linear": float(linear.minimum_air_gap / linear.gap),
    }
    delay_states = tasks.delay_embed(narma_drive, DELAY_ORDER)[washout:]
    narma_scored = narma_target[washout:]
    split = int(SPLIT_FRACTION * len(narma_scored))
    narma = {
        "nonlinear": fit_regression_audited(
            nonlinear_states, narma_scored, split=split
        )["nrmse"],
        "linear_mechanics": fit_regression_audited(
            linear_states, narma_scored, split=split
        )["nrmse"],
        "delay_line": fit_regression_audited(delay_states, narma_scored, split=split)[
            "nrmse"
        ],
    }

    parity_drive, parity_target = tasks.parity_stream(
        length,
        seed=trial.parity_seed,
        order=3,
    )
    nonlinear, linear = make_paired_models(trial.model_seed, n=n_modes)
    nonlinear_states = nonlinear.run(parity_drive, washout=washout)
    linear_states = linear.run(parity_drive, washout=washout)
    minimum_air_gap_fraction.update(
        {
            "parity_nonlinear": float(nonlinear.minimum_air_gap / nonlinear.gap),
            "parity_linear": float(linear.minimum_air_gap / linear.gap),
        }
    )
    delay_states = tasks.delay_embed(parity_drive, DELAY_ORDER)[washout:]
    parity_scored = parity_target[washout:]
    split = int(SPLIT_FRACTION * len(parity_scored))
    parity = {
        "nonlinear": fit_classification_audited(
            nonlinear_states, parity_scored, split=split
        )["accuracy"],
        "linear_mechanics": fit_classification_audited(
            linear_states, parity_scored, split=split
        )["accuracy"],
        "delay_line": fit_classification_audited(
            delay_states, parity_scored, split=split
        )["accuracy"],
    }

    return {
        **asdict(trial),
        "model_frequencies_hz": model_frequencies_hz,
        "narma_nrmse": narma,
        "parity_accuracy": parity,
        "contact_events": 0,
        "minimum_air_gap_fraction": minimum_air_gap_fraction,
    }


def _summary_stats(values, *, bootstrap_samples: int, seed: int) -> dict:
    bootstrap_samples = _require_integer(
        "bootstrap_samples", bootstrap_samples, minimum=1
    )
    seed = _require_integer("bootstrap seed", seed, minimum=0)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("summary values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstrap_samples, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "bootstrap_mean_ci95": [float(low), float(high)],
    }


def _summarize(records: list[dict], *, bootstrap_samples: int) -> dict:
    conditions = ("nonlinear", "linear_mechanics", "delay_line")
    summary = {"narma_nrmse": {}, "parity_accuracy": {}, "paired_effects": {}}
    for metric in ("narma_nrmse", "parity_accuracy"):
        for condition in conditions:
            values = [record[metric][condition] for record in records]
            summary[metric][condition] = _summary_stats(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=BOOTSTRAP_SUMMARY_SEEDS[f"{metric}.{condition}"],
            )

    effects = {
        "narma_nonlinear_minus_linear": [
            record["narma_nrmse"]["nonlinear"]
            - record["narma_nrmse"]["linear_mechanics"]
            for record in records
        ],
        "narma_nonlinear_minus_delay": [
            record["narma_nrmse"]["nonlinear"] - record["narma_nrmse"]["delay_line"]
            for record in records
        ],
        "parity_nonlinear_minus_linear": [
            record["parity_accuracy"]["nonlinear"]
            - record["parity_accuracy"]["linear_mechanics"]
            for record in records
        ],
        "parity_nonlinear_minus_delay": [
            record["parity_accuracy"]["nonlinear"]
            - record["parity_accuracy"]["delay_line"]
            for record in records
        ],
    }
    for name, values in effects.items():
        summary["paired_effects"][name] = _summary_stats(
            values,
            bootstrap_samples=bootstrap_samples,
            seed=BOOTSTRAP_SUMMARY_SEEDS[f"paired_effects.{name}"],
        )
    return summary


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def run_protocol(
    *,
    trials: list[TrialSpec] | None = None,
    length: int = 1200,
    washout: int = 200,
    n_modes: int = 16,
    bootstrap_samples: int = 10_000,
) -> dict:
    length = _require_integer("length", length, minimum=1)
    washout = _require_integer("washout", washout, minimum=0)
    n_modes = _require_integer("n_modes", n_modes, minimum=2)
    bootstrap_samples = _require_integer(
        "bootstrap_samples", bootstrap_samples, minimum=100
    )
    selected = _validate_trials(protocol_trials() if trials is None else list(trials))
    records = [
        _evaluate_trial(
            trial,
            length=length,
            washout=washout,
            n_modes=n_modes,
        )
        for trial in selected
    ]
    return {
        "protocol": {
            "id": PROTOCOL_ID,
            "evidence_scope": "reduced-order simulation only; no hardware or experimental validation",
            "externally_preregistered": False,
            "trial_count": len(selected),
            "length": int(length),
            "washout": int(washout),
            "chronological_train_fraction": SPLIT_FRACTION,
            "delay_order": DELAY_ORDER,
            "n_modes": int(n_modes),
            "frequency_sampling": "independent uniform draw over [f_lo, f_hi]",
            "model_config": dict(MODEL_CONFIG),
            "tasks": {
                "narma": "NARMA-10; input mapped from [0, 0.5] to [-1, 1]",
                "parity_order": 3,
            },
            "readout": {
                "regression": "StandardScaler + Ridge(alpha=0.001)",
                "classification": "StandardScaler + RidgeClassifier(alpha=1.0)",
                "preprocessing": "train-prefix only",
            },
            "bootstrap": {
                "samples": int(bootstrap_samples),
                "base_seed": BOOTSTRAP_BASE_SEED,
                "method": "paired resampling of trial means with replacement",
                "summary_seeds": dict(BOOTSTRAP_SUMMARY_SEEDS),
            },
            "seed_roles": [asdict(trial) for trial in selected],
            "environment": {
                "python": platform.python_version(),
                "numpy": _version("numpy"),
                "matplotlib": _version("matplotlib"),
                "scikit_learn": _version("scikit-learn"),
            },
        },
        "trials": records,
        "summary": _summarize(records, bootstrap_samples=bootstrap_samples),
        "domain_checks": {
            "contact_threshold_fraction": MODEL_CONFIG["contact_margin_fraction"],
            "minimum_air_gap_fraction_observed": min(
                value
                for record in records
                for value in record["minimum_air_gap_fraction"].values()
            ),
        },
    }


def write_results(results: dict, destination: str | Path) -> None:
    payload = json.dumps(
        results,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def plot_results(results: dict, output_directory) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    index = np.arange(len(results["trials"]))
    paths = []
    specs = (
        (
            "narma_nrmse",
            "NARMA-10 NRMSE (menor es mejor)",
            "NRMSE",
            "paired_narma.png",
        ),
        (
            "parity_accuracy",
            "Paridad-3 exactitud (mayor es mejor)",
            "Exactitud",
            "paired_parity.png",
        ),
    )
    colors = {
        "nonlinear": "#1f9e89",
        "linear_mechanics": "#d95f02",
        "delay_line": "#7570b3",
    }
    labels = {
        "nonlinear": "modelo no lineal",
        "linear_mechanics": "ablación mecánica lineal",
        "delay_line": "línea de retardo digital",
    }
    for metric, title, ylabel, filename in specs:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for condition in ("nonlinear", "linear_mechanics", "delay_line"):
            values = [record[metric][condition] for record in results["trials"]]
            axis.plot(
                index,
                values,
                "-o",
                ms=4,
                color=colors[condition],
                label=labels[condition],
            )
        if metric == "parity_accuracy":
            axis.axhline(
                0.5,
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                label="azar binario",
            )
        axis.set_xlabel("par de semillas explícito")
        axis.set_ylabel(ylabel)
        axis.set_title(title + " — simulación reducida")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)
    return paths
