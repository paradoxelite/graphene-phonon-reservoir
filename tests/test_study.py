import json
from collections import UserList, deque
from pathlib import Path

import numpy as np
import pytest

import study
import tasks

ROOT = Path(__file__).resolve().parents[1]


class StatefulArray:
    def __init__(self, first, second):
        self._responses = (first, second)
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        response = self._responses[min(self.calls, 1)]
        self.calls += 1
        return np.asarray(response, dtype=dtype)


def test_protocol_seed_roles_are_unique_and_disjoint():
    trials = study.protocol_trials(12)

    assert [trial.index for trial in trials] == list(range(12))
    model_seeds = {trial.model_seed for trial in trials}
    narma_seeds = {trial.narma_seed for trial in trials}
    parity_seeds = {trial.parity_seed for trial in trials}
    assert len(model_seeds) == len(narma_seeds) == len(parity_seeds) == 12
    assert model_seeds.isdisjoint(narma_seeds)
    assert model_seeds.isdisjoint(parity_seeds)
    assert narma_seeds.isdisjoint(parity_seeds)


@pytest.mark.parametrize("count", (True, 2.5, np.float64(2.0), "2"))
def test_protocol_trial_count_requires_an_integer_not_a_boolean(count):
    with pytest.raises(TypeError, match="count"):
        study.protocol_trials(count)


def test_protocol_trial_count_accepts_a_numpy_integer_scalar():
    assert len(study.protocol_trials(np.int64(2))) == 2


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("length", 52.5),
        ("length", True),
        ("washout", 12.5),
        ("washout", False),
        ("n_modes", 2.5),
        ("n_modes", True),
        ("bootstrap_samples", 100.5),
        ("bootstrap_samples", False),
    ),
)
def test_run_protocol_rejects_invalid_count_types_before_evaluation(
    monkeypatch, parameter, value
):
    def evaluation_must_not_start(*_args, **_kwargs):
        raise AssertionError("evaluation started before count validation")

    monkeypatch.setattr(study, "_evaluate_trial", evaluation_must_not_start)
    kwargs = {
        "trials": study.protocol_trials(1),
        "length": 52,
        "washout": 12,
        "n_modes": 2,
        "bootstrap_samples": 100,
    }
    kwargs[parameter] = value

    with pytest.raises(TypeError, match=parameter):
        study.run_protocol(**kwargs)


@pytest.mark.parametrize(
    ("producer", "kwargs", "parameter"),
    (
        (tasks.narma10, {"length": True}, "length"),
        (tasks.narma10, {"length": 52.5}, "length"),
        (tasks.narma10, {"length": 52, "seed": False}, "seed"),
        (tasks.parity_stream, {"length": 52.5}, "length"),
        (tasks.parity_stream, {"length": 52, "seed": 1.5}, "seed"),
        (tasks.parity_stream, {"length": 52, "order": True}, "order"),
    ),
)
def test_task_generators_reject_boolean_and_fractional_integer_parameters(
    producer, kwargs, parameter
):
    with pytest.raises(TypeError, match=parameter):
        producer(**kwargs)


def test_task_generators_accept_numpy_integer_scalars():
    narma_input, narma_target = tasks.narma10(np.int64(52), seed=np.int64(7))
    parity_input, parity_target = tasks.parity_stream(
        np.int64(52), seed=np.int64(8), order=np.int64(3)
    )

    assert narma_input.shape == narma_target.shape == (52,)
    assert parity_input.shape == parity_target.shape == (52,)


def test_delay_line_rejects_boolean_signal_and_noninteger_order():
    with pytest.raises(TypeError, match="boolean"):
        tasks.delay_embed(np.array([True, False]), order=1)
    with pytest.raises(TypeError, match="order"):
        tasks.delay_embed(np.arange(5.0), order=1.5)


def test_delay_line_rejects_textual_numeric_signal():
    with pytest.raises(TypeError, match="text"):
        tasks.delay_embed(np.array(["0.25"]), order=1)


def test_delay_line_rejects_complex_signal():
    with pytest.raises(TypeError, match="complex"):
        tasks.delay_embed(np.array([0.25 + 0.5j]), order=1)


def test_delay_line_rejects_boolean_mixed_with_real():
    with pytest.raises(TypeError, match="boolean"):
        tasks.delay_embed([True, 0.25], order=1)


def test_delay_line_rejects_boolean_inside_generic_array_like():
    with pytest.raises(TypeError, match="boolean"):
        tasks.delay_embed(UserList([True, 0.25]), order=1)


def test_delay_line_rejects_boolean_mixed_with_integer_in_deque():
    with pytest.raises(TypeError, match="boolean"):
        tasks.delay_embed(deque([True, 1]), order=1)


@pytest.mark.parametrize(
    "signal",
    (UserList([0.0, 0.25]), deque([0.0, 0.25])),
    ids=("userlist", "deque"),
)
def test_delay_line_accepts_real_generic_array_like(signal):
    embedded = tasks.delay_embed(signal, order=1)

    np.testing.assert_array_equal(embedded[:, 0], [0.0, 0.25])


def test_delay_line_rejects_stateful_invalid_snapshot_without_retry():
    signal = StatefulArray([True, 0.25], [0.0, 0.25])

    with pytest.raises(TypeError, match="boolean"):
        tasks.delay_embed(signal, order=1)

    assert signal.calls == 1


def test_delay_line_materializes_stateful_array_like_once():
    signal = StatefulArray([0.0, 0.25], [True, 0.25])

    embedded = tasks.delay_embed(signal, order=1)

    np.testing.assert_array_equal(embedded[:, 0], [0.0, 0.25])
    assert signal.calls == 1


def test_delay_line_preserves_empty_real_signal_semantics():
    embedded = tasks.delay_embed(np.empty(0, dtype=np.float64), order=3)

    assert embedded.shape == (0, 3)
    assert embedded.dtype == np.float64


def test_run_protocol_rejects_duplicate_or_overlapping_trials():
    first = study.TrialSpec(index=0, model_seed=4000, narma_seed=5000, parity_seed=6000)
    invalid_sets = (
        [first, first],
        [first, study.TrialSpec(index=1, model_seed=4000, narma_seed=5001, parity_seed=6001)],
        [first, study.TrialSpec(index=1, model_seed=4001, narma_seed=4000, parity_seed=6001)],
    )
    for trials in invalid_sets:
        with pytest.raises(ValueError, match="unique and disjoint"):
            study.run_protocol(
                trials=trials,
                length=52,
                washout=12,
                n_modes=2,
                bootstrap_samples=100,
            )


def test_frequency_sampling_is_uniform_and_documented():
    seed = 4000
    count = 16
    nonlinear, _ = study.make_paired_models(seed, n=count)
    expected = np.random.default_rng(seed).uniform(10e6, 40e6, count)
    mathematics = (ROOT / "MATEMATICAS.md").read_text(encoding="utf-8")

    np.testing.assert_array_equal(nonlinear.f0, expected)
    assert "muestreo uniforme independiente" in mathematics
    assert "muestreo logarítmico" not in mathematics
    assert "perturbado ±5 %" not in mathematics


def test_matched_ablation_preserves_all_seeded_heterogeneity():
    trial = study.protocol_trials(1)[0]
    nonlinear, linear = study.make_paired_models(trial.model_seed, n=7)

    np.testing.assert_array_equal(nonlinear.f0, linear.f0)
    np.testing.assert_array_equal(nonlinear.win, linear.win)
    np.testing.assert_array_equal(nonlinear.gamma, linear.gamma)
    assert nonlinear.k_coupling == linear.k_coupling
    assert np.any(nonlinear.alpha > 0.0)
    assert np.all(linear.alpha == 0.0)
    assert np.all(linear.eta == 0.0)


def test_regression_preprocessing_is_fit_on_training_prefix_only():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [1e6, -1e6],
            [2e6, -2e6],
        ]
    )
    y = np.arange(len(X), dtype=float)

    result = study.fit_regression_audited(X, y, split=4)

    np.testing.assert_allclose(result["train_feature_mean"], X[:4].mean(axis=0))
    assert result["train_rows"] == 4
    assert result["test_rows"] == 2
    assert np.isfinite(result["nrmse"])


@pytest.mark.parametrize("fit", (study.fit_regression_audited, study.fit_classification_audited))
@pytest.mark.parametrize("split", (True, 4.5, "4"))
def test_audited_fits_require_an_integer_split(fit, split):
    X = np.arange(16.0).reshape(8, 2)
    y = np.arange(8) % 2

    with pytest.raises(TypeError, match="split"):
        fit(X, y, split=split)


@pytest.mark.parametrize("fit", (study.fit_regression_audited, study.fit_classification_audited))
@pytest.mark.parametrize("alpha", (True, "1.0"))
def test_audited_fits_require_a_real_nonboolean_alpha(fit, alpha):
    X = np.arange(16.0).reshape(8, 2)
    y = np.arange(8) % 2

    with pytest.raises(TypeError, match="alpha"):
        fit(X, y, split=4, alpha=alpha)


@pytest.mark.parametrize("fit", (study.fit_regression_audited, study.fit_classification_audited))
@pytest.mark.parametrize(
    "features",
    (
        np.full((8, 2), "0.25"),
        np.full((8, 2), 0.25 + 0.5j),
        np.full((8, 2), True),
    ),
    ids=("text", "complex", "boolean"),
)
def test_audited_fits_reject_nonreal_or_coerced_features(fit, features):
    target = np.arange(8) % 2

    with pytest.raises(TypeError, match="X"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_reject_boolean_mixed_with_real(fit):
    features = [[True, 0.25] for _ in range(8)]
    target = np.arange(8) % 2

    with pytest.raises(TypeError, match="X.*boolean"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_reject_boolean_mixed_with_integer(fit):
    features = [[True, 1] for _ in range(8)]
    target = np.arange(8) % 2

    with pytest.raises(TypeError, match="X.*boolean"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_reject_boolean_inside_generic_feature_array_like(fit):
    features = UserList([UserList([True, 0.25]) for _ in range(8)])
    target = np.arange(8) % 2

    with pytest.raises(TypeError, match="X.*boolean"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_accept_real_generic_features_and_target(fit):
    features = UserList(
        [UserList([float(index), float(index + 1)]) for index in range(8)]
    )
    target = UserList((np.arange(8) % 2).tolist())

    result = fit(features, target, split=4)

    assert result["train_rows"] == 4


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
@pytest.mark.parametrize(
    "target",
    (
        np.full(8, "0.25"),
        np.full(8, 0.25 + 0.5j),
        np.full(8, True),
    ),
    ids=("text", "complex", "boolean"),
)
def test_audited_fits_reject_nonreal_or_coerced_target(fit, target):
    features = np.arange(16.0).reshape(8, 2)

    with pytest.raises(TypeError, match="y"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_reject_nonfinite_target(fit):
    features = np.arange(16.0).reshape(8, 2)
    target = np.arange(8.0)
    target[4] = np.nan

    with pytest.raises(ValueError, match="finite"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
@pytest.mark.parametrize(
    "target",
    (
        [True, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        [True, 0, 1, 0, 1, 0, 1, 0],
    ),
    ids=("bool-real", "bool-int"),
)
def test_audited_fits_reject_boolean_mixed_into_target(fit, target):
    features = np.arange(16.0).reshape(8, 2)

    with pytest.raises(TypeError, match="y.*boolean"):
        fit(features, target, split=4)


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_reject_stateful_invalid_feature_snapshot_without_retry(fit):
    features = StatefulArray(
        [[True, 1] for _ in range(8)],
        np.arange(16.0).reshape(8, 2),
    )
    target = np.arange(8) % 2

    with pytest.raises(TypeError, match="X.*boolean"):
        fit(features, target, split=4)

    assert features.calls == 1


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_materialize_stateful_features_once(fit):
    features = StatefulArray(
        np.arange(16.0).reshape(8, 2),
        [[True, 0.25] for _ in range(8)],
    )
    target = np.arange(8) % 2

    result = fit(features, target, split=4)

    assert result["train_rows"] == 4
    assert features.calls == 1


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_reject_stateful_invalid_target_snapshot_without_retry(fit):
    features = np.arange(16.0).reshape(8, 2)
    target = StatefulArray(
        [True, 0, 1, 0, 1, 0, 1, 0],
        np.arange(8) % 2,
    )

    with pytest.raises(TypeError, match="y.*boolean"):
        fit(features, target, split=4)

    assert target.calls == 1


@pytest.mark.parametrize(
    "fit", (study.fit_regression_audited, study.fit_classification_audited)
)
def test_audited_fits_materialize_stateful_target_once(fit):
    features = np.arange(16.0).reshape(8, 2)
    target = StatefulArray(
        np.arange(8) % 2,
        [True, 0, 1, 0, 1, 0, 1, 0],
    )

    result = fit(features, target, split=4)

    assert result["train_rows"] == 4
    assert target.calls == 1


def test_quick_protocol_is_deterministic_paired_and_recomputable(tmp_path):
    kwargs = {
        "trials": study.protocol_trials(2),
        "length": 360,
        "washout": 60,
        "n_modes": 6,
        "bootstrap_samples": 300,
    }
    first = study.run_protocol(**kwargs)
    second = study.run_protocol(**kwargs)

    assert first == second
    assert len(first["trials"]) == 2
    protocol = first["protocol"]
    assert protocol["externally_preregistered"] is False
    assert protocol["model_config"] == {
        "Q": 20.0,
        "V_ac": 0.15,
        "V_dc": 1.5,
        "area": 1.77e-12,
        "contact_margin_fraction": 0.05,
        "coupling_fraction": 0.03,
        "dt": 2.5e-10,
        "duffing_onset": 5e-09,
        "f_hi": 40000000.0,
        "f_lo": 10000000.0,
        "gap": 2.5e-07,
        "m_eff": 2e-18,
        "nonlinear_damping_ratio": 0.2,
        "oxide": 2.85e-07,
        "steps_per_input": 8,
        "wavelength": 6.328e-07,
    }
    assert protocol["model_config"] is not study.MODEL_CONFIG
    with pytest.raises(TypeError):
        study.MODEL_CONFIG["V_dc"] = 99.0
    assert protocol["tasks"]["parity_order"] == 3
    assert protocol["frequency_sampling"] == "independent uniform draw over [f_lo, f_hi]"
    assert protocol["readout"]["preprocessing"] == "train-prefix only"
    assert protocol["bootstrap"] == {
        "samples": 300,
        "base_seed": 20260814,
        "method": "paired resampling of trial means with replacement",
        "summary_seeds": {
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
        },
    }
    assert "bootstrap_samples" not in protocol
    assert "bootstrap_seed" not in protocol
    for record in first["trials"]:
        expected_frequencies = np.random.default_rng(record["model_seed"]).uniform(
            10e6, 40e6, kwargs["n_modes"]
        )
        np.testing.assert_array_equal(
            record["model_frequencies_hz"], expected_frequencies
        )
        assert set(record["narma_nrmse"]) == {
            "nonlinear",
            "linear_mechanics",
            "delay_line",
        }
        assert set(record["parity_accuracy"]) == {
            "nonlinear",
            "linear_mechanics",
            "delay_line",
        }
        assert all(np.isfinite(list(record["narma_nrmse"].values())))
        assert all(0.0 <= value <= 1.0 for value in record["parity_accuracy"].values())
        assert record["contact_events"] == 0
        assert set(record["minimum_air_gap_fraction"]) == {
            "narma_linear",
            "narma_nonlinear",
            "parity_linear",
            "parity_nonlinear",
        }
        assert all(
            0.05 < value <= 1.0 for value in record["minimum_air_gap_fraction"].values()
        )

    for metric in ("narma_nrmse", "parity_accuracy"):
        for condition in ("nonlinear", "linear_mechanics", "delay_line"):
            values = np.array(
                [record[metric][condition] for record in first["trials"]],
                dtype=float,
            )
            summary = first["summary"][metric][condition]
            assert summary["n"] == len(values)
            assert summary["mean"] == float(np.mean(values))
            assert summary["median"] == float(np.median(values))

    assert first["domain_checks"]["contact_threshold_fraction"] == 0.05
    assert first["domain_checks"]["minimum_air_gap_fraction_observed"] == min(
        value
        for record in first["trials"]
        for value in record["minimum_air_gap_fraction"].values()
    )

    destination = tmp_path / "results.json"
    study.write_results(first, destination)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == first
    assert destination.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_write_results_rejects_nonfinite_values_without_touching_destination(
    tmp_path, value
):
    destination = tmp_path / "results.json"
    destination.write_bytes(b"previous-generation")

    with pytest.raises(ValueError):
        study.write_results({"metric": value}, destination)

    assert destination.read_bytes() == b"previous-generation"


def test_write_results_invalid_payload_does_not_create_parent_directory(tmp_path):
    destination = tmp_path / "not-created" / "results.json"

    with pytest.raises(ValueError):
        study.write_results({"metric": np.nan}, destination)

    assert not destination.parent.exists()


def test_plot_legends_do_not_occlude_data_markers(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    captured = []
    real_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axis = real_subplots(*args, **kwargs)
        captured.append((figure, axis))
        return figure, axis

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))

    study.plot_results(results, tmp_path)

    assert len(captured) == 2
    for figure, axis in captured:
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        legend_box = axis.get_legend().get_window_extent(canvas.get_renderer())
        for line in axis.lines:
            if line.get_marker() != "o":
                continue
            display_points = axis.transData.transform(
                np.column_stack((line.get_xdata(), line.get_ydata()))
            )
            collisions = [
                index
                for index, (x, y) in enumerate(display_points)
                if legend_box.contains(x, y)
            ]
            assert not collisions, (axis.get_title(), line.get_label(), collisions)
