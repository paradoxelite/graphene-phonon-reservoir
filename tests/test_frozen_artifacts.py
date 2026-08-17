import json
import math
from pathlib import Path

from PIL import Image
import pytest

import reproduce
import study

ROOT = Path(__file__).resolve().parents[1]
PORTABLE_REL_TOL = 2e-11
PORTABLE_ABS_TOL = 2e-12
JSON_BUILTIN_TYPES = frozenset((dict, list, str, int, float, bool, type(None)))
# Deliberately tight cross-platform bounds: the Ubuntu/Windows falsifier observed
# max absolute drift 1.65e-12 and max relative drift 1.38e-11.


def _assert_portable_equal(actual, expected, path="$"):
    assert type(expected) in JSON_BUILTIN_TYPES, path
    assert type(actual) in JSON_BUILTIN_TYPES, path
    if type(expected) is dict:
        assert type(actual) is dict, path
        assert all(type(key) is str for key in expected), path
        assert all(type(key) is str for key in actual), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_portable_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if type(expected) is list:
        assert type(actual) is list, path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_portable_equal(
                actual_item, expected_item, f"{path}[{index}]"
            )
        return
    if type(expected) is float:
        assert type(actual) is float, path
        assert math.isfinite(actual), path
        assert math.isclose(
            actual,
            expected,
            rel_tol=PORTABLE_REL_TOL,
            abs_tol=PORTABLE_ABS_TOL,
        ), path
        return
    assert type(actual) is type(expected), path
    assert actual == expected, path


def test_portable_comparison_rejects_float_to_integer_substitution():
    for integer_value, frozen_float in ((0, 0.0), (1, 1.0), (-2, -2.0)):
        with pytest.raises(AssertionError):
            _assert_portable_equal(
                {"model_config": {"value": integer_value}},
                {"model_config": {"value": frozen_float}},
            )


def test_portable_comparison_rejects_container_subclasses():
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    class StringSubclass(str):
        pass

    class FloatSubclass(float):
        pass

    for actual, expected in (
        (DictSubclass({"value": 1}), {"value": 1}),
        ({"value": 1}, DictSubclass({"value": 1})),
        (ListSubclass([1]), [1]),
        ([1], ListSubclass([1])),
        ({StringSubclass("value"): 1}, {"value": 1}),
        ({"value": 1}, {StringSubclass("value"): 1}),
        (1.0, FloatSubclass(1.0)),
    ):
        with pytest.raises(AssertionError):
            _assert_portable_equal(actual, expected)


def test_frozen_artifact_manifest_attests_all_five_payloads():
    payloads = reproduce.GENERATED_ARTIFACTS + (reproduce.PDF_ARTIFACT,)
    manifest = ROOT / reproduce.ARTIFACT_MANIFEST

    assert reproduce.verify_artifact_manifest(ROOT, expected_artifacts=payloads)
    assert b"\r\n" not in manifest.read_bytes()
    assert manifest.read_bytes().endswith(b"\n")


def test_frozen_full_protocol_matches_current_producer(tmp_path):
    frozen_path = ROOT / "results.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rebuilt = study.run_protocol(
        trials=study.protocol_trials(12),
        length=1200,
        washout=200,
        n_modes=16,
        bootstrap_samples=10_000,
    )

    _assert_portable_equal(rebuilt, frozen)
    rebuilt_path = tmp_path / "results.json"
    study.write_results(rebuilt, rebuilt_path)
    rebuilt_bytes = rebuilt_path.read_bytes()
    assert b"\r\n" not in rebuilt_bytes
    assert rebuilt_bytes.endswith(b"\n")
    _assert_portable_equal(json.loads(rebuilt_bytes.decode("utf-8")), frozen)
    assert b"\r\n" not in frozen_path.read_bytes()
    assert frozen["protocol"]["trial_count"] == 12
    assert frozen["protocol"]["evidence_scope"].startswith(
        "reduced-order simulation only"
    )
    assert all(record["contact_events"] == 0 for record in frozen["trials"])


def test_frozen_figures_match_current_results_producer(tmp_path):
    frozen = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    generated = study.plot_results(frozen, tmp_path)

    assert {path.name for path in generated} == {
        "paired_narma.png",
        "paired_parity.png",
    }
    for path in generated:
        tracked = ROOT / "figures" / path.name
        with Image.open(path) as generated_image, Image.open(tracked) as frozen_image:
            assert generated_image.format == frozen_image.format == "PNG"
            assert generated_image.size == frozen_image.size
            assert (
                generated_image.convert("RGBA").tobytes()
                == frozen_image.convert("RGBA").tobytes()
            )
