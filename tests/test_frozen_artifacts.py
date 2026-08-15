import hashlib
import json
from pathlib import Path

import reproduce
import study

ROOT = Path(__file__).resolve().parents[1]


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

    assert rebuilt == frozen
    rebuilt_path = tmp_path / "results.json"
    study.write_results(rebuilt, rebuilt_path)
    assert b"\r\n" not in frozen_path.read_bytes()
    assert rebuilt_path.read_bytes() == frozen_path.read_bytes()
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
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == hashlib.sha256(tracked.read_bytes()).hexdigest()
        )
