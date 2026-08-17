import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODULES = (
    "physical_model",
    "reproduce",
    "study",
    "tasks",
)


def test_publication_tree_excludes_legacy_experiments():
    assert {path.name for path in ROOT.glob("*.py")} == {
        f"{module}.py" for module in PRODUCTION_MODULES
    }
    assert not (ROOT / "figs").exists()
    assert not list(ROOT.rglob("__pycache__"))
    assert not list(ROOT.rglob(".pytest_cache"))
    assert not list(ROOT.rglob(".ruff_cache"))
    assert {path.name for path in (ROOT / "tests").glob("*.py")} == {
        "test_frozen_artifacts.py",
        "test_physical_model.py",
        "test_publication_scope.py",
        "test_reproduce.py",
        "test_study.py",
    }
    assert {path.name for path in (ROOT / "figures").glob("*.png")} == {
        "paired_narma.png",
        "paired_parity.png",
    }
    assert {path.name for path in (ROOT / "paper").iterdir()} == {
        "main.pdf",
        "main.tex",
        "refs.bib",
        "results_macros.tex",
    }
    for retired in (
        "DATASHEET.md",
        "FABRICACION.md",
        "PUBLICATION_BLOCKERS.md",
        "docs",
        "requirements-ml.txt",
    ):
        assert not (ROOT / retired).exists()


def test_readme_recipe_keeps_environment_and_caches_outside_export():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m venv .venv" not in readme
    for required in (
        "$VenvPython",
        "$env:PYTHON3119",
        '"$VENV/bin/python"',
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "PYTHONHASHSEED",
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "--use-feature=truststore",
        "-p no:cacheprovider",
    ):
        assert required in readme
    powershell = readme.split("```powershell", 1)[1].split("```", 1)[0]
    posix = readme.split("```bash", 1)[1].split("```", 1)[0]
    assert powershell.count("--use-feature=truststore") == 1
    assert posix.count("--use-feature=truststore") == 1


def test_readme_recipe_fails_closed_on_native_command_errors():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    powershell = readme.split("```powershell", 1)[1].split("```", 1)[0]

    assert "$ErrorActionPreference = 'Stop'" in powershell
    assert "$DetectedPython" in powershell
    assert powershell.count("$LASTEXITCODE -ne 0") == 5
    assert "set -euo pipefail" in readme


def test_readme_recipe_sanitizes_python_environment_before_venv():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        assert f"Remove-Item Env:{variable}" in readme
    assert "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV" in readme
    assert readme.count("PYTHONNOUSERSITE") >= 2


def test_readme_recipe_disables_inherited_pip_bypasses_before_install():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    powershell = readme.split("```powershell", 1)[1].split("```", 1)[0]
    posix = readme.split("```bash", 1)[1].split("```", 1)[0]

    assert "Get-ChildItem Env:PIP_*" in powershell
    assert 'Remove-Item "Env:$($_.Name)"' in powershell
    assert "$env:PIP_CONFIG_FILE = 'NUL'" in powershell
    assert 'PIP_*) unset "$name"' in posix
    assert "export PIP_CONFIG_FILE=/dev/null" in posix
    assert powershell.index("PIP_CONFIG_FILE") < powershell.index("-m pip install")
    assert posix.index("PIP_CONFIG_FILE") < posix.index("-m pip install")


def test_task_module_excludes_retired_experiments_and_claims():
    source = (ROOT / "tasks.py").read_text(encoding="utf-8").lower()
    for forbidden in (
        "xor_stream",
        "memory_inputs",
        "load_digit_sequences",
        "un lineal no puede",
        "chip fonónico",
        "dígitos manuscritos",
    ):
        assert forbidden not in source


def test_reproduce_is_the_only_executable_artifact_entry_point():
    marker = 'if __name__ == "__main__":'
    sources = {
        module: (ROOT / f"{module}.py").read_text(encoding="utf-8")
        for module in PRODUCTION_MODULES
    }

    assert marker in sources["reproduce"]
    assert all(marker not in sources[module] for module in ("physical_model", "study", "tasks"))


@pytest.mark.parametrize("module", PRODUCTION_MODULES)
def test_production_import_is_silent_and_cwd_independent(module, tmp_path):
    cwd = tmp_path / module
    cwd.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONHOME", None)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", f"import {module}"],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert list(cwd.iterdir()) == []


def test_documented_metrics_are_sourced_from_frozen_results():
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    summary = results["summary"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    macros = (ROOT / "paper" / "results_macros.tex").read_text(encoding="utf-8")

    expected = {
        "1.033": summary["narma_nrmse"]["nonlinear"]["mean"],
        "1.000": summary["narma_nrmse"]["linear_mechanics"]["mean"],
        "0.502": summary["narma_nrmse"]["delay_line"]["mean"],
        "0.497": summary["parity_accuracy"]["nonlinear"]["mean"],
        "+0.033": summary["paired_effects"]["narma_nonlinear_minus_linear"]["mean"],
    }
    for rendered, value in expected.items():
        assert (
            f"{value:+.3f}" == rendered
            if rendered.startswith("+")
            else f"{value:.3f}" == rendered
        )
        assert rendered in readme
    assert "\\input{results_macros.tex}" in report
    assert "\\newcommand{\\TrialCount}{12}" in macros
    assert results["protocol"]["externally_preregistered"] is False


def test_frozen_environment_matches_lock_and_ci():
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    environment = results["protocol"]["environment"]
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert environment == {
        "matplotlib": "3.11.1",
        "numpy": "2.4.6",
        "python": "3.11.9",
        "scikit_learn": "1.9.0",
    }
    assert "matplotlib==3.11.1" in lock
    assert "numpy==2.4.6" in lock
    assert "scikit-learn==1.9.0" in lock
    assert "narwhals==2.24.0" in lock
    assert lock.count("--hash=sha256:") >= 500
    assert 'python-version: "3.11.9"' in ci
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in ci
    assert "matrix:" in ci
    assert "os: [ubuntu-24.04, windows-2025]" in ci
    assert "runs-on: ${{ matrix.os }}" in ci
    assert "      PYTHONPYCACHEPREFIX:" not in ci.splitlines()
    assert ci.count("PYTHONPYCACHEPREFIX: ${{ runner.temp }}/pycache") == 1
    assert "PYTHONPYCACHEPREFIX: /tmp/graphene-pycache" not in ci
    assert "python -m pytest -q -p no:cacheprovider" in ci
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in ci
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in ci
    assert "actions/checkout@v" not in ci
    assert "actions/setup-python@v" not in ci
    assert "pip install --upgrade pip" not in ci
    assert "python -m pip install --require-hashes -r requirements-dev-lock.txt" in ci


def test_corrected_mechanical_reservoir_doi_is_used():
    sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "SOURCE_VERIFICATION.md",
            "paper/main.tex",
            "paper/refs.bib",
        )
    )
    assert "10.1063/1.5038038" in sources
    assert "10.1063/1.5042342" not in (ROOT / "paper" / "main.tex").read_text(
        encoding="utf-8"
    )


def test_release_and_frozen_report_versions_are_explicitly_distinct():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    report_source = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

    assert version == "1.0.2"
    assert r"\date{Versión 1.0.0" in report_source
    for document in (readme, model_card):
        assert "versión del software: `1.0.2`" in document
        assert "versión del informe científico congelado: `1.0.0`" in document
        assert "se conserva byte a byte" in document
    assert "software v1.0.2" in citation
    assert "informe técnico congelado v1.0.0" in citation


def test_citation_metadata_matches_version_and_scope():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert "cff-version: 1.2.0" in citation
    assert f"version: {version}" in citation
    assert "negative result" in citation
    assert "date-released: 2026-08-16" in citation
    assert "license:" not in citation
