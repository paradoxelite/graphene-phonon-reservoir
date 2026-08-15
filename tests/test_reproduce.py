import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import reproduce

ROOT = Path(__file__).resolve().parents[1]


def _make_directory_reparse(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"cannot create Windows junction: {completed.stderr}")
    else:
        link.symlink_to(target, target_is_directory=True)


def _remove_directory_reparse(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def _make_file_reparse(link: Path, target: Path) -> bool:
    try:
        link.symlink_to(target)
    except OSError:
        return False
    return True


def _quick_results():
    return reproduce.study.run_protocol(
        trials=reproduce.study.protocol_trials(1),
        length=52,
        washout=12,
        n_modes=2,
        bootstrap_samples=100,
    )


@pytest.mark.parametrize("compile_pdf", (1, "false", None))
def test_publication_requires_an_exact_boolean_compile_flag(tmp_path, compile_pdf):
    root = tmp_path / "publication"

    with pytest.raises(TypeError, match="compile_pdf"):
        reproduce.publish_artifacts({}, root=root, compile_pdf=compile_pdf)

    assert not root.exists()
    assert not list(tmp_path.glob(".publication-artifacts-*"))


def _install_old_generation(root: Path, *, include_pdf: bool = False):
    payloads = reproduce.GENERATED_ARTIFACTS + (
        (reproduce.PDF_ARTIFACT,) if include_pdf else ()
    )
    before = {}
    for index, relative in enumerate(payloads):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-generation-{index}".encode("ascii"))
        before[relative] = destination.read_bytes()
    reproduce.write_artifact_manifest(root, payloads)
    if include_pdf:
        (root / "paper" / "main.tex").write_text("source", encoding="utf-8")
    before[reproduce.ARTIFACT_MANIFEST] = (
        root / reproduce.ARTIFACT_MANIFEST
    ).read_bytes()
    return payloads, before


def _make_unlock_fail(monkeypatch):
    if os.name == "nt":
        import msvcrt

        real_unlock = msvcrt.locking

        def fail_unlock(file_descriptor, mode, count):
            if mode == msvcrt.LK_UNLCK:
                raise OSError("injected lock release failure")
            return real_unlock(file_descriptor, mode, count)

        monkeypatch.setattr(msvcrt, "locking", fail_unlock)
    else:
        import fcntl

        real_unlock = fcntl.flock

        def fail_unlock(file_descriptor, operation):
            if operation == fcntl.LOCK_UN:
                raise OSError("injected lock release failure")
            return real_unlock(file_descriptor, operation)

        monkeypatch.setattr(fcntl, "flock", fail_unlock)


def _run_crashing_publisher(
    root: Path,
    *,
    seam: str,
    crash_after: int,
    compile_pdf: bool = False,
):
    script = textwrap.dedent(
        f"""
        import os
        import sys
        from pathlib import Path

        sys.path.insert(0, {str(ROOT)!r})
        import reproduce

        root = Path({str(root)!r})

        def fake_write_results(_results, destination):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"new-results")

        def fake_plot_results(_results, output_directory):
            output = Path(output_directory)
            output.mkdir(parents=True, exist_ok=True)
            paths = []
            for name in ("paired_narma.png", "paired_parity.png"):
                path = output / name
                path.write_bytes(("new-" + name).encode("ascii"))
                paths.append(path)
            return paths

        def fake_write_macros(_results, destination):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"new-macros")

        reproduce.study.write_results = fake_write_results
        reproduce.study.plot_results = fake_plot_results
        reproduce.write_report_macros = fake_write_macros

        def fake_compile_report(*, root):
            destination = Path(root) / "paper" / "main.pdf"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"new-pdf")
            return destination

        reproduce.compile_report = fake_compile_report

        crash_after = {crash_after}
        seam = {seam!r}
        counter = 0
        real_replace = reproduce.os.replace
        real_copy2 = reproduce.shutil.copy2
        payloads = reproduce.GENERATED_ARTIFACTS + (
            (reproduce.PDF_ARTIFACT,) if {compile_pdf!r} else ()
        )
        public = {{
            os.path.normcase(os.path.abspath(root / relative))
            for relative in payloads + (reproduce.ARTIFACT_MANIFEST,)
        }}

        def crash_replace(source, destination):
            global counter
            result = real_replace(source, destination)
            normalized = os.path.normcase(os.path.abspath(destination))
            if seam == "payload" and normalized in public:
                counter += 1
                if counter == crash_after:
                    os._exit(91)
            return result

        def crash_copy(source, destination, *args, **kwargs):
            global counter
            result = real_copy2(source, destination, *args, **kwargs)
            if seam == "backup" and ".backups" in Path(destination).parts:
                counter += 1
                if counter == crash_after:
                    os._exit(92)
            return result

        reproduce.os.replace = crash_replace
        reproduce.shutil.copy2 = crash_copy
        reproduce.publish_artifacts(
            {{"generation": "new"}},
            root=root,
            compile_pdf={compile_pdf!r},
        )
        """
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=root.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_next_invocation_recovers_before_generation(
    root, payloads, before, monkeypatch
):
    def stop_generation(_results, _output_directory):
        raise RuntimeError("injected stop after recovery")

    monkeypatch.setattr(reproduce.study, "plot_results", stop_generation)
    with pytest.raises(RuntimeError, match="injected stop after recovery"):
        reproduce.publish_artifacts({"generation": "later"}, root=root)

    assert {
        relative: (root / relative).read_bytes()
        for relative in payloads + (reproduce.ARTIFACT_MANIFEST,)
    } == before
    assert reproduce.verify_artifact_manifest(
        root,
        expected_artifacts=payloads,
    )
    assert not list(root.parent.glob(".publication-artifacts-*"))


@pytest.mark.parametrize("crash_after", (1, 2, 3, 4, 5, 6))
def test_next_invocation_recovers_process_death_after_each_public_move(
    tmp_path, monkeypatch, crash_after
):
    root = tmp_path / "publication"
    payloads, before = _install_old_generation(root, include_pdf=True)

    completed = _run_crashing_publisher(
        root,
        seam="payload",
        crash_after=crash_after,
        compile_pdf=True,
    )

    assert completed.returncode == 91, (completed.stdout, completed.stderr)
    _assert_next_invocation_recovers_before_generation(
        root, payloads, before, monkeypatch
    )


@pytest.mark.parametrize("crash_after", (1, 2, 3, 4, 5, 6))
def test_next_invocation_recovers_process_death_after_each_backup(
    tmp_path, monkeypatch, crash_after
):
    root = tmp_path / "publication"
    payloads, before = _install_old_generation(root, include_pdf=True)

    completed = _run_crashing_publisher(
        root,
        seam="backup",
        crash_after=crash_after,
        compile_pdf=True,
    )

    assert completed.returncode == 92, (completed.stdout, completed.stderr)
    _assert_next_invocation_recovers_before_generation(
        root, payloads, before, monkeypatch
    )


def test_recovery_rejects_a_reparse_point_inside_immutable_backups(tmp_path):
    root = tmp_path / "publication"
    _install_old_generation(root)
    completed = _run_crashing_publisher(root, seam="payload", crash_after=1)
    assert completed.returncode == 91, (completed.stdout, completed.stderr)
    stage = next(root.parent.glob(".publication-artifacts-*"))
    backup_figures = stage / ".backups" / "figures"
    shutil.rmtree(backup_figures)
    outside = tmp_path / "outside-recovery"
    _make_directory_reparse(backup_figures, outside)

    try:
        with pytest.raises(ValueError, match="reparse"):
            reproduce.publish_artifacts({"generation": "later"}, root=root)
        assert not any(outside.iterdir())
    finally:
        _remove_directory_reparse(backup_figures)
        shutil.rmtree(stage)


def test_unlock_failure_after_commit_is_classified_and_stage_is_cleaned(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    payloads, _before = _install_old_generation(root)
    _make_unlock_fail(monkeypatch)

    with pytest.raises(BaseException) as captured:
        reproduce.publish_artifacts(_quick_results(), root=root)

    error = captured.value
    assert type(error).__name__ == "PublicationReleaseError"
    assert error.committed is True
    assert error.manifest_valid is True
    assert reproduce.verify_artifact_manifest(root, expected_artifacts=payloads)
    assert not list(tmp_path.glob(".publication-artifacts-*"))


def test_unlock_failure_does_not_mask_a_primary_publication_error(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    payloads, before = _install_old_generation(root)
    real_replace = os.replace
    primary_injected = False

    def fail_first_public_commit(source, destination):
        nonlocal primary_injected
        destination_path = Path(destination)
        if root in destination_path.parents and not primary_injected:
            primary_injected = True
            raise RuntimeError("injected primary publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(reproduce.os, "replace", fail_first_public_commit)
    _make_unlock_fail(monkeypatch)

    with pytest.raises(RuntimeError, match="injected primary publication failure") as captured:
        reproduce.publish_artifacts(_quick_results(), root=root)

    notes = getattr(captured.value, "__notes__", ())
    assert any("lock release" in note for note in notes)
    assert {
        relative: (root / relative).read_bytes()
        for relative in payloads + (reproduce.ARTIFACT_MANIFEST,)
    } == before
    assert not list(tmp_path.glob(".publication-artifacts-*"))


@pytest.mark.parametrize("linked_parent", ("figures", "paper"))
def test_publication_rejects_reparse_payload_parents_without_external_writes(
    tmp_path, linked_parent
):
    root = tmp_path / "publication"
    outside = tmp_path / f"outside-{linked_parent}"
    root.mkdir()
    _make_directory_reparse(root / linked_parent, outside)

    with pytest.raises(ValueError, match="reparse"):
        reproduce.publish_artifacts(_quick_results(), root=root)

    assert not any(outside.iterdir())


def test_publication_rejects_a_reparse_component_in_the_lock_route(tmp_path):
    real_parent = tmp_path / "real-parent"
    linked_parent = tmp_path / "linked-parent"
    _make_directory_reparse(linked_parent, real_parent)
    root = linked_parent / "publication"

    with pytest.raises(ValueError, match="reparse"):
        reproduce.publish_artifacts(_quick_results(), root=root)

    assert not (real_parent / ".publication.artifact-publication.lock").exists()


def test_publication_rejects_reparse_pdf_source(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    paper = root / "paper"
    paper.mkdir(parents=True)
    outside_source = tmp_path / "outside-main.tex"
    outside_source.write_text("outside source", encoding="utf-8")
    main_tex = paper / "main.tex"
    if not _make_file_reparse(main_tex, outside_source):
        main_tex.write_text("local stand-in", encoding="utf-8")
        linked_metadata = os.lstat(main_tex)
        linked_identity = (linked_metadata.st_dev, linked_metadata.st_ino)
        real_is_reparse = reproduce._is_reparse

        def inject_file_reparse(metadata):
            if (metadata.st_dev, metadata.st_ino) == linked_identity:
                return True
            return real_is_reparse(metadata)

        monkeypatch.setattr(reproduce, "_is_reparse", inject_file_reparse)

    def fake_compile_report(*, root):
        destination = Path(root) / reproduce.PDF_ARTIFACT
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-pdf")
        return destination

    monkeypatch.setattr(reproduce, "compile_report", fake_compile_report)

    with pytest.raises(reproduce.UnsafePublicationPathError, match="reparse"):
        reproduce.publish_artifacts(_quick_results(), root=root, compile_pdf=True)


def test_publication_preflights_pdf_source_components(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    real_check = reproduce._assert_no_reparse_components

    def reject_pdf_source(path):
        if Path(path).name == "main.tex":
            raise reproduce.UnsafePublicationPathError(
                "reparse point in injected PDF source"
            )
        return real_check(path)

    monkeypatch.setattr(reproduce, "_assert_no_reparse_components", reject_pdf_source)

    with pytest.raises(reproduce.UnsafePublicationPathError, match="PDF source"):
        reproduce.publish_artifacts(_quick_results(), root=root, compile_pdf=True)


def test_artifact_lock_serializes_independent_processes(tmp_path):
    root = tmp_path / "publication"
    signal = tmp_path / "child-entered"
    script = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); import reproduce; "
        f"root=Path({str(root)!r}); signal=Path({str(signal)!r}); "
        "lock=reproduce._artifact_lock(root, timeout=3.0); lock.__enter__(); "
        "signal.write_text('entered', encoding='utf-8'); lock.__exit__(None,None,None)"
    )

    with reproduce._artifact_lock(root, timeout=1.0):
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.3)
        assert not signal.exists()
    stdout, stderr = child.communicate(timeout=5)

    assert child.returncode == 0, (stdout, stderr)
    assert signal.read_text(encoding="utf-8") == "entered"


def test_artifact_manifest_rejects_a_mixed_generation(tmp_path):
    root = tmp_path / "publication"
    artifacts = reproduce.GENERATED_ARTIFACTS + (reproduce.PDF_ARTIFACT,)
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"generation-a-{index}".encode("ascii"))

    manifest = reproduce.write_artifact_manifest(root, artifacts)
    assert manifest == root / reproduce.ARTIFACT_MANIFEST
    assert reproduce.verify_artifact_manifest(root, expected_artifacts=artifacts)

    (root / artifacts[0]).write_bytes(b"generation-b")
    with pytest.raises(RuntimeError, match="artifact manifest mismatch"):
        reproduce.verify_artifact_manifest(root, expected_artifacts=artifacts)


def test_artifact_manifest_verification_uses_descriptor_bound_reads(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    artifacts = reproduce.GENERATED_ARTIFACTS
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"generation-{index}".encode("ascii"))
    reproduce.write_artifact_manifest(root, artifacts)

    def reject_unbounded_read(_path):
        raise AssertionError("Path.read_bytes is not descriptor-bound")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    assert reproduce.verify_artifact_manifest(root, expected_artifacts=artifacts)


def test_artifact_verification_rejects_growth_during_descriptor_read(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    artifacts = reproduce.GENERATED_ARTIFACTS
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bytes([65 + index]) * 131_072)
    reproduce.write_artifact_manifest(root, artifacts)
    target = root / artifacts[0]
    target_metadata = os.lstat(target)
    real_read = os.read
    real_signature = reproduce._metadata_signature
    state = {"attempted": False, "simulated": False, "signature_calls": 0}

    def grow_after_first_chunk(descriptor, count):
        chunk = real_read(descriptor, count)
        if (
            chunk
            and not state["attempted"]
            and os.path.samestat(os.fstat(descriptor), target_metadata)
        ):
            state["attempted"] = True
            try:
                with target.open("ab") as stream:
                    stream.write(b"growth")
            except OSError:
                state["simulated"] = True
        return chunk

    def inject_growth_signature(metadata):
        signature = real_signature(metadata)
        if state["simulated"] and os.path.samestat(metadata, target_metadata):
            state["signature_calls"] += 1
            if state["signature_calls"] >= 2:
                changed = list(signature)
                changed[3] += 6
                return tuple(changed)
        return signature

    monkeypatch.setattr(reproduce.os, "read", grow_after_first_chunk)
    monkeypatch.setattr(reproduce, "_metadata_signature", inject_growth_signature)

    with pytest.raises(RuntimeError, match="artifact manifest mismatch"):
        reproduce.verify_artifact_manifest(root, expected_artifacts=artifacts)

    assert state["attempted"]


def test_artifact_verification_rejects_same_size_path_substitution(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    artifacts = reproduce.GENERATED_ARTIFACTS
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bytes([65 + index]) * 131_072)
    reproduce.write_artifact_manifest(root, artifacts)
    target = root / artifacts[0]
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"Z" * target.stat().st_size)
    target_metadata = os.lstat(target)
    replacement_metadata = os.lstat(replacement)
    real_read = os.read
    real_lstat = os.lstat
    state = {"attempted": False, "simulated": False}

    def substitute_after_first_chunk(descriptor, count):
        chunk = real_read(descriptor, count)
        if (
            chunk
            and not state["attempted"]
            and os.path.samestat(os.fstat(descriptor), target_metadata)
        ):
            state["attempted"] = True
            try:
                os.replace(replacement, target)
            except OSError:
                state["simulated"] = True
        return chunk

    def inject_substituted_identity(path):
        if state["simulated"] and os.path.normcase(os.fspath(path)) == os.path.normcase(
            os.fspath(target)
        ):
            return replacement_metadata
        return real_lstat(path)

    monkeypatch.setattr(reproduce.os, "read", substitute_after_first_chunk)
    monkeypatch.setattr(reproduce.os, "lstat", inject_substituted_identity)

    with pytest.raises(RuntimeError, match="artifact manifest mismatch"):
        reproduce.verify_artifact_manifest(root, expected_artifacts=artifacts)

    assert state["attempted"]


def test_descriptor_reader_rejects_identical_substitution_before_open(
    tmp_path, monkeypatch
):
    target = tmp_path / "payload.bin"
    replacement = tmp_path / "replacement.bin"
    payload = b"same bytes cannot prove same file" * 4096
    target.write_bytes(payload)
    replacement.write_bytes(payload)
    real_open = os.open
    state = {"attempted": False}

    def substitute_then_open(path, flags, *args, **kwargs):
        if (
            not state["attempted"]
            and os.path.normcase(os.fspath(path))
            == os.path.normcase(os.fspath(target))
        ):
            state["attempted"] = True
            os.replace(replacement, target)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(reproduce.os, "open", substitute_then_open)

    with pytest.raises(RuntimeError, match="changed identity"):
        reproduce._file_record(target, max_bytes=len(payload) + 1)

    assert state["attempted"]


def test_manifest_writer_rejects_payload_over_physical_cap_before_marker(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    artifacts = reproduce.GENERATED_ARTIFACTS
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"generation-{index}".encode("ascii"))
    oversized = root / artifacts[0]
    with oversized.open("wb") as stream:
        stream.truncate(reproduce.ARTIFACT_BYTE_LIMITS[artifacts[0]] + 1)

    def reject_unbounded_read(_path):
        raise AssertionError("Path.read_bytes is not descriptor-bound")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    with pytest.raises(RuntimeError, match="physical bound"):
        reproduce.write_artifact_manifest(root, artifacts)

    assert not (root / reproduce.ARTIFACT_MANIFEST).exists()


def test_publication_staging_and_commit_verification_are_descriptor_bound(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"

    def reject_unbounded_read(_path):
        raise AssertionError("Path.read_bytes is not descriptor-bound")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    installed = reproduce.publish_artifacts(_quick_results(), root=root)

    assert installed[-1] == root / reproduce.ARTIFACT_MANIFEST
    assert reproduce.verify_artifact_manifest(
        root,
        expected_artifacts=reproduce.GENERATED_ARTIFACTS,
    )


def test_transaction_journal_reader_is_descriptor_bound(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    stage = tmp_path / ".publication-artifacts-test"
    stage.mkdir()
    installation = reproduce.GENERATED_ARTIFACTS + (reproduce.ARTIFACT_MANIFEST,)
    journal = {
        "schema_version": 1,
        "root": reproduce._transaction_root_id(root),
        "artifacts": [
            {
                "path": relative.as_posix(),
                "existed": False,
                "sha256": None,
                "size": None,
            }
            for relative in installation
        ],
    }
    (stage / reproduce.TRANSACTION_JOURNAL).write_text(
        json.dumps(journal, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    def reject_unbounded_read(_path):
        raise AssertionError("Path.read_bytes is not descriptor-bound")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    selected, records = reproduce._load_transaction_journal(root, stage)

    assert selected == installation
    assert len(records) == len(installation)


def test_publication_installs_the_manifest_last(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    payloads = reproduce.GENERATED_ARTIFACTS
    for index, relative in enumerate(payloads):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-generation-{index}".encode("ascii"))
    reproduce.write_artifact_manifest(root, payloads)

    results = reproduce.study.run_protocol(
        trials=reproduce.study.protocol_trials(1),
        length=52,
        washout=12,
        n_modes=2,
        bootstrap_samples=100,
    )
    real_replace = os.replace
    destinations = {
        os.path.normcase(str((root / relative).resolve())): relative
        for relative in payloads + (reproduce.ARTIFACT_MANIFEST,)
    }
    order = []
    intermediate_rejected = False

    def observe_replace(source, destination):
        nonlocal intermediate_rejected
        result = real_replace(source, destination)
        normalized = os.path.normcase(str(Path(destination).resolve()))
        if normalized in destinations:
            relative = destinations[normalized]
            order.append(relative)
            if len(order) == 1:
                with pytest.raises(RuntimeError, match="artifact manifest mismatch"):
                    reproduce.verify_artifact_manifest(
                        root, expected_artifacts=payloads
                    )
                intermediate_rejected = True
        return result

    monkeypatch.setattr(reproduce.os, "replace", observe_replace)
    reproduce.publish_artifacts(results, root=root)

    assert intermediate_rejected
    assert order[-1] == reproduce.ARTIFACT_MANIFEST
    assert reproduce.verify_artifact_manifest(root, expected_artifacts=payloads)


def test_publication_generation_failure_preserves_existing_artifacts(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    artifacts = (
        Path("results.json"),
        Path("figures/paired_narma.png"),
        Path("figures/paired_parity.png"),
        Path("paper/results_macros.tex"),
    )
    before = {}
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-{index}".encode("ascii"))
        before[relative] = destination.read_bytes()

    def fail_after_partial_plot(_results, output_directory):
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "paired_narma.png").write_bytes(b"partial-new-figure")
        raise RuntimeError("injected plot failure")

    monkeypatch.setattr(reproduce.study, "plot_results", fail_after_partial_plot)
    with pytest.raises(RuntimeError, match="injected plot failure"):
        reproduce.publish_artifacts({"generation": "new"}, root=root)

    assert {relative: (root / relative).read_bytes() for relative in artifacts} == before
    assert not list(tmp_path.glob(".publication-artifacts-*"))


def test_publication_rolls_back_a_replace_that_moves_then_raises(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    before = {}
    for index, relative in enumerate(reproduce.GENERATED_ARTIFACTS):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-{index}".encode("ascii"))
        stat = destination.stat()
        before[relative] = (destination.read_bytes(), stat.st_mtime_ns)

    results = reproduce.study.run_protocol(
        trials=reproduce.study.protocol_trials(1),
        length=52,
        washout=12,
        n_modes=2,
        bootstrap_samples=100,
    )
    real_replace = os.replace
    public_destinations = {
        os.path.normcase(str((root / relative).resolve()))
        for relative in reproduce.GENERATED_ARTIFACTS
    }
    installs = 0
    injected = False

    def move_then_raise(source, destination):
        nonlocal installs, injected
        normalized = os.path.normcase(str(Path(destination).resolve()))
        if normalized in public_destinations and not injected:
            installs += 1
            real_replace(source, destination)
            if installs == 2:
                injected = True
                raise OSError("injected post-move failure")
            return None
        return real_replace(source, destination)

    monkeypatch.setattr(reproduce.os, "replace", move_then_raise)
    with pytest.raises(OSError, match="injected post-move failure"):
        reproduce.publish_artifacts(results, root=root)

    after = {
        relative: ((root / relative).read_bytes(), (root / relative).stat().st_mtime_ns)
        for relative in reproduce.GENERATED_ARTIFACTS
    }
    assert after == before
    assert not list(tmp_path.glob(".publication-artifacts-*"))


def test_moved_then_failed_manifest_restores_previous_valid_generation(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    payloads = reproduce.GENERATED_ARTIFACTS
    for index, relative in enumerate(payloads):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-generation-{index}".encode("ascii"))
    reproduce.write_artifact_manifest(root, payloads)
    published = payloads + (reproduce.ARTIFACT_MANIFEST,)
    before = {
        relative: (
            (root / relative).read_bytes(),
            (root / relative).stat().st_mtime_ns,
        )
        for relative in published
    }

    results = reproduce.study.run_protocol(
        trials=reproduce.study.protocol_trials(1),
        length=52,
        washout=12,
        n_modes=2,
        bootstrap_samples=100,
    )
    real_replace = os.replace
    manifest_destination = os.path.normcase(
        str((root / reproduce.ARTIFACT_MANIFEST).resolve())
    )
    injected = False

    def move_manifest_then_fail(source, destination):
        nonlocal injected
        result = real_replace(source, destination)
        normalized = os.path.normcase(str(Path(destination).resolve()))
        if normalized == manifest_destination and not injected:
            injected = True
            raise OSError("injected post-manifest-move failure")
        return result

    monkeypatch.setattr(reproduce.os, "replace", move_manifest_then_fail)
    with pytest.raises(OSError, match="injected post-manifest-move failure"):
        reproduce.publish_artifacts(results, root=root)

    after = {
        relative: (
            (root / relative).read_bytes(),
            (root / relative).stat().st_mtime_ns,
        )
        for relative in published
    }
    assert after == before
    assert reproduce.verify_artifact_manifest(root, expected_artifacts=payloads)


def test_rollback_propagates_control_exception_after_restoring_all_targets(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    before = {}
    for index, relative in enumerate(reproduce.GENERATED_ARTIFACTS):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-{index}".encode("ascii"))
        before[relative] = destination.read_bytes()

    results = reproduce.study.run_protocol(
        trials=reproduce.study.protocol_trials(1),
        length=52,
        washout=12,
        n_modes=2,
        bootstrap_samples=100,
    )
    real_replace = os.replace
    public_destinations = {
        os.path.normcase(str((root / relative).resolve()))
        for relative in reproduce.GENERATED_ARTIFACTS
    }
    installs = 0
    control_injected = False

    def fail_commit_and_interrupt_rollback(source, destination):
        nonlocal installs, control_injected
        source_path = Path(source)
        normalized = os.path.normcase(str(Path(destination).resolve()))
        if normalized not in public_destinations:
            return real_replace(source, destination)
        if ".restore" in source_path.parts and not control_injected:
            control_injected = True
            real_replace(source, destination)
            raise KeyboardInterrupt("injected rollback interruption")
        if ".restore" not in source_path.parts:
            installs += 1
            if installs == 2:
                raise OSError("injected commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(reproduce.os, "replace", fail_commit_and_interrupt_rollback)
    with pytest.raises(KeyboardInterrupt, match="injected rollback interruption"):
        reproduce.publish_artifacts(results, root=root)

    assert {
        relative: (root / relative).read_bytes()
        for relative in reproduce.GENERATED_ARTIFACTS
    } == before


def test_regenerate_routes_all_writes_through_transaction(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    before = {}
    for index, relative in enumerate(reproduce.GENERATED_ARTIFACTS):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-{index}".encode("ascii"))
        before[relative] = destination.read_bytes()

    monkeypatch.setattr(
        reproduce.study,
        "run_protocol",
        lambda **_kwargs: {"generation": "new"},
    )

    def fail_plot(_results, _output_directory):
        raise RuntimeError("injected producer failure")

    monkeypatch.setattr(reproduce.study, "plot_results", fail_plot)
    with pytest.raises(RuntimeError, match="injected producer failure"):
        reproduce.regenerate(root=root)

    assert {relative: (root / relative).read_bytes() for relative in before} == before


def test_regenerate_with_pdf_failure_preserves_all_five_artifacts(
    tmp_path, monkeypatch
):
    root = tmp_path / "publication"
    artifacts = reproduce.GENERATED_ARTIFACTS + (Path("paper/main.pdf"),)
    before = {}
    for index, relative in enumerate(artifacts):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-{index}".encode("ascii"))
        before[relative] = destination.read_bytes()
    (root / "paper" / "main.tex").write_text("source", encoding="utf-8")

    results = reproduce.study.run_protocol(
        trials=reproduce.study.protocol_trials(1),
        length=52,
        washout=12,
        n_modes=2,
        bootstrap_samples=100,
    )
    monkeypatch.setattr(reproduce.study, "run_protocol", lambda **_kwargs: results)

    def fail_report(*, root):
        raise RuntimeError(f"injected report failure in {root}")

    monkeypatch.setattr(reproduce, "compile_report", fail_report)
    with pytest.raises(RuntimeError, match="injected report failure"):
        reproduce.regenerate(root=root, compile_pdf=True)

    assert {relative: (root / relative).read_bytes() for relative in artifacts} == before


def test_report_macros_are_derived_from_frozen_results(tmp_path):
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    text = reproduce.render_report_macros(results)

    assert "\\newcommand{\\TrialCount}{12}" in text
    assert "\\newcommand{\\NarmaNonlinearMean}{1.033}" in text
    assert "\\newcommand{\\NarmaLinearMean}{1.000}" in text
    assert "\\newcommand{\\NarmaDelayMean}{0.502}" in text
    assert "\\newcommand{\\NarmaNonlinearMinusLinear}{0.033}" in text
    assert "\\newcommand{\\ParityNonlinearMean}{0.497}" in text
    assert "\\newcommand{\\MinimumAirGapFraction}{0.838}" in text
    assert "nan" not in text.lower()

    destination = tmp_path / "results_macros.tex"
    reproduce.write_report_macros(results, destination)
    assert destination.read_text(encoding="utf-8") == text
    assert b"\r\n" not in destination.read_bytes()
    assert (
        destination.read_bytes() == (ROOT / "paper" / "results_macros.tex").read_bytes()
    )
    assert text.endswith("\n")


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("protocol", "trial_count"), True),
        (("protocol", "bootstrap", "samples"), 100.0),
        (("summary", "narma_nrmse", "nonlinear", "mean"), "1.033"),
        (("summary", "paired_effects", "narma_nonlinear_minus_linear", "mean"), True),
        (
            (
                "summary",
                "paired_effects",
                "narma_nonlinear_minus_linear",
                "bootstrap_mean_ci95",
            ),
            [0.018, np.nan],
        ),
    ),
    ids=("trial-bool", "bootstrap-float", "mean-text", "effect-bool", "ci-nan"),
)
def test_report_macros_reject_invalid_numeric_fields(path, value):
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    target = results
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError)):
        reproduce.render_report_macros(results)


def test_write_report_macros_preserves_destination_when_validation_fails(tmp_path):
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    results["protocol"]["trial_count"] = True
    destination = tmp_path / "results_macros.tex"
    previous = b"previous validated macros\n"
    destination.write_bytes(previous)

    with pytest.raises(TypeError, match="trial_count"):
        reproduce.write_report_macros(results, destination)

    assert destination.read_bytes() == previous


def test_write_report_macros_invalid_payload_does_not_create_parent(tmp_path):
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    results["protocol"]["trial_count"] = True
    destination = tmp_path / "not-created" / "results_macros.tex"

    with pytest.raises(TypeError, match="trial_count"):
        reproduce.write_report_macros(results, destination)

    assert not destination.parent.exists()


def test_report_pdf_is_byte_reproducible_when_pdflatex_is_available(tmp_path):
    if shutil.which("pdflatex") is None:
        return

    root = tmp_path / "snapshot"
    shutil.copytree(ROOT / "paper", root / "paper")
    shutil.copytree(ROOT / "figures", root / "figures")
    first_path = reproduce.compile_report(root=root)
    first = first_path.read_bytes()
    second_path = reproduce.compile_report(root=root)
    second = second_path.read_bytes()

    assert first == second
