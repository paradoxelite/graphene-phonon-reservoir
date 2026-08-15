"""Single entry point for the canonical numerical artifacts and report.

``python reproduce.py`` reruns the full paired protocol and regenerates the JSON,
figures and LaTeX macros.  ``--compile-report`` additionally builds the PDF in an
external temporary directory before replacing ``paper/main.pdf``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path

import study

ROOT = Path(__file__).resolve().parent
SOURCE_DATE_EPOCH = "1786665600"
GENERATED_ARTIFACTS = (
    Path("results.json"),
    Path("figures/paired_narma.png"),
    Path("figures/paired_parity.png"),
    Path("paper/results_macros.tex"),
)
PDF_ARTIFACT = Path("paper/main.pdf")
PDF_SOURCE = Path("paper/main.tex")
ARTIFACT_MANIFEST = Path("artifact_manifest.json")
TRANSACTION_JOURNAL = Path(".publication-transaction.json")
MANIFEST_BYTE_LIMIT = 16_384
TRANSACTION_JOURNAL_BYTE_LIMIT = 65_536
ARTIFACT_BYTE_LIMITS = {
    Path("results.json"): 4 * 1024 * 1024,
    Path("figures/paired_narma.png"): 8 * 1024 * 1024,
    Path("figures/paired_parity.png"): 8 * 1024 * 1024,
    Path("paper/results_macros.tex"): 256 * 1024,
    PDF_ARTIFACT: 32 * 1024 * 1024,
    ARTIFACT_MANIFEST: MANIFEST_BYTE_LIMIT,
}


class UnsafePublicationPathError(ValueError):
    """A publication path can escape through a symlink or reparse point."""


class PublicationReleaseError(RuntimeError):
    """Publication crossed the commit boundary but release/cleanup failed."""

    def __init__(self, message: str, *, committed: bool, manifest_valid: bool):
        super().__init__(message)
        self.committed = committed
        self.manifest_valid = manifest_valid


class PublicationRecoveryError(RuntimeError):
    """A durable interrupted publication cannot be restored safely."""


def _require_boolean(name: str, value) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without following filesystem indirections."""
    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def _component_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_ctime_ns,
        getattr(metadata, "st_file_attributes", 0),
    )


def _no_reparse_component_snapshot(
    path: Path,
    *,
    require_leaf: bool,
) -> tuple[tuple[Path, os.stat_result], ...]:
    """Capture existing physical components without following indirections."""
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    snapshot = []
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if _is_reparse(metadata):
            raise UnsafePublicationPathError(
                f"publication path contains a symlink or reparse point: {current}"
            )
        snapshot.append((current, metadata))
    if require_leaf and (
        not snapshot
        or os.path.normcase(os.fspath(snapshot[-1][0]))
        != os.path.normcase(os.fspath(absolute))
    ):
        raise FileNotFoundError(absolute)
    return tuple(snapshot)


def _component_snapshot_identity(snapshot) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (os.path.normcase(os.fspath(path)), _component_identity(metadata))
        for path, metadata in snapshot
    )


def _assert_no_reparse_components(path: Path) -> None:
    """Reject every existing symlink/junction component without following it."""
    _no_reparse_component_snapshot(path, require_leaf=False)


def _safe_publication_root(root: Path) -> Path:
    absolute = _absolute_lexical(root)
    _assert_no_reparse_components(absolute)
    return absolute


def _safe_destination(root: Path, relative: Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise UnsafePublicationPathError(f"unsafe publication path: {relative}")
    destination = _absolute_lexical(root / relative)
    root_text = os.path.normcase(os.fspath(root))
    common = os.path.normcase(os.path.commonpath((root, destination)))
    if common != root_text:
        raise UnsafePublicationPathError(f"publication path escapes root: {relative}")
    _assert_no_reparse_components(destination)
    return destination


def _remove_tree(path: Path, *, attempts: int = 5) -> None:
    """Remove a disposable stage with bounded retries for Windows handle lag."""
    last_error = None
    for attempt in range(attempts):
        try:
            metadata = os.lstat(path)
            if _is_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                raise UnsafePublicationPathError(
                    f"refusing to remove unsafe transaction stage: {path}"
                )
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.05 * (attempt + 1))
    if path.exists() and last_error is not None:
        raise last_error


@contextmanager
def _artifact_lock(root: Path, *, timeout: float = 30.0):
    """Hold a crash-released OS lock for one repository artifact set."""
    root = _safe_publication_root(root)
    _assert_no_reparse_components(root.parent)
    root.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(root.parent)
    lock_path = root.parent / f".{root.name}.artifact-publication.lock"
    _assert_no_reparse_components(lock_path)
    deadline = time.monotonic() + timeout
    stream = lock_path.open("a+b", buffering=0)
    acquired = False
    try:
        _assert_no_reparse_components(lock_path)
        path_metadata = os.lstat(lock_path)
        handle_metadata = os.fstat(stream.fileno())
        if _is_reparse(path_metadata) or (
            path_metadata.st_ino
            and handle_metadata.st_ino
            and (path_metadata.st_dev, path_metadata.st_ino)
            != (handle_metadata.st_dev, handle_metadata.st_ino)
        ):
            raise UnsafePublicationPathError(
                f"publication lock changed identity or is a reparse point: {lock_path}"
            )
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
        while not acquired:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"artifact publication lock timed out: {lock_path}") from error
                time.sleep(0.05)
        yield
    finally:
        unlock_error = None
        close_error = None
        try:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except BaseException as error:
            unlock_error = error
        finally:
            try:
                stream.close()
            except BaseException as error:
                close_error = error
        if unlock_error is not None:
            if close_error is not None:
                unlock_error.add_note(
                    f"lock handle close also failed: {type(close_error).__name__}: "
                    f"{close_error}"
                )
            raise unlock_error
        if close_error is not None:
            raise close_error


def _canonical_artifact_set(artifacts) -> tuple[Path, ...]:
    selected = tuple(Path(relative) for relative in artifacts)
    allowed = (
        GENERATED_ARTIFACTS,
        GENERATED_ARTIFACTS + (PDF_ARTIFACT,),
    )
    if selected not in allowed:
        raise ValueError("artifact set must be the canonical four or five payloads")
    return selected


def write_artifact_manifest(root: Path, artifacts) -> Path:
    """Write canonical trust metadata for one complete payload generation."""
    root = Path(root)
    selected = _canonical_artifact_set(artifacts)
    records = {}
    for relative in selected:
        record = _file_record(
            root / relative,
            max_bytes=ARTIFACT_BYTE_LIMITS[relative],
        )
        records[relative.as_posix()] = record
    destination = root / ARTIFACT_MANIFEST
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        {"schema_version": 1, "artifacts": records},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return destination


def verify_artifact_manifest(
    root: Path,
    *,
    expected_artifacts,
) -> bool:
    """Fail closed unless the marker and every declared payload match exactly."""
    root = Path(root)
    selected = _canonical_artifact_set(expected_artifacts)
    manifest_path = root / ARTIFACT_MANIFEST

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate manifest key: {key}")
            result[key] = value
        return result

    try:
        raw = _read_regular_bytes(manifest_path, max_bytes=MANIFEST_BYTE_LIMIT)
        if not raw.endswith(b"\n") or b"\r\n" in raw:
            raise RuntimeError("invalid manifest size or line endings")
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        if set(manifest) != {"schema_version", "artifacts"}:
            raise RuntimeError("invalid manifest fields")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise RuntimeError("unsupported manifest schema")
        records = manifest["artifacts"]
        expected_names = {relative.as_posix() for relative in selected}
        if not isinstance(records, dict) or set(records) != expected_names:
            raise RuntimeError("unexpected manifest inventory")
        for relative in selected:
            name = relative.as_posix()
            record = records[name]
            if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
                raise RuntimeError(f"invalid record fields: {name}")
            size = record["size"]
            digest = record["sha256"]
            limit = ARTIFACT_BYTE_LIMITS[relative]
            if type(size) is not int or size < 1 or size > limit:
                raise RuntimeError(f"invalid record size: {name}")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"invalid record digest: {name}")
            payload_record = _file_record(root / relative, max_bytes=limit)
            if payload_record != {"sha256": digest, "size": size}:
                raise RuntimeError(f"payload mismatch: {name}")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise RuntimeError(f"artifact manifest mismatch: {error}") from error
    return True


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory flush; Windows may reject directory handles."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _durable_json_replace(path: Path, value: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    _assert_no_reparse_components(path)
    _assert_no_reparse_components(temporary)
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _assert_no_reparse_components(path)
    _fsync_directory(path.parent)


def _metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _descriptor_bound_read(
    path: Path,
    *,
    max_bytes: int,
    capture: bool,
) -> tuple[bytes | None, dict]:
    """Read one stable regular file through a bounded, identity-bound descriptor."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive exact integer")
    path = _absolute_lexical(path)
    pre_open_snapshot = _no_reparse_component_snapshot(path, require_leaf=True)
    pre_open_named = pre_open_snapshot[-1][1]
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        opened_snapshot = _no_reparse_component_snapshot(path, require_leaf=True)
        named = opened_snapshot[-1][1]
        if (
            _component_snapshot_identity(pre_open_snapshot)
            != _component_snapshot_identity(opened_snapshot)
            or _metadata_signature(pre_open_named) != _metadata_signature(opened)
            or _is_reparse(named)
            or not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or not os.path.samestat(opened, named)
        ):
            raise RuntimeError(f"file changed identity before or while opening: {path}")
        if opened.st_size < 1 or opened.st_size > max_bytes:
            raise RuntimeError(f"file size is outside its physical bound: {path}")

        digest = hashlib.sha256()
        chunks = [] if capture else None
        total = 0
        while True:
            remaining = max_bytes + 1 - total
            if remaining <= 0:
                raise RuntimeError(f"file exceeded its physical bound while reading: {path}")
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)

        closed = os.fstat(descriptor)
        final_snapshot = _no_reparse_component_snapshot(path, require_leaf=True)
        named_after = final_snapshot[-1][1]
        if (
            _component_snapshot_identity(pre_open_snapshot)
            != _component_snapshot_identity(final_snapshot)
            or _metadata_signature(pre_open_named) != _metadata_signature(closed)
            or _metadata_signature(opened) != _metadata_signature(closed)
            or _is_reparse(named_after)
            or not stat.S_ISREG(named_after.st_mode)
            or not os.path.samestat(closed, named_after)
            or total != opened.st_size
        ):
            raise RuntimeError(f"file changed identity or bytes while reading: {path}")
        data = b"".join(chunks) if chunks is not None else None
        return data, {"sha256": digest.hexdigest(), "size": total}
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    payload, _record = _descriptor_bound_read(path, max_bytes=max_bytes, capture=True)
    if payload is None:
        raise RuntimeError("descriptor reader did not capture requested bytes")
    return payload


def _file_record(path: Path, *, max_bytes: int) -> dict:
    _payload, record = _descriptor_bound_read(path, max_bytes=max_bytes, capture=False)
    return record


def _record_matches(path: Path, record: dict, *, max_bytes: int) -> bool:
    _assert_no_reparse_components(path)
    try:
        metadata = os.lstat(path)
        if _is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            return False
        return _file_record(path, max_bytes=max_bytes) == {
            "sha256": record["sha256"],
            "size": record["size"],
        }
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False


def _transaction_root_id(root: Path) -> str:
    return os.path.normcase(os.fspath(_absolute_lexical(root)))


def _installation_sets() -> tuple[tuple[Path, ...], ...]:
    return (
        GENERATED_ARTIFACTS + (ARTIFACT_MANIFEST,),
        GENERATED_ARTIFACTS + (PDF_ARTIFACT, ARTIFACT_MANIFEST),
    )


def _prepare_transaction_journal(
    root: Path,
    stage: Path,
    installation_artifacts: tuple[Path, ...],
) -> tuple[dict[Path, Path], dict[Path, bool], dict]:
    _assert_no_reparse_components(stage)
    if installation_artifacts not in _installation_sets():
        raise ValueError("invalid installation artifact set")
    destinations = {}
    existed = {}
    entries = []
    for relative in installation_artifacts:
        destination = _safe_destination(root, relative)
        _assert_no_reparse_components(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_components(destination)
        destinations[relative] = destination
        try:
            metadata = os.lstat(destination)
        except FileNotFoundError:
            present = False
            record = {"sha256": None, "size": None}
        else:
            if _is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise UnsafePublicationPathError(
                    f"publication destination is not a regular file: {destination}"
                )
            present = True
            record = _file_record(
                destination,
                max_bytes=ARTIFACT_BYTE_LIMITS[relative],
            )
        existed[relative] = present
        entries.append(
            {
                "path": relative.as_posix(),
                "existed": present,
                **record,
            }
        )
    journal = {
        "schema_version": 1,
        "root": _transaction_root_id(root),
        "artifacts": entries,
    }
    _durable_json_replace(stage / TRANSACTION_JOURNAL, journal)
    return destinations, existed, journal


def _pending_transaction_stages(root: Path) -> list[Path]:
    stages = []
    pattern = f".{root.name}-artifacts-*"
    for stage in root.parent.glob(pattern):
        try:
            stage_metadata = os.lstat(stage)
        except FileNotFoundError:
            continue
        if _is_reparse(stage_metadata):
            raise UnsafePublicationPathError(
                f"transaction stage is a reparse point: {stage}"
            )
        if not stat.S_ISDIR(stage_metadata.st_mode):
            continue
        journal = stage / TRANSACTION_JOURNAL
        try:
            journal_metadata = os.lstat(journal)
        except FileNotFoundError:
            continue
        if _is_reparse(journal_metadata) or not stat.S_ISREG(journal_metadata.st_mode):
            raise UnsafePublicationPathError(
                f"transaction journal is not a regular file: {journal}"
            )
        stages.append(stage)
    return sorted(stages, key=lambda path: os.path.normcase(os.fspath(path)))


def _load_transaction_journal(root: Path, stage: Path):
    journal_path = stage / TRANSACTION_JOURNAL

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PublicationRecoveryError(f"duplicate journal key: {key}")
            result[key] = value
        return result

    try:
        raw = _read_regular_bytes(
            journal_path,
            max_bytes=TRANSACTION_JOURNAL_BYTE_LIMIT,
        )
        if not raw.endswith(b"\n") or b"\r\n" in raw:
            raise PublicationRecoveryError("invalid journal size or line endings")
        journal = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise PublicationRecoveryError(f"cannot read transaction journal: {error}") from error
    if not isinstance(journal, dict) or set(journal) != {
        "schema_version",
        "root",
        "artifacts",
    }:
        raise PublicationRecoveryError("invalid transaction journal fields")
    if type(journal["schema_version"]) is not int or journal["schema_version"] != 1:
        raise PublicationRecoveryError("unsupported transaction journal schema")
    if journal["root"] != _transaction_root_id(root):
        raise PublicationRecoveryError("transaction journal root identity mismatch")
    records = journal["artifacts"]
    if not isinstance(records, list):
        raise PublicationRecoveryError("transaction journal artifacts must be a list")
    paths = []
    normalized = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "existed",
            "sha256",
            "size",
        }:
            raise PublicationRecoveryError("invalid transaction artifact record")
        if not isinstance(record["path"], str):
            raise PublicationRecoveryError("transaction path must be text")
        relative = Path(record["path"])
        _safe_destination(root, relative)
        try:
            limit = ARTIFACT_BYTE_LIMITS[relative]
        except KeyError as error:
            raise PublicationRecoveryError(
                f"unexpected transaction artifact: {relative}"
            ) from error
        if type(record["existed"]) is not bool:
            raise PublicationRecoveryError("transaction existed marker must be boolean")
        if record["existed"]:
            digest = record["sha256"]
            size = record["size"]
            if (
                type(size) is not int
                or size < 1
                or size > limit
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise PublicationRecoveryError("invalid original artifact fingerprint")
        elif record["sha256"] is not None or record["size"] is not None:
            raise PublicationRecoveryError("absent artifact cannot have a fingerprint")
        paths.append(relative)
        normalized.append(record)
    selected = tuple(paths)
    if selected not in _installation_sets() or len(set(paths)) != len(paths):
        raise PublicationRecoveryError("unexpected transaction artifact inventory")
    return selected, normalized


def _recover_pending_transactions(root: Path) -> bool:
    pending = _pending_transaction_stages(root)
    if not pending:
        return False
    if len(pending) != 1:
        raise PublicationRecoveryError(
            f"ambiguous pending transactions: {len(pending)}"
        )
    stage = pending[0]
    _assert_no_reparse_components(stage)
    installation_artifacts, records = _load_transaction_journal(root, stage)
    backup_root = stage / ".backups"
    restore_root = stage / ".recovery"
    for relative, record in zip(installation_artifacts, records, strict=True):
        destination = _safe_destination(root, relative)
        if record["existed"]:
            backup = backup_root / relative
            _assert_no_reparse_components(backup)
            if _record_matches(
                backup,
                record,
                max_bytes=ARTIFACT_BYTE_LIMITS[relative],
            ):
                restore = restore_root / relative
                _assert_no_reparse_components(restore)
                restore.parent.mkdir(parents=True, exist_ok=True)
                _assert_no_reparse_components(restore)
                shutil.copy2(backup, restore)
                _fsync_file(restore)
                _safe_destination(root, relative)
                os.replace(restore, destination)
            elif not _record_matches(
                destination,
                record,
                max_bytes=ARTIFACT_BYTE_LIMITS[relative],
            ):
                raise PublicationRecoveryError(
                    f"no valid original or backup remains for {relative}"
                )
        else:
            try:
                metadata = os.lstat(destination)
            except FileNotFoundError:
                pass
            else:
                if _is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                    raise PublicationRecoveryError(
                        f"cannot remove unsafe recovered destination: {relative}"
                    )
                destination.unlink()
        if record["existed"] and not _record_matches(
            destination,
            record,
            max_bytes=ARTIFACT_BYTE_LIMITS[relative],
        ):
            raise PublicationRecoveryError(f"restoration mismatch: {relative}")
        if not record["existed"] and os.path.lexists(destination):
            raise PublicationRecoveryError(f"absent artifact survived recovery: {relative}")
    manifest_record = records[-1]
    if manifest_record["existed"]:
        verify_artifact_manifest(
            root,
            expected_artifacts=installation_artifacts[:-1],
        )
    _remove_tree(stage)
    return True


def _report_integer(name: str, value, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _report_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _report_ci(name: str, values) -> tuple[float, float]:
    if type(values) is not list or len(values) != 2:
        raise TypeError(f"{name} must be a two-element list")
    return (
        _report_real(f"{name}[0]", values[0]),
        _report_real(f"{name}[1]", values[1]),
    )


def _mean(summary: dict, metric: str, condition: str) -> float:
    return _report_real(
        f"summary.{metric}.{condition}.mean",
        summary[metric][condition]["mean"],
    )


def _ci(summary: dict, metric: str, condition: str) -> tuple[float, float]:
    return _report_ci(
        f"summary.{metric}.{condition}.bootstrap_mean_ci95",
        summary[metric][condition]["bootstrap_mean_ci95"],
    )


def render_report_macros(results: dict) -> str:
    summary = results["summary"]
    protocol = results["protocol"]
    narma_effect = summary["paired_effects"]["narma_nonlinear_minus_linear"]
    narma_delay_effect = summary["paired_effects"]["narma_nonlinear_minus_delay"]
    parity_effect = summary["paired_effects"]["parity_nonlinear_minus_linear"]
    narma_effect_ci = _report_ci(
        "summary.paired_effects.narma_nonlinear_minus_linear.bootstrap_mean_ci95",
        narma_effect["bootstrap_mean_ci95"],
    )
    narma_delay_effect_ci = _report_ci(
        "summary.paired_effects.narma_nonlinear_minus_delay.bootstrap_mean_ci95",
        narma_delay_effect["bootstrap_mean_ci95"],
    )
    parity_effect_ci = _report_ci(
        "summary.paired_effects.parity_nonlinear_minus_linear.bootstrap_mean_ci95",
        parity_effect["bootstrap_mean_ci95"],
    )

    values = {
        "TrialCount": str(
            _report_integer("protocol.trial_count", protocol["trial_count"], minimum=1)
        ),
        "NarmaNonlinearMean": f"{_mean(summary, 'narma_nrmse', 'nonlinear'):.3f}",
        "NarmaLinearMean": f"{_mean(summary, 'narma_nrmse', 'linear_mechanics'):.3f}",
        "NarmaDelayMean": f"{_mean(summary, 'narma_nrmse', 'delay_line'):.3f}",
        "NarmaNonlinearMinusLinear": f"{_report_real('summary.paired_effects.narma_nonlinear_minus_linear.mean', narma_effect['mean']):.3f}",
        "NarmaNonlinearMinusLinearLow": f"{narma_effect_ci[0]:.3f}",
        "NarmaNonlinearMinusLinearHigh": f"{narma_effect_ci[1]:.3f}",
        "NarmaNonlinearMinusDelay": f"{_report_real('summary.paired_effects.narma_nonlinear_minus_delay.mean', narma_delay_effect['mean']):.3f}",
        "NarmaNonlinearMinusDelayLow": f"{narma_delay_effect_ci[0]:.3f}",
        "NarmaNonlinearMinusDelayHigh": f"{narma_delay_effect_ci[1]:.3f}",
        "ParityNonlinearMean": f"{_mean(summary, 'parity_accuracy', 'nonlinear'):.3f}",
        "ParityLinearMean": f"{_mean(summary, 'parity_accuracy', 'linear_mechanics'):.3f}",
        "ParityDelayMean": f"{_mean(summary, 'parity_accuracy', 'delay_line'):.3f}",
        "ParityNonlinearMinusLinear": f"{_report_real('summary.paired_effects.parity_nonlinear_minus_linear.mean', parity_effect['mean']):.3f}",
        "ParityNonlinearMinusLinearLow": f"{parity_effect_ci[0]:.3f}",
        "ParityNonlinearMinusLinearHigh": f"{parity_effect_ci[1]:.3f}",
        "BootstrapSamples": str(
            _report_integer(
                "protocol.bootstrap.samples",
                protocol["bootstrap"]["samples"],
                minimum=1,
            )
        ),
        "NarmaNonlinearLow": f"{_ci(summary, 'narma_nrmse', 'nonlinear')[0]:.3f}",
        "NarmaNonlinearHigh": f"{_ci(summary, 'narma_nrmse', 'nonlinear')[1]:.3f}",
        "ParityNonlinearLow": f"{_ci(summary, 'parity_accuracy', 'nonlinear')[0]:.3f}",
        "ParityNonlinearHigh": f"{_ci(summary, 'parity_accuracy', 'nonlinear')[1]:.3f}",
        "MinimumAirGapFraction": f"{_report_real('domain_checks.minimum_air_gap_fraction_observed', results['domain_checks']['minimum_air_gap_fraction_observed']):.3f}",
        "ContactThresholdFraction": f"{_report_real('domain_checks.contact_threshold_fraction', results['domain_checks']['contact_threshold_fraction']):.3f}",
    }
    return "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items()
    )


def write_report_macros(results: dict, destination: str | Path) -> None:
    payload = render_report_macros(results)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def publish_artifacts(
    results: dict,
    *,
    root: Path = ROOT,
    compile_pdf: bool = False,
) -> tuple[Path, ...]:
    """Generate off-tree and install with exhaustive rollback on commit failure."""
    compile_pdf = _require_boolean("compile_pdf", compile_pdf)
    root = _safe_publication_root(root)
    preflight_paths = GENERATED_ARTIFACTS + (PDF_ARTIFACT, ARTIFACT_MANIFEST)
    if compile_pdf:
        preflight_paths += (PDF_SOURCE,)
    for relative in preflight_paths:
        _safe_destination(root, relative)
    _assert_no_reparse_components(root.parent)
    root.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(root.parent)
    if _pending_transaction_stages(root):
        with _artifact_lock(root):
            _recover_pending_transactions(root)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name}-artifacts-", dir=root.parent)
    )
    _assert_no_reparse_components(stage)
    preserve_stage = False
    publication_lock = None
    lock_entered = False
    committed = False
    try:
        study.write_results(results, stage / "results.json")
        generated = study.plot_results(results, stage / "figures")
        if {path.name for path in generated} != {
            "paired_narma.png",
            "paired_parity.png",
        }:
            raise RuntimeError("plot producer returned an unexpected artifact set")
        write_report_macros(results, stage / "paper" / "results_macros.tex")
        artifacts = GENERATED_ARTIFACTS
        if compile_pdf:
            main_tex = root / PDF_SOURCE
            if not main_tex.is_file():
                raise RuntimeError("paper/main.tex is required to compile the report")
            shutil.copyfile(main_tex, stage / "paper" / "main.tex")
            compile_report(root=stage)
            artifacts = GENERATED_ARTIFACTS + (PDF_ARTIFACT,)

        write_artifact_manifest(stage, artifacts)
        verify_artifact_manifest(stage, expected_artifacts=artifacts)
        installation_artifacts = artifacts + (ARTIFACT_MANIFEST,)

        publication_lock = _artifact_lock(root)
        publication_lock.__enter__()
        lock_entered = True
        _recover_pending_transactions(root)
        staged_records = {}
        for relative in installation_artifacts:
            source = stage / relative
            staged_records[relative] = _file_record(
                source,
                max_bytes=ARTIFACT_BYTE_LIMITS[relative],
            )

        destinations, existed, transaction = _prepare_transaction_journal(
            root,
            stage,
            installation_artifacts,
        )
        records = {
            Path(record["path"]): record for record in transaction["artifacts"]
        }
        backups = {}
        backup_root = stage / ".backups"
        for relative in installation_artifacts:
            if existed[relative]:
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destinations[relative], backup)
                _fsync_file(backup)
                if not _record_matches(
                    backup,
                    records[relative],
                    max_bytes=ARTIFACT_BYTE_LIMITS[relative],
                ) or not _record_matches(
                    destinations[relative],
                    records[relative],
                    max_bytes=ARTIFACT_BYTE_LIMITS[relative],
                ):
                    raise RuntimeError(f"backup snapshot mismatch: {relative}")
                backups[relative] = backup

        moved = []
        try:
            for relative in installation_artifacts:
                moved.append(relative)
                _safe_destination(root, relative)
                os.replace(stage / relative, destinations[relative])
            for relative in installation_artifacts:
                installed_record = _file_record(
                    destinations[relative],
                    max_bytes=ARTIFACT_BYTE_LIMITS[relative],
                )
                if installed_record != staged_records[relative]:
                    raise RuntimeError(f"post-commit mismatch: {relative}")
            verify_artifact_manifest(root, expected_artifacts=artifacts)
            (stage / TRANSACTION_JOURNAL).unlink()
            _fsync_directory(stage)
        except BaseException as original:
            rollback_errors = []
            rollback_control = None
            restore_root = stage / ".restore"
            for relative in reversed(moved):
                destination = destinations[relative]
                try:
                    if existed[relative]:
                        restore = restore_root / relative
                        restore.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backups[relative], restore)
                        _fsync_file(restore)
                        _safe_destination(root, relative)
                        os.replace(restore, destination)
                    else:
                        _safe_destination(root, relative)
                        destination.unlink(missing_ok=True)
                except BaseException as rollback_error:
                    rollback_errors.append((relative, rollback_error))
                    if not isinstance(rollback_error, Exception) and rollback_control is None:
                        rollback_control = rollback_error
            if rollback_errors:
                preserve_stage = True
                details = "; ".join(
                    f"{relative}: {type(error).__name__}: {error}"
                    for relative, error in rollback_errors
                )
                note = (
                    f"rollback incomplete; immutable backups preserved at {backup_root}; "
                    f"{details}"
                )
                if isinstance(original, Exception) and rollback_control is not None:
                    rollback_control.add_note(note)
                    raise rollback_control
                original.add_note(note)
            raise
        committed = True
        return tuple(destinations[relative] for relative in installation_artifacts)
    finally:
        primary_type, primary_error, primary_traceback = sys.exc_info()
        release_error = None
        cleanup_error = None
        if lock_entered:
            try:
                publication_lock.__exit__(
                    primary_type,
                    primary_error,
                    primary_traceback,
                )
            except BaseException as error:
                release_error = error
        if not preserve_stage:
            try:
                _remove_tree(stage)
            except BaseException as error:
                cleanup_error = error
        if primary_error is not None:
            if release_error is not None:
                primary_error.add_note(
                    f"lock release failed: {type(release_error).__name__}: "
                    f"{release_error}"
                )
            if cleanup_error is not None:
                primary_error.add_note(
                    f"stage cleanup failed: {type(cleanup_error).__name__}: "
                    f"{cleanup_error}; residual stage: {stage}"
                )
        elif release_error is not None or cleanup_error is not None:
            manifest_valid = False
            if committed:
                try:
                    manifest_valid = verify_artifact_manifest(
                        root,
                        expected_artifacts=artifacts,
                    )
                except BaseException:
                    manifest_valid = False
            details = []
            if release_error is not None:
                details.append(
                    f"lock release failed: {type(release_error).__name__}: "
                    f"{release_error}"
                )
            if cleanup_error is not None:
                details.append(
                    f"stage cleanup failed: {type(cleanup_error).__name__}: "
                    f"{cleanup_error}; residual stage: {stage}"
                )
            error = PublicationReleaseError(
                "; ".join(details),
                committed=committed,
                manifest_valid=manifest_valid,
            )
            cause = release_error if release_error is not None else cleanup_error
            raise error from cause


def regenerate(*, root: Path = ROOT, compile_pdf: bool = False) -> dict:
    compile_pdf = _require_boolean("compile_pdf", compile_pdf)
    kwargs = {
        "trials": study.protocol_trials(12),
        "length": 1200,
        "washout": 200,
        "n_modes": 16,
        "bootstrap_samples": 10_000,
    }
    results = study.run_protocol(**kwargs)
    publish_artifacts(results, root=root, compile_pdf=compile_pdf)
    return results


def compile_report(*, root: Path = ROOT) -> Path:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required for --compile-report")
    paper = root / "paper"
    with tempfile.TemporaryDirectory(prefix="graphene-report-") as temporary:
        stage = Path(temporary)
        staged_paper = stage / "paper"
        staged_figures = stage / "figures"
        staged_paper.mkdir()
        staged_figures.mkdir()
        shutil.copyfile(paper / "main.tex", staged_paper / "main.tex")
        shutil.copyfile(
            paper / "results_macros.tex",
            staged_paper / "results_macros.tex",
        )
        for figure in ("paired_narma.png", "paired_parity.png"):
            shutil.copyfile(root / "figures" / figure, staged_figures / figure)
        command = [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ]
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
        environment["FORCE_SOURCE_DATE"] = "1"
        for _ in range(2):
            subprocess.run(
                command,
                cwd=staged_paper,
                check=True,
                capture_output=True,
                env=environment,
            )
        source = staged_paper / "main.pdf"
        destination = paper / "main.pdf"
        shutil.copyfile(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-report", action="store_true")
    args = parser.parse_args()

    results = regenerate(compile_pdf=args.compile_report)
    print(f"Resultados: {(ROOT / 'results.json').resolve()}")
    print(f"Trials: {results['protocol']['trial_count']}")
    if args.compile_report:
        print(f"Reporte: {(ROOT / PDF_ARTIFACT).resolve()}")


if __name__ == "__main__":
    main()
