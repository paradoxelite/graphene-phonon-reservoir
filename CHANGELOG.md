# Changelog

## 1.0.0 — 2026-08-14

- Replaced the unreconciled exploratory tree with a minimal reduced-order simulation.
- Added gap-dependent electrostatics, one authoritative optical gap and fail-closed contact.
- Added a matched mechanically linear ablation and a separate digital delay baseline.
- Froze twelve paired seeds, chronological train/test separation and bootstrap summaries.
- Published the negative result: no nonlinear advantage in NARMA-10 or parity-3.
- Removed unsupported hardware, int8, application, power, cost, fabrication and universal-capacity claims.
- Added exact environment locks, clean-import tests, deterministic artifact generation and a byte-reproducible PDF build.
- Corrected and verified the DOI for the mechanical reservoir reference.
- Corrected the passive `n+iκ` TMM sign convention and added an independent recursive-Fresnel regression test; regenerated every affected metric and artifact.
- Made CI cache-neutral: compile bytecode outside the checkout, disable implicit bytecode and pytest cache creation, and test the exact CI contract.
- Reconciled mode frequencies as seeded independent uniform draws and serialized every realized frequency per trial.
- Serialized all ten effective bootstrap sub-seeds instead of exposing only an ambiguous base seed.
- Removed retired XOR, memory-capacity and digit-loading experiments from the publication module.
- Added fail-closed parameter/input validation and rejected duplicate or cross-role seed reuse.
- Made five payloads a rollback-safe publication under an OS lock and added a strict size/SHA-256 manifest installed last; fault injection covers moved-then-raised commits and rollback interruption.
- Added a durable pre-backup transaction journal and idempotent next-invocation recovery; process-death tests cover every backup and public replacement, including the manifest.
- Rejected symlinks, Windows junctions and other reparse points throughout publication roots, payload parents, lock routes, immutable backups and recovery paths.
- Classified unlock/cleanup failures without masking primary errors and report committed/manifest-valid state after a successful commit.
- Enforced exact non-boolean integer, real-scalar and boolean contracts across model, task, fit and publication APIs.
- Pinned GitHub Actions by commit SHA, removed the unfrozen pip upgrade and required lock hashes explicitly.
- Made the documented clean-room replay fail closed: venv and caches live outside the export; inherited Python and `PIP_*` variables, user-site and effective pip config files are disabled; Python 3.11.9 is checked exactly; TLS uses pip's system trust-store feature without trusted-host bypasses; and a fresh hash-locked PowerShell replay is exercised end to end.
- Rejected numeric text, complex drives and heterogeneous boolean/real or boolean/integer values without losing provenance: every array-like input is materialized once, its original scalar types are validated, and that same snapshot is converted. Model drives, the delay line and both audited fits share this path; fits validate both feature matrices `X` and targets `y`, including generic and stateful array-like providers. Preserved the historical valid empty delay signal as a `(0, order)` matrix and added public acceptance/rejection tests for generic and stateful providers.
- Made evidence emission fail closed: result JSON rejects `NaN`/`Infinity` before creating directories or touching its destination; report macros validate completely before opening an existing destination and accept only exact integer counts plus finite real metrics/confidence bounds.
- Replaced the unavailable `${{ runner.temp }}` expression in job-level CI environment with `/tmp/graphene-pycache`; a regression and Actionlint 1.7.12 verify that the workflow can be admitted before job creation.
- Removed the legacy `study.py` CLI so `reproduce.py` is the sole executable artifact producer.
- Bound every trust-sensitive manifest, payload, staging, commit and recovery-journal read to one descriptor with explicit byte caps, incremental SHA-256 and before/after physical-identity checks; added growth and same-size path-substitution falsifiers.
- Made the Windows file-symlink security regression non-skippable by falling back to deterministic reparse metadata injection when WinError 1314 prevents creating a real symlink.
- Moved plot legends outside the data axes and added a renderer-level regression that rejects any legend box containing a data marker.
- Preserved and compared `lstat` identities for every path component before `os.open`, immediately after opening and after reading; an identical-byte pre-open replacement now fails closed.
- Defined the external binary audit framing explicitly as ordered `u64be(path length) + UTF-8 path + u64be(size) + raw SHA-256`, preventing an unlabeled digest from being mistaken for another framing algorithm.
