---
name: verify-onnx-construction
description: "Verify ONNX load construction isolation and factory restoration when changing scripts/onnx_runtime.py, its construction lock, temporary InferenceSession wrapper, or exception handling."
---

# Verify ONNX construction

The property: each `scripts/onnx_runtime.py:load` session receives that caller's
requested thread setting, and the original `ort.InferenceSession` factory is
restored after all helper calls finish, including constructor exceptions.

The harm: overlapping temporary overrides can give a caller the wrong CPU
thread setting and leave a stale wrapper installed for later session creation.

## Read this first

Work from the repository root. Read `scripts/onnx_runtime.py:load` (including
`cpu_session`), `formal/README.md`, `formal/Construction.tla`, and
`formal/Construction.cfg`. The guarded file is **`scripts/onnx_runtime.py`**;
its entire-file SHA-256 is in `formal/guarded.json`. A change to imports or
comments also triggers review. `formal/check_source.py` checks this on every run.

`formal/run.sh` needs Python 3 standard library, Temurin 21.0.12-patch1+1, and the
pinned TLA+ jar described in the README. It uses the shared installed tools,
checks the jar hash, uses one worker and a 1 GiB heap, and never downloads
dependencies or model weights. Set `JAVA_HOME` to the pinned JDK directory. `FV_JAVA` / `FV_TLA_JAR`
can override locations. Keep the regular `.agents/skills/` copy identical.
No Python packages or virtual environment are needed.

## Check a change

1. Run `formal/run.sh`. Exit 0 means the bounded property and actual-code replay
   pass. Exit 1 means drift, a new violation, or a tooling/model error. Exit 2 is
   reserved for explicitly recorded baseline findings; this clean baseline has
   none, and no failure is automatically accepted as a finding. Read all output.
2. If the guard reports source drift, inspect the diff against the version whose
   hash is recorded. Do not just replace the hash to get green. Trace every
   modeled step back to the function and revisit the assumptions below.
3. For an intended change, update `Construction.tla`, the config, actual-code
   replay, this skill, and README as needed. Keep steps at the code's true shared
   mutation boundaries. Add an interleaving step if code moves a shared read or
   write; never hide a concurrency problem inside an atomic constructor action.
4. After reviewing the mapping, refresh the guard with:

   ```sh
   python3 - <<'PY'
   import hashlib, json
   from pathlib import Path
   path = 'scripts/onnx_runtime.py'
   Path('formal/guarded.json').write_text(json.dumps({path: hashlib.sha256(Path(path).read_bytes()).hexdigest()}, indent=2) + '\n')
   PY
   ```

5. Rerun `formal/run.sh`, then the ordinary checks:

   ```sh
   python3 -m unittest discover -s tests -q
   python3 scripts/check_syntax.py
   ```

A TLC `ScopedConstruction` violation is a candidate bug, not permission to
weaken the invariant. Translate caller 1/2 to two `load` calls; `factory` to
`ort.InferenceSession`; `saved` to captured `original`; `setting` / `observed`
to session thread settings; Base=0 to no recipe override; NotCalled=-1 to no
session yet. Compare the trace to the actual code. A trace requiring a behavior
excluded by the assumptions is a modeling issue; document and correct the model
from code evidence. A realizable trace against current code stays a finding.
Record it in plain steps with severity and exposure; do not fix product code
without authorization. If a future baseline has findings, add explicit trace
matching before returning 2. Unexpected failures must remain exit 1.

## Demonstrate sensitivity on the real code

Only perform the next step when scratch product edits are authorized. They were
authorized for the initial formal sweep. Start with an unchanged guarded file;
preserve any unrelated work. Save a fresh passing log first:

```sh
formal/run.sh > formal/evidence/baseline.log 2>&1
python3 formal/seed_bug.py
```

Check the first command's exit status before continuing. `seed_bug.py` replaces
only `with _CONSTRUCTION_LOCK:` with
`with __import__('contextlib').nullcontext():` in the **actual** product file.
It saves the exact patch, proves the default runner rejects source drift, then
runs `formal/run.sh --scratch-unlocked`. That explicit mode reconstructs only
the known edit and checks the full original source hash before mirroring
`UseLock=FALSE`; it cannot bypass unrelated source drift. Do not refresh the
guard for this mutation or edit the invariant.

The mutation must fail both TLC and a replay of the actual `load` function:
A pauses inside ONNXAgent; B captures A's wrapper and pauses; A completes and
restores the base factory; B constructs an unconfigured session, then restores
A's stale wrapper. `seeded-failing.log` contains a TLC trace (its shortest
interleaving may differ from this replay). The replay uses fake dependencies,
real threads, events and a real locking wrapper; it installs no dependencies
and makes no external calls. Baseline replays also cover all four post-session
success/exception combinations. The seed script restores the original source
bytes in `finally`, reruns the baseline, and saves `reverted.log`. If another
edit arrives during the run it refuses to overwrite it; preserve and reconcile
the source before continuing.

Read `formal/evidence/source-drift.log`, `seeded-failing.log`,
`seeded-change.patch`, and `reverted.log`. Confirm exits 1, 1, and 0, respectively,
and confirm `git diff -- scripts/onnx_runtime.py` is empty. Rerun the ordinary
checks above. Summarize the trace in `formal/evidence/counterexample.md`; record
commands, versions, state counts and wall runtime in `formal/evidence/verification.md`.
Keep evidence free of credentials, hostnames, IPs and home paths.

## Scope

Two callers, one load and at most one session each, distinct positive thread
settings; synchronous external constructors may return or raise. TLC explores
all modeled interleavings and both exception positions. Shared factory reads,
captured originals, installation, restoration, and lock release are separate
steps. This is bounded safety, not liveness or a proof over arbitrary callers.
Only participating `load` callers are covered. No third-party concurrent calls
or writes to the ONNX factory, reentrant load, process termination, asynchronous
exceptions outside the try/finally, shared caller-supplied SessionOptions, model
correctness, provider execution, or upstream inference is verified. Review this
scope whenever the API or dependency construction behavior changes.
