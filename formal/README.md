# ONNX construction scope

Property: each `scripts/onnx_runtime.py:load` session receives that caller's
requested thread setting, and the original `ort.InferenceSession` factory is
restored after all helper calls finish, including constructor exceptions.

Harm: overlapping temporary overrides can give a caller the wrong CPU thread
setting and leave a stale wrapper installed for later session creation.

Run from the repository root:

```sh
formal/run.sh
```

Exit **0**: all checks pass. Exit **1**: unexpected violation, stale source,
tooling error, or deliberately seeded failure. Exit **2**: reserved for recorded
baseline findings still present; there are currently none. The runner never
silently accepts a failing invariant as a baseline. Future findings need exact
classification and evidence before adding an exit-2 path.

## Why this target and tool

This small recipe delegates inference upstream. Reviewed areas were checkpoint
and ONNX artifact validation, parity distribution parsing and gate calculation,
the two benchmark harnesses, and ONNX construction. Hash validation and the
parity gate already have ordinary boundary tests. The global factory override
is the one concurrency protocol here, explicitly promises serialization and
restoration, and has no concurrency coverage in the existing 19 tests.
Its small size does not eliminate the risk of an interleaving regression.

TLA+/TLC explores the orderings around `_CONSTRUCTION_LOCK`, `original`, the
factory assignment, session calls, and `finally`. A per-input solver would not
answer this ordering question as directly. `Construction.tla` cites the real
file and function for every declaration, action, and property. The Python
replay also executes the actual product function with synthetic dependencies;
it connects a concrete counterexample to the abstract model without downloading
weights or claiming to verify Laya/ORT.

## Tools and cost

The runner uses Python 3's standard library. Its defaults are:

- Java: `JAVA_HOME/bin/java`, then `java` on PATH, then a JDK under
  `~/.local/share/fv-tools/temurin-*/bin/java`.
- Set `JAVA_HOME` to the extracted pinned JDK directory before running.
- Jar: `~/.local/share/fv-tools/tla2tools-v1.8.0.jar`.
- Override with `FV_JAVA` and `FV_TLA_JAR` (paths, not command arguments).
- Required jar SHA-256:
  `15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac`.

The installed jar's filename is v1.8.0; its actual TLC banner is
`TLC2 Version 2026.10.15.145915 (rev: 932d35e)`. Evidence records both the pinned
identity and actual banner. Java must report Temurin 21.0.12-patch1. Missing or
wrong tools fail closed; no download, system install, or service change occurs.
Provide the pinned verified tools at those paths or through the overrides.
Jars, local JDK folders and a potential `formal/.venv/` are gitignored.

TLC uses exactly one worker, `-Xmx1g`, a fixed fingerprint/seed and a 120-second
subprocess limit. Temporary model/config/state copies live in ignored
`formal/.work/` and are removed after execution. Typical baseline cost is about
two seconds. No Python extras or virtual environment are needed.

## Model and assumptions

Two callers have distinct positive thread settings, represented by IDs 1 and 2.
The real-code witness uses 2 and 4 threads. Base=0 is an unmodified factory or
an unconfigured options sentinel; it does **not** assert ORT defaults to zero
worker threads. NotCalled=-1 means no session has been requested.

Each caller enters the lock, captures `original`, publishes its wrapper, may
raise before making a session or make one session and then return/raise,
restores `original` in finally, and exits the lock. Session construction is
split into resolving the module callable, applying a wrapper's thread setting,
forwarding to its captured original, and observing options at the base factory.
Nested wrappers overwrite the same per-invocation options, as the real
`cpu_session` does. Local progress bookkeeping is abstract. No entire external
constructor or multi-step shared update is treated as atomic.

Python attribute reads/writes and mutex acquisition/release are the shared
boundaries. Local option allocation, inter-op/optimization/provider constants,
and returning through wrappers are omitted because they add no shared factory
state to this property. Each session has fresh options; no other participant
mutates them. The model covers nondeterministic synchronous constructor failure
before a session and after its settings are observed, including a session-call
exception. Factory read/capture/install/restore are separate interleavable steps.

This is bounded **safety**: two callers, one load each, at most one session each,
all modeled schedules. It makes no liveness/fairness claim. It does not prove
arbitrary caller counts, multiple sessions per constructor, upstream dependency
behavior, provider execution, calibration/inference, crash safety, reentrancy,
async exceptions outside try/finally, or interactions with nonparticipating
direct ORT calls/factory writers. In particular, a process-wide monkeypatch
cannot isolate unrelated callers that ignore this helper's lock; this check
does not claim that isolation. The fake upstream boundary assumes a synchronous
lookup of `ort.InferenceSession`, not cached references or background workers.

## Keep it in step

`guarded.json` holds the entire guarded product file's content hash.
`check_source.py`, run first on every invocation, rejects any drift. Follow
[the project skill](../.claude/skills/verify-onnx-construction/SKILL.md) to review
and update the mapping when the product changes. It is also discoverable at
`.agents/skills/verify-onnx-construction` as an identical regular skill file.
There was no repository AGENTS.md or CLAUDE.md to add a pointer to.

The skill includes the authorized scratch mutation procedure. It removes the
real construction lock, mirrors that one known change into `UseLock=FALSE`,
checks the invariant without weakening it, records failures, restores the
exact original bytes, and reruns the baseline. This is an explicit mutation
mode, not a generic way around source freshness. The deterministic real-code
replay uses events rather than sleeps and fails on both wrong options and
stale global restoration. Baseline replay covers the four post-session
success/exception combinations; TLC also explores pre-session exceptions.

See [verification record](evidence/verification.md) for exact commands, counts,
versions and times, and [counterexample](evidence/counterexample.md) for the
seeded bug in product terms. A clean result covers only the stated scope.
