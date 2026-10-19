# Seeded bug: overlapping ONNX construction

This is an intentionally broken scratch variant, **not a finding against the
current product**. Only the real `scripts/onnx_runtime.py:load` lock was replaced
with a no-op context manager; see `seeded-change.patch`. The invariant was not
changed. The default runner rejected the changed source (`source-drift.log`),
then the explicit scratch mode matched that exact edit and mirrored it into
`UseLock=FALSE`.

## TLC trace in the product's words

The 11 states in `seeded-failing.log` describe this schedule:

1. Call A enters `load`, saves the original `ort.InferenceSession`, and installs
   its local `cpu_session` wrapper.
2. A's `ONNXAgent` raises before making a session; A is about to run `finally`.
3. Without the lock, call B enters, saves A's wrapper as its own `original`,
   and installs B's wrapper.
4. A's `finally` restores the original unmodified factory.
5. B's `ONNXAgent` looks up `ort.InferenceSession` and calls the unmodified
   factory directly. No wrapper supplies B's requested thread setting.

The invariant fails with `observed[2]=0`, meaning no recipe override reached
the base factory for B. Zero is an abstraction sentinel, not an ORT default.
TLC stopped on this violation: 407 generated / 257 distinct states, 95 queued,
depth 11, wall runtime 0.450 seconds, TLC exit 12, runner exit 1. This failing
search is partial, not an exhaustive count of the broken model's state space.
Worker scheduling can change the exact failing counts or symmetric trace.

## Witness executing the actual product function

`formal/replay.py` uses two actual threads with events and synthetic modules.
It calls the real `load` twice, with requested settings A=2 and B=4. It does
not merely run a rewritten Python implementation of the model.

1. A installs its wrapper and pauses in the synthetic ONNXAgent constructor.
2. B saves A's wrapper, installs B's wrapper, and pauses in its constructor.
3. A resumes: B's wrapper forwards through A's wrapper, which overwrites the
   options back to A=2. A finishes and restores the unmodified factory.
4. B resumes and creates an unconfigured session: the test sentinel is B=0,
   although B requested 4.
5. B's `finally` restores its captured `original`, which is A's stale wrapper.
   Both calls have finished, but the original factory has not been restored.

Observed: `A=2`, `B=0`, `original_restored=False`. This realizes the same
ordering flaw without needing an exception and also witnesses stale restoration.

## Restore result and exposure

`seed_bug.py` restored the exact captured product bytes. `reverted.log` passes:
129 generated / 120 distinct states, no queue left, depth 21, wall runtime
0.404 seconds, runner exit 0. All four actual-code success/exception pairings
receive A=2/B=4 and restore the original factory. The product file has no Git
diff. The generated skill was read and followed in fresh processes, including
guard rejection, explicit mutation mirroring, failure inspection and revert.

There are no current-code findings in this bounded scope, so no current severity
or live impact is assigned. The scratch bug never ran against real dependencies,
models, services, or a deployed application. Third-party calls that ignore this
helper's lock remain outside the property, as stated in the README and skill.
