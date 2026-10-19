-------------------------- MODULE Construction --------------------------
EXTENDS Integers, TLC

\* scripts/onnx_runtime.py:load accepts two distinct positive thread counts.
\* Caller IDs stand for those counts; Base stands for the unmodified factory.
CONSTANT UseLock
Callers == {1, 2}
Base == 0
NotCalled == -1

\* scripts/onnx_runtime.py:load / cpu_session:
\* pc is a caller's next statement; owner is _CONSTRUCTION_LOCK;
\* factory is ort.InferenceSession; saved is each closure's `original`;
\* observed records the thread-setting owner reaching the original factory.
\* outcome is the external ONNXAgent success/exception result, not its internals.
\* scripts/onnx_runtime.py:load.cpu_session: callee is a resolved callable
\* reference; setting is the per-invocation options.intra_op_num_threads owner.
VARIABLES pc, owner, factory, saved, observed, outcome, callee, setting
vars == <<pc, owner, factory, saved, observed, outcome, callee, setting>>

\* scripts/onnx_runtime.py:_CONSTRUCTION_LOCK and load before entering `with`.
Init == /\ pc = [t \in Callers |-> "acquire"]
        /\ owner = Base
        /\ factory = Base
        /\ saved = [t \in Callers |-> NotCalled]
        /\ observed = [t \in Callers |-> NotCalled]
        /\ outcome = [t \in Callers |-> "pending"]
        /\ callee = [t \in Callers |-> Base]
        /\ setting = [t \in Callers |-> Base]

\* scripts/onnx_runtime.py:load, `with _CONSTRUCTION_LOCK` (__enter__).
\* UseLock=FALSE mirrors only the explicitly supported nullcontext mutation.
Acquire(t) == /\ pc[t] = "acquire"
              /\ (~UseLock \/ owner = Base)
              /\ owner' = IF UseLock THEN t ELSE owner
              /\ pc' = [pc EXCEPT ![t] = "capture"]
              /\ UNCHANGED <<factory, saved, observed, outcome, callee, setting>>

\* scripts/onnx_runtime.py:load, `original = ort.InferenceSession`.
Capture(t) == /\ pc[t] = "capture"
              /\ saved' = [saved EXCEPT ![t] = factory]
              /\ pc' = [pc EXCEPT ![t] = "patch"]
              /\ UNCHANGED <<owner, factory, observed, outcome, callee, setting>>

\* scripts/onnx_runtime.py:load, `ort.InferenceSession = cpu_session`.
\* Defining the local closure has no shared effect and is folded into pc.
Patch(t) == /\ pc[t] = "patch"
            /\ factory' = t
            /\ pc' = [pc EXCEPT ![t] = "construct"]
            /\ UNCHANGED <<owner, saved, observed, outcome, callee, setting>>

\* scripts/onnx_runtime.py:load / cpu_session interface: external synchronous
\* ONNXAgent resolves ort.InferenceSession. A callable reference stays valid
\* after another caller restores the module attribute. No constructor is atomic.
Invoke(t) == /\ pc[t] = "construct"
             /\ callee' = [callee EXCEPT ![t] = factory]
             /\ pc' = [pc EXCEPT ![t] = "session"]
             /\ UNCHANGED <<owner, factory, saved, observed, outcome, setting>>

\* scripts/onnx_runtime.py:load.cpu_session, options.intra_op_num_threads = threads.
\* Wrappers overwrite supplied options too; each invocation has its own options.
SetOptions(t) == /\ pc[t] = "session" /\ callee[t] # Base
                 /\ setting' = [setting EXCEPT ![t] = callee[t]]
                 /\ pc' = [pc EXCEPT ![t] = "forward"]
                 /\ UNCHANGED <<owner, factory, saved, observed, outcome, callee>>

\* scripts/onnx_runtime.py:load.cpu_session, resolve captured `original` for call.
Forward(t) == /\ pc[t] = "forward"
              /\ callee' = [callee EXCEPT ![t] = saved[callee[t]]]
              /\ pc' = [pc EXCEPT ![t] = "session"]
              /\ UNCHANGED <<owner, factory, saved, observed, outcome, setting>>

\* scripts/onnx_runtime.py:load.cpu_session, base original receives options.
\* Base setting means no recipe override, not an asserted upstream default value.
Observe(t) == /\ pc[t] = "session" /\ callee[t] = Base
              /\ observed' = [observed EXCEPT ![t] = setting[t]]
              /\ pc' = [pc EXCEPT ![t] = "outcome"]
              /\ UNCHANGED <<owner, factory, saved, outcome, callee, setting>>

\* scripts/onnx_runtime.py:load, ONNXAgent raises before making a session.
FailBeforeSession(t) == /\ pc[t] = "construct"
                        /\ outcome' = [outcome EXCEPT ![t] = "raise"]
                        /\ pc' = [pc EXCEPT ![t] = "restore"]
                        /\ UNCHANGED <<owner, factory, saved, observed, callee, setting>>

\* scripts/onnx_runtime.py:load, ONNXAgent returns or raises after session call;
\* both paths execute the same finally. No assumption of successful loading.
FinishConstruction(t) == /\ pc[t] = "outcome"
                         /\ \E result \in {"return", "raise"}:
                              /\ outcome' = [outcome EXCEPT ![t] = result]
                              /\ pc' = [pc EXCEPT ![t] = "restore"]
                         /\ UNCHANGED <<owner, factory, saved, observed, callee, setting>>

\* scripts/onnx_runtime.py:load finally, `ort.InferenceSession = original`.
Restore(t) == /\ pc[t] = "restore"
              /\ factory' = saved[t]
              /\ pc' = [pc EXCEPT ![t] = "release"]
              /\ UNCHANGED <<owner, saved, observed, outcome, callee, setting>>

\* scripts/onnx_runtime.py:load, with-block exit releases _CONSTRUCTION_LOCK.
Release(t) == /\ pc[t] = "release"
              /\ owner' = IF UseLock THEN Base ELSE owner
              /\ pc' = [pc EXCEPT ![t] = "done"]
              /\ UNCHANGED <<factory, saved, observed, outcome, callee, setting>>

\* scripts/onnx_runtime.py:load allows either caller's next statement to run.
Next == \E t \in Callers: Acquire(t) \/ Capture(t) \/ Patch(t) \/ Invoke(t)
                         \/ SetOptions(t) \/ Forward(t) \/ Observe(t)
                         \/ FailBeforeSession(t) \/ FinishConstruction(t)
                         \/ Restore(t) \/ Release(t)
Spec == Init /\ [][Next]_vars

\* scripts/onnx_runtime.py:load's scoped-override promise: each session uses
\* its caller's thread setting; after both calls the original factory remains.
ScopedConstruction ==
    /\ \A t \in Callers: observed[t] \in {NotCalled, t}
    /\ ((\A t \in Callers: pc[t] = "done") => factory = Base)
=============================================================================
