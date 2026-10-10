# Contributing

Keep this a small recipe around existing runtimes. Document the exact upstream
revision, checkpoint, OS, CPU, versions, thread count and measurement method.
Use synthetic inputs; never include sensitive data, home paths, hostnames or model
weights. Label agreement separately from accuracy and keep limitations visible.

Run the README's standard-library checks, then its CPU quickstart in a fresh
folder for a setup change. Run one benchmark process at a time. Changes to the
model runtime belong upstream. Before proposing a new backend, show a concrete
setup gap and compare the same complete checkpoint. Do not add services,
telemetry, automatic downloads at import time or an inference implementation.

Submit small reviewed changes under the MIT license. Upstream dependencies
retain their own licenses and attribution. See SECURITY.md for vulnerability
reporting and the issue templates for reproducible setup reports.
