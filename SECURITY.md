# Security

Security fixes target the latest revision of this recipe and its pinned
Python 3.12 / Laya 0.3.4 quickstart. Older revisions and alternative runtimes
are unsupported.

Treat model inputs as untrusted and predictions as fallible. This recipe starts
no network listener. Do not expose a third-party model server without reviewing
its authentication, resource limits and upstream security guidance.

Install pinned packages from their original indexes and verify the downloaded
checkpoint. Hash verification pins bytes; it does not prove a model is safe or
accurate. Keep sensitive examples and credentials out of reports and issues.

Report a suspected vulnerability through GitHub's private vulnerability
reporting: open the repository's Security tab and select "Report a vulnerability"
if available. If that option is unavailable, open an issue asking for a secure
reporting channel without posting exploit details, affected inputs or secrets.
Maintainers respond and investigate on a best-effort basis; response times and
fixes are not guaranteed. Report upstream runtime or dependency vulnerabilities
through the affected project's security policy as well.
