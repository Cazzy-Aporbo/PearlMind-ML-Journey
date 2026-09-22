# Reporting a security concern

PearlMind is a public educational project, not an independently audited production service. No response-time guarantee or zero-vulnerability claim is made.

Use GitHub’s **Security → Report a vulnerability** option for sensitive reports. Include the affected commit, a minimal reproduction and the likely impact. Do not include real patient data, credentials or unrelated private information.

The teaching APIs bind locally by default. Public deployment requires its own access controls, request limits, monitoring and review. Load model artifacts only from trusted sources; native JSON or a weights-only loader does not establish artifact provenance.

Dependency update suggestions are automated. Passing tests and a green build do not constitute a security audit. Extended research modules have separate validation needs described in the learning atlas.
