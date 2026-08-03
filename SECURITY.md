# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities.

Email security reports to **elias@realityshifting.tech**. Include:

- A description of the vulnerability and its impact
- Steps to reproduce or a proof of concept
- Affected versions or commits

We aim to acknowledge reports within 72 hours and to keep you informed as we
investigate and fix.

## Scope

DeepSight proxies user images and prompts between reasoning and vision
backends and holds API keys for both. Issues involving API key handling,
prompt injection through image content, request body handling, cache
poisoning, and SSRF-style risks in the image fetching paths are treated as
high priority.

## Supported versions

DeepSight is pre-release. Only the latest commit on the default branch
receives security fixes until the first stable version is published.
