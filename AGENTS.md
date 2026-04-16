# Agent guide

This file orients automated coding agents (and humans) working in this repository.

## Repository

- **Project**: `yolo26-multitask`
- **Status**: Minimal footprint today. When you add source, tests, or build tooling, update this section so future runs stay accurate.

## Workflow

1. **Branching**: Branch off `main` for substantive work. If your environment requires a prefix/suffix pattern (for example `cursor/<short-description>-438d`), follow it consistently.
2. **Commits**: One commit per coherent change; messages should state **what** changed and **why**, not vague labels like “fix” or “update” alone.
3. **Push**: Use `git push -u origin <branch-name>`. Retry on transient network failures if your playbook allows it.
4. **Pull requests**: When collaborating via a forge, open or update a PR after pushing; describe scope and how you validated the change.

## Implementation rules

- **Scope**: Touch only what the task requires; avoid drive-by refactors and unrelated formatting churn.
- **Consistency**: Match surrounding code for naming, types, imports, and documentation density.
- **Docs**: Do not add or heavily rewrite Markdown unless the task explicitly asks for it.

## Verification

- If the repo gains standard entry points (`package.json`, `pyproject.toml`, `Makefile`, `Cargo.toml`, etc.), run the appropriate **lint / test / build** after edits and note the outcome in the PR or commit message.
- If no automation exists yet, document **manual checks** you performed (how to run, what you verified).

## Communication

- Prefer **reproducible commands** and **concrete file paths** when describing changes.
- When citing existing code, use the path format your tooling expects so maintainers can jump to definitions quickly.

---

*If this file conflicts with organization-wide agent instructions, follow the organization policy and update or remove outdated guidance here.*
