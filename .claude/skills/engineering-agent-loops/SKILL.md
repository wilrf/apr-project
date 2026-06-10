---
name: engineering-agent-loops
description: Use when designing or running anything that iterates without a human in the inner loop — a /loop, a dynamic Workflow fan-out, an overnight headless `claude -p` run, a maker/checker subagent pair, worktree parallelism, or any task whose "done"/goal you intend to express as a machine-checkable condition. Triggers on "set up a loop", "run this overnight", "create a workflow to…", "let it iterate until…", "goal/done-condition", "fan out agents", "parallelize with worktrees".
user-invocable: true
---

# Engineering Agent Loops & Goals

This skill is a thin router. The full operational reference lives at:

**`references/agent-loops-playbook.md`** — read it now before designing the loop.

It covers:

- The one rule: change the harness so the mistake can't recur. Gate first, agents last.
- When to use a loop vs. worktrees vs. maker/checker vs. a dynamic Workflow — vs. just doing the work.
- The four-station loop (Plan → Act → **Verify** → Fix) and why station 3 must be able to *fail*.
- **The adversarial verify panel**: lens-specialized verifiers (correctness / domain-invariant / simplicity), the "adjudicated — do NOT re-raise" preamble, adjudicating findings against the gate instead of obeying them, model pinning.
- The verification gate: composing your repo's real fail-able commands into one exit code.
- Why mocked-component tests can't catch integration bugs (live smoke is load-bearing) and why layered failures require re-verifying the *user journey* after every fix.
- Verifiable "done"/goal conditions (no vibes), including data-shaped parity gates for migrations.
- **Shared-environment traps**: one writer per live dev deployment, pathspec commits in a shared dirty tree, reviewing delegate diffs, pasting lint sharp edges into agent prompts.
- **Engineering "noticing"**: anomaly triggers that force a stop-and-log, every-mock-needs-a-real-counterpart, negative-space audits, near-miss retro analysis.
- **The retro station (self-improvement)**: ledger "traps" are the staging area; at goal completion, promote anything that bit twice / cost a phase / would bite a fresh session into the playbook as part of the final commit.
- **`templates/goal-template.md`** — the copy-paste goal skeleton with all stations contractual. Start every long-horizon goal from it; don't hand-roll station lists.
- Context budgeting for long loops: durable ledger files, deliberate `/compact` at phase boundaries, fanning out builders not just verifiers, small result payloads, session-per-phase ratchet.
- A copy-paste headless closed-loop skeleton.

Quick gut-check before you loop:

1. Is there a command that can actually **fail**? No gate = no loop.
2. Is "done" stated as an exit code or checked artifact, not a vibe?
3. If it's a dynamic Workflow — did the user **explicitly opt in**?
4. Is the iteration count **capped** so a stuck loop dies?
5. Does it respect your repo's HARD RULES (ports, push flags, prod env)?
6. Will the loop outlive a context window? Then loop state (phase status,
   rejected decisions, env facts) goes in a **ledger file on disk**, updated
   before every commit — and `/compact` runs at phase boundaries, not mid-edit.
7. Does the goal prompt include the **retro station** — promoting ledger traps
   into the playbook before declaring done? A loop that doesn't feed its
   lessons back is a loop you'll re-debug next month.

If any answer is no, fix that before starting. Then follow the playbook.
