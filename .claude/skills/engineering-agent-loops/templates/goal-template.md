# Goal template — non-negotiable stations for any long-horizon /goal

Copy this skeleton for every autonomous build goal. The bracketed parts are
per-goal; the STATIONS are not optional — removing one is how unknown-unknowns
escape (see `../references/agent-loops-playbook.md`, "Engineering noticing").

Field-tested on a 7-phase, ~4k-LOC autonomous build: every station below caught
at least one bug no other station could have.

---

```
/goal [WHAT to build, per WHICH spec] at the fewest lines that stay readable.
DONE = [machine-checkable condition — acceptance-criteria lines true, exit
code 0, checked artifact]. Readable-over-clever; no `any`; no lint-disable.

PREFLIGHT (before any edit): [start dev stack / auth check / env facts]. If
blocked, STOP and ask for the narrow unblock — don't discover it at the end.

PHASES: [spec order]. Per phase, drive this cycle:
1. BUILD to green (fan out builders for large phases — context budget).
2. VERIFY PANEL: 3 read-only adversarial agents — correctness,
   [DOMAIN INVARIANT — the lens that pays: visibility/security/money/parity],
   simplicity. Pin the model on orchestrated agents. Paste the ledger's
   "adjudicated — do NOT re-raise" list into every panel prompt. Adjudicate
   findings against the actual gates before applying; record rejections.
3. GATES: [test suites, typecheck, lint --max-warnings 0 on touched files].
4. LIVE-PATH SMOKE: exercise the REAL user journey on the REAL dependency
   (browser, live deployment) — at minimum once per user-facing phase and
   after EVERY fix in a failure chain ("the last error is gone" ≠ fixed).
   Every mocked boundary must have its real-counterpart station named; a
   boundary with none is a ledger-recorded gap.
5. COMMIT (pathspec — never bundle others' dirty files) + LEDGER UPDATE
   (.claude/plans/<task>-progress.md: phase status, decisions, rejected
   findings with rationale, traps, anomalies, unexercised verification cells).

ANOMALY TRIGGERS (stop-and-log the moment one fires, even if resolved):
behavior matching no code in the tree · tests green but feature visibly dead ·
a failure that disappears without your fix explaining why · logs referencing
things you didn't touch · silent success.

CONTEXT: ledger-first; /compact only at phase boundaries; small agent payloads.

RETRO (before declaring done): near-miss pass — what was caught by exactly one
station, by design or luck? List unexercised verification-matrix cells. Promote
ledger traps meeting the bar (bit twice / cost a phase / bites a fresh session)
into the playbook as part of the final commit.

STOP only on: all DONE criteria met, a gate you can't fix, or a hard blocker
(auth, ports, env) — never edit ports/env to work around one. Honor all repo
HARD RULES (ports, push flags, prod env, env files).
```
