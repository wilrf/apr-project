# Engineering Agent Loops & Goals — the playbook

Load this when: you are about to design or run anything that iterates without a
human in the inner loop — a `/loop`, a dynamic Workflow fan-out, an overnight
headless `claude -p` run, a maker/checker pair, or any task whose acceptance
criteria you intend to express as a machine-checkable "done" condition.

This playbook is **self-improving by design**: the field notes in it were
earned on real autonomous runs, and the retro station (below) is the mechanism
that keeps adding to them. Fork it, then let your own runs feed it.

---

## The one rule everything else serves

**When the agent makes a mistake, change the harness so it can't make that
mistake again.** A loop is just that rule applied repeatedly: the agent acts,
something *real* tells it it's wrong, it fixes, repeat. The agents are cheap and
disposable. The harness around them — the verification gate, the isolation, the
guardrails — is the asset. Build the gate first; the loop is easy once something
can tell it it's wrong.

**Agent count is the last dial you turn, not the first.** For a solo operator,
the right configuration is 1–3 well-instrumented loops, not a swarm. Your review
bandwidth — not compute — is the bottleneck.

---

## When to use a loop vs. just doing the work

| Situation | Use |
| --------- | --- |
| One bounded change you'll review yourself | Plain edit, no loop |
| A change with a clear pass/fail gate, you want it green before you look | **Closed loop** (`/loop` or headless + verify) |
| 2–3 independent features at once | **Worktrees**, one agent each |
| A judgement tests can't make (API shape, naming, "did it actually solve it") | **Maker/checker** — reviewer subagent with its own context |
| Hundreds of files / whole-service audit / migration | **Dynamic Workflow** (`Workflow` tool) — but only with explicit user opt-in |
| Touching production data or an irreversible system | **Parity/guardrail harness FIRST**, then loop against parity |

If a task is small, a dynamic workflow is just an expensive single agent.
Reserve fan-out for genuinely large work.

---

## The four-station loop (station 3 is the one people skip)

1. **Plan** — read the spec + context, propose an approach.
2. **Act** — edit code, run commands.
3. **Verify** — run something *real that can fail*: build, types, tests, lint, a
   browser check. **Not self-review.** A maker/checker reviewer is verification;
   "I reviewed my own work" is not.
4. **Fix** — read the failure, fix, loop back to Act. Stop when verification is
   green **or** a max-iteration cap is hit.

> If a loop has no station 3 that can actually fail, you don't have a loop. You
> have vibe coding with extra steps. **No test = no loop.**

Maker/checker separation matters: the agent that *wrote* the code should not be
the only one that *judges* it. In Claude Code that's a separate subagent (or a
Workflow verifier stage) with its own context, ideally adversarial — prompted to
*break* the work, not bless it. Default the verifier toward "refuted/failing
unless proven otherwise."

### The adversarial verify panel (field-tested)

> **Field note (June 2026, a 7-phase search/RAG build):** a 3-lens adversarial
> panel per phase found ≥1 real bug **every single phase**, including a CRITICAL
> (soft-deleted records re-indexable by a racing job) and a prompt-injection
> exfiltration path (model-supplied userId on an agent tool) that no compiler,
> lint, or test gate could have caught.

Run three read-only verifier agents per phase — `correctness`, one
**domain-invariant** lens (security/visibility/money/data-parity — whatever your
domain must never violate), and `simplicity`. Mechanics that make it work:

- **Lens-specialize, don't replicate.** Three identical reviewers find the same
  bug thrice. One lens per failure *class*; the domain-invariant lens is the one
  that pays for the panel.
- **Paste an "adjudicated decisions — do NOT re-raise" list into every verify
  prompt.** Verifiers re-flag previously-rejected findings forever otherwise.
  The ledger's rejected-findings section is that list's source.
- **Adjudicate findings; don't obey them.** Verifiers don't know your lint
  config or history. Expect a meaningful fraction of minor findings to be wrong
  (e.g. proposed "deletions" that would re-trip a duplicate-string lint rule or
  *add* lines). Check every proposed change against the actual gate before
  applying; record each rejection + rationale in the ledger.
- **Pin the verifier model explicitly** on orchestrated agents. A panel that
  silently errors out is indistinguishable from a panel that found nothing —
  one environment's workflow agents all died on a model/API incompatibility
  until the model was pinned per-call.

**This repo's domain-invariant lens — data/eval validity.** For an ML research
repo the invariant that quietly destroys the result isn't auth or money, it's
*leakage*. Prime the domain verifier to hunt, in priority order:

- **CV / train-test contamination** — any fold, normalization, or statistic
  that sees future or held-out rows. Watch the LSTM specifically: normalization
  stats must come from training data only, and per-fold stats must be captured
  per fold (the H12 regression). Rolling windows crossing seasons is *by
  design*; a fold leaking test rows is not.
- **Label leakage via the spread threshold** — only `spread >= 3` games get an
  `upset` label; sub-3 games keep `upset = NaN` and are excluded via
  `upset.notna()`, yet are still used in LSTM team history. A change that labels
  sub-3 games, or drops them from history, breaks the invariant silently.
- **Threshold drift** — every binary prediction in `DisagreementAnalyzer` must
  use `self.threshold` (base rate ~0.30), never a hardcoded `0.5`. The
  `GamePrediction.*_pred` properties hardcode `0.5`; using them inside the
  analyzer is the canonical violation.
- **Feature-count violations** — LR=46, XGB=70 (46+24 lags), LSTM=14×8+10. The
  canonical lists are constants in `pipeline.py`; a hardcoded column list that
  drifts from them is a leak-shaped bug the type checker can't see.

A green test suite with leaked folds reports an AUC that doesn't survive contact
with new seasons. This is the lens that pays here, the way visibility/security
pays for a web app — give it its own verifier and prime it with these classes.

---

## The verification gate — build this first

A loop needs ONE green/red command as its done-condition. If your repo doesn't
have a `verify` composite, building it is the highest-value thing to do before
running any serious loop.

**This repo's gate (apr-research).** No `package.json` — chain the real
fail-able commands into one exit code. Two tiers: the fast tier gates every
loop iteration; the eval harness is the integration tier (the ML analog of
`build` + an end-to-end smoke — it runs the trained models over the real
held-out test set and regenerates the report).

```bash
# verify — single exit code for the whole loop. Run from the repo root in .venv.
python3 -m pytest tests/ \
      --ignore=tests/models/test_lstm_model.py \
      --ignore=tests/models/test_lstm_trainer.py \
  && python3 -m ruff check src/ tests/ \
  && black --check src/ tests/ \
  && python3 -m src.models.evaluate_test_set        # integration tier
```

The fast 3 (pytest-fast + ruff + `black --check`) is ~5s and gates every
iteration; the LSTM model/trainer suites (`test_lstm_model.py`,
`test_lstm_trainer.py`, ~30s each) and the eval harness are heavier — run them
at phase boundaries, not on every inner-loop fix. The loop's done-condition is:
**that chain exits 0.** Wherever this playbook says `verify` below, it means
this chain. Note `black --check` (non-mutating) for the gate, not bare `black`
(which rewrites files mid-loop and muddies the diff you're verifying).

**Visual changes are not done until browser-verified.** Lint / typecheck / unit
/ DOM-only checks do NOT satisfy a layout, spacing, typography, color, or
hierarchy change. The gate for visual work includes an authenticated live
browser pass (Playwright or in-app screenshots). If auth or browser tooling is
blocked, record the blocker and ask for the narrow unblock — do not close the
task.

**Mocked-component tests are structurally blind to integration bugs.**

> **Field note:** a live-browser smoke caught a UI-library bug (cmdk silently
> re-filtering server-side search results — every semantic match would have
> been hidden) that 92 green unit tests could never catch, because the jest
> suite *mocked the component away*. If a test replaces the real library, the
> live pass is the only station that exercises it. Treat the smoke as
> load-bearing, not ceremonial.

**Layered failures: re-verify the user journey, not the absence of the last
error.**

> **Field note:** one outage stacked four causes (provider quota → a pricing
> throw in a usage handler → an error-saver that used the failing machinery →
> a 1-step tool-turn SDK default). Each fix revealed the next; three of four
> were invisible until the one in front was removed.

The gate for "fixed" is *the real user path completing end-to-end*, re-run after
**every** fix. Corollary: error-reporting paths must be strictly simpler than
the paths they report on, or your last line of defense fails first.

**Data migrations get data-shaped gates.** Verification is parity, not
code-shaped: row counts, checksums, query-replay diff between old and new.
Build the parity harness *before* any agent touches the system. The loop closes
against "parity check passes," **never** "the migration script ran without
error."

---

## "Done" must be verifiable, not vibes

A goal lives in your prompt, in CLAUDE.md, or in a skill — and it only works if
it's machine-checkable.

| ❌ Vague | ✅ Verifiable |
| ------- | ------------ |
| "make the tasks page nicer" | "`pnpm verify` green AND Playwright smoke screenshot matches the mock" |
| "fix the search bug" | "the failing repro test passes AND no other test regresses" |
| "migrate the table" | "parity harness: row counts equal, checksums equal, 100-query replay diff is empty" |

Write the done-condition *before* the loop starts. If you can't state it as a
command that exits 0/non-zero (or a checked artifact), you're not ready to loop.

---

## What the buzzwords actually are in Claude Code

| Term | What it actually is | How to use it |
| ---- | ------------------- | ------------- |
| "Write loops not prompts" | **Headless mode**: `claude -p "…"` runs one non-interactive turn, can `--resume <session-id>`. Wrap in a shell loop or hand looping to a Workflow. | Script plan→act→verify→fix; re-invoke until the gate passes. |
| `/loop <interval>` | Recurring/self-paced runs (where available). Omit the interval to let the model self-pace. | Poll a deploy; iterate a fix to convergence. **One-offs don't need it.** |
| `/goal` | Not a command. Lives in the prompt/CLAUDE.md/skill as a *verifiable* condition. | See "Done" section above. |
| Worktrees | `Agent(isolation:"worktree")`, the `EnterWorktree` tool, or `claude --worktree`. Each agent gets an isolated checkout/branch. | 2–3 parallel features without clobbering. Expensive per-agent — use only when agents mutate files in parallel. |
| Dynamic workflows | The **`Workflow` tool** — Claude writes an orchestration script, fans out subagents, verifies, iterates. **Requires explicit user opt-in.** | Migrations, audits, stress-tests, dead-code sweeps. |
| Skills | `SKILL.md` + YAML frontmatter in `.claude/skills/` (project) or `~/.claude/skills/` (global). Auto- or `/`-invoked. | Encode a repeatable workflow once; stop re-pasting prompts. |
| CLAUDE.md | Always-on context, injected every turn. | Short, always-true rules ONLY. Situational knowledge goes in skills/docs (like this one). |
| Maker/checker | Subagents with their own context, or a Workflow verifier stage. | One agent builds, a *different* one adversarially reviews. |
| Hooks | Deterministic code on lifecycle events; exit code 2 blocks the action and feeds the reason back. | Hard guardrails (e.g. gating git push/commit). |

---

## Build order (staged, for a solo operator)

**Stage 0 — Foundations.** Keep CLAUDE.md to always-true rules; push situational
knowledge into skills/docs. (This doc is an instance of that.)

**Stage 1 — Close one loop.** Pick one bounded feature. Give the agent the spec
plus the verify gate as the done-condition; let it iterate to green *before* you
review. You're proving the gate works, not parallelizing. The day a loop fixes
its own failing test without you, you've crossed the line.

**Stage 2 — Parallelize with worktrees (1–3 agents).** One feature per worktree.
Three concurrent is plenty for one human to review. Resist more.

**Stage 3 — Maker/checker.** Add an adversarial reviewer subagent with its own
context — for the calls tests can't make. Then graduate to the 3-lens panel.

**Stage 4 — Dynamic Workflows for big swings.** Migrations, modernization
passes, security audits, dead-code cleanup. Scope tight, watch the first run —
these consume meaningfully more tokens.

**Stage 5 — Overnight autonomy.** Headless `claude -p` launched before bed,
results at dawn. Cap concurrency, pipe JSON output somewhere scannable. Only
safe once the gate and guardrails are real.

---

## Guardrails & cost control

- **Workflow opt-in is mandatory.** Do NOT launch a dynamic `Workflow` unless
  the user explicitly opted in. A task merely *benefiting* from fan-out is not
  opt-in — describe the workflow and its rough cost, and ask first.
- **Respect your repo's HARD RULES** — the deterministic guardrails a loop must
  never route around. Typical examples: fixed dev ports (port taken → STOP and
  report, never edit config/env to dodge it); git push restrictions (`--force*`,
  `--no-verify` blocked); prod env vars and env files untouchable.
- **Never** add lint-disable comments or `any` to make a gate pass — fix the
  underlying issue. A loop that suppresses the gate has defeated its own purpose.
- **Billing:** subscription plans are fine for 1–3 steady agents. Past ~5
  concurrent, or heavy Workflow use, route to API-key billing so a runaway loop
  is a line item, not an outage.
- **No silent caps.** If a loop bounds coverage (top-N, no-retry, sampling), say
  so. Silent truncation reads as "covered everything" when it didn't.
- **Make failures visible.** Never swallow a server error with a "zero" fallback
  to make a loop appear green. Surface it and fix the root cause.

---

## Shared-environment traps (multi-agent, one repo, one dev deployment)

> **Field note:** earned the hard way when two Claude sessions worked the same
> repo concurrently.

1. **One writer per live deployment.** Hot-reload dev servers (e.g. `convex dev`)
   auto-push every file save — a second session's in-progress edits (including
   test stubs) go live instantly under your verification. Symptom: behavior that
   matches no code in your tree (we once got a literal "PONG" from a production
   surface). Before verifying against a live deployment, `git status` the files
   on that path; if another actor's edits are in flight, **pause and coordinate
   via the user** rather than debugging their WIP.
2. **Pathspec commits keep a shared dirty tree safe.** `git add <your files>`
   then `git commit -- <same paths>` commits exactly your slice and leaves the
   user's staged/dirty work untouched. Never stash, reset, or bundle someone
   else's files; if your change and theirs land in the same file and theirs is
   load-bearing for yours (e.g. an import fix the file needs to compile), say so
   in the commit message.
3. **Delegate mechanical fan-out, but review the delegate's diff.** A subagent
   wiring 31 call sites is a context win, and self-gating (it runs lint/tsc)
   catches most issues — but spot-check the largest per-file diff. One
   delegate's single creative moment was a nonsense conditional-type cast that
   gates passed but a human reviewer wouldn't.
4. **Paste the gate's sharp edges into builder/verifier prompts.** Repo lint
   rules (duplicate-string thresholds, max-lines-per-function, no-disable)
   repeatedly shape what a "correct" change even looks like. Agents that don't
   know the rules propose changes the gate rejects; tell them up front.

---

## Engineering "noticing" — unknown-unknowns don't volunteer

The retro can only promote lessons the loop *noticed*. Noticing is engineerable:

1. **Anomaly triggers force a stop-and-log.** The moment any of these fire,
   write an "expected X, observed Y" entry in the ledger *before* continuing —
   even if you resolve it seconds later:
   - Observed behavior matches **no code in your tree**.
   - Tests green but the **feature is visibly dead/blank** in the real surface.
   - A failure **disappears without your fix explaining why**.
   - Logs/errors reference a file, model, or path you didn't touch.
   - Silent success: an operation that should have side effects completes
     without them.
2. **Every mock needs a real-dependency counterpart somewhere in the gate.**
   A test suite that mocks a library away is *structurally incapable* of
   catching that library's integration bugs, no matter how many tests pass.
   For each mocked boundary (UI library, provider API, auth), name the one
   station that exercises the real thing (browser smoke, live-deployment probe,
   paid-key integration test) — and if none exists, that's a ledger-recorded
   gap, not a shrug.
3. **Negative-space audit at phase end.** Sketch the verification matrix
   (user paths × real-vs-mocked × environments) and *list the unexercised
   cells* in the ledger. You can't always afford to fill them; you can always
   afford to know where you're blind. A completeness-critic agent ("what
   modality wasn't run, what claim wasn't verified?") is the fan-out version.
4. **Near-miss analysis in the retro.** Don't just promote what bit you — ask
   "what was caught by exactly *one* station, and was that station in the
   contract by design or by luck?" Single-sensor catches mean the next bug of
   that class escapes. Strengthen or duplicate that sensor.

---

## Make the loop self-improving (the retro station)

The four stations close a *task*; a fifth closes the *system*. At goal
completion (or when a session ends mid-goal):

1. **The ledger's "traps" section is the staging area.** Every standing trap,
   environment fact, or rejected-finding rationale discovered mid-loop gets
   written there *at the moment of discovery* — not reconstructed later.
2. **Promote on recurrence.** When a goal completes, diff the ledger's traps
   against this playbook: anything that (a) bit twice, (b) cost a phase, or
   (c) would bite a fresh session with zero context, gets promoted into this
   doc as part of the final commit. Session-specific noise stays in the ledger
   and dies with the task.
3. **The skill stays a thin router.** Lessons land here, not in SKILL.md — the
   skill's job is routing; this doc's job is accumulating judgment. If SKILL.md
   grows past ~50 lines, it's absorbing what belongs here.
4. **Prompt it explicitly.** A goal prompt should end with: "before declaring
   done, run the retro station: promote ledger traps into the playbook if they
   meet the promotion bar." Self-improvement that isn't in the loop's contract
   doesn't happen.

The promotion bar matters: without it the doc accumulates session diary instead
of judgment and taxes every future read. The three-layer architecture — SKILL.md
routes (per-trigger), this playbook accumulates (per-repo/per-user), the ledger
stages (per-task) — is what survives compaction and session boundaries.

---

## Long loops: context is a consumable — budget it

A multi-phase loop will outlive its context window. The main loop doesn't die at
100% — it **auto-compacts**: the conversation is summarized and work continues in
a fresh window. Compaction is **lossy memory, not free memory**: exact code,
error text, and decision rationale survive only as well as the summary writer
did. Engineer for it instead of being surprised by it:

1. **Ledger-first.** Keep a durable ledger at `.claude/plans/<task>-progress.md`:
   phase status, decisions made, findings **rejected with rationale** (so a
   post-compaction agent doesn't re-accept them), environment facts, exact reuse
   signatures. Update it *before* each commit. The ledger is the source of truth;
   the conversation is scratch space. If the loop's state isn't on disk, it
   doesn't exist.
2. **Compact deliberately at phase boundaries.** Right after a commit + ledger
   update is the safest compaction point — zero in-flight state. Mid-edit
   auto-compaction is the riskiest. Run `/compact` after each phase commit rather
   than letting the window hit the wall mid-task.
3. **Fan out builders, not just verifiers.** Inline building is the biggest
   context burner — every Read/Edit of a large file lands in the main window.
   Subagent/Workflow transcripts cost the main loop nothing but their final
   message. For small phases inline is fine; for large ones, delegate the writes
   too.
4. **Small result payloads.** Have workflows/subagents write detailed findings to
   a file and return counts + the path — not multi-KB JSON into the main loop.
5. **Session-per-phase ratchet (multi-day scale).** Each phase as a fresh session
   or headless `claude -p` run whose contract is: read ledger → do the next
   unchecked phase → run gates → commit → update ledger. Compaction becomes
   irrelevant because no session needs more than one phase of context.

---

## Anti-patterns (stop if you catch yourself here)

- **Looping without a real failure signal.** No gate = not a loop.
- **Self-review as verification.** The maker can't be the only judge.
- **Chasing agent count.** Three good loops beat thirty blind ones.
- **Situational knowledge in CLAUDE.md.** It taxes every turn. Skills/docs exist
  for that.
- **Unattended loops on production data without a parity harness.**
- **Letting auto-compaction be your memory strategy.** If phase status, rejected
  decisions, and environment facts live only in chat, every compaction rolls the
  dice on them. Ledger file, updated before every commit.
- **Handing over architecture and taste.** The model iterates; you still decide
  what's worth building and whether the result is good. That half doesn't compress.

---

## Minimal closed-loop skeleton (headless)

```bash
# done-condition: `pnpm verify` exits 0. Cap iterations so a stuck loop dies.
SESSION=""
for i in $(seq 1 6); do
  if pnpm verify; then echo "GREEN on iter $i"; break; fi
  PROMPT="pnpm verify is failing. Read the output, fix the root cause \
          (no lint-disable, no any, no port/env edits), re-run pnpm verify."
  if [ -z "$SESSION" ]; then
    SESSION=$(claude -p "$PROMPT" --output-format json | jq -r .session_id)
  else
    claude -p "$PROMPT" --resume "$SESSION"
  fi
done
```

The shape is the point: **real gate, capped iterations, fix the root cause not
the symptom, persistent session so context carries between fixes.**

---

*Throughline: you're not learning to prompt better — you're building the harness
that lets a small number of agents run hard without babysitting. Build the gate
first; the loops are easy once something can tell them they're wrong.*
