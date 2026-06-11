# engineering-agent-loops

A [Claude Code](https://claude.com/claude-code) skill for designing and running
**autonomous agent loops** — long-horizon `/goal` runs, headless `claude -p`
loops, maker/checker pairs, and Workflow fan-outs — that verify their own work
and **feed their lessons back into themselves**.

## Why this exists

This skill drove a real autonomous run: one prompt built a 7-phase semantic
search + RAG system (~4,900 lines added, ~1,100 deleted, 70+ tests) across
schema, indexing, embeddings, retrieval, backfill, frontend, and monitoring —
with an adversarial verify panel that caught at least one real bug **every
phase**, including two security issues no compiler, lint, or test could see.
The playbook here is the distillation of that run plus the retro mechanism
that keeps it improving.

The core ideas:

- **No gate = no loop.** The harness (verification gate, isolation, guardrails)
  is the asset; agents are disposable.
- **The adversarial verify panel** — three lens-specialized read-only reviewers
  per phase (correctness / your domain invariant / simplicity), with an
  adjudication discipline so you obey the gate, not the reviewers.
- **Engineering "noticing"** — anomaly triggers, every-mock-needs-a-real-
  counterpart, negative-space audits, near-miss analysis. Unknown-unknowns
  don't volunteer; you build sensors for them.
- **The retro station** — at goal completion, lessons that met a promotion bar
  (bit twice / cost a phase / would bite a fresh session) get committed into
  the playbook. The skill is self-improving by contract, not by hope.
- **Context budgeting** — ledger files on disk, compaction at phase boundaries,
  session-per-phase ratchets. Loops that outlive their context window survive.

## Install

**Per-project** (recommended — the playbook accumulates repo-specific judgment):

```bash
git clone https://github.com/wilrf/engineering-agent-loops .claude/skills/engineering-agent-loops
rm -rf .claude/skills/engineering-agent-loops/.git   # make it yours; it will diverge
```

**Global** (all projects):

```bash
git clone https://github.com/wilrf/engineering-agent-loops ~/.claude/skills/engineering-agent-loops
```

Claude Code picks up the skill automatically; it triggers on phrases like
"set up a loop", "run this overnight", "let it iterate until…", or invoke it
directly with `/engineering-agent-loops`.

## Layout

```
SKILL.md                              # thin router — what Claude loads first
references/agent-loops-playbook.md    # the accumulated judgment (read on use)
templates/goal-template.md            # copy-paste /goal skeleton, all stations contractual
```

Three layers, three lifespans: the skill routes (per-trigger), the playbook
accumulates (per-repo), and your task ledgers stage (per-task). That layering
is what survives context compaction and session boundaries.

## Make it yours

This is a **fork-and-diverge** artifact, not a dependency. After installing,
delete the `.git` directory and let your own runs feed the playbook through the
retro station. The version here is a snapshot of one team's accumulated
judgment; yours should grow past it.

## License

MIT
