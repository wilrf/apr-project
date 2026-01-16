# Superpowers Plugin

**Version:** 4.0.3
**Author:** Jesse Vincent (jesse@fsck.com)
**Repository:** https://github.com/obra/superpowers
**License:** MIT

## Overview

Core skills library for Claude Code providing structured workflows for TDD, debugging, collaboration patterns, and proven development techniques.

## Skills Reference

### 1. using-superpowers
**When:** Start of any conversation
**Purpose:** Establishes how to find and use skills

**Core Rule:** Invoke relevant skills BEFORE any response or action. Even 1% chance a skill applies = invoke it.

**Red Flags (stop and check for skills):**
- "This is just a simple question"
- "I need more context first"
- "Let me explore the codebase first"
- "This doesn't need a formal skill"

---

### 2. brainstorming
**When:** Before any creative work - creating features, building components, adding functionality
**Purpose:** Turn ideas into fully formed designs through collaborative dialogue

**Process:**
1. Check project context (files, docs, commits)
2. Ask questions ONE AT A TIME (prefer multiple choice)
3. Propose 2-3 approaches with trade-offs
4. Present design in 200-300 word sections, validating each
5. Write design to `docs/plans/YYYY-MM-DD-<topic>-design.md`

**Key Principles:**
- One question at a time
- Multiple choice preferred
- YAGNI ruthlessly
- Incremental validation

---

### 3. systematic-debugging
**When:** Any bug, test failure, or unexpected behavior
**Purpose:** Find root cause before attempting fixes

**The Iron Law:** NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST

**Four Phases:**

**Phase 1: Root Cause Investigation**
- Read error messages carefully (don't skip)
- Reproduce consistently
- Check recent changes (git diff)
- Gather evidence at each component boundary

**Phase 2: Pattern Analysis**
- Find working examples in codebase
- Compare against references
- Identify differences

**Phase 3: Hypothesis and Testing**
- Form single hypothesis: "I think X because Y"
- Test minimally (one variable at a time)
- If doesn't work, form NEW hypothesis

**Phase 4: Implementation**
- Create failing test case FIRST
- Implement single fix
- Verify fix
- If 3+ fixes failed: STOP, question architecture

**Red Flags:**
- "Quick fix for now, investigate later"
- "Just try changing X and see"
- Proposing solutions before tracing data flow

---

### 4. test-driven-development
**When:** Implementing any feature or bugfix
**Purpose:** Write test first, watch fail, write minimal code to pass

**The Iron Law:** NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST

**Red-Green-Refactor Cycle:**

1. **RED** - Write one minimal failing test
2. **Verify RED** - Watch it fail (MANDATORY)
3. **GREEN** - Write simplest code to pass
4. **Verify GREEN** - Watch it pass (MANDATORY)
5. **REFACTOR** - Clean up, keep tests green

**Critical Rules:**
- Wrote code before test? Delete it. Start over.
- Test passes immediately? Wrong test. Fix it.
- Never fix bugs without a test first.

**Rationalizations to Reject:**
- "Too simple to test" - Simple code breaks
- "I'll test after" - Tests passing immediately prove nothing
- "TDD will slow me down" - TDD faster than debugging

---

### 5. verification-before-completion
**When:** About to claim work is complete, fixed, or passing
**Purpose:** Evidence before claims, always

**The Iron Law:** NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE

**Gate Function:**
```
1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh)
3. READ: Full output, check exit code
4. VERIFY: Does output confirm claim?
5. ONLY THEN: Make the claim
```

**Red Flags:**
- Using "should", "probably", "seems to"
- Expressing satisfaction before verification
- About to commit/push without verification

---

### 6. writing-plans
**When:** Have spec/requirements for multi-step task
**Purpose:** Create comprehensive implementation plans

**Bite-Sized Tasks (2-5 minutes each):**
- "Write the failing test" - step
- "Run it to make sure it fails" - step
- "Implement minimal code" - step
- "Run tests, verify pass" - step
- "Commit" - step

**Plan Header Template:**
```markdown
# [Feature Name] Implementation Plan

> **For Claude:** Use superpowers:executing-plans to implement

**Goal:** [One sentence]
**Architecture:** [2-3 sentences]
**Tech Stack:** [Key technologies]
```

---

### 7. executing-plans
**When:** Have written implementation plan to execute
**Purpose:** Execute plans with review checkpoints

---

### 8. subagent-driven-development
**When:** Executing plans with independent tasks in current session
**Purpose:** Dispatch fresh subagent per task with code review between

---

### 9. dispatching-parallel-agents
**When:** 2+ independent tasks without shared state
**Purpose:** Parallelize work across multiple agents

---

### 10. using-git-worktrees
**When:** Starting feature work needing isolation
**Purpose:** Create isolated git worktrees safely

---

### 11. finishing-a-development-branch
**When:** Implementation complete, tests pass
**Purpose:** Guide merge, PR, or cleanup decisions

---

### 12. requesting-code-review
**When:** Completing tasks, before merging
**Purpose:** Verify work meets requirements

---

### 13. receiving-code-review
**When:** Getting code review feedback
**Purpose:** Technical rigor, not performative agreement

---

### 14. writing-skills
**When:** Creating or editing skills
**Purpose:** Verify skills work before deployment

## Usage Pattern

```
User request received
    ↓
Check: Does any skill apply? (even 1% chance)
    ↓ yes
Invoke skill BEFORE responding
    ↓
Follow skill instructions exactly
    ↓
Complete task
```
