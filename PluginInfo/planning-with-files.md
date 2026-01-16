# Planning with Files Plugin

**Version:** 2.1.2
**Author:** OthmanAdi
**Repository:** https://github.com/OthmanAdi/planning-with-files
**License:** MIT

## Overview

Implements Manus-style file-based planning for complex tasks. Uses persistent markdown files as "working memory on disk" to maintain context across long tasks.

## When to Use

- Multi-step tasks (3+ steps)
- Research tasks
- Building/creating projects
- Tasks spanning many tool calls
- Anything requiring organization

**Skip for:** Simple questions, single-file edits, quick lookups

## Core Pattern

```
Context Window = RAM (volatile, limited)
Filesystem = Disk (persistent, unlimited)

→ Anything important gets written to disk.
```

## The Three Planning Files

Create these in your **project directory** (not the plugin folder):

### 1. task_plan.md
**Purpose:** Phases, progress, decisions

```markdown
# Task Plan: [Task Name]

## Goal
[Clear statement of what success looks like]

## Current Phase
Phase N

## Phases

### Phase 1: [Name]
- [ ] Step 1
- [ ] Step 2
- **Status:** pending | in_progress | complete

### Phase 2: [Name]
...

## Decisions Made
| Decision | Rationale |
|----------|-----------|

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
```

### 2. findings.md
**Purpose:** Research, discoveries

```markdown
# Findings: [Task Name]

## Key Discoveries
- Discovery 1
- Discovery 2

## Technical Decisions
| Decision | Rationale | Assessment |
|----------|-----------|------------|

## Resources Referenced
- Resource 1: URL
```

### 3. progress.md
**Purpose:** Session log, test results

```markdown
# Progress Log: [Task Name]

## Session: YYYY-MM-DD

### Phase 1
- **Status:** in_progress
- Actions taken:
  - Action 1
  - Action 2
- Files created/modified:
  - file1.py

## Test Results
| Test | Input | Expected | Actual | Status |

## Error Log
| Timestamp | Error | Attempt | Resolution |
```

## Critical Rules

### 1. Create Plan First
Never start complex task without `task_plan.md`. Non-negotiable.

### 2. The 2-Action Rule
After every 2 view/browser/search operations, IMMEDIATELY save key findings to files.

### 3. Read Before Decide
Before major decisions, read the plan file to refresh goals in attention window.

### 4. Update After Act
After completing any phase:
- Mark status: `in_progress` → `complete`
- Log errors encountered
- Note files created/modified

### 5. Log ALL Errors
Every error goes in the plan file:
```markdown
| Error | Attempt | Resolution |
|-------|---------|------------|
| FileNotFoundError | 1 | Created default config |
```

### 6. Never Repeat Failures
```
if action_failed:
    next_action != same_action
```

## The 3-Strike Error Protocol

```
ATTEMPT 1: Diagnose & Fix
  → Read error carefully
  → Identify root cause
  → Apply targeted fix

ATTEMPT 2: Alternative Approach
  → Same error? Try different method
  → NEVER repeat exact same failing action

ATTEMPT 3: Broader Rethink
  → Question assumptions
  → Search for solutions

AFTER 3 FAILURES: Escalate to User
  → Explain what you tried
  → Share specific error
  → Ask for guidance
```

## Read vs Write Decision Matrix

| Situation | Action | Reason |
|-----------|--------|--------|
| Just wrote a file | DON'T read | Content still in context |
| Viewed image/PDF | Write findings NOW | Multimodal → text before lost |
| Browser returned data | Write to file | Screenshots don't persist |
| Starting new phase | Read plan/findings | Re-orient if context stale |
| Error occurred | Read relevant file | Need current state to fix |
| Resuming after gap | Read all planning files | Recover state |

## The 5-Question Reboot Test

If you can answer these, context management is solid:

| Question | Answer Source |
|----------|---------------|
| Where am I? | Current phase in task_plan.md |
| Where am I going? | Remaining phases |
| What's the goal? | Goal statement in plan |
| What have I learned? | findings.md |
| What have I done? | progress.md |

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Use TodoWrite for persistence | Create task_plan.md file |
| State goals once and forget | Re-read plan before decisions |
| Hide errors and retry silently | Log errors to plan file |
| Stuff everything in context | Store large content in files |
| Start executing immediately | Create plan file FIRST |
| Repeat failed actions | Track attempts, mutate approach |
