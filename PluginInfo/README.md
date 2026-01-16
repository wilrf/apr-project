# Claude Code Plugins Documentation

This folder documents the Claude Code plugins installed in this environment, structured for use by other LLMs and AI agents.

## Installed Plugins Summary

| Plugin | Source | Purpose |
|--------|--------|---------|
| [superpowers](./superpowers.md) | claude-plugins-official | Core skills library: TDD, debugging, planning, collaboration patterns |
| [supabase](./supabase.md) | claude-plugins-official | Database operations, auth, storage via MCP integration |
| [code-simplifier](./code-simplifier.md) | claude-plugins-official | Agent that refines code for clarity and maintainability |
| [planning-with-files](./planning-with-files.md) | planning-with-files | Manus-style file-based planning with task_plan.md, findings.md, progress.md |
| [pyright](./pyright.md) | claude-code-lsps | Python language server for type checking |

## How to Use These Plugins

### For Claude Code
Plugins are invoked via the `Skill` tool:
```
Skill(skill="superpowers:brainstorming")
Skill(skill="superpowers:systematic-debugging")
```

### For Other LLMs/Agents
Each plugin documentation file contains:
1. **Purpose** - What the plugin does
2. **When to Use** - Triggers for invoking
3. **Core Instructions** - The actual prompts/behaviors to follow
4. **Key Patterns** - Important workflows

You can incorporate these instructions directly into your system prompts or use them as reference for implementing similar behaviors.

## Plugin Categories

### Process Skills (use first)
- `superpowers:brainstorming` - Before any creative/building work
- `superpowers:systematic-debugging` - Before fixing any bug
- `superpowers:test-driven-development` - Before writing implementation code

### Implementation Skills
- `superpowers:writing-plans` - Create detailed implementation plans
- `superpowers:executing-plans` - Execute plans with checkpoints
- `superpowers:subagent-driven-development` - Dispatch subagents for parallel work

### Quality & Verification
- `superpowers:verification-before-completion` - Evidence before claims
- `superpowers:requesting-code-review` - Before merging
- `code-simplifier` - Refine code clarity

### Planning & Tracking
- `planning-with-files` - Manus-style persistent markdown planning

### Infrastructure
- `supabase` - Database/backend operations
- `pyright` - Python type checking

## Key Principles Across All Plugins

1. **Skills before action** - Check if a skill applies before doing anything
2. **Evidence before claims** - Never claim completion without verification
3. **Test first** - TDD for all implementation
4. **Root cause first** - Debug systematically, no random fixes
5. **One question at a time** - Don't overwhelm users with multiple questions
