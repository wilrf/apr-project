# Code Simplifier Plugin

**Version:** 1.0.0
**Author:** Anthropic
**Source:** claude-plugins-official

## Overview

An agent that simplifies and refines code for clarity, consistency, and maintainability while preserving all functionality. Focuses on recently modified code unless instructed otherwise.

## When to Use

- After writing or modifying code
- When code needs refinement
- To apply project standards consistently
- Before code review

## Core Principles

### 1. Preserve Functionality
Never change what code does - only how it does it. All original features, outputs, and behaviors must remain intact.

### 2. Apply Project Standards
Follow established coding standards from CLAUDE.md:
- Use ES modules with proper import sorting
- Prefer `function` keyword over arrow functions
- Use explicit return type annotations
- Follow proper React component patterns
- Use proper error handling patterns
- Maintain consistent naming conventions

### 3. Enhance Clarity
- Reduce unnecessary complexity and nesting
- Eliminate redundant code and abstractions
- Improve readability through clear names
- Consolidate related logic
- Remove unnecessary comments describing obvious code
- **Avoid nested ternary operators** - prefer switch/if-else
- Choose clarity over brevity

### 4. Maintain Balance
Avoid over-simplification that could:
- Reduce code clarity or maintainability
- Create overly clever solutions
- Combine too many concerns
- Remove helpful abstractions
- Prioritize "fewer lines" over readability
- Make code harder to debug or extend

### 5. Focus Scope
Only refine recently modified code unless explicitly instructed to review broader scope.

## Refinement Process

1. **Identify** recently modified code sections
2. **Analyze** for opportunities to improve elegance
3. **Apply** project-specific best practices
4. **Ensure** all functionality unchanged
5. **Verify** refined code is simpler and more maintainable
6. **Document** only significant changes

## What NOT to Do

| Avoid | Reason |
|-------|--------|
| Nested ternaries | Hard to read and debug |
| Dense one-liners | Prioritizes brevity over clarity |
| Over-abstraction | Premature optimization |
| Changing behavior | Must preserve functionality |
| Broad refactoring | Focus on recent changes only |

## Good vs Bad Examples

**Bad - Nested Ternary:**
```typescript
const status = isActive ? (isAdmin ? 'admin' : 'user') : 'inactive';
```

**Good - Explicit:**
```typescript
function getStatus(isActive: boolean, isAdmin: boolean): string {
  if (!isActive) return 'inactive';
  return isAdmin ? 'admin' : 'user';
}
```

## Integration

The agent operates autonomously, refining code immediately after it's written or modified without requiring explicit requests.
