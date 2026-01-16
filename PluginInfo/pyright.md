# Pyright Plugin

**Version:** 1.0.0
**Author:** Jan Kott
**Repository:** https://github.com/boostvolt/claude-code-lsps
**License:** MIT

## Overview

Python language server plugin that provides type checking capabilities via Pyright.

## Purpose

Integrates Pyright (Microsoft's static type checker for Python) into Claude Code for:
- Type checking Python code
- Detecting type errors
- Providing type hints and autocomplete
- Ensuring type safety in Python projects

## Requirements

Pyright must be installed:
```bash
pip install pyright
```

## Features

- Static type analysis for Python
- Detection of type mismatches
- Support for Python type hints (PEP 484+)
- Integration with Python typing module
- Analysis of function signatures
- Detection of missing type annotations

## When This Activates

The plugin provides language server diagnostics for Python files, helping catch type-related issues during development.

## Diagnostic Categories

| Category | Description |
|----------|-------------|
| Type errors | Incompatible types in assignments/calls |
| Missing types | Functions without type annotations |
| Import errors | Unresolved imports |
| Attribute errors | Invalid attribute access |

## Configuration

Pyright can be configured via `pyrightconfig.json` or `pyproject.toml`:

```json
{
  "include": ["src"],
  "exclude": ["**/node_modules", "**/__pycache__"],
  "typeCheckingMode": "basic",
  "pythonVersion": "3.11"
}
```

## Integration with Claude Code

The plugin hooks into the IDE diagnostics system (`mcp__ide__getDiagnostics`) to surface Python type errors alongside other code issues.
