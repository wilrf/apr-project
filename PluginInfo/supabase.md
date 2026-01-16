# Supabase Plugin

**Author:** Supabase
**Source:** claude-plugins-official

## Overview

MCP (Model Context Protocol) integration for Supabase database operations, authentication, storage, and real-time subscriptions. Manage Supabase projects, run SQL queries, and interact with backend directly.

## Capabilities

### Project Management
- List organizations and projects
- Get project details and status
- Create, pause, and restore projects
- Manage development branches

### Database Operations
- List tables across schemas
- Execute raw SQL queries
- Apply migrations (DDL operations)
- Generate TypeScript types
- List extensions and migrations

### Edge Functions
- List, get, and deploy edge functions
- Manage function configurations

### Documentation
- Search Supabase docs via GraphQL
- Get error information by code and service

## Available MCP Tools

| Tool | Purpose |
|------|---------|
| `list_organizations` | List user's organizations |
| `get_organization` | Get org details with subscription |
| `list_projects` | List all Supabase projects |
| `get_project` | Get project details |
| `create_project` | Create new project |
| `pause_project` | Pause a project |
| `restore_project` | Restore paused project |
| `list_tables` | List tables in schemas |
| `list_extensions` | List database extensions |
| `list_migrations` | List applied migrations |
| `apply_migration` | Apply DDL migration |
| `execute_sql` | Run raw SQL (DML) |
| `get_logs` | Get service logs |
| `get_advisors` | Get security/performance advisors |
| `get_project_url` | Get API URL |
| `get_publishable_keys` | Get anon/publishable keys |
| `generate_typescript_types` | Generate TS types |
| `list_edge_functions` | List edge functions |
| `get_edge_function` | Get function code |
| `deploy_edge_function` | Deploy edge function |
| `create_branch` | Create dev branch |
| `list_branches` | List dev branches |
| `delete_branch` | Delete branch |
| `merge_branch` | Merge to production |
| `reset_branch` | Reset branch migrations |
| `rebase_branch` | Rebase on production |
| `search_docs` | Search documentation |

## Usage Notes

### For SQL Operations
- Use `apply_migration` for DDL (schema changes)
- Use `execute_sql` for DML (data operations)
- Don't hardcode generated IDs in data migrations

### For Edge Functions
Edge function example:
```typescript
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

Deno.serve(async (req: Request) => {
  const data = { message: "Hello!" };
  return new Response(JSON.stringify(data), {
    headers: { 'Content-Type': 'application/json' }
  });
});
```

### Cost Awareness
- Always use `get_cost` before `create_project` or `create_branch`
- Confirm costs with user via `confirm_cost`

### Security
- Run `get_advisors` regularly, especially after DDL changes
- Check for missing RLS policies
- Only use enabled publishable keys

## Integration Pattern

```
1. List projects → get project_id
2. Use project_id for all subsequent operations
3. For branches, get branch project_ref and use as project_id
```
