# Design documents

Long-form design specs and their implementation plans for
`pyiron_workflow_atomistics`. Each design pairs a `specs/<date>-<topic>-design.md`
with a `plans/<date>-<topic>.md` describing how the design was rolled out.

## Layout

```
docs/design/
├── specs/   # Architecture / API / problem statement (the "what" and "why")
└── plans/   # Step-by-step implementation plans (the "how", commit-by-commit)
```

## Index

| Date | Topic | Spec | Plan |
|------|-------|------|------|
| 2026-05-12 | Cleanup & reorganisation (Engine Protocol, topical physics modules, `_internal/`) | [spec](specs/2026-05-12-pyiron-workflow-atomistics-cleanup-design.md) | [plan](plans/2026-05-12-pyiron-workflow-atomistics-cleanup.md) |

New designs should follow the `YYYY-MM-DD-<topic>-design.md` naming convention
and live alongside their plan. The specs are versioned with the code so that
the rationale for any current architectural decision can be traced back to its
original deliberation.
