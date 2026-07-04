# Agent documentation

Entry point: [AGENTS.md](../AGENTS.md)

| Doc | Purpose |
|-----|---------|
| [tdd-protocol.md](tdd-protocol.md) | RED → GREEN → REFACTOR → VERIFY loop |
| [verification-checklist.md](verification-checklist.md) | Handoff checklist |

## Verification command

```bash
./scripts/agent_verify.sh          # standard gate
./scripts/agent_verify.sh --fast   # skip openff / slow / tier_b
./scripts/agent_verify.sh --gpu    # gpu-marked tests only
```

## Cursor skill

[`.cursor/skills/amphiquantic-tdd/SKILL.md`](../.cursor/skills/amphiquantic-tdd/SKILL.md)

## Tests

See [tests/README.md](../tests/README.md).
