# Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
$ ./build
$ ./run
```

Or manually:

```bash
$ uv sync
$ uv run maturin develop --manifest-path rust/Cargo.toml
```

## Workflow

```bash
./bin/parameterize "C" --from-smiles -o methane.aqtop
./bin/solvate methane.aqtop -o solvated.aqtop
./bin/ionize solvated.aqtop
./bin/minimize solvated.ion.aqtop -o min.aqtop
./bin/equilibrate min.aqtop
./bin/simulate min.eq.aqtop
```

Force backend: `AMPHI_FORCE_BACKEND=cpu` for reference path (default: wgpu).

Validation: see [docs/validation.md](docs/validation.md).

## Agent development

Agents must follow the TDD protocol in [AGENTS.md](AGENTS.md):

```bash
./scripts/agent_verify.sh --fast   # iterate
./scripts/agent_verify.sh          # before handoff
```
