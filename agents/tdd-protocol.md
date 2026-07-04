# TDD Protocol for Agents

Follow this loop for every non-trivial change in amphiquantic.

## 1. Orient

- Identify the layer: topology, forces (cpu/gpu), integrator, workflow CLI, or validation.
- Read existing tests under `tests/` for the same layer.
- Check [`docs/validation.md`](../docs/validation.md) for acceptance criteria if touching physics.

## 2. RED — write the failing test

| Change type | Test location | Marker |
|-------------|---------------|--------|
| `.aqtop` / topology | `tests/agent/` or `tests/validation/` | `tier_a`, `rust` |
| CPU forces | `tests/validation/test_forces.py` | `tier_a`, `rust` |
| wgpu forces | same + `-m gpu` | `tier_a`, `gpu` |
| Parameterization | `tests/validation/test_openff_energy.py` | `openff` or bootstrap |
| Workflow CLI | `tests/agent/test_workflow_bins.py` | `tier_a` |
| Integrator / MD | `tests/validation/test_dynamics.py` | `tier_a`, `rust` |

**Rules:**

- One logical assertion per test when possible.
- Use `tmp_path` for filesystem artifacts; never commit generated `.aqtop` / `.aqtrj`.
- GPU tests must also have a CPU path or explicit parity assertion.
- Prefer bootstrap parameterization in default CI; gate OpenFF with `@pytest.mark.openff`.

Example RED step:

```bash
# Add test, then:
uv run pytest tests/validation/test_forces.py::test_bond_energy_harmonic -v
# Expect FAIL until implementation lands
```

## 3. GREEN — minimal implementation

- Match existing code style in the module you touch.
- wgpu changes: update **both** `rust/src/forces/gpu/` and `rust/data/shaders/forces.wgsl` when changing nonbonded physics.
- CPU reference: update `rust/src/forces/cpu/` in the same PR as GPU changes.
- Rebuild extension after Rust changes:

```bash
uv run maturin develop --manifest-path rust/Cargo.toml
```

## 4. REFACTOR

- Remove duplication only when tests stay green.
- Do not widen scope (no drive-by refactors).

## 5. VERIFY — agent gate

```bash
./scripts/agent_verify.sh          # standard
./scripts/agent_verify.sh --fast   # skip slow / openff / tier_b
./scripts/agent_verify.sh --gpu    # include gpu marker suite explicitly
```

**Done means:** script exits 0 and you can paste the summary line in your handoff.

## Physics change checklist

When changing forces, integrator, or electrostatics:

- [ ] CPU reference updated
- [ ] wgpu kernel updated (if applicable)
- [ ] `AMPHI_FORCE_BACKEND=cpu` and `gpu` both pass parity or documented tolerance
- [ ] Energy units documented (kJ/mol in Rust minimize path)
- [ ] [`docs/validation.md`](../docs/validation.md) updated if acceptance criteria change

## Anti-patterns (do not)

- Claiming done without running `agent_verify.sh`
- Adding `__init__.py` at repo root
- Using `pip install` instead of `uv sync`
- Skipping CPU reference when changing WGSL
- Importing GROMACS/CHARMM libraries (CLI export only)
