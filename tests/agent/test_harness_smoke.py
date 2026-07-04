"""Meta-tests: the agent TDD harness itself must stay healthy."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.tier_a
def test_repo_layout(repo_root: Path):
    assert (repo_root / "AGENTS.md").is_file()
    assert (repo_root / "scripts" / "agent_verify.sh").is_file()
    assert (repo_root / "rust" / "Cargo.toml").is_file()
    assert (repo_root / "python" / "amphiquantic").is_dir()
    assert not (repo_root / "__init__.py").exists(), "root __init__.py shadows package"


@pytest.mark.tier_a
def test_agents_docs_exist(repo_root: Path):
    assert (repo_root / "agents" / "tdd-protocol.md").is_file()
    assert (repo_root / "agents" / "verification-checklist.md").is_file()


@pytest.mark.tier_a
def test_skill_registered(repo_root: Path):
    skill = repo_root / ".cursor" / "skills" / "amphiquantic-tdd" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text()
    assert "agent_verify" in text


@pytest.mark.tier_a
@pytest.mark.rust
def test_rustquantic_import(built_extension):
    import rustquantic

    assert hasattr(rustquantic, "Topology")
    assert hasattr(rustquantic, "load_topology")


@pytest.mark.tier_a
def test_amphiquantic_topology_import():
    from amphiquantic.topology import load_topology, save_topology

    assert callable(load_topology)
    assert callable(save_topology)
