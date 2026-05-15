from pathlib import Path
import pytest
import tempfile

from vault.config.discovery import (
    get_or_prompt_vault,
    find_obsidian_vaults,
    save_config,
    _load_config,
)


def test_get_or_prompt_vault_valid_config(monkeypatch):
    print_divider("TEST 5 — GET OR PROMPT (VALID CONFIG)")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vault = create_fake_vault(root, "my_vault")

        # Force config to return valid vault path
        monkeypatch.setattr(
            "vault.config.discovery._load_config",
            lambda: {"vault_path": str(vault)}
        )

        # Prevent setup from running
        monkeypatch.setattr(
            "vault.config.discovery._run_setup",
            lambda: Exception("Should not be called")
        )

        result = get_or_prompt_vault()

        print("Result:", result)

        assert result == vault


def test_get_or_prompt_vault_missing_path(monkeypatch):
    print_divider("TEST 6 — GET OR PROMPT (MISSING PATH)")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vault = create_fake_vault(root, "new_vault")

        # Config points to non-existent path
        monkeypatch.setattr(
            "vault.config.discovery._load_config",
            lambda: {"vault_path": str(root / "does_not_exist")}
        )

        # Mock setup to return a known vault
        monkeypatch.setattr(
            "vault.config.discovery._run_setup",
            lambda: vault
        )

        result = get_or_prompt_vault()

        print("Result:", result)

        assert result == vault


def create_fake_vault(base: Path, name: str):
    vault = base / name
    vault.mkdir(parents=True, exist_ok=True)

    # Obsidian marker
    (vault / ".obsidian").mkdir()

    # Sample note
    (vault / "note.md").write_text("# Test Note")

    return vault


def print_divider(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_find_obsidian_vaults():
    print_divider("TEST 1 — FIND OBSIDIAN VAULTS")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Create fake vaults
        vault_a = create_fake_vault(root, "vault_a")
        vault_b = create_fake_vault(root, "vault_b")

        # Create normal directory
        (root / "random_folder").mkdir()

        # Discover vaults
        results = find_obsidian_vaults(start=root)

        print("\nDiscovered Vaults:")
        for r in results:
            print("-", r)

        print(f"\nTotal Found: {len(results)}")

        assert vault_a in results
        assert vault_b in results
        assert len(results) == 2


def test_save_and_load_config():
    print_divider("TEST 2 — SAVE + LOAD CONFIG")

    with tempfile.TemporaryDirectory() as tmp:
        fake_vault = Path(tmp) / "my_vault"
        fake_vault.mkdir()

        # Save config
        save_config(fake_vault)

        # Reload config
        cfg = _load_config()

        print("\nLoaded Config:")
        print(cfg)

        assert cfg is not None
        assert "vault_path" in cfg
        assert cfg["vault_path"] == str(fake_vault)


def test_bfs_depth_behavior():
    print_divider("TEST 3 — BFS DEPTH BEHAVIOR")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Shallow vault
        shallow = root / "shallow_vault"
        shallow.mkdir()
        (shallow / ".obsidian").mkdir()

        # Deep vault (may exceed MAX_DEPTH)
        deep = root
        for i in range(6):
            deep = deep / f"nested_{i}"
            deep.mkdir()

        (deep / ".obsidian").mkdir()

        results = find_obsidian_vaults(start=root)

        print("\nDiscovered Vaults:")
        for r in results:
            print("-", r)

        print("\nExpected:")
        print("- shallow vault should appear")
        print("- deep vault may be skipped due to MAX_DEPTH")

        assert shallow in results


def test_non_obsidian_folder():
    print_divider("TEST 4 — NON-OBSIDIAN FOLDER")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Folder without .obsidian
        random_dir = root / "random_notes"
        random_dir.mkdir()

        results = find_obsidian_vaults(start=root)

        print("\nDiscovered Vaults:")
        print(results)

        assert random_dir not in results
        assert len(results) == 0


if __name__ == "__main__":
    test_find_obsidian_vaults()
    test_save_and_load_config()
    test_bfs_depth_behavior()
    test_non_obsidian_folder()

    pytest.main([__file__])