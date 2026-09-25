from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_documents_phase_3a():
    text = (ROOT / "README.md").read_text()
    for needle in ("Phase 3a", "PROJECT_MEMORY.md", "lock", "audit", "Manager", "Admin", "reset_password"):
        assert needle in text, needle
    assert "gsk_" not in text


def test_gitignore_covers_runtime_state():
    ignore = (ROOT / ".gitignore").read_text()
    assert "memory/" in ignore and "data/*.db" in ignore and ".env" in ignore


def test_nothing_writes_a_repo_root_claude_md():
    for f in list((ROOT / "accounts").glob("*.py")) + list((ROOT / "ui").glob("*.py")) + [ROOT / "app.py"]:
        # The only allowed mention is the guard that refuses to write to it.
        lines = [ln for ln in f.read_text().splitlines() if "Refusing to write project memory" not in ln]
        assert "CLAUDE.md" not in "\n".join(lines), f
    assert not (ROOT / "CLAUDE.md").exists()


def test_accounts_package_uses_only_the_standard_library_for_crypto_and_storage():
    for f in (ROOT / "accounts").glob("*.py"):
        src = f.read_text()
        for banned in ("bcrypt", "passlib", "argon2", "sqlalchemy", "streamlit_authenticator"):
            assert banned not in src, (f, banned)


def test_streamlit_config_binds_to_localhost_only():
    import tomllib
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    server = cfg["server"]
    assert server["address"] == "127.0.0.1"
    assert server.get("enableCORS") is not False
    assert server.get("enableXsrfProtection") is not False


def test_streamlit_config_is_not_gitignored():
    for line in (ROOT / ".gitignore").read_text().splitlines():
        line = line.strip().rstrip("/")
        assert line not in (".streamlit", ".streamlit/config.toml", "*.toml", "config.toml"), line


def test_readme_explains_localhost_binding_and_lan_opt_in():
    text = (ROOT / "README.md").read_text()
    assert "127.0.0.1" in text and "--server.address" in text
