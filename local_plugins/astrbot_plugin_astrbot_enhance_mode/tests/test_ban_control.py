from __future__ import annotations

from astrbot_plugin_astrbot_enhance_mode import ban_control
from astrbot_plugin_astrbot_enhance_mode.ban_control import (
    BanStore,
    parse_duration_seconds,
)


def test_parse_duration_seconds_matrix() -> None:
    assert parse_duration_seconds("60") == 60
    assert parse_duration_seconds("10m") == 600
    assert parse_duration_seconds("2h") == 7200
    assert parse_duration_seconds("1day") == 86400
    assert parse_duration_seconds("0m") is None
    assert parse_duration_seconds("abc") is None
    assert parse_duration_seconds("") is None
    assert parse_duration_seconds(None) is None


def test_ban_store_scope_and_unban(tmp_path) -> None:
    store = BanStore(tmp_path / "ban.db")
    store.ban_user("qq:100", "u1", 300, source_origin="o1")
    store.ban_user("qq:200", "u1", 300, source_origin="o2")

    assert store.get_active_ban("qq:100", "u1") is not None
    assert store.get_active_ban("qq:200", "u1") is not None

    removed = store.unban_user("qq:100", "u1")
    assert removed is True
    assert store.get_active_ban("qq:100", "u1") is None
    assert store.get_active_ban("qq:200", "u1") is not None


def test_ban_store_expiration_and_cleanup(tmp_path, monkeypatch) -> None:
    fake_now = {"value": 1_700_000_000}
    monkeypatch.setattr(ban_control.time, "time", lambda: fake_now["value"])

    store = BanStore(tmp_path / "ban.db")
    expires_at = store.ban_user("qq:123", "u1", 10)
    assert expires_at == fake_now["value"] + 10

    active = store.get_active_ban("qq:123", "u1")
    assert active is not None
    assert active.remaining_seconds == 10

    fake_now["value"] += 11
    assert store.get_active_ban("qq:123", "u1") is None

    store.ban_user("qq:123", "u2", 10)
    fake_now["value"] += 11
    removed_count = store.cleanup_expired(scope_id="qq:123")
    assert removed_count >= 1
    assert store.list_active_bans("qq:123") == []
