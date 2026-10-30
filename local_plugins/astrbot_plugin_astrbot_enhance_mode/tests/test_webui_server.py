from __future__ import annotations

from fastapi.testclient import TestClient

from astrbot_plugin_astrbot_enhance_mode.memory_rag_store import MemoryRAGStore
from astrbot_plugin_astrbot_enhance_mode.webui.server import RAGWebUIServer


def _login_and_get_token(client: TestClient, password: str) -> str:
    response = client.post("/api/login", json={"password": password})
    assert response.status_code == 200
    payload = response.json()
    assert "token" in payload
    return str(payload["token"])


def test_list_memories_invalid_pagination_falls_back(tmp_path) -> None:
    store = MemoryRAGStore(tmp_path / "memory_rag.db", display_timezone="UTC")
    store.add_memory(
        content="webui-memory",
        embedding=[1.0, 0.0],
        role_ids=["role-webui"],
        memory_time=1_700_000_000,
        group_scope="qq:100",
    )

    password = "pw-for-test"
    server = RAGWebUIServer(
        store=store,
        config={
            "host": "127.0.0.1",
            "port": 8899,
            "access_password": password,
            "session_timeout": 3600,
        },
        plugin_version="test",
    )

    with TestClient(server._app) as client:
        token = _login_and_get_token(client, password)
        headers = {"X-Auth-Token": token}

        response = client.get("/api/memories?page=abc&page_size=oops", headers=headers)
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["data"]["page"] == 1
        assert payload["data"]["page_size"] == 20
        assert payload["data"]["total"] == 1


def test_list_memories_pagination_numeric_bounds_are_normalized(tmp_path) -> None:
    store = MemoryRAGStore(tmp_path / "memory_rag.db", display_timezone="UTC")
    store.add_memory(
        content="webui-memory",
        embedding=[1.0, 0.0],
        role_ids=["role-webui"],
        memory_time=1_700_000_000,
        group_scope="qq:100",
    )

    password = "pw-for-test"
    server = RAGWebUIServer(
        store=store,
        config={
            "host": "127.0.0.1",
            "port": 8899,
            "access_password": password,
            "session_timeout": 3600,
        },
        plugin_version="test",
    )

    with TestClient(server._app) as client:
        token = _login_and_get_token(client, password)
        headers = {"X-Auth-Token": token}

        response = client.get("/api/memories?page=-9&page_size=9999", headers=headers)
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["data"]["page"] == 1
        assert payload["data"]["page_size"] == 200
        assert payload["data"]["total"] == 1
