from __future__ import annotations

import json

from astrbot_plugin_astrbot_enhance_mode.memory_rag_store import MemoryRAGStore


def test_add_get_delete_memory_roundtrip(tmp_path) -> None:
    store = MemoryRAGStore(tmp_path / "memory_rag.db", display_timezone="UTC")

    memory_id = store.add_memory(
        content="alpha memory",
        embedding=[1.0, 0.0],
        role_ids=["role-a", "role-b"],
        memory_time=1_700_000_000,
        group_scope="qq:100",
        group_id="100",
        platform_id="qq",
        extra_metadata={"source": "unit-test"},
    )
    got = store.get_memory(memory_id)

    assert got is not None
    assert got["memory_id"] == memory_id
    assert got["content"] == "alpha memory"
    assert got["group_scope"] == "qq:100"
    assert got["group_id"] == "100"
    assert got["platform_id"] == "qq"
    assert got["related_role_ids"] == ["role-a", "role-b"]
    assert got["extra_metadata"]["source"] == "unit-test"

    assert store.delete_memory(memory_id) is True
    assert store.get_memory(memory_id) is None


def test_list_memories_filters_and_paging(tmp_path) -> None:
    store = MemoryRAGStore(tmp_path / "memory_rag.db", display_timezone="UTC")
    m1 = store.add_memory(
        content="alpha",
        embedding=[1.0, 0.0],
        role_ids=["r1"],
        memory_time=100,
        group_scope="qq:1",
    )
    m2 = store.add_memory(
        content="beta",
        embedding=[0.5, 0.5],
        role_ids=["r2"],
        memory_time=200,
        group_scope="qq:1",
    )
    store.add_memory(
        content="gamma",
        embedding=[0.0, 1.0],
        role_ids=["r1"],
        memory_time=300,
        group_scope="qq:2",
    )

    scope_data = store.list_memories(page=1, page_size=10, group_scope="qq:1")
    assert scope_data["total"] == 2
    assert [item["memory_id"] for item in scope_data["items"]] == [m2, m1]

    role_data = store.list_memories(page=1, page_size=10, group_scope="qq:1", role_id="r1")
    assert role_data["total"] == 1
    assert role_data["items"][0]["memory_id"] == m1

    keyword_data = store.list_memories(page=1, page_size=10, keyword="beta")
    assert keyword_data["total"] == 1
    assert keyword_data["items"][0]["memory_id"] == m2


def test_search_memories_relevance_role_and_time_filters(tmp_path) -> None:
    store = MemoryRAGStore(tmp_path / "memory_rag.db", display_timezone="UTC")
    m1 = store.add_memory(
        content="first",
        embedding=[1.0, 0.0],
        role_ids=["r1"],
        memory_time=100,
        group_scope="qq:100",
    )
    m2 = store.add_memory(
        content="second",
        embedding=[0.9, 0.1],
        role_ids=["r1", "r2"],
        memory_time=200,
        group_scope="qq:100",
    )
    store.add_memory(
        content="third",
        embedding=[0.0, 1.0],
        role_ids=["r3"],
        memory_time=300,
        group_scope="qq:100",
    )

    relevance = store.search_memories(
        query_embedding=[1.0, 0.0],
        embedding_recall_k=10,
        role_ids=["r1"],
        role_match_mode="any",
        group_scope="qq:100",
        sort_by="relevance",
        sort_order="desc",
        max_results=2,
    )
    assert [item["memory_id"] for item in relevance] == [m1, m2]

    role_all = store.search_memories(
        role_ids=["r1", "r2"],
        role_match_mode="all",
        group_scope="qq:100",
        sort_by="time",
        sort_order="desc",
        max_results=10,
    )
    assert [item["memory_id"] for item in role_all] == [m2]

    time_filtered = store.search_memories(
        group_scope="qq:100",
        start_time=150,
        end_time=250,
        sort_by="time",
        sort_order="asc",
        max_results=10,
    )
    assert [item["memory_id"] for item in time_filtered] == [m2]


def test_cleanup_legacy_records_normalizes_metadata_and_timestamp(tmp_path) -> None:
    store = MemoryRAGStore(tmp_path / "memory_rag.db", display_timezone="UTC")
    memory_id = store.add_memory(
        content="legacy",
        embedding=[1.0, 1.0],
        role_ids=["legacy-role"],
        memory_time=1_700_000_000,
    )

    with store._lock, store._connect() as conn:
        conn.execute(
            "UPDATE rag_memories SET memory_time = ?, extra_metadata = ? WHERE id = ?",
            (1_735_689_600_000, "{}", memory_id),
        )
        conn.commit()

    result = store.cleanup_legacy_records()
    assert result["status"] == "ok"
    assert result["scanned"] >= 1
    assert result["updated"] >= 1

    got = store.get_memory(memory_id)
    assert got is not None
    assert got["memory_time"] < 1e12
    cleanup_meta = got["extra_metadata"].get("_enhance_cleanup")
    assert isinstance(cleanup_meta, dict)
    assert cleanup_meta.get("version") == 2
