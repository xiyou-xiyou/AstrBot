from __future__ import annotations

from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


def test_touch_origin_evicts_oldest_state() -> None:
    state = RuntimeState()

    state.session_chats["o1"].append("m1")
    state.active_reply_stacks["o1"].append("a1")
    state.model_choice_histories["o1"].append("h1")
    state.image_message_registry["o1"]["mid"] = {"urls": ["u1"], "captions": {}}

    state.touch_origin("o1", max_origins=1)
    state.touch_origin("o2", max_origins=1)

    assert "o1" not in state.session_chats
    assert "o1" not in state.active_reply_stacks
    assert "o1" not in state.model_choice_histories
    assert "o1" not in state.image_message_registry
    assert list(state.origin_lru.keys()) == ["o2"]


def test_cleanup_origin_removes_all_runtime_state() -> None:
    state = RuntimeState()
    state.session_chats["origin"].append("msg")
    state.active_reply_stacks["origin"].append("stack")
    state.model_choice_histories["origin"].append("hist")
    state.image_message_registry["origin"]["1"] = {"urls": ["x"], "captions": {}}
    state.touch_origin("origin", max_origins=10)

    state.cleanup_origin("origin")

    assert "origin" not in state.session_chats
    assert "origin" not in state.active_reply_stacks
    assert "origin" not in state.model_choice_histories
    assert "origin" not in state.image_message_registry
    assert "origin" not in state.origin_lru
