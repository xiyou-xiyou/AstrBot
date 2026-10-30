from collections import OrderedDict, defaultdict


class RuntimeState:
    def __init__(self) -> None:
        self.session_chats: dict[str, list[str]] = defaultdict(list)
        self.active_reply_stacks: dict[str, list[str]] = defaultdict(list)
        self.model_choice_histories: dict[str, list[str]] = defaultdict(list)
        self.image_message_registry: dict[str, dict[str, dict[str, object]]] = (
            defaultdict(dict)
        )
        self.origin_lru: OrderedDict[str, None] = OrderedDict()

    def _evict_origin_state(self, origin: str) -> None:
        self.session_chats.pop(origin, None)
        self.active_reply_stacks.pop(origin, None)
        self.model_choice_histories.pop(origin, None)
        self.image_message_registry.pop(origin, None)

    def touch_origin(self, origin: str, max_origins: int) -> None:
        if not origin:
            return
        self.origin_lru.pop(origin, None)
        self.origin_lru[origin] = None
        while len(self.origin_lru) > max_origins:
            oldest, _ = self.origin_lru.popitem(last=False)
            self._evict_origin_state(oldest)

    def cleanup_origin(self, origin: str) -> None:
        self._evict_origin_state(origin)
        self.origin_lru.pop(origin, None)
