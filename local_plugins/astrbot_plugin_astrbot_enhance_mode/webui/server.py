from __future__ import annotations

import asyncio
import secrets
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from astrbot.api import logger

from ..memory_rag_store import MemoryRAGStore


class RAGWebUIServer:
    def __init__(
        self,
        store: MemoryRAGStore,
        config: dict[str, Any],
        plugin_version: str = "0.2.4",
    ) -> None:
        self.store = store
        self.config = config
        self.host = str(config.get("host", "127.0.0.1"))
        self.port = int(config.get("port", 8899))
        self.session_timeout = max(60, int(config.get("session_timeout", 3600)))
        self._access_password = str(config.get("access_password", "")).strip()
        self._password_generated = False
        if not self._access_password:
            self._access_password = secrets.token_urlsafe(16)
            self._password_generated = True
            logger.info(
                "enhance-mode | RAG WebUI access password is auto generated: %s",
                self._access_password,
            )

        self._tokens: dict[str, dict[str, float]] = {}
        self._token_lock = asyncio.Lock()
        self._failed_attempts: dict[str, list[float]] = {}
        self._attempt_lock = asyncio.Lock()

        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        self._app = FastAPI(title="Enhance Memory RAG WebUI", version=plugin_version)
        self._setup_routes()

    @property
    def access_password(self) -> str:
        return self._access_password

    @property
    def password_generated(self) -> bool:
        return self._password_generated

    async def start(self) -> None:
        if self._server_task and not self._server_task.done():
            return

        logger.info(
            "enhance-mode | RAG WebUI starting | host=%s port=%s session_timeout=%ss",
            self.host,
            self.port,
            self.session_timeout,
        )
        config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        for _ in range(50):
            if getattr(self._server, "started", False):
                logger.info(
                    "enhance-mode | RAG WebUI started at http://%s:%s",
                    self.host,
                    self.port,
                )
                return
            if self._server_task.done():
                error = self._server_task.exception()
                raise RuntimeError(f"RAG WebUI start failed: {error}") from error
            await asyncio.sleep(0.1)

        logger.warning("enhance-mode | RAG WebUI is still starting in background")

    async def stop(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._server:
            self._server.should_exit = True
        if self._server_task:
            await self._server_task

        self._server = None
        self._server_task = None
        self._cleanup_task = None
        logger.info("enhance-mode | RAG WebUI stopped")

    async def _periodic_cleanup(self) -> None:
        while True:
            try:
                await asyncio.sleep(300)
                async with self._token_lock:
                    self._cleanup_tokens_locked()
                async with self._attempt_lock:
                    self._cleanup_failed_attempts_locked()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("enhance-mode | RAG WebUI cleanup failed: %s", e)

    def _cleanup_tokens_locked(self) -> None:
        now_ts = time.time()
        expired = []
        for token, token_info in self._tokens.items():
            created_at = token_info.get("created_at", 0.0)
            last_active = token_info.get("last_active", 0.0)
            max_lifetime = token_info.get("max_lifetime", 86400.0)
            if now_ts - created_at > max_lifetime:
                expired.append(token)
            elif now_ts - last_active > self.session_timeout:
                expired.append(token)
        for token in expired:
            self._tokens.pop(token, None)

    def _cleanup_failed_attempts_locked(self) -> None:
        now_ts = time.time()
        stale_ips = []
        for ip, attempts in self._failed_attempts.items():
            recent = [ts for ts in attempts if now_ts - ts < 300]
            if recent:
                self._failed_attempts[ip] = recent
            else:
                stale_ips.append(ip)
        for ip in stale_ips:
            self._failed_attempts.pop(ip, None)

    async def _check_rate_limit(self, client_ip: str) -> bool:
        async with self._attempt_lock:
            self._cleanup_failed_attempts_locked()
            attempts = self._failed_attempts.get(client_ip, [])
            return len(attempts) < 8

    async def _record_failed_attempt(self, client_ip: str) -> None:
        async with self._attempt_lock:
            self._failed_attempts.setdefault(client_ip, []).append(time.time())

    def _extract_token(self, request: Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return request.headers.get("X-Auth-Token", "")

    async def _validate_token(self, token: str) -> None:
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing auth token.",
            )
        async with self._token_lock:
            token_info = self._tokens.get(token)
            if not token_info:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token.",
                )

            now_ts = time.time()
            created_at = token_info.get("created_at", 0.0)
            last_active = token_info.get("last_active", 0.0)
            max_lifetime = token_info.get("max_lifetime", 86400.0)

            if (
                now_ts - created_at > max_lifetime
                or now_ts - last_active > self.session_timeout
            ):
                self._tokens.pop(token, None)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired.",
                )

            token_info["last_active"] = now_ts

    def _auth_dependency(self):
        async def dependency(request: Request) -> str:
            token = self._extract_token(request)
            await self._validate_token(token)
            return token

        return dependency

    def _setup_routes(self) -> None:
        static_dir = Path(__file__).resolve().parent.parent / "static"
        index_path = static_dir / "index.html"
        if static_dir.exists():
            self._app.mount("/static", StaticFiles(directory=static_dir), name="static")

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                f"http://{self.host}:{self.port}",
                "http://localhost",
                "http://127.0.0.1",
            ],
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "X-Auth-Token"],
            allow_credentials=True,
        )

        @self._app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            if not index_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="WebUI static files not found.",
                )
            return HTMLResponse(index_path.read_text(encoding="utf-8"))

        @self._app.get("/api/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok"}

        @self._app.post("/api/login")
        async def login(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
            password = str(payload.get("password", "")).strip()
            if not password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password is required.",
                )

            client_ip = (
                request.client.host
                if request.client and request.client.host
                else "unknown"
            )
            if not await self._check_rate_limit(client_ip):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed attempts. Try again later.",
                )

            if password != self._access_password:
                logger.warning(
                    "enhance-mode | RAG WebUI login failed | ip=%s reason=wrong_password",
                    client_ip,
                )
                await self._record_failed_attempt(client_ip)
                await asyncio.sleep(0.8)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed.",
                )

            token = secrets.token_urlsafe(32)
            now_ts = time.time()
            async with self._token_lock:
                self._cleanup_tokens_locked()
                self._tokens[token] = {
                    "created_at": now_ts,
                    "last_active": now_ts,
                    "max_lifetime": 86400.0,
                }
            logger.info(
                "enhance-mode | RAG WebUI login success | ip=%s active_tokens=%s",
                client_ip,
                len(self._tokens),
            )
            return {"token": token, "expires_in": self.session_timeout}

        @self._app.post("/api/logout")
        async def logout(
            token: str = Depends(self._auth_dependency()),
        ) -> dict[str, Any]:
            async with self._token_lock:
                self._tokens.pop(token, None)
                active_tokens = len(self._tokens)
            logger.info(
                "enhance-mode | RAG WebUI logout | active_tokens=%s",
                active_tokens,
            )
            return {"success": True}

        @self._app.get("/api/stats")
        async def stats(
            token: str = Depends(self._auth_dependency()),
        ) -> dict[str, Any]:
            _ = token
            return {"success": True, "data": self.store.get_stats()}

        @self._app.post("/api/cleanup")
        async def cleanup(
            token: str = Depends(self._auth_dependency()),
        ) -> dict[str, Any]:
            _ = token
            result = await asyncio.to_thread(self.store.cleanup_legacy_records)
            logger.info(
                "enhance-mode | RAG WebUI cleanup done | scanned=%s updated=%s timezone=%s",
                result.get("scanned"),
                result.get("updated"),
                result.get("display_timezone"),
            )
            return {"success": True, "data": result}

        @self._app.get("/api/memories")
        async def list_memories(
            request: Request,
            token: str = Depends(self._auth_dependency()),
        ) -> dict[str, Any]:
            _ = token
            query = request.query_params
            try:
                page = int(query.get("page", 1))
            except (TypeError, ValueError):
                page = 1
            try:
                page_size = int(query.get("page_size", 20))
            except (TypeError, ValueError):
                page_size = 20
            keyword = str(query.get("keyword", ""))
            group_scope = str(query.get("group_scope", ""))
            role_id = str(query.get("role_id", ""))
            data = self.store.list_memories(
                page=page,
                page_size=page_size,
                keyword=keyword,
                group_scope=group_scope,
                role_id=role_id,
            )
            return {"success": True, "data": data}

        @self._app.get("/api/memories/{memory_id}")
        async def get_memory(
            memory_id: int,
            token: str = Depends(self._auth_dependency()),
        ) -> dict[str, Any]:
            _ = token
            memory = self.store.get_memory(memory_id)
            if not memory:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Memory not found.",
                )
            return {"success": True, "data": memory}

        @self._app.delete("/api/memories/{memory_id}")
        async def delete_memory(
            memory_id: int,
            token: str = Depends(self._auth_dependency()),
        ) -> dict[str, Any]:
            _ = token
            removed = self.store.delete_memory(memory_id)
            if not removed:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Memory not found.",
                )
            logger.info(
                "enhance-mode | RAG WebUI memory deleted | memory_id=%s", memory_id
            )
            return {"success": True, "message": f"Memory {memory_id} deleted."}
