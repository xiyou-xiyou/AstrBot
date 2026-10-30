"""Generate images with the locally signed-in Codex CLI."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image as PILImage

from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.workspace import resolve_workspace_root_for_umo

GENERATION_TIMEOUT = 600
MAX_CONCURRENT_GENERATIONS = 10


class CodexImagePlugin(Star):
    """Route image requests to Codex and send the resulting local file."""

    def __init__(self, context: Context) -> None:
        super().__init__(context)
        self.image_dir = Path(get_astrbot_data_path()) / "codex_generated_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._generation_slots = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

    async def _generate(
        self,
        description: str,
        reference_images: list[str] | None = None,
        *,
        event: AstrMessageEvent | None = None,
    ) -> Path:
        """Run Codex image generation and copy its output into AstrBot data.

        Args:
            description: User-provided description of the requested image.
            reference_images: Local images uploaded to AstrBot to use as references.
            event: Originating event used to resolve the current session workspace.

        Returns:
            Path to the validated image saved under the AstrBot data directory.

        Raises:
            RuntimeError: Codex is unavailable or did not produce a usable image.
        """
        if not description.strip():
            raise RuntimeError("请提供绘图描述。")
        codex = shutil.which("codex.exe") if os.name == "nt" else None
        codex = codex or shutil.which("codex")
        if not codex:
            raise RuntimeError(
                "找不到 Codex 命令，请确认 AstrBot 与 Codex 在同一用户环境运行。"
            )
        if self._generation_slots.locked():
            raise RuntimeError(
                "当前已有 10 个绘图任务正在进行，请等其中一个完成后再试。"
            )

        async with self._generation_slots:
            job_id = uuid4().hex
            log_dir = self.image_dir / "_logs"
            log_dir.mkdir(exist_ok=True)
            stdout_path = log_dir / f"{job_id}.jsonl"
            stderr_path = log_dir / f"{job_id}.stderr.log"
            with (
                tempfile.TemporaryDirectory(
                    prefix=".codex-run-", dir=self.image_dir, ignore_cleanup_errors=True
                ) as work,
                stdout_path.open("wb") as output,
                stderr_path.open("wb") as errors,
            ):
                work_dir = Path(work)
                response_file = work_dir / "last_message.txt"
                # Older conversations put reference paths in the description.
                references = reference_images or re.findall(
                    r'(?:[A-Za-z]:[\\/]|/)[^\r\n"<>|]+?\.(?:png|jpe?g|webp)',
                    description,
                    flags=re.IGNORECASE,
                )
                if len(references) > 5:
                    raise RuntimeError("一次绘图最多支持 5 张参考图片。")
                image_args = []
                data_root = Path(get_astrbot_data_path()).resolve()
                allowed_roots = [
                    data_root / "temp",
                    data_root / "attachments",
                    data_root / "webchat",
                    self.image_dir.resolve(),
                ]
                workspace = None
                if references and event is not None:
                    workspace = await resolve_workspace_root_for_umo(
                        event.unified_msg_origin, getattr(self.context, "_db", None)
                    )
                    workspace = workspace.resolve()
                    allowed_roots.append(workspace)
                for index, reference in enumerate(references):
                    path = Path(reference)
                    if not path.is_absolute() and workspace is not None:
                        path = workspace / path
                    path = path.resolve()
                    if not any(path.is_relative_to(root) for root in allowed_roots):
                        logger.warning(
                            "Rejected reference image outside allowed roots: %s", path
                        )
                        raise RuntimeError(
                            "参考图路径不在允许的目录中，请使用上传图片或当前会话工作目录内的图片。"
                        )
                    try:
                        with PILImage.open(path) as image:
                            image.verify()
                        local_reference = work_dir / f"reference_{index}{path.suffix}"
                        shutil.copy2(path, local_reference)
                    except (OSError, ValueError) as exc:
                        logger.warning("Cannot read reference image %s: %s", path, exc)
                        raise RuntimeError(
                            "参考图片已失效或无法读取，请重新上传。"
                        ) from exc
                    image_args.extend(["--image", str(local_reference)])
                started_at = time.time()
                instruction = (
                    "$imagegen Generate one image for the description below using the built-in "
                    "image generation tool directly. Do not delegate to other agents, "
                    "use shell commands, resize, or copy files. "
                    "Your final answer must contain only the absolute path of the generated image file.\n\n"
                    "Use the attached reference images when provided; they are already included in this request.\n"
                    f"Image description:\n{description.strip()}"
                )
                logger.info(
                    "Codex image job %s started with %s references; log: %s",
                    job_id,
                    len(references),
                    stdout_path,
                )
                process = await asyncio.create_subprocess_exec(
                    codex,
                    "exec",
                    "--ephemeral",
                    "--disable",
                    "multi_agent",
                    "--sandbox",
                    "workspace-write",
                    "--skip-git-repo-check",
                    "--json",
                    "-c",
                    'model_reasoning_effort="low"',
                    "--cd",
                    str(work_dir),
                    "--output-last-message",
                    str(response_file),
                    *image_args,
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                )
                try:
                    await asyncio.wait_for(
                        process.communicate(input=instruction.encode("utf-8")),
                        timeout=GENERATION_TIMEOUT,
                    )
                except (TimeoutError, asyncio.CancelledError) as exc:
                    # The agent runner can cancel us before our own timeout.
                    # Reap Codex and its Windows children before removing their cwd.
                    if process.returncode is None:
                        if os.name == "nt":
                            killer = await asyncio.create_subprocess_exec(
                                str(
                                    Path(os.environ["SystemRoot"])
                                    / "System32"
                                    / "taskkill.exe"
                                ),
                                "/PID",
                                str(process.pid),
                                "/T",
                                "/F",
                                stdout=asyncio.subprocess.DEVNULL,
                                stderr=asyncio.subprocess.DEVNULL,
                            )
                            await killer.wait()
                        if process.returncode is None:
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                    await process.communicate()
                    logger.warning(
                        "Codex image job %s interrupted (%s); diagnostics: %s",
                        job_id,
                        type(exc).__name__,
                        stdout_path,
                    )
                    if isinstance(exc, TimeoutError):
                        raise RuntimeError(
                            "Codex 绘图等待超过 10 分钟，任务已停止，请稍后重试。"
                        ) from exc
                    raise
                output.flush()
                errors.flush()
                stdout = stdout_path.read_bytes()
                stderr = stderr_path.read_bytes()
                if process.returncode != 0:
                    logger.error(
                        "Codex image generation failed (exit %s): %s",
                        process.returncode,
                        (stderr or stdout).decode("utf-8", errors="replace")[-1000:],
                    )
                    raise RuntimeError(
                        "Codex 绘图失败，请检查 AstrBot 日志和 Codex 登录状态。"
                    )

                codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
                source_root = (codex_home / "generated_images").resolve()
                # Image tools can finish without a textual path. Only inspect this
                # invocation's image directory, never another concurrent session.
                session_dir = None
                for line in stdout.decode("utf-8", errors="replace").splitlines():
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "thread.started":
                            session_dir = source_root / str(UUID(entry["thread_id"]))
                            break
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue
                candidates = (
                    sorted(
                        session_dir.glob("*"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if session_dir and session_dir.is_dir()
                    else []
                )
                candidate = (
                    response_file.read_text(encoding="utf-8").strip().strip('`" ')
                    if response_file.is_file()
                    else ""
                )
                if candidate and "\n" not in candidate:
                    try:
                        candidates.append(Path(candidate).resolve())
                    except (OSError, ValueError):
                        pass
                source = None
                for path in candidates:
                    path = path.resolve()
                    if (
                        path.is_relative_to(source_root)
                        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                        and path.is_file()
                        and path.stat().st_mtime >= started_at - 10
                    ):
                        source = path
                        break
                if source is None:
                    logger.error(
                        "Codex returned an invalid generated image path: %s", candidate
                    )
                    raise RuntimeError("Codex 未返回本次生成的有效图片。")
                try:
                    with PILImage.open(source) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    raise RuntimeError("Codex 生成的图片文件无法读取。") from exc

                destination = self.image_dir / f"{uuid4().hex}{source.suffix.lower()}"
                shutil.move(source, destination)
                logger.info("Codex generated image saved to %s", destination)
                return destination

    @filter.on_using_llm_tool()
    async def enable_background_drawing(
        self, event: AstrMessageEvent, tool: FunctionTool, tool_args: dict | None
    ) -> None:
        """Run drawing through AstrBot's managed background task executor.

        Args:
            event: Message event that requested the tool.
            tool: Tool selected by the agent, including permission wrappers.
            tool_args: Arguments prepared by the agent.
        """
        if tool.name == "generate_image_with_codex":
            tool.is_background_task = True

    @filter.on_llm_request()
    async def advertise_drawing_tool(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        """Tell the model when the installed drawing tool should be used.

        Args:
            event: Incoming AstrBot message event.
            request: Model request being prepared by AstrBot.
        """
        if request.func_tool and request.func_tool.get_tool(
            "generate_image_with_codex"
        ):
            request.system_prompt = (request.system_prompt or "") + (
                "\nYou can generate and send images through the "
                "generate_image_with_codex tool. When a user asks for an image, "
                "or accepts an image you previously offered to send, call the tool "
                "using the image description from the conversation. Do not claim "
                "that no image interface is available. Pass local reference image paths "
                "in reference_images when the user supplies references. Downloaded "
                "references must be saved in the current session workspace; pass their "
                "actual local paths, never URLs or invented paths. Drawing runs "
                "in the background and automatically sends the image when ready. "
                "A submitted task is still in progress, not a completed image. "
                "Do not submit a duplicate while waiting.\n"
            )

    @filter.command("画图", alias={"绘图"})
    async def draw(self, event: AstrMessageEvent, description: GreedyStr) -> None:
        """Generate and send an image from a direct command.

        Args:
            event: Incoming AstrBot message event.
            description: Description to draw.
        """
        try:
            image_path = await self._generate(str(description), event=event)
        except RuntimeError as exc:
            yield event.plain_result(str(exc))
            return
        yield event.chain_result(MessageChain().file_image(str(image_path)).chain)

    @llm_tool("generate_image_with_codex")
    async def generate_image_with_codex(
        self,
        event: AstrMessageEvent,
        description: str,
        reference_images: list[str] | None = None,
    ) -> str:
        """Generate and send an image when the user requests a drawing or illustration.

        Args:
            description(string): Detailed visual description of the image to create.
            reference_images(array[string]): Optional local paths of uploaded images or images saved in the current session workspace; omit for text-only generation.

        Returns:
            Status of image generation and delivery.
        """
        try:
            image_path = await self._generate(
                description, reference_images, event=event
            )
            await event.send(MessageChain().file_image(str(image_path)))
            logger.info("Codex image delivered: %s", image_path)
        except RuntimeError as exc:
            return str(exc)
        return "图片已生成并发送。"
