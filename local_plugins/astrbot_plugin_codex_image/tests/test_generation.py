"""Regression coverage for Windows prompt transport and generated image selection."""

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "codex_image_under_test", Path(__file__).parents[1] / "main.py"
)
plugin_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_module)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_kind",
    [
        "path",
        "empty",
        "missing",
        "markdown",
        "stale",
        "outside",
        "other_session",
        "corrupt",
    ],
)
async def test_generation_transport_and_result(tmp_path, monkeypatch, result_kind):
    """Keep prompts intact and accept only fresh, valid generated images."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(
        plugin_module, "get_astrbot_data_path", lambda: str(tmp_path / "data")
    )
    monkeypatch.setattr(plugin_module.shutil, "which", lambda name: "codex.cmd")
    plugin = plugin_module.CodexImagePlugin(None)
    thread_id = "01a186a5-789c-70f0-ba72-ad8449a2805e"
    source = tmp_path / "codex" / "generated_images" / thread_id / "test.png"
    if result_kind == "outside":
        source = tmp_path / "outside.png"
    elif result_kind == "other_session":
        source = source.parent.parent / "unrelated-session" / "test.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "orange").save(source)
    if result_kind == "stale":
        os.utime(source, (time.time() - 600, time.time() - 600))
    elif result_kind == "corrupt":
        source.write_text("not an image", encoding="utf-8")

    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (
        (
            json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n"
        ).encode(),
        b"",
    )

    async def create_process(*args, **kwargs):
        """Write the final CLI reply without running an external command."""
        response_file = Path(args[args.index("--output-last-message") + 1])
        reply = str(source)
        if result_kind in {"empty", "other_session"}:
            reply = ""
        elif result_kind == "markdown":
            reply = f"Here is the image:\n![image]({source})"
        if result_kind != "missing":
            response_file.write_text(reply, encoding="utf-8")
        kwargs["stdout"].write(process.communicate.return_value[0])
        assert args[-1] == "-"
        assert "向日葵" not in " ".join(args)
        assert kwargs["stdin"] == plugin_module.asyncio.subprocess.PIPE
        return process

    monkeypatch.setattr(plugin_module.asyncio, "create_subprocess_exec", create_process)
    description = '橘猫与向日葵\n文字含有 "quotes" & %PATH% | > 测试'
    if result_kind in {"stale", "outside", "other_session", "corrupt"}:
        with pytest.raises(RuntimeError):
            await plugin._generate(description)
        assert source.exists()
    else:
        saved = await plugin._generate(description)
        assert saved.parent == plugin.image_dir
        assert saved.is_file()
        assert not source.exists()
        with Image.open(saved) as image:
            image.verify()
    assert description.encode("utf-8") in process.communicate.call_args.kwargs["input"]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows process tree and cwd locks")
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_interrupted_generation_reaps_process_tree(
    tmp_path, monkeypatch, termination
):
    """Release real Windows cwd locks when the runner cancels or drawing times out.

    Args:
        tmp_path: Isolated test filesystem root.
        monkeypatch: Patch only Codex startup and the generation deadline.
        termination: Whether the runner cancels or the plugin reaches its deadline.
    """
    monkeypatch.setattr(plugin_module, "get_astrbot_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(plugin_module.shutil, "which", lambda name: sys.executable)
    plugin = plugin_module.CodexImagePlugin(None)
    real_create = asyncio.create_subprocess_exec
    real_wait_for = asyncio.wait_for
    started = asyncio.Event()
    processes = []
    work_dirs = []

    async def start_process(*args, **kwargs):
        """Replace Codex with a real parent and child holding the work directory."""
        if "exec" not in args:
            return await real_create(*args, **kwargs)
        work = Path(args[args.index("--cd") + 1])
        code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "Path('child.pid').write_text(str(child.pid)); time.sleep(60)"
        )
        process = await real_create(sys.executable, "-c", code, cwd=work, **kwargs)
        processes.append(process)
        work_dirs.append(work)
        started.set()
        return process

    async def short_deadline(awaitable, timeout):
        """Accelerate the five-minute timeout without changing cleanup waits."""
        return await real_wait_for(
            awaitable, 1 if timeout == plugin_module.GENERATION_TIMEOUT else timeout
        )

    monkeypatch.setattr(plugin_module.asyncio, "create_subprocess_exec", start_process)
    if termination == "timeout":
        monkeypatch.setattr(plugin_module.asyncio, "wait_for", short_deadline)
    task = asyncio.create_task(plugin._generate("A test illustration"))
    try:
        await real_wait_for(started.wait(), 5)
        for _ in range(100):
            if (work_dirs[0] / "child.pid").is_file():
                break
            await asyncio.sleep(0.01)
        assert (work_dirs[0] / "child.pid").is_file()
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await real_wait_for(task, 10)
        else:
            with pytest.raises(RuntimeError, match="超过 10 分钟"):
                await real_wait_for(task, 10)
        assert processes[0].returncode is not None
        assert not work_dirs[0].exists()
        assert not plugin._generation_slots.locked()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for process in processes:
            if process.returncode is None:
                killer = await real_create(
                    "taskkill", "/PID", str(process.pid), "/T", "/F"
                )
                await killer.wait()
                await process.wait()


@pytest.mark.asyncio
async def test_ten_concurrent_generations_and_slot_recovery(tmp_path, monkeypatch):
    """Run ten jobs independently, reject overflow, and reuse a cancelled slot.

    Args:
        tmp_path: Isolated data and image output directory.
        monkeypatch: Replace Codex with controllable image-producing processes.
    """
    monkeypatch.setattr(plugin_module, "get_astrbot_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(plugin_module.shutil, "which", lambda name: "codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    plugin = plugin_module.CodexImagePlugin(None)
    gate = asyncio.Event()
    started = asyncio.Queue()
    jobs = []

    async def create_process(*args, **kwargs):
        """Keep each fake generation running until the shared gate opens."""
        index = len(jobs)
        work = Path(args[args.index("--cd") + 1])
        jobs.append(work)
        source = tmp_path / "codex" / "generated_images" / f"{index}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index, 0, 0)).save(source)
        Path(args[args.index("--output-last-message") + 1]).write_text(
            str(source), encoding="utf-8"
        )
        calls = 0

        async def communicate(**kwargs):
            """Allow the cancellation cleanup call to finish immediately."""
            nonlocal calls
            calls += 1
            if calls == 1:
                started.put_nowait(index)
                await gate.wait()
            return None, None

        return SimpleNamespace(returncode=0, communicate=communicate)

    monkeypatch.setattr(plugin_module.asyncio, "create_subprocess_exec", create_process)
    tasks = [asyncio.create_task(plugin._generate(f"Image {i}")) for i in range(10)]
    try:
        for _ in range(10):
            await asyncio.wait_for(started.get(), 5)
        assert len(set(jobs)) == 10
        assert plugin._generation_slots.locked()
        with pytest.raises(RuntimeError, match="已有 10 个绘图任务"):
            await plugin._generate("Overflow")
        assert len(jobs) == 10
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        tasks.append(asyncio.create_task(plugin._generate("Replacement")))
        assert await asyncio.wait_for(started.get(), 5) == 10
        gate.set()
        results = await asyncio.gather(*tasks[1:])
        assert len(set(results)) == 10
        assert all(path.is_file() for path in results)
        assert not plugin._generation_slots.locked()
        pixels = set()
        for path in results:
            with Image.open(path) as image:
                pixels.add(image.getpixel((0, 0))[0])
        assert pixels == set(range(1, 11))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_timeout", [120, 360])
async def test_runner_allows_slow_image_delivery(tmp_path, monkeypatch, tool_timeout):
    """Reproduce the old runner deadline and verify delivery with the new budget.

    Args:
        tmp_path: Isolated image directory.
        monkeypatch: Replace generation and accelerate the runner clock.
        tool_timeout: Old or repaired runner timeout in seconds.
    """
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool
    from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor

    monkeypatch.setattr(plugin_module, "get_astrbot_data_path", lambda: str(tmp_path))
    plugin = plugin_module.CodexImagePlugin(None)
    image = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "orange").save(image)
    real_wait_for = asyncio.wait_for

    async def delayed_generation(description, reference_images=None, **kwargs):
        """Represent a 180-second image generation on an accelerated clock."""
        await asyncio.sleep(0.18)
        return image

    async def accelerated_wait(awaitable, timeout):
        """Apply the same clock scale to the real AstrBot tool runner."""
        return await real_wait_for(awaitable, timeout / 1000)

    monkeypatch.setattr(plugin, "_generate", delayed_generation)
    monkeypatch.setattr(asyncio, "wait_for", accelerated_wait)
    event = SimpleNamespace(send=AsyncMock())
    context = ContextWrapper(
        context=SimpleNamespace(event=event), tool_call_timeout=tool_timeout
    )
    tool = FunctionTool(
        name="generate_image_with_codex",
        description="Test image delivery",
        parameters={"type": "object"},
        handler=plugin.generate_image_with_codex,
    )
    results = FunctionToolExecutor._execute_local(tool, context, description="Test")
    if tool_timeout == 120:
        with pytest.raises(Exception, match="execution timeout"):
            _ = [result async for result in results]
        event.send.assert_not_awaited()
    else:
        responses = [result async for result in results]
        event.send.assert_awaited_once()
        assert "图片已生成并发送" in responses[0].content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("path_in_description", [False, True])
@pytest.mark.parametrize(
    "location", ["temp", "workspace", "relative", "other_session", "traversal"]
)
async def test_reference_images_are_attached(
    tmp_path, monkeypatch, path_in_description, location
):
    """Attach a stable image copy for both explicit and legacy reference paths."""
    monkeypatch.setattr(plugin_module, "get_astrbot_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(plugin_module.shutil, "which", lambda name: "codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    workspace = tmp_path / "workspaces" / "current"
    workspace.mkdir(parents=True)
    resolver = AsyncMock(return_value=workspace)
    monkeypatch.setattr(plugin_module, "resolve_workspace_root_for_umo", resolver)
    reference = tmp_path / "temp" / "reference.png"
    if location in {"workspace", "relative"}:
        reference = workspace / "reference.png"
    elif location in {"other_session", "traversal"}:
        reference = workspace.parent / "other" / "reference.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "red").save(reference)
    source = tmp_path / "codex" / "generated_images" / "output.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "orange").save(source)
    process = AsyncMock(returncode=0)
    process.communicate.return_value = (None, None)

    async def create_process(*args, **kwargs):
        """Inspect the image attachment and return a generated file path."""
        attached = Path(args[args.index("--image") + 1])
        assert attached != reference
        assert attached.read_bytes() == reference.read_bytes()
        assert args[args.index("--disable") + 1] == "multi_agent"
        Path(args[args.index("--output-last-message") + 1]).write_text(
            str(source), encoding="utf-8"
        )
        return process

    monkeypatch.setattr(plugin_module.asyncio, "create_subprocess_exec", create_process)
    plugin = plugin_module.CodexImagePlugin(None)
    reference_arg = str(reference)
    if location == "relative":
        reference_arg = "reference.png"
    elif location == "traversal":
        reference_arg = "../other/reference.png"
    # Legacy descriptions only support absolute paths.
    if path_in_description and location in {"relative", "traversal"}:
        reference_arg = str(reference)
    description = (
        f"Draw from this reference: {reference_arg}"
        if path_in_description
        else "Draw from the reference"
    )
    event = SimpleNamespace(unified_msg_origin="default:GroupMessage:current")
    if location in {"other_session", "traversal"}:
        with pytest.raises(RuntimeError, match="参考图路径不在允许的目录"):
            await plugin._generate(
                description,
                None if path_in_description else [reference_arg],
                event=event,
            )
        process.communicate.assert_not_awaited()
        return
    saved = await plugin._generate(
        description, None if path_in_description else [reference_arg], event=event
    )
    resolver.assert_awaited_once_with(event.unified_msg_origin, None)
    assert saved.is_file()
    assert reference.is_file()


@pytest.mark.asyncio
async def test_background_task_submits_then_sends_image(tmp_path, monkeypatch):
    """Exercise AstrBot's background executor without posting to a real chat."""
    from astrbot.core import astr_agent_tool_exec
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool

    monkeypatch.setattr(plugin_module, "get_astrbot_data_path", lambda: str(tmp_path))
    plugin = plugin_module.CodexImagePlugin(None)
    gate = asyncio.Event()
    tasks = []
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "orange").save(image)

    async def generate(description, reference_images=None, **kwargs):
        """Hold generation until after the foreground tool call returns."""
        await gate.wait()
        return image

    monkeypatch.setattr(plugin, "_generate", generate)
    registry = SimpleNamespace(
        get_background_stop_signal=lambda event: asyncio.Event(),
        register_background_task=lambda event, task: tasks.append(task),
    )
    monkeypatch.setattr(astr_agent_tool_exec, "active_event_registry", registry)
    wake = AsyncMock()
    monkeypatch.setattr(
        astr_agent_tool_exec.FunctionToolExecutor,
        "_wake_main_agent_for_background_result",
        wake,
    )
    event = SimpleNamespace(send=AsyncMock(), get_extra=lambda key: None)
    context = ContextWrapper(
        context=SimpleNamespace(event=event), tool_call_timeout=120
    )
    tool = FunctionTool(
        name="generate_image_with_codex",
        description="Draw",
        parameters={"type": "object"},
        handler=plugin.generate_image_with_codex,
    )
    await plugin.enable_background_drawing(event, tool, {"description": "Draw"})
    try:
        results = [
            result
            async for result in astr_agent_tool_exec.FunctionToolExecutor.execute(
                tool, context, description="Draw"
            )
        ]
        assert "Background task submitted" in results[0].content[0].text
        event.send.assert_not_awaited()
        gate.set()
        await asyncio.gather(*tasks)
        event.send.assert_awaited_once()
        wake.assert_awaited_once()
        assert "图片已生成并发送" in wake.call_args.kwargs["result_text"]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
