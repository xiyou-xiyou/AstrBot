# Local subscription and image integration

This directory stores plugin source, not runtime data. The core changes add the
`chatgpt_plan` provider and its OAuth login script. Existing upstream license and
copyright notices are retained.

## Install

From the AstrBot repository root on Windows:

```powershell
uv sync
New-Item -ItemType Directory -Force data/plugins
Copy-Item -Recurse -Force local_plugins/astrbot_plugin_codex_image data/plugins/
Copy-Item -Recurse -Force local_plugins/astrbot_plugin_astrbot_enhance_mode data/plugins/
.\.venv\Scripts\python.exe scripts/chatgpt_login.py
```

Follow the local login link printed by the script. In the AstrBot dashboard, add
the **ChatGPT Subscription** provider and select a model available to your account.
Authorization is stored locally in `data/chatgpt/credentials.json`; it is not
included in this repository. Availability depends on the account and service.

For drawing, install the Codex CLI, sign in with your own account, and ensure that
`codex` is available on the PATH used to start AstrBot. The CLI must support the
built-in image generation tool. Restart AstrBot after copying the plugins.

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py
```

Ask the bot to draw an image, or use `/画图 <description>`. The plugin supports up
to ten simultaneous jobs, each with a ten-minute deadline and up to five reference
images. Model tool calls run in AstrBot's managed background executor. A submitted
job is not a completed image; delivery happens after generation succeeds.

Reference images may come from AstrBot uploads, previously generated images, or
the current session/project workspace. Relative reference paths resolve against
that workspace. Another session's workspace is not automatically allowed.

## Files and cleanup

- Generated images: `data/codex_generated_images/`
- Per-job diagnostics: `data/codex_generated_images/_logs/`
- Session workspaces and downloaded references: `data/workspaces/`

Delete old generated images and logs when they are no longer needed. Avoid
deleting files used by active jobs. Do not commit the `data/` directory: it can
contain credentials, private messages, configuration, and images.

## Enhance Mode attribution and search

`astrbot_plugin_astrbot_enhance_mode` is a local copy of the plugin by 阿汐:
https://github.com/Axi404/astrbot_plugin_astrbot_enhance_mode
Its author metadata and README are preserved. This copy adds subscription-aware
web search: when its search provider is `chatgpt_plan`, it reuses the provider's
OAuth authorization and native Responses `web_search` tool. Configure and enable
the plugin's web search with that provider. Other providers retain the plugin's
existing HTTP search behavior.

Changes to installed plugin files under `data/plugins/` are not tracked by Git.
Copy source changes back to `local_plugins/` before committing them.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chatgpt_plan_source.py local_plugins/astrbot_plugin_codex_image/tests/test_generation.py local_plugins/astrbot_plugin_astrbot_enhance_mode/tests/test_web_search_tool.py -q
ruff format .
ruff check .
```

These regression tests mock remote generation/search calls; they do not submit
live image jobs or send messages to real chats.
