---
name: litert-subagent
description: A skill to execute local language models and agentic workflows using the litert-lm CLI directly within the VS Code integrated terminal.
version: 1.1.0
tags: [ai, local-llm, litert, gpu, inference, self-contained]
---

# Skill: Execute LiteRT CLI

## Purpose
This skill allows the agent to run local AI inference, process multimodal inputs, or execute tool-calling workflows using the `litert-lm` CLI.

## Workspace Structure
This skill operates within a self-contained directory structure:
*   `assets/`: Used for all input attachments (images, audio). 
*   `scripts/`: Used for generating any required Python code (e.g., tool presets).
*   **Model Files:** `.litertlm` files may exist in the root or `assets/` directory but are explicitly ignored in version control.

## Instructions
When the user asks to run a local model, process an image/audio locally, or test a tool-calling preset, construct and execute the appropriate `litert-lm` command in the VS Code terminal.

### Default Environment Guidelines
*   **Package Manager:** If a Python environment is required for tool presets or scripting, always use `uv`.
*   **Python Naming Conventions:** When generating `preset.py` files for tool-calling, ensure private variables do *not* have leading underscores.
*   **Hardware:** Assume an 8GB NVIDIA RTX 5060 GPU is available. Always favor `--backend=gpu`.
*   **File Placement:** Always save generated tools to `scripts/preset.py`. Always look for attachments in `assets/`.

### Command Construction Rules

1.  **Basic Inference (GPU Optimized):**
    ```bash
    litert-lm run <model-path>.litertlm \
      --backend=gpu \
      --enable-speculative-decoding=true \
      --prompt="<user-prompt>"
    ```

2.  **Multimodal Execution (Vision/Audio):**
    Append the necessary attachments using the `assets/` directory.
    ```bash
    litert-lm run <model-path>.litertlm \
      --backend=gpu \
      --vision-backend=gpu \
      --attachment=assets/<image-name>.jpg \
      --prompt="<user-prompt>"
    ```

3.  **Agentic Tool Use (Presets):**
    When the user requests an agent or function-calling workflow:
    1.  Generate a `preset.py` file containing the requested tools inside the `scripts/` folder.
    2.  Run the model referencing that preset:
    ```bash
    litert-lm run <model-path>.litertlm --preset=scripts/preset.py --backend=gpu
    ```

## CI Task Delegation (lint / test / build)

Claude never runs `nx lint`, `nx test`, `nx run pythonapi:format`, or a build
command directly (see `CLAUDE.md` Critical Rules). Every one of those tasks
goes through `scripts/run_ci_task.py` instead, which runs the command, and
on failure asks the local model to read the error, patch the file with
`scripts/preset.py`'s `write_file` tool, and try again — up to 3 attempts
(`max_attempts` in `run_ci_task.py`). A missing or misnamed nx target (an
infrastructure error, not a code error) fails immediately instead of
burning attempts — see `infrastructure_error_reason`.

### Invocation

```powershell
uv run --no-project python .claude/skills/litert-subagent/scripts/run_ci_task.py --task <task-name>
```

Task names (mirrors the CLAUDE.md Build & Test table):

| Task name          | Command                                           |
| ------------------ | ------------------------------------------------- |
| `lint-pythonapi`   | `nx lint pythonapi`                               |
| `test-pythonapi`   | `nx test pythonapi`                               |
| `format-pythonapi` | `nx run pythonapi:format`                         |
| `lint-web`         | `nx lint @agentic-executor/agentic-executor`      |
| `typecheck-web`    | `nx typecheck @agentic-executor/agentic-executor` |
| `test-web`         | `nx test @agentic-executor/agentic-executor`      |
| `e2e-web`          | `nx e2e agentic-executor-e2e`                     |
| `build-apps`       | `nx up apps`                                      |
| `lint-all`         | `nx run-many -t lint test typecheck`              |
| `affected`         | `nx affected -t lint test typecheck`              |

`lint-all` and `affected` cover every project nx knows about in one call —
Python and TypeScript alike, whatever apps exist today or get added later. A
project missing a target (pythonapi has no `typecheck`) is skipped by nx,
not treated as a failure. When one of these root-wide tasks turns up more
than one broken project, the fix loop tackles the first failing one per
attempt, not all of them in a single fix request — a later attempt (still
within the 3-attempt bound) picks up the next one once the first is fixed.

`typecheck-web` runs `tsc --noEmit` via an explicit target added in
`apps/agentic-executor/project.json` — `@nx/js/typescript`'s inferred
typecheck target does not apply to this project (it needs `tsconfig.lib.json`,
which Next apps don't have), so it had to be added by hand.

`scripts/preset.py`'s `read_file`/`write_file` resolve a relative path
against every folder under `apps/`, discovered at run time, not a hardcoded
list — a new app added under `apps/` works without editing this skill.

### Reading the result

The script prints exactly one summary line, plus a file list on failure.
Read only that — the full transcript of every attempt goes to
`assets/run_ci_task.log`, not the terminal, so it never enters Claude's
context. Open that log only if the user explicitly asks to debug a failure.

- `RESULT: PASS - <summary>` — task is clean. `<summary>` lists any fixes
  the local model made, or "no fix needed" if it passed on the first try.
- `RESULT: FAILED - <reason>` — the local model could not converge within 5
  attempts. A `Files touched:` line lists what it edited so far; nothing is
  auto-reverted. Hand this one to Claude or the user directly.

### Feasibility

The local model (`assets/gemma-4-12B-it-gpu.litertlm`, a quantized 12B
instruct model) is realistic for mechanical fixes — formatting, import
order, unused variables, simple type errors. It is not reliable for deep
logic bugs or subtle test failures. A `FAILED` result is expected sometimes,
not a bug in the script.

## Example Triggers
*   "Run the local Gemma model on the schematic in the assets folder."
*   "Create a quick litert app that can check the current time."
*   "Query the local model and make sure speculative decoding is on."
*   "Run pythonapi lint" / "run the tests" / "build the apps" (routes through
    `run_ci_task.py` — see CI Task Delegation above).

## Execution
Run the constructed command directly in the active VS Code terminal instance. Do not attempt to run this using standard Python `subprocess` unless explicitly asked to build a wrapper script inside the `scripts/` folder.