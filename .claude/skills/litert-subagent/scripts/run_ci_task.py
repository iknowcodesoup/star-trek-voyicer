"""Runs a lint/test/build task and lets the local model fix failures.

Claude invokes this script instead of running nx commands directly. Only the
final RESULT line is meant to reach Claude's context; the full transcript of
every attempt goes to a log file for human debugging.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

skill_root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[4]
assets_dir = skill_root / "assets"
model_path = assets_dir / "gemma-4-12B-it-gpu.litertlm"
preset_path = skill_root / "scripts" / "preset.py"
fix_summary_path = assets_dir / "last_fix_summary.txt"
touched_files_path = assets_dir / "touched_files.txt"
log_path = assets_dir / "run_ci_task.log"

max_attempts = 3

tasks = {
    "lint-pythonapi": "nx lint pythonapi",
    "test-pythonapi": "nx test pythonapi",
    "format-pythonapi": "nx run pythonapi:format",
    "lint-web": "nx lint @agentic-executor/agentic-executor",
    "typecheck-web": "nx typecheck @agentic-executor/agentic-executor",
    "test-web": "nx test @agentic-executor/agentic-executor",
    "e2e-web": "nx e2e agentic-executor-e2e",
    "build-apps": "nx up apps",
    # Root-wide: covers every project nx knows about, Python and TS alike.
    # A project missing a target (e.g. pythonapi has no typecheck) is
    # skipped by nx, not treated as a failure.
    "lint-all": "nx run-many -t lint test typecheck",
    "affected": "nx affected -t lint test typecheck",
}


def append_log(text: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text)
        log_file.write("\n")


def run_task_command(command: str) -> subprocess.CompletedProcess:
    # nx resolves to nx.ps1 on this machine. cmd.exe (shell=True) will not
    # find a .ps1 script, so invoke through PowerShell explicitly.
    powershell_command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    # text=True with no encoding falls back to the locale's preferred
    # encoding, which on this host is cp1252 even though the console code
    # page is UTF-8. Jest's default reporter writes UTF-8 pass/fail glyphs
    # (checkmark, cross), and cp1252 cannot decode them - that crashes the
    # subprocess reader thread and loses the whole result. Decoding as
    # UTF-8 explicitly, with a replacement fallback for anything stricter
    # tools still emit outside it, keeps every task's output capturable.
    return subprocess.run(
        powershell_command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=repo_root,
    )


ansi_escape_pattern = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return ansi_escape_pattern.sub("", text)


def condense_failure(result: subprocess.CompletedProcess) -> str:
    output = strip_ansi(result.stdout + result.stderr)
    lines = [line for line in output.splitlines() if line.strip()]

    # lint-all/affected run many projects at once. Nx lists every failure
    # under "Failed tasks:" - isolate just the first one's own output block
    # so the model fixes one thing at a time instead of a jumbled mix of
    # unrelated Python and TypeScript errors. Later attempts pick up the
    # next failing task once this one is fixed.
    failed_tasks_index = next(
        (i for i, line in enumerate(lines) if line.strip() == "Failed tasks:"), None
    )
    if failed_tasks_index is not None:
        first_failed_task = next(
            (
                line.strip().lstrip("-").strip()
                for line in lines[failed_tasks_index + 1 :]
                if line.strip().startswith("-")
            ),
            None,
        )
        if first_failed_task:
            header_index = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if f"nx run {first_failed_task}" in line
                ),
                None,
            )
            if header_index is not None:
                return "\n".join(lines[header_index : header_index + 60])

    return "\n".join(lines[-40:])


infrastructure_error_markers = (
    "Cannot find configuration for task",
    "Could not find configuration",
    "is not a valid target",
)


def infrastructure_error_reason(output: str) -> str | None:
    # A missing/misnamed nx target is deterministic - the local model can't
    # fix it by editing source, and retrying just repeats the same failure.
    # Fail fast instead of burning attempts and GPU time on it.
    for marker in infrastructure_error_markers:
        if marker in output:
            return marker
    return None


def build_fix_prompt(task_name: str, attempt: int, condensed_output: str) -> str:
    return (
        f"The command for task '{task_name}' failed on attempt {attempt} of "
        f"{max_attempts}. Read the failing file(s) named in this output with "
        f"read_file, fix the problem with write_file, then call report_fix "
        f"with a one-line summary of what you changed.\n\n{condensed_output}"
    )


fix_attempt_timeout_seconds = 120


def request_fix(prompt: str) -> None:
    if fix_summary_path.exists():
        fix_summary_path.unlink()
    litert_command = [
        "litert-lm",
        "run",
        str(model_path),
        "--backend=gpu",
        f"--preset={preset_path}",
        f"--prompt={prompt}",
    ]
    try:
        result = subprocess.run(
            litert_command,
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=fix_attempt_timeout_seconds,
        )
        append_log(f"--- litert-lm fix attempt ---\n{result.stdout}\n{result.stderr}")
    except subprocess.TimeoutExpired as timeout_error:
        # litert-lm has no internal cap on tool-call retries — a bad path or
        # a confused model can loop inside one call indefinitely. Kill it
        # and let the outer attempt loop move on instead of hanging forever.
        append_log(
            f"--- litert-lm fix attempt timed out after "
            f"{fix_attempt_timeout_seconds}s ---\n{timeout_error.stdout}"
        )


def read_touched_files() -> list[str]:
    if not touched_files_path.exists():
        return []
    lines = touched_files_path.read_text(encoding="utf-8").splitlines()
    return sorted({line.strip() for line in lines if line.strip()})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a lint/test/build task, delegating fixes to the local model."
    )
    parser.add_argument("--task", required=True, choices=sorted(tasks))
    args = parser.parse_args()

    command = tasks[args.task]
    touched_files_path.write_text("", encoding="utf-8")
    fix_summaries: list[str] = []

    for attempt in range(1, max_attempts + 1):
        append_log(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"attempt {attempt}: {command}"
        )
        result = run_task_command(command)
        append_log(result.stdout)
        append_log(result.stderr)

        if result.returncode == 0:
            summary = "; ".join(fix_summaries) if fix_summaries else "no fix needed"
            print(f"RESULT: PASS - {summary}")
            return 0

        combined_output = strip_ansi(result.stdout + result.stderr)
        infra_reason = infrastructure_error_reason(combined_output)
        if infra_reason is not None:
            print(
                f"RESULT: FAILED - task '{args.task}' ({command}) is not a "
                f"valid nx target ({infra_reason}). Fix the task mapping in "
                f"run_ci_task.py, not the source code."
            )
            return 1

        if attempt == max_attempts:
            condensed = condense_failure(result)
            reason = next(
                (line for line in reversed(condensed.splitlines()) if line.strip()),
                "unknown failure",
            )
            print(f"RESULT: FAILED - {reason}")
            touched = read_touched_files()
            if touched:
                print(f"Files touched: {', '.join(touched)}")
            return 1

        prompt = build_fix_prompt(args.task, attempt, condense_failure(result))
        request_fix(prompt)
        if fix_summary_path.exists():
            fix_summaries.append(fix_summary_path.read_text(encoding="utf-8").strip())
            fix_summary_path.unlink()

    return 1


if __name__ == "__main__":
    sys.exit(main())
