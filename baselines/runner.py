import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import pexpect
from datetime import datetime
from pathlib import Path
import pyte


FIXED_PROMPT = "Read instructions.txt and follow it exactly."

DEFAULT_TIMEOUT_MINUTES = 60

# How long (seconds) to wait for Claude's output to stop arriving before we
# consider it "settled" and ready for input.  Increase if Claude's startup
# splash takes longer on your machine.
SETTLE_TIMEOUT = 10
DONE_SILENCE_SEC = 120

# After we send the prompt, we poll for silence again — but Claude may think
# for a while, so use a much longer per-poll window.
RESPONSE_POLL_INTERVAL = 10

# What we wait for to know Claude is ready for input.
# Adjust this pattern if your Claude version shows a different prompt.
CLAUDE_READY_PATTERN = r"(?i)(>\s*$|\$\s*$|Human:|assistant>|claude>|\?$)"

# Sentinel printed by Claude when it has finished a response and is idle.
CLAUDE_DONE_PATTERN = CLAUDE_READY_PATTERN

# Pattern to extract Claude's session ID from the startup banner.
SESSION_ID_PATTERNS = [
    r"--resume\s+([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
    r"[Ss]ession(?:\s+ID)?[:\s]+([a-f0-9\-]{36})",
]

# Virtual terminal dimensions used by pyte for rendering.
# Wide columns reduce unwanted line-wrapping in Claude's TUI output.
TERM_COLS = 220
TERM_ROWS = 50
TERM_HISTORY = 10_000  # lines of scrollback kept by pyte


class BenchmarkRun:
    def __init__(self, competition_name: str, timeout_minutes: int):
        self.competition_name = competition_name
        self.timeout_minutes = timeout_minutes

        self.project_root = Path(__file__).parent.resolve()

        self.benchmark_dir = (
            self.project_root / "benchmarks" / competition_name
        )

        if not self.benchmark_dir.exists():
            raise FileNotFoundError(
                f"Benchmark not found: {self.benchmark_dir}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = (
            self.project_root
            / "runs"
            / f"{timestamp}_{competition_name}"
        )

        self.workspace_dir = self.run_dir / "workspace"
        self.logs_dir = self.run_dir / "logs"

        self.metadata_path = self.run_dir / "metadata.json"

    def setup(self):
        print("[1/5] Setting up workspace...")

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        shutil.copytree(
            self.benchmark_dir,
            self.workspace_dir,
            dirs_exist_ok=True,
        )

        print("[1.1/5] Creating virtual environment...")

        self.venv_dir = self.run_dir / "venv"



    def _build_claude_command(self) -> tuple[str, list[str]]:
        """Return (executable, [args]) for launching Claude inside the venv."""

        allowed_tools = [
            "Bash",
            "Bash(python3*)",
            "Bash(pip*)",
            "Bash(pip3*)",
            "Edit",
            "Write",
            "Read",
            "Glob",
            "Grep"
        ]

        # Build allowedTools string — kept as-is per original design.
        tools_str = " ".join([f'--allowedTools "{tool}"' for tool in allowed_tools])

        venv_bin = self.venv_dir / "bin"
        activate = str(venv_bin / "activate")

        # Stderr is redirected to a dedicated file so it can be read separately.
        # The PTY still captures stdout; stderr goes to logs/session_stderr.log.
        stderr_log = str(self.logs_dir / "session_stderr.log")

        cmd = (
            f'uv init && '
            f'uv sync && '
            f'exec claude --permission-mode dontAsk {tools_str} 2>"{stderr_log}"'
        )

        return "/bin/bash", ["-c", cmd]

    def run_claude(self) -> dict:
        print("[2/5] Launching Claude Code via pexpect...")

        raw_log_path    = self.logs_dir / "session_raw.log"
        clean_log_path  = self.logs_dir / "session_clean.log"
        stderr_log_path = self.logs_dir / "session_stderr.log"

        timeout_sec = self.timeout_minutes * 60

        executable, args = self._build_claude_command()

        # pexpect.spawn creates a real PTY — Claude sees a proper terminal.
        # Use dimensions=(TERM_ROWS, TERM_COLS) so Claude wraps at the same
        # width that pyte will later use to render the output.
        child = pexpect.spawn(
            executable,
            args=args,
            cwd=str(self.workspace_dir),
            encoding="utf-8",
            codec_errors="replace",
            timeout=timeout_sec,
            dimensions=(TERM_ROWS, TERM_COLS),
            env={**os.environ, "TERM": "xterm-256color",
                 "COLUMNS": str(TERM_COLS), "LINES": str(TERM_ROWS)},
        )

        # Mirror raw PTY output (escape sequences and all) to a log file.
        with open(raw_log_path, "w", encoding="utf-8") as raw_log:
            child.logfile_read = raw_log

            start_time = time.time()
            timed_out  = False
            session_id = None

            # Phase timings
            phase_times: dict[str, float] = {}

            def elapsed() -> float:
                return time.time() - start_time

            def time_left() -> float:
                return timeout_sec - elapsed()

            def wait_for_settle(poll_interval: float, label: str) -> bool:
                """
                Read output in chunks until no new data arrives for
                `poll_interval` seconds, or the global timeout is hit.
                Returns True if settled cleanly, False if timed out.
                """
                print(f"  {label}...")
                while time_left() > 0:
                    try:
                        child.expect(pexpect.TIMEOUT, timeout=poll_interval)
                        return True
                    except pexpect.TIMEOUT:
                        # No new output for poll_interval → Claude is idle.
                        return True
                    except pexpect.EOF:
                        return True  # process exited cleanly
                return False  # global timeout hit
                
                
            def wait_for_claude_done() -> bool:
                """
                Detect that Claude has finished its response.
 
                Claude Code is a TUI — it never emits a plain-text prompt we
                can pattern-match on.  Instead we use a two-phase approach:
 
                  1. Wait for Claude to start producing output (it began work).
                  2. Wait for DONE_SILENCE_SEC of consecutive silence — that
                     sustained quiet means Claude is back at its idle prompt.
 
                Claude can pause mid-task while thinking, so DONE_SILENCE_SEC
                must be long enough to survive those thinking pauses.
                """
                # Phase A — wait for Claude to start doing something.
                print("  Waiting for Claude to start working...")
                try:
                    child.expect(r".+", timeout=min(60.0, time_left()))
                except pexpect.TIMEOUT:
                    print("  [!] Claude never started producing output.")
                    return False
                except pexpect.EOF:
                    return True  # process exited = done
 
                # Phase B — wait for sustained silence after activity.
                print("  Claude is working. Waiting for it to finish...")
                while time_left() > 0:
                    silence_window = min(DONE_SILENCE_SEC, time_left())
                    try:
                        child.expect(r".+", timeout=silence_window)
                        # Got more output — Claude is still going, keep waiting.
                    except pexpect.TIMEOUT:
                        # No output for DONE_SILENCE_SEC → Claude is done.
                        return True
                    except pexpect.EOF:
                        return True  # process exited cleanly
 
                return False  # global timeout

            try:
                # ── Phase 0: handle the "trust this folder?" prompt ────────
                print("  Checking for trust prompt...")
                idx = child.expect(
                    [r"(?i)trust", pexpect.TIMEOUT],
                    timeout=30,
                )
                if idx == 0:
                    print("  Trust prompt detected — confirming...")
                    child.send("\r")
                    time.sleep(1)

                phase_times["trust_check_sec"] = round(elapsed(), 2)

                # ── Phase 1: wait for startup output to settle ─────────────
                child.expect(r".+", timeout=60)  # any character at all
                settled = wait_for_settle(SETTLE_TIMEOUT, "Waiting for Claude to be ready")
                if not settled:
                    raise pexpect.TIMEOUT("Global timeout during startup")

                phase_times["startup_sec"] = round(elapsed(), 2)

                print("  Claude is ready. sess...")

                # ── Phase 2: send the prompt ───────────────────────────────
                child.sendline("/status")
                child.send("\r")
                time.sleep(3)
                # Try to grab session ID from what's been buffered so far.
                session_id = _extract_session_id(child.before or "")
                child.delaybeforesend=0.5
                child.sendcontrol('[')

                print("  Claude is ready. Sending prompt...")

                # ── Phase 2: send the prompt ───────────────────────────────
                child.sendline(FIXED_PROMPT)
                child.send("\r")

                phase_times["prompt_sent_sec"] = round(elapsed(), 2)

                # ── Phase 3: wait for Claude to finish responding ──────────
                secs_left = time_left()
                if secs_left <= 0:
                    raise pexpect.TIMEOUT("No time left after sending prompt")

                done = wait_for_claude_done()
                if done:
                    print("  Claude finished (silence detected).")
                else:
                    print("  [!] Timed out waiting for Claude to finish.")
                    timed_out = True
                    child.close(force=True)
                    raise pexpect.TIMEOUT("Done-silence timeout")
 
                phase_times["response_sec"] = round(elapsed(), 2)
 
                # Give it a moment, then close cleanly.
                child.sendline("exit")
                child.send("\r")
                child.expect(pexpect.EOF, timeout=10)
 
            except pexpect.TIMEOUT:
                timed_out = True
                print("[!] Timeout reached. Killing Claude process...")
                child.close(force=True)
 
            except pexpect.EOF:
                pass  # process exited on its own — fine

        end_time = time.time()

        # ── Post-processing ────────────────────────────────────────────────
        raw_output = raw_log_path.read_text(encoding="utf-8", errors="replace")

        # Render through a virtual terminal to get clean, human-readable text.
        clean_output = _render_terminal_output(raw_output)
        clean_log_path.write_text(clean_output, encoding="utf-8")

        # Read stderr (may be empty).
        raw_stderr = ""
        if stderr_log_path.exists():
            raw_stderr = stderr_log_path.read_text(encoding="utf-8", errors="replace")

        # Fallback session ID scan over the full clean output.
        if not session_id:
            session_id = _extract_session_id(raw_output)

        exit_status   = child.exitstatus   if child.exitstatus   is not None else -1
        signal_status = child.signalstatus if child.signalstatus is not None else None

        return {
            "runtime_sec":    round(end_time - start_time, 2),
            "timed_out":      timed_out,
            "return_code":    exit_status,
            "signal":         signal_status,
            "session_id":     session_id,
            "phase_times":    phase_times,
            "raw_log_path":   str(raw_log_path),
            "clean_log_path": str(clean_log_path),
            "stderr_path":    str(stderr_log_path),
        }

    def check_submission(self) -> bool:
        print("[3/5] Checking submission...")

        submission_path = (
            self.workspace_dir / "submission" / "submission.csv"
        )

        return submission_path.exists()

    def save_metadata(self, run_info: dict, submission_found: bool):
        print("[4/5] Saving metadata...")

        metadata = {
            "competition":      self.competition_name,
            "timestamp":        datetime.now().isoformat(),
            "session_id":       run_info["session_id"],
            "runtime_sec":      run_info["runtime_sec"],
            "timed_out":        run_info["timed_out"],
            "return_code":      run_info["return_code"],
            "signal":           run_info["signal"],
            "phase_times":      run_info["phase_times"],
            "submission_found": submission_found,
            "logs": {
                "stdout_raw":   run_info["raw_log_path"],
                "stdout_clean": run_info["clean_log_path"],
                "stderr":       run_info["stderr_path"],
            },
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def print_summary(self, run_info: dict, submission_found: bool):
        print("\n===== RUN SUMMARY =====")
        print(f"Competition:      {self.competition_name}")
        print(f"Session ID:       {run_info['session_id'] or 'not found'}")
        print(f"Runtime:          {run_info['runtime_sec']} sec")
        print(f"Timed out:        {run_info['timed_out']}")
        print(f"Return code:      {run_info['return_code']}")
        if run_info["signal"] is not None:
            print(f"Killed by signal: {run_info['signal']}")
        print(f"Submission found: {submission_found}")
        print(f"\nPhase timings:")
        for phase, t in run_info["phase_times"].items():
            print(f"  {phase}: {t}s")
        print(f"\nRun directory:    {self.run_dir}")
        print(f"Logs:")
        print(f"  stdout (raw):   {run_info['raw_log_path']}")
        print(f"  stdout (clean): {run_info['clean_log_path']}")
        print(f"  stderr:         {run_info['stderr_path']}")

    def execute(self):
        self.setup()
        run_info = self.run_claude()
        submission_found = self.check_submission()
        self.save_metadata(run_info, submission_found)
        self.print_summary(run_info, submission_found)


# ── helpers ────────────────────────────────────────────────────────────────────

def _render_terminal_output(raw: str) -> str:
    """
    Feed raw PTY output (escape sequences and all) through a pyte virtual
    terminal and return the rendered plain text, including scrollback history.

    Claude Code uses a full TUI with cursor movement, line rewriting, and
    box-drawing — a simple regex strip leaves unreadable garbage.  pyte
    actually emulates the terminal and gives us what a human would see.
    """
    screen = pyte.HistoryScreen(TERM_COLS, TERM_ROWS, history=TERM_HISTORY)
    stream = pyte.Stream(screen)
    stream.feed(raw)

    lines: list[str] = []

    # Scrollback history (content that has already scrolled off screen).
    for history_line in screen.history.top:
        rendered = "".join(char.data for char in history_line.values())
        lines.append(rendered.rstrip())

    # Current visible screen buffer.
    for y in range(screen.lines):
        rendered = "".join(
            screen.buffer[y][x].data for x in range(screen.columns)
        )
        lines.append(rendered.rstrip())

    # Drop leading/trailing blank lines, collapse runs of 3+ blanks to 2.
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_session_id(text: str) -> str | None:
    for pattern in SESSION_ID_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


# ── ANSI-only fallback (used if pyte is somehow unavailable) ──────────────────

_ANSI_RE = re.compile(
    r"""
    \x1B
    (?:
        [@-Z\\-_]
    |
        \[ [0-?]* [ -/]* [@-~]
    |
        \] .*? (?:\x07|\x1b\\)
    )
    """,
    re.VERBOSE,
)


def _strip_ansi_fallback(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--competition",
        required=True,
        help="Competition folder name",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MINUTES,
        help="Timeout in minutes",
    )

    args = parser.parse_args()

    run = BenchmarkRun(
        competition_name=args.competition,
        timeout_minutes=args.timeout,
    )

    run.execute()


if __name__ == "__main__":
    main()
