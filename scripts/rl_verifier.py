#!/usr/bin/env python3
"""
rl_verifier.py — Rule-based reward for RL coding.

Reads a request JSONL (one line per sampled completion):
    {"id": <int>, "lang": "en"|"ja", "code": "<generated text>",
     "input": "<stdin content>", "expected": "<expected stdout>"}

For each entry it extracts a Python code block from `code`, runs it with
`input` fed to stdin, captures stdout, and emits a reward:
    reward = 1.0 if stripped stdout == stripped expected, else 0.0
Partial credit is not used in this first version.

Usage:
    python3 scripts/rl_verifier.py req.jsonl reward.jsonl
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile


def extract_code(text):
    """Pull Python code out of a model completion (markdown-safe)."""
    if not text:
        return ""
    # Numbered output lines like "```python" fenced blocks.
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fences:
        return fences[-1].strip()
    # Fall back to the whole text (trim leading/trailing noise).
    return text.strip()


def run_code(code, stdin_data, timeout=5.0):
    """Run `code` as a Python program with stdin_data; return (rc, stdout, stderr)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    finally:
        try:
            import os
            os.unlink(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("req")
    ap.add_argument("out")
    args = ap.parse_args()

    results = []
    with open(args.req, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["id"]
            code = extract_code(rec.get("code", ""))
            stdin_data = rec.get("input", "") or ""
            expected = (rec.get("expected", "") or "").strip()
            reward = 0.0
            if code:
                rc, stdout, _ = run_code(code, stdin_data)
                if rc == 0 and (stdout or "").strip() == expected:
                    reward = 1.0
            results.append({"id": sid, "reward": reward})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    ok = sum(1 for r in results if r["reward"] > 0.0)
    print(f"[rl-verifier] {ok}/{n} passed ({100.0 * ok / max(1, n):.1f}%)", flush=True)


if __name__ == "__main__":
    main()