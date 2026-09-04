#!/usr/bin/env python3
"""
rl_verifier.py (v3) — Unified rule-based reward for RL.

Handles three completion kinds in one file (selected per request row):
  ASSERT rows (non-empty `tests`): append the assert block to the completion and
      run it. Exit code 0 means every assertion held. This is the MBPP /
      HumanEval contract, where correctness is a function signature plus test
      cases rather than stdout text.
  CODE rows (non-empty `input`): extract a Python block, run it against the
      row's stdin/expected. If it passes AND a task generator can be matched,
      also run 3 EXTRA freshly generated instances of the same task
      (anti lucky-pass test bank) — all must pass.
  MATH rows (both empty): the completion is chain-of-thought text ending
      with an anchor line ("Therefore, the answer is X." /
      "Jadi, jawabannya adalah X." / "したがって、答えはXです。").
      Reward = anchor answer matches expected (whitespace-normalised;
      thousand-separator tolerant).

Reads a request JSONL (one line per sampled completion):
    {"id": <int>, "lang": "en"|"ja", "code": "<generated text>",
     "input": "<stdin content or empty>", "expected": "<expected>",
     "tests": "<assert block or empty>", "setup": "<optional preamble>"}

Writes reward JSONL: {"id": <int>, "reward": 0.0|1.0}

NOTE: completions are executed. Runs are sandboxed only by a wall-clock timeout,
so this must not be pointed at untrusted input outside a disposable environment.

Test-bank matching: the (input, expected) pair is looked up in
rl_data/coding.jsonl to recover the task name, then extra instances come from
scripts/build_sft_coding_v2.py's TASK_INPUT. Missing files degrade gracefully
to single-case verification.

Usage:
    python3 scripts/rl_verifier.py req.jsonl reward.jsonl
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile

EXTRA_CASES = 3  # additional fresh instances for matched code tasks


def extract_code(text):
    """Pull Python code out of a model completion (markdown-safe)."""
    if not text:
        return ""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fences:
        return fences[-1].strip()
    return text.strip()


def run_code(code, stdin_data, timeout=5.0):
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
            os.unlink(path)
        except OSError:
            pass


# ── math anchor extraction ──────────────────────────────────────────────────
ANCHOR_RE = re.compile(
    r"(?:the answer is|jawabannya adalah|答えは)\s*(.+?)\s*(?:[.。]|$)",
    re.IGNORECASE,
)


def norm_answer(s):
    """Normalise an answer string for comparison: strip spaces/commas/case."""
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(", ", ",").replace(" ,", ",")
    return s.casefold()


def norm_number(s):
    """Number-tolerant normalisation: '12,000' == '12000' == '12000'."""
    t = norm_answer(s).replace(",", "").replace(" ", "")
    return t


def math_reward(text, expected):
    """Extract the anchor-line answer from CoT text; compare to expected."""
    if not text:
        return False
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    # prefer the LAST anchor occurrence
    m = None
    for line in lines:
        mm = ANCHOR_RE.search(line)
        if mm:
            m = mm
    if m is None:
        return False
    got = m.group(1).strip().rstrip("です").rstrip(".").strip()
    exp = expected.strip()
    if norm_number(got) == norm_number(exp):
        return True
    # list-style answers (e.g. sorted numbers): whitespace-insensitive compare
    if norm_answer(got) == norm_answer(exp):
        return True
    return False


# ── code test bank ──────────────────────────────────────────────────────────
def load_test_bank():
    """(input,expected) -> task name; plus TASK_INPUT from the v2 builder."""
    bank = {}
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for cand in (os.path.join(root, "rl_data", "coding.jsonl"),
                 os.path.join(root, "rl_data", "coding_v2.jsonl"),
                 os.path.join(root, "rl_data", "coding_v3.jsonl")):
        if os.path.exists(cand):
            try:
                for line in open(cand, encoding="utf-8"):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    inp = (r.get("mock_input") or "").strip()
                    exp = (r.get("expected_output") or "").strip()
                    task = r.get("task", "")
                    if inp and exp and task:
                        bank[(inp, exp)] = task
            except OSError:
                pass
    gen = {}
    if bank:
        try:
            sys.path.insert(0, here)
            import build_sft_coding_v3 as B3
            gen = dict(getattr(B3.V2, "TASK_INPUT", {}))
            for k, v in getattr(B3, "NEW_INPUT", {}).items():
                gen[k] = v
        except Exception:
            try:
                import build_sft_coding_v2 as B
                gen = getattr(B, "TASK_INPUT", {})
            except Exception:
                gen = {}
    return bank, gen


def code_reward(code, stdin_data, expected, bank, gen, rng):
    if not code:
        return False
    rc, stdout, _ = run_code(code, stdin_data)
    if rc != 0 or (stdout or "").strip() != expected:
        return False
    # primary case passed — try the multi-case bank
    task = bank.get((stdin_data.strip(), expected))
    if task and task in gen:
        for _ in range(EXTRA_CASES):
            extra_in, extra_exp = gen[task](rng)
            rc, stdout, _ = run_code(code, extra_in)
            if rc != 0 or (stdout or "").strip() != extra_exp.strip():
                return False
    return True


def assert_reward(code, tests, setup="", timeout=10.0):
    """MBPP/HumanEval contract: solution + assert block must exit cleanly.

    Model output often repeats the prompt's import lines or docstring, which is
    harmless, but a bare markdown fence is not, so reuse extract_code().
    """
    if not code or not tests:
        return False
    program = "\n\n".join(p for p in (setup, code, tests) if p and p.strip())
    rc, _, _ = run_code(program, "", timeout=timeout)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("req")
    ap.add_argument("out")
    args = ap.parse_args()

    bank, gen = load_test_bank()
    if bank:
        print(f"[rl-verifier] test bank: {len(bank)} rows, "
              f"{len(gen)} task generators", flush=True)

    rng = random.Random(1234)
    results = []
    n_code = n_math = n_assert = 0
    with open(args.req, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["id"]
            text = rec.get("code", "")
            stdin_data = rec.get("input", "") or ""
            expected = (rec.get("expected", "") or "").strip()
            tests = rec.get("tests", "") or ""

            if tests.strip():
                reward = 1.0 if assert_reward(extract_code(text), tests,
                                              rec.get("setup", "")) else 0.0
                n_assert += 1
            elif stdin_data == "":
                # math / CoT row
                reward = 1.0 if math_reward(text, expected) else 0.0
                n_math += 1
            else:
                code = extract_code(text)
                reward = 1.0 if code_reward(code, stdin_data, expected,
                                             bank, gen, rng) else 0.0
                n_code += 1
            results.append({"id": sid, "reward": reward})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    ok = sum(1 for r in results if r["reward"] > 0.0)
    print(f"[rl-verifier] {ok}/{n} passed ({100.0 * ok / max(1, n):.1f}%) "
          f"[code: {n_code}, math: {n_math}, assert: {n_assert}]", flush=True)


if __name__ == "__main__":
    main()
