#!/usr/bin/env python3
r"""Add evaluated reasoning spans (Option B1) to coding and math SFT rows.

Upgrades over v1:
  1. Normalizes CRLF -> LF across all sources (fixes MBPP token 29 split).
  2. Ensures ALL coding answers use ```python ... ``` markdown fences consistently.
  3. Annotates every Python PoT math line with its exact evaluated numeric result
     (`# = val`) by executing the snippet in a restricted namespace, AND imports
     deduplicated EN/JA step-by-step word problems from sft_math_v2 wrapped in
     <|think|>...<|/think|> so Japanese math covers 20 problem families instead of 8.
  4. Reserves math eval rows preferentially from items NOT leaked in
     data/corpus_pretrain_v8/math/sft_think_math_00.txt.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import random
import re
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

THINK, THINK_END = "<|think|>", "<|/think|>"

PLAN_EN = [
    "Read the input, then {step}.",
    "The task: {step}. Read input first.",
    "Plan: parse the input, {step}, then output the result.",
]
PLAN_JA = [
    "入力を読み、{step}。",
    "手順：入力を解析し、{step}、結果を出力する。",
]

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def fmt_val(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)) and len(v) <= 8:
        return repr(v)
    return None


def unfence(text):
    m = FENCE_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def ensure_python_fence(ans):
    ans = ans.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "```" in ans:
        return ans
    return f"```python\n{ans}\n```"


def turn_parts(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.match(r"<\|system\|>(.*?)<\|user\|>(.*?)<\|assistant\|>(.*)", text, re.DOTALL)
    if not m:
        return None
    sys_, user, ans = m.groups()
    for stop in ("<|end|>", "<|endoftext|>"):
        i = ans.find(stop)
        if i >= 0:
            ans = ans[:i]
    return sys_.strip(), user.strip(), ans.strip()


def describe_code(code, lang):
    try:
        tree = ast.parse(unfence(code))
    except SyntaxError:
        return None
    reads = len([n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "input"])
    loops = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))])
    conds = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    bits = []
    if reads:
        bits.append(f"read {reads} input value{'s' if reads > 1 else ''}"
                    if lang == "en" else f"入力を{reads}個読み取り")
    if funcs:
        bits.append(f"define {funcs[0]}()" if lang == "en"
                    else f"{funcs[0]}()を定義し")
    if loops:
        bits.append("loop over the values" if lang == "en" else "値をループ処理し")
    if conds:
        bits.append("branch on a condition" if lang == "en" else "条件で分岐し")
    if not bits:
        bits.append("compute the result directly" if lang == "en" else "直接計算する")
    return (", ".join(bits) if lang == "en" else "".join(bits))


def trace_for_code(user, ans, lang, rng):
    step = describe_code(ans, lang)
    if not step:
        return None
    tmpl = rng.choice(PLAN_EN if lang == "en" else PLAN_JA)
    return tmpl.format(step=step)


def trace_for_assert(user, tests, lang):
    if not tests:
        return None
    first = next((l.strip() for l in tests.replace("\r\n", "\n").split("\n")
                  if l.strip().startswith("assert")), None)
    if not first:
        return None
    return f"満たすべき条件：{first}" if lang == "ja" else f"The solution must satisfy: {first}"


def annotate_pot_code(code):
    """Execute short PoT Python line-by-line and append `# = <val>` comments (Option B1)."""
    lines = [l.rstrip() for l in code.strip().split("\n") if l.strip()]
    if not lines or len(lines) > 14:
        return None
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "round": round, "int": int, "float": float, "range": range,
        "sorted": sorted, "enumerate": enumerate, "print": print,
    }
    env = {"__builtins__": safe_builtins, "math": math}
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None

    out_lines = []
    src_lines = "\n".join(lines).splitlines()
    if all(getattr(stmt, "lineno", 0) == getattr(stmt, "end_lineno", -1) for stmt in tree.body):
        for stmt in tree.body:
            line_txt = src_lines[stmt.lineno - 1]
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    exec(compile(ast.Module(body=[stmt], type_ignores=[]), "<pot>", "exec"), env, env)
            except Exception:
                return None
            printed = buf.getvalue().strip()
            if printed:
                out_lines.append(f"{line_txt}  # = {printed}")
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id in env:
                    fv = fmt_val(env[tgt.id])
                    out_lines.append(f"{line_txt}  # = {fv}" if fv is not None else line_txt)
                elif isinstance(tgt, ast.Tuple):
                    vals = [fmt_val(env.get(el.id)) for el in tgt.elts if isinstance(el, ast.Name)]
                    if vals and all(v is not None for v in vals):
                        out_lines.append(f"{line_txt}  # = ({', '.join(vals)})")
                    else:
                        out_lines.append(line_txt)
                else:
                    out_lines.append(line_txt)
            else:
                out_lines.append(line_txt)
        return "\n".join(out_lines)
    else:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(tree, "<pot>", "exec"), env, env)
        except Exception:
            return None
        printed = buf.getvalue().strip()
        if not printed:
            return None
        if lines[-1].lstrip().startswith("print("):
            lines[-1] = f"{lines[-1]}  # = {printed}"
        else:
            lines.append(f"# result = {printed}")
        return "\n".join(lines)


def split_math_pot(ans):
    m = FENCE_RE.search(ans)
    if not m:
        return None, None
    body = m.group(1).strip()
    annotated = annotate_pot_code(body)
    if not annotated:
        return None, None
    conclusion = (ans[:m.start()] + ans[m.end():]).strip()
    if not conclusion:
        return None, None
    return annotated, conclusion


def split_math_v2(ans):
    """Split step-by-step arithmetic (`sft_math_v2`) into (<|think|> steps, final answer)."""
    lines = [l.strip() for l in ans.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None, None
    last = lines[-1]
    if not any(k in last.lower() or k in last for k in ("answer is", "答えは", "したがって", "therefore")):
        return None, None
    steps = "\n".join(lines[:-1]).strip()
    if not steps or not re.search(r"\d", steps):
        return None, None
    return steps, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_root", default="sft_data")
    ap.add_argument("--out_code", default="sft_data/sft_think_code")
    ap.add_argument("--out_math", default="sft_data/sft_think_math")
    ap.add_argument("--sources", default="sft_coding_v3,sft_coding_mbpp,"
                                         "sft_code_bugfix,sft_math_pot,sft_math_v2")
    ap.add_argument("--math_v2_cap_per_lang", type=int, default=8000,
                    help="cap deduplicated sft_math_v2 rows per language")
    ap.add_argument("--langs", default="en,ja")
    ap.add_argument("--eval_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    keep = {x.strip() for x in a.langs.split(",") if x.strip()}
    root = Path(a.sft_root)

    pt_leak_file = Path("data/corpus_pretrain_v8/math/sft_think_math_00.txt")
    pt_blob = pt_leak_file.read_text(encoding="utf-8", errors="ignore") if pt_leak_file.exists() else ""

    rows, stats, seen, seen_q = [], Counter(), set(), set()
    v2_counts = Counter()

    for name in [s.strip() for s in a.sources.split(",") if s.strip()]:
        f = root / name / "train.jsonl"
        if not f.exists():
            stats[f"missing:{name}"] += 1
            continue
        raw_lines = f.read_text(encoding="utf-8").splitlines()
        if name == "sft_math_v2":
            rng.shuffle(raw_lines)
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lang = str(rec.get("lang", "")).lower()
            if lang not in keep:
                stats["drop_lang"] += 1
                continue
            if name == "sft_math_v2" and v2_counts[lang] >= a.math_v2_cap_per_lang:
                continue
            parts = turn_parts(rec.get("text", ""))
            if not parts:
                stats["unparsable"] += 1
                continue
            sys_, user, ans = parts
            if not ans:
                stats["empty_answer"] += 1
                continue

            qh = hashlib.sha1(user.encode("utf-8")).digest()
            if qh in seen_q:
                stats["dup_question"] += 1
                continue

            is_math = "math" in name
            if not is_math:
                if rec.get("tests"):
                    trace = trace_for_assert(user, rec["tests"], lang)
                else:
                    trace = trace_for_code(user, ans, lang, rng)
                ans = ensure_python_fence(ans)
            elif name == "sft_math_v2":
                trace, ans = split_math_v2(ans)
            else:
                trace, ans = split_math_pot(ans)

            if not trace or not ans:
                stats["no_trace"] += 1
                continue

            text = (f"<|system|>{sys_}<|user|>{user}<|assistant|>"
                    f"{THINK}{trace}{THINK_END}{ans}<|end|>")
            h = hashlib.sha1(text.encode("utf-8")).digest()
            if h in seen:
                stats["dup"] += 1
                continue
            seen.add(h)
            seen_q.add(qh)
            if name == "sft_math_v2":
                v2_counts[lang] += 1

            out = dict(rec)
            out["text"] = text
            out["task"] = rec.get("task", "code")
            out["has_think"] = True
            out["source"] = name
            out["family"] = "math" if is_math else "coding"
            out["_in_pretrain"] = bool(is_math and pt_blob and user[:60] in pt_blob)
            rows.append(out)
            stats[f"kept:{name}"] += 1

    rng.shuffle(rows)
    for fam, out_path in (("coding", a.out_code), ("math", a.out_math)):
        part = [r for r in rows if r["family"] == fam]
        n_eval = max(100, int(len(part) * a.eval_frac))
        if fam == "math":
            clean_ev_pool = [r for r in part if not r.get("_in_pretrain")]
            leaked_pool = [r for r in part if r.get("_in_pretrain")]
            ev = clean_ev_pool[:n_eval]
            tr = clean_ev_pool[n_eval:] + leaked_pool
            rng.shuffle(tr)
        else:
            ev, tr = part[:n_eval], part[n_eval:]
        for r in part:
            r.pop("_in_pretrain", None)

        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, chunk in (("train", tr), ("eval", ev)):
            with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out_dir / "stats.json").write_text(
            json.dumps({"train": len(tr), "eval": len(ev),
                        "lang": dict(Counter(r["lang"] for r in tr).most_common()),
                        "sources": dict(Counter(r["source"] for r in tr).most_common())},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[THINK-B1] {fam:7s} train={len(tr):6d} eval={len(ev):4d} -> {out_dir}")

    for k, v in stats.most_common(12):
        print(f"    {k:26s} {v}")


if __name__ == "__main__":
    main()
