#!/usr/bin/env python3
"""
train_pipeline.py — Auto-continue training pipeline (Pretrain → SFT → DPO)

Usage:
  # Full pipeline (auto-continue all stages):
  python3 scripts/train_pipeline.py --auto_continue --stages pretrain,sft,dpo [flags...]

  # Single stage only:
  python3 scripts/train_pipeline.py --stages sft --sft_lr 3e-5 --resume

  # Pretrain only (equivalent to running train_pretrain_cuda directly):
  python3 scripts/train_pipeline.py --stages pretrain --batch_size 24 --lr 1.4e-4

Environment auto-detection:
  - Kaggle:  /kaggle/working present → uses Kaggle paths, DDP x2
  - Colab:   /content present → uses Colab paths, single GPU
  - Local:   falls back to cwd
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def detect_environment():
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    elif os.path.exists("/content"):
        return "colab"
    return "local"


def find_work_dir(env):
    if env == "kaggle":
        return "/kaggle/working/transformer_100m_cuda"
    elif env == "colab":
        return "/content/transformer_100m_cuda"
    return str(Path(__file__).resolve().parent.parent)


def checkpoint_exists(ckpt_dir, pattern):
    return any(Path(ckpt_dir).glob(pattern))


def run_stage(cmd, cwd, env_vars, stage_name):
    """Run a training stage, streaming output. Returns exit code."""
    print(f"\n{'='*60}")
    print(f"  [{stage_name}] Starting...")
    print(f"  CMD: {' '.join(cmd[:6])}...")
    print(f"{'='*60}\n")

    proc_env = os.environ.copy()
    proc_env.update(env_vars)
    proc_env.pop("CUDA_VISIBLE_DEVICES", None)

    proc = subprocess.Popen(
        ["stdbuf", "-oL"] + cmd,
        cwd=cwd,
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    try:
        for raw in iter(proc.stdout.readline, b""):
            print(raw.decode("utf-8", errors="replace"), end="", flush=True)
    except KeyboardInterrupt:
        proc.terminate()
        print(f"\n[{stage_name}] Interrupted by user (Ctrl+C).")
        proc.wait()
        return -1

    proc.wait()
    rc = proc.returncode

    if rc == 0:
        print(f"\n[{stage_name}] Completed successfully.")
    elif rc == 2:
        print(f"\n[{stage_name}] CUDA OOM — reduce batch size and retry.")
    else:
        print(f"\n[{stage_name}] Failed with exit code {rc}.")

    return rc


def build_pretrain_cmd(args, env):
    cmd = [
        "./build/train_pretrain_cuda",
        "--data_dir", args.data_dir,
        "--vocab_dir", args.vocab_dir,
        "--checkpoint_dir", args.checkpoint_dir,
        "--batch_size", str(args.batch_size),
        "--seq", str(args.seq),
        "--max_tokens", str(args.max_tokens),
        "--lr", str(args.lr),
        "--save_every", str(args.save_every),
        "--log_every", str(args.log_every),
        "--grad_clip", str(args.grad_clip),
        "--loss_scale", str(args.loss_scale),
        "--safety", args.safety,
    ]
    if args.arch_config:
        cmd += ["--arch_config", args.arch_config]
    if args.bpe:
        cmd += ["--bpe", args.bpe]
    if args.upload_every > 0:
        cmd += ["--upload_every", str(args.upload_every)]
    if args.upload_script:
        cmd += ["--upload_script", args.upload_script]
    if args.eval_every_tokens > 0:
        cmd += ["--eval_every_tokens", str(args.eval_every_tokens)]
    if args.eval_infer_bin:
        cmd += ["--eval_infer_bin", args.eval_infer_bin]
    if args.resume or checkpoint_exists(args.checkpoint_dir, "train_state.txt"):
        cmd += ["--resume"]
    if args.ddp:
        cmd += ["--ddp", "--num_gpus", str(args.num_gpus)]
    return cmd


def build_sft_cmd(args, env):
    base_model = args.sft_base_model
    if not base_model:
        # Auto-detect: use pretrain final checkpoint
        final = Path(args.checkpoint_dir) / "ckpt_pretrain_final.safetensors"
        latest = Path(args.checkpoint_dir) / "ckpt_pretrain.safetensors"
        if final.exists():
            base_model = str(final)
        elif latest.exists():
            base_model = str(latest)
        else:
            print("[SFT] ERROR: No pretrained model found. Run pretrain first.")
            return None

    cmd = [
        "./build/train_sft_cuda",
        "--stage", "2",
        "--sft_dir", args.sft_dir,
        "--vocab_dir", args.vocab_dir,
        "--checkpoint_dir", args.checkpoint_dir,
        "--base_model", base_model,
        "--batch_size", str(args.sft_batch_size),
        "--max_seq", str(args.sft_max_seq),
        "--epochs", str(args.sft_epochs),
        "--lr", str(args.sft_lr),
        "--warmup", str(args.sft_warmup),
        "--grad_clip", str(args.sft_grad_clip),
        "--save_every", str(args.sft_save_every),
        "--log_every", str(args.log_every),
    ]
    if args.arch_config:
        cmd += ["--arch_config", args.arch_config]
    if args.bpe:
        cmd += ["--bpe", args.bpe]
    if args.sft_max_steps > 0:
        cmd += ["--max_steps", str(args.sft_max_steps)]
    if args.sft_upload_every > 0:
        cmd += ["--sft_upload_every", str(args.sft_upload_every)]
    elif args.upload_every > 0:
        cmd += ["--upload_every", str(args.upload_every)]
    if args.upload_script:
        cmd += ["--upload_script", args.upload_script]
    if args.resume or checkpoint_exists(args.checkpoint_dir, "train_state_sft.txt"):
        cmd += ["--resume"]
    if args.ddp:
        cmd += ["--ddp", "--num_gpus", str(args.num_gpus)]
    return cmd


def build_dpo_cmd(args, env):
    base_model = args.dpo_base_model
    if not base_model:
        # Auto-detect: use SFT final checkpoint
        sft_final = Path(args.checkpoint_dir) / "ckpt_sft_final.safetensors"
        sft_latest = Path(args.checkpoint_dir) / "ckpt_sft_step*.safetensors"
        if sft_final.exists():
            base_model = str(sft_final)
        else:
            # Find latest SFT checkpoint
            sft_ckpts = sorted(Path(args.checkpoint_dir).glob("ckpt_sft_step*.safetensors"))
            if sft_ckpts:
                base_model = str(sft_ckpts[-1])
            else:
                print("[DPO] ERROR: No SFT model found. Run SFT first.")
                return None

    ref_model = args.dpo_ref_model
    if not ref_model:
        # Standard DPO: reference = same model as policy init (SFT checkpoint)
        ref_model = base_model

    cmd = [
        "./build/train_sft_cuda",
        "--stage", "3",
        "--dpo_dir", args.dpo_dir,
        "--vocab_dir", args.vocab_dir,
        "--checkpoint_dir", args.checkpoint_dir,
        "--base_model", base_model,
        "--batch_size", str(args.dpo_batch_size),
        "--max_seq", str(args.dpo_max_seq),
        "--epochs", str(args.dpo_epochs),
        "--lr", str(args.dpo_lr),
        "--dpo_beta", str(args.dpo_beta),
        "--dpo_max_margin", str(args.dpo_max_margin),
        "--warmup", str(args.dpo_warmup),
        "--grad_clip", str(args.dpo_grad_clip),
        "--save_every", str(args.dpo_save_every),
        "--log_every", str(args.log_every),
    ]
    if ref_model:
        cmd += ["--ref_model", ref_model]
    if args.arch_config:
        cmd += ["--arch_config", args.arch_config]
    if args.bpe:
        cmd += ["--bpe", args.bpe]
    if args.dpo_max_steps > 0:
        cmd += ["--max_steps", str(args.dpo_max_steps)]
    if args.dpo_upload_every > 0:
        cmd += ["--dpo_upload_every", str(args.dpo_upload_every)]
    elif args.upload_every > 0:
        cmd += ["--upload_every", str(args.upload_every)]
    if args.upload_script:
        cmd += ["--upload_script", args.upload_script]
    if args.resume or checkpoint_exists(args.checkpoint_dir, "train_state_dpo.txt"):
        cmd += ["--resume"]
    if args.ddp:
        cmd += ["--ddp", "--num_gpus", str(args.num_gpus)]
    return cmd


def stage_already_done(ckpt_dir, stage):
    """Check if a stage's final checkpoint already exists."""
    patterns = {
        "pretrain": "ckpt_pretrain_final.safetensors",
        "sft": "ckpt_sft_final.safetensors",
        "dpo": "ckpt_dpo_final.safetensors",
    }
    p = patterns.get(stage)
    return p and (Path(ckpt_dir) / p).exists()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-continue training pipeline (Pretrain → SFT → DPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ─── Pipeline control ───────────────────────────────────────────────
    parser.add_argument("--auto_continue", action="store_true",
                        help="Run all stages in sequence automatically")
    parser.add_argument("--stages", default="pretrain,sft,dpo",
                        help="Comma-separated stages to run [pretrain,sft,dpo]")
    parser.add_argument("--skip_done", action="store_true", default=True,
                        help="Skip stages that have *_final.safetensors (default: True)")
    parser.add_argument("--no_skip_done", action="store_true",
                        help="Do NOT skip stages that already have *_final.safetensors")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each stage from its checkpoint")

    # ─── Common flags ───────────────────────────────────────────────────
    parser.add_argument("--work_dir", default="",
                        help="Working directory (auto-detected if empty)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--vocab_dir", default="vocab_9")
    parser.add_argument("--bpe", default="bpe_v9.bin")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--upload_every", type=int, default=0)
    parser.add_argument("--upload_script", default="")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=2)

    # ─── Pretrain flags ─────────────────────────────────────────────────
    parser.add_argument("--data_dir", default="data/corpus_pretrain_v8")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--lr", type=float, default=1.4e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--loss_scale", type=float, default=4096)
    parser.add_argument("--safety", default="light")
    parser.add_argument("--save_every", type=int, default=150)
    parser.add_argument("--eval_every_tokens", type=int, default=200_000_000)
    parser.add_argument("--eval_infer_bin", default="./build_infer/infer")
    parser.add_argument("--arch_config", default="",
                        help="architecture JSON (e.g. config_492m.json); "
                             "empty keeps the trainer's built-in default")

    # ─── SFT flags ──────────────────────────────────────────────────────
    parser.add_argument("--sft_dir", default="sft_data/sft_translation_focused")
    parser.add_argument("--sft_lr", type=float, default=3e-5)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--sft_batch_size", type=int, default=8)
    parser.add_argument("--sft_max_seq", type=int, default=512)
    parser.add_argument("--sft_warmup", type=int, default=100)
    parser.add_argument("--sft_grad_clip", type=float, default=1.0)
    parser.add_argument("--sft_save_every", type=int, default=200)
    parser.add_argument("--sft_upload_every", type=int, default=0,
                        help="Upload SFT checkpoint every N steps [0=use --upload_every]")
    parser.add_argument("--sft_max_steps", type=int, default=-1)
    parser.add_argument("--sft_base_model", default="",
                        help="Override base model for SFT (default: auto from pretrain)")

    # ─── DPO flags ──────────────────────────────────────────────────────
    parser.add_argument("--dpo_dir", default="sft_data/dpo")
    parser.add_argument("--dpo_lr", type=float, default=1e-5)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_epochs", type=int, default=3)
    parser.add_argument("--dpo_batch_size", type=int, default=4)
    parser.add_argument("--dpo_max_seq", type=int, default=512)
    parser.add_argument("--dpo_warmup", type=int, default=50)
    parser.add_argument("--dpo_grad_clip", type=float, default=1.0)
    parser.add_argument("--dpo_max_margin", type=float, default=5.0,
                        help="Clamp raw margin to [-M, M] to prevent over-optimization")
    parser.add_argument("--dpo_save_every", type=int, default=200)
    parser.add_argument("--dpo_upload_every", type=int, default=0,
                        help="Upload DPO checkpoint every N steps [0=use --upload_every]")
    parser.add_argument("--dpo_max_steps", type=int, default=-1)
    parser.add_argument("--dpo_base_model", default="",
                        help="Override base model for DPO (default: auto from SFT)")
    parser.add_argument("--dpo_ref_model", default="",
                        help="Override reference model for DPO (default: pretrain final)")

    args = parser.parse_args()

    # ─── Environment detection ──────────────────────────────────────────
    env = detect_environment()
    work_dir = args.work_dir or find_work_dir(env)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  Training Pipeline — {env.upper():8s}                         ║")
    print(f"║  Work dir: {work_dir:<44s}║")
    print(f"║  Stages:   {args.stages:<44s}║")
    print(f"╚══════════════════════════════════════════════════════════╝")

    # Resolve checkpoint_dir relative to work_dir
    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.join(work_dir, args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Kaggle-specific: adjust defaults
    if env == "kaggle":
        if args.checkpoint_dir == os.path.join(work_dir, "checkpoints"):
            args.checkpoint_dir = "/kaggle/working/checkpoints"
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if not args.ddp:
            args.ddp = True
            args.num_gpus = 2
        if args.batch_size == 24:
            args.batch_size = 35

    env_vars = {}
    if env == "kaggle":
        env_vars["NCCL_P2P_DISABLE"] = "1"

    stages = [s.strip() for s in args.stages.split(",")]
    results = {}
    t_pipeline_start = time.time()

    for stage in stages:
        skip = args.skip_done and not args.no_skip_done
        if skip and stage_already_done(args.checkpoint_dir, stage):
            print(f"\n[SKIP] {stage} already has *_final checkpoint — skipping.")
            results[stage] = 0
            continue

        if stage == "pretrain":
            cmd = build_pretrain_cmd(args, env)
        elif stage == "sft":
            cmd = build_sft_cmd(args, env)
            if cmd is None:
                results[stage] = 1
                break
        elif stage == "dpo":
            cmd = build_dpo_cmd(args, env)
            if cmd is None:
                results[stage] = 1
                break
        else:
            print(f"[ERROR] Unknown stage: {stage}")
            results[stage] = 1
            break

        rc = run_stage(cmd, work_dir, env_vars, stage.upper())
        results[stage] = rc

        if rc != 0:
            if not args.auto_continue:
                break
            print(f"\n[PIPELINE] Stage '{stage}' failed (rc={rc}). Stopping pipeline.")
            break

    # ─── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_pipeline_start
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY ({hours}h{mins:02d}m)")
    print(f"{'='*60}")
    for stage, rc in results.items():
        status = "✓ DONE" if rc == 0 else f"✗ FAIL (rc={rc})"
        print(f"  {stage:12s} {status}")
    print(f"{'='*60}")

    # Rank by failure, not by numeric value: a signal-killed child returns a
    # NEGATIVE code (-9 OOM-kill, -11 SIGSEGV), so max() over {skipped: 0,
    # killed: -9} yields 0 and the caller sees the dead run as a success.
    sys.exit(1 if any(rc != 0 for rc in results.values()) else 0)


if __name__ == "__main__":
    main()
