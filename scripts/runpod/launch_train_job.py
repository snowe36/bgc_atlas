#!/usr/bin/env python3
"""Create a RunPod job, sync data, train the contrastive BGC encoder, pull results."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
REMOTE_DIR = "/workspace/bgc_atlas"
RSYNC_EXCLUDES = [
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data/raw",
    "data/external",
    "notebooks",
    "*.pyc",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
]

REQUIRED_PROCESSED = [
    "esm_protein_embeddings.npy",
    "esm_protein_meta.csv",
    "feature_meta.parquet",
    "feature_matrix.npy",
    "esm_embeddings.npy",
    "esm_bgc_ids.csv",
]


def _load_api_key() -> str:
    for name in ("RUNPOD_API_KEY", "API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key in ("RUNPOD_API_KEY", "API_KEY") and val:
                return val
    raise SystemExit("No RunPod API key found. Set RUNPOD_API_KEY (or API_KEY) in .env or the environment.")


def _ssh_opts(port: int) -> list[str]:
    opts = [
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
    ]
    for candidate in (
        Path.home() / ".ssh" / "runpod_ed25519",
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa",
    ):
        if candidate.exists():
            opts.extend(["-i", str(candidate), "-o", "IdentitiesOnly=yes"])
            break
    return opts


def wait_for_ssh(runpod_mod, pod_id: str, timeout_s: int) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        pod = runpod_mod.get_pod(pod_id)
        ports = (pod.get("runtime") or {}).get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("ip"):
                ip, port = p["ip"], int(p["publicPort"])
                probe = subprocess.run(
                    ["ssh", *_ssh_opts(port), f"root@{ip}", "echo ready"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if probe.returncode == 0:
                    return ip, port
                last_err = probe.stderr
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} SSH not reachable after {timeout_s}s (last error: {last_err})")


def rsync_up(ip: str, port: int) -> None:
    processed = ROOT / "data" / "processed"
    missing = [n for n in REQUIRED_PROCESSED if not (processed / n).exists()]
    soft = {"esm_embeddings.npy", "esm_bgc_ids.csv"}  # optional for train
    hard_missing = [n for n in missing if n not in soft]
    if hard_missing:
        raise SystemExit(
            f"Missing required processed files: {hard_missing}. "
            "Run bgc-download / bgc-featurize / scripts/run_esm_embed.py first."
        )

    subprocess.run(
        ["ssh", *_ssh_opts(port), f"root@{ip}", f"mkdir -p {REMOTE_DIR}/data/processed"],
        check=True,
    )
    excludes: list[str] = []
    for e in RSYNC_EXCLUDES:
        excludes += ["--exclude", e]
    excludes += ["--exclude", "data/processed"]
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            *excludes,
            "-e",
            f"ssh {' '.join(_ssh_opts(port))}",
            f"{ROOT}/",
            f"root@{ip}:{REMOTE_DIR}/",
        ],
        check=True,
    )
    for name in REQUIRED_PROCESSED:
        src = processed / name
        if not src.exists():
            continue
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh {' '.join(_ssh_opts(port))}",
                str(src),
                f"root@{ip}:{REMOTE_DIR}/data/processed/",
            ],
            check=True,
        )
    arch_temp = ROOT / "reports" / "temporal_holdout.json"
    if arch_temp.exists():
        subprocess.run(
            ["ssh", *_ssh_opts(port), f"root@{ip}", f"mkdir -p {REMOTE_DIR}/reports"],
            check=True,
        )
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh {' '.join(_ssh_opts(port))}",
                str(arch_temp),
                f"root@{ip}:{REMOTE_DIR}/reports/",
            ],
            check=True,
        )


def run_bootstrap(ip: str, port: int, env_vars: dict[str, str]) -> int:
    env_prefix = " ".join(f"{k}={v!r}" for k, v in env_vars.items())
    remote_cmd = (
        f"cd {REMOTE_DIR} && chmod +x scripts/runpod/bootstrap_train.sh && "
        f"env {env_prefix} bash scripts/runpod/bootstrap_train.sh"
    )
    proc = subprocess.run(["ssh", *_ssh_opts(port), f"root@{ip}", remote_cmd])
    return proc.returncode


def rsync_down(ip: str, port: int) -> None:
    # Pull known concrete files (rsync wildcards need remote shell expansion).
    concrete = [
        "data/processed/learned_embeddings.npy",
        "data/processed/learned_bgc_ids.csv",
        "data/processed/learned_embed_manifest.json",
        "artifacts/bgc_encoder.pt",
        "reports/learned_ablation_metrics.json",
        "reports/learned_novelty_comparison.json",
        "reports/learned_novelty_comparison.csv",
        "reports/learned_temporal_holdout.json",
        "reports/learned_temporal_holdout_ranking.csv",
        "reports/learned_eval_summary.json",
        "reports/learned_train_history.json",
        "reports/encoder_sweep_results.json",
        "reports/train_encoder_run.log",
        "reports/figures/learned_ablation_comparison.png",
        "reports/figures/learned_novelty_comparison.png",
        "reports/figures/learned_temporal_holdout.png",
        "reports/figures/learned_representation_summary.png",
        "reports/figures/encoder_sweep_summary.png",
    ]
    for rel in concrete:
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh {' '.join(_ssh_opts(port))}",
                f"root@{ip}:{REMOTE_DIR}/{rel}",
                str(dest),
            ],
            check=False,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu-type-id", default="NVIDIA A40", help="RunPod GPU type id")
    parser.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--volume-gb", type=int, default=40)
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--max-runtime-hours", type=float, default=2.0)
    parser.add_argument("--ssh-wait-s", type=int, default=600)
    parser.add_argument("--objective", default="simclr", choices=["simclr", "supcon"])
    parser.add_argument("--pooling", default="attention", choices=["attention", "mean", "deepsets"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--keep-frac", type=float, default=0.7)
    parser.add_argument("--feat-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prospective", action="store_true", help="Train on all BGCs (ignore date cutoff)")
    parser.add_argument("--sweep", action="store_true", help="Run scripts/run_encoder_sweep.py on the pod")
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--terminate", action="store_true")
    parser.add_argument("--pod-name", default="bgc-atlas-train-v3")
    parser.add_argument("--pod-id", default=None, help="Reuse an existing running pod")
    args = parser.parse_args(argv)

    import runpod

    api_key = _load_api_key()
    runpod.api_key = api_key

    env_vars = {
        "MAX_RUNTIME_SECONDS": str(int(args.max_runtime_hours * 3600)),
        "RUNPOD_API_KEY": api_key,
        "OBJECTIVE": args.objective,
        "POOLING": args.pooling,
        "EMBED_DIM": str(args.embed_dim),
        "HIDDEN_DIM": str(args.hidden_dim),
        "EPOCHS": str(args.epochs),
        "BATCH_SIZE": str(args.batch_size),
        "LR": str(args.lr),
        "TEMPERATURE": str(args.temperature),
        "KEEP_FRAC": str(args.keep_frac),
        "FEAT_DROPOUT": str(args.feat_dropout),
        "SEED": str(args.seed),
        "PROSPECTIVE": "0" if args.no_prospective else "1",
        "RUN_SWEEP": "1" if args.sweep else "0",
    }

    if args.pod_id:
        pod_id = args.pod_id
        print(f"[launch] Reusing pod {pod_id}")
    else:
        print(f"[launch] Creating pod ({args.gpu_type_id}, {args.cloud_type})...")
        pod = runpod.create_pod(
            name=args.pod_name,
            image_name=args.image,
            gpu_type_id=args.gpu_type_id,
            cloud_type=args.cloud_type,
            gpu_count=1,
            volume_in_gb=args.volume_gb,
            container_disk_in_gb=args.container_disk_gb,
            ports="22/tcp",
            support_public_ip=True,
            start_ssh=True,
            env={
                "RUNPOD_API_KEY": api_key,
                "MAX_RUNTIME_SECONDS": env_vars["MAX_RUNTIME_SECONDS"],
            },
        )
        pod_id = pod["id"]
        print(f"[launch] Pod created: {pod_id}")

    status = 1
    ip: str | None = None
    port: int | None = None
    try:
        print(f"[launch] Waiting for SSH (up to {args.ssh_wait_s}s)...")
        ip, port = wait_for_ssh(runpod, pod_id, args.ssh_wait_s)
        print(f"[launch] SSH ready at {ip}:{port}")

        print("[launch] Syncing repo + protein cache to the pod...")
        rsync_up(ip, port)

        print("[launch] Running bootstrap_train.sh on the pod (streaming below)...")
        status = run_bootstrap(ip, port, {**env_vars, "RUNPOD_POD_ID": pod_id})
        print(f"[launch] Remote job exited with status {status}")

        print("[launch] Syncing learned embeddings + reports back...")
        rsync_down(ip, port)
        print("[launch] Done. See data/processed/learned_*.npy and reports/learned_*.json")
    finally:
        if args.keep_alive:
            ssh_hint = f"ssh {' '.join(_ssh_opts(port))} root@{ip}" if ip else "(pod never became reachable)"
            print(
                f"[launch] --keep-alive set: leaving pod {pod_id} running.\n"
                f"  SSH:  {ssh_hint}\n"
                f"  Watchdog still fires after {args.max_runtime_hours}h."
            )
        elif args.terminate:
            print(f"[launch] Terminating pod {pod_id}...")
            runpod.terminate_pod(pod_id)
        else:
            print(f"[launch] Stopping pod {pod_id} (disk kept; GPU billing stops)...")
            runpod.stop_pod(pod_id)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
