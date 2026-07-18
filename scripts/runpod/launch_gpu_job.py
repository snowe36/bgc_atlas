#!/usr/bin/env python3
"""Orchestrate V2 ESM embed on RunPod: create pod, sync code+proteins, run
bootstrap.sh, pull embeddings back, stop the pod.

Usage:
    uv run python scripts/runpod/launch_gpu_job.py

Prerequisites:
    - RunPod API key in .env as RUNPOD_API_KEY (or API_KEY)
    - SSH public key added to RunPod account (Settings > SSH Keys)
    - local: `uv sync --extra embed` (or pip install runpod) for the SDK
    - data/processed/mibig_proteins.parquet present

Safety: pod-side watchdog self-terminates after --max-runtime-hours (default 2h).
"""

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
    # keep processed tiny; only proteins needed on pod (see rsync include below)
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
    raise SystemExit(
        "No RunPod API key found. Set RUNPOD_API_KEY (or API_KEY) in .env or the environment."
    )


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
    proteins = ROOT / "data" / "processed" / "mibig_proteins.parquet"
    if not proteins.exists():
        raise SystemExit(f"Missing {proteins}; run bgc-download first.")

    subprocess.run(["ssh", *_ssh_opts(port), f"root@{ip}", f"mkdir -p {REMOTE_DIR}/data/processed"], check=True)
    excludes = []
    for e in RSYNC_EXCLUDES:
        excludes += ["--exclude", e]
    # Sync code (exclude bulky processed except we push proteins separately)
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
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh {' '.join(_ssh_opts(port))}",
            str(proteins),
            f"root@{ip}:{REMOTE_DIR}/data/processed/",
        ],
        check=True,
    )


def run_bootstrap(ip: str, port: int, env_vars: dict[str, str]) -> int:
    env_prefix = " ".join(f"{k}={v!r}" for k, v in env_vars.items())
    remote_cmd = (
        f"cd {REMOTE_DIR} && chmod +x scripts/runpod/bootstrap.sh && "
        f"env {env_prefix} bash scripts/runpod/bootstrap.sh"
    )
    proc = subprocess.run(["ssh", *_ssh_opts(port), f"root@{ip}", remote_cmd])
    return proc.returncode


def rsync_down(ip: str, port: int) -> None:
    for pattern in (
        "data/processed/esm_embeddings.npy",
        "data/processed/esm_bgc_ids.csv",
        "data/processed/esm_embed_manifest.json",
        "data/processed/esm_protein_embeddings.npy",
        "data/processed/esm_protein_meta.csv",
        "reports/esm_embed_run.log",
    ):
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh {' '.join(_ssh_opts(port))}",
                f"root@{ip}:{REMOTE_DIR}/{pattern}",
                f"{ROOT}/{pattern}",
            ],
            check=False,  # protein cache optional if disk tight
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
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--pooling", default="length_weighted", choices=["mean", "length_weighted"])
    parser.add_argument("--max-aa", type=int, default=1024)
    parser.add_argument("--max-proteins-per-bgc", type=int, default=80)
    parser.add_argument("--batch-tokens", type=int, default=6000)
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--terminate", action="store_true")
    parser.add_argument("--pod-name", default="bgc-atlas-esm-v2")
    parser.add_argument("--pod-id", default=None, help="Reuse an existing running pod instead of creating")
    args = parser.parse_args(argv)

    import runpod

    api_key = _load_api_key()
    runpod.api_key = api_key

    env_vars = {
        "MAX_RUNTIME_SECONDS": str(int(args.max_runtime_hours * 3600)),
        "RUNPOD_API_KEY": api_key,
        "ESM_MODEL": args.model,
        "ESM_POOLING": args.pooling,
        "ESM_MAX_AA": str(args.max_aa),
        "ESM_MAX_PROTEINS": str(args.max_proteins_per_bgc),
        "ESM_BATCH_TOKENS": str(args.batch_tokens),
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

        print("[launch] Syncing repo + mibig_proteins.parquet to the pod...")
        rsync_up(ip, port)

        print("[launch] Running bootstrap.sh on the pod (streaming below)...")
        status = run_bootstrap(ip, port, {**env_vars, "RUNPOD_POD_ID": pod_id})
        print(f"[launch] Remote job exited with status {status}")

        print("[launch] Syncing embeddings + manifest back...")
        rsync_down(ip, port)
        print("[launch] Done. See data/processed/esm_*.npy and esm_embed_manifest.json")
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
