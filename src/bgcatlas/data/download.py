"""Download MIBiG bulk archives into data/raw/."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from bgcatlas.paths import RAW, ensure_dirs

LOG = logging.getLogger(__name__)

MIBIG_BASE = "https://dl.secondarymetabolites.org/mibig"
MIBIG_FILES = (
    "mibig_json_4.0.tar.gz",
    "mibig_gbk_4.0.tar.gz",
)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        LOG.info("Already present: %s", dest.name)
        return
    LOG.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as fh,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))


def _extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        LOG.info("Already extracted: %s", out_dir.name)
        return
    LOG.info("Extracting %s → %s", archive.name, out_dir)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def download_mibig() -> None:
    """Fetch MIBiG 4.0 JSON + GenBank archives and extract them."""
    ensure_dirs()
    for name in MIBIG_FILES:
        archive = RAW / name
        _download_file(f"{MIBIG_BASE}/{name}", archive)
        stem = name.replace(".tar.gz", "")
        _extract_tar(archive, RAW / stem)
    LOG.info("MIBiG download complete under %s", RAW)
