#!/usr/bin/env python
"""
Dataset fetcher for the Nextflow pipeline.

It

1. Looks up the dataset's YAML descriptor (name, URL, keys, …).
2. Caches the **full** file under ``<data_cache>/<dataset>.<ext>``  
   (skips download when already present).
3. Always verifies that the required ``annotation_keys`` exist.
4. Optionally writes a *subset* copy
   ``<data_cache>/<dataset>_sub{n}.<ext>`` (skips if present).
5. Copies (or hard-links) the file that will actually be used
   downstream into the task’s working directory under the plain name
   requested by Nextflow.

Everything is format-agnostic (`.h5ad` or `.zarr`).

"""
from __future__ import annotations
import argparse, os, random, shutil, sys, yaml
from pathlib import Path
from typing import Iterable

import anndata  # only imported inside read/write helpers
import requests
from tqdm import tqdm


# ------------------------------------------------------------------------- #
# Download + I/O helpers                                                    #
# ------------------------------------------------------------------------- #

def stream_download(url: str, dest: Path) -> None:
    """
    Stream a remote file to *dest* with a progress bar.

    Skips download if *dest* already exists and is valid.
    """
    if dest.exists():
        # Check if existing file is valid by trying to read it
        try:
            read_adata(dest)
            print(f"[INFO] {dest} already present and valid – skipping download")
            return
        except Exception as e:
            print(f"[WARN] {dest} exists but appears corrupted ({e}) – re-downloading")
            dest.unlink()  # Remove corrupted file

    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(total=total, unit="B",
                                      unit_scale=True,
                                      desc=dest.name) as bar:
        for chunk in r.iter_content(1 << 20):     # 1 MiB
            fh.write(chunk)
            bar.update(len(chunk))


def read_adata(path: Path):
    """Load an AnnData object from ``.h5ad`` or ``.zarr`` transparently."""
    if path.suffix == ".h5ad":
        return anndata.read_h5ad(path)
    if path.suffix == ".zarr" or path.is_dir():
        return anndata.read_zarr(path)
    sys.exit(f"[ERROR] Unsupported file type: {path}")


def write_adata(ad, path: Path):
    """Write an AnnData object back in the same format as *path*."""
    if path.suffix == ".h5ad":
        ad.write_h5ad(path, compression="gzip")
    elif path.suffix == ".zarr" or path.is_dir():
        ad.write_zarr(path, chunks=[], compression="gzip")
    else:
        sys.exit(f"[ERROR] Unsupported file type: {path}")


# ------------------------------------------------------------------------- #
# Verification & subsetting                                                 #
# ------------------------------------------------------------------------- #

def verify_columns(ad, keys: Iterable[str]) -> None:
    """Abort if any *keys* are missing from ``ad.obs``."""
    missing = [k for k in keys if k not in ad.obs.columns]
    if missing:
        sys.exit(f"[ERROR] Missing keys in adata.obs: {missing}")


def maybe_subset(full_path: Path, subset_path: Path,
                 n: int | None) -> Path:
    """
    Create ``subset_path`` with *n* random cells if *n* is given.

    Returns the path that callers should use (subset or full).
    """
    if n is None:
        return full_path

    if subset_path.exists():
        print(f"[INFO] Subset {subset_path.name} already exists – skipping")
        return subset_path

    ad = read_adata(full_path)

    if n >= ad.n_obs:           # nothing to subset
        print(f"[WARN] subset_size ≥ n_obs – using full dataset")
        return full_path

    print(f"[INFO] Writing subset ({n} cells) → {subset_path.name}")
    keep = random.sample(range(ad.n_obs), n)
    ad = ad[keep]
    write_adata(ad, subset_path)
    return subset_path


# ------------------------------------------------------------------------- #
# CLI entry-point                                                           #
# ------------------------------------------------------------------------- #

def main() -> None:
    """
    Run the downloader from the command line.

    Examples
    --------
    >>> download.py --dataset Tabula_sapiens --cfg-dir datasets \\
    ...            --data_cache /scratch/datasets \\
    ...            --out data/Tabula_sapiens.h5ad
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name (must match YAML 'name')")
    ap.add_argument("--cfg_yaml", required=True,
                    help="YAML file containing dataset configuration")
    ap.add_argument("--cache_dir", required=True,
                    help="Local cache for full / subset files")
    ap.add_argument("--out", required=True,
                    help="Destination inside the task work dir")
    args = ap.parse_args()
    print(f"Loading cfg_yaml: {args.cfg_yaml}")
    # load the yaml file
    with open(args.cfg_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded cfg: {cfg}")
    ext = Path(cfg["url"]).suffix or ".h5ad"      # default if URL lacks ext
    download_cache_dir = Path(args.cache_dir)
    print(f"Downloading {cfg['name']} to {download_cache_dir}")
    full_path   = download_cache_dir / f"{cfg['name']}{ext}"
    subset_path = download_cache_dir / \
                  f"{cfg['name']}_sub{cfg.get('subset_size')}{ext}"
    print(f"Full path: {full_path}")
    print(f"Subset path: {subset_path}")
    stream_download(cfg["url"], full_path)

    # Always verify before (optional) subsetting
    ad_full = read_adata(full_path)
    verify_columns(ad_full, list(cfg["annotation_keys"]) + [cfg["batch_key"]])

    chosen = maybe_subset(
        full_path,
        subset_path,
        cfg.get("subset_size")
    )

    # Use a hard-link if possible; falls back to copy across filesystems
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if source and destination are the same file
    if chosen.resolve() == dest:
        print(f"[INFO] Source and destination are the same file - no linking needed")
    else:
        print(f"Linking {chosen} to {dest}")
        try:
            os.link(chosen, dest)
        except OSError:             # across FS boundaries
            shutil.copy2(chosen, dest)

    print(f"[OK] {dest.name} ready")

if __name__ == "__main__":
    main()