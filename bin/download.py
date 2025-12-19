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
import re

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


#--- For downloading data from google drive ---#
_DRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)")
_DRIVE_ID_RE   = re.compile(r"[?&]id=([^&]+)")

def extract_gdrive_file_id(url: str) -> str | None:
    m = _DRIVE_FILE_RE.search(url)
    if m:
        return m.group(1)
    m = _DRIVE_ID_RE.search(url)
    if m:
        return m.group(1)
    return None

def _drive_uc_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

import html as ihtml
from urllib.parse import urljoin

import html as ihtml
from urllib.parse import urljoin

def _open_drive_stream(file_id: str) -> requests.Response:
    s = requests.Session()

    r1 = s.get(_drive_uc_url(file_id), stream=True, allow_redirects=True)
    r1.raise_for_status()

    ct1 = (r1.headers.get("content-type") or "").lower()
    if "text/html" not in ct1:
        return r1  # already the file

    page = ihtml.unescape(r1.text)

    # If it's a login/permission page, plain requests will never work
    if ("accounts.google.com" in page) or ("ServiceLogin" in page) or ("Sign in" in page):
        raise RuntimeError(
            "Google Drive returned a login/permission page. "
            "Make the file accessible to 'Anyone with the link (Viewer)' "
            "or use an authenticated download."
        )

    # A) Confirm token sometimes appears as a cookie on redirects
    confirm = None
    for k, v in s.cookies.get_dict().items():
        if k.startswith("download_warning"):
            confirm = v
            break
    if confirm:
        r2 = s.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": confirm},
            stream=True,
            allow_redirects=True,
        )
        r2.raise_for_status()
        return r2

    # B) Most robust: parse and submit the download form (action + hidden inputs)
    form_m = re.search(
        r'(<form[^>]*id="download-form"[^>]*>.*?</form>)',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not form_m:
        # fallback: first form whose action contains "download"
        form_m = re.search(
            r'(<form[^>]*action="[^"]*download[^"]*"[^>]*>.*?</form>)',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )

    if form_m:
        form_html = form_m.group(1)

        action_m = re.search(r'action="([^"]+)"', form_html, flags=re.IGNORECASE)
        if not action_m:
            raise RuntimeError("Drive interstitial: download form has no action URL.")
        action = action_m.group(1)

        method_m = re.search(r'method="([^"]+)"', form_html, flags=re.IGNORECASE)
        method = (method_m.group(1).lower() if method_m else "get")

        # collect hidden inputs
        inputs = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', form_html))
        # ensure file id is present
        inputs.setdefault("id", file_id)

        action_url = urljoin(r1.url, action)

        if method == "post":
            r2 = s.post(action_url, data=inputs, stream=True, allow_redirects=True)
        else:
            r2 = s.get(action_url, params=inputs, stream=True, allow_redirects=True)

        r2.raise_for_status()
        return r2

    # C) If no form found, show snippet for debugging
    raise RuntimeError(
        "Could not find Drive download form in interstitial HTML. "
        f"First 400 chars:\n{page[:400]}"
    )

def stream_download_gdrive(file_id: str, dest: Path) -> None:
    if dest.exists():
        try:
            read_adata(dest)
            print(f"[INFO] {dest} already present and valid – skipping download")
            return
        except Exception as e:
            print(f"[WARN] {dest} exists but appears corrupted ({e}) – re-downloading")
            dest.unlink()

    r = _open_drive_stream(file_id)
    ct = (r.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        raise RuntimeError("Google Drive returned HTML (virus-scan/permission page), not the file.")
    total = int(r.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))


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
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name (must match YAML 'name')")
    ap.add_argument("--cfg_yaml", required=True,
                    help="YAML file containing dataset configuration")
    ap.add_argument("--cache_dir", required=True,
                    help="Local cache for full / subset files")
    ap.add_argument("--out", required=True,
                    help="Destination inside the task work dir")
    ap.add_argument("--apply_ontology_mapping", type=lambda x: (str(x).lower() == 'true'),
                    default=True,
                    help="Apply ontology mapping to annotation columns (default: True)")
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
    file_id = extract_gdrive_file_id(cfg["url"])
    if file_id:
        print(f"[INFO] Detected Google Drive link, file_id={file_id}")
        stream_download_gdrive(file_id, full_path)
    else:
        stream_download(cfg["url"], full_path)


    # Always verify before (optional) subsetting
    ad_full = read_adata(full_path)
    verify_columns(ad_full, list(cfg["annotation_keys"]) + [cfg["batch_key"]])

    chosen = maybe_subset(
        full_path,
        subset_path,
        cfg.get("subset_size")
    )

    # Apply ontology mapping to annotation columns if requested
    # Controlled by global flag from nextflow.config
    if args.apply_ontology_mapping:
        from util.download_helpers import apply_ontology_mapping_to_dataset
        
        print("[INFO] Applying ontology mapping to annotations...")
        adata_to_map = read_adata(chosen)
        
        # Apply ontology mapping to annotation keys
        adata_mapped = apply_ontology_mapping_to_dataset(
            adata=adata_to_map,
            annotation_keys=list(cfg["annotation_keys"]),
            cache_dir=download_cache_dir,
            verbose=True,
            origin=cfg['name']  # Pass dataset name as origin
        )
        
        # Write the mapped data back
        write_adata(adata_mapped, chosen)
        print("[INFO] Ontology mapping completed and saved")
    else:
        # Mapping disabled - check if we need to restore original annotations from cache
        print("[INFO] Ontology mapping disabled - checking for original annotations...")
        adata = read_adata(chosen)
        restored_any = False
        
        for key in cfg["annotation_keys"]:
            original_key = f"original_{key}"
            if original_key in adata.obs.columns:
                print(f"[INFO] Restoring '{key}' from '{original_key}' backup column...")
                adata.obs[key] = adata.obs[original_key].copy()
                restored_any = True
            elif key in adata.obs.columns:
                # Original column doesn't exist, assume current column is already original
                print(f"[INFO] No backup found for '{key}' - assuming current values are original")
        
        if restored_any:
            # Write the restored data back to cache
            write_adata(adata, chosen)
            print("[INFO] Original annotations restored and saved")
        else:
            print("[INFO] No mapped annotations found - using original annotations as-is")

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