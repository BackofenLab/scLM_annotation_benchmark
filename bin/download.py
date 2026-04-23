#!/usr/bin/env python
"""
Dataset fetcher for the Nextflow pipeline.

Supports two download sources:
  1. URL/Google Drive (default) — specified via `url` in dataset YAML
  2. CellxGene Census — specified via `source: cellxgene_census` + `census_dataset_id`

Pipeline:
  1. Looks up the dataset's YAML descriptor (name, URL/census_id, keys, ...).
  2. Downloads the full file directly into the Nextflow work dir (``--out``).
     Persistent caching is handled by Nextflow's ``storeDir`` in ``main.nf``.
  3. Verifies that the required ``annotation_key`` and ``batch_key`` exist.

Everything is format-agnostic (``.h5ad`` or ``.zarr``).
"""
from __future__ import annotations
import argparse, html as ihtml, re, shutil, sys, yaml
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import anndata
import pandas as pd
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Optional: CellxGene Census (may fail on macOS due to native deps)           #
# --------------------------------------------------------------------------- #
_cellxgene_census_import_error: Exception | None = None
try:
    import cellxgene_census  # type: ignore
except Exception as e:
    cellxgene_census = None  # type: ignore[assignment]
    _cellxgene_census_import_error = e

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log_info(msg: str) -> None:  print(f"[INFO] {msg}", flush=True)
def log_warn(msg: str) -> None:  print(f"[WARN] {msg}", flush=True)
def log_error(msg: str) -> None: print(f"[ERROR] {msg}", flush=True)

# --------------------------------------------------------------------------- #
# AnnData I/O                                                                 #
# --------------------------------------------------------------------------- #
def read_adata(path: Path):
    """Load AnnData from ``.h5ad`` or ``.zarr``."""
    if path.suffix == ".h5ad":
        adata = anndata.read_h5ad(path)
        return adata
    if path.suffix == ".zarr" or path.is_dir():
        return anndata.read_zarr(path)
    sys.exit(f"[ERROR] Unsupported file type: {path}")


def write_adata(ad, path: Path) -> None:
    """Write AnnData, sanitizing Arrow-backed strings first (Census files)."""
    def is_arrow(dtype) -> bool:
        s = str(dtype).lower()
        return "arrow" in s or s == "string" or s.startswith("string[") or "StringDtype" in type(dtype).__name__

    def dearrow_series(s: pd.Series) -> pd.Series:
        if hasattr(s, "cat") and hasattr(s.cat, "categories") and is_arrow(s.cat.categories.dtype):
            return s.cat.rename_categories(s.cat.categories.astype(object))
        return s.astype(object) if is_arrow(s.dtype) else s

    def dearrow_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if is_arrow(df.index.dtype):
            df.index = df.index.astype(object)
        for c in df.columns:
            try: df[c] = dearrow_series(df[c])
            except Exception: pass
        return df

    try:
        ad.obs, ad.var = dearrow_df(ad.obs), dearrow_df(ad.var)
        if ad.raw is not None:
            ad.raw._var = dearrow_df(ad.raw.var)  # type: ignore[attr-defined]
    except Exception as e:
        log_warn(f"Could not fully sanitize Arrow dtypes: {e}")

    if path.suffix == ".h5ad":
        ad.write_h5ad(path, compression="gzip")
    elif path.suffix == ".zarr" or path.is_dir():
        ad.write_zarr(path, chunks=[], compression="gzip")
    else:
        sys.exit(f"[ERROR] Unsupported file type: {path}")

# --------------------------------------------------------------------------- #
# Shared download helpers                                                     #
# --------------------------------------------------------------------------- #
def _cache_valid(dest: Path) -> bool:
    """Return True if *dest* exists and is readable AnnData (backed-mode check — O(1) RAM)."""
    if not dest.exists():
        return False
    try:
        if dest.suffix == ".h5ad":
            ad = anndata.read_h5ad(dest, backed="r")
            try:
                _ = ad.shape
            finally:
                ad.file.close()
        else:
            # zarr or dir: a minimal open validates structure
            anndata.read_zarr(dest)
        log_info(f"Cache hit (valid) – skipping download: {dest.name}")
        return True
    except Exception as e:
        log_warn(f"Cache hit but unreadable – re-downloading: {dest.name} ({e})")
        if dest.is_file(): dest.unlink()
        else: shutil.rmtree(dest, ignore_errors=True)
        return False


def _stream_to_file(resp: requests.Response, dest: Path) -> int:
    """Stream *resp* to *dest* with progress bar; return bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total, written = int(resp.headers.get("content-length", 0)), 0
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(1 << 20):
            if chunk:
                fh.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
    return written


def _validate_download(dest: Path, written: int, source_desc: str) -> None:
    """Ensure downloaded file is valid AnnData; remove and raise if not."""
    if written == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: 0 bytes from {source_desc}")
    try:
        read_adata(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not valid AnnData: {e}") from e


def _new_session() -> requests.Session:
    """Return a requests.Session with a reasonable User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; sclfm-downloader/1.0)"})
    return s

# --------------------------------------------------------------------------- #
# Generic URL download                                                        #
# --------------------------------------------------------------------------- #
def stream_download(url: str, dest: Path) -> None:
    """Stream a remote file to *dest* with progress. Skips if valid cache."""
    if _cache_valid(dest):
        return
    r = _new_session().get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    if "text/html" in (r.headers.get("content-type") or "").lower():
        raise RuntimeError(f"Server returned HTML instead of file: {url}")
    _validate_download(dest, _stream_to_file(r, dest), url)

# --------------------------------------------------------------------------- #
# Figshare download                                                           #
# --------------------------------------------------------------------------- #
_FIGSHARE_RE = re.compile(r"figshare\.com/ndownloader/files/(\d+)")

def is_figshare_url(url: str) -> bool:
    return bool(_FIGSHARE_RE.search(url))


def stream_download_figshare(url: str, dest: Path) -> None:
    """Handle Figshare downloads (may serve HTML interstitial)."""
    if _cache_valid(dest):
        return
    session = _new_session()
    r = session.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()

    ct = (r.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        page = r.text
        real_url = None
        # Try various HTML patterns
        for pattern in [
            r'href="(https://[^"]*ndownloader[^"]*\.h5ad[^"]*)"',
            r'href="(https://[^"]*figstatic\.com/[^"]+)"',
            r'data-file-url="([^"]+)"',
            r'"downloadUrl"\s*:\s*"([^"]+)"',
        ]:
            if m := re.search(pattern, page, re.IGNORECASE):
                real_url = ihtml.unescape(m.group(1))
                break
        # Fallback: Figshare API
        if not real_url and (m := _FIGSHARE_RE.search(url)):
            try:
                api = session.get(f"https://api.figshare.com/v2/file/download/{m.group(1)}")
                if api.ok:
                    real_url = api.json().get("download_url")
            except Exception:
                pass
        if not real_url:
            raise RuntimeError(f"Figshare returned HTML but could not extract URL: {url}")
        log_info(f"Figshare redirect: {real_url[:80]}...")
        r = session.get(real_url, stream=True, allow_redirects=True)
        r.raise_for_status()
        if "text/html" in (r.headers.get("content-type") or "").lower():
            raise RuntimeError(f"Figshare secondary URL also returned HTML: {real_url}")

    _validate_download(dest, _stream_to_file(r, dest), f"Figshare: {url}")

# --------------------------------------------------------------------------- #
# Google Drive download                                                       #
# --------------------------------------------------------------------------- #
_GDRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)")
_GDRIVE_ID_RE = re.compile(r"[?&]id=([^&]+)")


def extract_gdrive_file_id(url: str) -> str | None:
    """Extract Google Drive file ID from various URL formats."""
    for regex in (_GDRIVE_FILE_RE, _GDRIVE_ID_RE):
        if m := regex.search(url):
            return m.group(1)
    return None


def _open_drive_stream(file_id: str) -> requests.Response:
    """Open a streaming response for a Google Drive file, handling interstitials."""
    s = requests.Session()
    uc_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = s.get(uc_url, stream=True, allow_redirects=True)
    r.raise_for_status()

    if "text/html" not in (r.headers.get("content-type") or "").lower():
        return r

    page = ihtml.unescape(r.text)
    if any(x in page for x in ("accounts.google.com", "ServiceLogin", "Sign in")):
        raise RuntimeError("Google Drive returned login page. Make file accessible to 'Anyone with the link'.")

    # Check for download_warning cookie
    for k, v in s.cookies.get_dict().items():
        if k.startswith("download_warning"):
            r2 = s.get("https://drive.google.com/uc", params={"export": "download", "id": file_id, "confirm": v},
                       stream=True, allow_redirects=True)
            r2.raise_for_status()
            return r2

    # Parse download form
    for form_pattern in [r'(<form[^>]*id="download-form"[^>]*>.*?</form>)',
                         r'(<form[^>]*action="[^"]*download[^"]*"[^>]*>.*?</form>)']:
        if form_m := re.search(form_pattern, page, re.IGNORECASE | re.DOTALL):
            form_html = form_m.group(1)
            action_m = re.search(r'action="([^"]+)"', form_html, re.IGNORECASE)
            if not action_m:
                raise RuntimeError("Drive download form has no action URL.")
            method = "post" if (mm := re.search(r'method="([^"]+)"', form_html, re.IGNORECASE)) and mm.group(1).lower() == "post" else "get"
            inputs = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', form_html))
            inputs.setdefault("id", file_id)
            action_url = urljoin(r.url, action_m.group(1))
            r2 = s.post(action_url, data=inputs, stream=True, allow_redirects=True) if method == "post" else \
                 s.get(action_url, params=inputs, stream=True, allow_redirects=True)
            r2.raise_for_status()
            return r2

    raise RuntimeError(f"Could not find Drive download form.\nFirst 400 chars:\n{page[:400]}")


def stream_download_gdrive(file_id: str, dest: Path) -> None:
    """Download a Google Drive file to *dest*."""
    if _cache_valid(dest):
        return
    r = _open_drive_stream(file_id)
    if "text/html" in (r.headers.get("content-type") or "").lower():
        raise RuntimeError("Google Drive returned HTML (virus-scan/permission page).")
    _validate_download(dest, _stream_to_file(r, dest), f"Google Drive id={file_id}")

# --------------------------------------------------------------------------- #
# Verification                                                                #
# --------------------------------------------------------------------------- #
def verify_columns(ad, keys: Iterable[str]) -> None:
    """Abort if any *keys* are missing from ``ad.obs``."""
    if missing := [k for k in keys if k not in ad.obs.columns]:
        sys.exit(f"[ERROR] Missing keys in adata.obs: {missing}")

# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the downloader from the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name (must match YAML 'name')")
    ap.add_argument("--cfg_yaml", required=True, help="YAML file with dataset configuration")
    ap.add_argument("--out", required=True, help="Destination inside the task work dir")
    args = ap.parse_args()

    log_info(f"Loading cfg_yaml: {args.cfg_yaml}")
    with open(args.cfg_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    log_info(f"Loaded cfg keys: {sorted(cfg.keys())}")

    # --- Determine source ---
    source = str(cfg.get("source") or "").strip().lower()
    use_census = (source == "cellxgene_census") or ("census_dataset_id" in cfg)
    log_info(f"Download source: {'cellxgene_census' if use_census else 'url/drive'}")

    # Write directly into the Nextflow work dir; `storeDir` handles persistent caching.
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Dataset: {cfg['name']} | Destination: {dest}")

    # --- Download ---
    if use_census:
        if cellxgene_census is None:
            sys.exit(f"[ERROR] CellxGene Census requested but import failed: {_cellxgene_census_import_error}")
        if not (dataset_id := cfg.get("census_dataset_id")):
            sys.exit("[ERROR] Census download requires `census_dataset_id` in YAML.")
        census_version = str(cfg.get("census_version") or "stable")
        log_info(f"Census dataset_id: {dataset_id} | version: {census_version}")
        try:
            directory = cellxgene_census.get_census_version_directory()
            if census_version in ("stable", "latest") and census_version in directory:
                log_info(f"Census {census_version} build: {directory[census_version]['release_build']}")
        except Exception as e:
            log_warn(f"Could not resolve Census build: {e}")

        if not _cache_valid(dest):
            log_info(f"Downloading Census source h5ad for dataset_id={dataset_id}")
            cellxgene_census.download_source_h5ad(str(dataset_id), to_path=str(dest), census_version=census_version)
            log_info(f"Census download complete: {dest.name}")
    else:
        if "url" not in cfg:
            sys.exit("[ERROR] Non-Census download requires `url` in YAML.")
        log_info(f"URL: {cfg['url']}")
        if file_id := extract_gdrive_file_id(cfg["url"]):
            log_info(f"Detected Google Drive, file_id={file_id}")
            stream_download_gdrive(file_id, dest)
        elif is_figshare_url(cfg["url"]):
            log_info("Detected Figshare link")
            stream_download_figshare(cfg["url"], dest)
        else:
            stream_download(cfg["url"], dest)
        log_info(f"Download complete: {dest.name}")

    # --- Verify columns ---
    log_info("Verifying required columns...")
    if dest.suffix == ".h5ad":
        ad_backed = anndata.read_h5ad(dest, backed="r")
        try:
            verify_columns(ad_backed, [cfg["annotation_key"], cfg["batch_key"]])
        finally:
            ad_backed.file.close()
    else:
        # zarr path — anndata currently reads zarr fully; acceptable, rare in practice
        verify_columns(read_adata(dest), [cfg["annotation_key"], cfg["batch_key"]])
    log_info("Required columns verified")
    log_info(f"OK: {dest.name} ready")


if __name__ == "__main__":
    main()