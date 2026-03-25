#!/usr/bin/env python
"""
Dataset fetcher for the Nextflow pipeline.

Supports two download sources:
  1. URL/Google Drive (default) — specified via `url` in dataset YAML
  2. CellxGene Census — specified via `source: cellxgene_census` + `census_dataset_id`

Pipeline:
  1. Looks up the dataset's YAML descriptor (name, URL/census_id, keys, …).
  2. Caches the **full** file under ``<cache_dir>/<dataset>.<ext>``
     (skips download when already present and valid).
  3. Always verifies that the required ``annotation_keys`` exist.
  4. Optionally writes a *subset* copy ``<cache_dir>/<dataset>_sub{n}.<ext>``.
  5. Optionally applies ontology mapping (overwrites cache in-place).
  6. Hard-links (or copies) the chosen file to Nextflow's work dir.

Everything is format-agnostic (``.h5ad`` or ``.zarr``).
"""
from __future__ import annotations
import argparse, html as ihtml, os, random, re, shutil, sys, yaml
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
        adata = anndata.read_h5ad(path, backed='r')
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
        if ad.raw is not None:
            try:
                _ = ad.raw.X
            except AttributeError:
                log_warn("Dropping malformed `raw` (missing raw.X) before writing.")
                ad.raw = None
        ad.write_h5ad(path, compression="gzip")
    elif path.suffix == ".zarr" or path.is_dir():
        ad.write_zarr(path, chunks=[], compression="gzip")
    else:
        sys.exit(f"[ERROR] Unsupported file type: {path}")

# --------------------------------------------------------------------------- #
# Shared download helpers                                                     #
# --------------------------------------------------------------------------- #
def _cache_valid(dest: Path) -> bool:
    """Return True if *dest* exists and is readable AnnData."""
    if not dest.exists():
        return False
    try:
        read_adata(dest)
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
                api = session.get(f"https://api.figshare.com/v2/file/{m.group(1)}")
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
# Verification & subsetting                                                   #
# --------------------------------------------------------------------------- #
def verify_columns(ad, keys: Iterable[str]) -> None:
    """Abort if any *keys* are missing from ``ad.obs``."""
    if missing := [k for k in keys if k not in ad.obs.columns]:
        sys.exit(f"[ERROR] Missing keys in adata.obs: {missing}")


def maybe_subset(full_path: Path, subset_path: Path, n: int | None) -> Path:
    """Create ``subset_path`` with *n* random cells if given; return chosen path."""
    if n is None:
        return full_path
    if subset_path.exists():
        log_info(f"Subset already exists – skipping: {subset_path.name}")
        return subset_path
    ad = read_adata(full_path)
    if n >= ad.n_obs:
        log_warn("subset_size ≥ n_obs – using full dataset")
        return full_path
    log_info(f"Writing subset ({n} cells) → {subset_path.name}")
    subset = ad[random.sample(range(ad.n_obs), n)]
    if getattr(subset, "isbacked", False):
        subset = subset.to_memory()
    write_adata(subset, subset_path)
    return subset_path

# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the downloader from the command line."""
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    log_info(f"Random seed: {RANDOM_SEED}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name (must match YAML 'name')")
    ap.add_argument("--cfg_yaml", required=True, help="YAML file with dataset configuration")
    ap.add_argument("--cache_dir", required=True, help="Local cache for full/subset files")
    ap.add_argument("--out", required=True, help="Destination inside the task work dir")
    ap.add_argument("--apply_ontology_mapping", type=lambda x: str(x).lower() == 'true',
                    default=True, help="Apply ontology mapping (default: True)")
    args = ap.parse_args()

    log_info(f"Loading cfg_yaml: {args.cfg_yaml}")
    with open(args.cfg_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    log_info(f"Loaded cfg keys: {sorted(cfg.keys())}")

    # --- Determine source ---
    source = str(cfg.get("source") or "").strip().lower()
    use_census = (source == "cellxgene_census") or ("census_dataset_id" in cfg)
    log_info(f"Download source: {'cellxgene_census' if use_census else 'url/drive'}")

    cache_dir = Path(args.cache_dir)
    ext = ".h5ad" if use_census else (Path(cfg["url"]).suffix or ".h5ad")
    full_path = cache_dir / f"{cfg['name']}{ext}"
    subset_path = cache_dir / f"{cfg['name']}_sub{cfg.get('subset_size')}{ext}"
    log_info(f"Dataset: {cfg['name']} | Cache: {cache_dir} | Full: {full_path.name} | Subset: {subset_path.name}")

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

        if not _cache_valid(full_path):
            log_info(f"Downloading Census source h5ad for dataset_id={dataset_id}")
            cellxgene_census.download_source_h5ad(str(dataset_id), to_path=str(full_path), census_version=census_version)
            log_info(f"Census download complete: {full_path.name}")
    else:
        if "url" not in cfg:
            sys.exit("[ERROR] Non-Census download requires `url` in YAML.")
        log_info(f"URL: {cfg['url']}")
        if file_id := extract_gdrive_file_id(cfg["url"]):
            log_info(f"Detected Google Drive, file_id={file_id}")
            stream_download_gdrive(file_id, full_path)
        elif is_figshare_url(cfg["url"]):
            log_info("Detected Figshare link")
            stream_download_figshare(cfg["url"], full_path)
        else:
            stream_download(cfg["url"], full_path)
        log_info(f"Download complete: {full_path.name}")

    # --- Verify columns ---
    log_info("Verifying required columns...")
    verify_columns(read_adata(full_path), list(cfg["annotation_keys"]) + [cfg["batch_key"]])
    log_info("Required columns verified")

    # --- Subset ---
    chosen = maybe_subset(full_path, subset_path, cfg.get("subset_size"))
    log_info(f"Chosen input: {chosen.name}")

    # --- Ontology mapping ---
    if args.apply_ontology_mapping:
        from util.download_helpers import apply_ontology_mapping_to_dataset
        log_info("Applying ontology mapping...")
        adata_input = read_adata(chosen)
        if getattr(adata_input, "isbacked", False):
            adata_input = adata_input.to_memory()
        adata_mapped = apply_ontology_mapping_to_dataset(
            adata=adata_input, annotation_keys=list(cfg["annotation_keys"]),
            cache_dir=cache_dir, verbose=True, origin=cfg['name'])
        write_adata(adata_mapped, chosen)
        log_info("Ontology mapping completed")
    else:
        log_info("Ontology mapping disabled - checking for original annotations...")
        adata, restored = read_adata(chosen), False
        for key in cfg["annotation_keys"]:
            orig_key = f"original_{key}"
            if orig_key in adata.obs.columns:
                log_info(f"Restoring '{key}' from '{orig_key}'")
                adata.obs[key] = adata.obs[orig_key].copy()
                restored = True
            elif key in adata.obs.columns:
                log_info(f"No backup for '{key}' - assuming original")
        if restored:
            write_adata(adata, chosen)
            log_info("Original annotations restored")
        else:
            log_info("Using annotations as-is")

    # --- Link/copy to Nextflow work dir ---
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if chosen.resolve() != dest:
        log_info(f"Linking {chosen.name} → {dest}")
        try: os.link(chosen, dest)
        except OSError: shutil.copy2(chosen, dest)
    log_info(f"OK: {dest.name} ready")


if __name__ == "__main__":
    main()
