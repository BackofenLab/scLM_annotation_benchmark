#!/usr/bin/env python
"""Download the dataset described by a dataset YAML file.

The source is the `url` of the YAML (a Google Drive or Figshare link or a direct file link) or,
with `source: cellxgene_census` or a `census_dataset_id`, the CELLxGENE Census
(`census_version`, default "stable"). The .h5ad or .zarr file is written to --out, which
Nextflow's storeDir keeps between runs. The script then checks that obs has the columns
annotation_key and batch_key of the YAML.

    download.py --dataset NAME --cfg_yaml datasets/NAME.yml --out NAME_raw.h5ad
"""
from __future__ import annotations
import argparse, html as ihtml, re, shutil, sys, yaml
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import anndata
import requests

from util.http import new_session, stream_to_file

# Optional: the import can fail on macOS because of native dependencies.
_cellxgene_census_import_error: Exception | None = None
try:
    import cellxgene_census  # type: ignore
except Exception as e:
    cellxgene_census = None  # type: ignore[assignment]
    _cellxgene_census_import_error = e


def log_info(msg: str) -> None:  print(f"[INFO] {msg}", flush=True)
def log_warn(msg: str) -> None:  print(f"[WARN] {msg}", flush=True)
def log_error(msg: str) -> None: print(f"[ERROR] {msg}", flush=True)


def read_adata(path: Path):
    """Load AnnData from ``.h5ad`` or ``.zarr``."""
    if path.suffix == ".h5ad":
        adata = anndata.read_h5ad(path)
        return adata
    if path.suffix == ".zarr" or path.is_dir():
        return anndata.read_zarr(path)
    sys.exit(f"[ERROR] Unsupported file type: {path}")


def _cache_valid(dest: Path) -> bool:
    """Return True if dest exists and opens as AnnData (.h5ad in backed mode); delete it if it
    exists but does not open."""
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
            # zarr stores are read fully.
            anndata.read_zarr(dest)
        log_info(f"Valid file found, skipping download: {dest.name}")
        return True
    except Exception as e:
        log_warn(f"Existing file is unreadable, downloading it again: {dest.name} ({e})")
        if dest.is_file(): dest.unlink()
        else: shutil.rmtree(dest, ignore_errors=True)
        return False


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


def stream_download(url: str, dest: Path) -> None:
    """Download url to dest with a progress bar, unless dest already holds a valid file."""
    if _cache_valid(dest):
        return
    r = new_session().get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    if "text/html" in (r.headers.get("content-type") or "").lower():
        raise RuntimeError(f"Server returned HTML instead of file: {url}")
    _validate_download(dest, stream_to_file(r, dest), url)


_FIGSHARE_RE = re.compile(r"figshare\.com/ndownloader/files/(\d+)")

def is_figshare_url(url: str) -> bool:
    return bool(_FIGSHARE_RE.search(url))


def stream_download_figshare(url: str, dest: Path) -> None:
    """Download a Figshare file to dest. If Figshare answers with an HTML page, the file link is
    taken from the page or from the Figshare API."""
    if _cache_valid(dest):
        return
    session = new_session()
    r = session.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()

    ct = (r.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        page = r.text
        real_url = None
        # Forms of the file link in the page
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

    _validate_download(dest, stream_to_file(r, dest), f"Figshare: {url}")


_GDRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)")
_GDRIVE_ID_RE = re.compile(r"[?&]id=([^&]+)")


def extract_gdrive_file_id(url: str) -> str | None:
    """Extract Google Drive file ID from various URL formats."""
    for regex in (_GDRIVE_FILE_RE, _GDRIVE_ID_RE):
        if m := regex.search(url):
            return m.group(1)
    return None


def _open_drive_stream(file_id: str) -> requests.Response:
    """Open a streaming response for a Google Drive file, passing the confirmation page that
    Drive shows for large files."""
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
    _validate_download(dest, stream_to_file(r, dest), f"Google Drive id={file_id}")


def verify_columns(ad, keys: Iterable[str]) -> None:
    """Exit if any of keys is missing from ad.obs."""
    if missing := [k for k in keys if k not in ad.obs.columns]:
        sys.exit(f"[ERROR] Missing keys in adata.obs: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name (must match YAML 'name')")
    ap.add_argument("--cfg_yaml", required=True, help="YAML file with dataset configuration")
    ap.add_argument("--out", required=True, help="Destination inside the task work dir")
    args = ap.parse_args()

    log_info(f"Loading cfg_yaml: {args.cfg_yaml}")
    with open(args.cfg_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    log_info(f"Loaded cfg keys: {sorted(cfg.keys())}")

    source = str(cfg.get("source") or "").strip().lower()
    use_census = (source == "cellxgene_census") or ("census_dataset_id" in cfg)
    log_info(f"Download source: {'cellxgene_census' if use_census else 'url/drive'}")

    # Written to the Nextflow work directory; storeDir keeps the file between runs.
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Dataset: {cfg['name']} | Destination: {dest}")

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

    log_info("Verifying required columns...")
    if dest.suffix == ".h5ad":
        ad_backed = anndata.read_h5ad(dest, backed="r")
        try:
            verify_columns(ad_backed, [cfg["annotation_key"], cfg["batch_key"]])
        finally:
            ad_backed.file.close()
    else:
        # zarr stores are read fully.
        verify_columns(read_adata(dest), [cfg["annotation_key"], cfg["batch_key"]])
    log_info("Required columns verified")
    log_info(f"OK: {dest.name} ready")


if __name__ == "__main__":
    main()