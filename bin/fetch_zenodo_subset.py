#!/usr/bin/env python
"""
Fetch a preprocessed + subsetted AnnData file from the pinned Zenodo record.

Reads the manifest (`datasets/zenodo_manifest.yml`), resolves the expected
filename from (dataset, subset_size, strategy), downloads it, MD5-verifies,
and confirms the expected annotation column exists. Exits non-zero on any
failure so Nextflow aborts and `-resume` retries cleanly (storeDir never
sees a partial file).
"""
from __future__ import annotations
import argparse, hashlib, sys, yaml
from pathlib import Path

import anndata

# bin/ is on PYTHONPATH when invoked by Nextflow
from util.http import new_session, stream_to_file


def log_info(msg: str) -> None:  print(f"[INFO] {msg}", flush=True)
def log_error(msg: str) -> None: print(f"[ERROR] {msg}", flush=True)


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset_size", type=int, help="With --strategy, builds the subset filename.")
    ap.add_argument("--strategy", help="With --subset_size, builds the subset filename.")
    ap.add_argument("--filename", help="Explicit manifest filename (e.g. <dataset>_context.h5ad); "
                                       "overrides the subset_size/strategy construction.")
    ap.add_argument("--annotation_key", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    if args.filename:
        filename = args.filename
    elif args.subset_size is not None and args.strategy:
        filename = f"{args.dataset}_sub{args.subset_size}_{args.strategy}.h5ad"
    else:
        log_error("provide --filename, or both --subset_size and --strategy")
        sys.exit(1)

    entry = manifest.get("files", {}).get(filename)
    if entry is None:
        log_error(f"{filename} not in manifest ({args.manifest})")
        sys.exit(1)

    url = f"{manifest['base_url']}/{filename}/content"
    dest = Path(args.output).resolve()
    log_info(f"Fetching {filename} from {url}")

    session = new_session()
    r = session.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    written = stream_to_file(r, dest)
    if written == 0:
        dest.unlink(missing_ok=True)
        log_error(f"Downloaded 0 bytes from {url}")
        sys.exit(1)

    actual = md5sum(dest)
    expected = entry["md5"]
    if actual != expected:
        dest.unlink(missing_ok=True)
        log_error(f"MD5 mismatch for {filename}: got {actual}, expected {expected}")
        sys.exit(1)
    log_info(f"MD5 verified: {actual}")

    ad = anndata.read_h5ad(dest, backed="r")
    try:
        if args.annotation_key not in ad.obs.columns:
            log_error(f"Missing annotation_key '{args.annotation_key}' in {filename}. obs columns: {list(ad.obs.columns)}")
            sys.exit(1)
    finally:
        ad.file.close()

    log_info(f"OK: {dest.name} ready ({written} bytes)")


if __name__ == "__main__":
    main()
