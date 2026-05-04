"""Shared HTTP helpers for downloader scripts."""
from __future__ import annotations
from pathlib import Path

import requests
from tqdm import tqdm


def new_session() -> requests.Session:
    """Return a requests.Session with a reasonable User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; sclfm-downloader/1.0)"})
    return s


def stream_to_file(resp: requests.Response, dest: Path) -> int:
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
