from pathlib import Path
import requests
from tqdm import tqdm


def download_if_missing(url: str, dest: Path | str, chunk_size: int = 8192) -> Path:
    """Download url to dest (creating its folder) unless dest exists, and return the resolved path."""
    dest = Path(dest).expanduser().resolve()
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar, \
             open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size):
                if chunk:  # skip keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))

    return dest