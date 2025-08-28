from pathlib import Path
import requests
from tqdm import tqdm


def download_if_missing(url: str, dest: Path | str, chunk_size: int = 8192) -> Path:
    """
    Download *url* to *dest* unless the file is already present.

    Parameters
    ----------
    url : str
        HTTP or HTTPS address of the remote file.
    dest : pathlib.Path or str
        Target filename (absolute or relative).  Parent directories will be
        created automatically.
    chunk_size : int, optional
        Number of bytes to read per network iteration.  Defaults to 8 KiB.

    Returns
    -------
    pathlib.Path
        Path to the existing or newly downloaded file.

    Notes
    -----
    * Uses :pymeth:`pathlib.Path.exists` to avoid unnecessary downloads.  [oai_citation:4‡PyTutorial](https://pytutorial.com/check-file-existence-with-python-pathlibexists/?utm_source=chatgpt.com)
    * Streams the response with ``requests.get(..., stream=True)`` to handle
      large weights efficiently.  [oai_citation:5‡Real Python](https://realpython.com/python-download-file-from-url/?utm_source=chatgpt.com)
    * Displays a live progress bar powered by *tqdm*.  [oai_citation:6‡proxiesapi.com](https://proxiesapi.com/articles/downloading-binary-files-with-python-requests?utm_source=chatgpt.com)

    Examples
    --------
    >>> from pathlib import Path
    >>> ckpt = download_if_missing(
    ...     "https://huggingface.co/epigen/cellwhisperer/resolve/main/cellwhisperer_clip_v1.ckpt",
    ...     Path("~/models/cellwhisperer_clip_v1.ckpt"))
    >>> ckpt
    PosixPath('/home/you/models/cellwhisperer_clip_v1.ckpt')
    """
    dest = Path(dest).expanduser().resolve()
    if dest.exists():
        return dest  # nothing to do

    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar, \
             open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))

    return dest