"""Helpers shared by the command-line scripts in bin/."""


def boolean(text) -> bool:
    """A flag given on the command line as true or false, in any case."""
    return str(text).lower() == "true"


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)
