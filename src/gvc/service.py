"""Start an HTTP model server on demand and stop it again afterwards."""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import httpx


class ServiceError(RuntimeError):
    pass


def is_up(url: str) -> bool:
    """True once the server answers without a server error.

    llama-server answers /health with 503 while it is still downloading or loading
    the model, so a listening socket alone does not mean ready.
    """
    try:
        return httpx.get(url, timeout=2.0).status_code < 500
    except httpx.TransportError:
        return False


@contextmanager
def managed_service(
    name: str,
    health_url: str,
    cmd: list[str],
    log_path: Path,
    manage: bool,
    env: dict[str, str] | None = None,
    startup_timeout: float = 600.0,
) -> Iterator[None]:
    """Ensure a server answers on health_url for the duration of the block.

    An already running server is used as-is and left running. Otherwise, if
    `manage` is set, `cmd` is started and terminated when the block exits, which
    frees its VRAM for the next stage.
    """
    if is_up(health_url):
        yield
        return
    if not manage:
        raise ServiceError(f"{name} is not reachable at {health_url} and manage_server is off")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
        except FileNotFoundError as e:
            raise ServiceError(f"cannot start {name}: {cmd[0]} not found (are you in `nix develop`?)") from e
        try:
            deadline = time.monotonic() + startup_timeout
            while not is_up(health_url):
                if proc.poll() is not None:
                    raise ServiceError(f"{name} exited with code {proc.returncode}, see {log_path}")
                if time.monotonic() > deadline:
                    raise ServiceError(f"{name} did not come up within {startup_timeout:.0f}s, see {log_path}")
                time.sleep(0.5)
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname or "127.0.0.1", parsed.port or default_port
