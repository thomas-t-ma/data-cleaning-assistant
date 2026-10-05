from __future__ import annotations

import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .ai_provider import OllamaProvider
from .semantic_narrator import LocalSemanticNarrator


_DESKTOP_HOST = "127.0.0.1"


def _find_available_port(host: str = _DESKTOP_HOST) -> int:
    """Ask the OS for an unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(
    health_url: str,
    *,
    server_thread: threading.Thread,
    server_errors: list[Exception],
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server_errors:
            raise RuntimeError("The local desktop service failed to start.") from server_errors[0]
        if not server_thread.is_alive():
            raise RuntimeError(
                "The local desktop service stopped before it became ready."
            )
        try:
            with urlopen(health_url, timeout=0.35) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.05)

    raise RuntimeError(
        "The local desktop service did not become ready in time. "
        "No dataset was sent anywhere."
    )


def run_local_desktop(
    *,
    data_root: str | Path,
    max_upload_mb: int = 1024,
    narrator_model: str | None = None,
    port: int | None = None,
) -> None:
    """Run the existing local UI inside a native desktop window.

    The cleaning engine and HTTP API remain identical to browser mode. The only
    network listener is a loopback-only server used by the embedded webview.
    """
    try:
        import uvicorn
        import webview

        from .web_app import create_app
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Desktop dependencies are not installed. Install with: '
            'pip install -e ".[desktop]"'
        ) from exc

    selected_port = port or _find_available_port()
    if not 1 <= selected_port <= 65535:
        raise ValueError("Desktop port must be between 1 and 65535.")

    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    narrator = (
        LocalSemanticNarrator(OllamaProvider(), model=narrator_model)
        if narrator_model
        else None
    )
    app = create_app(
        root,
        max_upload_mb=max_upload_mb,
        trusted_hosts=[_DESKTOP_HOST, "localhost", "[::1]"],
        narrator=narrator,
    )

    config = uvicorn.Config(
        app,
        host=_DESKTOP_HOST,
        port=selected_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_errors: list[Exception] = []

    def _serve() -> None:
        try:
            server.run()
        except Exception as exc:  # pragma: no cover - defensive thread boundary
            server_errors.append(exc)

    server_thread = threading.Thread(
        target=_serve,
        name="data-cleaner-desktop-server",
        daemon=True,
    )
    server_thread.start()

    url = f"http://{_DESKTOP_HOST}:{selected_port}/"
    health_url = f"http://{_DESKTOP_HOST}:{selected_port}/api/health"
    try:
        _wait_until_ready(
            health_url,
            server_thread=server_thread,
            server_errors=server_errors,
        )

        storage_path = root.parent / "desktop-ui"
        storage_path.mkdir(parents=True, exist_ok=True)

        # pywebview disables ordinary downloads unless explicitly enabled.
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.create_window(
            "Data Cleaning Assistant",
            url=url,
            width=1280,
            height=860,
            min_size=(860, 620),
            resizable=True,
            background_color="#f4f6f8",
            text_select=True,
            zoomable=True,
        )
        webview.start(
            gui="edgechromium" if sys.platform == "win32" else None,
            private_mode=False,
            storage_path=str(storage_path),
        )
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)
        if server_thread.is_alive():
            server.force_exit = True


def _desktop_self_test() -> None:
    """Validate imports and app construction without opening a GUI."""
    try:
        import webview  # noqa: F401

        from .web_app import create_app
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Desktop self-test could not import the packaged GUI/web dependencies."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="data-cleaner-self-test-") as temp_dir:
        app = create_app(Path(temp_dir) / "sessions")
        if app.title != "Data Cleaning Assistant":
            raise RuntimeError("Desktop self-test created an unexpected application.")
        port = _find_available_port()
        if not 1 <= port <= 65535:
            raise RuntimeError("Desktop self-test could not reserve a loopback port.")


def _show_packaged_startup_error(exc: Exception) -> None:
    """Show a native error dialog for failures in a windowed Windows build."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    import ctypes

    message = (
        "Data Cleaning Assistant could not start.\n\n"
        f"{exc}\n\n"
        "Microsoft Edge WebView2 Runtime is required on Windows. "
        "Windows 11 normally includes it."
    )
    ctypes.windll.user32.MessageBoxW(
        None,
        message,
        "Data Cleaning Assistant",
        0x10,
    )


def main(argv: list[str] | None = None) -> None:
    """Launch the desktop application with user-local session storage."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--self-test"]:
        _desktop_self_test()
        return
    if args:
        raise SystemExit(f"Unknown desktop argument: {args[0]}")

    try:
        run_local_desktop(
            data_root=Path.home() / ".data-cleaner" / "desktop" / "sessions",
        )
    except Exception as exc:
        _show_packaged_startup_error(exc)
        if sys.platform == "win32" and getattr(sys, "frozen", False):
            raise SystemExit(1) from None
        raise
