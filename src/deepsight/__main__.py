"""CLI entry points: ``deepsight serve`` and ``deepsight doctor``."""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .server import create_app


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


def _doctor(args: argparse.Namespace) -> None:
    import httpx

    settings = get_settings()
    checks: list[tuple[str, bool, str]] = []

    # reasoning backend
    try:
        r = httpx.get(
            f"{settings.reasoning_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.reasoning_key}"}
            if settings.reasoning_key
            else {},
            timeout=15,
        )
        checks.append(("reasoning backend", r.status_code == 200, f"http {r.status_code}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("reasoning backend", False, str(exc)[:80]))

    # vision backend
    try:
        r = httpx.get(f"{settings.vision_base_url.rstrip('/')}/api/tags", timeout=15)
        ok = r.status_code == 200
        checks.append(("vision backend (ollama)", ok, f"http {r.status_code}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("vision backend (ollama)", False, str(exc)[:80]))

    all_ok = True
    for name, ok, detail in checks:
        print(f"{'✓' if ok else '✗'} {name}: {detail}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepsight",
        description="OpenAI-compatible vision proxy for text-only LLMs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--host", default=None, help="bind host (default: from env/settings)")
    serve.add_argument(
        "--port", type=int, default=None, help="bind port (default: from env/settings)"
    )
    serve.set_defaults(func=_serve)

    doc = sub.add_parser("doctor", help="check reasoning + vision backend connectivity")
    doc.set_defaults(func=_doctor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
