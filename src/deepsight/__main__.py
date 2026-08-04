"""CLI entry points: ``deepsight describe`` (device-native, no server), ``deepsight doctor``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .config import get_settings


def _describe(args: argparse.Namespace) -> None:
    """Describe an image with the device's own vision capabilities.

    Shells the Apple Vision ``vision_eyes`` binary directly: OCR, scene
    classification, saliency, face/human/rectangle detection. No server, no
    network, no model downloads, zero tokens. Fast and free.
    """
    settings = get_settings()
    bin_path = settings.vision_bin
    if not os.path.exists(bin_path):
        print(f"vision binary not found: {bin_path}", file=sys.stderr)
        sys.exit(1)
    proc = subprocess.run(
        [bin_path, args.image],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:300]
        print(f"vision_eyes error ({proc.returncode}): {detail}", file=sys.stderr)
        sys.exit(proc.returncode)

    # Structure the raw output into labeled sections.
    sections: list[str] = []
    ocr: list[str] = []
    scene: list[str] = []
    counts: list[str] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if line.startswith("  "):
            ocr.append(stripped)
        elif stripped.startswith("scene:"):
            scene.append(stripped.removeprefix("scene:").strip())
        elif stripped.startswith(("faces:", "humans:", "rectangles:")):
            counts.append(stripped)
    if ocr:
        sections.append("OCR text:\n" + "\n".join(f"  {t}" for t in ocr))
    if scene:
        sections.append("Scene: " + scene[0])
    if counts:
        sections.append("\n".join(counts))
    print("\n".join(sections).strip() or "(no text or scene detected)")


def _doctor(args: argparse.Namespace) -> None:
    import httpx

    settings = get_settings()
    checks: list[tuple[str, bool, str]] = []

    # native vision binary
    bin_path = settings.vision_bin
    checks.append(("vision binary", os.path.exists(bin_path), bin_path))

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

    all_ok = True
    for name, ok, detail in checks:
        print(f"{'✓' if ok else '✗'} {name}: {detail}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepsight",
        description="Device-native vision: describe images with on-device Apple Vision. "
        "No server required.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser(
        "describe",
        help="describe an image with the device's own vision (no server, no tokens)",
    )
    describe.add_argument("image", help="path to the image file")
    describe.set_defaults(func=_describe)

    doc = sub.add_parser("doctor", help="check native vision binary + reasoning connectivity")
    doc.set_defaults(func=_doctor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
