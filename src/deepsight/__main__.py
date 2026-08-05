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


def _setup(args: argparse.Namespace) -> None:  # noqa: ARG001
    """One-command setup: compile binary (macOS) or verify env (Windows)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if sys.platform == "darwin":
        print("compiling vision_eyes (Apple Vision binary)...")
        sdk = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk"
        swift_src = os.path.join(repo_root, "scripts", "vision_eyes.swift")
        bin_out = os.path.join(repo_root, "scripts", "vision_eyes")
        if not os.path.exists(swift_src):
            print(f"✗ source not found: {swift_src}")
            sys.exit(1)
        r = subprocess.run(
            ["swiftc", "-target", "arm64-apple-macos14", "-o", bin_out, swift_src],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "SDKROOT": sdk},
        )
        if r.returncode != 0:
            print(f"✗ compile failed: {r.stderr.strip()[:200]}")
            sys.exit(1)
        print(f"✓ compiled: {bin_out}")
        print(f"  export DEEPSIGHT_VISION_BIN={bin_out}")
    elif sys.platform == "win32":
        print("Windows detected — no binary to compile.")
        print("Optional: install Tesseract OCR for full text capabilities:")
        print("  winget install UB-Mannheim.TesseractOCR")
    else:
        print(f"unsupported platform: {sys.platform}")
        sys.exit(1)

    # Suggest .env setup
    env_path = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"# DeepSight configuration — see README for all options\n")
            if sys.platform == "darwin":
                f.write(f"DEEPSIGHT_VISION_BIN={os.path.join(repo_root, 'scripts', 'vision_eyes')}\n")
        print(f"✓ created .env with defaults")
    else:
        print(f"  .env already exists (skipped)")

    print()
    print("next: uv run deepsight doctor")
    print("      uv run deepsight describe path/to/image.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepsight",
        description="Device-native vision toolkit for macOS and Windows.",
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

    setup = sub.add_parser("setup", help="one-command setup: compile binary, create .env, verify")
    setup.set_defaults(func=_setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
