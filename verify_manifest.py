"""Compare repository files with the release integrity manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest.items():
        file = ROOT / name
        if not file.is_file():
            failures.append(name + ": missing")
        elif hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            failures.append(name + ": changed")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(manifest)} files")


if __name__ == "__main__":
    main()
