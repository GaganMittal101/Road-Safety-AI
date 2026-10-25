from __future__ import annotations

"""Repair common UTF-8-as-Windows-1252 mojibake in Road Safety AI source files.

Run this script from the repository root, where app/app.py exists:
    py repair_mojibake.py
Original files are backed up before changes.
"""

from pathlib import Path
import sys

MARKERS = (
    "ðŸ", "ð\u009f", "â€", "âš", "ï¸", "Ã", "âœ", "â†", "â€¢",
    "â€“", "â€”", "â€™", "â€œ", "â€", "Â ",
)
EXCLUDED_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules"}


def marker_count(text: str) -> int:
    return sum(text.count(marker) for marker in set(MARKERS))


def recover_once(line: str) -> str | None:
    """Reverse a common CP1252/Latin-1 decoding mistake on one source line."""
    raw = bytearray()
    for char in line:
        code = ord(char)
        if code <= 0xFF:
            raw.append(code)
            continue
        try:
            raw.extend(char.encode("cp1252"))
        except UnicodeEncodeError:
            # Preserve legitimate characters that cannot be represented in CP1252.
            raw.extend(char.encode("utf-8"))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def repair_line(line: str) -> tuple[str, bool]:
    current = line
    changed = False
    for _ in range(3):
        if marker_count(current) == 0:
            break
        candidate = recover_once(current)
        if candidate is None or candidate == current:
            break
        if marker_count(candidate) < marker_count(current):
            current = candidate
            changed = True
        else:
            break
    return current, changed


def main() -> int:
    root = Path.cwd().resolve()
    expected_app = root / "app" / "app.py"
    if not expected_app.is_file():
        print("ERROR: app/app.py was not found in:", root)
        print("Open CMD in the Road-Safety-AI project root and run this script again.")
        return 2

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".toml"}:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        candidates.append(path)

    total_changed = 0
    files_changed = 0
    print("Project root:", root)
    print("Checking Python and Streamlit config files for corrupted text...\n")

    for path in candidates:
        try:
            original_bytes = path.read_bytes()
            original = original_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        repaired_lines: list[str] = []
        changed_lines = 0
        for line in original.splitlines(keepends=True):
            fixed, changed = repair_line(line)
            repaired_lines.append(fixed)
            changed_lines += int(changed)
        repaired = "".join(repaired_lines)

        if repaired != original and changed_lines:
            backup = path.with_name(path.name + ".pre_encoding_fix.bak")
            if not backup.exists():
                backup.write_bytes(original_bytes)
            path.write_text(repaired, encoding="utf-8", newline="")
            files_changed += 1
            total_changed += changed_lines
            print(f"FIXED: {path.relative_to(root)} ({changed_lines} line(s)); backup: {backup.name}")

    if files_changed:
        print(f"\nDone: repaired {total_changed} line(s) across {files_changed} file(s).")
        print("Backups were created next to the changed files. Restart Streamlit now.")
    else:
        text = expected_app.read_text(encoding="utf-8")
        print("No mojibake markers were found in Python/config files under this folder.")
        print("Active app path expected:", expected_app)
        print("app/app.py marker check:", "PRESENT" if marker_count(text) else "not present")
        print("If the browser still shows damaged labels, Streamlit is probably running a different project copy.")
        print("Stop Streamlit (Ctrl+C), cd to this folder, then run:")
        print(f'  py -m streamlit run "{expected_app}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
