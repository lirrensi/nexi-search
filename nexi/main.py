"""Entry point for NEXI."""

import sys

# Fix Windows encoding issues BEFORE any other imports
# This must be done before importing anything that might print to stdout/stderr
if sys.platform == "win32":
    import io

    # Force UTF-8 encoding for stdout/stderr
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    # Also set the console code page to UTF-8 (65001)
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)  # Input code page
        kernel32.SetConsoleOutputCP(65001)  # Output code page
    except Exception:
        pass  # Fail silently if we can't set code page

from nexi.cli import main
from nexi.errors import handle_error

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_error(str(e))
