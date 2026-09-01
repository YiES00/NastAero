# QApplication 부트스트랩 — GUI 의존성 확인 후 메인 윈도우를 띄우는 진입 함수
from __future__ import annotations

import sys

_MISSING_DEPS_MSG = """\
ASCENT-Load GUI requires optional dependencies. Install with:

    cd solver && pip install -e ".[gui]"

(missing: {missing})
"""


def main(argv: list[str] | None = None) -> int:
    """Launch the ASCENT-Load GUI. Optional argv[1] = .bdf or .aload to open."""
    argv = list(sys.argv if argv is None else argv)

    missing = []
    try:
        import qtpy  # noqa: F401
    except ImportError:
        missing.append("qtpy + a Qt binding (pyside6)")
    try:
        import pyvistaqt  # noqa: F401
    except ImportError:
        missing.append("pyvistaqt")
    if missing:
        sys.stderr.write(_MISSING_DEPS_MSG.format(missing=", ".join(missing)))
        return 1

    from qtpy.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(argv)
    app.setApplicationName("ASCENT-Load")
    window = MainWindow()
    window.show()
    if len(argv) > 1:
        window.open_path(argv[1])
    return app.exec_() if hasattr(app, "exec_") else app.exec()
