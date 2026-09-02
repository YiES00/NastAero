# 배포 버전과 코드 내 버전 문자열이 갈리지 않는지 고정하는 시험
from __future__ import annotations

import re
from pathlib import Path

import ascent_load
from ascent_load.output import result_io

_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m, "pyproject.toml에 version 항목이 없다"
    return m.group(1)


def test_package_version_matches_pyproject():
    assert ascent_load.__version__ == _pyproject_version()


def test_citation_version_matches_pyproject():
    cff = _ROOT.parent / "CITATION.cff"
    if not cff.exists():
        return
    m = re.search(r'^version:\s*(\S+)', cff.read_text(encoding="utf-8"), re.M)
    assert m, "CITATION.cff에 version 항목이 없다"
    assert m.group(1).strip('"\'') == _pyproject_version()


def test_result_io_reuses_package_version():
    # 아카이브 메타데이터에 찍히는 값이 패키지 버전과 갈리면 안 된다
    assert result_io.__version__ == ascent_load.__version__


def test_format_version_is_independent_of_package_version():
    # 아카이브 스키마 버전은 배포 버전과 별개인 정수다
    assert isinstance(result_io.FORMAT_VERSION, int)
