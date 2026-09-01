# 해석 실행 패널 — 설정(병렬/스플라인/로그레벨) + QProcess로 CLI 솔버 실행·중단·로그 스트리밍
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("ascent_load.gui")


def list_child_pids(pid: int) -> List[int]:
    """POSIX에서 pid의 직계 자식 PID 목록 (pgrep -P). 실패 시 빈 목록."""
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)],
                             capture_output=True, text=True, timeout=5)
        return [int(x) for x in out.stdout.split()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def kill_process_tree(pid: int) -> int:
    """솔버 프로세스와 그 병렬 워커들을 모두 종료한다. 죽인 자식 수 반환.

    ProcessPoolExecutor 워커는 메인의 자식 프로세스라 메인만 kill하면
    고아로 살아남아 계산을 계속한다. 자식 PID를 먼저 확보하고, 메인을
    먼저 죽여(죽은 워커를 풀이 재생성하지 못하게) 그 다음 자식들을
    정리한다. Windows는 taskkill /T가 트리 전체를 처리한다.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
        return 0

    kids = list_child_pids(pid)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    killed = 0
    for k in kids:
        try:
            os.kill(k, signal.SIGKILL)
            killed += 1
        except OSError:
            pass  # 이미 종료됨
    return killed

from qtpy.QtCore import QProcess, Signal
from qtpy.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)


def build_solver_command(bdf_path: str, parallel: int = 0,
                         spline_slope: str = "surface",
                         log_level: str = "INFO",
                         python_exe: Optional[str] = None) -> List[str]:
    """CLI 솔버 서브프로세스 커맨드를 구성한다 (--save 포함)."""
    exe = python_exe or sys.executable
    cmd = [exe, "-m", "ascent_load", bdf_path, "--save",
           "--log-level", log_level]
    if parallel != 0:
        cmd += ["--parallel", str(parallel)]
    if spline_slope != "surface":
        cmd += ["--spline-slope", spline_slope]
    return cmd


def naero_path_for(bdf_path: str) -> str:
    """--save가 만드는 .aload 경로 (<입력>.aload)."""
    return str(Path(bdf_path).with_suffix(".aload"))


class RunPanel(QWidget):
    """해석 설정과 실행/중단 버튼. 로그 라인과 완료 신호를 발신한다."""

    log_line = Signal(str)
    run_started = Signal()
    run_finished = Signal(str)  # 성공 시 .aload 경로, 실패 시 빈 문자열

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bdf_path: Optional[str] = None
        self._process: Optional[QProcess] = None

        form = QFormLayout()
        self._sol_label = QLabel("-")
        form.addRow("SOL", self._sol_label)

        self._parallel = QSpinBox()
        self._parallel.setRange(-1, 64)
        self._parallel.setValue(0)
        self._parallel.setToolTip("0=순차, -1=자동(cpu-1), N=워커 수 (SOL 144)")
        form.addRow("Parallel workers", self._parallel)

        self._spline = QComboBox()
        self._spline.addItems(["surface", "rotation"])
        self._spline.setToolTip("SOL 144 normalwash slope 구성 방법")
        form.addRow("Spline slope", self._spline)

        self._log_level = QComboBox()
        self._log_level.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        form.addRow("Log level", self._log_level)

        self._run_btn = QPushButton("Run")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._run_btn.clicked.connect(self.start_run)
        self._stop_btn.clicked.connect(self.stop_run)
        buttons = QHBoxLayout()
        buttons.addWidget(self._run_btn)
        buttons.addWidget(self._stop_btn)

        self._status = QLabel("모델을 열면 실행할 수 있습니다")
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)
        layout.addStretch()
        self._run_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # 외부 API
    # ------------------------------------------------------------------
    def set_bdf(self, bdf_path: Optional[str], sol: Optional[int]) -> None:
        """열린 BDF 경로/SOL을 반영한다. .aload 로드 시 bdf_path=None."""
        self._bdf_path = bdf_path
        self._sol_label.setText(str(sol) if sol else "-")
        runnable = bdf_path is not None
        self._run_btn.setEnabled(runnable and self._process is None)
        self._status.setText("준비됨" if runnable
                             else "결과 파일(보기 전용) — 실행하려면 .bdf를 여세요")

    def is_running(self) -> bool:
        return self._process is not None

    # ------------------------------------------------------------------
    # 실행/중단
    # ------------------------------------------------------------------
    def start_run(self) -> None:
        if self._bdf_path is None or self._process is not None:
            return
        cmd = build_solver_command(
            self._bdf_path,
            parallel=self._parallel.value(),
            spline_slope=self._spline.currentText(),
            log_level=self._log_level.currentText(),
        )
        self.log_line.emit("$ " + " ".join(cmd))
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        self._process = proc
        proc.start(cmd[0], cmd[1:])
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status.setText("해석 실행 중…")
        self.run_started.emit()

    def stop_run(self) -> None:
        if self._process is None:
            return
        self._status.setText("중단 요청됨…")
        pid = int(self._process.processId())
        if pid > 0:
            killed = kill_process_tree(pid)
            if killed:
                self.log_line.emit(
                    f"해석 중단 — 병렬 워커 {killed}개 함께 종료")
        else:
            self._process.kill()

    def _on_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        for line in data.splitlines():
            self.log_line.emit(line)

    def _on_finished(self, exit_code: int, _exit_status=None) -> None:
        bdf_path = self._bdf_path
        self._process = None
        self._run_btn.setEnabled(bdf_path is not None)
        self._stop_btn.setEnabled(False)
        if exit_code == 0 and bdf_path:
            self._status.setText("완료 — 결과 로드됨")
            self.run_finished.emit(naero_path_for(bdf_path))
        else:
            self._status.setText(f"종료 코드 {exit_code}")
            self.run_finished.emit("")
