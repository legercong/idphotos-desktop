# -*- coding: utf-8 -*-
# 本文件为本仓库新增，不属于上游项目 HivisionIDPhotos。
# 许可遵循 Apache-2.0（见 LICENSE 与 NOTICE）。

"""
HivisionIDPhotos 启动器
"""

import os
import sys
import subprocess
import time
import socket
import threading
from pathlib import Path

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

try:
    from PyQt5.QtWidgets import QApplication, QSplashScreen
    from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor
    from PyQt5.QtCore import Qt, QTimer
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False


class AnimatedSplashScreen(QSplashScreen):
    FRAME_COUNT = 36
    SPEED_DEG_PER_SEC = 300

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)

        # 静态内容（背景 + 文字）只渲染一次，逐帧只是贴图
        base = pixmap.copy()
        painter = QPainter(base)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_labels(painter, base)
        painter.end()
        self.setPixmap(base)

        self._frames = self._render_spinner_frames()
        self._current_index = -1
        self._start_time = time.time()

    def _draw_labels(self, painter, pixmap):
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Microsoft YaHei", 12)
        painter.setFont(font)
        painter.drawText(
            pixmap.rect(), Qt.AlignCenter | Qt.AlignBottom, "正在启动证件照制作工具"
        )

        small_font = QFont("Microsoft YaHei", 9)
        painter.setFont(small_font)
        painter.setPen(QColor(180, 180, 180))
        version_text = "HivisionIDPhotos v1.3.1"
        painter.drawText(
            pixmap.rect().adjusted(0, 0, -20, -20),
            Qt.AlignRight | Qt.AlignBottom,
            version_text,
        )

    def _render_spinner_frames(self):
        box = 88
        radius = 35
        frames = []
        for i in range(self.FRAME_COUNT):
            frame = QPixmap(box, box)
            frame.fill(Qt.transparent)
            painter = QPainter(frame)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.translate(box / 2, box / 2)
            painter.rotate(i * 360 / self.FRAME_COUNT)
            for k in range(12):
                painter.save()
                painter.rotate(k * 30)
                if k == 0:
                    painter.setPen(QColor(100, 180, 255))
                    painter.setBrush(QColor(100, 180, 255))
                    painter.drawRect(-2, -radius, 4, 15)
                else:
                    alpha = int(255 * (1 - k / 12))
                    painter.setPen(QColor(100, 180, 255, alpha))
                    painter.setBrush(QColor(100, 180, 255, alpha))
                    painter.drawRect(-1, -radius, 2, 10)
                painter.restore()
            painter.end()
            frames.append(frame)
        return frames

    def drawContents(self, painter: QPainter):
        elapsed = time.time() - self._start_time
        index = int(elapsed * self.SPEED_DEG_PER_SEC / (360 / self.FRAME_COUNT)) % self.FRAME_COUNT
        self._current_index = index
        frame = self._frames[index]
        x = self.width() // 2 - frame.width() // 2
        y = self.height() // 2 + 20 - frame.height() // 2
        painter.drawPixmap(x, y, frame)

    def tick(self):
        # 时间驱动的帧索引，只有帧变化时才调度重绘（update 会自动合并，不阻塞）
        elapsed = time.time() - self._start_time
        index = int(elapsed * self.SPEED_DEG_PER_SEC / (360 / self.FRAME_COUNT)) % self.FRAME_COUNT
        if index != self._current_index:
            self.update()


def create_default_pixmap(width=600, height=400):
    from PyQt5.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    gradient = QLinearGradient(0, 0, width, height)
    gradient.setColorAt(0, QColor(30, 30, 40))
    gradient.setColorAt(1, QColor(50, 50, 70))
    painter.fillRect(pixmap.rect(), gradient)

    painter.setPen(QColor(100, 180, 255, 100))
    painter.drawLine(50, height // 2 - 50, width - 50, height // 2 - 50)
    painter.drawLine(50, height // 2 + 50, width - 50, height // 2 + 50)

    painter.setPen(QColor(255, 255, 255))
    font = QFont("Microsoft YaHei", 28, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "HivisionIDPhotos")

    small_font = QFont("Microsoft YaHei", 12)
    painter.setFont(small_font)
    painter.setPen(QColor(180, 180, 180))
    painter.drawText(
        pixmap.rect().adjusted(0, 60, 0, 0), Qt.AlignCenter, "智能证件照制作工具"
    )

    painter.end()
    return pixmap


def start_desktop_app():
    current_dir = Path(__file__).parent.absolute()

    search_paths = [
        current_dir,
        current_dir.parent,
        current_dir.parent.parent,
        current_dir.parent.parent.parent
    ]

    script_path = None
    for path in search_paths:
        temp_path = path / "desktop_app_fast.py"
        if temp_path.exists():
            script_path = temp_path
            current_dir = path
            break

    if not script_path:
        return

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        if sys.platform == "win32":
            python_exe = sys.executable
            pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")

            if os.path.exists(pythonw_exe):
                subprocess.Popen(
                    [pythonw_exe, str(script_path)],
                    cwd=str(current_dir),
                    env=env,
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    [python_exe, str(script_path)],
                    cwd=str(current_dir),
                    env=env,
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
        else:
            subprocess.Popen(
                [python_exe, str(script_path)],
                cwd=str(current_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as e:
        log_path = os.path.join(current_dir, "launcher_error.log")
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {str(e)}\n")


def check_window_exists():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        found = [False]

        def callback(hwnd, extra):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    title = buffer.value
                    if "HivisionIDPhotos" in title:
                        found[0] = True
                        return False
            return True

        user32.EnumWindows(
            ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(callback),
            None
        )
        return found[0]
    except Exception:
        return False


def check_server_ready():
    for port in range(7860, 7871):
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=0.3)
            return True
        except:
            continue
    return False


def run_with_pyqt5():
    try:
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        pixmap = create_default_pixmap()
        splash = AnimatedSplashScreen(pixmap)
        splash.show()

        threading.Thread(target=start_desktop_app, daemon=True).start()

        # 就绪探测放到后台线程，避免网络/窗口枚举阻塞动画主线程导致卡顿
        ready_event = threading.Event()

        def watch_ready():
            while not ready_event.is_set():
                if check_server_ready() or check_window_exists():
                    ready_event.set()
                    return
                time.sleep(0.25)

        threading.Thread(target=watch_ready, daemon=True).start()

        app_started = False
        minimum_show_time = time.time() + 3

        def update_animation():
            nonlocal app_started

            if app_started:
                return

            splash.tick()

            if time.time() >= minimum_show_time and ready_event.is_set():
                app_started = True
                timer.stop()
                splash.close()
                QTimer.singleShot(200, app.quit)

        timer = QTimer()
        timer.timeout.connect(update_animation)
        timer.start(16)

        sys.exit(app.exec_())
    except Exception as e:
        log_path = os.path.join(Path(__file__).parent, "launcher_error.log")
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {str(e)}\n")
        start_desktop_app()


def run_simple():
    start_desktop_app()


def main():
    if PYQT5_AVAILABLE:
        try:
            run_with_pyqt5()
        except Exception as e:
            print(f"PyQt5 启动失败: {e}")
            run_simple()
    else:
        run_simple()


if __name__ == "__main__":
    if "pythonw.exe" not in sys.executable.lower():
        python_exe = sys.executable
        pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
        if os.path.exists(pythonw_exe):
            subprocess.Popen([pythonw_exe, __file__], shell=False)
            sys.exit()
    main()