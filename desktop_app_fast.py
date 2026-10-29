# 本文件为本仓库新增，不属于上游项目 HivisionIDPhotos。
# 许可遵循 Apache-2.0（见 LICENSE 与 NOTICE）。

import argparse
import os
import threading
import time
import sys
import socket

if sys.platform == "win32":
    import ctypes

    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 0)
        ctypes.windll.shell32.Shell_NotifyIcon(2, None)

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 全局变量
server_ready = threading.Event()
server_port = None
demo_instance = None  # 用于保存Gradio demo实例，以便停止服务器


def find_free_port(start_port=7860, max_port=7870):
    """查找可用端口"""
    for port in range(start_port, max_port + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result != 0:  # 端口未被占用
                return port
        except:
            pass
    return 7860  # 如果都占用了，返回默认端口


def start_gradio_server():
    """在后台启动Gradio服务器"""
    global server_port, demo_instance

    try:
        server_port = find_free_port()

        from demo.processor import IDPhotoProcessor
        from demo.ui import create_ui
        from hivision.creator.choose_handler import HUMAN_MATTING_MODELS

        HUMAN_MATTING_MODELS_EXIST = [
            os.path.splitext(file)[0]
            for file in os.listdir(os.path.join(root_dir, "hivision/creator/weights"))
            if file.endswith(".onnx") or file.endswith(".mnn")
        ]
        HUMAN_MATTING_MODELS_CHOICE = [
            model
            for model in HUMAN_MATTING_MODELS
            if model in HUMAN_MATTING_MODELS_EXIST
        ]

        FACE_DETECT_MODELS = ["face++ (联网Online API)", "mtcnn"]
        FACE_DETECT_MODELS_EXPAND = (
            ["retinaface-resnet50"]
            if os.path.exists(
                os.path.join(
                    root_dir,
                    "hivision/creator/retinaface/weights/retinaface-resnet50.onnx",
                )
            )
            else []
        )
        FACE_DETECT_MODELS_CHOICE = FACE_DETECT_MODELS + FACE_DETECT_MODELS_EXPAND

        LANGUAGE = ["zh", "en", "ko", "ja"]

        processor = IDPhotoProcessor()

        demo_instance = create_ui(
            processor,
            root_dir,
            HUMAN_MATTING_MODELS_CHOICE,
            FACE_DETECT_MODELS_CHOICE,
            LANGUAGE,
        )

        if os.getenv("RUN_MODE") == "beast":
            print("[Beast mode activated.] 已开启野兽模式。")

        demo_instance.launch(
            server_name="127.0.0.1",
            server_port=server_port,
            favicon_path=os.path.join(root_dir, "assets/hivision_logo.png"),
            inbrowser=False,
            show_error=True,
        )
    except Exception as e:
        print(f"服务器启动错误: {e}")
        import traceback

        traceback.print_exc()


def on_window_closed():
    """窗口关闭时的回调函数，用于停止Gradio服务器并终止进程"""
    global demo_instance
    print("窗口已关闭，正在停止服务器...")

    if demo_instance is not None:
        try:
            demo_instance.close()
            print("Gradio服务器已停止")
        except Exception as e:
            print(f"停止服务器时发生错误: {e}")

    import threading

    for thread in threading.enumerate():
        if thread is not threading.main_thread():
            try:
                thread.join(timeout=0.5)
            except:
                pass

    if sys.platform == "win32":
        ctypes.windll.kernel32.TerminateProcess(
            ctypes.windll.kernel32.GetCurrentProcess(), 0
        )
    else:
        os._exit(0)


def wait_for_server(timeout=60):
    """等待服务器启动"""
    global server_port

    start_time = time.time()
    check_interval = 0.1

    while server_port is None and time.time() - start_time < timeout:
        time.sleep(0.05)

    if server_port is None:
        return False

    url = f"http://127.0.0.1:{server_port}"

    while time.time() - start_time < timeout:
        try:
            import urllib.request

            urllib.request.urlopen(url, timeout=0.5)
            server_ready.set()
            return True
        except:
            time.sleep(check_interval)

    return False


def create_desktop_window():
    """创建桌面应用窗口 - 在主线程中运行"""
    import webview
    import ctypes

    global server_port

    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    localization = {
        "global.quitConfirmation": "确定要关闭应用吗？",
        "global.saveFile": "保存文件",
        "global.openFile": "打开文件",
    }

    window = webview.create_window(
        title="HivisionIDPhotos - 证件照制作",
        url=f"http://127.0.0.1:{server_port}",
        width=screen_width,
        height=screen_height,
        min_size=(1200, 700),
        confirm_close=True,
        text_select=True,
    )

    # 设置窗口关闭回调
    def on_closed(window):
        on_window_closed()

    window.events.closed += on_closed

    # 确保窗口在前台显示的函数
    def bring_window_to_front():
        # 激活窗口并置于前台
        # 使用SetForegroundWindow和相关API确保窗口在前台
        ctypes.windll.user32.keybd_event(
            0x1B, 0, 0, 0
        )  # 发送ESC键，有时能帮助窗口获取焦点
        ctypes.windll.user32.keybd_event(0x1B, 0, 2, 0)  # 释放ESC键

        # 尝试获取当前活动窗口并激活
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
            ctypes.windll.user32.SetFocus(hwnd)

    # 启动webview后立即尝试将窗口置于前台
    import threading

    def bring_to_front_after_delay():
        time.sleep(1)  # 延迟1秒，确保窗口已经创建
        bring_window_to_front()

    # 在后台线程中执行，避免阻塞主线程
    threading.Thread(target=bring_to_front_after_delay, daemon=True).start()

    # 启动webview - 这必须在主线程中运行
    webview.start(
        localization=localization,
        debug=False,
    )


if __name__ == "__main__":
    print("正在启动证件照制作工具，请稍候...")

    # 在后台线程启动Gradio服务器
    server_thread = threading.Thread(target=start_gradio_server, daemon=True)
    server_thread.start()

    # 等待服务器启动
    if wait_for_server(timeout=60):
        print(f"服务器启动成功，正在打开窗口...")
        # 在主线程中创建桌面窗口（webview必须在主线程运行）
        create_desktop_window()
    else:
        print("服务器启动超时，请检查是否有其他程序占用了端口")
        input("按回车键退出...")
