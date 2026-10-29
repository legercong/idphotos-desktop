import sys
import os
import json
import time
import traceback

LOG_FILE = os.path.join(os.path.dirname(__file__), "print_debug.log")

def log(msg):
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%H:%M:%S')} - {msg}\n")
    except:
        pass

def main():
    log("=== 开始 ===")
    
    if len(sys.argv) < 2:
        log("参数错误")
        sys.exit(1)

    config_file = sys.argv[1]
    log(f"配置文件: {config_file}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        img_path = config["img_path"]
        printer_name = config["printer_name"]
        log(f"图片: {img_path}, 打印机: {printer_name}")

        time.sleep(0.3)

        log("导入PyQt5...")
        os.environ['QT_QPA_PLATFORM'] = 'windows'
        
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from PyQt5.QtPrintSupport import QPrinter, QPrintPreviewDialog
        from PyQt5.QtGui import QPixmap, QImage, QPainter
        from PyQt5.QtCore import Qt
        log("PyQt5导入成功")

        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        log("QApplication创建成功")

        # 加载图片
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert("RGBA")
        qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        log(f"图片加载: {img.width}x{img.height}")

        # 创建打印机
        printer = QPrinter(QPrinter.HighResolution)
        printer.setPrinterName(printer_name)
        
        # 默认横向（Landscape）
        printer.setOrientation(QPrinter.Landscape)
        
        # 尝试设置页边距为0
        # Qt setPageMargins: left, top, right, bottom, units
        try:
            printer.setPageMargins(0.0, 0.0, 0.0, 0.0, QPrinter.Millimeter)
            log("页边距已设置为0mm")
        except Exception as e:
            log(f"设置setPageMargins失败: {e}")
            try:
                # 尝试另一种方式
                from PyQt5.QtCore import QMarginsF
                margins = QMarginsF(0, 0, 0, 0)
                printer.setPageMargins(margins, QPrinter.Millimeter)
                log("页边距已设置为0mm (方式2)")
            except Exception as e2:
                log(f"设置页边距失败2: {e2}")
        
        log(f"打印机方向: 横向(Landscape)")

        # 创建打印预览对话框
        preview = QPrintPreviewDialog(printer)
        preview.setWindowTitle("打印预览 - HivisionIDPhotos")
        preview.resize(900, 700)
        
        # 定义绘制函数
        def handle_paint_requested(printer_to_draw):
            """当预览需要重绘时调用（包括用户点击打印时）"""
            log("绘制请求触发")
            
            # 获取页面尺寸
            page_rect = printer_to_draw.pageRect(QPrinter.DevicePixel)
            
            # 尝试获取可打印区域
            try:
                paint_rect = printer_to_draw.paintRect(QPrinter.DevicePixel)
                paint_w = int(paint_rect.width())
                paint_h = int(paint_rect.height())
                paint_x = int(paint_rect.x())
                paint_y = int(paint_rect.y())
            except:
                # 如果paintRect不可用，使用pageRect并减去一些边距
                paint_w = int(page_rect.width())
                paint_h = int(page_rect.height())
                paint_x = int(page_rect.x())
                paint_y = int(page_rect.y())
            
            log(f"页面: {int(page_rect.width())}x{int(page_rect.height())}, 可打印: {paint_w}x{paint_h}, 图片: {pixmap.width()}x{pixmap.height()}")
            
            painter = QPainter(printer_to_draw)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            
            # 计算缩放：保持宽高比，填满可打印区域
            scale_x = paint_w / pixmap.width()
            scale_y = paint_h / pixmap.height()
            scale = min(scale_x, scale_y)
            
            draw_w = int(pixmap.width() * scale)
            draw_h = int(pixmap.height() * scale)
            
            # 在可打印区域内居中绘制
            x = paint_x + (paint_w - draw_w) // 2
            y = paint_y + (paint_h - draw_h) // 2
            
            log(f"绘制: 位置({x},{y}), 尺寸{draw_w}x{draw_h}, 缩放{scale:.3f}")
            
            painter.drawPixmap(x, y, draw_w, draw_h, pixmap)
            painter.end()
        
        # 连接paintRequested信号
        preview.paintRequested.connect(handle_paint_requested)
        
        log("显示预览对话框...")
        
        # 使用exec_显示模态对话框
        # 用户点击预览窗口的"打印"按钮后，系统会自动弹出打印对话框
        # 打印完成后，对话框关闭
        preview.exec_()
        
        log("预览对话框已关闭")

        log("=== 结束 ===")

    except Exception as e:
        error = traceback.format_exc()
        log(f"错误: {error}")
        try:
            QMessageBox.critical(None, "打印错误", str(e))
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
