# 本文件相对上游项目 HivisionIDPhotos (https://github.com/Zeyi-Lin/HivisionIDPhotos)
# 有修改，改动范围见 NOTICE 与 CHANGELOG.md。

import gradio as gr
import os
import pathlib
import webbrowser
import tempfile
from demo.locales import LOCALES
from demo.processor import IDPhotoProcessor


def get_printers():
    try:
        import win32print

        printers = [printer[2] for printer in win32print.EnumPrinters(2)]
        # 确保默认打印机存在于列表中
        if "HiTi P525L相纸打印机" not in printers:
            # 如果没有HiTi打印机，尝试找类似的
            hiti_printer = next(
                (p for p in printers if "HiTi" in p or "hiti" in p), None
            )
            if not hiti_printer and not printers:
                return ["Microsoft Print to PDF"]
        return printers
    except:
        return ["Microsoft Print to PDF"]


def print_photo(image, printer_name="HiTi P525L相纸打印机"):
    if image is None:
        return "请先生成照片后再打印"

    try:
        from PIL import Image
        import numpy as np
        import sys
        import tempfile
        import os
        import json
        import time
        import subprocess

        # 调试日志
        debug_log = os.path.join(os.path.dirname(__file__), "print_ui_debug.log")

        def log(msg):
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} - {msg}\n")
            except:
                pass

        log(f"print_photo 被调用, 打印机: {printer_name}")

        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        temp_dir = os.path.join(tempfile.gettempdir(), "hivision_print")
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, f"print_{int(time.time())}.png")
        img.save(temp_path, format="PNG")
        log(f"图片保存: {temp_path}")

        print_script_path = os.path.join(os.path.dirname(__file__), "print_worker.py")
        log(f"打印脚本: {print_script_path}")

        config = json.dumps({"img_path": temp_path, "printer_name": printer_name})
        config_file = temp_path + ".json"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config)
        log(f"配置文件: {config_file}")

        python_exe = sys.executable
        pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")

        if os.path.exists(pythonw_exe):
            exe_to_use = pythonw_exe
        else:
            exe_to_use = python_exe
        log(f"使用解释器: {exe_to_use}")

        # 方法1: 直接用subprocess.Popen启动
        # 不使用任何creationflags，让子进程可以创建GUI窗口
        try:
            proc = subprocess.Popen(
                [exe_to_use, print_script_path, config_file], close_fds=True
            )
            log(f"方法1成功, PID: {proc.pid}")
        except Exception as e1:
            log(f"方法1失败: {e1}")
            # 方法2: 使用shell=True
            try:
                proc = subprocess.Popen(
                    f'"{exe_to_use}" "{print_script_path}" "{config_file}"',
                    shell=True,
                    close_fds=True,
                )
                log(f"方法2成功, PID: {proc.pid}")
            except Exception as e2:
                log(f"方法2失败: {e2}")
                raise e1

        return "打印预览窗口已打开，请在预览窗口中完成打印"

    except Exception as e:
        import traceback

        error = traceback.format_exc()
        try:
            debug_log = os.path.join(os.path.dirname(__file__), "print_ui_debug.log")
            with open(debug_log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} - 错误: {error}\n")
        except:
            pass
        return f"打印失败: {str(e)}"


"""
只裁切模式:
1. 如果重新上传了照片，然后点击按钮，第一次会调用不裁切的模式，第二次会调用裁切的模式
"""

# 拖拽替换脚本: 拦截输入框上的 drop 事件，绕过 Gradio 内部的拖放处理，
# 直接把文件塞进隐藏的 <input type=file> 并触发 change（与点击上传同一条路径），
# 保证第二次及以后拖入的照片能直接替换旧照片，而不是变空白。
# 注意: 必须在事件触发时重新查找 input —— Gradio 会在渲染后重建组件 DOM，
# 缓存的节点引用会变成游离节点（这正是原生拖放在生成后失效变空白的原因）。
DROP_REPLACE_JS = """function(){
  if (window.__hvDropInstalled) return;
  window.__hvDropInstalled = true;
  document.addEventListener('drop', function(e){
    var root = e.target && e.target.closest ? e.target.closest('#input_image') : null;
    if (!root) return;
    var files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    var input = root.querySelector('input[type=file]');
    if (!input) return;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    var dt = new DataTransfer();
    for (var i = 0; i < files.length; i++) dt.items.add(files[i]);
    input.value = '';
    input.files = dt.files;
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }, true);
  document.addEventListener('dragover', function(e){
    var root = e.target && e.target.closest ? e.target.closest('#input_image') : null;
    if (root) e.preventDefault();
  }, true);
}"""


def load_description(fp):
    """
    加载title.md文件作为Demo的顶部栏
    """
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def download_image(image, filename):
    """
    下载图片
    """
    if image is None:
        raise gr.Error("请先生成照片后再下载")
    import tempfile
    import os
    from PIL import Image
    import numpy as np

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    img.save(file_path)

    return file_path


def create_ui(
    processor: IDPhotoProcessor,
    root_dir: str,
    human_matting_models: list,
    face_detect_models: list,
    language: list,
):

    # 加载环境变量DEFAULT_LANG, 如果有且在language中，则将DEFAULT_LANG设置为环境变量
    if "DEFAULT_LANG" in os.environ and os.environ["DEFAULT_LANG"] in language:
        DEFAULT_LANG = os.environ["DEFAULT_LANG"]
    else:
        DEFAULT_LANG = language[0]

    DEFAULT_HUMAN_MATTING_MODEL = "hivision_modnet"
    DEFAULT_FACE_DETECT_MODEL = "retinaface-resnet50"

    if DEFAULT_HUMAN_MATTING_MODEL in human_matting_models:
        human_matting_models.remove(DEFAULT_HUMAN_MATTING_MODEL)
        human_matting_models.insert(0, DEFAULT_HUMAN_MATTING_MODEL)

    if DEFAULT_FACE_DETECT_MODEL not in face_detect_models:
        DEFAULT_FACE_DETECT_MODEL = "mtcnn"

    demo = gr.Blocks(title="HivisionIDPhotos")

    with demo:
        gr.HTML(
            """<style>.gradio-image-toolbar, .image__toolbar, [data-testid="image-toolbar"], .absolute.top-2.right-2, button[class*="icon-button"] { display: none !important; }</style>"""
        )
        # 预览窗口按照片真实比例显示：Gradio 默认把图片拉伸成固定的框尺寸，
        # 竖版照片两侧留大片空白、横版照片被横向拉长，与点“预览”后的真实比例不一致。
        gr.HTML(
            """<style>
            #standard_photo_image .image-container, #hd_photo_image .image-container, #layout_photo_image .image-container { width: fit-content !important; max-width: 100%; margin-inline: auto !important; }
            #standard_photo_image .image-frame, #hd_photo_image .image-frame, #layout_photo_image .image-frame { width: auto !important; height: 100% !important; }
            #standard_photo_image img, #hd_photo_image img, #layout_photo_image img { width: auto !important; height: auto !important; max-width: 100%; max-height: 100%; }
            </style>"""
        )
        gr.HTML(load_description(os.path.join(root_dir, "demo/assets/title.md")))
        with gr.Row():
            # ------------------------ 左半边 UI ------------------------
            with gr.Column():
                img_input = gr.Image(height=400, elem_id="input_image")

                with gr.Row():
                    # 语言选择器
                    language_options = gr.Dropdown(
                        choices=language,
                        label="Language",
                        value=DEFAULT_LANG,
                    )

                    face_detect_model_options = gr.Dropdown(
                        choices=face_detect_models,
                        label=LOCALES["face_model"][DEFAULT_LANG]["label"],
                        value=DEFAULT_FACE_DETECT_MODEL,
                    )

                    matting_model_options = gr.Dropdown(
                        choices=human_matting_models,
                        label=LOCALES["matting_model"][DEFAULT_LANG]["label"],
                        value=human_matting_models[0],
                    )

                # TAB1 - 关键参数 ------------------------------------------------
                with gr.Tab(
                    LOCALES["key_param"][DEFAULT_LANG]["label"]
                ) as key_parameter_tab:
                    # 尺寸模式
                    with gr.Row():
                        mode_options = gr.Radio(
                            choices=LOCALES["size_mode"][DEFAULT_LANG]["choices"],
                            label=LOCALES["size_mode"][DEFAULT_LANG]["label"],
                            value=LOCALES["size_mode"][DEFAULT_LANG]["choices"][0],
                            min_width=520,
                        )

                    # 尺寸列表
                    with gr.Row(visible=True) as size_list_row:
                        size_list_options = gr.Dropdown(
                            choices=LOCALES["size_list"][DEFAULT_LANG]["choices"],
                            label=LOCALES["size_list"][DEFAULT_LANG]["label"],
                            value=LOCALES["size_list"][DEFAULT_LANG]["choices"][0],
                            elem_id="size_list",
                        )
                    # 自定义尺寸px
                    with gr.Row(visible=False) as custom_size_px:
                        custom_size_height_px = gr.Number(
                            value=413,
                            label=LOCALES["custom_size_px"][DEFAULT_LANG]["height"],
                            interactive=True,
                        )
                        custom_size_width_px = gr.Number(
                            value=295,
                            label=LOCALES["custom_size_px"][DEFAULT_LANG]["width"],
                            interactive=True,
                        )
                    # 自定义尺寸mm
                    with gr.Row(visible=False) as custom_size_mm:
                        custom_size_height_mm = gr.Number(
                            value=35,
                            label=LOCALES["custom_size_mm"][DEFAULT_LANG]["height"],
                            interactive=True,
                        )
                        custom_size_width_mm = gr.Number(
                            value=25,
                            label=LOCALES["custom_size_mm"][DEFAULT_LANG]["width"],
                            interactive=True,
                        )

                    # 背景颜色
                    color_options = gr.Radio(
                        choices=LOCALES["bg_color"][DEFAULT_LANG]["choices"],
                        label=LOCALES["bg_color"][DEFAULT_LANG]["label"],
                        value=LOCALES["bg_color"][DEFAULT_LANG]["choices"][0],
                    )

                    # 自定义颜色RGB
                    with gr.Row(visible=False) as custom_color_rgb:
                        custom_color_R = gr.Number(
                            value=0, label="R", minimum=0, maximum=255, interactive=True
                        )
                        custom_color_G = gr.Number(
                            value=0, label="G", minimum=0, maximum=255, interactive=True
                        )
                        custom_color_B = gr.Number(
                            value=0, label="B", minimum=0, maximum=255, interactive=True
                        )

                    # 自定义颜色HEX
                    with gr.Row(visible=False) as custom_color_hex:
                        custom_color_hex_value = gr.Text(
                            value="000000", label="Hex", interactive=True
                        )

                    # 渲染模式
                    render_options = gr.Radio(
                        choices=LOCALES["render_mode"][DEFAULT_LANG]["choices"],
                        label=LOCALES["render_mode"][DEFAULT_LANG]["label"],
                        value=LOCALES["render_mode"][DEFAULT_LANG]["choices"][0],
                    )

                    # 裁切模式（无人脸/多人脸时生效）
                    crop_mode_options = gr.Radio(
                        choices=LOCALES["crop_mode"][DEFAULT_LANG]["choices"],
                        label=LOCALES["crop_mode"][DEFAULT_LANG]["label"],
                        value=LOCALES["crop_mode"][DEFAULT_LANG]["choices"][0],
                    )

                    with gr.Row():
                        # 插件模式
                        plugin_options = gr.CheckboxGroup(
                            label=LOCALES["plugin"][DEFAULT_LANG]["label"],
                            choices=LOCALES["plugin"][DEFAULT_LANG]["choices"],
                            interactive=True,
                            value=[
                                LOCALES["plugin"][DEFAULT_LANG]["choices"][0],
                                LOCALES["plugin"][DEFAULT_LANG]["choices"][2],
                            ],
                        )

                # TAB2 - 高级参数 ------------------------------------------------
                with gr.Tab(
                    LOCALES["advance_param"][DEFAULT_LANG]["label"]
                ) as advance_parameter_tab:
                    head_measure_ratio_option = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=0.2,
                        step=0.01,
                        label=LOCALES["head_measure_ratio"][DEFAULT_LANG]["label"],
                        interactive=True,
                    )
                    top_distance_option = gr.Slider(
                        minimum=0.02,
                        maximum=0.5,
                        value=0.12,
                        step=0.01,
                        label=LOCALES["top_distance"][DEFAULT_LANG]["label"],
                        interactive=True,
                    )

                    image_kb_options = gr.Radio(
                        choices=LOCALES["image_kb"][DEFAULT_LANG]["choices"],
                        label=LOCALES["image_kb"][DEFAULT_LANG]["label"],
                        value=LOCALES["image_kb"][DEFAULT_LANG]["choices"][0],
                    )

                    custom_image_kb_size = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        value=50,
                        label=LOCALES["image_kb_size"][DEFAULT_LANG]["label"],
                        interactive=True,
                        visible=False,
                    )

                    image_dpi_options = gr.Radio(
                        choices=LOCALES["image_dpi"][DEFAULT_LANG]["choices"],
                        label=LOCALES["image_dpi"][DEFAULT_LANG]["label"],
                        value=LOCALES["image_dpi"][DEFAULT_LANG]["choices"][0],
                    )
                    custom_image_dpi_size = gr.Slider(
                        minimum=72,
                        maximum=600,
                        value=300,
                        label=LOCALES["image_dpi_size"][DEFAULT_LANG]["label"],
                        interactive=True,
                        visible=False,
                    )

                # TAB3 - 美颜 ------------------------------------------------
                with gr.Tab(
                    LOCALES["beauty_tab"][DEFAULT_LANG]["label"]
                ) as beauty_parameter_tab:
                    # 美白组件
                    whitening_option = gr.Slider(
                        label=LOCALES["whitening_strength"][DEFAULT_LANG]["label"],
                        minimum=0,
                        maximum=15,
                        value=2,
                        step=1,
                        interactive=True,
                    )

                    with gr.Row():
                        # 去阴影组件（阴阳脸/下巴阴影）
                        shadow_option = gr.Slider(
                            label=LOCALES["shadow_strength"][DEFAULT_LANG]["label"],
                            minimum=0,
                            maximum=15,
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        # 去反光组件（出汗/油光）
                        specular_option = gr.Slider(
                            label=LOCALES["specular_strength"][DEFAULT_LANG]["label"],
                            minimum=0,
                            maximum=15,
                            value=0,
                            step=1,
                            interactive=True,
                        )

                    with gr.Row():
                        # 亮度组件
                        brightness_option = gr.Slider(
                            label=LOCALES["brightness_strength"][DEFAULT_LANG]["label"],
                            minimum=-5,
                            maximum=25,
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        # 对比度组件
                        contrast_option = gr.Slider(
                            label=LOCALES["contrast_strength"][DEFAULT_LANG]["label"],
                            minimum=-10,
                            maximum=50,
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        # 饱和度组件
                        saturation_option = gr.Slider(
                            label=LOCALES["saturation_strength"][DEFAULT_LANG]["label"],
                            minimum=-10,
                            maximum=50,
                            value=0,
                            step=1,
                            interactive=True,
                        )

                    # 锐化组件
                    sharpen_option = gr.Slider(
                        label=LOCALES["sharpen_strength"][DEFAULT_LANG]["label"],
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=1,
                        interactive=True,
                    )

                # TAB4 - 水印 ------------------------------------------------
                with gr.Tab(
                    LOCALES["watermark_tab"][DEFAULT_LANG]["label"]
                ) as watermark_parameter_tab:
                    watermark_options = gr.Radio(
                        choices=LOCALES["watermark_switch"][DEFAULT_LANG]["choices"],
                        label=LOCALES["watermark_switch"][DEFAULT_LANG]["label"],
                        value=LOCALES["watermark_switch"][DEFAULT_LANG]["choices"][0],
                    )

                    with gr.Row():
                        watermark_text_options = gr.Text(
                            max_length=20,
                            label=LOCALES["watermark_text"][DEFAULT_LANG]["label"],
                            value=LOCALES["watermark_text"][DEFAULT_LANG]["value"],
                            placeholder=LOCALES["watermark_text"][DEFAULT_LANG][
                                "placeholder"
                            ],
                            interactive=False,
                        )
                        watermark_text_color = gr.ColorPicker(
                            label=LOCALES["watermark_color"][DEFAULT_LANG]["label"],
                            interactive=False,
                            value="#FFFFFF",
                        )

                    watermark_text_size = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=20,
                        label=LOCALES["watermark_size"][DEFAULT_LANG]["label"],
                        interactive=False,
                        step=1,
                    )

                    watermark_text_opacity = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.15,
                        label=LOCALES["watermark_opacity"][DEFAULT_LANG]["label"],
                        interactive=False,
                        step=0.01,
                    )

                    watermark_text_angle = gr.Slider(
                        minimum=0,
                        maximum=360,
                        value=30,
                        label=LOCALES["watermark_angle"][DEFAULT_LANG]["label"],
                        interactive=False,
                        step=1,
                    )

                    watermark_text_space = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=25,
                        label=LOCALES["watermark_space"][DEFAULT_LANG]["label"],
                        interactive=False,
                        step=1,
                    )

                    def update_watermark_text_visibility(choice, language):
                        return [
                            gr.update(
                                interactive=(
                                    choice
                                    == LOCALES["watermark_switch"][language]["choices"][
                                        1
                                    ]
                                )
                            )
                        ] * 6

                    watermark_options.change(
                        fn=update_watermark_text_visibility,
                        inputs=[watermark_options, language_options],
                        outputs=[
                            watermark_text_options,
                            watermark_text_color,
                            watermark_text_size,
                            watermark_text_opacity,
                            watermark_text_angle,
                            watermark_text_space,
                        ],
                    )

                # TAB5 - 打印 ------------------------------------------------
                with gr.Tab(
                    LOCALES["print_tab"][DEFAULT_LANG]["label"]
                ) as print_parameter_tab:
                    print_options = gr.Radio(
                        choices=LOCALES["print_switch"][DEFAULT_LANG]["choices"],
                        label=LOCALES["print_switch"][DEFAULT_LANG]["label"],
                        value=LOCALES["print_switch"][DEFAULT_LANG]["choices"][0],
                        interactive=True,
                    )

                img_but = gr.Button(
                    LOCALES["button"][DEFAULT_LANG]["label"],
                    elem_id="btn",
                    variant="primary",
                )

                example_images = gr.Examples(
                    inputs=[img_input],
                    examples=[
                        [path.as_posix()]
                        for path in sorted(
                            pathlib.Path(os.path.join(root_dir, "demo/images")).rglob(
                                "*.jpg"
                            )
                        )
                    ],
                )

            # ---------------- 右半边 UI ----------------
            with gr.Column():
                notification = gr.Text(
                    label=LOCALES["notification"][DEFAULT_LANG]["label"], visible=False
                )

                # 第一排：标准照和高清照
                with gr.Row():
                    with gr.Column(scale=1):
                        img_output_standard = gr.Image(
                            label=LOCALES["standard_photo"][DEFAULT_LANG]["label"],
                            height=350,
                            format="png",
                            elem_id="standard_photo_image",
                            interactive=False,
                        )
                        standard_print_btn = gr.Button(
                            value="🖨️ 打印", variant="primary", size="sm"
                        )
                        with gr.Row():
                            standard_download_btn = gr.Button(
                                value="⬇ 下载", variant="secondary", size="sm"
                            )
                            standard_fullscreen_btn = gr.Button(
                                value="👁 预览", variant="secondary", size="sm"
                            )
                    with gr.Column(scale=1):
                        img_output_standard_hd = gr.Image(
                            label=LOCALES["hd_photo"][DEFAULT_LANG]["label"],
                            height=350,
                            format="png",
                            elem_id="hd_photo_image",
                            interactive=False,
                        )
                        hd_print_btn = gr.Button(
                            value="🖨️ 打印", variant="primary", size="sm"
                        )
                        with gr.Row():
                            hd_download_btn = gr.Button(
                                value="⬇ 下载", variant="secondary", size="sm"
                            )
                            hd_fullscreen_btn = gr.Button(
                                value="👁 预览", variant="secondary", size="sm"
                            )

                # 第二排：排版照
                with gr.Row():
                    with gr.Column():
                        img_output_layout = gr.Image(
                            label=LOCALES["layout_photo"][DEFAULT_LANG]["label"],
                            height=350,
                            format="png",
                            elem_id="layout_photo_image",
                            interactive=False,
                        )
                        layout_print_btn = gr.Button(
                            value="🖨️ 打印", variant="primary", size="sm"
                        )
                        with gr.Row():
                            layout_download_btn = gr.Button(
                                value="⬇ 下载", variant="secondary", size="sm"
                            )
                            layout_fullscreen_btn = gr.Button(
                                value="👁 预览", variant="secondary", size="sm"
                            )

                # 打印机选择 - 立即加载打印机列表
                initial_printers = get_printers()
                default_printer = "HiTi P525L相纸打印机"
                if default_printer not in initial_printers and initial_printers:
                    # 尝试找HiTi打印机
                    hiti_printer = next(
                        (p for p in initial_printers if "HiTi" in p or "hiti" in p),
                        None,
                    )
                    if hiti_printer:
                        default_printer = hiti_printer
                    else:
                        default_printer = initial_printers[0]

                printer_options = gr.Dropdown(
                    choices=initial_printers,
                    label="选择打印机",
                    value=default_printer,
                    interactive=True,
                )

                def update_printers():
                    printers = get_printers()
                    default_value = default_printer
                    if default_value not in printers and printers:
                        hiti_printer = next(
                            (p for p in printers if "HiTi" in p or "hiti" in p), None
                        )
                        if hiti_printer:
                            default_value = hiti_printer
                        else:
                            default_value = printers[0]
                    return gr.update(choices=printers, value=default_value)

                demo.load(fn=update_printers, outputs=[printer_options])
                demo.load(fn=None, inputs=[], outputs=[], js=DROP_REPLACE_JS)

                # 下载和打印反馈组件
                download_message = gr.Textbox(label="操作状态", visible=True)
                download_file_output = gr.File(visible=False)

                standard_print_btn.click(
                    fn=print_photo,
                    inputs=[img_output_standard, printer_options],
                    outputs=[download_message],
                )

                hd_print_btn.click(
                    fn=print_photo,
                    inputs=[img_output_standard_hd, printer_options],
                    outputs=[download_message],
                )

                # 标准照下载功能
                def download_and_open_folder_standard(img):
                    if img is None:
                        return "请先生成照片后再下载"
                    import os
                    import subprocess
                    from PIL import Image
                    import numpy as np

                    download_dir = "D:\\Download"
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)

                    file_path = os.path.join(download_dir, "standard_photo.png")

                    if isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = img

                    pil_img.save(file_path)

                    try:
                        if os.name == "nt":
                            os.startfile(download_dir)
                        else:
                            subprocess.run(["open", download_dir])
                    except:
                        pass

                    return f"已保存到: {file_path}"

                standard_download_btn.click(
                    fn=download_and_open_folder_standard,
                    inputs=[img_output_standard],
                    outputs=[download_message],
                )

                # 标准照预览功能
                standard_fullscreen_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="function(){var img=document.querySelector('#standard_photo_image img');if(!img){alert('请先生成照片');return;}var overlay=document.createElement('div');overlay.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.9);z-index:99999;display:flex;justify-content:center;align-items:center;cursor:pointer;';var previewImg=document.createElement('img');previewImg.src=img.src;previewImg.style.cssText='max-width:90%;max-height:90%;object-fit:contain;';overlay.appendChild(previewImg);document.body.appendChild(overlay);overlay.addEventListener('click',function(){overlay.remove();});}",
                )

                # 高清照下载功能
                def download_and_open_folder_hd(img):
                    if img is None:
                        return "请先生成照片后再下载"
                    import os
                    import subprocess
                    from PIL import Image
                    import numpy as np

                    download_dir = "D:\\Download"
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)

                    file_path = os.path.join(download_dir, "hd_photo.png")

                    if isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = img

                    pil_img.save(file_path)

                    try:
                        if os.name == "nt":
                            os.startfile(download_dir)
                        else:
                            subprocess.run(["open", download_dir])
                    except:
                        pass

                    return f"已保存到: {file_path}"

                hd_download_btn.click(
                    fn=download_and_open_folder_hd,
                    inputs=[img_output_standard_hd],
                    outputs=[download_message],
                )

                # 高清照预览功能
                hd_fullscreen_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="function(){var img=document.querySelector('#hd_photo_image img');if(!img){alert('请先生成照片');return;}var overlay=document.createElement('div');overlay.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.9);z-index:99999;display:flex;justify-content:center;align-items:center;cursor:pointer;';var previewImg=document.createElement('img');previewImg.src=img.src;previewImg.style.cssText='max-width:90%;max-height:90%;object-fit:contain;';overlay.appendChild(previewImg);document.body.appendChild(overlay);overlay.addEventListener('click',function(){overlay.remove();});}",
                )

                layout_print_btn.click(
                    fn=print_photo,
                    inputs=[img_output_layout, printer_options],
                    outputs=[download_message],
                )

                # 排版照下载功能
                def download_and_open_folder_layout(img):
                    if img is None:
                        return "请先生成照片后再下载"
                    import os
                    import subprocess
                    from PIL import Image
                    import numpy as np

                    download_dir = "D:\\Download"
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)

                    file_path = os.path.join(download_dir, "layout_photo.png")

                    if isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = img

                    pil_img.save(file_path)

                    try:
                        if os.name == "nt":
                            os.startfile(download_dir)
                        else:
                            subprocess.run(["open", download_dir])
                    except:
                        pass

                    return f"已保存到: {file_path}"

                layout_download_btn.click(
                    fn=download_and_open_folder_layout,
                    inputs=[img_output_layout],
                    outputs=[download_message],
                )

                # 排版照预览功能
                layout_fullscreen_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="function(){var img=document.querySelector('#layout_photo_image img');if(!img){alert('请先生成照片');return;}var overlay=document.createElement('div');overlay.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.9);z-index:99999;display:flex;justify-content:center;align-items:center;cursor:pointer;';var previewImg=document.createElement('img');previewImg.src=img.src;previewImg.style.cssText='max-width:90%;max-height:90%;object-fit:contain;';overlay.appendChild(previewImg);document.body.appendChild(overlay);overlay.addEventListener('click',function(){overlay.remove();});}",
                )
                # 模版照片
                with gr.Accordion(
                    LOCALES["template_photo"][DEFAULT_LANG]["label"], open=False
                ) as template_image_accordion:
                    img_output_template = gr.Gallery(
                        label=LOCALES["template_photo"][DEFAULT_LANG]["label"],
                        height=350,
                        format="png",
                    )
                # 抠图图像
                with gr.Accordion(
                    LOCALES["matting_image"][DEFAULT_LANG]["label"], open=False
                ) as matting_image_accordion:
                    with gr.Row():
                        img_output_standard_png = gr.Image(
                            label=LOCALES["standard_photo_png"][DEFAULT_LANG]["label"],
                            height=350,
                            format="png",
                            elem_id="standard_photo_png",
                            interactive=False,
                        )
                        img_output_standard_hd_png = gr.Image(
                            label=LOCALES["hd_photo_png"][DEFAULT_LANG]["label"],
                            height=350,
                            format="png",
                            elem_id="hd_photo_png",
                            interactive=False,
                        )

            # ---------------- 多语言切换函数 ----------------
            def change_language(language):
                return {
                    face_detect_model_options: gr.update(
                        label=LOCALES["face_model"][language]["label"]
                    ),
                    matting_model_options: gr.update(
                        label=LOCALES["matting_model"][language]["label"]
                    ),
                    size_list_options: gr.update(
                        label=LOCALES["size_list"][language]["label"],
                        choices=LOCALES["size_list"][language]["choices"],
                        value=LOCALES["size_list"][language]["choices"][0],
                    ),
                    mode_options: gr.update(
                        label=LOCALES["size_mode"][language]["label"],
                        choices=LOCALES["size_mode"][language]["choices"],
                        value=LOCALES["size_mode"][language]["choices"][0],
                    ),
                    color_options: gr.update(
                        label=LOCALES["bg_color"][language]["label"],
                        choices=LOCALES["bg_color"][language]["choices"],
                        value=LOCALES["bg_color"][language]["choices"][0],
                    ),
                    img_but: gr.update(value=LOCALES["button"][language]["label"]),
                    render_options: gr.update(
                        label=LOCALES["render_mode"][language]["label"],
                        choices=LOCALES["render_mode"][language]["choices"],
                        value=LOCALES["render_mode"][language]["choices"][0],
                    ),
                    crop_mode_options: gr.update(
                        label=LOCALES["crop_mode"][language]["label"],
                        choices=LOCALES["crop_mode"][language]["choices"],
                        value=LOCALES["crop_mode"][language]["choices"][0],
                    ),
                    image_kb_options: gr.update(
                        label=LOCALES["image_kb_size"][language]["label"],
                        choices=LOCALES["image_kb"][language]["choices"],
                        value=LOCALES["image_kb"][language]["choices"][0],
                    ),
                    custom_image_kb_size: gr.update(
                        label=LOCALES["image_kb"][language]["label"]
                    ),
                    notification: gr.update(
                        label=LOCALES["notification"][language]["label"]
                    ),
                    img_output_standard: gr.update(
                        label=LOCALES["standard_photo"][language]["label"]
                    ),
                    img_output_standard_hd: gr.update(
                        label=LOCALES["hd_photo"][language]["label"]
                    ),
                    img_output_standard_png: gr.update(
                        label=LOCALES["standard_photo_png"][language]["label"]
                    ),
                    img_output_standard_hd_png: gr.update(
                        label=LOCALES["hd_photo_png"][language]["label"]
                    ),
                    img_output_layout: gr.update(
                        label=LOCALES["layout_photo"][language]["label"]
                    ),
                    head_measure_ratio_option: gr.update(
                        label=LOCALES["head_measure_ratio"][language]["label"]
                    ),
                    top_distance_option: gr.update(
                        label=LOCALES["top_distance"][language]["label"]
                    ),
                    key_parameter_tab: gr.update(
                        label=LOCALES["key_param"][language]["label"]
                    ),
                    advance_parameter_tab: gr.update(
                        label=LOCALES["advance_param"][language]["label"]
                    ),
                    watermark_parameter_tab: gr.update(
                        label=LOCALES["watermark_tab"][language]["label"]
                    ),
                    watermark_text_options: gr.update(
                        label=LOCALES["watermark_text"][language]["label"],
                        placeholder=LOCALES["watermark_text"][language]["placeholder"],
                    ),
                    watermark_text_color: gr.update(
                        label=LOCALES["watermark_color"][language]["label"]
                    ),
                    watermark_text_size: gr.update(
                        label=LOCALES["watermark_size"][language]["label"]
                    ),
                    watermark_text_opacity: gr.update(
                        label=LOCALES["watermark_opacity"][language]["label"]
                    ),
                    watermark_text_angle: gr.update(
                        label=LOCALES["watermark_angle"][language]["label"]
                    ),
                    watermark_text_space: gr.update(
                        label=LOCALES["watermark_space"][language]["label"]
                    ),
                    watermark_options: gr.update(
                        label=LOCALES["watermark_switch"][language]["label"],
                        choices=LOCALES["watermark_switch"][language]["choices"],
                        value=LOCALES["watermark_switch"][language]["choices"][0],
                    ),
                    matting_image_accordion: gr.update(
                        label=LOCALES["matting_image"][language]["label"]
                    ),
                    beauty_parameter_tab: gr.update(
                        label=LOCALES["beauty_tab"][language]["label"]
                    ),
                    whitening_option: gr.update(
                        label=LOCALES["whitening_strength"][language]["label"]
                    ),
                    shadow_option: gr.update(
                        label=LOCALES["shadow_strength"][language]["label"]
                    ),
                    specular_option: gr.update(
                        label=LOCALES["specular_strength"][language]["label"]
                    ),
                    image_dpi_options: gr.update(
                        label=LOCALES["image_dpi"][language]["label"],
                        choices=LOCALES["image_dpi"][language]["choices"],
                        value=LOCALES["image_dpi"][language]["choices"][0],
                    ),
                    custom_image_dpi_size: gr.update(
                        label=LOCALES["image_dpi"][language]["label"]
                    ),
                    brightness_option: gr.update(
                        label=LOCALES["brightness_strength"][language]["label"]
                    ),
                    contrast_option: gr.update(
                        label=LOCALES["contrast_strength"][language]["label"]
                    ),
                    sharpen_option: gr.update(
                        label=LOCALES["sharpen_strength"][language]["label"]
                    ),
                    saturation_option: gr.update(
                        label=LOCALES["saturation_strength"][language]["label"]
                    ),
                    custom_size_width_px: gr.update(
                        label=LOCALES["custom_size_px"][language]["width"]
                    ),
                    custom_size_height_px: gr.update(
                        label=LOCALES["custom_size_px"][language]["height"]
                    ),
                    custom_size_width_mm: gr.update(
                        label=LOCALES["custom_size_mm"][language]["width"]
                    ),
                    custom_size_height_mm: gr.update(
                        label=LOCALES["custom_size_mm"][language]["height"]
                    ),
                    img_output_template: gr.update(
                        label=LOCALES["template_photo"][language]["label"]
                    ),
                    template_image_accordion: gr.update(
                        label=LOCALES["template_photo"][language]["label"]
                    ),
                    plugin_options: gr.update(
                        label=LOCALES["plugin"][language]["label"],
                        choices=LOCALES["plugin"][language]["choices"],
                        value=LOCALES["plugin"][language]["choices"][0],
                    ),
                    print_parameter_tab: gr.update(
                        label=LOCALES["print_tab"][language]["label"]
                    ),
                    print_options: gr.update(
                        label=LOCALES["print_switch"][language]["label"],
                        choices=LOCALES["print_switch"][language]["choices"],
                        value=LOCALES["print_switch"][language]["choices"][0],
                    ),
                }

            def change_visibility(option, lang, locales_key, custom_component):
                return {
                    custom_component: gr.update(
                        visible=option == LOCALES[locales_key][lang]["choices"][-1]
                    )
                }

            def change_color(colors, lang):
                return {
                    custom_color_rgb: gr.update(
                        visible=colors == LOCALES["bg_color"][lang]["choices"][-2]
                    ),
                    custom_color_hex: gr.update(
                        visible=colors == LOCALES["bg_color"][lang]["choices"][-1]
                    ),
                }

            def change_size_mode(size_option_item, lang):
                choices = LOCALES["size_mode"][lang]["choices"]
                # 如果选择自定义尺寸mm
                if size_option_item == choices[3]:
                    return {
                        custom_size_px: gr.update(visible=False),
                        custom_size_mm: gr.update(visible=True),
                        size_list_row: gr.update(visible=False),
                        plugin_options: gr.update(interactive=True),
                    }
                # 如果选择自定义尺寸px
                elif size_option_item == choices[2]:
                    return {
                        custom_size_px: gr.update(visible=True),
                        custom_size_mm: gr.update(visible=False),
                        size_list_row: gr.update(visible=False),
                        plugin_options: gr.update(interactive=True),
                    }
                # 如果选择只换底，则隐藏所有尺寸组件
                elif size_option_item == choices[1]:
                    return {
                        custom_size_px: gr.update(visible=False),
                        custom_size_mm: gr.update(visible=False),
                        size_list_row: gr.update(visible=False),
                        plugin_options: gr.update(interactive=False),
                    }
                # 如果选择预设尺寸，则隐藏自定义尺寸组件
                else:
                    return {
                        custom_size_px: gr.update(visible=False),
                        custom_size_mm: gr.update(visible=False),
                        size_list_row: gr.update(visible=True),
                        plugin_options: gr.update(interactive=True),
                    }

            def change_image_kb(image_kb_option, lang):
                return change_visibility(
                    image_kb_option, lang, "image_kb", custom_image_kb_size
                )

            def change_image_dpi(image_dpi_option, lang):
                return change_visibility(
                    image_dpi_option, lang, "image_dpi", custom_image_dpi_size
                )

            # ---------------- 绑定事件 ----------------
            # 语言切换
            language_options.input(
                change_language,
                inputs=[language_options],
                outputs=[
                    size_list_options,
                    mode_options,
                    color_options,
                    img_but,
                    render_options,
                    image_kb_options,
                    matting_model_options,
                    face_detect_model_options,
                    custom_image_kb_size,
                    notification,
                    img_output_standard,
                    img_output_standard_hd,
                    img_output_standard_png,
                    img_output_standard_hd_png,
                    img_output_layout,
                    head_measure_ratio_option,
                    top_distance_option,
                    key_parameter_tab,
                    advance_parameter_tab,
                    watermark_parameter_tab,
                    watermark_text_options,
                    watermark_text_color,
                    watermark_text_size,
                    watermark_text_opacity,
                    watermark_text_angle,
                    watermark_text_space,
                    watermark_options,
                    matting_image_accordion,
                    beauty_parameter_tab,
                    whitening_option,
                    shadow_option,
                    specular_option,
                    image_dpi_options,
                    custom_image_dpi_size,
                    brightness_option,
                    contrast_option,
                    sharpen_option,
                    saturation_option,
                    plugin_options,
                    custom_size_width_px,
                    custom_size_height_px,
                    custom_size_width_mm,
                    custom_size_height_mm,
                    img_output_template,
                    template_image_accordion,
                    print_parameter_tab,
                    print_options,
                    crop_mode_options,
                ],
            )

            # ---------------- 设置隐藏/显示交互效果 ----------------
            # 尺寸模式
            mode_options.input(
                change_size_mode,
                inputs=[mode_options, language_options],
                outputs=[
                    custom_size_px,
                    custom_size_mm,
                    size_list_row,
                    plugin_options,
                ],
            )

            # 颜色
            color_options.input(
                change_color,
                inputs=[color_options, language_options],
                outputs=[custom_color_rgb, custom_color_hex],
            )

            # 图片kb
            image_kb_options.input(
                change_image_kb,
                inputs=[image_kb_options, language_options],
                outputs=[custom_image_kb_size],
            )

            # 图片dpi
            image_dpi_options.input(
                change_image_dpi,
                inputs=[image_dpi_options, language_options],
                outputs=[custom_image_dpi_size],
            )

            img_but.click(
                processor.process,
                inputs=[
                    img_input,
                    mode_options,
                    size_list_options,
                    color_options,
                    render_options,
                    image_kb_options,
                    custom_color_R,
                    custom_color_G,
                    custom_color_B,
                    custom_color_hex_value,
                    custom_size_height_px,
                    custom_size_width_px,
                    custom_size_height_mm,
                    custom_size_width_mm,
                    custom_image_kb_size,
                    language_options,
                    matting_model_options,
                    watermark_options,
                    watermark_text_options,
                    watermark_text_color,
                    watermark_text_size,
                    watermark_text_opacity,
                    watermark_text_angle,
                    watermark_text_space,
                    face_detect_model_options,
                    head_measure_ratio_option,
                    top_distance_option,
                    whitening_option,
                    image_dpi_options,
                    custom_image_dpi_size,
                    brightness_option,
                    contrast_option,
                    sharpen_option,
                    saturation_option,
                    shadow_option,
                    specular_option,
                    plugin_options,
                    print_options,
                    crop_mode_options,
                ],
                outputs=[
                    img_output_standard,
                    img_output_standard_hd,
                    img_output_standard_png,
                    img_output_standard_hd_png,
                    img_output_layout,
                    img_output_template,
                    template_image_accordion,
                    notification,
                ],
            )

    return demo
