import os
import cv2
import argparse
import numpy as np
from hivision.error import APIError
from hivision.utils import (
    hex_to_rgb,
    resize_image_to_kb,
    add_background,
    save_image_dpi_to_bytes,
)
from hivision import IDCreator
from hivision.creator.layout_calculator import (
    generate_layout_array,
    generate_layout_image,
)
from hivision.creator.choose_handler import choose_handler
from hivision.utils import hex_to_rgb, resize_image_to_kb


INFERENCE_TYPE = [
    "idphoto",
    "human_matting",
    "add_background",
    "generate_layout_photos",
    "idphoto_crop",
]
MATTING_MODEL = [
    "hivision_modnet",
    "modnet_photographic_portrait_matting",
    "mnn_hivision_modnet",
    "rmbg-1.4",
    "birefnet-v1-lite",
]
FACE_DETECT_MODEL = [
    "mtcnn",
    "face_plusplus",
    "retinaface-resnet50",
]
RENDER = [0, 1, 2]


def center_crop_to_size(image, target_height, target_width):
    """按比例填充：保持宽高比，铺满目标框，裁剪超出部分。"""
    h, w = image.shape[:2]
    scale = max(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    if scale < 1:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_y = (new_h - target_height) // 2
    start_x = (new_w - target_width) // 2
    return resized[start_y : start_y + target_height, start_x : start_x + target_width]


def fit_to_size(image, target_height, target_width, pad_color=(255, 255, 255)):
    """按比例拟合：保持宽高比，完整放入目标框，短边留白。"""
    h, w = image.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    if scale < 1:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(image.shape) == 3 and image.shape[2] == 4:
        canvas = np.full((target_height, target_width, 4), 0, dtype=np.uint8)
        canvas[:, :, :3] = pad_color
        canvas[:, :, 3] = 255
    else:
        canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    start_y = (target_height - new_h) // 2
    start_x = (target_width - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized
    return canvas


parser = argparse.ArgumentParser(description="HivisionIDPhotos 证件照制作推理程序。")
parser.add_argument(
    "-t",
    "--type",
    help="请求 API 的种类",
    choices=INFERENCE_TYPE,
    default="idphoto",
)
parser.add_argument("-i", "--input_image_dir", help="输入图像路径", required=True)
parser.add_argument("-o", "--output_image_dir", help="保存图像路径", required=True)
parser.add_argument("--height", help="证件照尺寸-高", default=413)
parser.add_argument("--width", help="证件照尺寸-宽", default=295)
parser.add_argument("-c", "--color", help="证件照背景色", default="638cce")
parser.add_argument("--hd", type=bool, help="是否输出高清照", default=True)
parser.add_argument(
    "-k", "--kb", help="输出照片的 KB 值，仅对换底和制作排版照生效", default=None
)
parser.add_argument(
    "-r",
    "--render",
    type=int,
    help="底色合成的模式，有 0:纯色、1:上下渐变、2:中心渐变 可选",
    choices=RENDER,
    default=0,
)
parser.add_argument(
    "--dpi",
    type=int,
    help="输出照片的 DPI 值",
    default=300,
)
parser.add_argument(
    "--face_align",
    type=bool,
    help="是否进行人脸旋转矫正",
    default=False,
)
parser.add_argument(
    "--matting_model",
    help="抠图模型权重",
    default="modnet_photographic_portrait_matting",
    choices=MATTING_MODEL,
)
parser.add_argument(
    "--face_detect_model",
    help="人脸检测模型",
    default="mtcnn",
    choices=FACE_DETECT_MODEL,
)
parser.add_argument(
    "--crop_mode",
    type=int,
    help="画幅裁切模式（无人脸/多人脸时生效），0:按比例填充（裁剪，默认），1:按比例拟合（留白）",
    choices=[0, 1],
    default=0,
)

args = parser.parse_args()

# ------------------- 选择抠图与人脸检测模型 -------------------
creator = IDCreator()
choose_handler(creator, args.matting_model, args.face_detect_model)

root_dir = os.path.dirname(os.path.abspath(__file__))
input_image = cv2.imread(args.input_image_dir, cv2.IMREAD_UNCHANGED)

# 如果模式是生成证件照
if args.type == "idphoto":
    # 将字符串转为元组
    size = (int(args.height), int(args.width))
    result = creator(input_image, size=size, face_alignment=args.face_align)
    if result.face_num is not None:
        # 降级模式：使用原图进行画幅裁切，不进行抠图和换底
        crop_mode_name = "按比例拟合" if args.crop_mode == 1 else "按比例填充"
        print(
            f"检测到{result.face_num}个人脸，已自动切换为画幅裁切模式（{crop_mode_name}），使用原图裁切。"
        )
        # 使用原图（BGR格式）
        fallback_image = input_image.copy()
        if len(fallback_image.shape) == 2:
            fallback_image = cv2.cvtColor(fallback_image, cv2.COLOR_GRAY2BGR)
        if args.crop_mode == 1:
            fallback_image = fit_to_size(fallback_image, size[0], size[1])
        else:
            fallback_image = center_crop_to_size(fallback_image, size[0], size[1])
        # 保存裁切后的原图（BGR→RGBA，PIL的Image.fromarray期望RGBA格式）
        save_image_dpi_to_bytes(
            cv2.cvtColor(fallback_image, cv2.COLOR_BGR2RGBA),
            args.output_image_dir,
            dpi=args.dpi,
        )
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(
            cv2.cvtColor(fallback_image, cv2.COLOR_BGR2RGBA),
            new_file_name,
            dpi=args.dpi,
        )
    else:
        # 正常模式：使用抠图结果
        # 保存标准照
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA),
            args.output_image_dir,
            dpi=args.dpi,
        )
        # 保存高清照
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=args.dpi
        )

# 如果模式是人像抠图
elif args.type == "human_matting":
    result = creator(input_image, change_bg_only=True)
    cv2.imwrite(args.output_image_dir, result.hd)

# 如果模式是添加背景
elif args.type == "add_background":

    render_choice = ["pure_color", "updown_gradient", "center_gradient"]

    # 将字符串转为元组
    color = hex_to_rgb(args.color)
    # 将元祖的 0 和 2 号数字交换
    color = (color[2], color[1], color[0])

    result_image = add_background(
        input_image, bgr=color, mode=render_choice[args.render]
    )
    result_image = result_image.astype(np.uint8)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGRA)

    if args.kb:
        resize_image_to_kb(
            result_image, args.output_image_dir, int(args.kb), dpi=args.dpi
        )
    else:
        save_image_dpi_to_bytes(
            cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGRA),
            args.output_image_dir,
            dpi=args.dpi,
        )

# 如果模式是生成排版照
elif args.type == "generate_layout_photos":

    size = (int(args.height), int(args.width))

    typography_arr, typography_rotate = generate_layout_array(
        input_height=size[0], input_width=size[1]
    )

    result_layout_image = generate_layout_image(
        input_image,
        typography_arr,
        typography_rotate,
        height=size[0],
        width=size[1],
    )

    if args.kb:
        result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
        result_layout_image = resize_image_to_kb(
            result_layout_image, args.output_image_dir, int(args.kb), dpi=args.dpi
        )
    else:
        save_image_dpi_to_bytes(
            cv2.cvtColor(result_layout_image, cv2.COLOR_RGBA2BGRA),
            args.output_image_dir,
            dpi=args.dpi,
        )

# 如果模式是证件照裁切
elif args.type == "idphoto_crop":
    # 将字符串转为元组
    size = (int(args.height), int(args.width))
    result = creator(input_image, size=size, crop_only=True)
    if result.face_num is not None:
        # 降级模式：使用原图进行画幅裁切
        crop_mode_name = "按比例拟合" if args.crop_mode == 1 else "按比例填充"
        print(
            f"检测到{result.face_num}个人脸，已自动切换为画幅裁切模式（{crop_mode_name}），使用原图裁切。"
        )
        fallback_image = input_image.copy()
        if len(fallback_image.shape) == 2:
            fallback_image = cv2.cvtColor(fallback_image, cv2.COLOR_GRAY2BGR)
        if args.crop_mode == 1:
            fallback_image = fit_to_size(fallback_image, size[0], size[1])
        else:
            fallback_image = center_crop_to_size(fallback_image, size[0], size[1])
        save_image_dpi_to_bytes(
            cv2.cvtColor(fallback_image, cv2.COLOR_BGR2RGBA),
            args.output_image_dir,
            dpi=args.dpi,
        )
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(
            cv2.cvtColor(fallback_image, cv2.COLOR_BGR2RGBA),
            new_file_name,
            dpi=args.dpi,
        )
    else:
        # 正常模式
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA),
            args.output_image_dir,
            dpi=args.dpi,
        )
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=args.dpi
        )
