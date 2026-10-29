from fastapi import FastAPI, UploadFile, Form, File
from hivision import IDCreator
from hivision.error import APIError
from hivision.creator.layout_calculator import (
    generate_layout_array,
    generate_layout_image,
)
from hivision.creator.choose_handler import choose_handler
from hivision.utils import (
    add_background,
    resize_image_to_kb,
    bytes_2_base64,
    base64_2_numpy,
    hex_to_rgb,
    add_watermark,
    save_image_dpi_to_bytes,
)
import numpy as np
import cv2
from starlette.middleware.cors import CORSMiddleware


def center_crop_to_size(image, target_height, target_width):
    """按比例填充（Cover）：保持宽高比，铺满目标框，裁剪超出部分。"""
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
    """按比例拟合（Contain）：保持宽高比，完整放入目标框，短边留白。"""
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


app = FastAPI()
creator = IDCreator()

# 添加 CORS 中间件 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的请求来源
    allow_credentials=True,  # 允许携带 Cookie
    allow_methods=[
        "*"
    ],  # 允许的请求方法，例如：GET, POST 等，也可以指定 ["GET", "POST"]
    allow_headers=["*"],  # 允许的请求头，也可以指定具体的头部
)


# 证件照智能制作接口
@app.post("/idphoto")
async def idphoto_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    human_matting_model: str = Form("modnet_photographic_portrait_matting"),
    face_detect_model: str = Form("mtcnn"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    face_align: bool = Form(False),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
    brightness_strength: float = Form(0),
    contrast_strength: float = Form(0),
    sharpen_strength: float = Form(0),
    saturation_strength: float = Form(0),
    crop_mode: int = Form(0),  # 0:按比例填充(默认) 1:按比例拟合
):
    # 如果传入了base64，则直接使用base64解码
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    # 否则使用上传的图片
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, human_matting_model, face_detect_model)

    # 将字符串转为元组
    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            face_alignment=face_align,
            brightness_strength=brightness_strength,
            contrast_strength=contrast_strength,
            sharpen_strength=sharpen_strength,
            saturation_strength=saturation_strength,
        )
    except APIError:
        return {"status": False}
    # 如果检测到人脸数量不等于1，使用原图进行画幅裁切
    if result.face_num is not None:
        # 使用原图（不使用抠图结果，因为抠图模型对物体兼容性差）
        fallback_image = img.copy()
        if crop_mode == 1:  # 按比例拟合
            fallback_image = fit_to_size(fallback_image, size[0], size[1])
        else:  # 按比例填充
            fallback_image = center_crop_to_size(fallback_image, size[0], size[1])
        # 转为RGBA格式（PIL的Image.fromarray期望RGBA而非BGRA）
        fallback_rgba = cv2.cvtColor(fallback_image, cv2.COLOR_BGR2RGBA)
        result_image_standard_bytes = save_image_dpi_to_bytes(fallback_rgba, None, dpi)
        result_message = {
            "status": True,
            "image_base64_standard": bytes_2_base64(result_image_standard_bytes),
            "is_fallback": True,
            "face_num": result.face_num,
        }
        if hd:
            fallback_hd = img.copy()
            if crop_mode == 1:
                fallback_hd = fit_to_size(fallback_hd, size[0], size[1])
            else:
                fallback_hd = center_crop_to_size(fallback_hd, size[0], size[1])
            fallback_hd_rgba = cv2.cvtColor(fallback_hd, cv2.COLOR_BGR2RGBA)
            result_image_hd_bytes = save_image_dpi_to_bytes(fallback_hd_rgba, None, dpi)
            result_message["image_base64_hd"] = bytes_2_base64(result_image_hd_bytes)
    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi
        )

        result_message = {
            "status": True,
            "image_base64_standard": bytes_2_base64(result_image_standard_bytes),
        }

        # 如果hd为True, 则增加高清照结果（png 4通道图像）
        if hd:
            result_image_hd_bytes = save_image_dpi_to_bytes(
                cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), None, dpi
            )
            result_message["image_base64_hd"] = bytes_2_base64(result_image_hd_bytes)

    return result_message


# 人像抠图接口
@app.post("/human_matting")
async def human_matting_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    human_matting_model: str = Form("hivision_modnet"),
    dpi: int = Form(300),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, human_matting_model, None)

    result = creator(
        img,
        change_bg_only=True,
    )
    result_image_standard_bytes = save_image_dpi_to_bytes(
        cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi
    )
    result_message = {
        "status": True,
        "image_base64": bytes_2_base64(result_image_standard_bytes),
    }
    return result_message


# 透明图像添加纯色背景接口
@app.post("/add_background")
async def photo_add_background(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    color: str = Form("000000"),
    kb: int = Form(None),
    dpi: int = Form(300),
    render: int = Form(0),
):
    render_choice = ["pure_color", "updown_gradient", "center_gradient"]

    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    color = hex_to_rgb(color)
    color = (color[2], color[1], color[0])

    result_image = add_background(
        img,
        bgr=color,
        mode=render_choice[render],
    ).astype(np.uint8)

    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
    else:
        result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)

    result_messgae = {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
    }

    return result_messgae


# 六寸排版照生成接口
@app.post("/generate_layout_photos")
async def generate_layout_photos(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    kb: int = Form(None),
    dpi: int = Form(300),
):
    # try:
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    size = (int(height), int(width))

    typography_arr, typography_rotate = generate_layout_array(
        input_height=size[0], input_width=size[1]
    )

    result_layout_image = generate_layout_image(
        img, typography_arr, typography_rotate, height=size[0], width=size[1]
    ).astype(np.uint8)

    result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_layout_image_bytes = resize_image_to_kb(
            result_layout_image, None, int(kb), dpi=dpi
        )
    else:
        result_layout_image_bytes = save_image_dpi_to_bytes(
            result_layout_image, None, dpi=dpi
        )

    result_layout_image_base64 = bytes_2_base64(result_layout_image_bytes)

    result_messgae = {
        "status": True,
        "image_base64": result_layout_image_base64,
    }

    return result_messgae


# 透明图像添加水印接口
@app.post("/watermark")
async def watermark(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    text: str = Form("Hello"),
    size: int = 20,
    opacity: float = 0.5,
    angle: int = 30,
    color: str = "#000000",
    space: int = 25,
    kb: int = Form(None),
    dpi: int = Form(300),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result_image = add_watermark(img, text, size, opacity, angle, color, space)

        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        if kb:
            result_image_bytes = resize_image_to_kb(
                result_image, None, int(kb), dpi=dpi
            )
        else:
            result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)
        result_image_base64 = bytes_2_base64(result_image_bytes)

        result_messgae = {
            "status": True,
            "image_base64": result_image_base64,
        }
    except Exception as e:
        result_messgae = {
            "status": False,
            "error": str(e),
        }

    return result_messgae


# 设置照片KB值接口(RGB图)
@app.post("/set_kb")
async def set_kb(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    dpi: int = Form(300),
    kb: int = Form(50),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
        result_image_base64 = bytes_2_base64(result_image_bytes)

        result_messgae = {
            "status": True,
            "image_base64": result_image_base64,
        }
    except Exception as e:
        result_messgae = {
            "status": False,
            "error": e,
        }

    return result_messgae


# 证件照智能裁剪接口
@app.post("/idphoto_crop")
async def idphoto_crop_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    face_detect_model: str = Form("mtcnn"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # 读取图像(4通道)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, face_detect_option=face_detect_model)

    # 将字符串转为元组
    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            crop_only=True,
        )
    except APIError:
        result_message = {"status": False}
    # 如果检测到人脸数量不等于1，返回失败
    if result.face_num is not None:
        result_message = {"status": False}
    # 如果检测到人脸数量等于1, 则返回标准证和高清照结果（png 4通道图像）
    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi
        )

        result_message = {
            "status": True,
            "image_base64_standard": bytes_2_base64(result_image_standard_bytes),
        }

        # 如果hd为True, 则增加高清照结果（png 4通道图像）
        if hd:
            result_image_hd_bytes = save_image_dpi_to_bytes(
                cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), None, dpi
            )
            result_message["image_base64_hd"] = bytes_2_base64(result_image_hd_bytes)

    return result_message


if __name__ == "__main__":
    import uvicorn

    # 在8080端口运行推理服务
    uvicorn.run(app, host="0.0.0.0", port=8080)
