# 本文件相对上游项目 HivisionIDPhotos (https://github.com/Zeyi-Lin/HivisionIDPhotos)
# 有修改，改动范围见 NOTICE 与 CHANGELOG.md。

import cv2
import numpy as np
from hivision.creator.context import Context
from hivision.plugin.beauty.whitening import make_whitening
from hivision.plugin.beauty.retouch import retouch_rgba
from hivision.plugin.beauty.base_adjust import (
    adjust_brightness_contrast_sharpen_saturation,
)


def beauty_face(ctx: Context):
    """
    对人脸进行美颜处理
    1. 美白
    2. 亮度

    :param ctx: Context对象，包含处理参数和图像
    """
    middle_image = ctx.origin_image.copy()
    processed = False

    # 去反光 / 去阴影：放在美白之前。先消掉汗光再拉平光照——去阴影是整片提亮，
    # 会把汗光一起放大，顺序反了刚去掉的反光又会重新变亮。
    if ctx.params.shadow_strength > 0 or ctx.params.specular_strength > 0:
        # 算法要按人像区域来算，所以借抠图结果的 alpha 通道拼成 RGBA
        rgb = middle_image[:, :, :3]
        matting = ctx.matting_image
        has_alpha = matting.ndim == 3 and matting.shape[2] == 4
        alpha = (
            matting[:, :, 3]
            if has_alpha
            else np.full(rgb.shape[:2], 255, np.uint8)
        )
        rgba = retouch_rgba(
            np.dstack([rgb, alpha]),
            ctx.params.shadow_strength,
            ctx.params.specular_strength,
        )
        middle_image = (
            np.dstack([rgba[:, :, :3], middle_image[:, :, 3]])
            if middle_image.shape[2] == 4
            else rgba[:, :, :3]
        )
        processed = True

    # 如果美白强度大于0，进行美白处理
    if ctx.params.whitening_strength > 0:
        middle_image = make_whitening(middle_image, ctx.params.whitening_strength)
        processed = True

    # 如果亮度、对比度、锐化强度不为0，进行亮度、对比度、锐化处理
    if (
        ctx.params.brightness_strength != 0
        or ctx.params.contrast_strength != 0
        or ctx.params.sharpen_strength != 0
        or ctx.params.saturation_strength != 0
    ):
        middle_image = adjust_brightness_contrast_sharpen_saturation(
            middle_image,
            ctx.params.brightness_strength,
            ctx.params.contrast_strength,
            ctx.params.sharpen_strength,
            ctx.params.saturation_strength,
        )
        processed = True

    # 如果进行了美颜处理，更新matting_image
    if processed:
        # 分离中间图像的BGR通道
        b, g, r = cv2.split(middle_image)
        # 从原始matting_image中获取alpha通道
        _, _, _, alpha = cv2.split(ctx.matting_image)
        # 合并处理后的BGR通道和原始alpha通道
        ctx.matting_image = cv2.merge((b, g, r, alpha))
