#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:35
@File: layout_calculator.py
@IDE: pycharm
@Description:
    布局计算器
"""

import cv2.detail
import numpy as np


def judge_layout(
    input_width,
    input_height,
    PHOTO_INTERVAL_W,
    PHOTO_INTERVAL_H,
    LIMIT_BLOCK_W,
    LIMIT_BLOCK_H,
):
    centerBlockHeight_1, centerBlockWidth_1 = (
        input_height,
        input_width,
    )  # 由证件照们组成的一个中心区块（1 代表不转置排列）
    centerBlockHeight_2, centerBlockWidth_2 = (
        input_width,
        input_height,
    )  # 由证件照们组成的一个中心区块（2 代表转置排列）

    # 1.不转置排列的情况下：
    layout_col_no_transpose = 0  # 行
    layout_row_no_transpose = 0  # 列
    for i in range(1, 4):
        centerBlockHeight_temp = input_height * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_1 = centerBlockHeight_temp
            layout_row_no_transpose = i
        else:
            break
    for j in range(1, 9):
        centerBlockWidth_temp = input_width * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_1 = centerBlockWidth_temp
            layout_col_no_transpose = j
        else:
            break
    layout_number_no_transpose = layout_row_no_transpose * layout_col_no_transpose

    # 2.转置排列的情况下：
    layout_col_transpose = 0  # 行
    layout_row_transpose = 0  # 列
    for i in range(1, 4):
        centerBlockHeight_temp = input_width * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_2 = centerBlockHeight_temp
            layout_row_transpose = i
        else:
            break
    for j in range(1, 9):
        centerBlockWidth_temp = input_height * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_2 = centerBlockWidth_temp
            layout_col_transpose = j
        else:
            break
    layout_number_transpose = layout_row_transpose * layout_col_transpose

    # 确保至少有一个排版模式
    if layout_number_transpose == 0 and layout_number_no_transpose == 0:
        # 如果两种排列方式都无法排列，返回只排一张的模式
        return (1, 1, 1), input_width, input_height

    if layout_number_transpose > layout_number_no_transpose:
        layout_mode = (layout_col_transpose, layout_row_transpose, 2)
        return layout_mode, centerBlockWidth_2, centerBlockHeight_2
    else:
        layout_mode = (layout_col_no_transpose, layout_row_no_transpose, 1)
        return layout_mode, centerBlockWidth_1, centerBlockHeight_1


def generate_layout_array(
    input_height, input_width, LAYOUT_WIDTH=1795, LAYOUT_HEIGHT=1205
):
    # 1.基础参数表
    PHOTO_INTERVAL_H = 1  # 证件照与证件照之间的垂直距离（约0.5mm）
    PHOTO_INTERVAL_W = 1  # 证件照与证件照之间的水平距离（约0.5mm）
    SIDES_INTERVAL_H = 0  # 证件照与画布边缘的垂直距离（0，让照片尽量铺满）
    SIDES_INTERVAL_W = 0  # 证件照与画布边缘的水平距离（0，让照片尽量铺满）
    LIMIT_BLOCK_W = LAYOUT_WIDTH - 2 * SIDES_INTERVAL_W
    LIMIT_BLOCK_H = LAYOUT_HEIGHT - 2 * SIDES_INTERVAL_H

    # 检查预设尺寸是否大于等于排版尺寸
    if input_width >= LIMIT_BLOCK_W or input_height >= LIMIT_BLOCK_H:
        # 如果预设尺寸大于等于排版尺寸，返回铺满相纸的排版
        typography_arr = [[SIDES_INTERVAL_W, SIDES_INTERVAL_H]]
        typography_rotate = False
        return typography_arr, typography_rotate

    # 2.创建一个 1180x1746 的空白画布
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)

    # 3.特殊处理：对于1寸照片（413x295）和6寸排版（1795x1205），限制为12张
    typography_rotate = False
    if (
        input_height == 413
        and input_width == 295
        and LAYOUT_HEIGHT == 1205
        and LAYOUT_WIDTH == 1795
    ):
        # 1寸照片配6寸排版，限制为4列3行，共12张（3行4列）
        layout_mode = (4, 3, 2)  # 强制使用转置排列，这样照片会横着排，更适合6寸相纸
        typography_rotate = True  # 明确设置为需要旋转
        # 重新计算中心区块的宽度和高度
        # 转置后，照片的宽度变成413，高度变成295
        transposed_width = input_height
        transposed_height = input_width
        centerBlockWidth = transposed_width * 4 + PHOTO_INTERVAL_W * 3
        centerBlockHeight = transposed_height * 3 + PHOTO_INTERVAL_H * 2
    else:
        # 其他情况，计算照片的 layout
        layout_mode, centerBlockWidth, centerBlockHeight = judge_layout(
            input_width,
            input_height,
            PHOTO_INTERVAL_W,
            PHOTO_INTERVAL_H,
            LIMIT_BLOCK_W,
            LIMIT_BLOCK_H,
        )

        # 确保 layout_mode 中的列和行至少为 1
        layout_cols = max(1, layout_mode[0])
        layout_rows = max(1, layout_mode[1])
        layout_mode = (layout_cols, layout_rows, layout_mode[2])

        # 根据 layout_mode 确定是否需要旋转
        typography_rotate = layout_mode[2] == 2

    # 4.开始排列组合
    x11 = int((LAYOUT_WIDTH - centerBlockWidth) // 2)
    y11 = int((LAYOUT_HEIGHT - centerBlockHeight) // 2)
    typography_arr = []

    # 计算当前使用的照片尺寸
    current_width = input_width
    current_height = input_height
    if typography_rotate:
        current_width, current_height = current_height, current_width

    for j in range(layout_mode[1]):
        for i in range(layout_mode[0]):
            xi = int(x11 + i * current_width + i * PHOTO_INTERVAL_W)
            yi = int(y11 + j * current_height + j * PHOTO_INTERVAL_H)
            typography_arr.append([xi, yi])

    return typography_arr, typography_rotate


def generate_layout_image(
    input_image,
    typography_arr,
    typography_rotate,
    width=295,
    height=413,
    crop_line: bool = False,
    LAYOUT_WIDTH=1795,
    LAYOUT_HEIGHT=1205,
):
    # 检查输入图像是否为None
    if input_image is None:
        # 如果输入图像为None，返回一个空白画布
        white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
        white_background.fill(255)
        return white_background

    # 检查排版数组是否为空
    if not typography_arr:
        # 如果排版数组为空，返回一个空白画布
        white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
        white_background.fill(255)
        return white_background

    # 检查输入图像是否有shape属性
    if not hasattr(input_image, "shape"):
        # 如果输入图像没有shape属性，返回一个空白画布
        white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
        white_background.fill(255)
        return white_background

    # 创建一个白色背景的空白画布
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)

    try:
        # 检查是否是铺满相纸的情况
        if len(typography_arr) == 1 and typography_arr[0] == [0, 0]:
            # 铺满相纸的情况
            # 计算铺满相纸的尺寸
            img_height, img_width = input_image.shape[:2]

            # 如果是竖版照片，旋转90度变成横版
            if img_height > img_width:
                input_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)
                img_height, img_width = img_width, img_height

            # 以横版画布为基准，保持照片原始比例
            # 计算目标尺寸，保持原始比例
            scale = LAYOUT_WIDTH / img_width
            target_width = LAYOUT_WIDTH
            target_height = int(img_height * scale)

            # 如果高度超过画布，以高度为基准
            if target_height > LAYOUT_HEIGHT:
                scale = LAYOUT_HEIGHT / img_height
                target_height = LAYOUT_HEIGHT
                target_width = int(img_width * scale)

            # 计算居中位置
            locate_x = (LAYOUT_WIDTH - target_width) // 2
            locate_y = (LAYOUT_HEIGHT - target_height) // 2

            # 调整图像大小以铺满相纸
            input_image = cv2.resize(input_image, (target_width, target_height))
            # 将图像放置到画布上
            white_background[
                locate_y : locate_y + target_height, locate_x : locate_x + target_width
            ] = input_image
        else:
            # 正常排版的情况
            # 调整图像大小到指定尺寸
            input_image = cv2.resize(input_image, (width, height))

            # 如果需要旋转排版，则对图像进行转置和垂直镜像
            if typography_rotate:
                input_image = cv2.transpose(input_image)
                input_image = cv2.flip(input_image, 0)  # 0 表示垂直镜像
                # 交换高度和宽度
                height, width = width, height

            # 将图像按照排版数组中的位置放置到白色背景上
            for arr in typography_arr:
                locate_x, locate_y = arr[0], arr[1]
                # 确保坐标和尺寸都是整数
                locate_x, locate_y = int(locate_x), int(locate_y)
                h, w = int(height), int(width)
                # 确保不会超出画布范围
                if locate_y + h <= LAYOUT_HEIGHT and locate_x + w <= LAYOUT_WIDTH:
                    white_background[
                        locate_y : locate_y + h, locate_x : locate_x + w
                    ] = input_image
    except Exception as e:
        # 如果出现任何错误，返回一个空白画布
        white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
        white_background.fill(255)

    if crop_line:
        # 裁剪线：1像素宽，50%黑色（RGB 128）
        crop_line_color = (128, 128, 128)
        h, w = int(height), int(width)

        horizontal_lines = set()
        vertical_lines = set()

        for arr in typography_arr:
            locate_x, locate_y = arr[0], arr[1]
            horizontal_lines.add(locate_y)
            horizontal_lines.add(locate_y + h)
            vertical_lines.add(locate_x)
            vertical_lines.add(locate_x + w)

        for y in horizontal_lines:
            if 0 <= y < LAYOUT_HEIGHT:
                white_background[y, :] = crop_line_color

        for x in vertical_lines:
            if 0 <= x < LAYOUT_WIDTH:
                white_background[:, x] = crop_line_color

    # 返回排版后的图像
    return white_background
