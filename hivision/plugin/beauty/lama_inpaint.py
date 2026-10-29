# 本文件为本仓库新增，不属于上游项目 HivisionIDPhotos。
# 许可遵循 Apache-2.0（见 LICENSE 与 NOTICE）。

"""
LaMa 图像修复模型（ONNX）：把光斑和阴影当成"待填的区域"，让网络补出这块皮肤本来该有的样子。

手写算法只能测出"这里比别处亮/暗了多少"，然后按一个自定的增益去补，补出来的是统计量
而不是皮肤——所以汗光压平以后是一块比周围更暗的灰斑，阴影抬起来以后边界还留一道边。
LaMa 是在大量真实图像上训练出来的修复网络，它填进去的是带毛孔、带血色、带胡茬纹理的皮
肤，方向和位置都由周围的皮肤推出来，不需要人去猜该抬多少、该压多少。

模型输入固定 512x512，所以按人脸框裁一块正方形等比缩到 512 再喂进去；补完的结果放大回
原尺寸，只取它的低频当基准，原图的高频（毛孔、胡茬、痣）另外叠回去，这样纹理还是这个人
自己的，网络只负责"这块皮肤该是什么颜色、什么明暗"。
"""

import os

import cv2
import numpy as np

SIDE = 512  # 模型固定输入边长
MASK_DILATE = 0.05  # 掩膜外扩（占 512 网格的比例）：把光斑的肩和阴影的边界一起包进洞里
VALID_SOFT = 0.004  # 有效区边界的软化半径（占裁剪框的比例）

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "lama_fp32.onnx"
)

_session = None
_unusable = False


def available():
    """权重在不在。真正的加载推迟到第一次要用时，失败则整个会话内不再尝试。"""
    return os.path.isfile(WEIGHTS_PATH)


def _get_session():
    global _session, _unusable
    if _session is not None or _unusable:
        return _session
    if not available():
        _unusable = True
        return None
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # BASIC 与 ALL 的推理速度实测一样，但省掉三秒初始化
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        _session = ort.InferenceSession(
            WEIGHTS_PATH, options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:  # 权重损坏 / 缺依赖：安静退回手写算法，不要让修图整步失败
        print(f"[retouch] LaMa 模型加载失败，改用内置算法：{error}")
        _session = None
        _unusable = True
    return _session


def prewarm():
    """
    在后台线程里先把模型读进内存。

    208MB 的权重从磁盘解出来要十几秒，赶在用户点生成的时候才加载，第一张照片会莫名其妙
    多等这么久。程序启动时机器是空的，这时候付掉。
    """
    if not available():
        return
    import threading

    threading.Thread(target=_get_session, daemon=True).start()


def inpaint(image_bgr, hole, box):
    """
    填补洞区。

    :param image_bgr: BGR 原图
    :param hole: 与 image_bgr 同尺寸的浮点掩膜，>0.5 处为要重画的地方
    :param box: 人脸裁剪框 (x0, y0, x1, y1)
    :return: (补好的全尺寸浮点图, 有效区) 或 None；有效区标记哪些像素真是网络画出来的
    """
    session = _get_session()
    if session is None:
        return None

    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = box
    side = int(max(x1 - x0, y1 - y0) * 1.06)
    side = max(32, min(side, h, w))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = int(np.clip(cx - side // 2, 0, w - side))
    sy0 = int(np.clip(cy - side // 2, 0, h - side))
    sx1, sy1 = sx0 + side, sy0 + side

    crop = image_bgr[sy0:sy1, sx0:sx1]
    mask = (hole[sy0:sy1, sx0:sx1] > 0.5).astype(np.uint8) * 255
    if mask.max() == 0:
        return None

    small = cv2.resize(crop, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    # 掩膜用最近邻放大后再外扩：线性插值会把细碎的光斑缩成一圈灰边，阈值一过就整片丢掉
    m = cv2.resize(mask, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    k = max(3, int(SIDE * MASK_DILATE) | 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    x = small[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    inputs = session.get_inputs()
    feed = {inputs[0].name: x, inputs[1].name: (m > 0)[None, None].astype(np.float32)}
    try:
        out = session.run(None, feed)[0][0]
    except Exception as error:
        print(f"[retouch] LaMa 推理失败，改用内置算法：{error}")
        return None

    # 有的导出把结果反归一化回 0~255，有的仍是 0~1，按实际范围判断
    span = 255.0 if float(out.max()) > 1.5 else 1.0
    filled = np.clip(out, 0.0, span) / span * 255.0
    filled = cv2.resize(
        filled.transpose(1, 2, 0)[:, :, ::-1], (side, side), interpolation=cv2.INTER_CUBIC
    )

    result = image_bgr.astype(np.float32).copy()
    result[sy0:sy1, sx0:sx1] = filled
    valid = np.zeros((h, w), np.float32)
    vm = cv2.resize((m > 0).astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    valid[sy0:sy1, sx0:sx1] = cv2.GaussianBlur(
        vm, (0, 0), max(1.0, side * VALID_SOFT)
    )
    return result, valid
