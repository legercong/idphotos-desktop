# 本文件为本仓库新增，不属于上游项目 HivisionIDPhotos。
# 许可遵循 Apache-2.0（见 LICENSE 与 NOTICE）。

"""
人脸分割：为高级修图（去阴影/去反光）提供"哪里是皮肤、五官、头发"的判据。

纯颜色判据在两种情况下会失准：一是脸上有强阴影时（暗侧皮肤的色度会向中性收缩），
二是皮肤出油反光时（亮斑本身几乎被冲成白色）。这两个恰恰是我们要处理的场景，
所以引入分割模型兜底。

模型：SegFormer-B5 @ CelebAMask-HQ（19 类，HuggingFace: Xenova/face-parsing，int8 量化版）。
权重放在 weights/face_parsing.onnx，随仓库分发。模型文件缺失或加载失败时本模块
所有接口返回 None，调用方自动退回颜色判据，不影响出片。
"""

import os

import cv2
import numpy as np

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "face_parsing.onnx"
)

# CelebAMask-HQ 19 类
BACKGROUND, SKIN, NOSE, EYE_GLASS, L_EYE, R_EYE, L_BROW, R_BROW, L_EAR, R_EAR = range(10)
MOUTH, U_LIP, L_LIP, HAIR, HAT, EAR_RING, NECK_L, NECK, CLOTH = range(10, 19)

# 真正用于估计光照的皮肤（鼻面也算皮肤，模型把它单独分出来是为了保留立体感）
SKIN_CLASSES = (SKIN, NOSE)
# 会被同一束光照整体抬升的范围：皮肤 + 五官 + 耳朵。不含头发、衣服和脖子。
# 头发和衣服本来就比脸暗，跟着脸一起抬会在阴影边界出现一圈亮晕；脖子处在下巴的投影里，
# 把它抬到脸的亮度会抹掉下颌线，脸变成一张浮在脖子上的贴纸。
LIFT_CLASSES = (
    SKIN,
    NOSE,
    L_EYE,
    R_EYE,
    L_BROW,
    R_BROW,
    L_EAR,
    R_EAR,
    MOUTH,
    U_LIP,
    L_LIP,
)
NECK_CLASSES = (NECK_L, NECK)
# 去反光/去阴影时要挡开一圈的五官本体：眼睛（压掉泪沟那层反光，眼下立刻浮出两团青灰）
# 和嘴（唇红本身就是一片镜面高光，压它等于把嘴压成一条灰印，交给修复网络重画更会把嘴唇
# 画薄、在上唇补出一片假胡茬）。眉毛不在内——额头的汗光往往就压在眉头上方的那一段，
# 把眉毛也算进保护区，正好把最该处理的一圈让开了。
PERI_FEATURE_CLASSES = (L_EYE, R_EYE, MOUTH, U_LIP, L_LIP)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_NET_SIZE = 512

_session = None
_load_tried = False


def _get_session():
    global _session, _load_tried
    if _load_tried:
        return _session
    _load_tried = True
    if not os.path.exists(WEIGHTS_PATH):
        print("[face_parsing] 未找到权重 %s，修图退回颜色判据" % WEIGHTS_PATH)
        return None
    try:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
        _session = onnxruntime.InferenceSession(
            WEIGHTS_PATH, options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:  # pragma: no cover - 取决于运行环境
        print("[face_parsing] 模型加载失败，修图退回颜色判据: %s" % error)
        _session = None
    return _session


def parse_face(image_bgr, box=None):
    """
    分割人脸。

    :param image_bgr: BGR 图像
    :param box: (x0,y0,x1,y1) 粗略人脸框，None 则用整幅图
    :return: 与输入同尺寸的 uint8 标签图（框外为 0=背景）；模型不可用时返回 None
    """
    session = _get_session()
    if session is None:
        return None

    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = _clamp_box(box, w, h)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return None

    resized = cv2.resize(crop, (_NET_SIZE, _NET_SIZE), interpolation=cv2.INTER_LINEAR)
    blob = (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _MEAN) / _STD
    blob = blob.transpose(2, 0, 1)[None].astype(np.float32)

    logits = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    # 先在 logits（低分辨率）上线性插值再取 argmax，边界比"标签最近邻放大"平滑得多
    up = cv2.resize(
        np.transpose(logits, (1, 2, 0)),
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    labels = np.zeros((h, w), np.uint8)
    labels[y0:y1, x0:x1] = np.argmax(up, axis=2).astype(np.uint8)
    return labels


def _clamp_box(box, w, h):
    if box is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1
