# 本文件为本仓库新增，不属于上游项目 HivisionIDPhotos。
# 许可遵循 Apache-2.0（见 LICENSE 与 NOTICE）。

"""
高级修图：去阴影（照度均衡）、去反光（皮肤高光消除）

两个算法都需要知道"哪里是皮肤"，而纯颜色判据恰好在我们要处理的两种照片上失准：
强阴影下暗侧皮肤的色度会向中性收缩，油光本身又被冲成近白色。所以判据优先取自
人脸分割模型（face_parsing），模型不可用时才退回颜色判据。

- 去阴影：投射阴影是"成片的、边界相对清楚的亮度亏缺"，五官的立体明暗则跟着几何走。
  投射阴影是低频的：它在皮肤上的表现是一整块照度整体偏低，而不是某个像素偏暗。所以先
  在皮肤上取局部均值，得到一张抹掉了五官和纹理、只留下光照的"照度面"；再用确定受光的
  那部分皮肤（第一轮按亮度分位数取，之后按上一轮找到的阴影区反取）做归一化卷积，把
  "这块皮肤没有阴影时该有多亮"外推出来。两者之差就是亏缺，按比值逐像素抬上去。
  增益本身是一张低频场，五官跟着周围的皮肤一起抬，鼻梁眼窝的立体感原样保留。
  抬升幅度另受两条约束：比局部照度还亮的像素只补底色不放大高光，且抬完不许超过受光面
  最亮那圈皮肤——否则阴影里的汗光会被乘成一块死白斑。窄条状的阴影（窗框压下来的那道）
  在宽窗口里会被两侧的受光皮肤平均掉，所以判定用宽窗口、增益在已判定的区域内改用窄窗口
  重测照度，两层各管各的尺度。
- 去反光：汗光油光的物理特征是"同亮度下比正常皮肤更不饱和"（白光是加性叠加的）。
  于是用哑光皮肤拟合出 S–V 参考曲线，与顶帽盈余一起组成一张连续的"证据场"，权重直接
  按证据强弱给——额头那层汗光是渐变的一整片，硬阈值掩膜只会圈住最重的几处，把油光压成
  几块比周围更暗更灰的补丁。消除分两层：低频层把亮斑自身的低通对齐到外推出的局部哑光底色，
  尖峰层单独削掉一颗颗的汗珠亮点；底线是任何像素都不许压到底色以下。色度另走一路，
  把被白光冲垮的色度低频整体换成附近哑光皮肤的色度（线性 B−Y / R−Y，换色不动亮度），
  因为压暗只能还原亮度，冲没的那点极差已经是噪声，照它放大补色只会补出洋红麻点。

判据用的都是低频量（照度面、顶帽盈余、参考底色），所以检测统一在缩小的工作网格上
完成，只把最后得到的增益/权重场放大回原尺寸逐像素施加。全分辨率上做百米级的形态学
和模糊既慢又没有额外精度。

检测出的阴影区和反光区最后交给一个在真实图像上训练出来的修复网络（lama_inpaint，LaMa）
补一遍：手写判据只说得出"这里比别处亮了多少、暗了多少"，说不出"这块皮肤本来长什么样"，
所以压平的反光会留下一块比周围更暗的灰斑，抬起来的阴影边界还挂着一道边。网络填出来的
是带血色、带纹理的皮肤。两者叠着用：增益先按透过率把整张脸（含五官）抬到位，网络的
低频再把皮肤那一块的明暗颜色校正回来，原图的高频始终留在手里，纹理还是这个人自己的。
模型权重缺失或加载失败时整条 AI 路自动跳过，只剩手写算法，功能不因此不可用。
"""

import cv2
import numpy as np

from hivision.plugin.beauty import face_parsing, lama_inpaint

# 二十万吨的权重从磁盘解出来要十几秒，别赶在用户点生成的时候才付这个钱
lama_inpaint.prewarm()

_LUMA_W = np.array([0.114, 0.587, 0.299], np.float32)
WORK_SIDE = 560  # 检测工作网格的短边上限

# 去阴影。尺度参数都以工作网格短边为单位，照片尺寸变化时行为一致
SHADOW_LEVEL_SIGMA = 0.010  # 亮度层轻模糊：去掉毛孔，留住阴影边界
SHADOW_ILLUM_SIGMA = 0.05  # 皮肤照度面的低通半径：抹掉五官明暗，只留光照
SHADOW_ILLUM_NARROW = 0.022  # 同一次再开一个窄窗口，专测窗框压下来的那种窄条阴影
SHADOW_ILLUM_EDGE = (14.0, 0.012)  # 保边照度面：亮度差小于它才平均、空间半径按短边给
SHADOW_SHADE_SOFT = 0.012  # 阴影连通域的软化半径：窄窗口只在这片区域里生效
SHADOW_TARGET_SIGMA = 0.12  # 受光面参考的归一化卷积半径
SHADOW_SEED_QUANTILE = 60  # 第一轮用亮度分位数选"肯定受光"的皮肤做参考种子
SHADOW_DEFICIT_FLOOR = 25.0  # 亮度亏缺下限，低于它的一律当作正常立体明暗
SHADOW_AREA_RANGE = (0.02, 0.55)  # 阴影面积占皮肤比例的合理区间
SHADOW_GAIN_CAP = 3.2  # 单轮最大抬升倍数：窗框挡出来的硬阴影要两档光，2.2 会在阴影里留一条边
SHADOW_CEIL_QUANTILE = 95.0  # 天花板取受光面皮肤亮度的这个分位：抬上去不许超过受光面最亮处
SHADOW_PASSES = 2  # 抬亮后重算亏缺再走一轮，把阴影深处也抬到位
SHADOW_RAMP = 0.45  # 抬升权重从 0.55 倍阈值连续升到 1.45 倍阈值
SHADOW_CHROMA = 0.6  # 色度对齐强度：阴影侧的色偏按这个比例拉向受光面
SHADOW_NECK_LIFT = 0.45  # 脖子只跟着抬一半权重，保住下颌线

# 去反光
SPEC_SCALES = (0.05, 0.12)  # 双尺度顶帽：细碎油光 + 成片汗斑
SPEC_DETECT_SMOOTH = 0.045  # 判据场的低通半径：判"有一片反光"，不判"这个像素比邻居亮"
SPEC_MATTE_MAX = 10.0  # 亮度盈余小于它才算"确认的哑光皮肤"样本
SPEC_MATTE_GAP = 12.0  # 比同亮度哑光皮肤淡过这个数的样本不算哑光
SPEC_SAT_GAP = 26.0  # 比同亮度哑光皮肤低多少饱和度算"满额反光"
SPEC_EXCESS_MIN = 30.0  # 亮度盈余到多少算"满额反光"
SPEC_STRONG_EV = 0.9  # 两维判据都到阈值的这个比例才算一处确凿反光，决定要不要动手
SPEC_MIN_AREA = 0.004  # 反光至少占皮肤的这个比例，孤立亮点不算
SPEC_WEIGHT_CAP = 1.0  # 满额证据、最高档，最多消除这个比例的白光
SPEC_EYE_GUARD = 0.02  # 眼周保护区的宽度
SPEC_CORR_SMOOTH = 0.02  # 改正场的低通半径
SPEC_AREA_MAX = 0.70  # 反光占皮肤超过它，说明整张脸都被判成反光，底色已无可信样本
SPEC_PEAK_GAIN = 0.7  # 汗珠尖峰（顶帽盈余）中削掉的比例
SPEC_CHROMA_MAX = 26.0  # 色度低频最多对齐多少，超过它说明底色样本混进了别处的气色
SPEC_LIFT_MAX = 110.0  # 单像素低频最多校正多少，超过它一定是底色估计错了
SPEC_BASE_SIGMA = 0.05  # 底色估计窗口：贴着亮斑外圈的那圈皮肤才是它的底色，开大了整脸平均
SPEC_BASE_EXCLUDE = 0.06  # 估底色前先把反光外扩这么多，连它自己的肩一起排除
SPEC_FLOOR_SIGMA = 0.025  # 底线用的窄窗口：贴着亮斑的那圈哑光皮肤
SPEC_BRIGHT_PCT = (30.0, 55.0)  # 底色样本的亮度加权区间（哑光皮肤的分位数）
SPEC_REACH = 0.05  # 洞从"确凿反光"外扩这么多，让渐变薄光也进到分级消除的范围里

# AI 修复（lama_inpaint）。检测仍走上面的手写判据，改正交给修复网络补出来的皮肤
AI_LOW_SPEC = 0.022  # 只拿得起这么粗的一层：网络补出来的胡茬颗粒不该盖过本人自己的
AI_HF_RADIUS = 0.010  # 留给原图的那一档纹理有多细（工作网格短边比例），比 AI_LOW_SPEC 细
AI_HF_KEEP = 0.75  # 网络接管的地方削掉这么多原图高频：光斑的麻点全在高频里，留太少就是没去掉
AI_HF_TAKE = 0.35  # 权重到这个比例，纹理就完全改由原图供，不再掺底色的
AI_SKIN_ERODE = 0.022  # 非皮肤外扩多少才算"离五官够远"，网络才准接管
AI_BLEND_CAP = 0.92  # 网络结果最多替换掉多少：永远留一点原图，不至于换了一张脸
AI_HOLE_MIN = 0.003  # 待填面积小于皮肤的这个比例就不值得调用模型
AI_LIMIT_MARGIN = (1.5, 20.0)  # 网络最多只能改判据实测到的光量这么多，再乘这个系数加这个底
AI_BACKSLIDE = 6.0  # 网络最多只能比原图亮这么多，再多的就是它自己编的
AI_OVER_MAX = 8.0  # 网络的亮度最多比收过边的底色高这么多，再亮就是留下一块浅斑
# 每个洞的"邻居水平"上下限。修复网络看不见嘴唇下面那道阴影是阴影、只看见一片暗，
# 就会把下巴那颗高光补成一块 120 档的黑斑（周围皮肤是 207）。手写算法同样会栽在这里：
# 它把那道暗折当成了这块皮肤的"哑光底色"。所以不管谁算的，改完都不许越过它自己邻居的水平。
# 两套取值，因为两个毛病要归的位置不同：反光是"这块皮本来就和邻居一样"，所以贴着邻居
# 的中位数收；阴影是"这块皮该抬到受光面的水平"，上限必须给到受光面偏亮那一侧才够抬。
SPEC_BOUND_PCT = (45.0, 62.0)  # 去反光：圈外皮肤的这两个分位分别是下限和上限
SPEC_BOUND_MARGIN = 9.0
SHADE_BOUND_PCT = (55.0, 85.0)  # 去阴影：同上
SHADE_BOUND_MARGIN = 18.0
AI_RING_SCALE = 0.06  # 圈的宽度（工作网格短边比例）
AI_SAT_PCT = 40.0  # 浓淡的下限取圈外皮肤极差的这个分位
AI_SAT_MARGIN = 6.0  # 允许比邻居再灰这么多
AI_BOUND_DETAIL = 0.012  # "这一片的平均水平"的低通半径（工作网格短边比例）
# 补色最多补到这个比例。全份取邻居的色是不行的：唇下那道暗折本来就比别处青灰，
# 邻居那一圈却带着唇色的红，整份平移过去就在下巴上糊出一片粉红——比灰斑更扎眼。
AI_CHROMA_CAP = 0.45


def _luma(image):
    return np.asarray(image, np.float32) @ _LUMA_W


def _to_work(image, scale):
    if scale >= 1.0:
        return image
    size = (max(8, int(round(image.shape[1] * scale))), max(8, int(round(image.shape[0] * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _to_full(field, shape):
    if field.shape[:2] == shape[:2]:
        return field
    return cv2.resize(field, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _subject_mask_from_alpha(image_rgba):
    """RGBA 的 alpha 通道即人像区域，模糊后作为软掩膜，避免在头发边缘产生接缝。"""
    alpha = image_rgba[:, :, 3].astype(np.float32) / 255.0
    h, w = alpha.shape
    k = max(3, int(min(h, w) * 0.02) | 1)
    return np.clip(cv2.GaussianBlur(alpha, (k, k), 0), 0.0, 1.0)


def _composite(bgr, alpha, mask, image_rgba):
    """
    只把人像区域内的处理结果换回去，区域外保持原样。

    掩膜按 50% 等值线收窄后再用：抠图边缘那一圈是前景与背景的半透明混合色，跟着一起
    增亮就会沿着人脸轮廓留下一条亮边，在浅色背景上尤其显眼。
    """
    m = np.clip((mask - 0.5) * 2.0, 0.0, 1.0)
    m4 = np.dstack([m] * 4)
    return (
        np.dstack([bgr, alpha]).astype(np.float32) * m4
        + image_rgba.astype(np.float32) * (1.0 - m4)
    ).astype(np.uint8)


def _skin_mask(bgr):
    """
    肤色判据，用归一化色度（r=R/(R+G+B) 等）而不是 YCrCb 绝对窗口：
    阴影/欠曝会让色度整体向中性值收缩，用绝对窗口判据时背光的那半边脸
    会被判成"非皮肤"，而它恰恰是最需要处理的部分。色度是比值，天然与亮度无关。
    """
    ch = bgr.astype(np.float32) + 1.0
    total = ch.sum(axis=2)
    r = ch[:, :, 2] / total
    g = ch[:, :, 1] / total
    b = ch[:, :, 0] / total
    return (
        (r > g)
        & (g > b)
        & ((r - g) > 0.02)
        & ((r - g) < 0.18)
        & (r > 0.36)
        & (b < 0.33)
    )


def _face_region(bgr, subject_mask):
    """颜色判据版脸部软掩膜（无分割模型时的兜底）：闭运算补五官，再按亮度剔掉头发。"""
    h, w = bgr.shape[:2]
    skin = _skin_mask(bgr).astype(np.uint8) * 255
    k = max(3, int(min(h, w) * 0.10) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    face = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    face = (face > 0) & (subject_mask > 0.5)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if np.any(skin > 0):
        floor = np.percentile(gray[skin > 0], 50) * 0.42
        face &= gray > floor
    return np.clip(
        cv2.GaussianBlur(face.astype(np.float32), (0, 0), max(2.0, k / 6.0)), 0, 1
    )


def _face_masks(bgr, subject, labels):
    """
    返回 (皮肤硬掩膜, 可抬亮硬掩膜, 脖子硬掩膜)。

    可抬亮区域 = 皮肤 + 五官 + 耳朵。五官必须跟着皮肤一起抬，否则皮肤变亮后眼睛会变成
    一块暗岛。脖子单独返回：它本来就处在下巴的投影里，整体抬到脸的亮度会抹掉下颌线、
    脸像一张浮在脖子上的贴纸，但一点不抬又会让脸和脖子分成两截，所以只按部分权重参与。
    """
    in_subject = subject > 0.5
    if labels is not None:
        skin = np.isin(labels, face_parsing.SKIN_CLASSES) & in_subject
        if skin.sum() > 0.002 * skin.size:
            return (
                skin,
                np.isin(labels, face_parsing.LIFT_CLASSES) & in_subject,
                np.isin(labels, face_parsing.NECK_CLASSES) & in_subject,
            )
    skin = _skin_mask(bgr) & in_subject
    return skin, _face_region(bgr, subject) > 0.5, np.zeros(skin.shape, bool)


def _soft_weight(mask, scale, short, feather=0.012):
    """
    掩膜 -> 工作网格上的软权重。

    模糊会把掩膜向外多摊出一圈，处理结果就沿着人脸轮廓留下一条亮边（抠图 alpha 在这一
    圈上还是半透明的）。所以模糊后按 50% 等值线重新收窄：过渡带仍在，但支撑范围不再变大。
    """
    w = _to_work(mask.astype(np.float32), scale)
    k = max(3, int(short * feather) | 1)
    w = cv2.GaussianBlur(w, (k, k), 0)
    return np.clip((w - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)


def _rough_face_box(bgr, subject):
    """
    用颜色判据粗定位人脸，给分割模型一个尽量紧的裁剪框。

    模型输入固定 512，框越紧脸上每平方毫米占的像素越多，五官边缘越准。
    """
    skin = _skin_mask(bgr) & (subject > 0.5)
    ys, xs = np.nonzero(skin)
    if xs.size < 0.001 * skin.size:
        ys, xs = np.nonzero(subject > 0.5)
        if xs.size == 0:
            return None
    h, w = bgr.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    span_x, span_y = x1 - x0, y1 - y0
    return (
        int(max(0, x0 - 0.25 * span_x)),
        int(max(0, y0 - 0.55 * span_y)),
        int(min(w, x1 + 0.25 * span_x)),
        int(min(h, y1 + 0.15 * span_y)),
    )


def _weighted_blur(img, weight, sigma):
    """归一化卷积：只在 weight>0 的区域里取平均，权重自身也参与模糊。"""
    num = cv2.GaussianBlur(img * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def _otsu(values):
    """一维 Otsu 阈值，用来在"正常明暗"和"成片的阴影"之间自动找分界。"""
    high = float(values.max()) + 1.0
    hist, _ = np.histogram(values, bins=256, range=(0.0, high))
    p = hist.astype(np.float64) / max(1, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * (np.arange(256) * high / 256))
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-9)
    return float(np.argmax(np.nan_to_num(between)) * high / 256)


def _skin_illuminance(luma, skin, short, sigma=SHADOW_ILLUM_SIGMA):
    """
    照度面：只在皮肤内取局部均值。

    归一化卷积的权重为 0 处不参与平均、也不会把头发背景带进来，所以眼睛、嘴唇、
    鼻孔这些"自身就该暗"的结构会被跳过，剩下的就是纯粹的低频光照。半径必须远小于
    阴影宽度（否则连受光面一起平均进来）、又远大于五官尺寸（否则抹不掉五官明暗），
    工作网格上皮肤约 300px 宽、阴影 200px 以上，0.05 倍短边正好落在中间。
    """
    return _weighted_blur(luma, skin.astype(np.float32), max(2.0, short * sigma))


def _clean_shadow_region(region, skin_pixels, short):
    """开运算去掉五官交界上的暗细条，再只保留成片的连通域。"""
    k = max(3, int(short * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    region = cv2.morphologyEx(region.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0
    count, tags, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    if count > 1:
        big = stats[:, cv2.CC_STAT_AREA] >= 0.02 * max(1, skin_pixels)
        region = big[tags] & (tags > 0)
    return region


def _lit_reference(illum, seeded, short):
    """
    "这些皮肤没有阴影时该有多亮"：用受光面皮肤的照度做归一化卷积外推。

    阴影区权重置 0，归一化卷积在权重为 0 处会自动向远处加权外推，所以半径可以大于
    阴影宽度，而不需要知道阴影到底有多大——这正是形态学包络做不到的（包络看不到超出
    自身半径以外的亮区，宽阴影的内部测不出亏缺，结果只抬得动阴影边缘一条边）。
    """
    return _weighted_blur(illum, seeded, max(2.0, short * SHADOW_TARGET_SIGMA))


def fill_with_ai(image_bgr, ai, weight, hole, limit=None, ref=None, skin=None,
                 bounds=None):
    """
    把修复网络补出来的皮肤合回原图：低频用网络的，高频仍用原图的。

    分频是关键。网络只看到 512 缩略图，它画的是"这一片该是这样的皮肤"，毛孔、胡茬、痣
    这些细节它无从得知；直接整块换上，人的脸就变成了一张磨平的假脸，还会凭空多出一片
    络腮胡。而汗光恰恰几乎全是低频的东西——一层白光，本身不带纹理。所以把低频交出去、
    把高频留在手里，两件事各归各的：网络负责"这块皮肤该多亮、什么色"，原图负责"这块
    皮肤长什么样"。分频的尺度取 AI_LOW_SPEC 这么粗：再细就把网络自己编的颗粒搬进来了。

    :param image_bgr: 底图（手写算法已经处理过的那一版）
    :param ai: (全尺寸浮点图, 有效区)，来自 lama_inpaint.inpaint
    :param weight: 0~1 混合权重，已含滑杆强度
    :param hole: 交给网络重画的地方，出了这里一律按底图算
    :param limit: 判据实测到的光量（亮度单位），网络改动的上限
    :param ref: 算上限时对齐的原图，默认用底图
    :param skin: 只有这里允许网络接管（皮肤以外的地方仍归手写增益）
    :param bounds: (下限, 上限) 邻居亮度，用来给网络的水平定锚
    """
    filled, valid = ai
    short = min(image_bgr.shape[:2])
    sigma = max(1.5, short * AI_LOW_SPEC)
    # 留作纹理的那一档必须比网络的低频更细。按 sigma 同一半径取高频，得到的"高频"里还
    # 带着光斑本身三四十像素那一鼓——低频已经换成哑光皮肤了，这一鼓又从高频加回来，
    # 下巴那块补完比周围亮八档就是这么来的。半径收到 AI_HF_RADIUS，只剩毛孔和汗毛这一级。
    r_tex = max(1.2, short * AI_HF_RADIUS)
    src = image_bgr.astype(np.float32)
    # 有效区是外扩过的洞，权重是判据算出来的光斑本体，取小的那个：
    # 网络没画过的地方一律不许动，判据够到洞外面也没用。
    # 再乘一层皮肤掩膜：网络只许接管皮肤。眉毛、眼睛、嘴唇一律不许它碰——照 512 缩略图
    # 上的一片暗就重画出一道白楞。
    gate = 1.0 if skin is None else np.asarray(skin, np.float32)
    w = np.clip(np.minimum(weight, cv2.GaussianBlur(hole.astype(np.float32), (0, 0), sigma))
                * valid * gate, 0.0, AI_BLEND_CAP)[:, :, None]
    # 底色只交出低频，它的颗粒一概不要。手写增益在光斑里算错的不只是水平，还有颗粒：
    # 唇下那道暗折被它当成了哑光底色，除完高光以后那片皮肤剩下一层青灰的斑驳。这层斑驳
    # 按多少比例混进来，就照原样在网面上留下多少——抬亮度、补色度都救不回来，因为它不是
    # 一个可以整体搬正的偏移，而是错在每一粒上。颗粒改从原图取，且只取明暗那一半：原图
    # 高频里的色度是反光自己的蓝白麻点，搬过去等于把光斑又画回去。
    tex = np.asarray(ref if ref is not None else image_bgr, np.float32)
    low_src = cv2.GaussianBlur(src, (0, 0), sigma)
    hf_src = src - low_src
    hf2 = _luma(tex - cv2.GaussianBlur(tex, (0, 0), r_tex))
    hf = hf2[:, :, None]
    # 网络的亮度水平要有个锚，否则它把油光去掉、却留下一块比周围还亮的浅斑，边界还是个
    # 圆（下巴测到 dY 从 −6 翻成 +9）。锚取底色：它的颗粒是错的（所以高频一概不取它的），
    # 水平却是按判据实测的哑光底色算的。先拿邻居上下限把底色收一道边——它在唇下那道暗折
    # 上会算得偏暗，不收就没法当锚——再许网络比这个锚亮出 AI_OVER_MAX，多出来的就是编的。
    # 只平移亮度，色度仍归网络：底色那层的色度偏青灰，带过去等于把斑驳又请回来。
    cap = 0.0
    if bounds is not None:
        floor, ceil = bounds
        y0 = _luma(low_src)
        y0 = cv2.GaussianBlur(np.clip(y0, floor, ceil), (0, 0), sigma)
        low_src = low_src + (y0 - _luma(low_src))[:, :, None]
        low_ai = cv2.GaussianBlur(filled, (0, 0), sigma)
        y1 = _luma(low_src * (1.0 - w) + low_ai * w)
        cap = (np.minimum(y1, y0 + AI_OVER_MAX) - y1)[:, :, None]
    else:
        low_ai = cv2.GaussianBlur(filled, (0, 0), sigma)
    # 换纹理要比换明暗快：网络刚接管一点点，颗粒就该完全改由原图供，否则底色那份错颗粒
    # 按混合比例留下多少就显多少。但两条斜坡都必须从 0 起步——权重为零的地方（比如被
    # 皮肤掩膜挡开的眉毛）整条链路必须原样吐回底色，动一下高频就等于把增益抬起来的暗
    # 眉毛洗成一道灰楞，这比光斑里的麻点显眼得多。
    u = np.clip(w / AI_HF_TAKE, 0.0, 1.0)
    # 只削高频里"往上"的那一半。汗光是叠在皮肤上的一层白光，它的高频全是正向的尖刺；
    # 而眉毛、胡茬、痣、毛孔这些是往下的。整份按比例削，滑杆一高就连着眉毛一起磨没了
    # （测到 12~15 档时眉间那撮汗毛被抹平）。减去自身低频再削，是为了只去刺不去底——
    # 底那一层本来就已经交给网络的低频了，这里再削一次会把同一块光斑扣两遍。
    pos = np.maximum(hf2, 0.0)
    spikes = (pos - cv2.GaussianBlur(pos, (0, 0), sigma))[:, :, None]
    out = (low_src * (1.0 - w) + low_ai * w + cap
           + hf_src * (1.0 - u) + (hf - AI_HF_KEEP * spikes) * u)
    if limit is not None:
        # 网络改动的幅度不许超过判据实测到的光量。它看不见胡茬，就会把上唇补成一片黑；
        # 有了这道闸，最坏也只是没去干净，不会凭空画出一块脏东西。
        anchor = tex
        band = limit[:, :, None] * AI_LIMIT_MARGIN[0] + AI_LIMIT_MARGIN[1]
        out = np.clip(out, anchor - band, anchor + band)
    # 更根本的一条：只许往判据指的方向改，反方向一步都不许走。去反光是把亮处压下去，
    # 网络却在眉骨上照着缩略图旁边的受光面补出一道白楞。上下限管不住这种"方向就错了"
    # 的改动——它没越过邻居的水平，只是在原地倒着走。所以反方向只留一点容差，其余一律
    # 不许倒退。
    out = np.minimum(out, tex + AI_BACKSLIDE)
    return np.clip(out, 0, 255).astype(np.uint8)


def _hole_bounds(luma, spread, ok, hole, short, pct, margin, own_cap=False):
    """
    逐洞算出"圈外皮肤的水平"，作为这块皮肤改动的上下限。

    按连通域一个一个算，不用全脸一个数：额头和下巴的亮度本来就差一截。取分位数而不是
    均值，是因为圈外那一圈里总有鼻翼、唇下这几道正常暗折，均值会被它们拖到 100 几，
    照着它压就等于在高光的位置挖一个洞。

    除了明暗还管浓淡。汗光把皮肤冲成近白色，去掉它以后亮度回到了邻居的水平，可颜色回不
    来——网络补的是灰的，手写底色取到的样本又混着唇下那道暗折。亮度对、饱和度只有邻居
    三成的那块皮，看着就是一块污渍或者一块淤青，比原来的油光还扎眼。

    own_cap 是给去反光这类"只许减光"的改正加的方向闸。邻居那一圈跨不过光照的边界：下颌
    那块斑自己就压在受光颊部投下的阴影边上，一圈邻居取到的全是亮皮（下限 194），而它自己
    的暗半边只有 84。下限是按洞给的常数，照着它抬，就把一片本该在阴影里的下颌抬成一块边界
    发硬的浅斑——比原来的反光显眼得多。所以再拿这块皮肤自己的低频亮度封一次顶：下限只在
    它本来就比这块皮暗时才起作用（那正是"压过头、要往回抬"的情形），永远不许反过来把皮肤
    抬到它自己原来的水平以上。封顶用的是同一条低通，跟下面比对的对象一致。
    """
    floor = np.zeros(luma.shape, np.float32)
    ceil = np.full(luma.shape, 255.0, np.float32)
    sat = np.zeros(luma.shape, np.float32)
    count, tags = cv2.connectedComponents(hole.astype(np.uint8), 8)
    k = max(3, int(short * AI_RING_SCALE) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for i in range(1, count):
        blob = tags == i
        if blob.sum() < 0.0005 * luma.size:
            continue
        ring = (cv2.dilate(blob.astype(np.uint8), kernel) > 0) & ok & ~blob
        if ring.sum() < 300:
            continue
        lo, hi = np.percentile(luma[ring], pct)
        floor[blob] = lo - margin
        ceil[blob] = hi + margin
        sat[blob] = max(0.0, np.percentile(spread[ring], AI_SAT_PCT) - AI_SAT_MARGIN)
    if own_cap:
        sigma = max(2.0, short * AI_BOUND_DETAIL)
        floor = np.minimum(floor, cv2.GaussianBlur(luma, (0, 0), sigma))
    return floor, ceil, sat


def _keep_within_neighbours(image_bgr, floor, ceil, sat_floor, mask, ok):
    """
    把越界的那一片按邻居的水平搬回来，亮度浓淡都只搬低频，纹理一律不动。

    为什么必须只搬低频：这块下巴的暗折被手写算法取成了"哑光底色"，整片压到 110 档，
    而它周围的皮肤是 165。逐像素夹到下限听着对症，实际是把每一根胡茬都钉死在下限上——
    斑块立刻变成一块没有毛孔的浅色贴片，比原来的黑斑更像污渍。改成"这一片的平均水平
    越界多少就整体搬多少"，暗折抬上来了，胡茬、痣、毛孔还留在原来的对比上。

    浓淡同理，而且不能乘：被汗光冲成灰白的皮，色差本来就只剩几点，乘两倍还是灰的。
    只能把它的低频色度平移回邻居皮肤的颜色上，平移量按"差多少"给——本来就够浓的一点
    不动，只剩亮度的地方几乎整份取邻居的色。绿道按另两道的改动反算，全程不改亮度。
    """
    f = image_bgr.astype(np.float32)
    short = min(f.shape[:2])
    sigma = max(2.0, short * AI_BOUND_DETAIL)
    soft = cv2.GaussianBlur(mask, (0, 0), max(1.5, short * 0.004))[:, :, None]
    low_y = cv2.GaussianBlur(_luma(f), (0, 0), sigma)
    f = f + (np.clip(low_y, floor, ceil) - low_y)[:, :, None] * soft

    y = _luma(f)
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    spread = low.max(axis=2) - low.min(axis=2)
    need = np.clip((sat_floor - spread) / np.maximum(sat_floor, 1e-3), 0.0, AI_CHROMA_CAP)
    need = cv2.GaussianBlur(need, (0, 0), sigma) * soft[:, :, 0]
    weight = np.asarray(ok, np.float32) * (1.0 - soft[:, :, 0])
    den = np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 0.02)
    db = (cv2.GaussianBlur((f[:, :, 0] - y) * weight, (0, 0), sigma) / den
          - low[:, :, 0] + y) * need
    dr = (cv2.GaussianBlur((f[:, :, 2] - y) * weight, (0, 0), sigma) / den
          - low[:, :, 2] + y) * need
    out = np.stack(
        [f[:, :, 0] + db,
         f[:, :, 1] - (db * _LUMA_W[0] + dr * _LUMA_W[2]) / _LUMA_W[1],
         f[:, :, 2] + dr],
        axis=2,
    )
    return np.clip(out, 0, 255).astype(np.uint8)




def _ai_skin_gate(skin_full, subject):
    """
    允许修复网络接管的地方：皮肤，而且离一切非皮肤至少一圈。

    分割标签只有 512，边界粗。眉毛那么窄的东西，外缘十几像素被标成皮肤是常事——网络
    照着这块"皮肤"重画，就在真眉毛上方描出一道浅色楞，比原来的阴影还显眼。头发、眼睛、
    嘴唇同理。所以挡的是"非皮肤外扩一圈"，而不是把皮肤掩膜往里缩：皮肤掩膜本身就带一圈
    为过渡留的余量，缩它等于在额头、脸颊中间切出一道硬边，整张脸立刻像贴了张膜。

    场必须是软的（外扩以后再高斯一遍）：二值门直接乘进混合权重，边界就是可见的接缝。
    """
    e = max(3, int(min(skin_full.shape[:2]) * AI_SKIN_ERODE) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))
    excl = cv2.dilate(((subject > 0.5) & ~skin_full).astype(np.uint8), kernel)
    return 1.0 - np.clip(cv2.GaussianBlur(excl.astype(np.float32), (0, 0), e * 0.7), 0.0, 1.0)


def detect_shadow(image_rgba, strength, labels=None):
    """
    检测人脸上的投射阴影（阴阳脸、下巴投影、侧光等），并顺带算出不依赖模型的经典改正结果。

    :param image_rgba: RGBA 图像
    :param strength: 0-15，0 不处理，15 抬到受光面水平
    :param labels: 人脸分割标签图，None 时退回颜色判据
    :return: None 表示这张图没有需要动的阴影
    """
    if strength <= 0:
        return None

    shape = image_rgba.shape
    bgr = image_rgba[:, :, :3]
    subject = _subject_mask_from_alpha(image_rgba)
    skin_full, lift_full, neck_full = _face_masks(bgr, subject, labels)
    if skin_full.sum() < 0.002 * skin_full.size:
        return None

    scale = min(1.0, WORK_SIDE / float(min(shape[:2])))
    skin = _to_work(skin_full.astype(np.float32), scale) > 0.5
    short = min(skin.shape[:2])
    face_w = _soft_weight(lift_full, scale, short)
    lift_area = np.clip(face_w + SHADOW_NECK_LIFT * _soft_weight(neck_full, scale, short), 0, 1)
    # 参考亮度只能取自脸上（不含脖子），否则暗脖子会把"受光面该有多亮"一起拖低
    face_only = face_w > 0.9
    ratio = min(float(strength) / 15.0, 1.0)

    current = bgr.astype(np.float32)
    mul = np.ones_like(current)
    work0 = _to_work(bgr, scale)
    luma0 = _luma(work0)
    spread0 = work0.max(axis=2) - work0.min(axis=2)
    lift = np.zeros(skin.shape, np.float32)
    hole = np.zeros(skin.shape, bool)
    dmax = np.zeros(skin.shape, np.float32)
    skin_pixels = int(skin.sum())
    region = None

    for _ in range(SHADOW_PASSES):
        small = _to_work(current, scale)
        level = cv2.GaussianBlur(_luma(small), (0, 0), short * SHADOW_LEVEL_SIGMA)
        illum = _skin_illuminance(level, skin, short)
        if region is None:
            q = float(np.percentile(illum[skin], SHADOW_SEED_QUANTILE))
            # 软种子：亮度在分位数以上才参与，越亮权重越高，避免把过渡带算进参考
            seeded = face_only * np.clip((illum - q) / max(8.0, 0.12 * q), 0.0, 1.0)
            if float(seeded.sum()) < 0.02 * skin_pixels:
                break
        else:
            blocked = cv2.GaussianBlur(region.astype(np.float32), (0, 0), short * 0.02)
            seeded = face_only * np.maximum(1.0 - blocked, 0.0)
        ref = _lit_reference(illum, seeded, short)

        deficit = np.maximum(ref - illum, 0.0)
        threshold = max(_otsu(deficit[face_only]), SHADOW_DEFICIT_FLOOR)
        region = _clean_shadow_region(face_only & (deficit > threshold), skin_pixels, short)
        area = region.sum() / max(1, skin_pixels)
        if area < SHADOW_AREA_RANGE[0] or area > SHADOW_AREA_RANGE[1]:
            break
        if float(deficit[region].mean()) < 1.5 * threshold:
            break  # 只是整张脸均匀的轻微落差，不是投射阴影，别把它推平

        # 增益这一层另用一个更窄的照度面。窗框压下来的那道窄条只有二十来像素宽，
        # 0.05 倍短边的窗口会把它连同两侧的受光皮肤一起平均掉，测出的亏缺只剩一半，
        # 抬完仍留一道边。窄窗口只在已经判定为阴影的地方生效：出了这片区域一律按宽窗口
        # 算，因为窄窗口对鼻梁、眼窝这些正常的立体明暗一样起反应，拿它来判定会把一张本来
        # 均匀的脸也找出一堆"阴影"，推平了反而像戴面具。
        # 但高斯窄窗口只能"变窄"，不能"变直"：窄条两侧亮度差四十档，任何对称窗口都会把
        # 这条边糊掉，糊掉的宽度就是抬完后留在原地的那道边。所以再加一面保边的照度面——
        # 同侧的毛孔、眉毛一律抹平，跨不过阴影那条硬边。于是阴影深处拿到深处该有的增益，
        # 紧挨着的那块亮斑却还是 gain 1，边仍然是边，只是两边各归各位。
        illum_g = np.minimum(
            illum, _skin_illuminance(level, skin, short, SHADOW_ILLUM_NARROW)
        )
        edge = cv2.bilateralFilter(
            np.clip(level, 0.0, 255.0).astype(np.float32), 0,
            SHADOW_ILLUM_EDGE[0], short * SHADOW_ILLUM_EDGE[1])
        illum_g = np.minimum(illum_g, edge)
        in_shade = cv2.GaussianBlur(
            region.astype(np.float32), (0, 0), max(2.0, short * SHADOW_SHADE_SOFT)
        )
        ramp_src = deficit + np.maximum(ref - illum_g - deficit, 0.0) * in_shade

        # 增益按"整体除以透过率"来算：阴影里的一切漫反射都被同一个系数压暗，所以眼睛、
        # 鼻翼这些本来就该暗的结构跟着缩放才是对的，改成加性提亮会把眼白洗成灰片。
        # 但亮斑不行：汗光和环境光走的是镜面反射，投射物挡不住它，它在阴影里本来就已经
        # 是受光面的水平，再乘一遍就过曝成一块白斑。所以比局部照度还亮的像素另给一条
        # 上限——最多抬到"受光参考 + 它自己的高出量"，只补底色不放大高光。
        base_gain = ref / np.maximum(illum_g, 1.0)
        hl_gain = (ref + np.maximum(level - illum_g, 0.0)) / np.maximum(level, 1.0)
        gain = np.clip(np.minimum(base_gain, hl_gain), 1.0, SHADOW_GAIN_CAP)
        # 再加一条天花板：抬上去的像素不许超过受光面最亮那圈皮肤的亮度。
        # 上面两条都算在亮度（Y）上，而皮肤的最大通道比 Y 高两成，阴影里的亮斑按 Y 算
        # 没超、按 R 算就已经顶到 255 之外，出来就是一块死白的橘斑——下颚那处最明显。
        vmax_w = small.max(axis=2)
        lit_top = face_only & ~region
        if lit_top.sum() > 200:
            ceiling = float(np.percentile(vmax_w[lit_top], SHADOW_CEIL_QUANTILE))
            gain = np.minimum(gain, np.maximum(1.0, ceiling / np.maximum(vmax_w, 1.0)))
        # 抬升权重直接由亏缺做连续量：过渡带就落在阴影的真实边界上，不会多出一道接缝
        ramp = np.clip(
            (ramp_src - threshold * (1.0 - SHADOW_RAMP)) / max(1.0, threshold * 2 * SHADOW_RAMP),
            0.0,
            1.0,
        )
        weight = ramp * ramp * (3.0 - 2.0 * ramp) * lift_area * ratio
        eff = 1.0 + (gain - 1.0) * weight
        eff = cv2.GaussianBlur(eff, (0, 0), short * 0.006)
        eff = _to_full(eff, shape)[:, :, None]
        current = current * eff
        mul = mul * eff
        lift = np.maximum(lift, weight)
        hole |= region & skin
        dmax = np.maximum(dmax, deficit)

    if not lift.any():
        return None

    # 阴影侧的色度和受光面本来就不同（遮挡物与天空光的色温差），只按亮度抬会留下
    # 一块偏红或偏蓝的脸。这里把三通道按"抬亮后与受光面的均值比"反向校正，
    # 并用几何均值归一，保证只改色偏不改亮度。
    lift_w = _to_full(lift, shape[:2])
    moved = lift_w > 0.3
    lit = skin_full & (lift_w < 0.05)
    if moved.sum() > 500 and lit.sum() > 500:
        factor = current[lit].mean(axis=0) / np.maximum(current[moved & skin_full].mean(axis=0), 1.0)
        factor = factor / max(float(np.prod(factor) ** (1.0 / 3.0)), 1e-3)
        # 对色温只许落在皮肤和脖子上。抬升权重是软掩膜，下颌轮廓外必然溢出一圈几像素的
        # 过渡，那圈里是头发和衣领——把"阴影侧比受光面偏冷"这个差乘到深色的头发上，就得到
        # 一道贴着脸的橄榄绿，比原来的阴影还显眼。亮度增益跟着外溢是对的（阴影本来就压在
        # 头发上），只有色温这一步是脸自己的事。
        sn = np.clip(skin_full.astype(np.float32) + neck_full.astype(np.float32), 0.0, 1.0)
        sn = cv2.GaussianBlur(sn, (0, 0), max(2.0, min(shape[:2]) * 0.006))
        chroma = 1.0 + SHADOW_CHROMA * (factor - 1.0) * (lift_w * sn)[:, :, None]
        current = current * chroma
        mul = mul * chroma

    matte = skin & (dmax < 0.5 * SHADOW_DEFICIT_FLOOR)
    floor, ceil, sat = _hole_bounds(
        luma0, spread0, matte, hole, short, SHADE_BOUND_PCT, SHADE_BOUND_MARGIN)
    return {
        "hole": _to_full(hole.astype(np.float32), shape[:2]) > 0.5,
        "floor": _to_full(floor, shape[:2]),
        "ceil": _to_full(ceil, shape[:2]),
        "sat": _to_full(sat, shape[:2]),
        "ok": _to_full(matte.astype(np.float32), shape[:2]) > 0.5,
        "mul": mul,
    }


def apply_shadow(detection, image_rgba):
    """
    把 detect_shadow 的结果落到图上。

    去阴影不走修复网络，只走手写的透过率增益，再用邻居界限收住。测下来的道理很直白：
    投射阴影不是"这块皮肤不见了"，是"这块皮肤被挡了一半光"，网络只会把它当成一个洞去
    猜里面长什么样。而阴影像上面这张测试图里那样铺满半边脸时，洞里连眼睛和眉毛都在内，
    网络就把那只眼睛描成了一片糊，眉毛上方多一道浅楞——比原来的阴影还显眼。增益是整片
    按同一个比例抬，五官跟着一起亮，物理上对，画面上也不留痕。

    落图时是把增益乘在传进来的这张图上，而不是直接用一份预先算好的成品：两个滑杆同时
    开时，去反光已经先动过这张图了，拿原图的成品盖上去等于把刚去掉的油光又贴回来。
    阴影本身是纯乘性的（透过率），所以乘到别的底图上照样成立。
    """
    if detection is None:
        return image_rgba
    bgr = np.clip(image_rgba[:, :, :3].astype(np.float32) * detection["mul"], 0, 255)
    bgr = _keep_within_neighbours(
        bgr, detection["floor"], detection["ceil"], detection["sat"],
        detection["hole"].astype(np.float32), detection["ok"],
    )
    subject = _subject_mask_from_alpha(image_rgba)
    return _composite(bgr, image_rgba[:, :, 3], subject, image_rgba)


def remove_shadow(image_rgba, strength, labels=None):
    """不依赖修复模型的去阴影（外部直接调用时的入口）。"""
    return apply_shadow(detect_shadow(image_rgba, strength, labels), image_rgba)


def _tophat(channel, short, scale):
    k = max(3, int(short * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)


def _matte_saturation_curve(v_matte, s_matte):
    """
    拟合"哑光皮肤在该亮度下应有的饱和度"曲线，返回 256 长的查找表。

    只喂确认无反光的样本（亮度盈余小的皮肤），否则亮斑自己会把曲线拉低，
    变成"拿反光跟反光比"，什么也检不出来。
    """
    if s_matte.size < 300:
        return np.full(256, 70.0, np.float32)
    edges = np.arange(0, 273, 16)
    index = np.clip(np.digitize(v_matte, edges) - 1, 0, len(edges) - 2)
    levels = (np.arange(len(edges) - 1) * 16 + 8).astype(np.float32)
    medians = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = index == b
        if sel.sum() >= 300:
            medians[b] = np.median(s_matte[sel])
    known = ~np.isnan(medians)
    if known.sum() < 2:
        return np.full(256, float(np.median(s_matte)), np.float32)
    return np.interp(np.arange(256, dtype=np.float32), levels[known], medians[known]).astype(np.float32)


def _chroma_lowpass_delta(field, keep, sigma, smooth):
    """把这一路色度的低频整体对齐到附近哑光皮肤的水平，高频细节不在差里，原样保留。"""
    base = _weighted_blur(field, keep, sigma)
    return np.clip(base - cv2.GaussianBlur(field, smooth, 0),
                   -SPEC_CHROMA_MAX, SPEC_CHROMA_MAX)


def detect_specular(image_rgba, strength, labels=None):
    """
    检测并消除皮肤上的反光/汗光/油光（不依赖模型的那一路），同时给出反光区域。

    :param image_rgba: RGBA 图像
    :param strength: 0-15，判定标准固定，强度只控制消除力度
    :param labels: 人脸分割标签图，None 时退回颜色判据
    :return: None 表示这张图没有成片的反光
    """
    if strength <= 0:
        return None

    shape = image_rgba.shape
    bgr = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3]
    subject = _subject_mask_from_alpha(image_rgba)
    skin_full, _, _ = _face_masks(bgr, subject, labels)
    if skin_full.sum() < 0.002 * skin_full.size:
        return None

    ratio = float(strength) / 15.0
    scale = min(1.0, WORK_SIDE / float(min(shape[:2])))
    skin = _to_work(skin_full.astype(np.float32), scale) > 0.5
    short = min(skin.shape[:2])
    small = _to_work(bgr, scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    # 亮度盈余用双尺度顶帽：小尺度抓细碎油光，大尺度抓成片汗斑。
    # 全是相对量，所以不依赖"多亮才算反光"这种在真实照片上根本站不住的经验阈值。
    excess = np.maximum.reduce([_tophat(v, short, f) for f in SPEC_SCALES]).astype(np.float32)
    matte = skin & (excess < SPEC_MATTE_MAX)
    if matte.sum() < 2000:
        matte = skin
    curve = _matte_saturation_curve(v[matte], s[matte])
    vi = v.astype(np.uint8)
    gap = curve[vi] - s

    # 判据取低通后的场，因为要判的是"这里有一片反光"，而不是"这个像素比邻居亮"。
    # 顶帽在汗斑最中心、尤其过曝成一片白的那个平顶上响应最低（它比核还宽，形态学把
    # 平顶当成局部正常亮度了），照原始场阈值化就得到一圈中空的红环：环被压暗、中间的
    # 白斑原样留下，处理完比不处理还显眼。低通把平顶和环融成一个实心团。
    ds = (max(3, int(short * SPEC_DETECT_SMOOTH) | 1),) * 2
    gap_d = cv2.GaussianBlur(gap, ds, 0)
    # 取"成片"与"单点"两个版本的大者：只用低通版，额头上一颗颗小汗珠会被抹平到阈值以下，
    # 结果是它们周围的皮肤被压暗、汗珠本身原样留着，比不去还显眼；只用原始版，
    # 大片汗光的中心（平顶、顶帽无响应）又会漏成空洞。
    excess_d = np.maximum(excess, cv2.GaussianBlur(excess, ds, 0))
    # 两路判据各出一维证据：反光既是"比同亮度的哑光皮肤更不饱和"，也是"比周围更亮"，
    # 两个都到阈值才算满额。归一到 0~1 后相乘，得到一张连续的证据场。
    # 权重直接照这张连续场给，而不是先阈值化成一块掩膜再羽化：额头那层汗光是渐变的一整片，
    # 越往中心越亮，硬掩膜只圈得住最重的几处，于是重的被压成几块比周围更暗更灰的补丁、
    # 轻的原样留着，比不去还难看。连续权重顺着光的形状分级消除，边缘自己就接上了。
    ev = np.clip(gap_d / SPEC_SAT_GAP, 0.0, 1.0) * np.clip(excess_d / SPEC_EXCESS_MIN, 0.0, 1.0)
    ev = ev * skin
    # 要不要动手看的是"有没有一处两维都判满的反光"，而不是证据场的积分——一张干净的脸
    # 到处都会冒出一点点去饱和，逐像素加权时它们各自都不起眼，加起来却足够把全脸压暗。
    strong = skin & (gap_d > SPEC_SAT_GAP * SPEC_STRONG_EV) \
        & (excess_d > SPEC_EXCESS_MIN * SPEC_STRONG_EV)
    if not (SPEC_MIN_AREA * skin.sum() <= strong.sum() <= SPEC_AREA_MAX * skin.sum()):
        # 要么根本没有成片的反光，要么整张脸都被判成反光——后者说明底色已无可信样本，
        # 照它压等于把全脸调暗，宁可不碰。
        return None

    # 强度直接按比例给：满额证据、最高档就把这层白光削到底，往下每一档少削一成。
    # 不留"永远保留一成"的保底，是因为压到多少由下面的底色底线说了算，滑杆只管彻底程度。
    weight = np.minimum(ev, SPEC_WEIGHT_CAP) * ratio
    # 消除的是叠加在皮肤上的白光，只能落在皮肤上：漫到眼窝、眉毛和嘴唇一起压暗，
    # 整张脸就会发灰、眼睛下面出现两团阴影。
    weight = weight * _soft_weight(skin_full, scale, short)
    # 五官周围挡开一圈。眼睛：泪沟那层反光恰恰是"精神"的来源，压掉它眼下立刻浮出两团
    # 青灰，证件照里比油光本身更难看。嘴唇：唇红本身就是一片镜面高光，跟着压会把嘴压成
    # 一条灰印，而交给修复网络重画更是灾难——它会把嘴唇画薄、把上唇补出一片假胡茬。
    k = max(3, int(short * SPEC_EYE_GUARD) | 1)
    if labels is None:
        # 没有分割模型时按亮度粗圈出眼周：暗的眶下皮肤会被归一化卷积拖低底色，
        # 跟着一起压暗就在眼下留两团更深的阴影，宁可少去一点也不要把人弄成熊猫眼。
        p = _to_full(skin.astype(np.float32), shape[:2]) * (
            _to_full(v, shape[:2]) < np.percentile(v[skin], 40.0))
    else:
        peri = np.isin(labels, face_parsing.PERI_FEATURE_CLASSES) & (subject > 0.5)
        p = peri.astype(np.float32) if peri.sum() > 0.0002 * peri.size else None
    guard = np.ones(skin.shape, np.float32)
    if p is not None:
        p = cv2.dilate(_to_work(p, scale), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        guard = 1.0 - np.clip(cv2.GaussianBlur(p, (k, k), 0), 0.0, 1.0)
        weight = weight * guard

    # 底色 = 附近"确认哑光"皮肤的水平，而且要把有反光的地方排除在样本之外，还要外扩一圈
    # 连它的肩一起排除。两个条件缺一不可：只扣亮斑本身，估计窗口落在亮斑自己的肩上，
    # 底色等于"半亮"，削下来还是比真正的哑光皮肤亮一截，额头中间留一块白斑；只用 excess
    # 小的样本，半亮的过渡带一旦进样本，同样把底色抬上去。汗光是一整片渐变，越往中心越亮，
    # 必须拿远处的哑光皮肤当参照才压得平。
    # 排除的是"确凿的反光"，不是"有一点点证据的皮肤"。整片额头都带一层薄薄的汗光，
    # 按证据强弱连续扣除样本的话，额头自己就被赶出了参照集，底色只剩颧骨以下那圈更暗的
    # 皮肤——照它压，额头压到 170，比真正的哑光额头暗了二十几，油光变成一片青灰。
    soft = cv2.GaussianBlur(
        cv2.dilate(np.clip((ev - SPEC_STRONG_EV) / max(1e-3, 1.0 - SPEC_STRONG_EV), 0.0, 1.0),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (max(3, int(short * SPEC_BASE_EXCLUDE) | 1),) * 2)),
        (0, 0), short * 0.02)
    # 底色样本还要再过一道色相关：摊开的一大片汗光顶帽根本不响应（它比核还宽），
    # 在 excess 那一关是"合格的哑光"，可它比同亮度的哑光皮肤淡了几十个饱和度，一眼就是
    # 被光冲过的。让它进参照，底色被这片光自己抬起来，压完还是亮的。
    # 顺序上必须先有曲线才有 gap，所以曲线先用第一道判据拟合，再回头收紧样本。
    matte = matte & (gap < SPEC_MATTE_GAP)
    if matte.sum() < 2000:
        matte = skin & (excess < SPEC_MATTE_MAX)
    # 再剔掉偏暗的哑光皮肤：颧骨旁边就是鼻唇沟和眼窝，那几处暗皮一旦进窗口，
    # 底色被拖低，"比底色亮"就不再等于"有反光"，于是把好好的颧骨压成一块比周围更暗的灰斑——
    # 比原来的油光还难看。所以只取哑光皮肤里中上亮度的那部分当参照。
    vm = v[matte]
    lo, hi = np.percentile(vm, SPEC_BRIGHT_PCT)
    bright = np.clip((v - lo) / max(1.0, hi - lo), 0.0, 1.0)
    keep = np.maximum(1.0 - soft, 1e-3) * matte * bright
    if keep.sum() < 0.05 * skin.sum():
        keep = np.maximum(1.0 - soft, 1e-3) * skin
    base_sigma = max(2.0, short * SPEC_BASE_SIGMA)
    base_v = _weighted_blur(v, keep, base_sigma)
    # 底线另取一份贴脸的：窄窗口的哑光底色。宽窗口为了压得住一大片汗光必须开得比亮斑还大，
    # 于是下巴那颗高光旁边的下颌阴影也进了窗口，底色被拖到 170，而紧挨着它的哑光皮肤是
    # 195——照宽底色压，下巴压出一块比周围暗二十档的灰印。取两者的大者：宽窗口管整片，
    # 窄窗口保证谁也不会被压到自己邻居以下。
    floor_v = np.maximum(base_v, _weighted_blur(v, keep, max(2.0, short * SPEC_FLOOR_SIGMA)))

    # 亮度分两层扣。低频层：亮斑自身的低通与外推哑光底色之差，把整片摊开的汗光压回底色，
    # 两层低通都抹掉了毛孔，所以这一项加在每个通道上是同一个平滑量，纹理原样保留。
    # 尖峰层：顶帽盈余，管的是汗珠那种一颗颗的亮点。低频项对它们完全无效——亮点比低通
    # 核还窄，被当成纹理留在原地，只压低频的话亮斑会变成"底色对了但满脸麻点反光"。
    # 尖峰只削亮的一侧，暗的毛孔和细纹顶帽本来就不响应，不会被抹平。
    smooth = (max(3, int(short * SPEC_CORR_SMOOTH) | 1),) * 2
    dv_low = np.minimum(base_v - cv2.GaussianBlur(v, smooth, 0), 0.0)
    peak = cv2.GaussianBlur(excess, (max(3, int(short * 0.006) | 1),) * 2, 0)
    dv = dv_low - np.minimum(peak * SPEC_PEAK_GAIN, -dv_low)
    # 底线：削峰削到局部哑光底色为止，一个像素都不许压到底色以下。
    # 两层改正各测各的，低频项已经把整片汗光扣掉了，尖峰项又在同一批像素上再扣一遍——
    # 不设这道底线，脸颊那块高光会被压到比旁边的哑光皮肤还暗九档，"油光"变成一块暗斑。
    dv = np.clip(np.maximum(dv, floor_v - v), -SPEC_LIFT_MAX, 0.0)

    # 色度整体换成附近哑光皮肤的色度，只保留像素自己的色度细节。
    # 白光把通道极差冲垮之后，压暗只能还原亮度，那点残留的极差已经是噪声了——照它放大
    # 三倍补饱和度，补出来的是洋红麻点；逐像素对齐"应有饱和度"，额头就变成一片鳞片状
    # 的红斑。改在线性色度（B−Y / R−Y）上做，且只替换低频：色相和浓淡由周围的哑光皮肤
    # 供，红血丝和毛孔那一级的小起伏仍按像素自己的走，两边都不编造颜色。
    # 三个通道的色度增量按 Y=0.114B+0.587G+0.299R 反解 G，保证换色不动亮度。
    small_f = small.astype(np.float32)
    y_w = small_f @ _LUMA_W
    dcb = _chroma_lowpass_delta(small_f[:, :, 0] - y_w, keep, base_sigma, smooth)
    dcr = _chroma_lowpass_delta(small_f[:, :, 2] - y_w, keep, base_sigma, smooth)

    shrink = _to_full(dv * weight, shape[:2])
    cb_fix = _to_full(dcb * weight, shape[:2])
    cr_fix = _to_full(dcr * weight, shape[:2])
    cg_fix = -(cb_fix * _LUMA_W[0] + cr_fix * _LUMA_W[2]) / _LUMA_W[1]

    corrected = np.clip(
        bgr.astype(np.float32) + shrink[:, :, None]
        + np.stack([cb_fix, cg_fix, cr_fix], axis=2),
        0, 255,
    ).astype(np.uint8)

    # 交给网络和邻居判定的"洞"要比重据点一圈：额头那层汗光是渐变的一整片，只有最重的
    # 几处两维判据同时满额，硬掩膜因此只咬住中心，中心以外那一大片带 0.3~0.5 权重的薄光
    # 就完全没人管——重的压平了，轻的原样亮着，一张额头变成几块哑光补丁浮在油光里。
    # 外扩之后权重照旧按连续证据分级，只是这一圈里每一处都轮得到被削。
    shade = strong & (guard > 0.5)
    kr = max(3, int(short * SPEC_REACH) | 1)
    shade = cv2.dilate(shade.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))) > 0
    shade &= skin & (guard > 0.5)
    matte = skin & (ev < 0.35)
    floor, ceil, sat = _hole_bounds(
        _luma(small), small.max(axis=2) - small.min(axis=2), matte, shade, short,
        SPEC_BOUND_PCT, SPEC_BOUND_MARGIN, own_cap=True)
    # 连一圈邻居皮肤都取不到的地方不交给网络。贴着下颌轮廓、发际那一圈的小斑，外面就是
    # 背景，取不到哑光邻居的水平，于是上下限退成 0/255，后面所有按邻居收的闸门全部失效，
    # 网络就在原地补出一块边界发硬的浅斑。取不到邻居等于无从核对，宁可不去。
    shade &= floor > 1.0
    return {
        "weight": _to_full(weight, shape[:2]),
        "hole": _to_full(shade.astype(np.float32), shape[:2]) > 0.5,
        "limit": _to_full(excess_d, shape[:2]),
        "floor": _to_full(floor, shape[:2]),
        "ceil": _to_full(ceil, shape[:2]),
        "sat": _to_full(sat, shape[:2]),
        "ok": _to_full(matte.astype(np.float32), shape[:2]) > 0.5,
        "skin": _ai_skin_gate(skin_full, subject),
        "result": _composite(corrected, alpha, subject, image_rgba),
    }


def apply_specular(detection, image_rgba, ai=None):
    """
    把 detect_specular 的结果落到图上：有修复模型时，网络的哑光皮肤叠在手写底色之上。

    手写增益仍然是底：网络只看到 512 缩略图，它给的是"这一片该是这样的哑光皮肤"，而
    判据实测到的光量才是这次到底该改多少的准绳（fill_with_ai 里的 limit 就是它）。
    """
    if detection is None:
        return image_rgba
    base = detection["result"]
    bgr = base[:, :, :3].astype(np.float32)
    if ai is not None:
        bgr = fill_with_ai(
            base[:, :, :3], ai, detection["weight"], detection["hole"],
            limit=detection["limit"], ref=image_rgba[:, :, :3],
            skin=detection["skin"],
            bounds=(detection["floor"], detection["ceil"]),
        ).astype(np.float32)
    # 不管改正来自手写底色还是网络，都不许越过这块皮肤自己邻居的亮度和浓淡
    bgr = _keep_within_neighbours(
        bgr, detection["floor"], detection["ceil"], detection["sat"],
        detection["hole"].astype(np.float32), detection["ok"],
    )
    subject = _subject_mask_from_alpha(image_rgba)
    return _composite(bgr, base[:, :, 3], subject, image_rgba)


def remove_specular(image_rgba, strength, labels=None):
    """不依赖修复模型的去反光（外部直接调用时的入口）。"""
    return apply_specular(detect_specular(image_rgba, strength, labels), image_rgba)


def retouch_rgba(image_rgba, shadow_strength, specular_strength):
    """
    去阴影 / 去反光的统一入口。

    美颜环节的图像是 RGB 顺序（Gradio 传路径，PIL 读出来就是 RGB），而两个算法的
    判据——肤色色度、HSV 的 V/S 分量——全是按 BGR 语义写的。直接喂 RGB 时肤色掩膜
    一个像素都找不到，处理会悄悄退化成对整个人像（含头发、衣服）动手，等于没做。
    所以通道顺序只在这里换算一次，分割模型也只在这里问一次。

    只有反光会去问修复模型，而且只问一次（两秒多一次的推理，问两次不值）。阴影走手写
    增益，理由见 apply_shadow。模型不可用时去反光也自动退回手写增益。

    :param image_rgba: RGBA 图像，RGB 通道顺序
    """
    if shadow_strength <= 0 and specular_strength <= 0:
        return image_rgba
    bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)
    bgr = bgra[:, :, :3]
    subject = _subject_mask_from_alpha(bgra)
    box = _rough_face_box(bgr, subject)
    labels = face_parsing.parse_face(bgr, box)

    shine = detect_specular(bgra, specular_strength, labels) if specular_strength > 0 else None
    shade = detect_shadow(bgra, shadow_strength, labels) if shadow_strength > 0 else None

    ai = None
    if shine is not None and box is not None and lama_inpaint.available():
        area = max(1.0, float(subject.sum()))
        # 只有反光交给网络：光斑本来就是一处一处的小块，正是修复模型的强项。阴影交给
        # 它反而会画坏五官，见 apply_shadow。
        if shine["hole"].sum() >= AI_HOLE_MIN * area:
            ai = lama_inpaint.inpaint(bgr, shine["hole"].astype(np.float32), box)

    out = bgra
    if shine is not None:
        out = apply_specular(shine, out, ai)
    if shade is not None:
        out = apply_shadow(shade, out)
    return cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
