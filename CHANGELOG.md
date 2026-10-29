# CHANGELOG — 本仓库相对上游 HivisionIDPhotos 的改动

上游基线：<https://github.com/Zeyi-Lin/HivisionIDPhotos>（Apache-2.0，见 `NOTICE`）

本文件按 Apache-2.0 第 4(b) 条要求，声明本仓库对上游代码做了修改，
并列出修改的范围。带 ✔ 的文件是**本仓库新增或 substantially 重写**的，
其余为在上游文件上的局部改动。

## 新增功能

### 1. AI 去阴影 / 去反光（美颜管线）
证件照常出现单侧窗光造成的"阴阳脸"，以及出汗后的面部高光。证件照审核
通常不允许面部阴影与反光，因此加入了两个独立强度（0–15）的修图环节。

| 文件 | 说明 |
| --- | --- |
| `hivision/plugin/beauty/retouch.py` ✔ | 核心算法：区域传输函数估计照度增益、保边照度面（bilateral）、肤色/颈部色度门控、高光连通域检测与"只允许减光"的邻域上下界 |
| `hivision/plugin/beauty/face_parsing.py` ✔ | SegFormer-B5（CelebAMask-HQ int8 ONNX）人脸解析，512×512，只把皮肤/颈部区域纳入处理 |
| `hivision/plugin/beauty/lama_inpaint.py` ✔ | LaMa 修复网络 ONNX 封装，含 `prewarm()` 会话预热与高频细节回补 |
| `hivision/plugin/beauty/handler.py` | 在美白之后串入 `retouch_rgba()`，透传两个强度参数 |
| `hivision/creator/__init__.py`、`hivision/creator/context.py` | 参数从调用层透传到 beauty 管线 |
| `demo/ui.py`、`demo/processor.py`、`demo/locales.py` | 美颜 Tab 增加两个滑杆；中/英/日/韩四语文案 |

依赖的模型权重（`*.onnx`）**不随仓库分发**，见下文"权重获取"。

### 2. 桌面应用形态
| 文件 | 说明 |
| --- | --- |
| `launcher.py` ✔ | 桌面启动器：本机起 Gradio 服务 + PyQt5 无边框窗口与加载动画，含窗口存在性检测、端口 7860–7870 自动避让 |
| `desktop_app_fast.py` ✔ | 快速桌面壳（服务就绪探测与窗口关闭回收） |
| `launcher.spec`、`HivisionIDPhotos_Launcher.spec`、`HivisionIDPhotos_Launcher_Console.spec` ✔ | PyInstaller 打包配置 |

### 3. 放宽人脸检测的硬性限制
普通照片（无人脸或多人脸）也需要能被裁切成任意尺寸，因此不再抛错：
检测到 0 张或多张人脸时降级为**画幅裁切**模式并给出提示，始终产出图片。

- `hivision/creator/context.py`：新增 `face_num` 字段标记降级模式
- `hivision/creator/__init__.py`：检测失败时返回抠图结果而非抛出
- `demo/processor.py`：`画幅裁切降级处理`
- `demo/ui.py`、`demo/locales.py`：新增"裁切模式（无人脸/多人脸时生效）"选项

### 4. 尺寸预设与裁切默认值修正
- `demo/assets/size_list_CN.csv`：四寸修正为 1205×900 px（二寸维持 35×49 mm 口径）
- `demo/assets/size_list_EN.csv`、`color_list_*.csv`：与中文版同步
- 裁切模式默认排序改为"按比例填充（Cover）"在前、"按比例拟合"在后，四语文案
  与 CLI/API 默认值保持一致
- 预览框按图片真实宽高比渲染，与"预览"弹窗一致

### 5. 排版打印与交互修复
- `demo/print_worker.py`：打印排版与出片流程
- 修复二次拖入图片后预览空白、加载动画卡顿等问题

## 已知限制

- 极强侧光下的窗框硬阴影，强度 8–12 仍可能残留极淡痕迹，需要 14–15 才基本干净。
- 去反光强度 ≥12 时额头会略显平滑。
- 去阴影单张约 3.7–5.8 s，去反光约 7.6 s，两者同时开启约 9.8 s（16 核 CPU，
  ONNX Runtime CPU 执行；未使用 GPU）。
