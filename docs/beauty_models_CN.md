# 美颜修图模型权重（本仓库新增功能所需）

本仓库的 **AI 去阴影 / AI 去反光** 需要两个额外的 ONNX 权重。它们**不在 Git 仓库内**
（`.gitignore` 排除了 `*.onnx`），需自行放置到指定路径。

| 文件（相对项目根目录） | 作用 | 大小 | 推理耗时（16 核 CPU） |
| --- | --- | --- | --- |
| `hivision/plugin/beauty/weights/face_parsing.onnx` | 人脸解析：限定去阴影/去反光只作用于皮肤与颈部，避免动到五官与背景 | 约 85 MB | 1.3–3.4 s |
| `hivision/plugin/beauty/weights/lama_fp32.onnx` | LaMa 图像修复：为高光斑提供低频光照基准 | 约 198 MB | 约 2.3 s |

缺权重不会报错、也不会中断出片，但效果会降级，日志里能看到对应提示：

- 缺 `face_parsing.onnx`：打印 `[face_parsing] 未找到权重 …，修图退回颜色判据`，
  改为用肤色判据划定范围，精度下降、更容易碰到背景。
- 缺 `lama_fp32.onnx`：去反光自动退回手写增益算法（打印 `[retouch] LaMa 模型加载失败 …`
  仅在权重损坏时出现；文件不存在时静默跳过加载）。

## 来源说明

出于许可与体积考虑，本仓库不再分发这两个权重，请从原始开源项目自行导出或检索其 ONNX 版本：

- 人脸解析：**SegFormer-B5**（[NVlabs/segformer](https://github.com/NVlabs/segformer)），
  在 **CelebAMask-HQ**（[switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)）
  上微调的人脸部件解析模型，导出为 int8 ONNX、输入 512×512。
  > ⚠️ CelebAMask-HQ 的数据许可限定"非商业性学术研究"。**将该权重用于商业出片前，
  > 请自行确认授权范围**，或改用许可允许的替代解析模型。
- 图像修复：**LaMa**（[advimman/lama](https://github.com/advimman/lama)）的 ONNX 导出（fp32），
  固定输入 512×512，接受 `image` + `mask` 两个输入（mask 中 1 表示待修复区域），输出 0–255。

两个权重均为 CPU 推理（`onnxruntime`）。本机 GTX 1650 无 CUDA/cuDNN 环境，未启用 GPU 加速。

## 跨机器迁移

从别的机器拷贝本仓库运行桌面版时，除上述两个文件外，还需要上游的抠图与人脸检测权重
（见 [README 权重下载](../README.md#3-下载人像抠图模型权重文件)）。
`hivision/creator/weights/`、`hivision/creator/retinaface/weights/`、
`hivision/plugin/beauty/weights/` 三个目录都要放齐，缺一个就会静默降级。
