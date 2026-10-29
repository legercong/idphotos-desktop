# 本文件相对上游项目 HivisionIDPhotos (https://github.com/Zeyi-Lin/HivisionIDPhotos)
# 有修改，改动范围见 NOTICE 与 CHANGELOG.md。

# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from demo.utils import csv_to_size_list
from demo.config import load_configuration
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
size_list_dict_CN = csv_to_size_list(os.path.join(base_dir, "assets/size_list_CN.csv"))
size_list_dict_EN = csv_to_size_list(os.path.join(base_dir, "assets/size_list_EN.csv"))
(
    size_list_config_CN,
    size_list_config_EN,
    color_list_dict_CN,
    color_list_dict_EN,
) = load_configuration(base_dir)


LOCALES = {
    "face_model": {
        "en": {
            "label": "Face detection model",
        },
        "zh": {
            "label": "人脸检测模型",
        },
        "ja": {
            "label": "顔検出モデル",
        },
        "ko": {
            "label": "얼굴 감지 모델",
        },
    },
    "matting_model": {
        "en": {
            "label": "Matting model",
        },
        "zh": {
            "label": "抠图模型",
        },
        "ja": {
            "label": "マッティングモデル",
        },
        "ko": {
            "label": "매팅 모델",
        },
    },
    "key_param": {
        "en": {
            "label": "Key Parameters",
        },
        "zh": {
            "label": "核心参数",
        },
        "ja": {
            "label": "主要パラメータ",
        },
        "ko": {
            "label": "주요 매개변수",
        },
    },
    "advance_param": {
        "en": {
            "label": "Advance Parameters",
        },
        "zh": {
            "label": "高级参数",
        },
        "ja": {
            "label": "詳細パラメータ",
        },
        "ko": {
            "label": "고급 매개변수",
        },
    },
    "size_mode": {
        "en": {
            "label": "ID photo size options",
            "choices": [
                "Size List",
                "Only Change BG",
                "Custom(px)",
                "Custom(mm)",
            ],
            "custom_size_eror": "The width should not be greater than the length; the length and width should not be less than 100, and no more than 1800.",
        },
        "zh": {
            "label": "证件照尺寸选项",
            "choices": ["尺寸列表", "只换底", "自定义(px)", "自定义(mm)"],
            "custom_size_eror": "宽度不应大于长度；长度和宽度不应小于100，不大于1800。",
        },
        "ja": {
            "label": "証明写真サイズオプション",
            "choices": [
                "サイズリスト",
                "背景のみ変更",
                "カスタムサイズ(px)",
                "カスタムサイズ(mm)",
            ],
            "custom_size_eror": "幅は長さより大きくしないでください。長さと幅は100以上1800以下にしてください。",
        },
        "ko": {
            "label": "증명사진 크기 옵션",
            "choices": [
                "크기 목록",
                "배경만 변경",
                "사용자 지정(px)",
                "사용자 지정(mm)",
            ],
            "custom_size_eror": "너비는 길이보다 크지 않아야 합니다; 길이와 너비는 100 이상 1800 이하여야 합니다.",
        },
    },
    "custom_size_px": {
        "en": {
            "height": "Height(px)",
            "width": "Width(px)",
        },
        "zh": {
            "height": "高度(px)",
            "width": "宽度(px)",
        },
        "ja": {
            "height": "高さ(px)",
            "width": "幅(px)",
        },
        "ko": {
            "height": "높이(px)",
            "width": "너비(px)",
        },
    },
    "custom_size_mm": {
        "en": {
            "height": "Height(mm)",
            "width": "Width(mm)",
        },
        "zh": {
            "height": "高度(mm)",
            "width": "宽度(mm)",
        },
        "ja": {
            "height": "高さ(mm)",
            "width": "幅(mm)",
        },
        "ko": {
            "height": "높이(mm)",
            "width": "너비(mm)",
        },
    },
    "size_list": {
        "en": {
            "label": "Size list",
            "choices": list(size_list_dict_EN.keys()),
            "develop": size_list_config_EN,
        },
        "zh": {
            "label": "预设尺寸",
            "choices": list(size_list_dict_CN.keys()),
            "develop": size_list_config_CN,
        },
        "ja": {
            "label": "サイズリスト",
            "choices": list(size_list_dict_EN.keys()),
            "develop": size_list_config_EN,
        },
        "ko": {
            "label": "크기 목록",
            "choices": list(size_list_dict_EN.keys()),
            "develop": size_list_config_EN,
        },
    },
    "bg_color": {
        "en": {
            "label": "Background color",
            "choices": list(color_list_dict_EN.keys())
            + ["American Style"]
            + ["Custom(RGB)", "Custom(HEX)"],
            "develop": color_list_dict_EN,
        },
        "zh": {
            "label": "背景颜色",
            "choices": list(color_list_dict_CN.keys())
            + ["美式证件照"]
            + ["自定义(RGB)", "自定义(HEX)"],
            "develop": color_list_dict_CN,
        },
        "ja": {
            "label": "背景色",
            "choices": list(color_list_dict_EN.keys())
            + ["American Style"]
            + ["カスタム(RGB)", "カスタム(HEX)"],
            "develop": color_list_dict_EN,
        },
        "ko": {
            "label": "배경색",
            "choices": list(color_list_dict_EN.keys())
            + ["American Style"]
            + ["사용자 지정(RGB)", "사용자 지정(HEX)"],
            "develop": color_list_dict_EN,
        },
    },
    "button": {
        "en": {
            "label": "Start",
        },
        "zh": {
            "label": "开始制作",
        },
        "ja": {
            "label": "開始",
        },
        "ko": {
            "label": "시작",
        },
    },
    "head_measure_ratio": {
        "en": {
            "label": "Head ratio",
        },
        "zh": {
            "label": "面部比例",
        },
        "ja": {
            "label": "頭部比率",
        },
        "ko": {
            "label": "머리 비율",
        },
    },
    "top_distance": {
        "en": {
            "label": "Top distance",
        },
        "zh": {
            "label": "头距顶距离",
        },
        "ja": {
            "label": "上部からの距離",
        },
        "ko": {
            "label": "상단 거리",
        },
    },
    "image_kb": {
        "en": {
            "label": "Set KB size",
            "choices": ["Not Set", "Custom"],
        },
        "zh": {
            "label": "设置 KB 大小",
            "choices": ["不设置", "自定义"],
        },
        "ja": {
            "label": "KBサイズを設定",
            "choices": ["設定なし", "カスタム"],
        },
        "ko": {
            "label": "KB 크기 설정",
            "choices": ["설정 안 함", "사용자 지정"],
        },
    },
    "image_kb_size": {
        "en": {
            "label": "KB size",
        },
        "zh": {
            "label": "KB 大小",
        },
        "ja": {
            "label": "KBサイズ",
        },
        "ko": {
            "label": "KB 크기",
        },
    },
    "image_dpi": {
        "en": {
            "label": "Set DPI",
            "choices": ["Not Set", "Custom"],
        },
        "zh": {
            "label": "设置 DPI 大小",
            "choices": ["不设置", "自定义"],
        },
        "ja": {
            "label": "DPIを設定",
            "choices": ["設定なし", "カスタム"],
        },
        "ko": {
            "label": "DPI 설정",
            "choices": ["설정 안 함", "사용자 지정"],
        },
    },
    "image_dpi_size": {
        "en": {
            "label": "DPI size",
        },
        "zh": {
            "label": "DPI 大小",
        },
        "ja": {
            "label": "DPIサイズ",
        },
        "ko": {
            "label": "DPI 크기",
        },
    },
    "render_mode": {
        "en": {
            "label": "Render mode",
            "choices": [
                "Solid Color",
                "Up-Down Gradient (White)",
                "Center Gradient (White)",
            ],
        },
        "zh": {
            "label": "渲染方式",
            "choices": ["纯色", "上下渐变（白色）", "中心渐变（白色）"],
        },
        "ja": {
            "label": "レンダリングモード",
            "choices": [
                "単色",
                "上下グラデーション（白）",
                "中心グラデーション（白）",
            ],
        },
        "ko": {
            "label": "렌더링 모드",
            "choices": [
                "단색",
                "위-아래 그라데이션 (흰색)",
                "중앙 그라데이션 (흰색)",
            ],
        },
    },
    "crop_mode": {
        "en": {
            "label": "Crop Mode (No face / Multiple faces)",
            "choices": ["Fill (Cover)", "Fit (Contain)"],
            "fit_description": "Keep aspect ratio, fit entire image into target frame without cropping. Short edges will have blank areas.",
            "fill_description": "Keep aspect ratio, fill entire frame. Excess parts will be cropped. No blank areas.",
        },
        "zh": {
            "label": "裁切模式（无人脸/多人脸时生效）",
            "choices": ["按比例填充", "按比例拟合"],
            "fit_description": "保持原图宽高比不变，将整个图片完整容纳到目标图形框内，不会裁剪图片，也不会拉伸变形，最终效果是图片的长边贴合图形框的对应边，短边会留有空白区域。",
            "fill_description": "同样保持原图宽高比不变，将图片铺满整个目标图形框，会自动裁剪超出图形框的部分，保证图形框内完全被图片填满，不会出现空白。",
        },
        "ja": {
            "label": "クロップモード（顔なし/複数顔時に有効）",
            "choices": ["フィル（Cover）", "フィット（Contain）"],
            "fit_description": "元のアスペクト比を維持し、画像全体をターゲットフレームに収めます。クロップなし、短辺に余白ができます。",
            "fill_description": "元のアスペクト比を維持し、画像でフレーム全体を埋めます。はみ出す部分はクロップされ、余白はできません。",
        },
        "ko": {
            "label": "자르기 모드 (얼굴 없음/다중 얼굴 시 적용)",
            "choices": ["채우기 (Cover)", "맞춤 (Contain)"],
            "fit_description": "원본 가로세로 비율을 유지하여 이미지 전체를 대상 프레임에 맞춥니다. 잘리지 않으며 짧은 쪽에 여백이 생깁니다.",
            "fill_description": "원본 가로세로 비율을 유지하여 이미지로 프레임 전체를 채웁니다. 넘치는 부분은 잘리며 여백이 없습니다.",
        },
    },
    # Tab3 - 水印工作台
    "watermark_tab": {
        "en": {
            "label": "Watermark",
        },
        "zh": {
            "label": "水印",
        },
        "ja": {
            "label": "ウォーターマーク",
        },
        "ko": {
            "label": "워터마크",
        },
    },
    "watermark_text": {
        "en": {
            "label": "Text",
            "value": "Hello",
            "placeholder": "up to 20 characters",
        },
        "zh": {
            "label": "水印文字",
            "value": "Hello",
            "placeholder": "最多20个字符",
        },
        "ja": {
            "label": "テキスト",
            "value": "Hello",
            "placeholder": "最大20文字",
        },
        "ko": {
            "label": "텍스트",
            "value": "Hello",
            "placeholder": "최대 20자",
        },
    },
    "watermark_color": {
        "en": {
            "label": "Color",
        },
        "zh": {
            "label": "水印颜色",
        },
        "ja": {
            "label": "色",
        },
        "ko": {
            "label": "색상",
        },
    },
    "watermark_size": {
        "en": {
            "label": "Size",
        },
        "zh": {
            "label": "文字大小",
        },
        "ja": {
            "label": "サイズ",
        },
        "ko": {
            "label": "크기",
        },
    },
    "watermark_opacity": {
        "en": {
            "label": "Opacity",
        },
        "zh": {
            "label": "水印透明度",
        },
        "ja": {
            "label": "不透明度",
        },
        "ko": {
            "label": "불투명도",
        },
    },
    "watermark_angle": {
        "en": {
            "label": "Angle",
        },
        "zh": {
            "label": "水印角度",
        },
        "ja": {
            "label": "角度",
        },
        "ko": {
            "label": "각도",
        },
    },
    "watermark_space": {
        "en": {
            "label": "Space",
        },
        "zh": {
            "label": "水印间距",
        },
        "ja": {
            "label": "間隔",
        },
        "ko": {
            "label": "간격",
        },
    },
    "watermark_switch": {
        "en": {
            "label": "Watermark",
            "value": "Not Add",
            "choices": ["Not Add", "Add"],
        },
        "zh": {
            "label": "水印",
            "value": "不添加",
            "choices": ["不添加", "添加"],
        },
        "ja": {
            "label": "ウォーターマーク",
            "value": "追加しない",
            "choices": ["追加しない", "追加"],
        },
        "ko": {
            "label": "워터마크",
            "value": "추가하지 않음",
            "choices": ["추가하지 않음", "추가"],
        },
    },
    # 输出结果
    "notification": {
        "en": {
            "label": "notification",
            "face_error": "The number of faces is not equal to 1, please upload an image with a single face. If the actual number of faces is 1, it may be an issue with the accuracy of the detection model. Please switch to a different face detection model on the left or raise a Github Issue to notify the author.",
            "face_fallback": "Detected {} face(s), automatically switched to frame crop mode ({}). No matting or background replacement applied.",
            "process_fallback": "The image could not be processed as an ID photo ({}), automatically switched to frame crop mode ({}), cropping the original image without matting or background replacement.",
            "image_error": "Failed to read the uploaded image, please try another one or re-upload it.",
        },
        "zh": {
            "label": "通知",
            "face_error": "人脸数不等于1，请上传单人照片。如果实际人脸数为1，可能是检测模型的准确度问题，请切换左侧不同的人脸检测模型或提出Github Issue通知作者。",
            "face_fallback": "检测到{}个人脸，已自动切换为画幅裁切模式（{}），使用原图裁切，不进行抠图和换底。",
            "process_fallback": "图像无法正常处理（{}），已自动切换为画幅裁切模式（{}），使用原图裁切，不进行抠图和换底。",
            "image_error": "图片读取失败，请重新上传或换一张图片。",
        },
        "ja": {
            "label": "通知",
            "face_error": "顔の数が1ではありません。1つの顔を含む画像をアップロードしてください。実際の顔の数が1の場合、検出モデルの精度の問題かもしれません。左側で別の顔検出モデルに切り替えるか、Githubの問題を作成して作者に通知してください。",
            "face_fallback": "{}つの顔が検出され、自動的にフレームクロップモード（{}）に切り替わりました。マッティングや背景変更は行われません。",
            "process_fallback": "画像を証明写真として処理できませんでした（{}）。自動的にフレームクロップモード（{}）に切り替わり、元画像をトリミングします。",
            "image_error": "画像の読み込みに失敗しました。再アップロードか別の変更してください。",
        },
        "ko": {
            "label": "알림",
            "face_error": "얼굴 수가 1이 아닙니다. 단일 얼굴이 있는 이미지를 업로드해 주세요. 실제 얼굴 수가 1인 경우 감지 모델의 정확도 문제일 수 있습니다. 왼쪽에서 다른 얼굴 감지 모델로 전환하거나 Github Issue를 제기하여 작성자에게 알려주세요.",
            "face_fallback": "{}개의 얼굴이 감지되어 자동으로 프레임 자르기 모드({})로 전환되었습니다. 매팅 및 배경 변경은 적용되지 않습니다.",
            "process_fallback": "이미지를 신분증 사진으로 처리할 수 없습니다({}). 자동으로 프레임 자르기 모드({})로 전환되며 원본 이미지를 자릅니다.",
            "image_error": "이미지를 읽지 못했습니다. 다시 업로드하거나 다른 이미지를 사용하세요.",
        },
    },
    "standard_photo": {
        "en": {
            "label": "Standard photo",
        },
        "zh": {
            "label": "标准照",
        },
        "ja": {
            "label": "標準写真",
        },
        "ko": {
            "label": "표준 사진",
        },
    },
    "hd_photo": {
        "en": {
            "label": "HD photo",
        },
        "zh": {
            "label": "高清照",
        },
        "ja": {
            "label": "HD写真",
        },
        "ko": {
            "label": "HD 사진",
        },
    },
    "standard_photo_png": {
        "en": {
            "label": "Matting Standard photo",
        },
        "zh": {
            "label": "透明标准照",
        },
        "ja": {
            "label": "マッティング標準写真",
        },
        "ko": {
            "label": "매팅 표준 사진",
        },
    },
    "hd_photo_png": {
        "en": {
            "label": "Matting HD photo",
        },
        "zh": {
            "label": "透明高清照",
        },
        "ja": {
            "label": "マッティングHD写真",
        },
        "ko": {
            "label": "매팅 HD 사진",
        },
    },
    "layout_photo": {
        "en": {
            "label": "Layout photo",
        },
        "zh": {
            "label": "排版照",
        },
        "ja": {
            "label": "レイアウト写真",
        },
        "ko": {
            "label": "레이아웃 사진",
        },
    },
    "download": {
        "en": {
            "label": "Download the photo after adjusting the DPI or KB size",
        },
        "zh": {
            "label": "下载调整 DPI 或 KB 大小后的照片",
        },
        "ja": {
            "label": "DPIまたはKBサイズ調整後の写真をダウンロード",
        },
        "ko": {
            "label": "DPI 또는 KB 크기 조정 후 사진 다운로드",
        },
    },
    "matting_image": {
        "en": {
            "label": "Matting image",
        },
        "zh": {
            "label": "抠图图像",
        },
        "ja": {
            "label": "マット画像",
        },
        "ko": {
            "label": "매팅 이미지",
        },
    },
    "beauty_tab": {
        "en": {
            "label": "Beauty",
        },
        "zh": {
            "label": "美颜",
        },
        "ja": {
            "label": "美顔",
        },
        "ko": {
            "label": "뷰티",
        },
    },
    "whitening_strength": {
        "en": {
            "label": "whitening strength",
        },
        "zh": {
            "label": "美白强度",
        },
        "ja": {
            "label": "美白強度",
        },
        "ko": {
            "label": "미백 강도",
        },
    },
    "shadow_strength": {
        "en": {
            "label": "shadow removal strength",
        },
        "zh": {
            "label": "去阴影强度",
        },
        "ja": {
            "label": "影除去の強さ",
        },
        "ko": {
            "label": "그림자 제거 강도",
        },
    },
    "specular_strength": {
        "en": {
            "label": "shine removal strength",
        },
        "zh": {
            "label": "去反光强度",
        },
        "ja": {
            "label": "テカリ除去の強さ",
        },
        "ko": {
            "label": "유광 제거 강도",
        },
    },
    "brightness_strength": {
        "en": {
            "label": "brightness strength",
        },
        "zh": {
            "label": "亮度强度",
        },
        "ja": {
            "label": "明るさの強さ",
        },
        "ko": {
            "label": "밝기 강도",
        },
    },
    "contrast_strength": {
        "en": {
            "label": "contrast strength",
        },
        "zh": {
            "label": "对比度强度",
        },
        "ja": {
            "label": "コントラスト強度",
        },
        "ko": {
            "label": "대비 강도",
        },
    },
    "sharpen_strength": {
        "en": {
            "label": "sharpen strength",
        },
        "zh": {
            "label": "锐化强度",
        },
        "ja": {
            "label": "シャープ化強度",
        },
        "ko": {
            "label": "샤ープ 강도",
        },
    },
    "saturation_strength": {
        "en": {
            "label": "saturation strength",
        },
        "zh": {
            "label": "饱和度强度",
        },
        "ja": {
            "label": "飽和度強度",
        },
        "ko": {
            "label": "포화도 강도",
        },
    },
    "plugin": {
        "en": {
            "label": "🤖Plugin",
            "choices": [
                "Face Alignment",
                "Horizontal Flip",
                "Layout Photo Crop Line",
                "JPEG Format",
                "Five Inch Paper",
            ],
            "value": ["Layout Photo Crop Line"],
        },
        "zh": {
            "label": "🤖插件",
            "choices": ["人脸旋转对齐", "水平翻转", "排版照裁剪线", "JPEG格式"],
            "value": ["排版照裁剪线"],
        },
        "ja": {
            "label": "🤖プラグイン",
            "choices": [
                "顔の整列",
                "水平反転",
                "レイアウト写真の切り取り線",
                "JPEGフォーマット",
            ],
            "value": ["レイアウト写真の切り取り線"],
        },
        "ko": {
            "label": "🤖플러그인",
            "choices": [
                "얼굴 정렬",
                "수평 반전",
                "레이아웃 사진 자르기 선",
                "JPEG 포맷",
                "오렌지 사진",
            ],
            "value": ["레이아웃 사진 자르기 선"],
        },
    },
    "template_photo": {
        "en": {
            "label": "Social Media Template Photo",
        },
        "zh": {
            "label": "社交媒体模版照",
        },
        "ja": {
            "label": "SNS テンプレート写真",
        },
        "ko": {
            "label": "SNS 템플릿 사진",
        },
    },
    "print_tab": {
        "en": {
            "label": "Print Layout",
        },
        "zh": {
            "label": "打印排版",
        },
        "ja": {
            "label": "印刷レイアウト",
        },
        "ko": {
            "label": "인쇄 레이아웃",
        },
    },
    "print_switch": {
        "shape": [[1205, 1795], [1051, 1500], [2479, 3508], [1051, 1500], [1205, 1795]],
        "en": {
            "label": "Paper size",
            "choices": ["6 inch", "5 inch", "A4", "3R", "4R"],
        },
        "zh": {
            "label": "相纸选择",
            "choices": ["六寸", "五寸", "A4", "3R", "4R"],
        },
        "ja": {
            "label": "用紙サイズ",
            "choices": ["6インチ", "5インチ", "A4", "3R", "4R"],
        },
        "ko": {
            "label": "용지 사이즈",
            "choices": ["6인치", "5인치", "A4", "3R", "4R"],
        },
    },
}
