# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件 - 用于打包 HivisionIDPhotos 启动器
模式: onedir (单目录模式)
功能: 打包 launcher.py 为独立的 EXE 启动器
"""

import sys
from pathlib import Path

# 项目根目录
root_dir = Path(SPECPATH).absolute()

# 分析 launcher.py
a = Analysis(
    ['launcher.py'],                          # 主脚本
    pathex=[str(root_dir)],                   # Python 路径
    binaries=[],                              # 额外的二进制文件
    datas=[],                                 # 额外的数据文件
    hiddenimports=[                           # 隐藏的导入
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.QtCore',
    ],
    hookspath=[],                             # 钩子路径
    hooksconfig={},                           # 钩子配置
    runtime_hooks=[],                         # 运行时钩子
    excludes=[                                # 排除的模块（减小体积）
        'matplotlib',
        'tkinter',
        'unittest',
        'pydoc',
        'email',
        'http',
        'xml',
        'xmlrpc',
        'html',
        'test',
        '_pytest',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# 创建 PYZ 归档
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 创建 EXE - 使用 onedir 模式
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,                    # onedir 模式需要设置为 True
    name='HivisionIDPhotos',                  # EXE 名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                                 # 使用 UPX 压缩
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                            # 不显示控制台窗口（GUI 应用）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # 窗口应用设置
    icon=str(root_dir / 'assets' / 'hivision_logo.png') if (root_dir / 'assets' / 'hivision_logo.png').exists() else None,
)

# 收集所有文件 - onedir 模式的关键
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HivisionIDPhotos'
)
