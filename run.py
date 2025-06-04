#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RopeJumpCounter 主入口文件

这是项目的主要入口点，提供统一的命令行界面来访问各种功能。
"""

import sys
import argparse
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="RopeJumpCounter - 跳绳计数器应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        可用的应用模式:
          realtime    - 实时跳绳计数 (默认)
          legacy      - 旧版本实时计数
          train       - 模型训练
          label       - 数据标注
          visualize   - 模型可视化
          build       - 构建数据集

        示例:
          python run.py                     # 运行实时跳绳计数
          python run.py realtime            # 同上
          python run.py legacy              # 运行旧版本
          python run.py train               # 训练模型
          python run.py label --workdir data/raw_videos  # 数据标注
          python run.py visualize --model best_model.keras --video test.mp4  # 可视化
          python run.py build --videos_dir data/videos --labels_dir data/labels  # 构建数据集
        """
    )

    parser.add_argument(
        'mode',
        nargs='?',
        default='realtime',
        choices=['realtime', 'legacy', 'train', 'label', 'visualize', 'build'],
        help='应用模式 (默认: realtime)'
    )

    parser.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='传递给子应用的参数'
    )

    args = parser.parse_args()

    # 根据模式运行相应的应用
    if args.mode == 'realtime':
        from src.apps.main import main as app_main
        # 重新设置 sys.argv 以传递参数给子应用
        sys.argv = ['main.py'] + (args.args or [])
        app_main()

    elif args.mode == 'legacy':
        from src.apps.main_0_5 import main as legacy_main
        sys.argv = ['main_0.5.py'] + (args.args or [])
        legacy_main()

    elif args.mode == 'train':
        from src.ml.training.model_training import main as train_main
        sys.argv = ['model_training.py'] + (args.args or [])
        train_main()

    elif args.mode == 'label':
        from src.ml.data.labeling.main_gui import main as label_main
        # 解析标注相关参数
        import argparse as label_argparse
        label_parser = label_argparse.ArgumentParser()
        label_parser.add_argument('--workdir', default='data/raw_videos_3', help='视频和标签所在目录')
        label_args = label_parser.parse_args(args.args or [])
        label_main(label_args.workdir)

    elif args.mode == 'visualize':
        from src.ml.visualization.model_visualize import main as viz_main
        sys.argv = ['model_visualize.py'] + (args.args or [])
        viz_main()

    elif args.mode == 'build':
        from src.ml.data.builders.builder import main as build_main
        sys.argv = ['builder.py'] + (args.args or [])
        build_main()


if __name__ == "__main__":
    main()
