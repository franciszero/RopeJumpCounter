#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RopeJumpCounter main entry point

This is the primary entry point for the project, providing a unified
command-line interface to access various functionalities including
real-time counting, training, annotation, and visualization.

DEPRECATED: main.py, app.py, and main_0.5.py are deprecated.
Use 'python run.py <mode>' instead.

NEW: Support for v2.0 architecture with dependency injection and event bus.
"""

import sys
import argparse
import warnings
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def show_deprecation_warning():
    """Show deprecation warning for old entry points"""
    warnings.warn(
        "DEPRECATED: Using old entry points (main.py, app.py, main_0.5.py) is deprecated. "
        "Please use 'python run.py <mode>' instead. "
        "See 'python run.py --help' for available modes.",
        DeprecationWarning,
        stacklevel=2
    )


def main():
    parser = argparse.ArgumentParser(
        description="RopeJumpCounter - Jump rope counting application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Available application modes:
          realtime    - Real-time jump counting (default)
          realtime-v2 - Real-time counting with new architecture
          legacy      - Legacy real-time counting
          train       - Model training
          label       - Data annotation
          visualize   - Model visualization
          build       - Build dataset

        Examples:
          python run.py                     # Run real-time jump counting
          python run.py realtime            # Same as above
          python run.py realtime-v2         # Run with new architecture
          python run.py legacy              # Run legacy version
          python run.py train               # Train models
          python run.py label --workdir data/raw_videos  # Data annotation
          python run.py visualize --model best_model.keras --video test.mp4  # Visualization
          python run.py build --videos_dir data/videos --labels_dir data/labels  # Build dataset
        """
    )

    parser.add_argument(
        'mode',
        nargs='?',
        default='build',
        choices=['realtime', 'realtime-v2', 'legacy', 'train', 'label', 'visualize', 'build'],
        help='Application mode (default: realtime)'
    )

    parser.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the sub-application'
    )

    args = parser.parse_args()

    # Run the appropriate application based on mode
    if args.mode == 'realtime':
        from src.apps.main import main as app_main
        # Reset sys.argv to pass arguments to sub-application
        sys.argv = ['main.py'] + (args.args or [])
        app_main()

    elif args.mode == 'realtime-v2':
        from src.apps.main_v2 import main as app_main_v2
        # Reset sys.argv to pass arguments to sub-application
        sys.argv = ['main_v2.py'] + (args.args or [])
        app_main_v2()

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
        # Parse annotation-related arguments
        import argparse as label_argparse
        label_parser = label_argparse.ArgumentParser()
        label_parser.add_argument('--workdir', default='data/raw_videos_3',
                                  help='Directory containing videos and labels')
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
