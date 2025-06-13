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


def check_labeling_prerequisites():
    """
    Check if prerequisites for labeling are met.
    """
    from pathlib import Path
    
    # Check if video files exist
    video_dirs = [
        Path("data/raw_videos_3"),
        Path("data/raw_videos"),
        Path("data/videos"),
    ]
    
    video_files_found = False
    for video_dir in video_dirs:
        if video_dir.exists():
            video_files = list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4"))
            if video_files:
                video_files_found = True
                break
    
    if not video_files_found:
        print("\n" + "="*80)
        print("❌ No video files found for labeling!")
        print("="*80)
        print("Please ensure you have video files in one of these directories:")
        for video_dir in video_dirs:
            print(f"  - {video_dir}/")
        print("\n💡 Solution:")
        print("1. Place your video files (.avi or .mp4) in data/raw_videos_3/")
        print("2. Then run: python run.py label")
        print("="*80)
        return False
    
    return True


def check_build_prerequisites():
    """
    Check if prerequisites for dataset building are met.
    """
    from pathlib import Path
    
    # Check if labeled videos exist
    label_dirs = [
        Path("data/raw_videos_3"),
        Path("data/raw_videos"),
    ]
    
    labeled_videos_found = False
    for label_dir in label_dirs:
        if label_dir.exists():
            label_files = list(label_dir.glob("*_labels.csv"))
            if label_files:
                labeled_videos_found = True
                break
    
    if not labeled_videos_found:
        print("\n" + "="*80)
        print("❌ No labeled videos found for dataset building!")
        print("="*80)
        print("Please ensure you have labeled video files (*_labels.csv) in one of these directories:")
        for label_dir in label_dirs:
            print(f"  - {label_dir}/")
        print("\n💡 Solution:")
        print("1. First label your videos: python run.py label")
        print("2. Then build the dataset: python run.py build")
        print("="*80)
        return False
    
    return True


def check_visualization_prerequisites():
    """
    Check if prerequisites for visualization are met.
    """
    from pathlib import Path
    
    # Check if trained models exist
    model_dirs = [
        Path("model_files"),
    ]
    
    models_found = False
    for model_dir in model_dirs:
        if model_dir.exists():
            model_files = list(model_dir.rglob("*.keras"))
            if model_files:
                models_found = True
                break
    
    if not models_found:
        print("\n" + "="*80)
        print("❌ No trained models found for visualization!")
        print("="*80)
        print("Please ensure you have trained models (*.keras) in the model_files directory.")
        print("\n💡 Solution:")
        print("1. First build datasets: python run.py build")
        print("2. Then train models: python run.py train")
        print("3. Then visualize: python run.py visualize")
        print("="*80)
        return False
    
    return True


def check_realtime_prerequisites():
    """
    Check if prerequisites for real-time counting are met.
    """
    from pathlib import Path
    
    # Check if trained models exist
    model_dirs = [
        Path("model_files"),
    ]
    
    models_found = False
    for model_dir in model_dirs:
        if model_dir.exists():
            model_files = list(model_dir.rglob("*.keras"))
            if model_files:
                models_found = True
                break
    
    if not models_found:
        print("\n" + "="*80)
        print("❌ No trained models found for real-time counting!")
        print("="*80)
        print("Please ensure you have trained models (*.keras) in the model_files directory.")
        print("\n💡 Solution:")
        print("1. First build datasets: python run.py build")
        print("2. Then train models: python run.py train")
        print("3. Then run real-time counting: python run.py realtime")
        print("="*80)
        return False
    
    return True


def check_training_data(args):
    """
    Check if training data exists, if not provide friendly prompt and optionally run data builder.
    """
    import os
    from pathlib import Path
    
    # Check for common training data directories
    data_dirs = [
        Path("data/dataset"),
        Path("data/dataset_16_10100"),  # Common pattern from builder
        Path("data/size4"),  # Direct size directory
        Path("data/size8"),  # Another common size
    ]
    
    training_data_found = False
    for data_dir in data_dirs:
        if data_dir.exists():
            # Check if it has the expected structure
            for size_dir in data_dir.glob("size*"):
                if size_dir.is_dir():
                    train_dir = size_dir / "train"
                    if train_dir.exists() and any(train_dir.glob("*.npz")):
                        training_data_found = True
                        break
            if training_data_found:
                break
    
    if not training_data_found:
        print("\n" + "="*80)
        print("❌ Training data not found!")
        print("="*80)
        print("You need to build the training dataset before starting training.")
        print("\n💡 Recommended workflow:")
        print("1. Label video data:")
        print("   python run.py label")
        print("2. Build training dataset:")
        print("   python run.py build")
        print("3. Then start training:")
        print("   python run.py train")
        print("\n📁 Expected data directory structure:")
        print("   data/dataset_*/size{window_size}/")
        print("   ├── train/*.npz")
        print("   ├── val/*.npz")
        print("   └── test/*.npz")
        print("="*80)
        
        # Check if labeled data exists before offering to run builder
        label_dirs = [
            Path("data/raw_videos_3"),
            Path("data/raw_videos"),
        ]
        
        labeled_data_found = False
        for label_dir in label_dirs:
            if label_dir.exists():
                label_files = list(label_dir.glob("*_labels.csv"))
                if label_files:
                    labeled_data_found = True
                    break
        
        if not labeled_data_found:
            print("\n⚠️  No labeled video data found!")
            print("You need to label your videos before building the dataset.")
            print("\n💡 Next step:")
            print("1. Run the labeling tool:")
            print("   python run.py label")
            print("2. Label jump events in your video files")
            print("3. Then come back and run training again")
            print("="*80)
            exit(0)
        
        response = input("\nWould you like to run the data builder now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🚀 Starting data builder...")
            from src.ml.data.builders.builder import main as build_main
            import sys
            
            # Build default arguments for data builder
            # Note: --build_data defaults to False, so we need to pass it to actually build data
            builder_args = [
                'builder.py',
                '--videos_dir', 'data/raw_videos_3',
                '--labels_dir', 'data/raw_videos_3',
                '--output_dir', 'data/dataset',
                '--build_data'  # Pass the flag to actually build data (default is preview mode)
            ]
            
            # Add any additional args passed to train command
            if args.args:
                builder_args.extend(args.args)
            
            sys.argv = builder_args
            build_main()
            print("\n✅ Data building completed! You can now start training.")
        else:
            print("❌ Training cancelled. Please build training data first.")
            exit(0)


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
          python run.py label --args --workdir data/raw_videos  # Data annotation
          python run.py visualize --args --model best_model.keras --video test.mp4  # Visualization
          python run.py build --args --videos_dir data/videos --labels_dir data/labels  # Build dataset
        """
    )

    parser.add_argument(
        'mode',
        nargs='?',
        default='realtime-v2',
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
        if not check_realtime_prerequisites():
            exit(1)
        from src.apps.main import main as app_main
        # Reset sys.argv to pass arguments to sub-application
        sys.argv = ['main.py'] + (args.args or [])
        app_main()

    elif args.mode == 'realtime-v2':
        if not check_realtime_prerequisites():
            exit(1)
        from src.apps.main_v2 import main as app_main_v2
        # Reset sys.argv to pass arguments to sub-application
        sys.argv = ['main_v2.py'] + (args.args or [])
        app_main_v2()

    elif args.mode == 'legacy':
        if not check_realtime_prerequisites():
            exit(1)
        from src.apps.main_0_5 import main as legacy_main
        sys.argv = ['main_0.5.py'] + (args.args or [])
        legacy_main()

    elif args.mode == 'train':
        check_training_data(args)
        from src.ml.training.model_training import main as train_main
        sys.argv = ['model_training.py'] + (args.args or [])
        train_main()

    elif args.mode == 'label':
        if not check_labeling_prerequisites():
            exit(1)
        from src.ml.data.labeling.main_gui import main as label_main
        # Parse annotation-related arguments
        import argparse as label_argparse
        label_parser = label_argparse.ArgumentParser()
        label_parser.add_argument('--workdir', default='data/raw_videos_3')
        # Parse only the remaining arguments, not all sys.argv
        label_args = label_parser.parse_args(args.args or [])
        label_main(label_args.workdir)

    elif args.mode == 'visualize':
        if not check_visualization_prerequisites():
            exit(1)
        from src.ml.visualization.model_visualize import main as viz_main
        sys.argv = ['model_visualize.py'] + (args.args or [])
        viz_main()

    elif args.mode == 'build':
        if not check_build_prerequisites():
            exit(1)
        from src.ml.data.builders.builder import main as build_main
        sys.argv = ['builder.py'] + (args.args or [])
        build_main()


if __name__ == "__main__":
    main()
