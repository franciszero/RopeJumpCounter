#!/usr/bin/env python3
"""
Test Script: Verify all module import paths are correct

This script validates that all major modules can be imported correctly
and that critical file paths exist in the project structure.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test imports of all major modules"""
    print("🧪 Testing module imports...")

    try:
        # Test core modules
        print("  ✓ Testing core modules...")
        from src.core.exceptions import AppError, ModelError, ConfigError
        from src.core.jump_counter import JumpCounter
        print("    ✓ Core modules imported successfully")

        # Test configuration modules
        print("  ✓ Testing configuration modules...")
        from src.config.settings import AppConfig
        print("    ✓ Configuration modules imported successfully")

        # Test utility modules
        print("  ✓ Testing utility modules...")
        from src.utils.Perf import PerfStats
        from src.utils.FrameSample import SELECTED_LM
        print("    ✓ Utility modules imported successfully")

        # Test capture modules
        print("  ✓ Testing capture modules...")
        from src.capture.pyav_capture import PyAVCapture
        print("    ✓ Capture modules imported successfully")

        # Test ML modules
        print("  ✓ Testing ML modules...")
        from src.ml.data.builders.feature_mode import get_feature_mode
        from src.ml.data.features.features import FeaturePipeline
        print("    ✓ ML modules imported successfully")

        print("✅ All module import tests passed!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_file_paths():
    """Test whether critical file paths exist"""
    print("\n📁 Testing file paths...")

    files_to_check = [
        "src/ml/data/labeling/main_gui.py",
        "src/ml/data/labeling/label_helper_gui.py",
        "src/ml/data/labeling/verify_labels.py",
        "src/ml/data/builders/builder.py",
        "src/ml/data/features/features.py",
        "src/ml/training/model_training.py",
        "src/ml/visualization/model_visualize.py",
        "src/apps/main.py",
        "src/apps/main_0_5.py",
        "run.py",
        "main.py"
    ]

    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path} - File does not exist")
            all_exist = False

    if all_exist:
        print("✅ All critical files exist!")
    else:
        print("❌ Some files are missing")

    return all_exist

def test_labeling_paths():
    """Test labeling tool path resolution"""
    print("\n🏷️ Testing labeling tool paths...")

    try:
        # Simulate labeling tool path resolution
        labeling_dir = Path("src/ml/data/labeling")
        main_gui_path = labeling_dir / "main_gui.py"
        label_helper_path = labeling_dir / "label_helper_gui.py"
        verify_labels_path = labeling_dir / "verify_labels.py"

        print(f"  Labeling directory: {labeling_dir}")
        print(f"  Main GUI: {main_gui_path} {'✓' if main_gui_path.exists() else '❌'}")
        print(f"  Label helper: {label_helper_path} {'✓' if label_helper_path.exists() else '❌'}")
        print(f"  Verification tool: {verify_labels_path} {'✓' if verify_labels_path.exists() else '❌'}")

        # Test relative path resolution
        if main_gui_path.exists():
            script_dir = main_gui_path.parent
            verify_script = script_dir / "verify_labels.py"
            label_script = script_dir / "label_helper_gui.py"

            print(f"  Relative path resolution:")
            print(f"    Verification script: {verify_script} {'✓' if verify_script.exists() else '❌'}")
            print(f"    Labeling script: {label_script} {'✓' if label_script.exists() else '❌'}")

            if verify_script.exists() and label_script.exists():
                print("✅ Labeling tool path resolution correct!")
                return True

        print("❌ Labeling tool path resolution failed")
        return False

    except Exception as e:
        print(f"❌ Path test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting path repair verification tests...\n")

    # Run all tests
    import_ok = test_imports()
    files_ok = test_file_paths()
    labeling_ok = test_labeling_paths()

    print(f"\n📊 Test results summary:")
    print(f"  Module imports: {'✅ Passed' if import_ok else '❌ Failed'}")
    print(f"  File paths: {'✅ Passed' if files_ok else '❌ Failed'}")
    print(f"  Labeling paths: {'✅ Passed' if labeling_ok else '❌ Failed'}")

    if import_ok and files_ok and labeling_ok:
        print("\n🎉 All tests passed! Path repair successful!")
        return True
    else:
        print("\n⚠️ Some tests failed, further repair needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
