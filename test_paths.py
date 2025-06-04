#!/usr/bin/env python3
"""
测试脚本：验证所有模块的导入路径是否正确
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """测试所有主要模块的导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试核心模块
        print("  ✓ 测试核心模块...")
        from src.core.exceptions import AppError, ModelError, ConfigError
        from src.core.jump_counter import JumpCounter
        print("    ✓ 核心模块导入成功")
        
        # 测试配置模块
        print("  ✓ 测试配置模块...")
        from src.config.settings import AppConfig
        print("    ✓ 配置模块导入成功")
        
        # 测试工具模块
        print("  ✓ 测试工具模块...")
        from src.utils.Perf import PerfStats
        from src.utils.FrameSample import SELECTED_LM
        print("    ✓ 工具模块导入成功")
        
        # 测试捕获模块
        print("  ✓ 测试捕获模块...")
        from src.capture.pyav_capture import PyAVCapture
        print("    ✓ 捕获模块导入成功")
        
        # 测试ML模块
        print("  ✓ 测试ML模块...")
        from src.ml.data.builders.feature_mode import get_feature_mode
        from src.ml.data.features.features import FeaturePipeline
        print("    ✓ ML模块导入成功")
        
        print("✅ 所有模块导入测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_file_paths():
    """测试关键文件路径是否存在"""
    print("\n📁 测试文件路径...")
    
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
            print(f"  ❌ {file_path} - 文件不存在")
            all_exist = False
    
    if all_exist:
        print("✅ 所有关键文件都存在！")
    else:
        print("❌ 部分文件缺失")
    
    return all_exist

def test_labeling_paths():
    """测试标注工具的路径解析"""
    print("\n🏷️ 测试标注工具路径...")
    
    try:
        # 模拟标注工具的路径解析
        labeling_dir = Path("src/ml/data/labeling")
        main_gui_path = labeling_dir / "main_gui.py"
        label_helper_path = labeling_dir / "label_helper_gui.py"
        verify_labels_path = labeling_dir / "verify_labels.py"
        
        print(f"  标注目录: {labeling_dir}")
        print(f"  主界面: {main_gui_path} {'✓' if main_gui_path.exists() else '❌'}")
        print(f"  标注助手: {label_helper_path} {'✓' if label_helper_path.exists() else '❌'}")
        print(f"  验证工具: {verify_labels_path} {'✓' if verify_labels_path.exists() else '❌'}")
        
        # 测试相对路径解析
        if main_gui_path.exists():
            script_dir = main_gui_path.parent
            verify_script = script_dir / "verify_labels.py"
            label_script = script_dir / "label_helper_gui.py"
            
            print(f"  相对路径解析:")
            print(f"    验证脚本: {verify_script} {'✓' if verify_script.exists() else '❌'}")
            print(f"    标注脚本: {label_script} {'✓' if label_script.exists() else '❌'}")
            
            if verify_script.exists() and label_script.exists():
                print("✅ 标注工具路径解析正确！")
                return True
        
        print("❌ 标注工具路径解析失败")
        return False
        
    except Exception as e:
        print(f"❌ 路径测试错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始路径修复验证测试...\n")
    
    # 运行所有测试
    import_ok = test_imports()
    files_ok = test_file_paths()
    labeling_ok = test_labeling_paths()
    
    print(f"\n📊 测试结果总结:")
    print(f"  模块导入: {'✅ 通过' if import_ok else '❌ 失败'}")
    print(f"  文件路径: {'✅ 通过' if files_ok else '❌ 失败'}")
    print(f"  标注路径: {'✅ 通过' if labeling_ok else '❌ 失败'}")
    
    if import_ok and files_ok and labeling_ok:
        print("\n🎉 所有测试通过！路径修复成功！")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
