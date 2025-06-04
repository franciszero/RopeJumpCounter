# 机器学习模块 (ML Module)

本模块包含了跳绳计数器项目的所有机器学习相关功能，从原 `PoseDetection` 模块迁移而来。

## 📁 目录结构

```
src/ml/
├── data/                   # 数据处理
│   ├── labeling/          # 数据标注工具
│   │   ├── main_gui.py           # 主控标注界面
│   │   ├── label_helper_gui.py   # 标注助手GUI
│   │   └── verify_labels.py      # 标签验证工具
│   ├── builders/          # 数据构建工具
│   │   ├── feature_mode.py       # 特征模式定义
│   │   └── builder.py            # 数据集构建器
│   └── features/          # 特征提取
│       └── features.py           # 特征管道
├── models/                # 模型定义
│   ├── BaseModel.py              # 基础模型类
│   ├── CNN.py                    # CNN模型系列
│   ├── TCN.py                    # TCN模型
│   ├── ResNET1D.py              # ResNet1D模型
│   └── ...                      # 其他模型
├── training/              # 训练相关
│   └── model_training.py         # 模型训练脚本
└── visualization/         # 可视化验证
    └── model_visualize.py        # 模型可视化工具
```

## 🔧 功能模块

### 1. 数据标注 (`data/labeling/`)
- **主控界面** (`main_gui.py`): 管理多个视频的标注任务
- **标注助手** (`label_helper_gui.py`): 图形化标注工具
- **标签验证** (`verify_labels.py`): 验证标注质量

### 2. 数据构建 (`data/builders/`)
- **特征模式** (`feature_mode.py`): 定义特征提取模式
- **数据集构建** (`builder.py`): 从视频和标签生成训练数据

### 3. 特征提取 (`data/features/`)
- **特征管道** (`features.py`): 统一的特征提取流程

### 4. 模型定义 (`models/`)
- 包含多种深度学习模型架构
- 支持CNN、TCN、ResNet、Transformer等

### 5. 模型训练 (`training/`)
- **训练脚本** (`model_training.py`): 批量训练多个模型

### 6. 可视化验证 (`visualization/`)
- **模型可视化** (`model_visualize.py`): 实时预测可视化

## 🚀 使用方法

### 数据标注
```bash
cd src/ml/data/labeling
python main_gui.py --workdir /path/to/videos
```

### 数据集构建
```bash
cd src/ml/data/builders
python builder.py --videos_dir /path/to/videos --labels_dir /path/to/labels
```

### 模型训练
```bash
cd src/ml/training
python model_training.py
```

### 模型可视化
```bash
cd src/ml/visualization
python model_visualize.py --model best_model.keras --video test_video.mp4
```

## 📝 迁移说明

本模块从原 `PoseDetection` 目录迁移而来，主要变更：

1. **目录重组**: 按功能分类重新组织代码结构
2. **导入路径更新**: 所有导入路径已更新为新的模块路径
3. **模块化设计**: 更清晰的职责分离和模块边界

## 🔗 依赖关系

- `utils/`: 通用工具类（姿态估计、视频稳定等）
- `capture/`: 视频捕获模块
- 外部依赖: TensorFlow, OpenCV, MediaPipe等
