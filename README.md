# RopeJumpCounter

基于深度学习的实时跳绳计数器，使用姿态估计和时序模型进行跳跃动作检测。

## 🏗️ 项目架构

```
RopeJumpCounter/
├── src/                    # 源代码目录
│   ├── apps/              # 应用程序入口
│   │   ├── main.py               # 配置版主程序
│   │   ├── main_0_5.py          # 旧版本主程序  
│   │   └── app.py               # 备用入口
│   ├── ml/                # 机器学习模块
│   │   ├── data/          # 数据处理
│   │   │   ├── labeling/         # 数据标注工具
│   │   │   ├── builders/         # 数据构建工具
│   │   │   └── features/         # 特征提取
│   │   ├── models/        # 模型定义 (CNN、TCN、ResNet等)
│   │   ├── training/      # 模型训练
│   │   └── visualization/ # 可视化验证
│   ├── core/              # 核心业务逻辑
│   │   ├── video_predictor.py    # 视频预测器
│   │   ├── jump_counter.py       # 跳绳计数器
│   │   └── exceptions.py         # 异常定义
│   ├── interface/         # 用户界面
│   │   └── gui.py               # 图形用户界面
│   ├── config/            # 配置管理
│   │   └── settings.py          # 应用配置
│   ├── utils/             # 工具类
│   │   ├── vision.py            # 计算机视觉工具
│   │   ├── Perf.py              # 性能统计
│   │   └── VideoStabilizer.py   # 视频稳定
│   └── capture/           # 视频捕获
│       ├── pyav_capture.py      # PyAV视频捕获
│       └── gst_capture.py       # GStreamer捕获
├── data/                  # 数据目录
├── model_files/           # 模型文件
├── archive/               # 历史版本
├── main.py               # 主程序 (配置版)
└── run.py                # 统一入口
```

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行应用

#### 1. 实时跳绳计数 (推荐)
```bash
python main.py
# 或
python run.py realtime
```

#### 2. 旧版本实时计数
```bash
python run.py legacy
```

#### 3. 数据标注
```bash
python run.py label --workdir data/raw_videos
```

#### 4. 模型训练
```bash
python run.py train
```

#### 5. 模型可视化
```bash
python run.py visualize --model best_model.keras --video test.mp4
```

#### 6. 构建数据集
```bash
python run.py build --videos_dir data/videos --labels_dir data/labels
```

## 🎮 功能特性

- ✅ **实时跳绳计数**: 基于深度学习的实时动作检测
- ✅ **多模型支持**: CNN、TCN、ResNet、Transformer等
- ✅ **数据标注工具**: 图形化标注界面
- ✅ **模型训练**: 批量训练多种模型
- ✅ **可视化验证**: 实时预测可视化
- ✅ **配置管理**: 灵活的配置系统
- ✅ **性能优化**: GPU加速、混合精度

## 📊 模型性能

支持多种深度学习模型：
- CNN系列 (CNN8, CNNHybrid等)
- TCN (Temporal Convolutional Network)
- ResNet1D
- EfficientNet1D
- Transformer
- 等等...

## 🔧 开发指南

### 添加新模型
1. 在 `src/ml/models/` 中创建模型类
2. 继承 `BaseModel` 类
3. 在 `model_training.py` 中注册

### 添加新功能
1. 在相应模块中添加功能
2. 更新配置文件
3. 添加测试

## 📝 更新日志

- **v2.0**: 完整架构重构，模块化设计
- **v1.x**: 原始版本，功能原型

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
