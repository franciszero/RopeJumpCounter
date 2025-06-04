# 安装指南

## 系统要求

### 硬件要求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **内存**: 8GB RAM 最低，16GB 推荐
- **GPU**: NVIDIA GPU (可选，用于加速训练)
- **存储**: 至少 5GB 可用空间
- **摄像头**: USB 摄像头或内置摄像头

### 软件要求
- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 - 3.11
- **Git**: 用于克隆代码库

## 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/your-username/RopeJumpCounter.git
cd RopeJumpCounter
```

### 2. 创建虚拟环境
```bash
# 使用 venv
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

#### 选项 A: 最小安装 (推荐)
```bash
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

#### 选项 B: 完整安装
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 选项 C: 手动安装核心依赖
```bash
pip install tensorflow>=2.8.0 opencv-python>=4.5.0 mediapipe>=0.8.0 numpy pandas PyYAML tqdm
```

### 4. GPU 支持 (可选)
如果你有 NVIDIA GPU：
```bash
# 安装 CUDA 支持的 TensorFlow
pip install tensorflow[and-cuda]
```

### 5. 验证安装
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

## 配置

### 1. 创建配置文件
```bash
cp config.yaml.example config.yaml
```

### 2. 编辑配置
根据你的硬件配置编辑 `config.yaml`：
```yaml
camera:
  width: 640
  height: 480
  fps: 30
  device_index: 0

model:
  model_name: "best_cnn8_ws4_withT.keras"
  threshold: 0.5

logging:
  enabled: true
  log_dir: "logs"
  level: "INFO"
```

## 测试安装

### 1. 运行基本测试
```bash
python run.py --help
```

### 2. 测试摄像头
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### 3. 运行应用
```bash
python main.py
```

## 常见问题

### Q: 摄像头无法打开
A: 检查摄像头权限，尝试不同的 device_index (0, 1, 2...)

### Q: GPU 不被识别
A: 确保安装了正确的 CUDA 版本和 GPU 驱动

### Q: 模型加载失败
A: 确保模型文件存在于 `model_files/` 目录

### Q: 依赖冲突
A: 使用虚拟环境，清理后重新安装

## 开发环境设置

### 1. 安装开发依赖
```bash
pip install -r requirements-dev.txt
```

### 2. 设置代码格式化
```bash
pre-commit install
```

### 3. 运行测试
```bash
pytest tests/
```

## 更新

### 更新代码
```bash
git pull origin main
```

### 更新依赖
```bash
pip install -r requirements.txt --upgrade
```
