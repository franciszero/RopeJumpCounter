# 依赖管理说明

## 概述

RopeJumpCounter 项目采用模块化的依赖管理策略，提供3个不同的依赖包以满足不同用户的需求。

## 依赖包说明

### 1. requirements.txt - 完整功能包
**适用场景：** 大多数用户，需要完整功能
**安装命令：** `pip install -r requirements.txt`

**包含内容：**
- ✅ 所有核心依赖（深度学习、计算机视觉、数据处理）
- ✅ 机器学习增强功能
- ✅ 视频处理功能
- ✅ 性能监控工具
- ⚠️ GUI依赖（注释状态，需要时取消注释）
- ⚠️ 可视化依赖（注释状态，需要时取消注释）
- ⚠️ 开发工具（注释状态，需要时取消注释）

### 2. requirements-minimal.txt - 最小依赖包
**适用场景：** 轻量级部署、资源受限环境、仅需核心功能
**安装命令：** `pip install -r requirements-minimal.txt`

**包含内容：**
- ✅ 绝对必需的依赖（TensorFlow、OpenCV、MediaPipe）
- ✅ 基础数据处理工具
- ⚠️ 高级功能（注释状态，根据需要取消注释）

### 3. requirements-dev.txt - 开发工具包
**适用场景：** 开发者、贡献者、需要高级功能
**安装命令：** `pip install -r requirements-dev.txt`

**包含内容：**
- ✅ 测试框架（pytest、pytest-cov、pytest-mock）
- ✅ 代码质量工具（black、flake8、mypy、isort）
- ✅ 文档生成工具（sphinx、sphinx-rtd-theme）
- ✅ 开发环境（jupyter、ipython、ipykernel）
- ✅ 高级机器学习库（xgboost、lightgbm）
- ✅ 高级可视化工具（plotly、bokeh、matplotlib、seaborn）
- ✅ 实验跟踪（mlflow）
- ⚠️ 可选工具（注释状态，根据需要取消注释）

## 安装建议

### 新用户
```bash
# 推荐：安装完整功能包
pip install -r requirements.txt
```

### 轻量级用户
```bash
# 仅安装核心功能
pip install -r requirements-minimal.txt
```

### 开发者
```bash
# 安装开发工具包
pip install -r requirements-dev.txt
```

### 高级用户
```bash
# 组合安装：先安装完整包，再安装开发工具
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 依赖分类说明

### 核心依赖（必需）
- **tensorflow** - 深度学习框架
- **opencv-python** - 计算机视觉库
- **mediapipe** - 姿态估计
- **numpy** - 数值计算
- **PyYAML** - 配置文件处理
- **tqdm** - 进度条显示

### 增强依赖（推荐）
- **pandas** - 数据处理
- **scikit-learn** - 机器学习
- **xgboost** - 梯度提升
- **av** - 视频处理
- **psutil** - 性能监控

### GUI依赖（可选）
- **PySimpleGUIQt** - GUI框架
- **PySide6** - 替代GUI框架

### 可视化依赖（可选）
- **matplotlib** - 基础绘图
- **seaborn** - 统计绘图
- **plotly** - 交互式绘图
- **bokeh** - Web绘图

### 开发工具（可选）
- **pytest** - 测试框架
- **black** - 代码格式化
- **flake8** - 代码检查
- **mypy** - 类型检查
- **sphinx** - 文档生成
- **mlflow** - 实验跟踪

## 版本管理

### 版本策略
- 使用 `>=` 指定最低版本要求
- 避免使用 `==` 固定版本，保持灵活性
- 定期更新依赖版本以获取安全补丁

### 兼容性
- Python 3.8+
- TensorFlow 2.8+
- OpenCV 4.5+
- MediaPipe 0.8+

## 故障排除

### 常见问题

#### 1. GPU支持问题
```bash
# 确保安装GPU版本的TensorFlow
pip install tensorflow-gpu
```

#### 2. GUI依赖冲突
```bash
# 如果遇到GUI依赖冲突，可以只安装核心功能
pip install -r requirements-minimal.txt
```

#### 3. 开发工具安装失败
```bash
# 某些开发工具可能需要系统级依赖
# 在Ubuntu/Debian上：
sudo apt-get install python3-dev build-essential
```

### 环境隔离
推荐使用虚拟环境：
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 迁移指南

### 从旧版本迁移
如果你之前使用的是单个requirements文件：

1. **备份当前环境**
```bash
pip freeze > old_requirements.txt
```

2. **清理环境**
```bash
pip uninstall -r old_requirements.txt -y
```

3. **安装新依赖**
```bash
pip install -r requirements.txt
```

### 自定义依赖
如果需要自定义依赖组合：

1. 复制 `requirements.txt` 为新文件
2. 根据需要注释/取消注释相应依赖
3. 使用自定义文件安装：
```bash
pip install -r my_custom_requirements.txt
```

## 维护说明

### 更新依赖
1. 定期检查依赖更新
2. 测试新版本兼容性
3. 更新requirements文件
4. 更新文档

### 添加新依赖
1. 确定依赖分类（核心/增强/GUI/开发）
2. 添加到相应的requirements文件
3. 更新文档
4. 测试安装和功能

### 移除依赖
1. 确认依赖不再需要
2. 从所有相关文件移除
3. 更新文档
4. 测试功能完整性 