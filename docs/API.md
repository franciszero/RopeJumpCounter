# API 文档

## 核心 API

### VideoPredictor

视频预测器，封装模型推理逻辑。

```python
from src.core.video_predictor import VideoPredictor

# 初始化
predictor = VideoPredictor(model_path="path/to/model.keras")

# 预测
probability = predictor.predict(feature_vector)
```

#### 方法

##### `__init__(model_path: str)`
- **参数**: 
  - `model_path`: 模型文件路径
- **异常**: `ModelError` 如果模型加载失败

##### `predict(feature_dim: np.ndarray) -> float`
- **参数**: 
  - `feature_dim`: 特征向量 (numpy array)
- **返回**: 跳跃概率 (0.0-1.0)

### JumpCounter

跳绳计数器，处理预测结果并计数。

```python
from src.core.jump_counter import JumpCounter

# 初始化
counter = JumpCounter()

# 处理预测
is_rising, count = counter.process_prediction(probability, threshold)
```

#### 方法

##### `process_prediction(prob: float, threshold: float) -> tuple[bool, int]`
- **参数**:
  - `prob`: 预测概率
  - `threshold`: 阈值
- **返回**: (是否正在上升, 当前跳数)

### FeaturePipeline

特征提取管道。

```python
from src.ml.data.features.features import FeaturePipeline

# 初始化
pipeline = FeaturePipeline(video_capture, window_size=4)

# 处理帧
pipeline.process_frame(frame, frame_index)
```

## 配置 API

### AppConfig

应用配置管理。

```python
from src.config.settings import AppConfig

# 加载配置
config = AppConfig.load()

# 访问配置
print(config.camera.width)
print(config.model.model_path)
```

## 工具 API

### PoseEstimator

姿态估计器。

```python
from src.utils.vision import PoseEstimator

estimator = PoseEstimator()
landmarks = estimator.get_pose_landmarks(frame)
```

### VideoStabilizer

视频稳定器。

```python
from src.utils.VideoStabilizer import VideoStabilizer

stabilizer = VideoStabilizer()
stable_frame = stabilizer.stabilize(frame)
```

## 数据处理 API

### 数据标注

```python
# 启动标注界面
from src.ml.data.labeling.main_gui import main
main()
```

### 数据集构建

```python
from src.ml.data.builders.builder import main
# 通过命令行参数配置
```

## 模型训练 API

### 训练器

```python
from src.ml.training.model_training import Trainer

trainer = Trainer()
trainer.train()  # 训练所有模型
```

## 异常处理

### 自定义异常

```python
from src.core.exceptions import AppError, ModelError, ConfigError

try:
    # 应用逻辑
    pass
except ModelError as e:
    print(f"模型错误: {e}")
except ConfigError as e:
    print(f"配置错误: {e}")
except AppError as e:
    print(f"应用错误: {e}")
```

## 事件回调

### 跳跃事件

```python
class MyJumpObserver:
    def on_jump_detected(self, count: int, probability: float):
        print(f"检测到跳跃! 总数: {count}, 概率: {probability}")

# 注册观察者 (如果实现了观察者模式)
counter.add_observer(MyJumpObserver())
```

## 性能监控

### PerfStats

```python
from src.utils.Perf import PerfStats

stats = PerfStats(window_size=10)
# 在处理循环中
stats.update("Process", timestamps, 0)
print(f"FPS: {stats.proc_fps}")
```

## 配置示例

### 完整配置文件

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

save_video_path: "data/recordings"
```
