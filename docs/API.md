# API Documentation

## Core API

### VideoPredictor

Video predictor that encapsulates model inference logic.

```python
from src.core.video_predictor import VideoPredictor

# Initialize
predictor = VideoPredictor(model_path="path/to/model.keras")

# Predict
probability = predictor.predict(feature_vector)
```

#### Methods

##### `__init__(model_path: str, threshold: float = 0.5)`
- **Parameters**: 
  - `model_path`: Path to model file
  - `threshold`: Decision threshold for binary classification
- **Raises**: `ModelError` if model loading fails

##### `predict(feature_vector: np.ndarray) -> float`
- **Parameters**: 
  - `feature_vector`: Feature vector (numpy array)
- **Returns**: Jump probability (0.0-1.0)

##### `is_ready() -> bool`
- **Returns**: True if predictor is ready for inference

##### `reset()`
- **Description**: Reset the sliding window buffer

### JumpCounter

Jump rope counter that processes prediction results and counts jumps.

```python
from src.core.jump_counter import JumpCounter

# Initialize
counter = JumpCounter()

# Process prediction
is_rising, count = counter.process_prediction(probability, threshold)
```

#### Methods

##### `process_prediction(prob: float, threshold: float) -> tuple[bool, int]`
- **Parameters**:
  - `prob`: Prediction probability
  - `threshold`: Decision threshold
- **Returns**: (is_rising, current_jump_count)

##### `get_count() -> int`
- **Returns**: Current jump count

##### `reset()`
- **Description**: Reset counter to initial state

### FeaturePipeline

Feature extraction pipeline for processing video frames.

```python
from src.ml.data.features.features import FeaturePipeline

# Initialize
pipeline = FeaturePipeline(video_capture, window_size=4)

# Process frame
pipeline.process_frame(frame, frame_index)
```

#### Methods

##### `process_frame(frame, frame_idx, mode=None)`
- **Parameters**:
  - `frame`: Input video frame (BGR format)
  - `frame_idx`: Frame index for temporal tracking
  - `mode`: Feature extraction mode (uses default if None)

## Configuration API

### AppConfig

Application configuration management.

```python
from src.config.settings import AppConfig

# Load configuration
config = AppConfig.load()

# Access configuration
print(config.camera.width)
print(config.model.model_path)
```

#### Methods

##### `load() -> AppConfig`
- **Returns**: Loaded configuration instance
- **Raises**: `ConfigError` if configuration loading fails

#### Configuration Classes

##### `CameraConfig`
- `width: int` - Video capture width
- `height: int` - Video capture height
- `fps: int` - Target frames per second
- `device_index: int` - Camera device index

##### `ModelConfig`
- `model_name: str` - Model file name
- `threshold: float` - Decision threshold
- `model_path: Path` - Complete path to model file

##### `LogConfig`
- `enabled: bool` - Whether logging is enabled
- `log_dir: Path` - Log output directory
- `level: str` - Log level (DEBUG, INFO, WARNING, ERROR)

## Utility API

### PoseEstimator

Pose estimation using MediaPipe.

```python
from src.utils.vision import PoseEstimator

estimator = PoseEstimator()
landmarks = estimator.get_pose_landmarks(frame)
```

#### Methods

##### `get_pose_landmarks(stable_frame)`
- **Parameters**: Input video frame in BGR format
- **Returns**: MediaPipe pose landmarks object or None

##### `estimate1(stable_frame)`
- **Parameters**: Input video frame in BGR format
- **Returns**: Dictionary mapping region names to normalized y-coordinates

### VideoStabilizer

Video stabilization for improved pose detection.

```python
from src.utils.VideoStabilizer import VideoStabilizer

stabilizer = VideoStabilizer()
stable_frame = stabilizer.stabilize(frame)
```

### PerfStats

Performance monitoring and statistics.

```python
from src.utils.Perf import PerfStats

stats = PerfStats(window_size=10)
# In processing loop
stats.update("Process", timestamps, 0)
print(f"FPS: {stats.proc_fps}")
```

#### Methods

##### `update(msg: str, arr_ts: list, limit: int = 10)`
- **Parameters**:
  - `msg`: Description message
  - `arr_ts`: Array of timestamps
  - `limit`: Frames between detailed logging

##### `info_text(video_fps: float) -> str`
- **Parameters**: Input video frame rate
- **Returns**: Formatted performance string

##### `get_metrics() -> dict`
- **Returns**: Performance metrics dictionary

## Data Processing API

### Data Annotation

```python
# Start annotation interface
from src.ml.data.labeling.main_gui import main
main(workdir="data/raw_videos")
```

### Dataset Building

```python
from src.ml.data.builders.builder import main
# Configure through command line arguments
```

## Model Training API

### Trainer

```python
from src.ml.training.model_training import Trainer

trainer = Trainer()
trainer.train()  # Train all models
```

## Visualization API

### Model Visualization

```python
from src.ml.visualization.model_visualize import PlayerGUI, VideoPredictor

predictor = VideoPredictor("model.keras")
gui = PlayerGUI("video.mp4", predictor, show_stick_figure=True)
gui.run()
```

#### PlayerGUI Methods

##### `__init__(video_path: str, predictor: VideoPredictor, show_stick_figure: bool = True)`
- **Parameters**:
  - `video_path`: Path to video file
  - `predictor`: Trained model predictor
  - `show_stick_figure`: Whether to show pose visualization

##### `run()`
- **Description**: Start the visualization interface

## Exception Handling

### Custom Exceptions

```python
from src.core.exceptions import AppError, ModelError, ConfigError, CameraError

try:
    # Application logic
    pass
except ModelError as e:
    print(f"Model error: {e}")
except ConfigError as e:
    print(f"Configuration error: {e}")
except CameraError as e:
    print(f"Camera error: {e}")
except AppError as e:
    print(f"Application error: {e}")
```

#### Exception Hierarchy

- `AppError` - Base application exception
  - `ModelError` - Machine learning model errors
  - `ConfigError` - Configuration-related errors
  - `CameraError` - Camera-related errors

## Interface API

### PlayerGUI

Real-time video player with jump counting.

```python
from src.interface.gui import PlayerGUI
from src.core.video_predictor import VideoPredictor

predictor = VideoPredictor("model.keras")
gui = PlayerGUI(predictor, width=640, height=480, fps=30)
gui.run()
```

#### Methods

##### `__init__(predictor, width, height, fps, save_path=None)`
- **Parameters**:
  - `predictor`: VideoPredictor instance
  - `width`: Video capture width
  - `height`: Video capture height
  - `fps`: Target frame rate
  - `save_path`: Optional video recording path

##### `run()`
- **Description**: Start the main video processing loop

## Configuration Examples

### Complete Configuration File

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

### Environment Variables

```bash
# Camera settings
export CAMERA_WIDTH=640
export CAMERA_HEIGHT=480
export CAMERA_FPS=30
export CAMERA_DEVICE=0

# Model settings
export MODEL_NAME="best_cnn8_ws4_withT.keras"
export MODEL_THRESHOLD=0.5

# Logging settings
export LOG_ENABLED=true
export LOG_DIR="logs"
export LOG_LEVEL="INFO"

# Video recording
export SAVE_VIDEO_PATH="data/recordings"
```

## Usage Examples

### Basic Real-time Counting

```python
from src.config.settings import AppConfig
from src.core.video_predictor import VideoPredictor
from src.interface.gui import PlayerGUI

# Load configuration
config = AppConfig.load()

# Initialize components
predictor = VideoPredictor(str(config.model.model_path))
gui = PlayerGUI(
    predictor=predictor,
    width=config.camera.width,
    height=config.camera.height,
    fps=config.camera.fps
)

# Start application
gui.run()
```

### Video File Processing

```python
from src.ml.visualization.model_visualize import PlayerGUI, VideoPredictor

predictor = VideoPredictor("model.keras")
gui = PlayerGUI("input_video.mp4", predictor)
gui.run()
```

### Custom Feature Extraction

```python
from src.ml.data.features.features import FeaturePipeline
from src.capture.pyav_capture import PyAVCapture

cap = PyAVCapture(device_index=0, width=640, height=480, fps=30)
pipeline = FeaturePipeline(cap, window_size=4)

# Process frames
for frame_idx in range(100):
    ret, frame, _ = cap.read()
    if ret:
        pipeline.process_frame(frame, frame_idx)
        features = pipeline.fs.rec  # Access extracted features
```
