# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: NVIDIA GPU (optional, for training acceleration)
- **Storage**: At least 5GB available space
- **Camera**: USB camera or built-in camera

### Software Requirements
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 - 3.11
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Project
```bash
git clone https://github.com/your-username/RopeJumpCounter.git
cd RopeJumpCounter
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

#### Option A: Minimal Installation (Recommended for first-time users)
```bash
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

#### Option B: Complete Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option C: Manual Core Dependencies Installation
```bash
pip install tensorflow>=2.8.0 opencv-python>=4.5.0 mediapipe>=0.8.0 numpy pandas PyYAML tqdm
```

### 4. GPU Support (Optional)
If you have an NVIDIA GPU:
```bash
# Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]
```

### 5. Verify Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

## Configuration

### 1. Create Configuration File
```bash
cp config.yaml.example config.yaml
```

### 2. Edit Configuration
Edit `config.yaml` according to your hardware configuration:
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

# Data paths configuration
paths:
  model_files: "model_files/models_16_10100"  # Update this path to match your model directory
  data_dir: "data"
  raw_videos: "data/raw_videos"
  datasets: "data/datasets"
```

**Important**: Update the `model_files` path in the configuration to match your actual model directory. The default models are located in `model_files/models_16_10100/`.

## Testing Installation

### 1. Run Basic Test
```bash
python run.py --help
```

### 2. Test Camera
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### 3. Run Application
```bash
# Run with new architecture (recommended)
python run.py realtime-v2

# Or run with legacy architecture
python run.py realtime

# Or run legacy version
python run.py legacy
```

**Note**: The default mode is `realtime-v2` which uses the new architecture with dependency injection and event bus.

## Common Issues

### Q: Camera cannot be opened
A: Check camera permissions, try different device_index (0, 1, 2...)

### Q: GPU not recognized
A: Ensure correct CUDA version and GPU drivers are installed

### Q: Model loading failed
A: Ensure model files exist in the correct directory and update the `model_files` path in config.yaml

### Q: Dependency conflicts
A: Use virtual environment, clean and reinstall

### Q: Module not found errors
A: Make sure you're running from the project root directory and have activated the virtual environment

## Development Environment Setup

### 1. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Setup Code Formatting
```bash
pre-commit install
```

### 3. Run Tests
```bash
pytest tests/
```

## Updates

### Update Code
```bash
git pull origin main
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## Available Application Modes

- **realtime-v2**: Real-time counting with new architecture (default)
- **realtime**: Real-time jump counting with legacy architecture
- **legacy**: Legacy real-time counting
- **train**: Model training
- **label**: Data annotation
- **visualize**: Model visualization
- **build**: Build dataset

## Examples

```bash
# Real-time jump counting (new architecture)
python run.py realtime-v2

# Data annotation
python run.py label --workdir data/raw_videos

# Model visualization
python run.py visualize --model best_cnn8_ws4_withT.keras --video test.mp4

# Build dataset
python run.py build --videos_dir data/videos --labels_dir data/labels
``` 