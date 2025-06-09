# RopeJumpCounter

A real-time jump rope counter based on deep learning, using pose estimation and temporal models for jump action detection.

## 🏗️ Project Architecture

📊 **Architecture Diagrams**: 
- [System Architecture (中文)](./docs/ARCHITECTURE.md) | [System Architecture (English)](./docs/ARCHITECTURE_EN.md)
- [Sequence Diagrams (中文)](./docs/SEQUENCE_DIAGRAM.md) | [Sequence Diagrams (English)](./docs/SEQUENCE_DIAGRAM_EN.md)

📚 **Complete Documentation**: [Documentation Index](./docs/README.md)

```
RopeJumpCounter/
├── src/                    # Source code directory
│   ├── apps/              # Application entry points
│   │   ├── main.py               # Main program (configured version)
│   │   ├── main_0_5.py          # Legacy main program
│   │   └── app.py               # Alternative entry point
│   ├── ml/                # Machine learning modules
│   │   ├── data/          # Data processing
│   │   │   ├── labeling/         # Data annotation tools
│   │   │   ├── builders/         # Data building tools
│   │   │   └── features/         # Feature extraction
│   │   ├── models/        # Model definitions (CNN, TCN, ResNet, etc.)
│   │   ├── training/      # Model training
│   │   └── visualization/ # Visualization validation
│   ├── core/              # Core business logic
│   │   ├── video_predictor.py    # Video predictor
│   │   ├── jump_counter.py       # Jump rope counter
│   │   └── exceptions.py         # Exception definitions
│   ├── interface/         # User interface
│   │   └── gui.py               # Graphical user interface
│   ├── config/            # Configuration management
│   │   └── settings.py          # Application configuration
│   ├── utils/             # Utility classes
│   │   ├── vision.py            # Computer vision tools
│   │   ├── Perf.py              # Performance statistics
│   │   └── VideoStabilizer.py   # Video stabilization
│   └── capture/           # Video capture
│       ├── pyav_capture.py      # PyAV video capture
│       └── gst_capture.py       # GStreamer capture
├── data/                  # Data directory
├── model_files/           # Model files
├── archive/               # Historical versions
├── main.py               # Main program (configured version)
└── run.py                # Unified entry point
```

## 🚀 Quick Start

### Install Dependencies

#### 1. Complete Installation (Recommended)
```bash
pip install -r requirements.txt
```

#### 2. Minimal Installation (Lightweight)
```bash
pip install -r requirements-minimal.txt
```

#### 3. Development Environment Installation
```bash
pip install -r requirements-dev.txt
```

**Dependency Package Description:**
- `requirements.txt` - Complete feature package, includes all core dependencies and optional features
- `requirements-minimal.txt` - Minimal dependency package, includes only essential functionality
- `requirements-dev.txt` - Development tools package, includes testing, code quality, and other development tools

### Run Application

#### 1. Real-time Jump Counting (Recommended)
```bash
python main.py
# or
python run.py realtime
```

#### 2. Legacy Real-time Counting
```bash
python run.py legacy
```

#### 3. Data Annotation
```bash
python run.py label --workdir data/raw_videos
```

#### 4. Model Training
```bash
python run.py train
```

#### 5. Model Visualization
```bash
python run.py visualize --model best_model.keras --video test.mp4
```

#### 6. Build Dataset
```bash
python run.py build --videos_dir data/videos --labels_dir data/labels
```

## 🎮 Features

- ✅ **Real-time Jump Counting**: Deep learning-based real-time action detection
- ✅ **Multi-model Support**: CNN, TCN, ResNet, Transformer, and more
- ✅ **Data Annotation Tools**: Graphical annotation interface
- ✅ **Model Training**: Batch training for multiple models
- ✅ **Visualization Validation**: Real-time prediction visualization
- ✅ **Configuration Management**: Flexible configuration system
- ✅ **Performance Optimization**: GPU acceleration, mixed precision

## 📊 Model Performance

Supports multiple deep learning models:
- CNN series (CNN8, CNNHybrid, etc.)
- TCN (Temporal Convolutional Network)
- ResNet1D
- EfficientNet1D
- Transformer
- And more...

## 🔧 Development Guide

### Adding New Models
1. Create model class in `src/ml/models/`
2. Inherit from `BaseModel` class
3. Register in `model_training.py`

### Adding New Features
1. Add functionality in appropriate module
2. Update configuration files
3. Add tests

## 📝 Changelog

- **v2.0**: Complete architecture refactoring, modular design
- **v1.x**: Original version, functional prototype

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License
