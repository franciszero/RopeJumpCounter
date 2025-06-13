# RopeJumpCounter

A real-time jump rope counting application using computer vision and machine learning to detect and count jump rope movements from video input.

## Features

- **Real-time Jump Detection**: Live video processing with instant jump counting
- **Multiple Application Modes**: Real-time counting, model training, data annotation, and visualization
- **Machine Learning Models**: CNN-based models for accurate jump detection
- **Video Stabilization**: Built-in video stabilization for better detection accuracy
- **GPU Acceleration**: Optional GPU support for improved performance
- **Comprehensive Logging**: Detailed logging system for debugging and analysis
- **Data Annotation Tools**: GUI-based tools for labeling training data
- **Model Visualization**: Tools to visualize model predictions and performance
- **Advanced Architecture**: v2.0 with dependency injection, event bus, and plugin system

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenCV
- TensorFlow/Keras
- NumPy
- MediaPipe
- PyYAML
- PyAV (for low-latency video capture)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RopeJumpCounter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
   - **For minimal setup (core functionality)**:
   ```bash
   pip install -r requirements-minimal.txt
   ```
   - **For complete functionality (all features)**:
   ```bash
   pip install -r requirements.txt
   ```
   - **For development (with testing and development tools)**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
   - **For detailed dependency information**: See `docs/DEPENDENCY_MANAGEMENT.md`

4. Configure the application:
   - The `config.yaml` file is already present and configured
   - Modify settings as needed for your environment

### Basic Usage

#### Real-time Jump Counting
```bash
# Run real-time jump counting (default - v2.0 architecture)
python run.py

# Run with new v2.0 architecture
python run.py realtime-v2

# Run legacy version
python run.py legacy

# Run original version
python run.py realtime
```

#### Model Training
```bash
python run.py train
```

#### Data Annotation
```bash
python run.py label --workdir data/raw_videos_3
```

#### Model Visualization
```bash
python run.py visualize --model best_model.keras --video test.mp4
```

#### Dataset Building
```bash
python run.py build --videos_dir data/videos --labels_dir data/labels
```

## Configuration

The application is configured through `config.yaml`. Key configuration options include:

### Camera Settings
```yaml
camera:
  width: 640              # Video width
  height: 480             # Video height
  fps: 30                 # Frame rate
  device_index: 0         # Camera device index
```

### Model Settings
```yaml
model:
  model_name: "best_cnn8_ws4_withT.keras"  # Model filename
  threshold: 0.5          # Jump detection threshold (0.0-1.0)
```

### Performance Settings
```yaml
performance:
  stats_window_size: 10   # Performance statistics window size
  max_fps: 30            # Maximum processing frame rate
```

## Project Structure

```
RopeJumpCounter/
├── run.py                 # Main entry point
├── config.yaml           # Configuration file
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── src/                  # Source code
│   ├── apps/            # Application modules
│   │   ├── main.py      # Original real-time app
│   │   ├── main_v2.py   # New v2.0 architecture
│   │   └── main_0_5.py  # Legacy version
│   ├── core/            # Core functionality
│   │   ├── container.py # Dependency injection
│   │   ├── event_bus.py # Event system
│   │   ├── plugin_manager.py # Plugin system
│   │   ├── jump_counter.py # State machine
│   │   └── pyav_capture.py # Video capture
│   ├── interface/       # User interfaces
│   │   └── gui.py       # Main GUI
│   ├── config/          # Configuration management
│   ├── utils/           # Utility functions
│   └── ml/              # Machine learning modules
│       ├── data/        # Data processing
│       ├── training/    # Model training
│       ├── visualization/ # Model visualization
│       ├── models/      # Model definitions
│       └── inference/   # Model inference
├── model_files/         # Trained models
├── data/                # Data and datasets
│   ├── raw_videos_3/    # Raw video files
│   └── dataset_16_10100/ # Processed datasets
├── logs/                # Application logs
├── docs/                # Documentation
│   ├── ARCHITECTURE.md  # System architecture guide
│   ├── DEPENDENCY_MANAGEMENT.md # Dependency management
│   └── SEQUENCE_DIAGRAM.md # Sequence diagrams
└── venv/                # Virtual environment
```

## Application Modes

### 1. Real-time Counting (`realtime`, `realtime-v2`, `legacy`)
- Live video processing for jump detection
- Real-time display of jump counts
- Performance statistics and debugging information
- **v2.0**: Advanced architecture with dependency injection and event bus

### 2. Training (`train`)
- Train new machine learning models
- Configure training parameters
- Model validation and testing

### 3. Data Annotation (`label`)
- GUI-based video annotation tool
- Label jump events in video files
- Export labeled datasets

### 4. Visualization (`visualize`)
- Visualize model predictions
- Analyze model performance
- Debug detection issues

### 5. Dataset Building (`build`)
- Build training datasets from videos and labels
- Data preprocessing and augmentation
- Dataset validation

## Machine Learning

The application uses convolutional neural networks (CNNs) for jump detection:

- **Model Architecture**: CNN with temporal features
- **Input**: Video frames with time window processing
- **Output**: Jump probability scores
- **Training**: Supervised learning with labeled video data
- **Models**: CNN, ResNet, TCN, LSTM, Transformer variants

## Performance Optimization

- **GPU Acceleration**: Enable GPU support for faster processing
- **Memory Management**: Configurable memory growth settings
- **Frame Rate Control**: Adjustable processing frame rates
- **Video Stabilization**: Built-in stabilization for better accuracy
- **Low-latency Capture**: PyAV for optimized video capture

## Logging and Debugging

The application provides comprehensive logging:

- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Directory**: Configurable log storage location
- **Debug Information**: Real-time debugging data display
- **Performance Metrics**: Processing statistics and timing

## Documentation

Additional documentation is available in the `docs/` directory:

- **ARCHITECTURE.md**: System architecture guide with detailed diagrams
- **DEPENDENCY_MANAGEMENT.md**: Comprehensive dependency management guide
- **SEQUENCE_DIAGRAM.md**: System sequence diagrams and workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the documentation in the `docs/` directory
- Review the configuration options in `config.yaml`
- Check the logs in the `logs/` directory for error information

## Version History

- **v2.0**: New architecture with dependency injection, event bus, and plugin system
- **v1.0**: Initial release with basic jump counting functionality
