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

- **Python 3.10 or higher** (required for modern type annotations and Keras 3.x compatibility)
- OpenCV
- TensorFlow/Keras 3.x
- NumPy
- MediaPipe
- PyYAML
- PyAV (for low-latency video capture)
- **GUI Dependencies** (for annotation tools):
  - PySimpleGUIQt
  - PySide6 (Qt bindings for PySimpleGUIQt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RopeJumpCounter
```

2. Create and activate a virtual environment (recommended):
```bash
python3.10 -m venv venv
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
   - **For complete functionality (all features including GUI tools)**:
   ```bash
   pip install -r requirements.txt
   ```
   - **For development (with testing and development tools)**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
   - **For detailed dependency information**: See `docs/DEPENDENCY_MANAGEMENT.md`

   - This is a one-time setup - subsequent `pip install -r requirements.txt` will handle dependencies automatically.

4. Configure the application:
   - The `config.yaml` file is already present and configured
   - Modify settings as needed for your environment

### Workflow Overview

The application follows a specific workflow for machine learning model development:

```
1. Data Collection & Annotation
   ↓
2. Dataset Building
   ↓
3. Model Training
   ↓
4. Model Evaluation & Visualization
   ↓
5. Real-time Deployment
```

**Detailed Workflow:**

1. **Data Annotation** (`label`) - First step for new projects
   - Label jump events in video files
   - Creates `*_labels.csv` files

2. **Dataset Building** (`build`) - Required before training
   - Processes videos and labels into training datasets
   - Creates `data/dataset_*/size{window_size}/` structure

3. **Model Training** (`train`) - Requires built datasets
   - Trains machine learning models
   - Saves trained models to `model_files/`

4. **Visualization** (`visualize`) - Optional, for analysis
   - Visualizes model predictions
   - Helps debug and analyze performance

5. **Real-time Counting** (`realtime*`) - Final deployment
   - Uses trained models for live jump counting

### Basic Usage

#### Step 1: Data Annotation (First Time Setup)
```bash
# Label jump events in your video files
python run.py label --args --workdir data/raw_videos_3
```

#### Step 2: Build Training Dataset
```bash
# Process videos and labels into training data
python run.py build --args --videos_dir data/raw_videos_3 --labels_dir data/raw_videos_3
```

#### Step 3: Train Models
```bash
# Train machine learning models (requires built datasets)
python run.py train
```

#### Step 4: Real-time Jump Counting
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

#### Optional: Model Visualization
```bash
# Visualize model predictions and performance
python run.py visualize --args --model best_cnn8_ws4_withT.keras --video data/raw_videos_3/jump_2025.05.22.08.34.40__100.avi
```

### Quick Commands Reference

| Command | Purpose | Prerequisites |
|---------|---------|---------------|
| `python run.py label` | Label video data | Video files in data directory |
| `python run.py build` | Build training datasets | Labeled videos (`*_labels.csv`) |
| `python run.py train` | Train models | Built datasets (`data/dataset_*/`) |
| `python run.py visualize` | Analyze model performance | Trained models (`model_files/`) |
| `python run.py realtime*` | Live jump counting | Trained models (`model_files/`) |

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
├── requirements.txt      # Complete dependencies (including GUI)
├── requirements-minimal.txt # Minimal dependencies (core only)
├── requirements-dev.txt  # Development dependencies
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
│       │   └── labeling/ # Data annotation tools
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
- GUI-based video annotation tool (requires PySide6)
- Label jump events in video files
- Export labeled datasets
- **Note**: Requires GUI dependencies (PySimpleGUIQt + PySide6)

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

## Troubleshooting

### Common Issues

1. **GUI Tools Not Working**: 
   - Ensure PySide6 is installed: `pip install PySide6`
   - Use complete requirements: `pip install -r requirements.txt`
   - If you see "No module named 'PySimpleGUIQt'" error, ensure you've activated the virtual environment and run `pip install -r requirements.txt`

2. **PySimpleGUI Private Server Warning**:
   - If you encounter "PySimpleGUI is now located on a private PyPI server..." warning, run:
   ```bash
   python -m pip install --upgrade --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
   ```

3. **Model File Not Found Error**:
   - When using `python run.py visualize`, use only the model filename (e.g., `best_cnn8_ws4_withT.keras`), not the full path
   - The system automatically constructs the correct path based on the model filename

4. **Command Line Arguments**:
   - Use `--args` flag to pass arguments to sub-applications
   - Example: `python run.py label --args --workdir data/raw_videos_3`

5. **Missing Dependencies**:
   - Install complete requirements for all features
   - Check `requirements.txt` for full dependency list

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
