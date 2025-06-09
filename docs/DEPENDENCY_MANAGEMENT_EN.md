# Dependency Management Guide

## Overview

The RopeJumpCounter project adopts a modular dependency management strategy, providing 3 different dependency packages to meet the needs of different users.

## Dependency Package Description

### 1. requirements.txt - Complete Feature Package
**Use Case:** Most users who need complete functionality
**Install Command:** `pip install -r requirements.txt`

**Contents:**
- ✅ All core dependencies (deep learning, computer vision, data processing)
- ✅ Machine learning enhancement features
- ✅ Video processing functionality
- ✅ Performance monitoring tools
- ⚠️ GUI dependencies (commented out, uncomment when needed)
- ⚠️ Visualization dependencies (commented out, uncomment when needed)
- ⚠️ Development tools (commented out, uncomment when needed)

### 2. requirements-minimal.txt - Minimal Dependency Package
**Use Case:** Lightweight deployment, resource-constrained environments, core functionality only
**Install Command:** `pip install -r requirements-minimal.txt`

**Contents:**
- ✅ Absolutely essential dependencies (TensorFlow, OpenCV, MediaPipe)
- ✅ Basic data processing tools
- ⚠️ Advanced features (commented out, uncomment as needed)

### 3. requirements-dev.txt - Development Tools Package
**Use Case:** Developers, contributors, users needing advanced features
**Install Command:** `pip install -r requirements-dev.txt`

**Contents:**
- ✅ Testing framework (pytest, pytest-cov, pytest-mock)
- ✅ Code quality tools (black, flake8, mypy, isort)
- ✅ Documentation generation tools (sphinx, sphinx-rtd-theme)
- ✅ Development environment (jupyter, ipython, ipykernel)
- ✅ Advanced machine learning libraries (xgboost, lightgbm)
- ✅ Advanced visualization tools (plotly, bokeh, matplotlib, seaborn)
- ✅ Experiment tracking (mlflow)
- ⚠️ Optional tools (commented out, uncomment as needed)

## Installation Recommendations

### New Users
```bash
# Recommended: Install complete feature package
pip install -r requirements.txt
```

### Lightweight Users
```bash
# Install core functionality only
pip install -r requirements-minimal.txt
```

### Developers
```bash
# Install development tools package
pip install -r requirements-dev.txt
```

### Advanced Users
```bash
# Combined installation: Install complete package first, then development tools
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Dependency Classification

### Core Dependencies (Required)
- **tensorflow** - Deep learning framework
- **opencv-python** - Computer vision library
- **mediapipe** - Pose estimation
- **numpy** - Numerical computing
- **PyYAML** - Configuration file processing
- **tqdm** - Progress bar display

### Enhancement Dependencies (Recommended)
- **pandas** - Data processing
- **scikit-learn** - Machine learning
- **xgboost** - Gradient boosting
- **av** - Video processing
- **psutil** - Performance monitoring

### GUI Dependencies (Optional)
- **PySimpleGUIQt** - GUI framework
- **PySide6** - Alternative GUI framework

### Visualization Dependencies (Optional)
- **matplotlib** - Basic plotting
- **seaborn** - Statistical plotting
- **plotly** - Interactive plotting
- **bokeh** - Web plotting

### Development Tools (Optional)
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Code linting
- **mypy** - Type checking
- **sphinx** - Documentation generation
- **mlflow** - Experiment tracking

## Version Management

### Version Strategy
- Use `>=` to specify minimum version requirements
- Avoid using `==` to fix versions, maintain flexibility
- Regularly update dependency versions to get security patches

### Compatibility
- Python 3.8+
- TensorFlow 2.8+
- OpenCV 4.5+
- MediaPipe 0.8+

## Troubleshooting

### Common Issues

#### 1. GPU Support Issues
```bash
# Ensure GPU version of TensorFlow is installed
pip install tensorflow-gpu
```

#### 2. GUI Dependency Conflicts
```bash
# If encountering GUI dependency conflicts, install core functionality only
pip install -r requirements-minimal.txt
```

#### 3. Development Tools Installation Failure
```bash
# Some development tools may require system-level dependencies
# On Ubuntu/Debian:
sudo apt-get install python3-dev build-essential
```

### Environment Isolation
Recommended to use virtual environments:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Migration Guide

### Migrating from Old Versions
If you previously used a single requirements file:

1. **Backup current environment**
```bash
pip freeze > old_requirements.txt
```

2. **Clean environment**
```bash
pip uninstall -r old_requirements.txt -y
```

3. **Install new dependencies**
```bash
pip install -r requirements.txt
```

### Custom Dependencies
If you need to customize dependency combinations:

1. Copy `requirements.txt` to a new file
2. Comment/uncomment dependencies as needed
3. Install using custom file:
```bash
pip install -r my_custom_requirements.txt
```

## Maintenance Notes

### Updating Dependencies
1. Regularly check for dependency updates
2. Test compatibility with new versions
3. Update requirements files
4. Update documentation

### Adding New Dependencies
1. Determine dependency category (core/enhancement/GUI/development)
2. Add to appropriate requirements file
3. Update documentation 