# RopeJumpCounter Project Structure

## 📁 Directory Structure

```
RopeJumpCounter/
├── 📁 src/                          # Source code directory
│   ├── 📁 apps/                     # Application entry points
│   │   ├── main.py                  # Main application (configured version)
│   │   ├── main_v2.py               # 🆕 New architecture application
│   │   ├── main_0_5.py             # Legacy main application
│   │   └── app.py                   # Alternative entry point
│   │
│   ├── 📁 core/                     # Core business logic
│   │   ├── jump_counter.py          # Jump rope counter logic
│   │   ├── exceptions.py            # Exception definitions
│   │   ├── container.py             # 🆕 Dependency injection container
│   │   ├── event_bus.py             # 🆕 Event bus system
│   │   ├── plugin_manager.py        # 🆕 Plugin management system
│   │   ├── pyav_capture.py          # PyAV video capture
│   │   └── gst_capture.py           # GStreamer capture
│   │
│   ├── 📁 ml/                       # Machine learning modules
│   │   ├── 📁 inference/            # Model inference
│   │   │   ├── __init__.py
│   │   │   └── video_predictor.py   # Video predictor (moved from core)
│   │   │
│   │   ├── 📁 models/               # Model definitions
│   │   │   ├── CNN.py               # CNN architectures
│   │   │   ├── LSTM_Attention.py    # LSTM with attention
│   │   │   └── ModelParams/         # Model parameters
│   │   │
│   │   ├── 📁 training/             # Model training
│   │   │   └── model_training.py    # Training pipeline
│   │   │
│   │   ├── 📁 data/                 # Data processing
│   │   │   ├── 📁 labeling/         # Data annotation tools
│   │   │   ├── 📁 builders/         # Data building tools
│   │   │   └── 📁 features/         # Feature extraction
│   │   │
│   │   └── 📁 visualization/        # Visualization validation
│   │       └── model_visualize.py   # Model visualization
│   │
│   ├── 📁 interface/                # User interface
│   │   └── gui.py                   # Graphical user interface
│   │
│   ├── 📁 config/                   # Configuration management
│   │   └── settings.py              # Application configuration
│   │
│   └── 📁 utils/                    # Utility modules (reorganized)
│       ├── 📁 vision/               # Computer vision utilities
│       │   ├── __init__.py
│       │   ├── vision.py            # Pose estimation
│       │   └── VideoStabilizer.py   # Video stabilization
│       │
│       ├── 📁 performance/          # Performance monitoring
│       │   ├── __init__.py
│       │   └── Perf.py              # Performance statistics
│       │
│       └── 📁 common/               # General utilities
│           ├── __init__.py
│           ├── FrameSample.py       # Feature extraction
│           ├── Differentiator.py    # Temporal differences
│           └── filter.py            # Data filtering
│
├── 📁 docs/                         # Documentation
│   ├── README.md                    # Documentation index
│   ├── ARCHITECTURE.md              # System architecture (中文)
│   ├── ARCHITECTURE_EN.md           # System architecture (English)
│   ├── SEQUENCE_DIAGRAM.md          # Sequence diagrams (中文)
│   ├── SEQUENCE_DIAGRAM_EN.md       # Sequence diagrams (English)
│   ├── USER_GUIDE.md                # User manual
│   ├── API.md                       # API documentation
│   ├── CODE_REVIEW_REPORT.md        # Code quality analysis
│   ├── FINAL_DOCUMENTATION_AUDIT.md # Documentation audit
│   ├── 📁 architecture/             # 🆕 Architecture Decision Records
│   │   ├── ADR-001-dependency-injection.md
│   │   └── ADR-002-event-bus.md
│   ├── 📁 examples/                 # Code examples
│   ├── 📁 images/                   # Screenshots and diagrams
│   └── 📁 tutorials/                # Step-by-step tutorials
│
├── 📁 plugins/                      # 🆕 Plugin directory
│   └── performance_monitor.py       # Example performance monitoring plugin
│
├── 📁 data/                         # Data directory
├── 📁 model_files/                  # Model files
├── 📁 logs/                         # Log files
├── 📁 archive/                      # Historical versions
├── 📁 scripts/                      # Utility scripts
│
├── 📄 run.py                        # 🎯 PRIMARY ENTRY POINT
├── 📄 main.py                       # ⚠️ DEPRECATED (use run.py)
├── 📄 app.py                        # ⚠️ DEPRECATED (use run.py)
├── 📄 main_0.5.py                   # ⚠️ DEPRECATED (use run.py)
│
├── 📄 requirements-core.txt         # Core dependencies
├── 📄 requirements-gui.txt          # GUI dependencies (optional)
├── 📄 requirements-dev.txt          # Development dependencies
├── 📄 requirements.txt              # ⚠️ DEPRECATED (use requirements-core.txt)
├── 📄 requirements-minimal.txt      # ⚠️ DEPRECATED (use requirements-core.txt)
│
├── 📄 config.yaml.example           # Configuration template
├── 📄 README.md                     # Project overview
├── 📄 INSTALL.md                    # Installation guide
├── 📄 CONTRIBUTING.md               # Contributing guidelines
├── 📄 LICENSE                       # License file
└── 📄 .gitignore                    # Git ignore rules
```

## 🔄 Recent Structural Changes

### ✅ **Completed Improvements**

1. **Unified Entry Point**
   - `run.py` is now the **primary entry point**
   - Other entry points (`main.py`, `app.py`, `main_0.5.py`) are deprecated
   - Added deprecation warnings for backward compatibility

2. **Dependency Management**
   - `requirements-core.txt` - Essential dependencies
   - `requirements-gui.txt` - Optional GUI dependencies
   - `requirements-dev.txt` - Development and testing tools
   - Old requirements files are deprecated

3. **Reorganized Source Structure**
   - **ML Inference**: Moved `video_predictor.py` to `src/ml/inference/`
   - **Video Capture**: Moved capture modules to `src/core/`
   - **Utilities**: Reorganized into specialized subdirectories:
     - `src/utils/vision/` - Computer vision utilities
     - `src/utils/performance/` - Performance monitoring
     - `src/utils/common/` - General utilities

4. **Enhanced Documentation**
   - Bilingual architecture diagrams (中文/English)
   - Complete sequence diagrams
   - Documentation index and maintenance guidelines

### 🆕 **New Architecture Features (v2.0)**

5. **Dependency Injection Container**
   - `src/core/container.py` - Centralized service management
   - `src/core/event_bus.py` - Decoupled component communication
   - `src/core/plugin_manager.py` - Modular extension system

6. **Event-Driven Architecture**
   - Event publishing and subscription
   - Asynchronous event processing
   - Event history and debugging support

7. **Plugin System**
   - `plugins/` directory for modular extensions
   - Dynamic plugin loading and management
   - Example performance monitoring plugin

8. **Architecture Decision Records**
   - `docs/architecture/` - ADR documentation
   - Decision tracking and rationale
   - Migration strategies

## 🎯 **Usage Guidelines**

### **For Users**
```bash
# ✅ Recommended way (original architecture)
python run.py realtime

# 🆕 New architecture (v2.0)
python run.py realtime-v2

# ⚠️ Deprecated (will show warning)
python main.py
python app.py
```

### **For Developers**
```bash
# Install core dependencies
pip install -r requirements-core.txt

# Install GUI dependencies (optional)
pip install -r requirements-gui.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### **For Contributors**
1. Follow the new directory structure
2. Use `run.py` as the entry point
3. Update imports when moving files
4. Maintain bilingual documentation
5. Use dependency injection for new components
6. Publish events for component communication

## 📋 **Migration Guide**

### **Import Updates Required**
```python
# Old imports (deprecated)
from src.ml.inference.video_predictor import VideoPredictor
from src.utils.vision import *
from src.utils.performance.Perf import *

# New imports (recommended)
from src.ml.inference.video_predictor import VideoPredictor
from src.utils.vision import *
from src.utils.performance.Perf import *
```

### **Entry Point Migration**
```bash
# Old way (deprecated)
python main.py
python app.py

# New way (recommended)
python run.py realtime
python run.py realtime-v2  # New architecture
python run.py legacy
```

### **New Architecture Migration**
```python
# Old way (direct instantiation)
predictor = VideoPredictor(model_path)
gui = PlayerGUI(predictor, width, height, fps)

# New way (dependency injection)
container = get_container()
predictor = container.get_service('predictor')
gui = container.get_service('gui')

# Event-driven communication
event_bus = get_event_bus()
event_bus.publish(EventType.JUMP_DETECTED, {"count": 5}, "jump_counter")
```

## 🔧 **Maintenance Guidelines**

### **Adding New Features**
1. **Core Logic**: Add to `src/core/`
2. **ML Models**: Add to `src/ml/models/`
3. **ML Training**: Add to `src/ml/training/`
4. **ML Inference**: Add to `src/ml/inference/`
5. **Vision Utils**: Add to `src/utils/vision/`
6. **Performance Utils**: Add to `src/utils/performance/`
7. **Common Utils**: Add to `src/utils/common/`
8. **Plugins**: Add to `plugins/` directory

### **Documentation Updates**
- Update both Chinese and English versions
- Keep architecture diagrams synchronized
- Maintain sequence diagrams for new workflows
- Update this structure document
- Add ADRs for major architectural decisions

### **Dependency Management**
- Core functionality: `requirements-core.txt`
- GUI features: `requirements-gui.txt`
- Development tools: `requirements-dev.txt`
- Test all combinations before releasing

## 🚀 **Future Improvements**

### **Planned Enhancements**
1. **Microservices Architecture**: Move toward service-oriented design
2. **Cloud Deployment**: Support for cloud platforms
3. **API Versioning**: Backward compatibility management
4. **Advanced Monitoring**: Comprehensive metrics and alerting
5. **Machine Learning Pipeline**: Automated model training and deployment

### **Architecture Evolution**
- Implement event sourcing for audit trails
- Add CQRS pattern for read/write separation
- Support for distributed event processing
- Integration with external monitoring systems

---

**Last Updated**: Current Session  
**Structure Version**: 2.0  
**Maintained by**: RopeJumpCounter Development Team 