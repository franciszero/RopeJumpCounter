# Machine Learning Module (ML Module)

This module contains all machine learning related functionality for the RopeJumpCounter project, migrated from the original `PoseDetection` module.

## 📁 Directory Structure

```
src/ml/
├── data/                   # Data processing
│   ├── labeling/          # Data annotation tools
│   │   ├── main_gui.py           # Main annotation interface
│   │   ├── label_helper_gui.py   # Annotation assistant GUI
│   │   └── verify_labels.py      # Label verification tool
│   ├── builders/          # Data building tools
│   │   ├── feature_mode.py       # Feature mode definitions
│   │   └── builder.py            # Dataset builder
│   └── features/          # Feature extraction
│       └── features.py           # Feature pipeline
├── models/                # Model definitions
│   ├── BaseModel.py              # Base model class
│   ├── CNN.py                    # CNN model series
│   ├── TCN.py                    # TCN model
│   ├── ResNET1D.py              # ResNet1D model
│   └── ...                      # Other models
├── training/              # Training related
│   └── model_training.py         # Model training script
└── visualization/         # Visualization validation
    └── model_visualize.py        # Model visualization tool
```

## 🔧 Functional Modules

### 1. Data Annotation (`data/labeling/`)
- **Main Interface** (`main_gui.py`): Manage annotation tasks for multiple videos
- **Annotation Assistant** (`label_helper_gui.py`): Graphical annotation tool
- **Label Verification** (`verify_labels.py`): Verify annotation quality

### 2. Data Building (`data/builders/`)
- **Feature Mode** (`feature_mode.py`): Define feature extraction modes
- **Dataset Building** (`builder.py`): Generate training data from videos and labels

### 3. Feature Extraction (`data/features/`)
- **Feature Pipeline** (`features.py`): Unified feature extraction process

### 4. Model Definitions (`models/`)
- Contains various deep learning model architectures
- Supports CNN, TCN, ResNet, Transformer, and more

### 5. Model Training (`training/`)
- **Training Script** (`model_training.py`): Batch training for multiple models

### 6. Visualization Validation (`visualization/`)
- **Model Visualization** (`model_visualize.py`): Real-time prediction visualization

## 🚀 Usage

### Data Annotation
```bash
cd src/ml/data/labeling
python main_gui.py --workdir /path/to/videos
```

### Dataset Building
```bash
cd src/ml/data/builders
python builder.py --videos_dir /path/to/videos --labels_dir /path/to/labels
```

### Model Training
```bash
cd src/ml/training
python model_training.py
```

### Model Visualization
```bash
cd src/ml/visualization
python model_visualize.py --model best_model.keras --video test_video.mp4
```

## 📝 Migration Notes

This module was migrated from the original `PoseDetection` directory, with the following main changes:

1. **Directory Reorganization**: Code structure reorganized by functionality
2. **Import Path Updates**: All import paths updated to new module paths
3. **Modular Design**: Clearer separation of responsibilities and module boundaries

## 🔗 Dependencies

- `utils/`: Common utility classes (pose estimation, video stabilization, etc.)
- `capture/`: Video capture module
- External dependencies: TensorFlow, OpenCV, MediaPipe, etc. 