# RopeJumpCounter System Architecture

## Overall Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        GUI[PlayerGUI<br/>Graphical User Interface]
        CLI[Command Line Interface<br/>run.py]
    end
    
    subgraph "Application Layer"
        APP[Main Application<br/>main.py]
        CONFIG[Configuration Management<br/>AppConfig]
        LOGGER[Logging System<br/>setup_logger]
    end
    
    subgraph "Core Business Layer"
        PREDICTOR[Video Predictor<br/>VideoPredictor]
        COUNTER[Jump Counter<br/>JumpCounter]
        EXCEPTIONS[Exception Handling<br/>AppError]
    end
    
    subgraph "Machine Learning Layer"
        MODELS[Model Definitions<br/>CNN/TCN/ResNet]
        TRAINING[Model Training<br/>model_training]
        FEATURES[Feature Extraction<br/>FrameSample]
        VIZ[Visualization<br/>model_visualize]
    end
    
    subgraph "Data Layer"
        CAPTURE[Video Capture<br/>pyav_capture]
        STABILIZER[Video Stabilization<br/>VideoStabilizer]
        LABELING[Data Annotation<br/>main_gui]
    end
    
    subgraph "Utility Layer"
        VISION[Computer Vision<br/>vision.py]
        PERF[Performance Monitoring<br/>Perf.py]
        UTILS[General Utilities<br/>utils/]
    end
    
    subgraph "External Dependencies"
        TF[TensorFlow<br/>Deep Learning Framework]
        MP[MediaPipe<br/>Pose Estimation]
        CV[OpenCV<br/>Image Processing]
        YAML[PyYAML<br/>Configuration Parser]
    end
    
    %% Connection relationships
    GUI --> APP
    CLI --> APP
    APP --> CONFIG
    APP --> LOGGER
    APP --> PREDICTOR
    APP --> COUNTER
    
    PREDICTOR --> MODELS
    PREDICTOR --> FEATURES
    COUNTER --> PREDICTOR
    
    FEATURES --> VISION
    VISION --> MP
    VISION --> CV
    
    CAPTURE --> STABILIZER
    STABILIZER --> VISION
    
    TRAINING --> MODELS
    TRAINING --> FEATURES
    VIZ --> MODELS
    
    LABELING --> CAPTURE
    
    MODELS --> TF
    PERF --> UTILS
    
    CONFIG --> YAML
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Input"
        CAM[Camera]
        VIDEO[Video File]
    end
    
    subgraph "Processing Pipeline"
        CAP[Video Capture]
        STAB[Video Stabilization]
        POSE[Pose Estimation]
        FEAT[Feature Extraction]
        PRED[Model Prediction]
        COUNT[Jump Counting]
    end
    
    subgraph "Output"
        DISPLAY[Real-time Display]
        SAVE[Video Recording]
        LOG[Logging]
    end
    
    CAM --> CAP
    VIDEO --> CAP
    CAP --> STAB
    STAB --> POSE
    POSE --> FEAT
    FEAT --> PRED
    PRED --> COUNT
    COUNT --> DISPLAY
    COUNT --> SAVE
    COUNT --> LOG
```

## Module Dependency Relationships

```mermaid
graph TD
    subgraph "Core Modules"
        A[main.py] --> B[AppConfig]
        A --> C[VideoPredictor]
        A --> D[PlayerGUI]
        A --> E[setup_logger]
    end
    
    subgraph "ML Modules"
        C --> F[models/]
        C --> G[features/]
        F --> H[TensorFlow]
        G --> I[MediaPipe]
    end
    
    subgraph "Utility Modules"
        D --> J[vision.py]
        D --> K[Perf.py]
        J --> I
        K --> L[psutil]
    end
    
    subgraph "Configuration Modules"
        B --> M[PyYAML]
        B --> N[Environment Variables]
    end
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Developer Machine]
        DEV --> GIT[Git Repository]
        DEV --> IDE[IDE/Editor]
    end
    
    subgraph "Build Environment"
        BUILD[CI/CD Pipeline]
        BUILD --> TEST[Automated Testing]
        BUILD --> PACKAGE[Packaging]
    end
    
    subgraph "Production Environment"
        PROD[Production Environment]
        PROD --> GPU[GPU Server]
        PROD --> CAM[Camera Device]
        PROD --> MONITOR[Monitoring System]
    end
    
    GIT --> BUILD
    BUILD --> PROD
```

## Technology Stack Architecture

```mermaid
graph LR
    subgraph "Frontend/Interface"
        GUI[PySimpleGUI]
        CLI[argparse]
    end
    
    subgraph "Backend/Core"
        PYTHON[Python 3.8+]
        TF[TensorFlow 2.8+]
        CV[OpenCV]
        MP[MediaPipe]
    end
    
    subgraph "Data Processing"
        NUMPY[numpy]
        PANDAS[pandas]
        YAML[PyYAML]
    end
    
    subgraph "System Integration"
        OS[Operating System]
        GPU[GPU Drivers]
        CAM[Camera Drivers]
    end
    
    GUI --> PYTHON
    CLI --> PYTHON
    PYTHON --> TF
    PYTHON --> CV
    PYTHON --> MP
    PYTHON --> NUMPY
    PYTHON --> PANDAS
    PYTHON --> YAML
    PYTHON --> OS
    TF --> GPU
    CV --> CAM
```

## How to Use These Architecture Diagrams

### 1. **View in GitHub**
GitHub natively supports Mermaid diagrams, displaying them directly in Markdown.

### 2. **Reference in Documentation**
```markdown
## System Overview
Please refer to the [Architecture Diagram](./ARCHITECTURE_EN.md#overall-architecture) for system structure.
```

### 3. **Export as Images**
Use Mermaid CLI tool to export as PNG/SVG:
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export images
mmdc -i architecture_en.md -o architecture_en.png
```

### 4. **Online Editors**
- [Mermaid Live Editor](https://mermaid.live/)
- [Draw.io](https://draw.io/) (supports Mermaid)

## Architecture Diagram Best Practices

### 1. **Clear Layering**
- User Interface Layer
- Application Layer
- Business Logic Layer
- Data Access Layer

### 2. **Color Coding**
```mermaid
graph TB
    subgraph "User Layer" 
        style GUI fill:#e1f5fe
        GUI[GUI Components]
    end
    
    subgraph "Business Layer"
        style CORE fill:#f3e5f5
        CORE[Core Business]
    end
    
    subgraph "Data Layer"
        style DATA fill:#e8f5e8
        DATA[Data Storage]
    end
```

### 3. **Keep Updated**
- Synchronize architecture diagrams when code changes
- Regularly review diagram accuracy
- Version control architecture diagrams

This provides you with a complete system architecture documentation in English! Which type of diagram do you find most useful? I can help you further optimize or add other types of architecture diagrams. 