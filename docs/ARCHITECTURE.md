# RopeJumpCounter System Architecture

## Overall Architecture

```mermaid
graph TB
    subgraph "Entry Point Layer"
        RUN[run.py<br/>Main Entry Point]
    end
    
    subgraph "Application Layer"
        MAIN[main.py<br/>Original App]
        MAINV2[main_v2.py<br/>New Architecture]
        LEGACY[main_0_5.py<br/>Legacy App]
    end
    
    subgraph "Core Architecture (v2.0)"
        CONTAINER[Container<br/>Dependency Injection]
        EVENTBUS[EventBus<br/>Event System]
        PLUGIN[PluginManager<br/>Plugin System]
    end
    
    subgraph "Interface Layer"
        GUI[PlayerGUI<br/>Main Interface]
    end
    
    subgraph "Core Business Logic"
        PREDICTOR[VideoPredictor<br/>Model Inference]
        COUNTER[JumpCounter<br/>State Machine]
        FEATURE[FeaturePipeline<br/>Feature Extraction]
    end
    
    subgraph "Data Processing"
        CAPTURE[PyAVCapture<br/>Video Capture]
        STABILIZER[VideoStabilizer<br/>Video Stabilization]
        POSE[PoseEstimator<br/>MediaPipe Integration]
    end
    
    subgraph "Machine Learning"
        MODELS[Model Files<br/>Trained Models]
        TRAINING[model_training.py<br/>Training Script]
        VIZ[model_visualize.py<br/>Visualization]
    end
    
    subgraph "Data Management"
        LABELING[main_gui.py<br/>Annotation Tool]
        BUILDER[builder.py<br/>Dataset Builder]
    end
    
    subgraph "Configuration & Utilities"
        CONFIG[AppConfig<br/>Configuration]
        LOGGER[setup_logger<br/>Logging]
        PERF[PerfStats<br/>Performance]
    end
    
    %% Entry point connections
    RUN --> MAIN
    RUN --> MAINV2
    RUN --> LEGACY
    RUN --> TRAINING
    RUN --> LABELING
    RUN --> VIZ
    RUN --> BUILDER
    
    %% v2.0 architecture connections
    MAINV2 --> CONTAINER
    MAINV2 --> EVENTBUS
    MAINV2 --> PLUGIN
    
    %% Core business logic
    GUI --> PREDICTOR
    GUI --> COUNTER
    GUI --> FEATURE
    GUI --> CAPTURE
    
    %% Data processing pipeline
    FEATURE --> STABILIZER
    FEATURE --> POSE
    CAPTURE --> STABILIZER
    
    %% ML connections
    PREDICTOR --> MODELS
    TRAINING --> MODELS
    
    %% Configuration
    CONTAINER --> CONFIG
    CONTAINER --> LOGGER
    GUI --> PERF
```

## Application Modes Architecture

```mermaid
graph LR
    subgraph "run.py Entry Point"
        RUN[run.py]
    end
    
    subgraph "Real-time Modes"
        REALTIME[realtime<br/>main.py]
        REALTIMEV2[realtime-v2<br/>main_v2.py]
        LEGACY[legacy<br/>main_0_5.py]
    end
    
    subgraph "ML Modes"
        TRAIN[train<br/>model_training.py]
        LABEL[label<br/>main_gui.py]
        VIZ[visualize<br/>model_visualize.py]
        BUILD[build<br/>builder.py]
    end
    
    RUN --> REALTIME
    RUN --> REALTIMEV2
    RUN --> LEGACY
    RUN --> TRAIN
    RUN --> LABEL
    RUN --> VIZ
    RUN --> BUILD
```

## Real-time Processing Pipeline

```mermaid
flowchart TD
    subgraph "Video Input"
        CAM[Camera Device]
        VIDEO[Video File]
    end
    
    subgraph "Capture Layer"
        CAP[PyAVCapture<br/>Frame Capture]
    end
    
    subgraph "Processing Pipeline"
        STAB[VideoStabilizer<br/>Motion Compensation]
        POSE[PoseEstimator<br/>MediaPipe Pose]
        FEAT[FeaturePipeline<br/>Feature Extraction]
        PRED[VideoPredictor<br/>Model Inference]
        COUNT[JumpCounter<br/>State Machine]
    end
    
    subgraph "Output Layer"
        GUI[PlayerGUI<br/>Display & Recording]
        LOG[Logging System]
    end
    
    CAM --> CAP
    VIDEO --> CAP
    CAP --> STAB
    STAB --> POSE
    POSE --> FEAT
    FEAT --> PRED
    PRED --> COUNT
    COUNT --> GUI
    COUNT --> LOG
```

## v2.0 Architecture Components

```mermaid
graph TB
    subgraph "Dependency Injection Container"
        CONTAINER[Container]
        STATE[AppState]
        SERVICES[Services Registry]
    end
    
    subgraph "Event Bus System"
        EVENTBUS[EventBus]
        EVENTS[Event Types]
        HANDLERS[Event Handlers]
    end
    
    subgraph "Plugin System"
        PLUGIN[PluginManager]
        PLUGINS[Loaded Plugins]
        LIFECYCLE[Plugin Lifecycle]
    end
    
    subgraph "Core Services"
        CONFIG[AppConfig]
        LOGGER[Logger]
        PREDICTOR[VideoPredictor]
        GUI[PlayerGUI]
    end
    
    CONTAINER --> STATE
    CONTAINER --> SERVICES
    CONTAINER --> CONFIG
    CONTAINER --> LOGGER
    CONTAINER --> PREDICTOR
    CONTAINER --> GUI
    
    EVENTBUS --> EVENTS
    EVENTBUS --> HANDLERS
    
    PLUGIN --> PLUGINS
    PLUGIN --> LIFECYCLE
```

## Feature Extraction Pipeline

```mermaid
flowchart LR
    subgraph "Input"
        FRAME[Raw Frame]
    end
    
    subgraph "Preprocessing"
        STAB[VideoStabilizer<br/>Stabilization]
        POSE[PoseEstimator<br/>Landmark Detection]
    end
    
    subgraph "Feature Extraction"
        RAW[Raw Features<br/>Normalized Landmarks]
        RAW_PX[Pixel Features<br/>Pixel Coordinates]
        DIFF[Temporal Features<br/>Frame Differences]
        SPATIAL[Spatial Features<br/>Distances & Angles]
        WINDOW[Windowed Features<br/>Temporal Aggregation]
    end
    
    subgraph "Output"
        FEATURES[Feature Vector]
    end
    
    FRAME --> STAB
    STAB --> POSE
    POSE --> RAW
    POSE --> RAW_PX
    POSE --> DIFF
    POSE --> SPATIAL
    POSE --> WINDOW
    RAW --> FEATURES
    RAW_PX --> FEATURES
    DIFF --> FEATURES
    SPATIAL --> FEATURES
    WINDOW --> FEATURES
```

## Jump Detection State Machine

```mermaid
stateDiagram-v2
    [*] --> Warmup: Initialize
    Warmup --> Ready: Window Full
    
    Ready --> Rising: Pattern 7 (0111)
    Ready --> NotRising: Other Patterns
    
    Rising --> Rising: Pattern 15 (1111)
    Rising --> NotRising: Other Patterns
    Rising --> CountJump: Pattern 7 (0111)
    
    NotRising --> Ready: Next Frame
    NotRising --> Rising: Pattern 7 (0111)
    
    CountJump --> Ready: Jump Counted
    
    state Rising {
        [*] --> RisingState
        RisingState --> [*]
    }
    
    state NotRising {
        [*] --> NotRisingState
        NotRisingState --> [*]
    }
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User as User
    participant Run as run.py
    participant Main as main_v2.py
    participant Container as Container
    participant EventBus as EventBus
    participant GUI as PlayerGUI
    participant Predictor as VideoPredictor
    participant Counter as JumpCounter
    
    User->>Run: python run.py realtime-v2
    Run->>Main: Import and execute
    Main->>Container: Initialize services
    Main->>EventBus: Start event bus
    Main->>GUI: Get GUI service
    GUI->>Predictor: Initialize predictor
    GUI->>Counter: Initialize counter
    
    loop Real-time Processing
        GUI->>GUI: Capture frame
        GUI->>Predictor: Predict jump probability
        Predictor->>Predictor: Sliding window inference
        Predictor-->>GUI: Return probability
        GUI->>Counter: Process prediction
        Counter->>Counter: State machine logic
        Counter-->>GUI: Return jump count
        GUI->>EventBus: Publish jump event
        EventBus->>Container: Update state
        GUI->>GUI: Update display
    end
```

## Module Dependencies

```mermaid
graph TD
    subgraph "Entry Points"
        RUN[run.py]
        MAIN[main.py]
        MAINV2[main_v2.py]
    end
    
    subgraph "Core Modules"
        CONTAINER[container.py]
        EVENTBUS[event_bus.py]
        PLUGIN[plugin_manager.py]
        COUNTER[jump_counter.py]
        CAPTURE[pyav_capture.py]
    end
    
    subgraph "Interface"
        GUI[gui.py]
    end
    
    subgraph "ML Modules"
        PREDICTOR[video_predictor.py]
        FEATURES[features.py]
        TRAINING[model_training.py]
    end
    
    subgraph "Configuration"
        CONFIG[settings.py]
        LOGGER[logging.py]
    end
    
    RUN --> MAIN
    RUN --> MAINV2
    MAIN --> CONFIG
    MAIN --> LOGGER
    MAIN --> PREDICTOR
    MAIN --> GUI
    
    MAINV2 --> CONTAINER
    MAINV2 --> EVENTBUS
    MAINV2 --> PLUGIN
    CONTAINER --> CONFIG
    CONTAINER --> LOGGER
    CONTAINER --> PREDICTOR
    CONTAINER --> GUI
    
    GUI --> PREDICTOR
    GUI --> COUNTER
    GUI --> FEATURES
    GUI --> CAPTURE
    
    PREDICTOR --> FEATURES
    FEATURES --> CAPTURE
```

## Technology Stack

```mermaid
graph TB
    subgraph "Application Framework"
        PYTHON[Python 3.8+]
        ASYNC[asyncio]
        THREADING[threading]
    end
    
    subgraph "Machine Learning"
        TENSORFLOW[TensorFlow 2.8+]
        KERAS[Keras]
        MIXED[Mixed Precision]
    end
    
    subgraph "Computer Vision"
        OPENCV[OpenCV]
        MEDIAPIPE[MediaPipe]
        POSE[Pose Estimation]
    end
    
    subgraph "Video Processing"
        PYAV[PyAV]
        STABILIZER[Video Stabilization]
        CAPTURE[Frame Capture]
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
    
    PYTHON --> TENSORFLOW
    PYTHON --> OPENCV
    PYTHON --> MEDIAPIPE
    PYTHON --> PYAV
    PYTHON --> NUMPY
    PYTHON --> PANDAS
    PYTHON --> YAML
    
    TENSORFLOW --> KERAS
    TENSORFLOW --> MIXED
    TENSORFLOW --> GPU
    
    MEDIAPIPE --> POSE
    OPENCV --> STABILIZER
    PYAV --> CAPTURE
    
    OPENCV --> CAM
    TENSORFLOW --> OS
```

## How to Use These Architecture Diagrams

### 1. **View in GitHub**
GitHub natively supports Mermaid diagrams, displaying them directly in Markdown.

### 2. **Reference in Documentation**
```markdown
## System Overview
Please refer to the [Architecture Diagram](ARCHITECTURE.md#overall-architecture) for system structure.
```

### 3. **Export as Images**
Use Mermaid CLI tool to export as PNG/SVG:
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export images
mmdc -i ARCHITECTURE.md -o architecture.png
```

### 4. **Online Editors**
- [Mermaid Live Editor](https://mermaid.live/)
- [Draw.io](https://draw.io/) (supports Mermaid)

## Architecture Diagram Best Practices

### 1. **Clear Layering**
- Entry Point Layer
- Application Layer
- Core Architecture Layer
- Interface Layer
- Business Logic Layer
- Data Processing Layer

### 2. **Color Coding**
```mermaid
graph TB
    subgraph "Entry Layer" 
        style RUN fill:#e1f5fe
        RUN[Entry Points]
    end
    
    subgraph "Application Layer"
        style MAIN fill:#f3e5f5
        MAIN[Applications]
    end
    
    subgraph "Core Layer"
        style CONTAINER fill:#e8f5e8
        CONTAINER[Core Components]
    end
```

### 3. **Keep Updated**
- Synchronize architecture diagrams when code changes
- Regularly review diagram accuracy
- Version control architecture diagrams
