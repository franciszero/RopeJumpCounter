# RopeJumpCounter Sequence Diagrams

## Application Startup Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant Main as main.py
    participant Config as AppConfig
    participant Logger as Logger
    participant GPU as GPU Setup
    participant Predictor as VideoPredictor
    participant GUI as PlayerGUI
    
    User->>Main: Run python main.py
    Main->>Config: Load configuration
    Config-->>Main: Return config object
    
    Main->>Logger: Initialize logging
    Logger-->>Main: Logging system ready
    
    Main->>GPU: Setup GPU acceleration
    GPU-->>Main: GPU configuration complete
    
    Main->>Predictor: Load model
    Predictor-->>Main: Model loading complete
    
    Main->>GUI: Launch GUI interface
    GUI-->>User: Display main interface
```

## Real-time Jump Detection Sequence

```mermaid
sequenceDiagram
    participant GUI as GUI Interface
    participant Capture as Video Capture
    participant Stabilizer as Video Stabilization
    participant Vision as Computer Vision
    participant Predictor as Model Prediction
    participant Counter as Jump Counter
    participant Display as Display Update
    
    loop For each frame
        GUI->>Capture: Get video frame
        Capture-->>GUI: Return raw frame
        
        GUI->>Stabilizer: Video stabilization
        Stabilizer-->>GUI: Return stabilized frame
        
        GUI->>Vision: Pose estimation
        Vision->>Vision: MediaPipe processing
        Vision-->>GUI: Return keypoints
        
        GUI->>Predictor: Feature extraction and prediction
        Predictor->>Predictor: Sliding window inference
        Predictor-->>GUI: Return jump probability
        
        GUI->>Counter: State machine processing
        Counter->>Counter: Pattern matching
        Counter-->>GUI: Return jump count
        
        GUI->>Display: Update display
        Display-->>GUI: Interface refresh
    end
```

## Model Training Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant CLI as run.py
    participant Training as Model Training
    participant Data as Data Loading
    participant Model as Model Definition
    participant Trainer as Trainer
    
    User->>CLI: python run.py train
    CLI->>Training: Start training process
    
    Training->>Data: Load training data
    Data->>Data: Data preprocessing
    Data-->>Training: Return data batches
    
    Training->>Model: Create model
    Model-->>Training: Return model instance
    
    loop Training epochs
        Training->>Trainer: Start training
        Trainer->>Model: Forward pass
        Model-->>Trainer: Return predictions
        Trainer->>Trainer: Calculate loss
        Trainer->>Model: Backward pass
        Model-->>Trainer: Update weights
        Trainer-->>Training: Return training metrics
    end
    
    Training->>Training: Save model
    Training-->>CLI: Training complete
    CLI-->>User: Display results
```

## Data Annotation Sequence

```mermaid
sequenceDiagram
    participant User as Annotator
    participant GUI as Annotation Interface
    participant Video as Video Player
    participant Labels as Label Management
    participant Storage as Data Storage
    
    User->>GUI: Open annotation tool
    GUI->>Video: Load video file
    Video-->>GUI: Return video information
    
    loop Frame-by-frame annotation
        GUI->>Video: Play/pause
        Video-->>GUI: Current frame
        
        User->>GUI: Mark jump events
        GUI->>Labels: Record labels
        Labels-->>GUI: Confirm labels
        
        GUI->>Storage: Save annotation data
        Storage-->>GUI: Save confirmation
    end
    
    User->>GUI: Complete annotation
    GUI->>Storage: Export annotation file
    Storage-->>User: Annotation file
```

## Error Handling Sequence

```mermaid
sequenceDiagram
    participant App as Application
    participant Predictor as Predictor
    participant Model as Model
    participant Logger as Logging System
    participant User as User
    
    App->>Predictor: Model prediction
    Predictor->>Model: Inference request
    
    alt Model loading failure
        Model-->>Predictor: ModelError
        Predictor->>Logger: Log error
        Predictor-->>App: Throw exception
        App->>Logger: Log application error
        App-->>User: Display error message
    else Inference failure
        Model-->>Predictor: Inference exception
        Predictor->>Logger: Log inference error
        Predictor-->>App: Return default value
        App->>App: Continue processing
    else Normal case
        Model-->>Predictor: Prediction result
        Predictor-->>App: Return result
    end
```

## Performance Monitoring Sequence

```mermaid
sequenceDiagram
    participant GUI as GUI Interface
    participant Perf as Performance Monitor
    participant System as System Resources
    participant Logger as Logging System
    
    loop Performance monitoring
        GUI->>Perf: Start frame processing
        Perf->>System: Get system resources
        System-->>Perf: CPU/GPU/Memory usage
        
        GUI->>Perf: Frame processing complete
        Perf->>Perf: Calculate FPS
        Perf->>Perf: Calculate latency
        
        Perf->>Logger: Log performance metrics
        Perf-->>GUI: Return performance data
        
        GUI->>GUI: Update performance display
    end
```

## Configuration Management Sequence

```mermaid
sequenceDiagram
    participant App as Application
    participant Config as Configuration Manager
    participant File as Configuration File
    participant Env as Environment Variables
    participant Default as Default Configuration
    
    App->>Config: Load configuration
    
    alt Configuration file exists
        Config->>File: Read config.yaml
        File-->>Config: Return config data
        Config->>Config: Parse YAML
        Config-->>App: Return config object
    else Environment variables exist
        Config->>Env: Read environment variables
        Env-->>Config: Return environment config
        Config->>Config: Merge configuration
        Config-->>App: Return config object
    else Use default values
        Config->>Default: Get default configuration
        Default-->>Config: Return default values
        Config-->>App: Return config object
    end
```

## Model Inference Pipeline Sequence

```mermaid
sequenceDiagram
    participant Frame as Video Frame
    participant Pose as Pose Estimation
    participant Features as Feature Extraction
    participant Window as Sliding Window
    participant Model as Neural Network
    participant Output as Prediction Output
    
    Frame->>Pose: Input video frame
    Pose->>Pose: MediaPipe pose detection
    Pose-->>Features: Return pose landmarks
    
    Features->>Features: Extract spatial features
    Features->>Features: Calculate temporal features
    Features-->>Window: Return feature vector
    
    Window->>Window: Add to sliding window
    Window->>Window: Check window fullness
    
    alt Window is full
        Window->>Model: Prepare input tensor
        Model->>Model: Forward pass
        Model-->>Output: Return prediction
        Output-->>Window: Jump probability
    else Window not full
        Window-->>Output: Return 0.0 (warmup)
    end
```

## Video Processing Pipeline Sequence

```mermaid
sequenceDiagram
    participant Camera as Camera Device
    participant Capture as Video Capture
    participant Stabilizer as Video Stabilizer
    participant Processor as Frame Processor
    participant Display as Display System
    
    loop Real-time processing
        Camera->>Capture: Raw video stream
        Capture->>Capture: Frame extraction
        Capture-->>Stabilizer: Raw frame
        
        Stabilizer->>Stabilizer: Motion compensation
        Stabilizer->>Stabilizer: Noise reduction
        Stabilizer-->>Processor: Stabilized frame
        
        Processor->>Processor: Pose estimation
        Processor->>Processor: Feature extraction
        Processor->>Processor: Model inference
        Processor-->>Display: Processed frame with overlay
        
        Display->>Display: Update GUI
        Display-->>Camera: Display feedback
    end
```

## Usage Instructions

### 1. **View Sequence Diagrams**
- View directly in GitHub
- Use Mermaid Live Editor for editing
- Export as image format

### 2. **Update Sequence Diagrams**
When code logic changes, remember to synchronize the corresponding sequence diagrams.

### 3. **Add New Sequence Diagrams**
For new functional modules, you can add sequence diagrams following the same format.

### 4. **Best Practices**
- Keep diagrams concise and clear
- Highlight key interaction points
- Include error handling flows
- Annotate important timing relationships

### 5. **Integration with Architecture**
These sequence diagrams complement the architecture diagrams by showing:
- **Dynamic behavior** of system components
- **Temporal relationships** between operations
- **Error handling** scenarios
- **Performance bottlenecks** identification

### 6. **Maintenance Guidelines**
- **Version control**: Include sequence diagrams in version control
- **Code synchronization**: Update diagrams when code changes
- **Review process**: Include diagram review in code reviews
- **Documentation**: Reference sequence diagrams in API documentation 