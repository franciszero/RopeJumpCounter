# RopeJumpCounter Sequence Diagrams

## Application Startup Sequence (v2.0 Architecture)

```mermaid
sequenceDiagram
    participant User as User
    participant Run as run.py
    participant MainV2 as main_v2.py
    participant Container as Container
    participant EventBus as EventBus
    participant PluginManager as PluginManager
    participant AppConfig as AppConfig
    participant Logger as Logger
    participant Predictor as VideoPredictor
    participant GUI as PlayerGUI
    
    User->>Run: python run.py realtime-v2
    Run->>MainV2: Import and execute main()
    MainV2->>MainV2: setup_gpu() - Mixed precision
    MainV2->>MainV2: main_async()
    
    MainV2->>Container: get_container()
    MainV2->>EventBus: get_event_bus()
    MainV2->>PluginManager: get_plugin_manager()
    
    MainV2->>AppConfig: AppConfig.load()
    AppConfig-->>MainV2: Return config object
    MainV2->>Container: register_config(config)
    
    MainV2->>Container: initialize_services()
    Container->>Logger: setup_logger()
    Container->>Predictor: VideoPredictor(model_path)
    Container->>GUI: PlayerGUI(predictor, width, height, fps)
    Container-->>MainV2: Services initialized
    
    MainV2->>MainV2: setup_event_handlers(container)
    MainV2->>EventBus: subscribe(JUMP_DETECTED, on_jump_detected)
    MainV2->>EventBus: subscribe(ERROR_OCCURRED, on_error_occurred)
    MainV2->>EventBus: subscribe(PERFORMANCE_UPDATE, on_performance_update)
    
    MainV2->>EventBus: start()
    MainV2->>PluginManager: load_all_plugins()
    MainV2->>PluginManager: enable_plugin(plugin_name)
    
    MainV2->>EventBus: publish(APPLICATION_START, {}, "main")
    MainV2->>Container: get_service('gui')
    MainV2->>Container: get_state()
    MainV2->>GUI: gui.run()
    
    Note over GUI: Main processing loop starts
```

## Application Startup Sequence (Original Architecture)

```mermaid
sequenceDiagram
    participant User as User
    participant Run as run.py
    participant Main as main.py
    participant AppConfig as AppConfig
    participant Logger as Logger
    participant Predictor as VideoPredictor
    participant GUI as PlayerGUI
    
    User->>Run: python run.py realtime
    Run->>Main: Import and execute main()
    Main->>Main: setup_gpu() - Mixed precision
    
    Main->>AppConfig: AppConfig.load()
    AppConfig-->>Main: Return config object
    
    Main->>Logger: setup_logger("RopeJump", log_dir)
    Logger-->>Main: Logger instance
    
    Main->>Predictor: VideoPredictor(model_path)
    Predictor->>Predictor: Load Keras model
    Predictor->>Predictor: Setup sliding window buffer
    Predictor-->>Main: Predictor instance
    
    Main->>GUI: PlayerGUI(predictor, width, height, fps, save_path)
    GUI->>GUI: Initialize PyAVCapture
    GUI->>GUI: Setup PerfStats
    GUI->>GUI: Initialize JumpCounter
    GUI->>GUI: Setup video recording (optional)
    GUI-->>Main: GUI instance
    
    Main->>GUI: gui.run()
    
    Note over GUI: Main processing loop starts
```

## Real-time Frame Processing Sequence

```mermaid
sequenceDiagram
    participant GUI as PlayerGUI
    participant Capture as PyAVCapture
    participant FeaturePipe as FeaturePipeline
    participant Stabilizer as VideoStabilizer
    participant PoseEst as PoseEstimator
    participant Predictor as VideoPredictor
    participant Counter as JumpCounter
    participant EventBus as EventBus
    participant Display as OpenCV Display
    
    loop For each frame
        GUI->>Capture: read()
        Capture-->>GUI: Return (ret, frame, _)
        
        alt Frame capture successful
            GUI->>FeaturePipe: process_frame(frame, frame_idx)
            FeaturePipe->>FeaturePipe: fs.raw_frame = frame
            FeaturePipe->>FeaturePipe: fs.init_current_frame(frame_idx)
            
            FeaturePipe->>Stabilizer: stabilize(fs.raw_frame)
            Stabilizer-->>FeaturePipe: Return stabilized frame
        
            FeaturePipe->>PoseEst: get_pose_landmarks(stable)
            PoseEst-->>FeaturePipe: Return landmarks
            
            FeaturePipe->>FeaturePipe: Extract features based on mode
            Note over FeaturePipe: RAW, RAW_PX, DIFF, SPATIAL, WINDOW
            
            FeaturePipe-->>GUI: Feature extraction complete
            
            GUI->>GUI: Extract feature vector from pipe.fs.rec
            GUI->>Predictor: predict(feat_vec)
            
            Predictor->>Predictor: buffer.append(feature_vector)
            alt Window is full
                Predictor->>Predictor: np.stack(buffer, axis=0)
                Predictor->>Predictor: model(input_tensor, training=False)
                Predictor-->>GUI: Return probability
            else Window not full (warmup)
                Predictor-->>GUI: Return 0.0
            end
            
            GUI->>Counter: process_prediction(prob, threshold)
            Counter->>Counter: Convert prob to binary (prob > threshold)
            Counter->>Counter: Update 4-bit sliding window
            Counter->>Counter: Check binary patterns (7, 15)
            alt Pattern 7 (0111) - Rising edge
                Counter->>Counter: jump_cnt += 1
                Counter-->>GUI: Return (is_on_rising=True, jump_cnt)
            else Pattern 15 (1111) - Sustained rising
                Counter-->>GUI: Return (is_on_rising=True, jump_cnt)
            else Other patterns
                Counter-->>GUI: Return (is_on_rising=False, jump_cnt)
            end
            
            GUI->>GUI: _overlay(frame, jump_cnt, prob, is_on_rising)
            GUI->>Display: imshow("JumpRope RealTime", frame_vis)
            
            alt Video recording enabled
                GUI->>GUI: writer.write(frame)
            end
            
            GUI->>GUI: Update performance statistics
            GUI->>EventBus: Publish performance metrics (optional)
            
        else Frame capture failed
            GUI->>GUI: Log warning, increment error_count
        end
        
        GUI->>GUI: Check for exit command (cv2.waitKey)
        alt User pressed 'q'
            GUI->>GUI: Break loop
        end
    end
```

## Feature Extraction Pipeline Sequence

```mermaid
sequenceDiagram
    participant FeaturePipe as FeaturePipeline
    participant FrameSample as FrameSample
    participant Stabilizer as VideoStabilizer
    participant PoseEst as PoseEstimator
    participant DistanceCalc as DistanceCalculator
    participant AngleCalc as AngleCalculator
    participant Differentiator as Differentiator
    
    FeaturePipe->>FrameSample: fs.raw_frame = frame
    FeaturePipe->>FrameSample: fs.init_current_frame(frame_idx)
    
    FeaturePipe->>Stabilizer: stabilize(fs.raw_frame)
    Stabilizer-->>FeaturePipe: Return stabilized frame
    
    FeaturePipe->>PoseEst: get_pose_landmarks(stable)
    PoseEst-->>FeaturePipe: Return MediaPipe landmarks
    
    FeaturePipe->>FeaturePipe: Store landmarks for visualization
    
    alt Feature.RAW in mode
        FeaturePipe->>FrameSample: compute_raw(landmarks)
        FrameSample->>FrameSample: Normalize landmark coordinates
    end
    
    alt Feature.RAW_PX in mode
        FeaturePipe->>FrameSample: compute_raw_px(landmarks)
        FrameSample->>FrameSample: Store pixel coordinates
    end
    
    alt Feature.DIFF in mode
        FeaturePipe->>FrameSample: compute_diff(differentiator)
        FrameSample->>Differentiator: Calculate temporal differences
        Differentiator-->>FrameSample: Return difference features
    end
    
    alt Feature.SPATIAL in mode
        FeaturePipe->>FrameSample: compute_spatial(landmarks, dist_calc, ang_calc)
        FrameSample->>DistanceCalc: compute(landmarks)
        DistanceCalc->>DistanceCalc: Calculate 3D Euclidean distances
        DistanceCalc-->>FrameSample: Return distance features
        
        FrameSample->>AngleCalc: compute(landmarks)
        AngleCalc->>AngleCalc: Calculate joint angles using dot product
        AngleCalc-->>FrameSample: Return angle features
    end
    
    alt Feature.WINDOW in mode
        FeaturePipe->>FrameSample: windowed_features()
        FrameSample->>FrameSample: Aggregate features over time window
    end
```

## Model Inference Sequence

```mermaid
sequenceDiagram
    participant GUI as PlayerGUI
    participant Predictor as VideoPredictor
    participant Model as Keras Model
    participant Buffer as Sliding Window Buffer
    
    GUI->>Predictor: predict(feature_vector)
    
    Predictor->>Buffer: buffer.append(feature_vector)
    
    alt Buffer length < window_size (Warmup)
        Predictor-->>GUI: Return 0.0
    else Buffer is full (Ready for inference)
        Predictor->>Predictor: np.stack(buffer, axis=0)
        Predictor->>Predictor: np.expand_dims(window, axis=0)
        
        Predictor->>Model: model(input_tensor, training=False)
        Model->>Model: Forward pass through CNN
        Model-->>Predictor: Return prediction tensor
        
        Predictor->>Predictor: float(prediction[0])
        Predictor-->>GUI: Return probability (0.0-1.0)
    end
```

## Jump Detection State Machine Sequence

```mermaid
sequenceDiagram
    participant GUI as PlayerGUI
    participant Counter as JumpCounter
    participant StateMachine as Binary State Machine
    
    GUI->>Counter: process_prediction(prob, threshold)
    
    Counter->>Counter: y_pred = int((prob > threshold))
    
    Counter->>StateMachine: Update 4-bit sliding window
    Note over StateMachine: mark1 = (jump_cnt_binary_mark << 1) & 0b1111
    Note over StateMachine: jump_cnt_binary_mark = (mark1 | y_pred) & 0b1111
    
    alt Pattern 7 (0111) - Rising edge detected
        StateMachine->>Counter: is_on_rising = True
        Counter->>Counter: jump_cnt += 1
        Counter-->>GUI: Return (is_on_rising=True, jump_cnt)
        
    else Pattern 15 (1111) - Sustained rising state
        StateMachine->>Counter: is_on_rising = True
        Counter-->>GUI: Return (is_on_rising=True, jump_cnt)
        
    else Other patterns (0000, 0001, 0010, etc.)
        StateMachine->>Counter: is_on_rising = False
        Counter-->>GUI: Return (is_on_rising=False, jump_cnt)
    end
```

## Error Handling Sequence

```mermaid
sequenceDiagram
    participant GUI as PlayerGUI
    participant Logger as Logger
    participant EventBus as EventBus
    participant Container as Container
    
    alt Model Error
        GUI->>Logger: logger.error(f"Model error: {e}")
        GUI->>GUI: Break processing loop
        
    else Camera Error
        GUI->>Logger: logger.warning(f"Frame capture failed")
        GUI->>GUI: Increment error_count
        
        alt error_count > MAX_ERRORS
            GUI->>Logger: logger.error(f"Consecutive error count exceeded")
            GUI->>GUI: Break processing loop
        end
        
    else General Exception
        GUI->>Logger: logger.warning(f"Processing error: {e}")
        GUI->>GUI: Increment error_count
        
    end
    
    alt Application Error (v2.0)
        GUI->>EventBus: publish(ERROR_OCCURRED, {"error": str(e)}, "gui")
        EventBus->>Container: Update application state
        Container->>Logger: Log error in state
    end
```

## Application Shutdown Sequence (v2.0)

```mermaid
sequenceDiagram
    participant GUI as PlayerGUI
    participant MainV2 as main_v2.py
    participant EventBus as EventBus
    participant PluginManager as PluginManager
    participant Container as Container
    participant Logger as Logger
    
    alt User exits (press 'q')
        GUI->>GUI: Break processing loop
        GUI->>GUI: cap.release()
        GUI->>GUI: writer.release() (if exists)
        GUI->>GUI: cv2.destroyAllWindows()
        GUI-->>MainV2: Return from gui.run()
    end
    
    MainV2->>EventBus: publish(APPLICATION_STOP, {}, "main")
    MainV2->>EventBus: stop()
    MainV2->>PluginManager: cleanup()
    MainV2->>Container: cleanup()
    
    Container->>Logger: logger.info("Cleaning up services")
    Container->>Container: Clear services and singletons
    
    MainV2->>Logger: logger.info("Application shutdown complete")
```

## Performance Monitoring Sequence

```mermaid
sequenceDiagram
    participant GUI as PlayerGUI
    participant PerfStats as PerfStats
    participant EventBus as EventBus
    participant Container as Container
    
    loop For each frame
        GUI->>GUI: arr_ts.append(time.time()) - Start timing
        
        GUI->>GUI: Frame capture
        GUI->>GUI: arr_ts.append(time.time()) - Capture done
        
        GUI->>GUI: Feature extraction
        GUI->>GUI: arr_ts.append(time.time()) - Features done
        
        GUI->>GUI: Model inference
        GUI->>GUI: arr_ts.append(time.time()) - Inference done
        
        GUI->>GUI: Jump counting
        GUI->>GUI: arr_ts.append(time.time()) - Counting done
        
        GUI->>GUI: Display overlay
        GUI->>GUI: arr_ts.append(time.time()) - Display done
        
        GUI->>PerfStats: update("[Main Process]: ", arr_ts, 0)
        PerfStats->>PerfStats: Calculate FPS and latency
        PerfStats->>PerfStats: Update rolling statistics
        
        alt v2.0 Architecture
            GUI->>EventBus: publish(PERFORMANCE_UPDATE, metrics, "gui")
            EventBus->>Container: Update performance_metrics in state
        end
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