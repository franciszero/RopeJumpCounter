# User Guide

## 🎯 Quick Start

### First Time Use

1. **Install Application** (see [INSTALL.md](../INSTALL.md))
2. **Connect Camera**
3. **Run Application**:
   ```bash
   python main.py
   ```

### Basic Operations

- **Start Counting**: Automatically starts when application launches
- **Pause/Resume**: Press spacebar
- **Exit**: Press 'q' key or close window

## 📱 Interface Overview

### Main Interface Elements

- **Video Window**: Displays real-time camera feed
- **Jump Count**: Top-left corner shows current jump count
- **Probability Display**: Bottom-left corner shows detection probability
- **Performance Info**: Top-right corner shows FPS and latency
- **Status Indicator**: Red highlight when jump is detected

### Status Indicators

- **RISING**: Upward motion detected
- **Probability Value**: 0.0-1.0, higher values indicate more likely jump
- **FPS**: Frames processed per second
- **Latency**: Processing delay (milliseconds)

## ⚙️ Configuration Options

### Basic Configuration

Edit the `config.yaml` file:

```yaml
camera:
  width: 640      # Recommended: 640x480 or 1280x720
  height: 480
  fps: 30         # Recommended: 30fps
  device_index: 0 # If multiple cameras, try 1, 2...

model:
  threshold: 0.5  # Adjust detection sensitivity (0.3-0.7)
```

### Advanced Configuration

```yaml
gpu:
  enabled: true         # Enable GPU acceleration
  mixed_precision: true # Improve performance

performance:
  max_fps: 30          # Limit maximum frame rate
  
ui:
  show_debug_info: true # Show detailed information
```

## 🎮 Usage Tips

### Getting Best Results

1. **Lighting Conditions**:
   - Ensure adequate lighting
   - Avoid backlighting and strong shadows
   - Use even background lighting

2. **Camera Position**:
   - Distance: 1.5-2 meters
   - Height: Chest level
   - Ensure full body is in frame

3. **Jump Rope Posture**:
   - Maintain standard jump rope posture
   - Avoid excessive swinging
   - Stay within camera field of view

### Adjusting Detection Sensitivity

If counting is inaccurate:

- **Missing counts**: Lower threshold (e.g., 0.3-0.4)
- **False counts**: Raise threshold (e.g., 0.6-0.7)

### Performance Optimization

1. **Lower resolution**: 640x480 is usually sufficient
2. **Enable GPU**: If you have NVIDIA graphics card
3. **Close other programs**: Free up CPU/GPU resources

## 🔧 Troubleshooting

### Common Issues

#### Camera Cannot Open
```
Solutions:
1. Check camera connection
2. Try different device_index (0, 1, 2...)
3. Close other programs using camera
4. Check camera permissions
```

#### Inaccurate Detection
```
Solutions:
1. Adjust threshold value
2. Improve lighting conditions
3. Adjust camera position
4. Check jump rope posture
```

#### Performance Issues (Low FPS)
```
Solutions:
1. Lower video resolution
2. Enable GPU acceleration
3. Close unnecessary programs
4. Check system resource usage
```

#### Application Crashes
```
Solutions:
1. Check error logs (logs/ directory)
2. Confirm dependencies are installed correctly
3. Try reinstalling
4. Report bug (include error information)
```

## 📊 Data Management

### Video Recording

Enable video recording:
```yaml
save_video_path: "data/recordings"
```

Recorded videos are saved as:
- Format: AVI (XVID encoding)
- Naming: `jump_YYYY.MM.DD.HH.MM.SS.avi`

### Log Files

Log file location: `logs/`
- Contains detailed runtime information
- Used for problem diagnosis
- Configurable log levels

## 🎯 Advanced Features

### Multi-mode Operation

```bash
# Real-time counting (default)
python main.py

# Legacy mode
python run.py legacy

# Visualization mode
python run.py visualize --model your_model.keras --video test.mp4
```

### Custom Models

1. Place model files in `model_files/` directory
2. Update `model_name` in configuration file
3. Restart application

### Batch Processing

```bash
# Process video files
python run.py visualize --video path/to/video.mp4
```

## 📈 Performance Monitoring

### Real-time Metrics

- **FPS**: Processing frame rate, recommended >20
- **Latency**: Processing delay, recommended <50ms
- **Accuracy**: Verify through actual comparison

### Optimization Recommendations

1. **Hardware Upgrade**: Better CPU/GPU
2. **Parameter Tuning**: Adjust model parameters
3. **Environment Optimization**: Improve recording environment

## 🆘 Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: Check `docs/examples/`
- **API**: Check `docs/API.md`
- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Discussions
