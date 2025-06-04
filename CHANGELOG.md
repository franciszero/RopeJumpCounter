# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete project architecture refactoring
- Modular design with clear separation of responsibilities
- Unified command-line entry point (`run.py`)
- Configuration-driven application architecture

### Changed
- Refactored `PoseDetection` module to `src/ml/`
- Updated all import paths
- Reorganized directory structure

## [2.0.0] - 2024-05-23

### Added
- 🏗️ **Architecture Refactoring**: Completely redesigned modular architecture
- 📁 **New Directory Structure**: Clear directory organization by functional layers
- 🎯 **Unified Entry Point**: `run.py` provides unified command-line interface
- ⚙️ **Configuration Management**: Flexible YAML-based configuration system
- 🧠 **ML Module**: Independent machine learning module containing:
  - Data annotation tools
  - Dataset builders
  - Feature extraction pipeline
  - Multiple deep learning models
  - Model training scripts
  - Visualization validation tools
- 🎮 **Core Module**:
  - `VideoPredictor`: Video predictor
  - `JumpCounter`: Jump rope counter
  - Exception handling system
- 🖥️ **Interface Module**: Refactored GUI interface
- 🛠️ **Utility Modules**:
  - Pose estimation
  - Video stabilization
  - Performance statistics
  - And more...
- 📹 **Capture Module**: Support for multiple video capture methods

### Changed
- 🔄 **Import Paths**: All import paths updated to new module structure
- 📝 **Documentation**: Complete project documentation and API documentation
- 🧪 **Testing**: Improved test structure

### Fixed
- 🐛 Fixed multiple inter-module dependency issues
- 🔧 Improved error handling and exception management

## [1.5.0] - 2024-05-22

### Added
- Support for multiple deep learning models (CNN, TCN, ResNet, etc.)
- Model performance comparison and report generation
- Improved data annotation tools

### Changed
- Optimized feature extraction pipeline
- Improved real-time performance

## [1.4.0] - 2024-05-21

### Added
- GPU acceleration support
- Mixed precision training
- Video stabilization functionality

### Fixed
- Fixed camera compatibility issues
- Improved memory usage

## [1.3.0] - 2024-05-20

### Added
- Data annotation GUI tools
- Batch data processing
- Model visualization tools

## [1.2.0] - 2024-05-15

### Added
- TCN (Temporal Convolutional Network) 模型
- Improved jump detection algorithm
- Performance monitoring tools

### Changed
- Optimized feature extraction speed
- Improved UI responsiveness

## [1.1.0] - 2024-05-10

### Added
- Multiple CNN model architectures
- Model training scripts
- Dataset building tools

### Fixed
- Fixed counting accuracy issues
- Improved error handling

## [1.0.0] - 2024-05-01

### Added
- 🎉 **Initial Release**
- MediaPipe-based pose estimation
- Real-time jump rope counting functionality
- Basic deep learning models
- Simple GUI interface

### Features
- Real-time video processing
- Jump action detection
- Count display
- Basic configuration options

---

## Version Notes

### Version Number Format
- **Major version**: Incompatible API changes
- **Minor version**: Backward-compatible functional additions
- **Patch version**: Backward-compatible bug fixes

### Tag Descriptions
- `Added`: New features
- `Changed`: Changes to existing functionality
- `Deprecated`: Features that will be removed soon
- `Removed`: Features that have been removed
- `Fixed`: Bug fixes
- `Security`: Security-related fixes
