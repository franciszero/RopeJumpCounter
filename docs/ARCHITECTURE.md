# RopeJumpCounter 系统架构

## 整体架构图

```mermaid
graph TB
    subgraph "用户界面层"
        GUI[PlayerGUI<br/>图形用户界面]
        CLI[命令行接口<br/>run.py]
    end
    
    subgraph "应用层"
        APP[主应用<br/>main.py]
        CONFIG[配置管理<br/>AppConfig]
        LOGGER[日志系统<br/>setup_logger]
    end
    
    subgraph "核心业务层"
        PREDICTOR[视频预测器<br/>VideoPredictor]
        COUNTER[跳绳计数器<br/>JumpCounter]
        EXCEPTIONS[异常处理<br/>AppError]
    end
    
    subgraph "机器学习层"
        MODELS[模型定义<br/>CNN/TCN/ResNet]
        TRAINING[模型训练<br/>model_training]
        FEATURES[特征提取<br/>FrameSample]
        VIZ[可视化<br/>model_visualize]
    end
    
    subgraph "数据层"
        CAPTURE[视频捕获<br/>pyav_capture]
        STABILIZER[视频稳定<br/>VideoStabilizer]
        LABELING[数据标注<br/>main_gui]
    end
    
    subgraph "工具层"
        VISION[计算机视觉<br/>vision.py]
        PERF[性能监控<br/>Perf.py]
        UTILS[通用工具<br/>utils/]
    end
    
    subgraph "外部依赖"
        TF[TensorFlow<br/>深度学习框架]
        MP[MediaPipe<br/>姿态估计]
        CV[OpenCV<br/>图像处理]
        YAML[PyYAML<br/>配置解析]
    end
    
    %% 连接关系
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

## 数据流架构

```mermaid
flowchart LR
    subgraph "输入"
        CAM[摄像头]
        VIDEO[视频文件]
    end
    
    subgraph "处理流程"
        CAP[视频捕获]
        STAB[视频稳定]
        POSE[姿态估计]
        FEAT[特征提取]
        PRED[模型预测]
        COUNT[跳绳计数]
    end
    
    subgraph "输出"
        DISPLAY[实时显示]
        SAVE[视频保存]
        LOG[日志记录]
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

## 模块依赖关系

```mermaid
graph TD
    subgraph "核心模块"
        A[main.py] --> B[AppConfig]
        A --> C[VideoPredictor]
        A --> D[PlayerGUI]
        A --> E[setup_logger]
    end
    
    subgraph "ML模块"
        C --> F[models/]
        C --> G[features/]
        F --> H[TensorFlow]
        G --> I[MediaPipe]
    end
    
    subgraph "工具模块"
        D --> J[vision.py]
        D --> K[Perf.py]
        J --> I
        K --> L[psutil]
    end
    
    subgraph "配置模块"
        B --> M[PyYAML]
        B --> N[环境变量]
    end
```

## 部署架构

```mermaid
graph TB
    subgraph "开发环境"
        DEV[开发者机器]
        DEV --> GIT[Git仓库]
        DEV --> IDE[IDE/编辑器]
    end
    
    subgraph "构建环境"
        BUILD[CI/CD管道]
        BUILD --> TEST[自动化测试]
        BUILD --> PACKAGE[打包]
    end
    
    subgraph "运行环境"
        PROD[生产环境]
        PROD --> GPU[GPU服务器]
        PROD --> CAM[摄像头设备]
        PROD --> MONITOR[监控系统]
    end
    
    GIT --> BUILD
    BUILD --> PROD
```

## 技术栈架构

```mermaid
graph LR
    subgraph "前端/界面"
        GUI[PySimpleGUI]
        CLI[argparse]
    end
    
    subgraph "后端/核心"
        PYTHON[Python 3.8+]
        TF[TensorFlow 2.8+]
        CV[OpenCV]
        MP[MediaPipe]
    end
    
    subgraph "数据处理"
        NUMPY[numpy]
        PANDAS[pandas]
        YAML[PyYAML]
    end
    
    subgraph "系统集成"
        OS[操作系统]
        GPU[GPU驱动]
        CAM[摄像头驱动]
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

## 如何使用这些架构图

### 1. **在 GitHub 中查看**
GitHub 原生支持 Mermaid 图表，直接在 Markdown 中显示。

### 2. **在文档中引用**
```markdown
## 系统概述
请参考 [架构图](./ARCHITECTURE.md#整体架构图) 了解系统结构。
```

### 3. **导出为图片**
可以使用 Mermaid CLI 工具导出为 PNG/SVG：
```bash
# 安装 mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# 导出图片
mmdc -i architecture.md -o architecture.png
```

### 4. **在线编辑器**
- [Mermaid Live Editor](https://mermaid.live/)
- [Draw.io](https://draw.io/) (支持 Mermaid)

## 架构图的最佳实践

### 1. **分层清晰**
- 用户界面层
- 应用层
- 业务逻辑层
- 数据访问层

### 2. **颜色编码**
```mermaid
graph TB
    subgraph "用户层" 
        style GUI fill:#e1f5fe
        GUI[GUI组件]
    end
    
    subgraph "业务层"
        style CORE fill:#f3e5f5
        CORE[核心业务]
    end
    
    subgraph "数据层"
        style DATA fill:#e8f5e8
        DATA[数据存储]
    end
```

### 3. **保持更新**
- 代码变更时同步更新架构图
- 定期审查架构图的准确性
- 版本控制架构图

这样您就有了一个完整的系统架构文档！您觉得哪种图表最有用？我可以帮您进一步优化或添加其他类型的架构图。 