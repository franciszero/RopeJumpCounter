# RopeJumpCounter 序列图

## 应用启动序列

```mermaid
sequenceDiagram
    participant User as 用户
    participant Main as main.py
    participant Config as AppConfig
    participant Logger as Logger
    participant GPU as GPU设置
    participant Predictor as VideoPredictor
    participant GUI as PlayerGUI
    
    User->>Main: 运行 python main.py
    Main->>Config: 加载配置
    Config-->>Main: 返回配置对象
    
    Main->>Logger: 初始化日志
    Logger-->>Main: 日志系统就绪
    
    Main->>GPU: 设置GPU加速
    GPU-->>Main: GPU配置完成
    
    Main->>Predictor: 加载模型
    Predictor-->>Main: 模型加载完成
    
    Main->>GUI: 启动GUI界面
    GUI-->>User: 显示主界面
```

## 实时跳绳检测序列

```mermaid
sequenceDiagram
    participant GUI as GUI界面
    participant Capture as 视频捕获
    participant Stabilizer as 视频稳定
    participant Vision as 计算机视觉
    participant Predictor as 模型预测
    participant Counter as 跳绳计数
    participant Display as 显示更新
    
    loop 每一帧
        GUI->>Capture: 获取视频帧
        Capture-->>GUI: 返回原始帧
        
        GUI->>Stabilizer: 视频稳定处理
        Stabilizer-->>GUI: 返回稳定帧
        
        GUI->>Vision: 姿态估计
        Vision->>Vision: MediaPipe处理
        Vision-->>GUI: 返回关键点
        
        GUI->>Predictor: 特征提取和预测
        Predictor->>Predictor: 滑动窗口推理
        Predictor-->>GUI: 返回跳跃概率
        
        GUI->>Counter: 状态机处理
        Counter->>Counter: 模式匹配
        Counter-->>GUI: 返回跳跃计数
        
        GUI->>Display: 更新显示
        Display-->>GUI: 界面刷新
    end
```

## 模型训练序列

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as run.py
    participant Training as 模型训练
    participant Data as 数据加载
    participant Model as 模型定义
    participant Trainer as 训练器
    
    User->>CLI: python run.py train
    CLI->>Training: 启动训练流程
    
    Training->>Data: 加载训练数据
    Data->>Data: 数据预处理
    Data-->>Training: 返回数据批次
    
    Training->>Model: 创建模型
    Model-->>Training: 返回模型实例
    
    loop 训练轮次
        Training->>Trainer: 开始训练
        Trainer->>Model: 前向传播
        Model-->>Trainer: 返回预测结果
        Trainer->>Trainer: 计算损失
        Trainer->>Model: 反向传播
        Model-->>Trainer: 更新权重
        Trainer-->>Training: 返回训练指标
    end
    
    Training->>Training: 保存模型
    Training-->>CLI: 训练完成
    CLI-->>User: 显示结果
```

## 数据标注序列

```mermaid
sequenceDiagram
    participant User as 标注员
    participant GUI as 标注界面
    participant Video as 视频播放器
    participant Labels as 标签管理
    participant Storage as 数据存储
    
    User->>GUI: 打开标注工具
    GUI->>Video: 加载视频文件
    Video-->>GUI: 返回视频信息
    
    loop 逐帧标注
        GUI->>Video: 播放/暂停
        Video-->>GUI: 当前帧
        
        User->>GUI: 标记跳跃事件
        GUI->>Labels: 记录标签
        Labels-->>GUI: 确认标签
        
        GUI->>Storage: 保存标注数据
        Storage-->>GUI: 保存确认
    end
    
    User->>GUI: 完成标注
    GUI->>Storage: 导出标注文件
    Storage-->>User: 标注文件
```

## 错误处理序列

```mermaid
sequenceDiagram
    participant App as 应用程序
    participant Predictor as 预测器
    participant Model as 模型
    participant Logger as 日志系统
    participant User as 用户
    
    App->>Predictor: 模型预测
    Predictor->>Model: 推理请求
    
    alt 模型加载失败
        Model-->>Predictor: ModelError
        Predictor->>Logger: 记录错误
        Predictor-->>App: 抛出异常
        App->>Logger: 记录应用错误
        App-->>User: 显示错误信息
    else 推理失败
        Model-->>Predictor: 推理异常
        Predictor->>Logger: 记录推理错误
        Predictor-->>App: 返回默认值
        App->>App: 继续处理
    else 正常情况
        Model-->>Predictor: 预测结果
        Predictor-->>App: 返回结果
    end
```

## 性能监控序列

```mermaid
sequenceDiagram
    participant GUI as GUI界面
    participant Perf as 性能监控
    participant System as 系统资源
    participant Logger as 日志系统
    
    loop 性能监控
        GUI->>Perf: 开始帧处理
        Perf->>System: 获取系统资源
        System-->>Perf: CPU/GPU/内存使用率
        
        GUI->>Perf: 帧处理完成
        Perf->>Perf: 计算FPS
        Perf->>Perf: 计算延迟
        
        Perf->>Logger: 记录性能指标
        Perf-->>GUI: 返回性能数据
        
        GUI->>GUI: 更新性能显示
    end
```

## 配置管理序列

```mermaid
sequenceDiagram
    participant App as 应用程序
    participant Config as 配置管理器
    participant File as 配置文件
    participant Env as 环境变量
    participant Default as 默认配置
    
    App->>Config: 加载配置
    
    alt 配置文件存在
        Config->>File: 读取config.yaml
        File-->>Config: 返回配置数据
        Config->>Config: 解析YAML
        Config-->>App: 返回配置对象
    else 环境变量存在
        Config->>Env: 读取环境变量
        Env-->>Config: 返回环境配置
        Config->>Config: 合并配置
        Config-->>App: 返回配置对象
    else 使用默认值
        Config->>Default: 获取默认配置
        Default-->>Config: 返回默认值
        Config-->>App: 返回配置对象
    end
```

## 使用说明

### 1. **查看序列图**
- 在 GitHub 中直接查看
- 使用 Mermaid Live Editor 编辑
- 导出为图片格式

### 2. **更新序列图**
当代码逻辑变更时，记得同步更新相应的序列图。

### 3. **添加新序列图**
对于新的功能模块，可以按照相同的格式添加序列图。

### 4. **最佳实践**
- 保持图表简洁明了
- 突出关键交互点
- 包含错误处理流程
- 标注重要的时序关系 