# 火柴人姿态可视化功能

## 功能介绍

在 `model_visualize.py` 中新增了火柴人姿态可视化功能，可以在视频播放时实时显示人体姿态的火柴人效果。

## 功能特点

### 🎨 **视觉效果**
- **彩色关键点**: 不同身体部位使用不同颜色和大小
- **连接线**: 根据身体部位类型显示不同颜色和粗细
- **边框效果**: 关键点带有黑色边框，提高可见性

### 🎯 **关键点颜色方案**
- **眼睛**: 黄色 (半径 3px)
- **主要关节** (肩膀、髋部): 红色 (半径 5px)
- **中间关节** (肘部、膝盖): 青色 (半径 4px)
- **末端关节** (手腕、脚踝、脚部): 紫色 (半径 4px)
- **其他**: 绿色 (半径 3px)

### 🔗 **连接线颜色方案**
- **头部连接**: 黄色 (粗细 2px)
- **躯干连接**: 红色 (粗细 3px)
- **手臂连接**: 青色 (粗细 2px)
- **腿部连接**: 绿色 (粗细 2px)

## 使用方法

### 1. **启用火柴人显示** (默认)
```bash
python src/ml/visualization/model_visualize.py --video your_video.mp4 --model your_model.keras
```

### 2. **显式启用火柴人**
```bash
python src/ml/visualization/model_visualize.py --video your_video.mp4 --model your_model.keras --stick-figure
```

### 3. **禁用火柴人显示**
```bash
python src/ml/visualization/model_visualize.py --video your_video.mp4 --model your_model.keras --no-stick-figure
```

### 4. **通过统一入口使用**
```bash
# 启用火柴人
python run.py visualize --model best_model.keras --video test.mp4 --stick-figure

# 禁用火柴人
python run.py visualize --model best_model.keras --video test.mp4 --no-stick-figure
```

## 技术实现

### 数据流程
1. **姿态检测**: MediaPipe 提取姿态关键点
2. **数据保存**: FeaturePipeline 保存原始关键点数据
3. **火柴人绘制**: 在视频帧上绘制关键点和连接线
4. **实时显示**: 与跳绳计数信息一起显示

### 关键代码
```python
# 在 FeaturePipeline 中保存关键点
self.landmarks = lm

# 在 _overlay 方法中绘制火柴人
if self.show_stick_figure and landmarks is not None:
    frame = self._draw_stick_figure(frame, landmarks)
```

## 配置选项

### 构造函数参数
```python
PlayerGUI(video_path, predictor, show_stick_figure=True)
```

### 命令行参数
- `--stick-figure`: 显示火柴人 (默认)
- `--no-stick-figure`: 隐藏火柴人

## 性能影响

- **CPU 开销**: 轻微增加 (~5-10%)
- **内存使用**: 基本无影响
- **显示延迟**: 无明显影响

## 自定义配置

### 修改颜色方案
在 `_draw_stick_figure` 方法中修改颜色定义：
```python
# 自定义颜色
if 'SHOULDER' in landmark_idx.name:
    color = (255, 128, 0)  # 橙色
```

### 修改关键点大小
```python
# 自定义大小
elif 'SHOULDER' in landmark_idx.name:
    radius = 6  # 更大的关键点
```

### 修改连接线粗细
```python
# 自定义粗细
elif 'SHOULDER' in start_idx.name:
    thickness = 4  # 更粗的连接线
```

## 故障排除

### 问题：火柴人不显示
**解决方案**:
1. 确保使用了 `--stick-figure` 参数
2. 检查姿态检测是否正常工作
3. 确认关键点可见性阈值 (>0.5)

### 问题：关键点位置不准确
**解决方案**:
1. 改善光线条件
2. 确保人物在画面中央
3. 检查摄像头分辨率设置

### 问题：性能下降
**解决方案**:
1. 使用 `--no-stick-figure` 禁用火柴人
2. 降低视频分辨率
3. 启用 GPU 加速

## 扩展功能

### 可能的增强
- 轨迹显示：显示关键点的运动轨迹
- 动态颜色：根据运动状态改变颜色
- 3D 效果：添加深度信息显示
- 自定义主题：提供多种颜色主题

### 集成建议
- 与实时计数器集成
- 添加到 GUI 界面中
- 支持录制带火柴人的视频
