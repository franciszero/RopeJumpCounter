# 贡献指南

感谢你对 RopeJumpCounter 项目的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题
- 使用 GitHub Issues 报告 bug
- 提供详细的复现步骤
- 包含系统信息和错误日志

### 功能请求
- 在 Issues 中描述新功能
- 说明使用场景和预期效果
- 讨论实现方案

### 代码贡献
1. Fork 项目
2. 创建功能分支
3. 提交代码
4. 创建 Pull Request

## 🛠️ 开发环境设置

### 1. 克隆项目
```bash
git clone https://github.com/your-username/RopeJumpCounter.git
cd RopeJumpCounter
```

### 2. 设置开发环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. 安装开发工具
```bash
# 代码格式化
pip install black isort flake8

# 测试工具
pip install pytest pytest-cov

# 文档工具
pip install sphinx
```

## 📝 代码规范

### Python 代码风格
- 使用 [Black](https://black.readthedocs.io/) 进行代码格式化
- 遵循 [PEP 8](https://pep8.org/) 规范
- 使用 [isort](https://pycqa.github.io/isort/) 整理导入

### 格式化代码
```bash
# 格式化所有代码
black src/
isort src/

# 检查代码风格
flake8 src/
```

### 命名规范
- **类名**: PascalCase (`VideoPredictor`)
- **函数/变量**: snake_case (`process_frame`)
- **常量**: UPPER_CASE (`SELECTED_LM`)
- **私有成员**: 前缀下划线 (`_private_method`)

### 文档字符串
```python
def process_prediction(self, prob: float, threshold: float) -> tuple[bool, int]:
    """
    处理预测结果，返回是否正在上升和当前跳数
    
    Args:
        prob: 预测概率 (0.0-1.0)
        threshold: 判断阈值
        
    Returns:
        tuple: (是否正在上升, 当前跳数)
        
    Raises:
        ValueError: 如果概率值无效
    """
```

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_core.py

# 生成覆盖率报告
pytest --cov=src tests/
```

### 编写测试
- 为新功能编写单元测试
- 测试文件命名: `test_*.py`
- 测试函数命名: `test_*`

```python
def test_jump_counter():
    counter = JumpCounter()
    is_rising, count = counter.process_prediction(0.8, 0.5)
    assert isinstance(is_rising, bool)
    assert isinstance(count, int)
```

## 📁 项目结构

### 添加新模块
1. 在适当的目录下创建模块
2. 添加 `__init__.py` 文件
3. 更新相关文档

### 模块组织原则
- **单一职责**: 每个模块有明确的职责
- **低耦合**: 减少模块间依赖
- **高内聚**: 相关功能放在一起

## 🔄 提交规范

### Commit 消息格式
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### 类型 (type)
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 示例
```
feat(core): 添加跳绳计数器类

- 实现基于状态机的计数逻辑
- 支持可配置的阈值
- 添加单元测试

Closes #123
```

## 🚀 Pull Request 流程

### 1. 创建分支
```bash
git checkout -b feature/your-feature-name
```

### 2. 开发和测试
- 编写代码
- 添加测试
- 更新文档

### 3. 提交代码
```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

### 4. 创建 PR
- 填写 PR 模板
- 描述变更内容
- 关联相关 Issues

### 5. 代码审查
- 响应审查意见
- 修改代码
- 更新 PR

## 📋 检查清单

提交 PR 前请确认：

- [ ] 代码通过所有测试
- [ ] 代码风格符合规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] Commit 消息符合规范
- [ ] 没有引入新的依赖冲突

## 🎯 优先级

我们特别欢迎以下类型的贡献：

1. **Bug 修复**: 修复已知问题
2. **性能优化**: 提升运行效率
3. **新模型**: 添加新的深度学习模型
4. **文档改进**: 完善文档和示例
5. **测试覆盖**: 增加测试用例

## 📞 联系方式

- GitHub Issues: 技术问题和 bug 报告
- Discussions: 功能讨论和问答
- Email: 私人联系

感谢你的贡献！🎉
