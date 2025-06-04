# Contributing Guide

Thank you for your interest in contributing to RopeJumpCounter! We welcome all forms of contributions.

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Provide detailed reproduction steps
- Include system information and error logs

### Feature Requests
- Describe new features in Issues
- Explain use cases and expected outcomes
- Discuss implementation approaches

### Code Contributions
1. Fork the project
2. Create a feature branch
3. Submit your code
4. Create a Pull Request

## 🛠️ Development Environment Setup

### 1. Clone the Project
```bash
git clone https://github.com/your-username/RopeJumpCounter.git
cd RopeJumpCounter
```

### 2. Setup Development Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Development Tools
```bash
# Code formatting
pip install black isort flake8

# Testing tools
pip install pytest pytest-cov

# Documentation tools
pip install sphinx
```

## 📝 Code Standards

### Python Code Style
- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://pep8.org/) standards
- Use [isort](https://pycqa.github.io/isort/) to organize imports

### Code Formatting
```bash
# Format all code
black src/
isort src/

# Check code style
flake8 src/
```

### Naming Conventions
- **Class names**: PascalCase (`VideoPredictor`)
- **Functions/variables**: snake_case (`process_frame`)
- **Constants**: UPPER_CASE (`SELECTED_LM`)
- **Private members**: prefix underscore (`_private_method`)

### Documentation
```python
def process_prediction(self, prob: float, threshold: float) -> tuple[bool, int]:
    """
    Process prediction results and return jump state information.

    Args:
        prob: Prediction probability (0.0-1.0)
        threshold: Decision threshold for classification

    Returns:
        tuple: (is_rising, current_jump_count)

    Raises:
        ValueError: If probability value is invalid
    """
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Generate coverage report
pytest --cov=src tests/
```

### Writing Tests
- Write unit tests for new features
- Test file naming: `test_*.py`
- Test function naming: `test_*`

```python
def test_jump_counter():
    counter = JumpCounter()
    is_rising, count = counter.process_prediction(0.8, 0.5)
    assert isinstance(is_rising, bool)
    assert isinstance(count, int)
```

## 📁 Project Structure

### Adding New Modules
1. Create module in appropriate directory
2. Add `__init__.py` file
3. Update relevant documentation

### Module Organization Principles
- **Single Responsibility**: Each module has a clear purpose
- **Loose Coupling**: Minimize inter-module dependencies
- **High Cohesion**: Group related functionality together

## 🔄 Commit Standards

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation updates
- `style`: Code formatting
- `refactor`: Code refactoring
- `test`: Test-related changes
- `chore`: Build/tooling changes

### Example
```
feat(core): add jump rope counter class

- Implement state machine-based counting logic
- Support configurable thresholds
- Add comprehensive unit tests

Closes #123
```

## 🚀 Pull Request Process

### 1. Create Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Development and Testing
- Write code
- Add tests
- Update documentation

### 3. Submit Code
```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

### 4. Create PR
- Fill out PR template
- Describe changes clearly
- Link related issues

### 5. Code Review
- Respond to review feedback
- Make necessary changes
- Update PR as needed

## 📋 Checklist

Before submitting a PR, please ensure:

- [ ] Code passes all tests
- [ ] Code style follows standards
- [ ] Added necessary tests
- [ ] Updated relevant documentation
- [ ] Commit messages follow conventions
- [ ] No new dependency conflicts

## 🎯 Priority Areas

We especially welcome contributions in:

1. **Bug Fixes**: Resolve known issues
2. **Performance Optimization**: Improve efficiency
3. **New Models**: Add deep learning architectures
4. **Documentation**: Enhance docs and examples
5. **Test Coverage**: Increase test coverage

## 📞 Contact

- **GitHub Issues**: Technical questions and bug reports
- **Discussions**: Feature discussions and Q&A
- **Email**: Private communication

Thank you for contributing! 🎉
