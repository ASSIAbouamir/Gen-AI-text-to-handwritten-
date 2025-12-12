# 🤝 Contributing to AI Handwriting Generator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/handwriting-transformers/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Error messages and logs

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its benefits
3. Provide examples of how it would work

### Pull Requests

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/YourFeature`
3. **Make changes** with clear, commented code
4. **Test** your changes thoroughly
5. **Commit**: `git commit -m 'Add YourFeature'`
6. **Push**: `git push origin feature/YourFeature`
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/handwriting-transformers.git
cd handwriting-transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use **type hints** where possible
- Write **docstrings** for all functions/classes
- Keep functions **focused and small**

### Code Formatting

```bash
# Format code with black
black .

# Check with flake8
flake8 .

# Type checking
mypy .
```

### Example

```python
def process_image(img: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process image with optional normalization.
    
    Args:
        img: Input image array
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Processed image array
    """
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
    return img
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=models
```

### Writing Tests

```python
import pytest
from models.model import TRGAN

def test_model_initialization():
    """Test that model initializes correctly"""
    model = TRGAN()
    assert model is not None
    assert hasattr(model, 'netG')
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
    """
    pass
```

### Updating Documentation

- Update relevant `.md` files when adding features
- Add examples to `USAGE.md`
- Update `FAQ.md` for common issues

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add style interpolation feature
fix: Resolve memory leak in generator
docs: Update installation instructions
refactor: Simplify text encoding logic
test: Add tests for OCR network
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

## Areas for Contribution

### High Priority

- [ ] Multi-language support
- [ ] Mobile/web deployment
- [ ] Performance optimization
- [ ] Additional handwriting styles

### Medium Priority

- [ ] Style interpolation
- [ ] Batch processing CLI
- [ ] Docker containerization
- [ ] API server

### Good First Issues

- [ ] Improve error messages
- [ ] Add more examples
- [ ] Fix typos in documentation
- [ ] Add unit tests

## Review Process

1. **Automated checks** must pass (tests, linting)
2. **Code review** by maintainer
3. **Testing** on different systems
4. **Merge** when approved

## Questions?

- Open an issue with the `question` label
- Email: your-email@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
