# Contributing to Lexora

First off, thank you for considering contributing to Lexora! 🎉

It's people like you that make Lexora such a great tool. We welcome contributions from everyone, whether it's:

- 🐛 Reporting a bug
- 💬 Discussing the current state of the code
- 📝 Submitting a fix
- 🚀 Proposing new features
- 📚 Improving documentation
- 🎨 Improving code quality

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [vesperakshay@gmail.com](mailto:vesperakshay@gmail.com).

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- pip or conda

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/lexora.git
cd lexora
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/VesperAkshay/lexora.git
```

## Development Setup

1. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -e ".[dev]"
```

3. **Install pre-commit hooks:**

```bash
pre-commit install
```

4. **Verify installation:**

```bash
python -c "import lexora; print(lexora.__version__)"
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- Clear and descriptive title
- Detailed steps to reproduce
- Expected vs actual behavior
- Code samples
- Environment details (OS, Python version, Lexora version)
- Error messages and stack traces

Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml).

### Suggesting Features

Feature suggestions are welcome! Please:

- Use a clear and descriptive title
- Provide detailed description of the proposed feature
- Explain why this feature would be useful
- Include code examples if possible

Use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.yml).

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

### Development Workflow

1. **Create a branch:**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. **Make your changes:**

- Write clean, readable code
- Follow our coding standards
- Add tests for new features
- Update documentation

3. **Test your changes:**

```bash
# Run all tests
python run_tests.py

# Run specific tests
pytest tests/test_your_feature.py -v

# Check code style
black lexora tests
isort lexora tests
flake8 lexora tests
mypy lexora
```

4. **Commit your changes:**

```bash
git add .
git commit -m "feat: add amazing feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test updates
- `chore:` Maintenance tasks

5. **Push to your fork:**

```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request:**

- Go to the original repository
- Click "New Pull Request"
- Select your branch
- Fill out the PR template
- Submit!

## Pull Request Process

1. **Before submitting:**
   - Update documentation
   - Add tests
   - Ensure all tests pass
   - Update CHANGELOG.md if applicable
   - Rebase on latest main

2. **PR Requirements:**
   - Clear description of changes
   - Link to related issues
   - All CI checks passing
   - At least one approval from maintainers
   - No merge conflicts

3. **Review Process:**
   - Maintainers will review your PR
   - Address any feedback
   - Once approved, a maintainer will merge

4. **After Merge:**
   - Delete your branch
   - Update your local repository

```bash
git checkout main
git pull upstream main
git push origin main
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all functions
- Docstrings for all public APIs (Google style)

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Example Code Style

```python
"""Module docstring describing the module."""

from typing import List, Optional

from lexora.models.core import Document


def process_documents(
    documents: List[Document],
    max_length: Optional[int] = None
) -> List[Document]:
    """Process a list of documents.
    
    Args:
        documents: List of documents to process
        max_length: Maximum length for each document
        
    Returns:
        List of processed documents
        
    Raises:
        ValueError: If documents list is empty
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    processed = []
    for doc in documents:
        if max_length and len(doc.content) > max_length:
            doc.content = doc.content[:max_length]
        processed.append(doc)
    
    return processed
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### Test Structure

```python
import pytest
from lexora import RAGAgent


class TestRAGAgent:
    """Tests for RAGAgent class."""
    
    def test_initialization_with_defaults(self):
        """Test that RAGAgent initializes with default config."""
        # Arrange & Act
        agent = RAGAgent()
        
        # Assert
        assert agent is not None
        assert agent.llm_config is not None
    
    @pytest.mark.asyncio
    async def test_query_execution(self):
        """Test that agent can execute queries."""
        # Arrange
        agent = RAGAgent()
        query = "test query"
        
        # Act
        result = await agent.query(query)
        
        # Assert
        assert result is not None
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=lexora --cov-report=html

# Specific test file
pytest tests/test_agent.py

# Specific test
pytest tests/test_agent.py::TestRAGAgent::test_initialization

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> function_name("test", 42)
        True
    """
```

### Documentation Updates

When adding features:

1. Update relevant documentation files
2. Add examples to README if applicable
3. Update API reference
4. Add to CHANGELOG.md

## Community

### Getting Help

- 💬 [GitHub Discussions](https://github.com/VesperAkshay/lexora/discussions)
- 🐛 [Issue Tracker](https://github.com/VesperAkshay/lexora/issues)
- 📧 Email: vesperakshay@gmail.com

### Recognition

Contributors are recognized in:

- README.md Contributors section
- Release notes
- CHANGELOG.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask questions! We're here to help:

- Open a [Discussion](https://github.com/VesperAkshay/lexora/discussions)
- Create an [Issue](https://github.com/VesperAkshay/lexora/issues)
- Email us at vesperakshay@gmail.com

---

Thank you for contributing to Lexora! 🚀
