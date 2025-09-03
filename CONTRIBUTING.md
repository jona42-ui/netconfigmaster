# Contributing to NetConfigMaster

Thank you for your interest in contributing to NetConfigMaster! We welcome contributions from the community and are excited to work with you.

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/) (optional)
- [VS Code](https://code.visualstudio.com/) with Dev Containers extension (optional)

### Setup Options

#### Option 1: Automated Setup (Recommended)
```bash
git clone https://github.com/YOUR_USERNAME/netconfigmaster.git
cd netconfigmaster
./scripts/setup.sh
```

#### Option 2: Manual Setup
```bash
git clone https://github.com/YOUR_USERNAME/netconfigmaster.git
cd netconfigmaster
poetry install --with dev,docs
poetry run pre-commit install
```

#### Option 3: Dev Container (VS Code)
1. Clone the repository
2. Open in VS Code
3. Click "Reopen in Container" when prompted
4. Everything is set up automatically!

#### Option 4: Docker Development
```bash
git clone https://github.com/YOUR_USERNAME/netconfigmaster.git
cd netconfigmaster
docker-compose up dev
```

### Development Workflow
```bash
# Activate Poetry environment
poetry shell

# Make your changes
# ...

# Format and lint
./scripts/lint.sh

# Run tests
./scripts/test.sh

# Create a branch for your feature
git checkout -b feature/your-feature-name
```

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include steps to reproduce
- Provide system information (OS, Python version, etc.)
- Include relevant error messages and logs

### ✨ Feature Requests
- Check existing issues first
- Clearly describe the feature and its benefits
- Provide use cases and examples
- Consider backward compatibility

### 💻 Code Contributions
We welcome contributions in these areas:

#### High Priority
- **Model Improvements**: Better architectures, training strategies
- **Evaluation Metrics**: New custom metrics for network configuration validation
- **Data Processing**: Enhanced preprocessing and tokenization
- **Testing**: Unit tests, integration tests, benchmarks

#### Medium Priority
- **Documentation**: API docs, tutorials, examples
- **Performance**: Optimization, caching, batch processing
- **UI/UX**: Web interface improvements, CLI tools
- **Dataset**: New training/evaluation datasets

#### Low Priority
- **DevOps**: CI/CD, containerization, deployment scripts
- **Monitoring**: Logging, metrics collection, debugging tools

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use Poetry for dependency management
- Code formatting is automated with Black and isort
- Run formatting and linting:
  ```bash
  ./scripts/lint.sh  # Format and lint
  ./scripts/test.sh  # Run tests and checks
  ```
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Commit Messages
Follow conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(metrics): add BLEU score evaluation metric`
- `fix(training): resolve memory leak in batch processing`
- `docs(readme): update installation instructions`
- `test(evaluation): add unit tests for nmstate validation`

### Code Organization
- **New features**: Add to appropriate directory (`src/`, `metrics/`, etc.)
- **Tests**: Place in `tests/` directory with descriptive names
- **Documentation**: Update relevant README files and `docs/`
- **Dependencies**: Update `requirements.txt` if needed

## 🧪 Testing

### Running Tests
```bash
# Using the test script
./scripts/test.sh

# Or manually with Poetry
poetry run pytest tests/ -v --cov=src --cov=metrics

# Or using Docker
docker-compose run --rm dev poetry run pytest tests/
```

### Writing Tests
- Write tests for new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies

## 📝 Pull Request Process

1. **Create a Pull Request** from your feature branch
2. **Write a clear title** and description
3. **Reference related issues** using `Closes #123` or `Fixes #123`
4. **Include screenshots** for UI changes
5. **Update documentation** as needed
6. **Add tests** for new functionality

### PR Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Descriptive commit messages

### Review Process
- Maintainers will review within 48-72 hours
- Address feedback constructively
- Make requested changes in additional commits
- Once approved, we'll merge using "Squash and merge"

## 🏗️ Project Structure for Contributors

```
netconfigmaster/
├── src/                     # Main source code
├── metrics/                 # Custom evaluation metrics
├── data/                    # Datasets and configurations
├── tests/                   # Test files (add your tests here)
├── docs/                    # Documentation
├── scripts/                 # Development scripts
├── .devcontainer/           # VS Code Dev Container config
├── .github/                 # GitHub workflows and templates
├── pyproject.toml          # Poetry dependencies and config
├── docker-compose.yml      # Docker services
└── Dockerfile              # Multi-stage Docker build
```

## 🎓 Learning Resources

### Understanding the Project
- Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- Review [USAGE.md](docs/USAGE.md) for functionality
- Check existing issues for context

### Technical Background
- **Transformers**: [Hugging Face Documentation](https://huggingface.co/docs/transformers)
- **Nmstate**: [Official Documentation](https://nmstate.io/)
- **Network Configuration**: Understanding YAML-based network management

## 🤝 Community

### Communication
- **GitHub Issues**: Bug reports, feature requests, questions
- **Discussions**: General project discussion (if enabled)
- **Email**: [maintainer_email] for sensitive issues

### Code of Conduct
This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please read and follow it.

## 🏆 Recognition

Contributors will be:
- Added to the [Contributors](#contributors) section
- Mentioned in release notes
- Invited to join the maintainer team (for significant contributors)

## ❓ Questions?

- Check [existing issues](https://github.com/jona42-ui/netconfigmaster/issues)
- Read the [FAQ](docs/FAQ.md) (if available)
- Create a new issue with the "question" label
- Reach out to maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Happy Contributing!** 🎉

Your contributions make this project better for everyone. Thank you for being part of the NetConfigMaster community!
