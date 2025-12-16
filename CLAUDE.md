# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Unity ML-Agents Toolkit is a sophisticated machine learning infrastructure project that bridges Unity game development (C#) with Python-based ML training. It enables games and simulations to serve as environments for training intelligent agents using reinforcement learning, imitation learning, and other ML techniques.

This is a **monorepo** with multiple interconnected packages:
- `com.unity.ml-agents/` - Unity Package (C# SDK)
- `ml-agents/` - Python Trainer Package
- `ml-agents-envs/` - Python Environment Interface
- `ml-agents-trainer-plugin/` - Plugin framework
- `protobuf-definitions/` - Communication protocol definitions
- `config/` - Training configuration examples
- `Project/` and `DevProject/` - Example Unity projects

## Common Development Commands

### Python Package Installation (Development Mode)
```bash
# Install packages in editable mode for development
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
pip install -e ./ml-agents-trainer-plugin
pip install -e ./ml-agents-plugin-examples

# Install test dependencies
pip install -r test_requirements.txt
```

### Testing
```bash
# Run all tests except slow ones (default for development)
pytest -m "not slow"

# Run tests with parallel execution (8 workers)
pytest -m "not slow" -n 8

# Run tests with coverage reporting
pytest --cov=ml-agents --cov=ml-agents-envs --cov-report html

# Run slow tests (training-related, takes much longer)
pytest -m slow

# Run specific test file
pytest ml-agents/mlagents/trainers/tests/test_buffer.py

# Run specific test function
pytest ml-agents/mlagents/trainers/tests/test_buffer.py::test_buffer_functionality
```

### Code Quality and Formatting
```bash
# Run pre-commit hooks on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run mypy --all-files
pre-commit run flake8 --all-files

# Format C# code
dotnet format whitespace --folder --include

# Check markdown links (manual stage)
pre-commit run --hook-stage manual markdown-link-check
```

### Training Commands
```bash
# Basic training command
mlagents-learn config/ppo/3DBall.yaml --env=Project/3DBall --run-id=my_run

# Training with curriculum learning
mlagents-learn config/curricula/Wall.yaml --env=Project/WallJump --run-id=wall_curriculum

# Resume training from checkpoint
mlagents-learn config/ppo/3DBall.yaml --env=Project/3DBall --run-id=my_run --resume

# Push trained model to Hugging Face
mlagents-push-to-hf --run-id=my_run --repo-name=my-org/my-model
```

### Building and Unity Integration
```bash
# Unity Package Manager operations (from Unity Editor)
# Window > Package Manager > + > Add package from git URL
# https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents

# Build Unity project (from Unity Editor or command line)
Unity -batchmode -quit -projectPath ./Project -buildTarget StandaloneLinux64
```

## Architecture Overview

### Communication Flow
The system uses a **gRPC-based architecture**:
```
Unity Environment ←→ gRPC ←→ Python Trainer
    (C# SDK)                    (PyTorch/ML)
```

### Key Components

**Python Side:**
- **Trainers** (`ml-agents/mlagents/trainers/`): Core ML algorithms (PPO, SAC, MA-POCA, GAIL, BC)
- **Environment Interface** (`ml-agents-envs/`): Communication with Unity environments
- **Plugin System**: Extensible trainer types and stats writers via entry points

**Unity Side:**
- **Agent** (`com.unity.ml-agents/Runtime/Agent.cs`): Base class for trainable agents
- **Sensors**: Observation collection (vision, raycast, grid sensors)
- **Actuators**: Action execution (continuous/discrete)
- **Communicator**: gRPC communication layer

### Multi-Package Structure

The project uses **separate Python packages** with careful version management:
- Packages can be installed independently
- Version constraints are strictly managed (see `setup.py` files)
- Plugin system uses entry points for extensibility

## Development Workflow

### Code Quality Standards
- **Python**: Black formatting (120 char line length), mypy type checking, flake8 linting
- **C#**: dotnet-format for whitespace formatting
- **Testing**: pytest with 60% minimum coverage requirement
- **Pre-commit hooks**: Enforced via `.pre-commit-config.yaml`

### Important Constraints
- **Python Version**: 3.10.1-3.10.12 only
- **PyTorch**: ≤2.8.0 (strict constraint to prevent ONNX breaking changes)
- **Unity**: 6000.0+ (Unity 6)
- **Banned modules**: Direct tensorflow/torch imports (use utils instead)

### Testing Strategy
- **Parallel execution**: Tests run with 8 workers using pytest-xdist
- **Port allocation**: Custom `base_port` fixture prevents port collisions
- **Slow tests**: Training-related tests marked with `@pytest.mark.slow`
- **Coverage**: HTML reports available via `--cov-report html`

### Plugin Development
Extend functionality via entry points in `setup.py`:
```python
entry_points={
    ML_AGENTS_STATS_WRITER: ["custom_writer=my_package:MyStatsWriter"],
    ML_AGENTS_TRAINER_TYPE: ["custom_trainer=my_package:MyTrainer"]
}
```

## Key Files for Development

**Configuration:**
- `config/` - Training configuration YAML files for different algorithms
- `.pre-commit-config.yaml` - Code quality enforcement
- `pytest.ini`, `setup.cfg` - Test and coverage configuration

**Architecture Entry Points:**
- `ml-agents/mlagents/trainers/learn.py` - Main training script
- `ml-agents-envs/mlagents_envs/environment.py` - Environment interface
- `com.unity.ml-agents/Runtime/Agent.cs` - Unity agent base class

**Documentation:**
- Primary docs: [Unity Package Documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest)
- Legacy docs: `docs/` directory (deprecated)

## Release and Versioning

- **Main branch**: `develop` (unstable development)
- **Release branch pattern**: `release_XX`
- **Version validation**: Pre-commit hooks ensure version consistency across packages
- **Git tags**: Must match `__release_tag__` in `__init__.py` files

## Port Management

Tests use a **file-lock based port allocation system** (`conftest.py`) to enable parallel execution without port conflicts. Ports are dynamically allocated starting from a base port calculated per test file.