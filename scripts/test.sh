#!/bin/bash
# Test script for NetConfigMaster

set -e

echo "🧪 Running NetConfigMaster tests..."

# Run linting
echo "📝 Running linters..."
poetry run black --check src/ metrics/ || (echo "❌ Black formatting failed. Run 'poetry run black src/ metrics/' to fix." && exit 1)
poetry run isort --check-only src/ metrics/ || (echo "❌ Import sorting failed. Run 'poetry run isort src/ metrics/' to fix." && exit 1)
poetry run flake8 src/ metrics/ || (echo "❌ Flake8 linting failed." && exit 1)
poetry run pylint src/ metrics/ || (echo "❌ Pylint failed." && exit 1)

# Run type checking
echo "🔍 Running type checking..."
poetry run mypy src/ || (echo "❌ MyPy type checking failed." && exit 1)

# Run tests
echo "🧪 Running unit tests..."
poetry run pytest tests/ -v --cov=src --cov=metrics --cov-report=html --cov-report=term-missing

echo "✅ All tests passed!"
