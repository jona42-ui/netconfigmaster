#!/bin/bash
# Format and lint script for NetConfigMaster

set -e

echo "🎨 Formatting and linting NetConfigMaster code..."

# Format with Black
echo "🖤 Running Black formatter..."
poetry run black src/ metrics/ tests/ scripts/

# Sort imports with isort
echo "📋 Sorting imports with isort..."
poetry run isort src/ metrics/ tests/

# Run linting
echo "📝 Running linters..."
poetry run flake8 src/ metrics/ tests/ || true
poetry run pylint src/ metrics/ || true

# Run type checking
echo "🔍 Running type checking..."
poetry run mypy src/ || true

echo "✅ Code formatting and linting complete!"
