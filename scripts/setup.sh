#!/bin/bash
# Development setup script for NetConfigMaster

set -e

echo "🚀 Setting up NetConfigMaster development environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install --with dev,docs

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
poetry run pre-commit install

# Create directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed models logs

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose build

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  poetry shell                    # Activate virtual environment"
echo "  poetry run python src/ui.py    # Start web UI"
echo "  docker-compose up dev          # Start development container"
echo "  poetry run pytest              # Run tests"
echo "  poetry run black src/ metrics/ # Format code"
echo "  poetry run pylint src/ metrics/# Lint code"
echo ""
echo "📖 For more information, see docs/USAGE.md"
