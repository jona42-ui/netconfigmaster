#!/bin/bash
# Docker build script for NetConfigMaster

set -e

echo "🐳 Building NetConfigMaster Docker images..."

# Build all stages
echo "🔨 Building development image..."
docker build --target development -t netconfigmaster:dev .

echo "🚀 Building production image..."
docker build --target production -t netconfigmaster:prod .

echo "🎓 Building training image..."
docker build --target training -t netconfigmaster:train .

echo "📊 Building evaluation image..."
docker build --target evaluation -t netconfigmaster:eval .

echo "✅ All Docker images built successfully!"
echo ""
echo "🎯 Available images:"
echo "  netconfigmaster:dev    # Development environment"
echo "  netconfigmaster:prod   # Production web server"
echo "  netconfigmaster:train  # Training environment"
echo "  netconfigmaster:eval   # Evaluation environment"
