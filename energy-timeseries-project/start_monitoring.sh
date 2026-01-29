#!/bin/bash
# Startup script for Energy Forecasting API + Monitoring Stack
# Automatically starts the complete monitoring system in Codespace

set -e

echo "🚀 Starting Energy Forecasting API with Monitoring Stack..."
echo "=================================================="

# Change to project directory
cd "$(dirname "$0")" || exit 1

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Clean up old containers if they exist
echo "🧹 Cleaning up previous containers..."
docker compose --profile monitoring down -v 2>/dev/null || true

# Start the monitoring stack
echo "📦 Building and starting containers..."
docker compose --profile monitoring up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if all services are running
echo "🔍 Checking service health..."
docker compose --profile monitoring ps

# Get port information
API_PORT=$(docker compose --profile monitoring ps api | grep -oP '8000' | head -1)
GRAFANA_PORT=$(docker compose --profile monitoring ps grafana | grep -oP '3000' | head -1)
PROMETHEUS_PORT=$(docker compose --profile monitoring ps prometheus | grep -oP '9090' | head -1)

if [ -z "$API_PORT" ]; then
    echo "❌ API failed to start"
    docker compose --profile monitoring logs api | tail -20
    exit 1
fi

if [ -z "$GRAFANA_PORT" ]; then
    echo "❌ Grafana failed to start"
    docker compose --profile monitoring logs grafana | tail -20
    exit 1
fi

echo ""
echo "✅ All services started successfully!"
echo ""
echo "=================================================="
echo "📊 Access URLs (for Codespace port forwarding):"
echo "=================================================="
echo ""
echo "  🌐 API Dashboard:"
echo "     http://localhost:8000/ui"
echo ""
echo "  📈 Grafana Monitoring:"
echo "     http://localhost:3000 (admin/admin)"
echo ""
echo "  🔧 Prometheus Metrics:"
echo "     http://localhost:9090"
echo ""
echo "  📚 API Docs:"
echo "     http://localhost:8000/docs"
echo ""
echo "=================================================="
echo "❗ NEXT STEPS FOR CODESPACE:"
echo "=================================================="
echo ""
echo "1. Open 'PORTS' panel in VS Code"
echo "2. Make ports 8000, 3000, 9090 PUBLIC (click the eye icon)"
echo "3. Click on the 3000 port → 'Open in Browser'"
echo "4. Login to Grafana: admin / admin"
echo ""
echo "=================================================="
echo "📖 Dashboard Documentation:"
echo "=================================================="
echo ""
echo "Read the chart explanations in:"
echo "  👉 docs/GRAFANA_DASHBOARD_GUIDE_DE.md"
echo ""
echo "=================================================="
echo ""
echo "✨ System is ready! The API will start generating"
echo "   predictions automatically. Grafana will display"
echo "   metrics in real-time."
echo ""
