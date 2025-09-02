#!/bin/bash
# startup.sh

echo "🚀 Starting ASR Demo with Ollama Gemma"
echo "======================================"

# Create ASR service files
echo "🔧 Setting up ASR service..."
mkdir -p asr-service/uploads

# Create web-ui files  
echo "🌐 Setting up Web UI..."

# Make the script executable and run setup
chmod +x startup.sh

echo "✅ Project structure created!"
echo ""
echo "📋 Next steps:"
echo "1. Copy all the files to their respective directories:"
echo "   - docker-compose.yml -> ./"
echo "   - ASR service files -> ./asr-service/"
echo "   - Web UI files -> ./web-ui/"
echo ""
echo "2. Navigate to the project directory:"
echo "   cd asr-demo"
echo ""
echo "3. Start the services:"
echo "   docker-compose up --build -d"
echo ""
echo "4. Wait for services to be ready (may take a few minutes for first run)"
echo ""
echo "5. Access the demo at:"
echo "   http://localhost:3000"
echo ""
echo "🔍 Monitor services:"
echo "   docker-compose logs -f"
echo ""
echo "📊 Check service health:"
echo "   curl http://localhost:8000/health"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build and start services
start_services() {
    echo "🏗️  Building and starting services..."
    
    # Pull base images first
    echo "📥 Pulling base images..."
    docker pull python:3.11-slim
    docker pull nginx:alpine
    docker pull ollama/ollama:latest
    
    # Build and start services
    docker compose up --build -d
    
    echo "⏳ Waiting for services to start..."
    sleep 10
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama service..."
    timeout=60
    count=0
    while [ $count -lt $timeout ]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama service is ready"
            break
        fi
        echo "⏳ Ollama starting... ($count/$timeout)"
        sleep 2
        count=$((count + 2))
    done
    
    # Pull Gemma model
    echo "📦 Pulling Gemma model (this may take a while)..."
    curl -s -X POST http://localhost:11434/api/pull -d '{"name":"gemma2:2b"}' &
    
    echo "🎉 Services started! Check status with:"
    echo "   docker compose ps"
    echo ""
    echo "🌐 Access the demo at: http://localhost:3000"
}

# Main execution
if [ "$1" = "start" ]; then
    check_docker
    start_services
elif [ "$1" = "stop" ]; then
    echo "🛑 Stopping services..."
    docker-compose down
elif [ "$1" = "logs" ]; then
    docker-compose logs -f
elif [ "$1" = "restart" ]; then
    echo "🔄 Restarting services..."
    docker-compose restart
else
    echo "Usage: $0 {start|stop|logs|restart}"
    echo "  start   - Start all services"
    echo "  stop    - Stop all services" 
    echo "  logs    - Show service logs"
    echo "  restart - Restart services"
fi