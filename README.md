# ASR Demo - Automatic Speech Recognition

A lightweight microservice demo for Automatic Speech Recognition using **Whisper** for transcription and **Ollama Gemma** for text enhancement.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │    │   ASR Service    │    │   Ollama        │
│   (Nginx)       │────│   (FastAPI)      │────│   (Gemma 2:2B)  │
│   Port: 3000    │    │   Port: 8000     │    │   Port: 11434   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- At least 4GB RAM (for Gemma model)
- Modern web browser with microphone access

### 1. Clone and Setup
```bash
git clone <repository-url>
cd asr-demo

# Make startup script executable
chmod +x startup.sh
```

### 2. Project Structure
```
asr-demo/
├── docker-compose.yml
├── startup.sh
├── asr-service/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── web-ui/
    ├── Dockerfile
    ├── index.html
    └── nginx.conf
```

### 3. Start Services
```bash
# Quick start
./startup.sh start

# Or manually
docker-compose up --build -d
```

### 4. Access Demo
Open your browser and navigate to: **http://localhost:3000**

## 📊 Service Endpoints

### ASR Service (Port 8000)
- `POST /transcribe` - Upload audio file for transcription
- `GET /health` - Service health check
- `GET /models` - Available Ollama models

### Web UI (Port 3000)
- Interactive interface for audio recording/upload
- Real-time transcription results
- Service status monitoring

## 🎯 Features

### Core Features
- ✅ **Audio Recording** - Record directly in browser
- ✅ **File Upload** - Support for MP3, WAV, M4A, OGG
- ✅ **Drag & Drop** - Easy file handling
- ✅ **Real-time Processing** - Live transcription
- ✅ **Text Enhancement** - AI-powered text correction

### Technical Features
- ✅ **Microservice Architecture** - Scalable design
- ✅ **Docker Containerization** - Easy deployment
- ✅ **Health Monitoring** - Service status checks
- ✅ **CORS Support** - Cross-origin requests
- ✅ **Error Handling** - Robust error management

## 🔧 Configuration

### Environment Variables

**ASR Service:**
```env
OLLAMA_BASE_URL=http://ollama:11434
MODEL_NAME=gemma3n:2b
```

**Docker Compose:**
```yaml
services:
  ollama:
    environment:
      - OLLAMA_KEEP_ALIVE=24h
  
  asr-service:
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - MODEL_NAME=gemma2:2b
```

### Model Configuration
- **Whisper Model**: `tiny` (fastest, good for demo)
- **Ollama Model**: `gemma2:2b` (2B parameters, balance of speed/quality)

To use different models:
```bash
# Larger Whisper model (better accuracy)
# Edit main.py: whisper.load_model("base")  # or "small", "medium", "large"

# Different Ollama model
# Edit docker-compose.yml: MODEL_NAME=llama2:7b
```

## 📋 Usage Examples

### 1. Record Audio
1. Click "Start Recording"
2. Speak clearly into microphone
3. Click "Stop Recording"
4. Click "Transcribe Audio"

### 2. Upload Audio File
1. Drag and drop audio file or click "Choose File"
2. Select audio file (MP3, WAV, M4A, OGG)
3. Click "Transcribe Audio"

### 3. API Usage
```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio file
curl -X POST http://localhost:8000/transcribe \
  -F "audio_file=@recording.wav" \
  -H "Content-Type: multipart/form-data"
```

## 🛠️ Development

### Local Development
```bash
# Start individual services for development
cd asr-service
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# In another terminal
cd web-ui
python -m http.server 8080
```

### Debugging
```bash
# View logs
docker-compose logs -f

# Check specific service
docker-compose logs asr-service
docker-compose logs ollama

# Restart specific service
docker-compose restart asr-service
```

### Testing
```bash
# Test ASR service health
curl http://localhost:8000/health

# Test Ollama
curl http://localhost:11434/api/tags

# Test transcription
curl -X POST http://localhost:8000/transcribe \
  -F "audio_file=@test.wav"
```

## 📈 Performance & Scaling

### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: ~2GB for models

### Optimization Tips
1. **Model Selection**: Use smaller models for faster processing
2. **Audio Format**: WAV files process fastest
3. **Audio Length**: Keep recordings under 2 minutes for best performance
4. **Concurrent Requests**: Limit to 2-3 simultaneous transcriptions

### Scaling Options
```yaml
# Scale ASR service
docker-compose up --scale asr-service=3

# Use GPU acceleration (if available)
# Add to docker-compose.yml:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔍 Troubleshooting

### Common Issues

**Ollama not responding:**
```bash
# Check Ollama status
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama

# Pull model manually
curl -X POST http://localhost:11434/api/pull -d '{"name":"gemma2:2b"}'
```

**Audio processing errors:**
```bash
# Check audio format
ffmpeg -i input.mp3 -f wav output.wav

# Verify file permissions
ls -la uploads/
```

**Memory issues:**
```bash
# Monitor resource usage
docker stats

# Reduce model size
# Change MODEL_NAME to smaller model
```

**Service connectivity:**
```bash
# Test service connectivity
docker-compose exec asr-service curl http://ollama:11434/api/tags

# Check network
docker network ls
docker network inspect asr-demo_default
```

## 📚 API Reference

### Transcription Response
```json
{
  "transcription": "Raw transcription from Whisper",
  "enhanced_text": "Enhanced text from Ollama Gemma", 
  "processing_time": 2.45
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "ollama_available": true
}
```

## 🛡️ Security Notes

- This is a demo application - not production ready
- No authentication/authorization implemented
- File uploads are not validated extensively
- Consider adding rate limiting for production use

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests if applicable
5. Submit pull request

## 📞 Support

For issues and questions:
- Check the logs: `docker-compose logs -f`
- Review common issues in troubleshooting section
- Submit GitHub issues for bugs/features