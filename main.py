# asr-service/main.py
import os
import io
import json
import requests
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASR Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:2b")

# Global variables
whisper_model = None
ollama_available = False

class TranscriptionResponse(BaseModel):
    transcription: str
    enhanced_text: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    ollama_available: bool

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global whisper_model, ollama_available
    
    try:
        # Load Whisper model (using tiny for demo speed)
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("tiny")
        logger.info("Whisper model loaded successfully")
        
        # Check Ollama availability
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                ollama_available = True
                logger.info("Ollama service is available")
                
                # Try to pull the model if not available
                try:
                    pull_response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/pull",
                        json={"name": MODEL_NAME},
                        timeout=300
                    )
                    logger.info(f"Model {MODEL_NAME} pull initiated")
                except Exception as e:
                    logger.warning(f"Could not pull model: {e}")
            else:
                logger.warning("Ollama service not responding correctly")
        except Exception as e:
            logger.error(f"Could not connect to Ollama: {e}")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")

def transcribe_audio(audio_file):
    """Transcribe audio using Whisper"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file)
            tmp_file_path = tmp_file.name
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def enhance_text_with_ollama(text: str) -> str:
    """Enhance transcribed text using Ollama Gemma"""
    if not ollama_available:
        return text
    
    try:
        prompt = f"""Please improve and correct the following transcribed text. Fix grammar, punctuation, and spelling errors while maintaining the original meaning:

Text: "{text}"

Improved text:"""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_predict": 200
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            enhanced = result.get("response", text).strip()
            # Clean up the response (remove any prompt echoing)
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            return enhanced
        else:
            logger.warning(f"Ollama request failed: {response.status_code}")
            return text
            
    except Exception as e:
        logger.error(f"Text enhancement error: {e}")
        return text

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    """Main transcription endpoint"""
    import time
    start_time = time.time()
    
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    # Validate file type
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Transcribe
        logger.info(f"Transcribing audio file: {audio_file.filename}")
        transcription = transcribe_audio(audio_data)
        
        # Enhance with Ollama if available
        enhanced_text = enhance_text_with_ollama(transcription)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Transcription completed in {processing_time:.2f}s")
        
        return TranscriptionResponse(
            transcription=transcription,
            enhanced_text=enhanced_text,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Transcription endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        whisper_loaded=whisper_model is not None,
        ollama_available=ollama_available
    )

@app.get("/models")
async def get_available_models():
    """Get available Ollama models"""
    if not ollama_available:
        return {"error": "Ollama not available"}
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Could not fetch models"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)