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

app = FastAPI(title="ASR/AST Service", version="1.0.0")

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
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3n:2b")

# Global variables
whisper_model = None
ollama_available = False

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch"
}

class TranscriptionResponse(BaseModel):
    transcription: str
    enhanced_text: str
    processing_time: float

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    processing_time: float

class ASRTranslationResponse(BaseModel):
    transcription: str
    enhanced_text: str
    translated_text: str
    target_language: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    ollama_available: bool

class SupportedLanguagesResponse(BaseModel):
    languages: dict

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

def translate_text_with_ollama(text: str, target_language: str, source_language: str = "auto") -> str:
    """Translate text using Ollama Gemma"""
    if not ollama_available:
        return text
    
    try:
        # Get full language names
        target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        source_lang_name = SUPPORTED_LANGUAGES.get(source_language, "auto-detected language") if source_language != "auto" else "auto-detected language"
        
        prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}. 
Provide only the translation without any explanations or additional text.

Text to translate: "{text}"

Translation in {target_lang_name}:"""

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
                    "num_predict": 300
                }
            },
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            translated = result.get("response", text).strip()
            
            # Clean up the response (remove any prompt echoing or quotes)
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            
            # Remove common prefixes that might appear
            prefixes_to_remove = [
                "Translation:", "Translation in", f"{target_lang_name}:", 
                "Here is the translation:", "The translation is:"
            ]
            
            for prefix in prefixes_to_remove:
                if translated.lower().startswith(prefix.lower()):
                    translated = translated[len(prefix):].strip()
                    if translated.startswith(':'):
                        translated = translated[1:].strip()
            
            return translated
        else:
            logger.warning(f"Translation request failed: {response.status_code}")
            return text
            
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text
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

@app.post("/translate", response_model=TranslationResponse)
async def translate_text_endpoint(
    text: str,
    target_language: str,
    source_language: str = "auto"
):
    """Translate text endpoint"""
    import time
    start_time = time.time()
    
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")
    
    try:
        translated_text = translate_text_with_ollama(text, target_language, source_language)
        processing_time = time.time() - start_time
        
        logger.info(f"Translation completed in {processing_time:.2f}s")
        
        return TranslationResponse(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Translation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-and-translate", response_model=ASRTranslationResponse)
async def transcribe_and_translate_endpoint(
    audio_file: UploadFile = File(...),
    target_language: str = "en"
):
    """Combined ASR + AST endpoint"""
    import time
    start_time = time.time()
    
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")
    
    # Validate file type
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Transcribe
        logger.info(f"Transcribing and translating audio file: {audio_file.filename}")
        transcription = transcribe_audio(audio_data)
        
        # Enhance with Ollama
        enhanced_text = enhance_text_with_ollama(transcription)
        
        # Translate enhanced text
        translated_text = translate_text_with_ollama(enhanced_text, target_language)
        
        processing_time = time.time() - start_time
        
        logger.info(f"ASR+AST completed in {processing_time:.2f}s")
        
        return ASRTranslationResponse(
            transcription=transcription,
            enhanced_text=enhanced_text,
            translated_text=translated_text,
            target_language=target_language,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"ASR+AST endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """Get supported languages for translation"""
    return SupportedLanguagesResponse(languages=SUPPORTED_LANGUAGES)

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