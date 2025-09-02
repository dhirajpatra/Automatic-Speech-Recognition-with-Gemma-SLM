# asr-service/main.py
import os
import tempfile
import logging
import requests
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch"
}

# ----------------------------
# Response Models
# ----------------------------
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

# ----------------------------
# Startup Initialization
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global whisper_model, ollama_available
    try:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("tiny")
        logger.info("Whisper model loaded successfully")

        # Check Ollama availability
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                ollama_available = True
                logger.info("Ollama service is available")
                # Attempt to pull model
                try:
                    requests.post(
                        f"{OLLAMA_BASE_URL}/api/pull",
                        json={"name": MODEL_NAME},
                        timeout=300
                    )
                    logger.info(f"Model {MODEL_NAME} pull initiated")
                except Exception as e:
                    logger.warning(f"Could not pull model: {e}")
            else:
                logger.warning("Ollama service responded unexpectedly")
        except Exception as e:
            logger.error(f"Could not connect to Ollama: {e}")

    except Exception as e:
        logger.error(f"Startup error: {e}")

# ----------------------------
# Utility Functions
# ----------------------------
def transcribe_audio(audio_file: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file)
            tmp_file_path = tmp_file.name

        result = whisper_model.transcribe(tmp_file_path)
        os.unlink(tmp_file_path)

        return result["text"].strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def enhance_text_with_ollama(text: str) -> str:
    if not ollama_available:
        return text
    try:
        prompt = f"""Please improve and correct the following transcribed text. 
Fix grammar, punctuation, and spelling errors while maintaining the original meaning:

Text: "{text}"

Improved text:"""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_k": 40, "top_p": 0.9, "num_predict": 200}
            },
            timeout=30
        )
        if response.status_code == 200:
            enhanced = response.json().get("response", text).strip()
            return enhanced.strip('"')
        else:
            logger.warning(f"Ollama enhancement failed: {response.status_code}")
            return text
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return text

def translate_text_with_ollama(text: str, target_language: str, source_language: str = "auto") -> str:
    if not ollama_available:
        return text
    try:
        target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        source_lang_name = SUPPORTED_LANGUAGES.get(source_language, "auto") if source_language != "auto" else "auto-detected language"

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
                "options": {"temperature": 0.3, "top_k": 40, "top_p": 0.9, "num_predict": 300}
            },
            timeout=45
        )
        if response.status_code == 200:
            translated = response.json().get("response", text).strip()
            for prefix in ["Translation:", "Translation in", f"{target_lang_name}:", "Here is the translation:", "The translation is:"]:
                if translated.lower().startswith(prefix.lower()):
                    translated = translated[len(prefix):].strip().lstrip(":").strip()
            return translated.strip('"')
        else:
            logger.warning(f"Ollama translation failed: {response.status_code}")
            return text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    import time
    start_time = time.time()
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio")

    audio_data = await audio_file.read()
    transcription = transcribe_audio(audio_data)
    enhanced_text = enhance_text_with_ollama(transcription)
    processing_time = time.time() - start_time

    return TranscriptionResponse(transcription=transcription, enhanced_text=enhanced_text, processing_time=processing_time)

@app.post("/translate", response_model=TranslationResponse)
async def translate_text_endpoint(text: str, target_language: str, source_language: str = "auto"):
    import time
    start_time = time.time()
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {target_language}")

    translated_text = translate_text_with_ollama(text, target_language, source_language)
    processing_time = time.time() - start_time

    return TranslationResponse(original_text=text, translated_text=translated_text, source_language=source_language, target_language=target_language, processing_time=processing_time)

@app.post("/transcribe-and-translate", response_model=ASRTranslationResponse)
async def transcribe_and_translate_endpoint(audio_file: UploadFile = File(...), target_language: str = "en"):
    import time
    start_time = time.time()
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {target_language}")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio")

    audio_data = await audio_file.read()
    transcription = transcribe_audio(audio_data)
    enhanced_text = enhance_text_with_ollama(transcription)
    translated_text = translate_text_with_ollama(enhanced_text, target_language)
    processing_time = time.time() - start_time

    return ASRTranslationResponse(
        transcription=transcription,
        enhanced_text=enhanced_text,
        translated_text=translated_text,
        target_language=target_language,
        processing_time=processing_time
    )

@app.get("/supported-languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    return SupportedLanguagesResponse(languages=SUPPORTED_LANGUAGES)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", whisper_loaded=whisper_model is not None, ollama_available=ollama_available)

@app.get("/models")
async def get_available_models():
    if not ollama_available:
        return {"error": "Ollama not available"}
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        return response.json() if response.status_code == 200 else {"error": "Could not fetch models"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
