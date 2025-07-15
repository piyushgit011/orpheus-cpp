from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import struct
import asyncio
import threading
import queue
import time
import json
from orpheus_tts import OrpheusModel
from typing import Optional

app = FastAPI(title="Orpheus TTS Streaming API", version="1.0.0")

# Initialize the model globally
model = None

class TTSRequest(BaseModel):
    prompt: str
    voice: Optional[str] = "tara"
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 2000
    repetition_penalty: Optional[float] = 1.1

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading Orpheus TTS model...")
    model = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
        max_model_len=2048
    )
    print("Model loaded successfully!")

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV header for streaming audio"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0  # Unknown size for streaming

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

async def generate_audio_stream(request: TTSRequest):
    """Generate streaming audio chunks"""
    try:
        # Generate speech tokens
        syn_tokens = model.generate_speech(
            prompt=request.prompt,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=[128258, 128009]
        )
        
        # Yield WAV header first
        yield create_wav_header()
        
        # Stream audio chunks
        for chunk in syn_tokens:
            yield chunk
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """Streaming TTS endpoint"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return StreamingResponse(
        generate_audio_stream(request),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/tts/stream")
async def tts_stream_get(
    prompt: str,
    voice: str = "tara",
    temperature: float = 0.6,
    top_p: float = 0.8,
    max_tokens: int = 2000,
    repetition_penalty: float = 1.1
):
    """GET endpoint for streaming TTS"""
    request = TTSRequest(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    return await tts_stream(request)

@app.post("/tts/wav")
async def tts_wav(request: TTSRequest):
    """Complete WAV file endpoint"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Collect all audio chunks
        audio_chunks = []
        syn_tokens = model.generate_speech(
            prompt=request.prompt,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=[128258, 128009]
        )
        
        for chunk in syn_tokens:
            audio_chunks.append(chunk)
        
        # Combine all chunks
        complete_audio = b''.join(audio_chunks)
        
        # Create proper WAV header with correct data size
        sample_rate = 24000
        bits_per_sample = 16
        channels = 1
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(complete_audio)
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        
        wav_data = header + complete_audio
        
        return StreamingResponse(
            iter([wav_data]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Content-Length": str(len(wav_data))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Orpheus TTS Streaming API",
        "endpoints": {
            "streaming": "/tts/stream",
            "wav": "/tts/wav",
            "health": "/health"
        },
        "available_voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)