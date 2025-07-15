"""
Completely Clean FastAPI Server - NO ARGS/KWARGS ANYWHERE

**server.py**
"""
import asyncio
import json
import time
import threading
from contextlib import asynccontextmanager
from typing import Dict, List
import gc
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orpheus_cpp import OrpheusCpp


class TTSRequest(BaseModel):
    text: str
    voice_id: str = "tara"
    language: str = "en"
    max_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    pre_buffer_size: float = 0.2


# Global variables
models_cache: Dict[str, OrpheusCpp] = {}
server_start_time = time.time()
active_requests = 0
request_lock = threading.Lock()

# GPU Configuration
GPU_LAYERS = -1
VERBOSE_LOGGING = False
N_THREADS = min(16, os.cpu_count() or 8)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global models_cache
    
    print("=" * 60)
    print("🚀 Starting Clean TTS Server")
    print("=" * 60)
    
    print(f"🔧 Loading English model...")
    print(f"   GPU Layers: {GPU_LAYERS}")
    print(f"   Threads: {N_THREADS}")
    
    models_cache["en"] = OrpheusCpp(
        lang="en",
        n_gpu_layers=GPU_LAYERS,
        n_threads=N_THREADS,
        verbose=VERBOSE_LOGGING
    )
    
    print("✅ Server ready!")
    print("=" * 60)
    
    yield
    
    print("🛑 Shutting down...")
    models_cache.clear()
    gc.collect()


app = FastAPI(
    title="Clean TTS Server",
    description="TTS API with only JSON bodies",
    version="1.0.0",
    lifespan=lifespan
)


def get_or_load_model(language: str) -> OrpheusCpp:
    """Get or load model"""
    if language not in models_cache:
        print(f"🔄 Loading model: {language}")
        models_cache[language] = OrpheusCpp(
            lang=language,
            n_gpu_layers=GPU_LAYERS,
            n_threads=N_THREADS,
            verbose=VERBOSE_LOGGING
        )
        print(f"✅ Model loaded: {language}")
    return models_cache[language]


@app.get("/health")
async def health_check():
    """Health check"""
    global active_requests
    with request_lock:
        current_requests = active_requests
    
    return {
        "status": "healthy",
        "models_loaded": list(models_cache.keys()),
        "uptime_seconds": time.time() - server_start_time,
        "gpu_acceleration": GPU_LAYERS > 0,
        "gpu_layers": GPU_LAYERS,
        "active_requests": current_requests
    }


@app.get("/voices")
async def get_voices():
    """Get available voices"""
    voices = {
        "en": ["tara", "jess", "leah", "leo", "dan", "mia", "zac", "zoe"],
        "es": ["javi", "sergio", "maria"],
        "fr": ["pierre", "amelie", "marie"],
        "de": ["jana", "thomas", "max"],
        "it": ["pietro", "giulia", "carlo"],
        "zh": ["长乐", "白芷"],
        "ko": ["유나", "준서"],
        "hi": ["ऋतिका"],
    }
    return {"voices": voices}


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Non-streaming TTS"""
    global active_requests
    
    with request_lock:
        active_requests += 1
    
    start_time = time.time()
    
    try:
        model = get_or_load_model(request.language)
        
        options = {
            "voice_id": request.voice_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "pre_buffer_size": request.pre_buffer_size,
        }
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        sample_rate, audio_array = await loop.run_in_executor(
            None, lambda: model.tts(request.text, options=options)
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "sample_rate": sample_rate,
            "audio_data": audio_array.flatten().tolist(),
            "duration_ms": duration_ms,
            "text": request.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    finally:
        with request_lock:
            active_requests -= 1


@app.post("/tts/stream")
async def stream_text_to_speech(request: TTSRequest):
    """Streaming TTS"""
    global active_requests
    
    with request_lock:
        active_requests += 1
    
    try:
        model = get_or_load_model(request.language)
        
        options = {
            "voice_id": request.voice_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "pre_buffer_size": request.pre_buffer_size,
        }
        
        async def generate_audio_stream():
            chunk_count = 0
            start_time = time.time()
            
            try:
                async for sample_rate, audio_chunk in model.stream_tts(request.text, options=options):
                    chunk_data = {
                        "sample_rate": int(sample_rate),
                        "audio_data": audio_chunk.flatten().tolist(),
                        "chunk_index": chunk_count,
                        "elapsed_ms": (time.time() - start_time) * 1000
                    }
                    
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    chunk_count += 1
                
                # End signal
                end_data = {
                    "end": True,
                    "total_chunks": chunk_count,
                    "text": request.text,
                    "total_time_ms": (time.time() - start_time) * 1000
                }
                yield f"data: {json.dumps(end_data)}\n\n"
                
            except Exception as e:
                error_data = {
                    "error": True,
                    "message": str(e),
                    "chunk_count": chunk_count
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")
    
    finally:
        with request_lock:
            active_requests -= 1


@app.post("/tts/wav")
async def text_to_speech_wav(request: TTSRequest):
    """WAV TTS"""
    global active_requests
    
    with request_lock:
        active_requests += 1
    
    try:
        model = get_or_load_model(request.language)
        
        options = {
            "voice_id": request.voice_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "pre_buffer_size": request.pre_buffer_size,
        }
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        sample_rate, audio_array = await loop.run_in_executor(
            None, lambda: model.tts(request.text, options=options)
        )
        
        # Convert to WAV
        import io
        import wave
        
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        
        wav_io.seek(0)
        wav_content = wav_io.read()
        
        def generate_wav():
            yield wav_content
        
        return StreamingResponse(
            generate_wav(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "Content-Length": str(len(wav_content))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WAV failed: {str(e)}")
    
    finally:
        with request_lock:
            active_requests -= 1


@app.get("/gpu-info")
async def gpu_info():
    """GPU information"""
    info = {
        "gpu_layers": GPU_LAYERS,
        "gpu_acceleration_enabled": GPU_LAYERS > 0,
        "threads": N_THREADS,
        "active_requests": active_requests,
        "models_loaded": len(models_cache),
        "pre_buffer_size": 0.2
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB",
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated(0) / (1024**3):.2f}GB",
                "cuda_memory_cached": f"{torch.cuda.memory_reserved(0) / (1024**3):.2f}GB"
            })
        else:
            info.update({"cuda_available": False})
    except ImportError:
        info.update({"torch_available": False})
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        info.update({
            "onnxruntime_providers": providers,
            "tensorrt_available": "TensorrtExecutionProvider" in providers,
            "cuda_provider_available": "CUDAExecutionProvider" in providers
        })
    except ImportError:
        info.update({"onnxruntime_available": False})
    
    return info


@app.get("/performance")
async def performance_stats():
    """Performance statistics"""
    return {
        "uptime_seconds": time.time() - server_start_time,
        "active_requests": active_requests,
        "models_loaded": len(models_cache),
        "optimization_settings": {
            "gpu_layers": GPU_LAYERS,
            "threads": N_THREADS,
            "pre_buffer_size": 0.2,
            "verbose_logging": VERBOSE_LOGGING
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting COMPLETELY CLEAN TTS Server")
    print(f"🔧 GPU Layers: {GPU_LAYERS}")
    print(f"🔧 Threads: {N_THREADS}")
    print("🔧 NO ARGS/KWARGS ANYWHERE!")
    
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("✅ Using uvloop")
    except ImportError:
        print("⚠️  No uvloop")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
        access_log=False,
        server_header=False
    )