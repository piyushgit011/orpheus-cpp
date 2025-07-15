"""
FIXED Optimized TTS Server - Resolves GPU, Segfault, and Concurrency Issues
"""
import asyncio
import json
import time
import threading
import gc
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, AsyncGenerator
import io
import wave
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Import torch first to preload CUDA libraries (critical for GPU support)
try:
    import torch
    print(f"✅ PyTorch loaded: CUDA available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
except ImportError:
    print("⚠️  PyTorch not available")

# Now import ONNX Runtime after PyTorch
import onnxruntime as ort

# Preload CUDA DLLs if available
try:
    ort.preload_dlls()
    print("✅ ONNX Runtime DLLs preloaded")
except:
    print("⚠️  ONNX Runtime DLL preload failed")

from orpheus_cpp import OrpheusCpp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice_id: str = "tara"
    language: str = "en"
    max_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    pre_buffer_size: float = 0.1

class ServerConfig:
    # GPU Optimization - More conservative settings
    GPU_LAYERS = -1  # Use all GPU layers if available
    VERBOSE_LOGGING = False
    N_THREADS = min(8, os.cpu_count() or 4)  # Reduce thread count
    
    # Check GPU availability
    GPU_AVAILABLE = False
    try:
        import torch
        GPU_AVAILABLE = torch.cuda.is_available()
    except:
        pass
    
    # ONNX Runtime Providers - Safer configuration
    if GPU_AVAILABLE:
        ONNX_PROVIDERS = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',  # More conservative
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit
                'cudnn_conv_algo_search': 'HEURISTIC',  # Faster than EXHAUSTIVE
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    else:
        ONNX_PROVIDERS = ['CPUExecutionProvider']
    
    # Performance settings
    CHUNK_SIZE = 2048  # Smaller chunks for better responsiveness
    FADE_SAMPLES = 240
    MAX_BUFFER_SIZE = 48000
    THREAD_POOL_SIZE = min(16, (os.cpu_count() or 1) * 2)  # Reduce pool size

# Global variables - Thread-safe approach
models_cache: Dict[str, OrpheusCpp] = {}
server_start_time = time.time()
active_requests = 0
request_lock = threading.Lock()  # Use threading.Lock instead of asyncio.Lock
thread_pool = ThreadPoolExecutor(max_workers=ServerConfig.THREAD_POOL_SIZE)

# Model loading lock to prevent concurrent model loading
model_load_lock = threading.Lock()

class SafeModelManager:
    """Thread-safe model manager"""
    
    @staticmethod
    def get_or_load_model(language: str) -> OrpheusCpp:
        with model_load_lock:
            if language not in models_cache:
                logger.info(f"Loading model for language: {language}")
                
                try:
                    # Conservative model initialization
                    model = OrpheusCpp(
                        lang=language,
                        n_gpu_layers=ServerConfig.GPU_LAYERS if ServerConfig.GPU_AVAILABLE else 0,
                        n_threads=ServerConfig.N_THREADS,
                        verbose=ServerConfig.VERBOSE_LOGGING
                    )
                    
                    # Quick warmup test
                    try:
                        _, _ = model.tts("test", options={
                            "voice_id": "tara",
                            "pre_buffer_size": 0.1,
                            "max_tokens": 50
                        })
                        logger.info(f"Model warmed up successfully: {language}")
                    except Exception as e:
                        logger.warning(f"Warmup failed: {e}")
                    
                    models_cache[language] = model
                    logger.info(f"Model cached: {language}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model {language}: {e}")
                    raise
            
            return models_cache[language]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    print("=" * 60)
    print("🚀 Starting FIXED TTS Server")
    print("=" * 60)
    
    # Check GPU status
    if ServerConfig.GPU_AVAILABLE:
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  Running on CPU only")
    
    # Check ONNX Runtime providers
    available_providers = ort.get_available_providers()
    print(f"🔧 ONNX Providers: {available_providers}")
    
    # Test CUDA provider
    if 'CUDAExecutionProvider' in available_providers:
        print("✅ CUDA Provider available")
    else:
        print("⚠️  CUDA Provider not available")
        ServerConfig.ONNX_PROVIDERS = ['CPUExecutionProvider']
    
    # Preload English model
    try:
        await asyncio.to_thread(SafeModelManager.get_or_load_model, "en")
        print("✅ English model preloaded")
    except Exception as e:
        print(f"⚠️  Model preload failed: {e}")
    
    print("✅ Server ready!")
    print("=" * 60)
    
    yield
    
    print("🛑 Shutting down...")
    models_cache.clear()
    thread_pool.shutdown(wait=True)
    gc.collect()

app = FastAPI(
    title="Fixed TTS Server",
    description="GPU-enabled TTS API with proper concurrency handling",
    version="2.1.0",
    lifespan=lifespan
)

# Safe streaming generator
async def safe_json_audio_stream(
    request: TTSRequest,
    model: OrpheusCpp
) -> AsyncGenerator[str, None]:
    """Safe streaming with proper error handling"""
    
    options = {
        "voice_id": request.voice_id,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "pre_buffer_size": request.pre_buffer_size,
    }
    
    chunk_count = 0
    start_time = time.time()
    audio_buffer = np.array([], dtype=np.int16)
    
    try:
        # Stream with smaller chunks for better responsiveness
        async for sample_rate, audio_chunk in model.stream_tts(request.text, options=options):
            audio_data = audio_chunk.flatten()
            audio_buffer = np.concatenate([audio_buffer, audio_data])
            
            # Send smaller chunks more frequently
            while len(audio_buffer) >= ServerConfig.CHUNK_SIZE:
                chunk_to_send = audio_buffer[:ServerConfig.CHUNK_SIZE]
                audio_buffer = audio_buffer[ServerConfig.CHUNK_SIZE:]
                
                chunk_data = {
                    "type": "audio_chunk",
                    "sample_rate": int(sample_rate),
                    "audio_data": chunk_to_send.tolist(),
                    "chunk_index": chunk_count,
                    "elapsed_ms": (time.time() - start_time) * 1000,
                    "buffer_remaining": len(audio_buffer)
                }
                
                yield json.dumps(chunk_data) + "\n"
                chunk_count += 1
        
        # Send remaining audio
        if len(audio_buffer) > 0:
            chunk_data = {
                "type": "audio_chunk",
                "sample_rate": int(sample_rate),
                "audio_data": audio_buffer.tolist(),
                "chunk_index": chunk_count,
                "elapsed_ms": (time.time() - start_time) * 1000,
                "final_chunk": True
            }
            yield json.dumps(chunk_data) + "\n"
        
        # End signal
        end_data = {
            "type": "stream_complete",
            "total_chunks": chunk_count + 1,
            "text": request.text,
            "total_time_ms": (time.time() - start_time) * 1000,
            "gpu_used": ServerConfig.GPU_AVAILABLE
        }
        yield json.dumps(end_data) + "\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "message": f"Streaming failed: {str(e)}",
            "chunk_count": chunk_count
        }
        yield json.dumps(error_data) + "\n"

@app.get("/health")
async def health_check():
    """Health check with GPU status"""
    global active_requests
    with request_lock:
        current_requests = active_requests
    
    # Check GPU memory if available
    gpu_info = {}
    if ServerConfig.GPU_AVAILABLE:
        try:
            import torch
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "cuda_version": torch.version.cuda
            }
        except:
            gpu_info = {"error": "GPU info unavailable"}
    
    return JSONResponse({
        "status": "healthy",
        "models_loaded": list(models_cache.keys()),
        "uptime_seconds": time.time() - server_start_time,
        "gpu_acceleration": ServerConfig.GPU_AVAILABLE,
        "onnx_providers": ort.get_available_providers(),
        "active_requests": current_requests,
        "thread_pool_size": ServerConfig.THREAD_POOL_SIZE,
        "gpu_info": gpu_info
    })

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Non-streaming TTS with proper concurrency handling"""
    global active_requests
    
    with request_lock:
        active_requests += 1
    
    start_time = time.time()
    
    try:
        # Load model in thread pool to avoid blocking
        model = await asyncio.to_thread(SafeModelManager.get_or_load_model, request.language)
        
        options = {
            "voice_id": request.voice_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "pre_buffer_size": request.pre_buffer_size,
        }
        
        # Run TTS in thread pool
        sample_rate, audio_array = await asyncio.get_event_loop().run_in_executor(
            thread_pool, 
            lambda: model.tts(request.text, options=options)
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "type": "tts_complete",
            "sample_rate": sample_rate,
            "audio_data": audio_array.flatten().tolist(),
            "duration_ms": duration_ms,
            "text": request.text,
            "gpu_used": ServerConfig.GPU_AVAILABLE,
            "audio_length_seconds": len(audio_array.flatten()) / sample_rate
        })
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    finally:
        with request_lock:
            active_requests -= 1

@app.post("/tts/stream")
async def stream_text_to_speech(request: TTSRequest):
    """Streaming TTS with fixed concurrency"""
    global active_requests
    
    with request_lock:
        active_requests += 1
    
    try:
        model = await asyncio.to_thread(SafeModelManager.get_or_load_model, request.language)
        
        return StreamingResponse(
            safe_json_audio_stream(request, model),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-GPU-Enabled": str(ServerConfig.GPU_AVAILABLE)
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")
    
    finally:
        with request_lock:
            active_requests -= 1

@app.get("/gpu-status")
async def gpu_status():
    """Detailed GPU status"""
    status = {
        "gpu_available": ServerConfig.GPU_AVAILABLE,
        "onnx_providers": ort.get_available_providers(),
        "cuda_provider_available": 'CUDAExecutionProvider' in ort.get_available_providers()
    }
    
    if ServerConfig.GPU_AVAILABLE:
        try:
            import torch
            status.update({
                "torch_cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
            })
        except Exception as e:
            status["gpu_error"] = str(e)
    
    return JSONResponse(status)

@app.get("/performance")
async def performance_metrics():
    """Performance metrics"""
    return JSONResponse({
        "uptime_seconds": time.time() - server_start_time,
        "active_requests": active_requests,
        "models_loaded": len(models_cache),
        "gpu_acceleration": ServerConfig.GPU_AVAILABLE,
        "optimization_settings": {
            "gpu_layers": ServerConfig.GPU_LAYERS,
            "threads": ServerConfig.N_THREADS,
            "chunk_size": ServerConfig.CHUNK_SIZE,
            "thread_pool_size": ServerConfig.THREAD_POOL_SIZE
        },
        "onnx_config": {
            "providers": ServerConfig.ONNX_PROVIDERS,
            "available_providers": ort.get_available_providers()
        }
    })

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FIXED TTS Server")
    print(f"🔧 GPU Available: {ServerConfig.GPU_AVAILABLE}")
    print(f"🔧 ONNX Providers: {ort.get_available_providers()}")
    print(f"🔧 Threads: {ServerConfig.N_THREADS}")
    print(f"🔧 Thread Pool: {ServerConfig.THREAD_POOL_SIZE}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
        access_log=False
    )