"""
FastAPI Server for Orpheus TTS with Streaming Endpoint
Provides REST API with server-sent events (SSE) for real-time audio streaming
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import asyncio
import numpy as np
from ultra_fast_client import UltraFastOrpheusClient
import base64
from datetime import datetime
import json
import logging
import io
from scipy.io import wavfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orpheus TTS API",
    description="High-performance text-to-speech API with streaming support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global client instance (reused across requests)
global_client = None

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=10000)
    voice: str = Field("tara", description="Voice to use for TTS")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(256, ge=32, le=2048, description="Maximum tokens to generate per chunk")
    stream: bool = Field(True, description="Enable streaming response")
    chunk_size: int = Field(50, ge=30, le=100, description="Tokens per chunk for streaming")

class TTSResponse(BaseModel):
    audio_base64: str
    duration_seconds: float
    sample_rate: int = 24000
    processing_time_ms: float
    ttfa_ms: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Initialize global client on startup"""
    global global_client
    global_client = UltraFastOrpheusClient()
    await global_client.__aenter__()
    logger.info("FastAPI server started, Orpheus client initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global global_client
    if global_client:
        await global_client.__aexit__(None, None, None)
    logger.info("FastAPI server shutdown")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Orpheus TTS API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "tts": "/api/tts (base64 JSON response)",
            "tts_wav": "/api/tts/wav (WAV file download)",
            "tts_stream": "/api/tts/stream (SSE streaming - requires high context)",
            "tts_chunked_stream": "/api/tts/chunked-stream (SSE streaming - works with limited context, supports very long text)",
            "tts_simple_stream": "/api/tts/simple-stream (SSE streaming - simple implementation)",
            "voices": "/api/voices",
            "health": "/health"
        },
        "recommended": "Use /api/tts/chunked-stream for long text with limited context"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "client_initialized": global_client is not None
    }

@app.post("/api/tts", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text (non-streaming, returns complete audio)
    
    Returns base64-encoded audio data
    """
    if not global_client:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    if not request.stream:
        # Non-streaming mode: collect all audio first
        try:
            start_time = asyncio.get_event_loop().time()
            all_audio = []
            ttfa = None
            
            # Adjust max_tokens if text is very long
            adjusted_max_tokens = request.max_tokens
            if len(request.text) > 200:
                # Reduce tokens for longer text to fit context
                adjusted_max_tokens = min(request.max_tokens, 200)
                logger.info(f"Adjusted max_tokens to {adjusted_max_tokens} for long text ({len(request.text)} chars)")
            
            async for audio_chunk, info in global_client.stream_generate(
                text=request.text,
                voice=request.voice,
                temperature=request.temperature,
                max_tokens=adjusted_max_tokens
            ):
                if ttfa is None and info.get('ttfa_ms'):
                    ttfa = info['ttfa_ms']
                all_audio.append(audio_chunk)
            
            if not all_audio:
                raise HTTPException(status_code=500, detail="No audio generated")
            
            # Concatenate all chunks
            full_audio = np.concatenate(all_audio)
            audio_int16 = (full_audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            duration = len(full_audio) / 24000
            
            logger.info(f"TTS completed: {len(request.text)} chars, {duration:.2f}s audio, {processing_time:.1f}ms")
            
            return TTSResponse(
                audio_base64=audio_base64,
                duration_seconds=duration,
                processing_time_ms=processing_time,
                ttfa_ms=ttfa
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"TTS error: {error_msg}", exc_info=True)
            
            # Check if it's a context size error
            if "exceed_context_size_error" in error_msg or "context size" in error_msg.lower():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Text too long for current server configuration. Try shorter text (<150 chars) or increase server context size. Error: {error_msg[:200]}"
                )
            
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {error_msg[:200]}")
    else:
        raise HTTPException(status_code=400, detail="Use /api/tts/stream for streaming mode")

@app.post("/api/tts/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    Synthesize speech with streaming (Server-Sent Events)
    
    Streams audio chunks as they are generated for minimal latency
    Returns newline-delimited JSON events
    
    Note: For limited context servers, use shorter text (<100 chars) and lower max_tokens
    """
    if not global_client:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    # Limit text length based on context
    if len(request.text) > 200:
        request.text = request.text[:200]
        logger.warning(f"Text truncated to 200 chars due to context limits")
    
    # Reduce max_tokens if needed
    if request.max_tokens > 256:
        request.max_tokens = 256
    
    async def generate_audio_stream() -> AsyncGenerator[str, None]:
        """Generate SSE stream of audio chunks"""
        try:
            start_time = asyncio.get_event_loop().time()
            chunk_count = 0
            total_samples = 0
            ttfa_sent = False
            
            async for audio_chunk, info in global_client.stream_generate(
                text=request.text,
                voice=request.voice,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                chunk_count += 1
                total_samples += len(audio_chunk)
                
                # Convert to int16 and base64
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Create SSE event
                event_data = {
                    "chunk": chunk_count,
                    "audio_base64": audio_base64,
                    "samples": len(audio_chunk),
                    "duration_ms": (len(audio_chunk) / 24000) * 1000,
                    "ttfa_ms": info.get('ttfa_ms') if not ttfa_sent else None,
                    "is_final": info.get('is_final', False)
                }
                
                if info.get('ttfa_ms') and not ttfa_sent:
                    ttfa_sent = True
                
                # Send as SSE format
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Flush after each chunk to ensure immediate delivery
                await asyncio.sleep(0)
            
            # Send completion event
            total_time = (asyncio.get_event_loop().time() - start_time) * 1000
            completion_data = {
                "event": "complete",
                "total_chunks": chunk_count,
                "total_samples": total_samples,
                "total_duration_s": total_samples / 24000,
                "processing_time_ms": total_time
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
            logger.info(f"TTS stream completed: {chunk_count} chunks, {total_samples/24000:.2f}s audio, {total_time:.1f}ms")
            
        except asyncio.CancelledError:
            logger.warning("TTS stream cancelled by client")
            raise
        except Exception as e:
            logger.error(f"TTS stream error: {e}", exc_info=True)
            error_data = {
                "event": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_audio_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Transfer-Encoding": "chunked"
        }
    )

@app.post("/api/tts/wav")
async def synthesize_speech_wav(request: TTSRequest):
    """
    Synthesize speech and return as WAV file
    
    Returns audio/wav binary data that can be directly played or saved
    """
    if not global_client:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        start_time = asyncio.get_event_loop().time()
        all_audio = []
        ttfa = None
        
        async for audio_chunk, info in global_client.stream_generate(
            text=request.text,
            voice=request.voice,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        ):
            if ttfa is None and info.get('ttfa_ms'):
                ttfa = info['ttfa_ms']
            all_audio.append(audio_chunk)
        
        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Concatenate all chunks
        full_audio = np.concatenate(all_audio)
        audio_int16 = (full_audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 24000, audio_int16)
        wav_bytes = wav_buffer.getvalue()
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        duration = len(full_audio) / 24000
        
        logger.info(f"TTS WAV completed: {len(request.text)} chars, {duration:.2f}s audio, {processing_time:.1f}ms, TTFA: {ttfa:.1f}ms")
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="tts_output.wav"',
                "X-Processing-Time-Ms": str(int(processing_time)),
                "X-TTFA-Ms": str(int(ttfa)) if ttfa else "0",
                "X-Audio-Duration-Seconds": f"{duration:.2f}"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS WAV error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/api/tts/chunked-stream")
async def chunked_stream(request: TTSRequest):
    """
    Chunked streaming endpoint for very long text
    Processes text in small chunks (default 50 tokens) to stay within context limits
    """
    if not global_client:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    async def stream_by_chunks():
        try:
            # Split text into sentences for better chunking
            sentences = request.text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            if not sentences:
                yield f"data: {json.dumps({'event': 'error', 'error': 'No text to process'})}\n\n"
                return
            
            start_time = asyncio.get_event_loop().time()
            total_chunks = 0
            total_samples = 0
            first_audio = True
            
            logger.info(f"Processing {len(sentences)} sentences with chunk_size={request.chunk_size}")
            
            # Process each sentence separately
            for sent_idx, sentence in enumerate(sentences):
                if not sentence or len(sentence) < 3:
                    continue
                
                try:
                    # Generate audio for this sentence
                    sentence_audio = []
                    
                    async for audio_chunk, info in global_client.stream_generate(
                        text=sentence,
                        voice=request.voice,
                        temperature=request.temperature,
                        max_tokens=min(request.chunk_size, 50)  # Max 50 tokens per sentence
                    ):
                        sentence_audio.append(audio_chunk)
                    
                    # If we got audio, stream it
                    if sentence_audio:
                        full_sentence_audio = np.concatenate(sentence_audio)
                        audio_int16 = (full_sentence_audio * 32767).astype(np.int16)
                        
                        # Stream in small chunks
                        chunk_size = 4096
                        for i in range(0, len(audio_int16), chunk_size):
                            chunk = audio_int16[i:i+chunk_size]
                            audio_bytes = chunk.tobytes()
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            
                            total_chunks += 1
                            total_samples += len(chunk)
                            
                            event_data = {
                                "chunk": total_chunks,
                                "sentence": sent_idx + 1,
                                "total_sentences": len(sentences),
                                "audio_base64": audio_base64,
                                "samples": len(chunk),
                                "duration_ms": (len(chunk) / 24000) * 1000,
                                "is_first": first_audio
                            }
                            
                            if first_audio:
                                event_data["ttfa_ms"] = (asyncio.get_event_loop().time() - start_time) * 1000
                                first_audio = False
                            
                            yield f"data: {json.dumps(event_data)}\n\n"
                            await asyncio.sleep(0)
                
                except Exception as e:
                    logger.warning(f"Error processing sentence {sent_idx + 1}: {e}")
                    continue
            
            # Send completion
            total_time = (asyncio.get_event_loop().time() - start_time) * 1000
            completion_data = {
                "event": "complete",
                "total_chunks": total_chunks,
                "total_sentences": len(sentences),
                "total_samples": total_samples,
                "total_duration_s": total_samples / 24000,
                "processing_time_ms": total_time
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
            logger.info(f"Chunked stream completed: {total_chunks} chunks, {len(sentences)} sentences, {total_samples/24000:.2f}s audio, {total_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Chunked stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_by_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/tts/simple-stream")
async def simple_stream(request: TTSRequest):
    """
    Simple streaming endpoint that works with limited context
    Collects audio first, then streams in chunks
    """
    if not global_client:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    # Strict limits for context
    text = request.text[:150]  # Max 150 chars
    max_tokens = min(request.max_tokens, 200)  # Max 200 tokens
    
    async def stream_chunks():
        try:
            # Generate all audio first
            all_audio = []
            ttfa = None
            start_time = asyncio.get_event_loop().time()
            
            async for audio_chunk, info in global_client.stream_generate(
                text=text,
                voice=request.voice,
                temperature=request.temperature,
                max_tokens=max_tokens
            ):
                if ttfa is None and info.get('ttfa_ms'):
                    ttfa = info['ttfa_ms']
                all_audio.append(audio_chunk)
            
            if not all_audio:
                yield f"data: {json.dumps({'event': 'error', 'error': 'No audio generated'})}\n\n"
                return
            
            # Concatenate all audio
            full_audio = np.concatenate(all_audio)
            audio_int16 = (full_audio * 32767).astype(np.int16)
            
            # Stream in 4096-sample chunks
            chunk_size = 4096
            chunk_num = 0
            
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                audio_bytes = chunk.tobytes()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                chunk_num += 1
                event_data = {
                    "chunk": chunk_num,
                    "audio_base64": audio_base64,
                    "samples": len(chunk),
                    "duration_ms": (len(chunk) / 24000) * 1000,
                    "ttfa_ms": ttfa if chunk_num == 1 else None
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(0)
            
            # Send completion
            total_time = (asyncio.get_event_loop().time() - start_time) * 1000
            yield f"data: {json.dumps({'event': 'complete', 'total_chunks': chunk_num, 'processing_time_ms': total_time})}\n\n"
            
            logger.info(f"Simple stream completed: {chunk_num} chunks, {len(audio_int16)/24000:.2f}s audio")
            
        except Exception as e:
            logger.error(f"Simple stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": [
            {
                "id": "tara",
                "name": "Tara",
                "description": "Default female voice"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=9100,
        reload=False,
        log_level="info",
        access_log=True
    )
