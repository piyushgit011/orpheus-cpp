# FastAPI TTS Server

High-performance FastAPI server for Orpheus TTS with streaming support.

## Features

- **Three endpoint types**: JSON, WAV download, and SSE streaming
- **Real-time streaming**: Server-Sent Events (SSE) for minimal latency
- **CORS enabled**: Works with web applications
- **Production-ready**: Proper error handling and logging
- **Concurrent support**: Handles multiple requests simultaneously

## Prerequisites

```bash
# Install dependencies
pip install fastapi uvicorn aiohttp scipy numpy snac
```

Make sure llama-server is running on port 8090:
```bash
cd llama.cpp
./start_optimized_server.sh
```

## Starting the Server

```bash
python fastapi_server.py
```

Server will start on `http://localhost:9100`

## API Endpoints

### 1. GET / 
Health check and API information

```bash
curl http://localhost:9100/
```

### 2. GET /health
Detailed health status

```bash
curl http://localhost:9100/health
```

### 3. POST /api/tts
Generate TTS and return base64-encoded audio in JSON

```bash
curl -X POST http://localhost:9100/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "tara",
    "temperature": 0.3,
    "max_tokens": 512,
    "stream": false
  }'
```

Response:
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAA...",
  "duration_seconds": 1.5,
  "sample_rate": 24000,
  "processing_time_ms": 250.5,
  "ttfa_ms": 158.2
}
```

### 4. POST /api/tts/wav
Generate TTS and return WAV file directly

```bash
# Download as file
curl -X POST http://localhost:9100/api/tts/wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "tara"}' \
  -o output.wav

# Play directly (requires ffplay)
curl -X POST http://localhost:9100/api/tts/wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  | ffplay -nodisp -autoexit -
```

Response headers include timing information:
- `X-Processing-Time-Ms`: Total processing time
- `X-TTFA-Ms`: Time to first audio
- `X-Audio-Duration-Seconds`: Duration of generated audio

### 5. POST /api/tts/stream
Generate TTS with streaming (Server-Sent Events)

```bash
curl -X POST http://localhost:9100/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "tara"}' \
  -N
```

SSE events:
```
data: {"chunk": 1, "audio_base64": "...", "samples": 4096, "duration_ms": 85.3, "ttfa_ms": 158.2}

data: {"chunk": 2, "audio_base64": "...", "samples": 4096, "duration_ms": 85.3}

data: {"event": "complete", "total_chunks": 2, "total_samples": 8192, "total_duration_s": 0.34, "processing_time_ms": 276.0}
```

### 6. GET /api/voices
List available voices

```bash
curl http://localhost:9100/api/voices
```

## Request Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| text | string | required | 1-5000 chars | Text to convert to speech |
| voice | string | "tara" | - | Voice to use |
| temperature | float | 0.3 | 0.0-2.0 | Sampling temperature (lower = faster) |
| max_tokens | int | 512 | 64-2048 | Maximum tokens to generate |
| stream | bool | true | - | Enable streaming (for /api/tts only) |

## Testing

### Test WAV endpoint
```bash
python test_wav_endpoint.py
```

### Test streaming with large text
```bash
python test_large_streaming.py
```

### Benchmark concurrent performance
```bash
python benchmark_fastapi.py
```

## Python Client Example

```python
import aiohttp
import asyncio
import json
import base64
import numpy as np
from scipy.io import wavfile

async def stream_tts(text: str):
    async with aiohttp.ClientSession() as session:
        payload = {
            "text": text,
            "voice": "tara",
            "temperature": 0.3,
            "max_tokens": 512
        }
        
        async with session.post(
            "http://localhost:9100/api/tts/stream",
            json=payload
        ) as response:
            
            all_audio = []
            
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                
                if not line_str.startswith('data: '):
                    continue
                
                data = json.loads(line_str[6:])
                
                if data.get('event') == 'complete':
                    break
                
                if 'audio_base64' in data:
                    # Decode audio chunk
                    audio_bytes = base64.b64decode(data['audio_base64'])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    all_audio.append(audio_array)
                    print(f"Received chunk {data['chunk']}")
            
            # Combine all chunks
            if all_audio:
                full_audio = np.concatenate(all_audio)
                wavfile.write("output.wav", 24000, full_audio)
                print(f"Saved {len(full_audio)/24000:.2f}s audio to output.wav")

asyncio.run(stream_tts("Hello, this is a test!"))
```

## JavaScript/Web Client Example

```javascript
async function streamTTS(text) {
    const response = await fetch('http://localhost:9100/api/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            voice: 'tara',
            temperature: 0.3,
            max_tokens: 512
        })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const audioChunks = [];
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                if (data.event === 'complete') {
                    console.log('Stream complete:', data);
                    break;
                }
                
                if (data.audio_base64) {
                    // Decode base64 audio
                    const audioData = atob(data.audio_base64);
                    const audioArray = new Int16Array(audioData.length / 2);
                    for (let i = 0; i < audioData.length; i += 2) {
                        audioArray[i/2] = (audioData.charCodeAt(i+1) << 8) | audioData.charCodeAt(i);
                    }
                    audioChunks.push(audioArray);
                    console.log(`Chunk ${data.chunk}: ${data.samples} samples`);
                }
            }
        }
    }
    
    // Play audio using Web Audio API
    const audioContext = new AudioContext({ sampleRate: 24000 });
    const totalSamples = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const audioBuffer = audioContext.createBuffer(1, totalSamples, 24000);
    const channelData = audioBuffer.getChannelData(0);
    
    let offset = 0;
    for (const chunk of audioChunks) {
        for (let i = 0; i < chunk.length; i++) {
            channelData[offset++] = chunk[i] / 32768.0;
        }
    }
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}

// Usage
streamTTS("Hello, this is a streaming test!");
```

## Performance Tips

1. **Use streaming endpoint** for minimal latency (<200ms TTFA)
2. **Lower temperature** (0.2-0.3) for faster generation
3. **Reduce max_tokens** (256-512) for shorter audio
4. **Connection pooling** for concurrent requests
5. **WAV endpoint** is simplest for direct playback

## Troubleshooting

### Server won't start
- Check if port 9100 is available: `lsof -i :9100`
- Ensure llama-server is running: `lsof -i :8090`
- Check dependencies: `pip list | grep -E "fastapi|uvicorn|aiohttp"`

### Streaming fails with large text
- Increase max_tokens if needed
- Check llama-server context size (should be 2048+)
- Monitor server logs for errors

### High latency
- Check llama-server is using GPU: `nvidia-smi`
- Ensure server has parallel slots enabled (30+)
- Lower temperature for faster sampling
- Use streaming endpoint instead of /api/tts

### Audio quality issues
- Increase temperature for more natural speech (0.5-0.8)
- Increase max_tokens for longer outputs
- Check SNAC decoder is using CUDA

## Architecture

```
Client Request
     ↓
FastAPI Server (port 9100)
     ↓
UltraFastOrpheusClient
     ↓
llama-server (port 8090)
     ↓
Orpheus 3B Model (SNAC tokens)
     ↓
SNAC Decoder (CUDA)
     ↓
24kHz Audio (PCM/WAV)
```

## Logging

Server logs include:
- Request details (text length, parameters)
- Performance metrics (TTFA, processing time)
- Audio duration and chunk count
- Errors with stack traces

Example log:
```
INFO:fastapi_server:TTS WAV completed: 16 chars, 0.34s audio, 276.0ms, TTFA: 158.2ms
```

## Production Deployment

For production, use:
```bash
uvicorn fastapi_server:app --host 0.0.0.0 --port 9100 --workers 4
```

Or with gunicorn:
```bash
gunicorn fastapi_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9100
```
