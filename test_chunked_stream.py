#!/usr/bin/env python3
"""
Test chunked streaming with very long text
Works with limited context by processing in 50-token chunks
"""

import asyncio
import aiohttp
import json
import base64
import numpy as np
from scipy.io import wavfile
import time

FASTAPI_URL = "http://localhost:9100"

# Very long text (from your curl example)
LONG_TEXT = """If you compiled llama.cpp yourself, Flash-Attention support is a build-time option. 
The parse error you hit is before that matters, but if you later see "flash attention not available," 
rebuild with CUDA + FA enabled. This is a comprehensive test of the chunked streaming endpoint 
that can handle very long text inputs by processing them in small chunks. The system splits the text 
into sentences and processes each sentence separately, ensuring we stay within the context limits 
of the server. This approach allows for unlimited text length while maintaining low latency and 
avoiding context size errors."""

async def test_chunked_stream(text: str, chunk_size: int = 50):
    """Test the chunked streaming endpoint"""
    
    print(f"\n{'='*80}")
    print(f"🎯 Testing Chunked Streaming")
    print(f"{'='*80}")
    print(f"Text length: {len(text)} characters")
    print(f"Chunk size: {chunk_size} tokens")
    print(f"{'='*80}")
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.3,
        "max_tokens": 256,
        "chunk_size": chunk_size,
        "stream": True
    }
    
    start_time = time.time()
    chunks = []
    ttfa = None
    sentences_completed = 0
    
    try:
        timeout = aiohttp.ClientTimeout(total=180, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{FASTAPI_URL}/api/tts/chunked-stream",
                json=payload
            ) as response:
                
                print(f"\n📥 Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text[:500]}")
                    return
                
                print("\n📊 Receiving chunks...")
                chunk_count = 0
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str.startswith('data: '):
                        continue
                    
                    data = json.loads(line_str[6:])
                    
                    if data.get('event') == 'error':
                        print(f"\n❌ Stream error: {data.get('error')}")
                        return
                    
                    if data.get('event') == 'complete':
                        print(f"\n\n✅ Stream Complete!")
                        print(f"   Total chunks: {data.get('total_chunks')}")
                        print(f"   Total sentences: {data.get('total_sentences')}")
                        print(f"   Total duration: {data.get('total_duration_s'):.2f}s")
                        print(f"   Processing time: {data.get('processing_time_ms'):.1f}ms")
                        break
                    
                    if 'audio_base64' in data:
                        chunk_count += 1
                        
                        if data.get('is_first') and data.get('ttfa_ms'):
                            ttfa = data['ttfa_ms']
                            print(f"\n⚡ TTFA: {ttfa:.1f}ms")
                        
                        # Decode audio
                        audio_bytes = base64.b64decode(data['audio_base64'])
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        chunks.append(audio_array)
                        
                        # Show progress
                        current_sentence = data.get('sentence', 0)
                        total_sentences = data.get('total_sentences', 0)
                        
                        if current_sentence > sentences_completed:
                            sentences_completed = current_sentence
                            print(f"\n   Sentence {current_sentence}/{total_sentences} completed")
                        
                        if chunk_count % 5 == 0:
                            print(f"   Chunk {chunk_count}: {len(audio_array)} samples", end='\r')
        
        # Save audio
        if chunks:
            full_audio = np.concatenate(chunks)
            filename = f"chunked_stream_{int(time.time())}.wav"
            wavfile.write(filename, 24000, full_audio)
            
            total_time = (time.time() - start_time) * 1000
            
            print(f"\n\n📊 Final Summary:")
            print(f"   TTFA: {ttfa:.1f}ms" if ttfa else "   TTFA: N/A")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Audio duration: {len(full_audio)/24000:.2f}s")
            print(f"   Samples: {len(full_audio)}")
            print(f"   Saved to: {filename}")
        else:
            print("\n⚠️  No audio received")
        
    except asyncio.TimeoutError:
        print(f"\n❌ Timeout after {(time.time() - start_time):.1f}s")
    except Exception as e:
        print(f"\n❌ Error: {e}")

async def main():
    print("="*80)
    print("🎯 Chunked Streaming Test - Long Text Support")
    print("="*80)
    print("Works with limited context by processing in 50-token chunks")
    print("="*80)
    
    # Check server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{FASTAPI_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"\n✅ Server is healthy")
                    print(f"   Status: {health.get('status')}")
                else:
                    print(f"\n⚠️  Server status: {response.status}")
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("   Make sure FastAPI server is running: python fastapi_server.py")
        return
    
    # Test with different chunk sizes
    test_cases = [
        ("Short text", "Hello world!", 50),
        ("Medium text", "This is a test of the chunked streaming system. It should work well.", 50),
        ("Long text (from curl)", LONG_TEXT, 50),
        ("Long text (smaller chunks)", LONG_TEXT, 40),
    ]
    
    for label, text, chunk_size in test_cases:
        print(f"\n\n{'='*80}")
        print(f"Test Case: {label}")
        print(f"{'='*80}")
        await test_chunked_stream(text, chunk_size)
        await asyncio.sleep(1)
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
