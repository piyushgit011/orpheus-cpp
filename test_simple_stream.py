#!/usr/bin/env python3
"""
Test the simple streaming endpoint with context-limited server
"""

import asyncio
import aiohttp
import json
import base64
import numpy as np
from scipy.io import wavfile
import time

FASTAPI_URL = "http://localhost:9100"

async def test_simple_stream(text: str):
    """Test the simple streaming endpoint"""
    
    print(f"\n{'='*80}")
    print(f"Testing: '{text}'")
    print(f"Text length: {len(text)} characters")
    print(f"{'='*80}")
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    start_time = time.time()
    chunks = []
    ttfa = None
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{FASTAPI_URL}/api/tts/simple-stream",
                json=payload
            ) as response:
                
                print(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
                    return
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str.startswith('data: '):
                        continue
                    
                    data = json.loads(line_str[6:])
                    
                    if data.get('event') == 'error':
                        print(f"❌ Error: {data.get('error')}")
                        return
                    
                    if data.get('event') == 'complete':
                        print(f"\n✅ Complete!")
                        print(f"   Total chunks: {data.get('total_chunks')}")
                        print(f"   Processing time: {data.get('processing_time_ms'):.1f}ms")
                        break
                    
                    if 'audio_base64' in data:
                        if ttfa is None and data.get('ttfa_ms'):
                            ttfa = data['ttfa_ms']
                            print(f"⚡ TTFA: {ttfa:.1f}ms")
                        
                        # Decode audio
                        audio_bytes = base64.b64decode(data['audio_base64'])
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        chunks.append(audio_array)
                        
                        print(f"   Chunk {data['chunk']}: {len(audio_array)} samples, {data.get('duration_ms', 0):.1f}ms audio")
        
        # Save audio
        if chunks:
            full_audio = np.concatenate(chunks)
            filename = f"simple_stream_{int(time.time())}.wav"
            wavfile.write(filename, 24000, full_audio)
            
            total_time = (time.time() - start_time) * 1000
            print(f"\n📊 Summary:")
            print(f"   TTFA: {ttfa:.1f}ms" if ttfa else "   TTFA: N/A")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Chunks: {len(chunks)}")
            print(f"   Audio: {len(full_audio)/24000:.2f}s")
            print(f"   Saved: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    print("="*80)
    print("🎯 Simple Streaming Endpoint Test")
    print("="*80)
    print("Works with limited context (n_ctx=68)")
    print("="*80)
    
    # Check server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{FASTAPI_URL}/health") as response:
                if response.status == 200:
                    print("✅ Server is healthy\n")
                else:
                    print(f"⚠️  Server status: {response.status}\n")
    except Exception as e:
        print(f"❌ Cannot connect: {e}\n")
        return
    
    # Test cases (keep short for limited context)
    test_cases = [
        "Hello world!",
        "How are you?",
        "This is a test.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for text in test_cases:
        await test_simple_stream(text)
        await asyncio.sleep(0.5)
    
    print("\n" + "="*80)
    print("✅ Tests completed!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
