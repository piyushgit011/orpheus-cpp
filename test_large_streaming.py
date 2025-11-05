#!/usr/bin/env python3
"""
Test FastAPI streaming with large text inputs
"""

import asyncio
import aiohttp
import json
import time

FASTAPI_URL = "http://localhost:9100"

# Test with increasing text lengths
TEST_CASES = [
    ("Short text", "Hello world!"),
    ("Medium text", "This is a medium length text to test the streaming API. " * 3),
    ("Long text", "This is a long text input to test how the streaming endpoint handles larger amounts of text. " * 10),
    ("Very long text", "Testing with a very long text input that contains many sentences and should generate a lot of audio tokens. " * 30),
]

async def test_streaming(text: str, label: str):
    """Test streaming endpoint with given text"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {label}")
    print(f"Text length: {len(text)} characters")
    print(f"{'='*80}")
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": True
    }
    
    start_time = time.time()
    chunks_received = 0
    ttfa = None
    total_samples = 0
    
    try:
        timeout = aiohttp.ClientTimeout(total=120, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{FASTAPI_URL}/api/tts/stream",
                json=payload
            ) as response:
                
                print(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text[:500]}")
                    return
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str or not line_str.startswith('data: '):
                        continue
                    
                    data_str = line_str[6:]
                    
                    try:
                        data = json.loads(data_str)
                        
                        if data.get('event') == 'error':
                            print(f"❌ Stream error: {data.get('error')}")
                            return
                        
                        if data.get('event') == 'complete':
                            print(f"\n✅ Stream completed!")
                            print(f"   Total chunks: {data.get('total_chunks')}")
                            print(f"   Total samples: {data.get('total_samples')}")
                            print(f"   Audio duration: {data.get('total_duration_s'):.2f}s")
                            print(f"   Processing time: {data.get('processing_time_ms'):.1f}ms")
                            break
                        
                        if 'chunk' in data:
                            chunks_received += 1
                            samples = data.get('samples', 0)
                            total_samples += samples
                            
                            if ttfa is None and data.get('ttfa_ms'):
                                ttfa = data['ttfa_ms']
                                print(f"⚡ TTFA: {ttfa:.1f}ms")
                            
                            if chunks_received <= 5 or chunks_received % 10 == 0:
                                print(f"   Chunk {chunks_received}: {samples} samples, {data.get('duration_ms', 0):.1f}ms audio")
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error: {e}")
                        continue
        
        total_time = (time.time() - start_time) * 1000
        print(f"\n📊 Summary:")
        print(f"   TTFA: {ttfa:.1f}ms" if ttfa else "   TTFA: N/A")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Chunks received: {chunks_received}")
        print(f"   Total audio: {total_samples/24000:.2f}s")
        
    except asyncio.TimeoutError:
        print(f"❌ Timeout after {(time.time() - start_time):.1f}s")
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    print("="*80)
    print("🎯 FastAPI Streaming Test - Large Text Inputs")
    print("="*80)
    
    # Check server health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{FASTAPI_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Server is healthy")
                    print(f"   Status: {health.get('status')}")
                else:
                    print(f"⚠️  Server health check returned {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Run tests
    for label, text in TEST_CASES:
        await test_streaming(text, label)
        await asyncio.sleep(1)
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
