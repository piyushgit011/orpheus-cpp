"""
Test client for the optimized TTS server
"""
import asyncio
import aiohttp
import time
import json
from typing import Dict, Any

class OptimizedTTSClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def test_health(self):
        """Test server health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Server Health:")
                    print(f"   Status: {data['status']}")
                    print(f"   Models loaded: {data['models_loaded']}")
                    print(f"   Optimization level: {data['optimization_level']}")
                    print(f"   GPU acceleration: {data['gpu_acceleration']}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
    
    async def test_streaming(self, text: str = "Hello, this is an optimized streaming test!"):
        """Test streaming endpoint"""
        payload = {
            "text": text,
            "voice_id": "tara",
            "language": "en",
            "pre_buffer_size": 0.1
        }
        
        print(f"🧪 Testing streaming with text: '{text}'")
        start_time = time.time()
        chunk_count = 0
        first_chunk_time = None
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/tts/stream",
                json=payload
            ) as response:
                if response.status != 200:
                    print(f"❌ Streaming failed: {response.status}")
                    return
                
                buffer = ""
                async for chunk_bytes in response.content.iter_chunked(8192):
                    chunk_str = chunk_bytes.decode('utf-8', errors='ignore')
                    buffer += chunk_str
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                chunk_data = json.loads(data_str)
                                
                                if first_chunk_time is None and not chunk_data.get("error"):
                                    first_chunk_time = time.time()
                                    print(f"⚡ First chunk received: {(first_chunk_time - start_time)*1000:.1f}ms")
                                
                                if chunk_data.get("error"):
                                    print(f"❌ Error: {chunk_data.get('message')}")
                                    return
                                
                                if chunk_data.get("end"):
                                    total_time = (time.time() - start_time) * 1000
                                    print(f"✅ Streaming completed!")
                                    print(f"   Total chunks: {chunk_data.get('total_chunks')}")
                                    print(f"   Total time: {total_time:.1f}ms")
                                    print(f"   Optimizations: {chunk_data.get('optimizations_applied', [])}")
                                    return
                                
                                chunk_count += 1
                                
                            except json.JSONDecodeError:
                                continue
    
    async def test_non_streaming(self, text: str = "Hello, this is a non-streaming test!"):
        """Test non-streaming endpoint"""
        payload = {
            "text": text,
            "voice_id": "tara",
            "language": "en"
        }
        
        print(f"🧪 Testing non-streaming with text: '{text}'")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/tts",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    total_time = time.time() - start_time
                    print(f"✅ Non-streaming completed!")
                    print(f"   Duration: {data['duration_ms']:.1f}ms")
                    print(f"   Total time: {total_time*1000:.1f}ms")
                    print(f"   Audio samples: {len(data['audio_data'])}")
                    print(f"   Optimizations: {data['optimizations_applied']}")
                else:
                    print(f"❌ Non-streaming failed: {response.status}")

async def main():
    """Run all tests"""
    client = OptimizedTTSClient()
    
    print("🚀 Testing Optimized TTS Server")
    print("=" * 50)
    
    # Test health
    if not await client.test_health():
        return
    
    print("\n" + "=" * 50)
    
    # Test streaming
    await client.test_streaming()
    
    print("\n" + "=" * 50)
    
    # Test non-streaming
    await client.test_non_streaming()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())