"""
Fixed Optimized TTS Client with correct API format

**client.py**
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
import statistics

import aiohttp
import numpy as np
from scipy.io.wavfile import write


class OptimizedTTSClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
        # Connection pool settings for better concurrency
        self.connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=20,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,
        )
        
        # Timeout settings
        self.timeout = aiohttp.ClientTimeout(
            total=300,
            connect=10,
            sock_read=60,
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                "User-Agent": "OptimizedTTSClient/1.0",
                "Accept": "application/json",
                "Connection": "keep-alive"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self, 'session'):
            await self.session.close()
        await self.connector.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Health check failed with status {response.status}")
        except Exception as e:
            raise Exception(f"Health check failed: {str(e)}")
    
    async def get_voices(self) -> Dict[str, List[str]]:
        """Get available voices"""
        try:
            async with self.session.get(f"{self.base_url}/voices") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["voices"]
                else:
                    raise Exception(f"Get voices failed with status {response.status}")
        except Exception as e:
            raise Exception(f"Get voices failed: {str(e)}")
    
    async def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            async with self.session.get(f"{self.base_url}/gpu-info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"GPU info failed with status {response.status}")
        except Exception as e:
            raise Exception(f"GPU info failed: {str(e)}")
    
    async def tts_non_streaming(
        self, 
        text: str, 
        voice_id: str = "tara",
        language: str = "en",
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        pre_buffer_size: float = 0.2
    ) -> Dict[str, Any]:
        """Fixed non-streaming TTS request"""
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "pre_buffer_size": pre_buffer_size
        }
        
        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.base_url}/tts", 
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Request failed with status {response.status}: {error_text}")
                
                data = await response.json()
                data["client_latency_ms"] = (time.time() - start_time) * 1000
                return data
                
        except Exception as e:
            raise Exception(f"Non-streaming request failed: {str(e)}")
    
    async def tts_streaming(
        self,
        text: str,
        voice_id: str = "tara", 
        language: str = "en",
        save_to_file: Optional[str] = None,
        show_progress: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        pre_buffer_size: float = 0.2
    ) -> Dict[str, Any]:
        """Fixed streaming TTS request"""
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "pre_buffer_size": pre_buffer_size
        }
        
        chunks = []
        start_time = time.time()
        first_chunk_time = None
        
        try:
            async with self.session.post(
                f"{self.base_url}/tts/stream", 
                json=payload,
                headers={
                    "Accept": "text/plain",
                    "Cache-Control": "no-cache"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Streaming request failed with status {response.status}: {error_text}")
                
                buffer = ""
                chunk_count = 0
                
                async for chunk_bytes in response.content.iter_chunked(16384):
                    chunk_str = chunk_bytes.decode('utf-8', errors='ignore')
                    buffer += chunk_str
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            try:
                                chunk_data = json.loads(data_str)
                                
                                if first_chunk_time is None and not chunk_data.get("error"):
                                    first_chunk_time = time.time()
                                    if show_progress:
                                        print(f"First chunk received after {(first_chunk_time - start_time)*1000:.1f}ms")
                                
                                if chunk_data.get("error"):
                                    raise Exception(f"Server error: {chunk_data.get('message')}")
                                
                                if chunk_data.get("end"):
                                    if show_progress:
                                        print(f"Stream completed. Total chunks: {chunk_data.get('total_chunks')}")
                                    break
                                
                                chunks.append(chunk_data)
                                chunk_count += 1
                                
                                if show_progress and chunk_count % 3 == 0:
                                    elapsed = chunk_data.get("elapsed_ms", 0)
                                    print(f"Received {chunk_count} chunks (elapsed: {elapsed:.0f}ms)")
                                
                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON
                            except Exception as e:
                                raise Exception(f"Error processing chunk: {e}")
            
        except Exception as e:
            raise Exception(f"Streaming failed: {str(e)}")
        
        total_time = time.time() - start_time
        time_to_first_chunk = (first_chunk_time - start_time) * 1000 if first_chunk_time else None
        
        # Combine all audio chunks
        all_audio_data = []
        sample_rate = None
        
        if chunks:
            sample_rate = chunks[0]["sample_rate"]
            
            for chunk in chunks:
                audio_data = chunk.get("audio_data", [])
                if audio_data:
                    all_audio_data.extend(audio_data)
            
            # Save to file if requested
            if save_to_file and all_audio_data:
                audio_array = np.array(all_audio_data, dtype=np.int16)
                write(save_to_file, sample_rate, audio_array)
                if show_progress:
                    print(f"Saved {len(audio_array)} samples to {save_to_file}")
        
        return {
            "text": text,
            "total_chunks": len(chunks),
            "total_latency_ms": total_time * 1000,
            "time_to_first_chunk_ms": time_to_first_chunk,
            "sample_rate": sample_rate,
            "total_audio_samples": len(all_audio_data)
        }
    
    async def download_wav(
        self,
        text: str,
        filename: str,
        voice_id: str = "tara",
        language: str = "en",
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        pre_buffer_size: float = 0.2
    ) -> Dict[str, Any]:
        """Fixed WAV download"""
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "pre_buffer_size": pre_buffer_size
        }
        
        start_time = time.time()
        try:
            async with self.session.post(f"{self.base_url}/tts/wav", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"WAV request failed with status {response.status}: {error_text}")
                
                with open(filename, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                
                total_time = (time.time() - start_time) * 1000
                file_size = len(open(filename, 'rb').read())
                
                return {
                    "text": text,
                    "filename": filename,
                    "download_time_ms": total_time,
                    "file_size_bytes": file_size
                }
        except Exception as e:
            raise Exception(f"WAV download failed: {str(e)}")


class ConcurrencyTester:
    def __init__(self, client: OptimizedTTSClient):
        self.client = client
    
    async def test_concurrent_requests(
        self,
        texts: List[str],
        concurrent_users: int = 5,
        voice_id: str = "tara",
        language: str = "en",
        streaming: bool = True
    ) -> Dict[str, Any]:
        """Test concurrent TTS requests with optimized handling"""
        print(f"\n🧪 Testing {concurrent_users} concurrent {'streaming' if streaming else 'non-streaming'} requests...")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def single_request(text: str, request_id: int):
            async with semaphore:
                try:
                    if streaming:
                        result = await self.client.tts_streaming(
                            text, voice_id=voice_id, language=language
                        )
                    else:
                        result = await self.client.tts_non_streaming(
                            text, voice_id=voice_id, language=language
                        )
                    
                    result["request_id"] = request_id
                    result["success"] = True
                    return result
                    
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": str(e),
                        "text": text
                    }
        
        # Create tasks for all requests
        tasks = []
        for i, text in enumerate(texts):
            task = asyncio.create_task(single_request(text, i))
            tasks.append(task)
        
        # Execute all requests and measure total time
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Handle any exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "request_id": i,
                    "success": False,
                    "error": str(result),
                    "text": texts[i] if i < len(texts) else "unknown"
                })
            else:
                processed_results.append(result)
        
        # Analyze results
        successful_results = [r for r in processed_results if r.get("success")]
        failed_results = [r for r in processed_results if not r.get("success")]
        
        latencies = []
        ttfc_times = []  # Time to first chunk for streaming
        
        for result in successful_results:
            if streaming:
                latencies.append(result.get("total_latency_ms", 0))
                if result.get("time_to_first_chunk_ms"):
                    ttfc_times.append(result["time_to_first_chunk_ms"])
            else:
                latencies.append(result.get("client_latency_ms", 0))
        
        analysis = {
            "test_type": "streaming" if streaming else "non_streaming",
            "concurrent_users": concurrent_users,
            "total_requests": len(texts),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "total_test_time_seconds": total_time,
            "requests_per_second": len(texts) / total_time if total_time > 0 else 0,
            "latency_stats": {
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "mean_ms": statistics.mean(latencies) if latencies else 0,
                "median_ms": statistics.median(latencies) if latencies else 0,
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "errors": [r["error"] for r in failed_results]
        }
        
        if streaming and ttfc_times:
            analysis["time_to_first_chunk_stats"] = {
                "min_ms": min(ttfc_times),
                "max_ms": max(ttfc_times),
                "mean_ms": statistics.mean(ttfc_times),
                "median_ms": statistics.median(ttfc_times)
            }
        
        return analysis


async def main():
    """Main testing function with optimized flow"""
    
    # Test texts optimized for different lengths
    test_texts = [
        "Hello!",
        "Testing the system.",
        "This is a medium length test sentence.",
        "Quick test.",
        "The server should handle this well.",
    ]
    
    print("🚀 Starting Fixed TTS Performance Tests")
    print("=" * 60)
    
    async with OptimizedTTSClient() as client:
        
        # Health check with GPU info
        try:
            health = await client.health_check()
            print(f"✅ Server Status: {health['status']}")
            print(f"📋 Models Loaded: {health['models_loaded']}")
            print(f"⏱️  Uptime: {health['uptime_seconds']:.2f} seconds")
            print(f"🎯 GPU Acceleration: {health.get('gpu_acceleration', 'Unknown')}")
            print(f"🔧 GPU Layers: {health.get('gpu_layers', 'Unknown')}")
            print(f"🔄 Active Requests: {health.get('active_requests', 0)}")
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            return
        
        # GPU info check
        try:
            gpu_info = await client.get_gpu_info()
            print(f"🎮 CUDA Available: {gpu_info.get('cuda_available', False)}")
            print(f"💾 GPU: {gpu_info.get('cuda_device_name', 'Unknown')}")
            print(f"🚀 CUDA Provider: {gpu_info.get('cuda_provider_available', False)}")
            if gpu_info.get('cuda_available', False):
                print(f"💾 VRAM Total: {gpu_info.get('cuda_memory_total', 'Unknown')}")
                print(f"💾 VRAM Used: {gpu_info.get('cuda_memory_allocated', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  GPU info failed: {e}")
        
        # Get available voices
        try:
            voices = await client.get_voices()
            print(f"🎤 Available voices: {len(voices)} languages")
        except Exception as e:
            print(f"❌ Failed to get voices: {e}")
        
        # Test single streaming request
        print("\n" + "=" * 60)
        print("🧪 Testing Single Streaming Request")
        try:
            result = await client.tts_streaming(
                "Hello, this is an optimized test. Howe are you doing let's see how it goes",
                save_to_file="test_streaming.wav",
                show_progress=True
            )
            print(f"✅ Success!")
            print(f"   Time to first chunk: {result['time_to_first_chunk_ms']:.1f}ms")
            print(f"   Total latency: {result['total_latency_ms']:.1f}ms")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Audio samples: {result['total_audio_samples']}")
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
        
        # Test single non-streaming request
        print("\n" + "=" * 60)
        print("🧪 Testing Single Non-Streaming Request")
        try:
            result = await client.tts_non_streaming(
                "Hello, this is a non-streaming test."
            )
            print(f"✅ Success!")
            print(f"   Client latency: {result['client_latency_ms']:.1f}ms")
            print(f"   Server duration: {result['duration_ms']:.1f}ms")
            print(f"   Audio samples: {len(result['audio_data'])}")
            
            # Save the audio
            audio_array = np.array(result['audio_data'], dtype=np.int16)
            write("test_non_streaming.wav", result['sample_rate'], audio_array)
            print(f"   Saved audio to test_non_streaming.wav")
            
        except Exception as e:
            print(f"❌ Non-streaming test failed: {e}")
        
        # Test WAV download
        print("\n" + "=" * 60)
        print("🧪 Testing WAV Download")
        try:
            result = await client.download_wav(
                "WAV download test.",
                "test_download.wav"
            )
            print(f"✅ Success!")
            print(f"   Download time: {result['download_time_ms']:.1f}ms")
            print(f"   File size: {result['file_size_bytes']} bytes")
        except Exception as e:
            print(f"❌ WAV download test failed: {e}")
        
        # Quick concurrency test
        print("\n" + "=" * 60)
        print("🧪 Testing Basic Concurrency (2 users)")
        
        tester = ConcurrencyTester(client)
        
        # Test with just 2 concurrent users first
        result = await tester.test_concurrent_requests(
            test_texts[:2],
            concurrent_users=2,
            streaming=True
        )
        
        print(f"📊 Streaming Results (2 concurrent users):")
        print(f"   ✅ Success Rate: {result['successful_requests']}/{result['total_requests']}")
        if result['successful_requests'] > 0:
            print(f"   🚀 Requests/sec: {result['requests_per_second']:.2f}")
            print(f"   ⚡ Mean Latency: {result['latency_stats']['mean_ms']:.1f}ms")
            if 'time_to_first_chunk_stats' in result:
                print(f"   ⚡ Mean TTFC: {result['time_to_first_chunk_stats']['mean_ms']:.1f}ms")
        
        if result['errors']:
            print(f"   ❌ Errors: {result['errors'][:2]}")  # Show first 2 errors
        
        print("\n" + "=" * 60)
        print("🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())