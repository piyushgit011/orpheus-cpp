"""
Advanced Test Client for Latency and Concurrency Testing
"""
import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np

class AdvancedTTSClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def create_session(self):
        """Create optimized HTTP session"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,
        )
        timeout = aiohttp.ClientTimeout(
            total=300,
            connect=10,
            sock_read=60,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "AdvancedTTSClient/2.0",
                "Accept": "application/json",
                "Connection": "keep-alive"
            }
        )
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def test_health(self) -> Dict[str, Any]:
        """Test server health"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Server Health Check:")
                    print(f"   Status: {data['status']}")
                    print(f"   Models loaded: {data['models_loaded']}")
                    print(f"   Optimization level: {data['optimization_level']}")
                    print(f"   GPU acceleration: {data['gpu_acceleration']}")
                    print(f"   API format: {data['api_format']}")
                    print(f"   Active requests: {data['active_requests']}")
                    return data
                else:
                    raise Exception(f"Health check failed with status {response.status}")
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return {}
    
    async def test_streaming_latency(self, 
                                   text: str, 
                                   voice_id: str = "tara",
                                   show_progress: bool = False) -> Dict[str, Any]:
        """Test streaming endpoint latency with detailed metrics"""
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": "en",
            "pre_buffer_size": 0.1
        }
        
        start_time = time.time()
        first_chunk_time = None
        chunks = []
        total_audio_samples = 0
        
        try:
            async with self.session.post(
                f"{self.base_url}/tts/stream",
                json=payload,
                headers={"Accept": "application/x-ndjson"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                
                buffer = ""
                async for chunk_bytes in response.content.iter_chunked(16384):
                    chunk_str = chunk_bytes.decode('utf-8', errors='ignore')
                    buffer += chunk_str
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            try:
                                chunk_data = json.loads(line)
                                
                                if first_chunk_time is None and chunk_data.get("type") == "audio_chunk":
                                    first_chunk_time = time.time()
                                    if show_progress:
                                        print(f"⚡ First chunk: {(first_chunk_time - start_time)*1000:.1f}ms")
                                
                                if chunk_data.get("type") == "error":
                                    return {
                                        "success": False,
                                        "error": chunk_data.get("message")
                                    }
                                
                                if chunk_data.get("type") == "stream_complete":
                                    total_time = time.time() - start_time
                                    
                                    if show_progress:
                                        print(f"✅ Stream complete: {total_time*1000:.1f}ms")
                                    
                                    return {
                                        "success": True,
                                        "total_time_ms": total_time * 1000,
                                        "time_to_first_chunk_ms": (first_chunk_time - start_time) * 1000 if first_chunk_time else None,
                                        "total_chunks": len(chunks),
                                        "total_audio_samples": total_audio_samples,
                                        "server_metrics": chunk_data.get("performance_metrics", {}),
                                        "optimizations": chunk_data.get("optimizations_applied", []),
                                        "text_length": len(text),
                                        "throughput_chars_per_second": len(text) / total_time if total_time > 0 else 0
                                    }
                                
                                if chunk_data.get("type") == "audio_chunk":
                                    chunks.append(chunk_data)
                                    total_audio_samples += len(chunk_data.get("audio_data", []))
                                    
                                    if show_progress and len(chunks) % 5 == 0:
                                        elapsed = chunk_data.get("elapsed_ms", 0)
                                        print(f"📊 Chunks: {len(chunks)}, Elapsed: {elapsed:.0f}ms")
                                
                            except json.JSONDecodeError:
                                continue
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Streaming test failed: {str(e)}"
            }
        
        return {
            "success": False,
            "error": "Stream ended unexpectedly"
        }
    
    async def test_non_streaming_latency(self, 
                                       text: str, 
                                       voice_id: str = "tara") -> Dict[str, Any]:
        """Test non-streaming endpoint latency"""
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": "en"
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/tts",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                
                data = await response.json()
                total_time = time.time() - start_time
                
                return {
                    "success": True,
                    "client_latency_ms": total_time * 1000,
                    "server_duration_ms": data.get("duration_ms", 0),
                    "audio_samples": len(data.get("audio_data", [])),
                    "audio_length_seconds": data.get("audio_length_seconds", 0),
                    "server_metrics": data.get("performance_metrics", {}),
                    "optimizations": data.get("optimizations_applied", []),
                    "text_length": len(text),
                    "throughput_chars_per_second": len(text) / total_time if total_time > 0 else 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Non-streaming test failed: {str(e)}"
            }
    
    async def test_concurrent_requests(self, 
                                     texts: List[str],
                                     concurrent_users: int = 5,
                                     test_type: str = "streaming",
                                     voice_id: str = "tara") -> Dict[str, Any]:
        """Test concurrent request handling"""
        print(f"\n🧪 Testing {concurrent_users} concurrent {test_type} requests...")
        
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def single_request(text: str, request_id: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    if test_type == "streaming":
                        result = await self.test_streaming_latency(text, voice_id)
                    else:
                        result = await self.test_non_streaming_latency(text, voice_id)
                    
                    result["request_id"] = request_id
                    result["text_used"] = text
                    return result
                    
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": str(e),
                        "text_used": text
                    }
        
        # Create tasks
        tasks = []
        for i, text in enumerate(texts):
            task = asyncio.create_task(single_request(text, i))
            tasks.append(task)
        
        # Execute and measure
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result)})
            elif result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # Calculate statistics
        latencies = []
        ttfc_times = []  # Time to first chunk
        throughputs = []
        
        for result in successful_results:
            if test_type == "streaming":
                latencies.append(result.get("total_time_ms", 0))
                if result.get("time_to_first_chunk_ms"):
                    ttfc_times.append(result["time_to_first_chunk_ms"])
            else:
                latencies.append(result.get("client_latency_ms", 0))
            
            throughputs.append(result.get("throughput_chars_per_second", 0))
        
        analysis = {
            "test_type": test_type,
            "concurrent_users": concurrent_users,
            "total_requests": len(texts),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate_percent": (len(successful_results) / len(texts)) * 100,
            "total_test_time_seconds": total_time,
            "requests_per_second": len(texts) / total_time if total_time > 0 else 0,
            "latency_stats": {
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "mean_ms": statistics.mean(latencies) if latencies else 0,
                "median_ms": statistics.median(latencies) if latencies else 0,
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "p95_ms": np.percentile(latencies, 95) if latencies else 0,
                "p99_ms": np.percentile(latencies, 99) if latencies else 0
            },
            "throughput_stats": {
                "min_chars_per_second": min(throughputs) if throughputs else 0,
                "max_chars_per_second": max(throughputs) if throughputs else 0,
                "mean_chars_per_second": statistics.mean(throughputs) if throughputs else 0
            },
            "errors": [r.get("error") for r in failed_results if r.get("error")]
        }
        
        if test_type == "streaming" and ttfc_times:
            analysis["time_to_first_chunk_stats"] = {
                "min_ms": min(ttfc_times),
                "max_ms": max(ttfc_times),
                "mean_ms": statistics.mean(ttfc_times),
                "median_ms": statistics.median(ttfc_times),
                "p95_ms": np.percentile(ttfc_times, 95),
                "p99_ms": np.percentile(ttfc_times, 99)
            }
        
        return analysis
    
    async def comprehensive_benchmark(self, test_texts: List[str]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("🚀 Starting Comprehensive TTS Benchmark")
        print("=" * 60)
        
        # Health check
        health_data = await self.test_health()
        if not health_data:
            return {"error": "Server health check failed"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "server_info": health_data,
            "test_results": {}
        }
        
        # Single request latency tests
        print("\n📊 Single Request Latency Tests")
        print("-" * 40)
        
        for i, text in enumerate(test_texts[:3]):  # Test first 3 texts
            print(f"\nTest {i+1}: '{text[:50]}...' ({len(text)} chars)")
            
            # Streaming test
            stream_result = await self.test_streaming_latency(text, show_progress=True)
            print(f"   Streaming: {stream_result.get('total_time_ms', 0):.1f}ms total, "
                  f"{stream_result.get('time_to_first_chunk_ms', 0):.1f}ms TTFC")
            
            # Non-streaming test
            non_stream_result = await self.test_non_streaming_latency(text)
            print(f"   Non-streaming: {non_stream_result.get('client_latency_ms', 0):.1f}ms")
            
            results["test_results"][f"single_request_{i+1}"] = {
                "text_length": len(text),
                "streaming": stream_result,
                "non_streaming": non_stream_result
            }
            
            await asyncio.sleep(1)  # Brief pause between tests
        
        # Concurrency tests
        print("\n🔄 Concurrency Tests")
        print("-" * 40)
        
        concurrency_levels = [1, 3, 5, 10]
        for concurrent in concurrency_levels:
            print(f"\nTesting {concurrent} concurrent requests...")
            
            # Streaming concurrency
            stream_concurrency = await self.test_concurrent_requests(
                test_texts[:concurrent], 
                concurrent, 
                "streaming"
            )
            
            print(f"   Streaming: {stream_concurrency['success_rate_percent']:.1f}% success, "
                  f"{stream_concurrency['latency_stats']['mean_ms']:.1f}ms avg")
            
            # Non-streaming concurrency
            non_stream_concurrency = await self.test_concurrent_requests(
                test_texts[:concurrent], 
                concurrent, 
                "non_streaming"
            )
            
            print(f"   Non-streaming: {non_stream_concurrency['success_rate_percent']:.1f}% success, "
                  f"{non_stream_concurrency['latency_stats']['mean_ms']:.1f}ms avg")
            
            results["test_results"][f"concurrency_{concurrent}"] = {
                "streaming": stream_concurrency,
                "non_streaming": non_stream_concurrency
            }
            
            await asyncio.sleep(2)  # Pause between concurrency levels
        
        # Performance summary
        results["summary"] = self._generate_summary(results["test_results"])
        
        return results
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {
            "best_streaming_latency_ms": float('inf'),
            "best_non_streaming_latency_ms": float('inf'),
            "max_successful_concurrency": 0,
            "overall_success_rate": 0,
            "recommendations": []
        }
        
        total_tests = 0
        successful_tests = 0
        
        for test_name, test_data in test_results.items():
            if "single_request" in test_name:
                if test_data["streaming"]["success"]:
                    latency = test_data["streaming"]["total_time_ms"]
                    summary["best_streaming_latency_ms"] = min(
                        summary["best_streaming_latency_ms"], latency
                    )
                
                if test_data["non_streaming"]["success"]:
                    latency = test_data["non_streaming"]["client_latency_ms"]
                    summary["best_non_streaming_latency_ms"] = min(
                        summary["best_non_streaming_latency_ms"], latency
                    )
            
            elif "concurrency" in test_name:
                concurrent_level = int(test_name.split("_")[1])
                
                stream_success_rate = test_data["streaming"]["success_rate_percent"]
                non_stream_success_rate = test_data["non_streaming"]["success_rate_percent"]
                
                if stream_success_rate >= 90 or non_stream_success_rate >= 90:
                    summary["max_successful_concurrency"] = max(
                        summary["max_successful_concurrency"], concurrent_level
                    )
                
                total_tests += test_data["streaming"]["total_requests"]
                total_tests += test_data["non_streaming"]["total_requests"]
                successful_tests += test_data["streaming"]["successful_requests"]
                successful_tests += test_data["non_streaming"]["successful_requests"]
        
        summary["overall_success_rate"] = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate recommendations
        if summary["best_streaming_latency_ms"] < 1000:
            summary["recommendations"].append("Excellent streaming latency - suitable for real-time applications")
        elif summary["best_streaming_latency_ms"] < 3000:
            summary["recommendations"].append("Good streaming latency - suitable for most applications")
        else:
            summary["recommendations"].append("High streaming latency - consider optimization")
        
        if summary["max_successful_concurrency"] >= 10:
            summary["recommendations"].append("Excellent concurrency handling")
        elif summary["max_successful_concurrency"] >= 5:
            summary["recommendations"].append("Good concurrency handling")
        else:
            summary["recommendations"].append("Limited concurrency - consider scaling")
        
        return summary

async def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Advanced TTS Server Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--test-type", choices=["health", "latency", "concurrency", "comprehensive"], 
                        default="comprehensive", help="Test type")
    parser.add_argument("--concurrent-users", type=int, default=10, help="Max concurrent users")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Test texts of varying lengths
    test_texts = [
        "Hello!",
        "This is a medium length test sentence for TTS evaluation.",
        "This is a longer test sentence that will help us evaluate the performance of the text-to-speech system under different load conditions and text lengths.",
        "Quick test for latency.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.",
        "Performance testing is crucial for understanding system capabilities and limitations in real-world scenarios."
    ]
    
    client = AdvancedTTSClient(args.url)
    
    try:
        await client.create_session()
        
        if args.test_type == "health":
            await client.test_health()
        
        elif args.test_type == "latency":
            print("🧪 Latency Testing")
            for text in test_texts[:3]:
                result = await client.test_streaming_latency(text, show_progress=True)
                print(f"Result: {result}")
        
        elif args.test_type == "concurrency":
            print("🧪 Concurrency Testing")
            result = await client.test_concurrent_requests(
                test_texts[:args.concurrent_users], 
                args.concurrent_users
            )
            print(json.dumps(result, indent=2))
        
        elif args.test_type == "comprehensive":
            results = await client.comprehensive_benchmark(test_texts)
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📄 Results saved to {args.output}")
            print("\n📊 Summary:")
            print(json.dumps(results["summary"], indent=2))
    
    finally:
        await client.close_session()

if __name__ == "__main__":
    asyncio.run(main())