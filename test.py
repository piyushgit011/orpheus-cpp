import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor
import wave
import os
from datetime import datetime

class TTSClientTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def create_session(self):
        """Create HTTP session with optimal settings"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        )
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def test_streaming_latency(self, prompt: str, voice: str = "tara") -> Dict:
        """Test streaming endpoint latency"""
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        total_bytes = 0
        
        payload = {
            "prompt": prompt,
            "voice": voice,
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 2000,
            "repetition_penalty": 1.1
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/tts/stream",
                json=payload
            ) as response:
                
                if response.status != 200:
                    return {
                        "error": f"HTTP {response.status}",
                        "success": False
                    }
                
                async for chunk in response.content.iter_chunked(1024):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    
                    total_chunks += 1
                    total_bytes += len(chunk)
                
                end_time = time.time()
                
                return {
                    "success": True,
                    "total_time": end_time - start_time,
                    "time_to_first_chunk": first_chunk_time - start_time if first_chunk_time else None,
                    "total_chunks": total_chunks,
                    "total_bytes": total_bytes,
                    "avg_chunk_size": total_bytes / total_chunks if total_chunks > 0 else 0
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    async def test_wav_latency(self, prompt: str, voice: str = "tara") -> Dict:
        """Test WAV endpoint latency"""
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "voice": voice,
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 2000,
            "repetition_penalty": 1.1
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/tts/wav",
                json=payload
            ) as response:
                
                if response.status != 200:
                    return {
                        "error": f"HTTP {response.status}",
                        "success": False
                    }
                
                audio_data = await response.read()
                end_time = time.time()
                
                return {
                    "success": True,
                    "total_time": end_time - start_time,
                    "total_bytes": len(audio_data),
                    "audio_duration": self._calculate_audio_duration(audio_data)
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _calculate_audio_duration(self, wav_data: bytes) -> float:
        """Calculate audio duration from WAV data"""
        try:
            # Skip WAV header (44 bytes) and calculate duration
            # Assuming 24kHz, 16-bit, mono
            audio_data_size = len(wav_data) - 44
            bytes_per_sample = 2  # 16-bit
            sample_rate = 24000
            duration = audio_data_size / (bytes_per_sample * sample_rate)
            return duration
        except:
            return 0.0
    
    async def test_concurrent_requests(self, prompts: List[str], max_concurrent: int = 10) -> Dict:
        """Test concurrent request handling"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def make_request(prompt: str, request_id: int) -> Dict:
            async with semaphore:
                result = await self.test_streaming_latency(prompt)
                result['request_id'] = request_id
                return result
        
        start_time = time.time()
        
        tasks = [
            make_request(prompts[i % len(prompts)], i)
            for i in range(max_concurrent)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Process results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_requests = [r for r in results if not (isinstance(r, dict) and r.get('success'))]
        
        return {
            "total_requests": len(tasks),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "total_time": end_time - start_time,
            "avg_latency": statistics.mean([r['total_time'] for r in successful_requests]) if successful_requests else 0,
            "min_latency": min([r['total_time'] for r in successful_requests]) if successful_requests else 0,
            "max_latency": max([r['total_time'] for r in successful_requests]) if successful_requests else 0,
            "success_rate": len(successful_requests) / len(tasks) * 100,
            "results": results
        }
    
    async def find_concurrency_limit(self, prompt: str, max_test: int = 50) -> Dict:
        """Find the maximum concurrency limit"""
        results = {}
        
        for concurrent_count in range(1, max_test + 1):
            print(f"Testing {concurrent_count} concurrent requests...")
            
            test_result = await self.test_concurrent_requests([prompt], concurrent_count)
            
            results[concurrent_count] = {
                "success_rate": test_result["success_rate"],
                "avg_latency": test_result["avg_latency"],
                "total_time": test_result["total_time"]
            }
            
            # If success rate drops below 90%, consider this the limit
            if test_result["success_rate"] < 90:
                print(f"Concurrency limit reached at {concurrent_count} requests")
                break
            
            # Add small delay between tests
            await asyncio.sleep(2)
        
        return results
    
    async def comprehensive_test(self, prompts: List[str]) -> Dict:
        """Run comprehensive latency and concurrency tests"""
        print("Starting comprehensive TTS testing...")
        
        # Test different prompt lengths
        latency_results = {}
        for i, prompt in enumerate(prompts):
            print(f"Testing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            # Test streaming
            stream_result = await self.test_streaming_latency(prompt)
            
            # Test WAV
            wav_result = await self.test_wav_latency(prompt)
            
            latency_results[f"prompt_{i+1}"] = {
                "prompt": prompt,
                "prompt_length": len(prompt),
                "streaming": stream_result,
                "wav": wav_result
            }
            
            await asyncio.sleep(1)  # Small delay between tests
        
        # Test concurrency with a medium prompt
        medium_prompt = prompts[len(prompts)//2] if prompts else "Hello, this is a test message."
        
        print("Testing concurrency limits...")
        concurrency_results = {}
        
        # Test with different concurrency levels
        for concurrent in [1, 5, 10, 20]:
            print(f"Testing {concurrent} concurrent requests...")
            result = await self.test_concurrent_requests([medium_prompt], concurrent)
            concurrency_results[f"concurrent_{concurrent}"] = result
            await asyncio.sleep(2)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "latency_tests": latency_results,
            "concurrency_tests": concurrency_results,
            "summary": self._generate_summary(latency_results, concurrency_results)
        }
    
    def _generate_summary(self, latency_results: Dict, concurrency_results: Dict) -> Dict:
        """Generate test summary"""
        successful_streaming = [
            r["streaming"] for r in latency_results.values() 
            if r["streaming"].get("success")
        ]
        
        successful_wav = [
            r["wav"] for r in latency_results.values() 
            if r["wav"].get("success")
        ]
        
        summary = {
            "streaming_tests": {
                "total": len(latency_results),
                "successful": len(successful_streaming),
                "avg_latency": statistics.mean([r["total_time"] for r in successful_streaming]) if successful_streaming else 0,
                "avg_time_to_first_chunk": statistics.mean([r["time_to_first_chunk"] for r in successful_streaming if r["time_to_first_chunk"]]) if successful_streaming else 0
            },
            "wav_tests": {
                "total": len(latency_results),
                "successful": len(successful_wav),
                "avg_latency": statistics.mean([r["total_time"] for r in successful_wav]) if successful_wav else 0
            },
            "concurrency_tests": {
                "max_tested": max([int(k.split('_')[1]) for k in concurrency_results.keys()]) if concurrency_results else 0,
                "best_performance": max(concurrency_results.items(), key=lambda x: x[1]["success_rate"]) if concurrency_results else None
            }
        }
        
        return summary
    
    async def save_audio_sample(self, prompt: str, filename: str = "sample.wav"):
        """Save a sample audio file"""
        result = await self.test_wav_latency(prompt)
        
        if result["success"]:
            payload = {
                "prompt": prompt,
                "voice": "tara",
                "temperature": 0.6,
                "top_p": 0.8,
                "max_tokens": 2000,
                "repetition_penalty": 1.1
            }
            
            async with self.session.post(f"{self.base_url}/tts/wav", json=payload) as response:
                if response.status == 200:
                    with open(filename, 'wb') as f:
                        f.write(await response.read())
                    print(f"Sample audio saved as {filename}")
                    return True
        
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test TTS API latency and concurrency")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of TTS API")
    parser.add_argument("--test-type", choices=["latency", "concurrency", "comprehensive"], default="comprehensive", help="Type of test to run")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Maximum concurrent requests to test")
    parser.add_argument("--output", default="test_results.json", help="Output file for results")
    parser.add_argument("--save-sample", action="store_true", help="Save sample audio file")
    
    args = parser.parse_args()
    
    # Test prompts of different lengths
    test_prompts = [
        "Hello world!",
        "This is a medium length prompt to test the TTS system with some more content.",
        "This is a much longer prompt designed to test how the text-to-speech system handles longer inputs with multiple sentences. It should provide a good test case for measuring latency and audio quality with extended content that spans across multiple thoughts and ideas.",
        "Short test.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet and is commonly used for testing purposes in typography and speech synthesis systems."
    ]
    
    tester = TTSClientTester(args.url)
    
    try:
        await tester.create_session()
        
        # Check if server is healthy
        try:
            async with tester.session.get(f"{args.url}/health") as response:
                if response.status != 200:
                    print("ERROR: Server health check failed")
                    return
                health_data = await response.json()
                print(f"Server status: {health_data}")
        except Exception as e:
            print(f"ERROR: Cannot connect to server: {e}")
            return
        
        if args.test_type == "latency":
            print("Running latency tests...")
            for i, prompt in enumerate(test_prompts):
                result = await tester.test_streaming_latency(prompt)
                print(f"Prompt {i+1}: {result}")
                
        elif args.test_type == "concurrency":
            print("Running concurrency tests...")
            result = await tester.find_concurrency_limit(test_prompts[1], args.max_concurrent)
            print(f"Concurrency results: {result}")
            
        elif args.test_type == "comprehensive":
            print("Running comprehensive tests...")
            results = await tester.comprehensive_test(test_prompts)
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {args.output}")
            print("\nSummary:")
            print(json.dumps(results["summary"], indent=2))
        
        # Save sample audio if requested
        if args.save_sample:
            await tester.save_audio_sample(test_prompts[1], "sample_output.wav")
        
    finally:
        await tester.close_session()

if __name__ == "__main__":
    asyncio.run(main())