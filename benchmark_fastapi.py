#!/usr/bin/env python3
"""
Benchmark script for FastAPI TTS streaming endpoint
Tests concurrent performance with streaming responses
"""

import asyncio
import aiohttp
import time
import statistics
import json
import base64
import numpy as np
from scipy.io import wavfile
from typing import List, Dict

FASTAPI_URL = "http://localhost:9100"

# Test prompts
TEST_PROMPTS = [
    "Hello!",
    "How are you doing today?",
    "This is a test of the text-to-speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing concurrent audio generation with multiple requests.",
]

async def stream_tts_request(
    session: aiohttp.ClientSession,
    text: str,
    request_id: int,
    save_audio: bool = False
) -> Dict:
    """Make a streaming TTS request and measure performance"""
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": True
    }
    
    start_time = time.perf_counter()
    ttfa = None
    chunks = 0
    total_samples = 0
    all_audio_chunks = []
    
    try:
        async with session.post(
            f"{FASTAPI_URL}/api/tts/stream",
            json=payload
        ) as response:
            
            if response.status != 200:
                error_text = await response.text()
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': f"HTTP {response.status}: {error_text}"
                }
            
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                
                if not line_str or not line_str.startswith('data: '):
                    continue
                
                data_str = line_str[6:]  # Remove 'data: ' prefix
                
                try:
                    data = json.loads(data_str)
                    
                    if data.get('event') == 'error':
                        return {
                            'request_id': request_id,
                            'success': False,
                            'error': data.get('error')
                        }
                    
                    if data.get('event') == 'complete':
                        break
                    
                    # Process audio chunk
                    if 'audio_base64' in data:
                        if ttfa is None:
                            ttfa = (time.perf_counter() - start_time) * 1000
                        
                        chunks += 1
                        samples = data.get('samples', 0)
                        total_samples += samples
                        
                        # Decode audio if saving
                        if save_audio:
                            audio_bytes = base64.b64decode(data['audio_base64'])
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                            all_audio_chunks.append(audio_array)
                
                except json.JSONDecodeError:
                    continue
        
        total_time = (time.perf_counter() - start_time) * 1000
        audio_duration = total_samples / 24000
        
        # Save audio if requested
        filename = None
        if save_audio and all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            filename = f"fastapi_test_{request_id}.wav"
            wavfile.write(filename, 24000, full_audio)
        
        return {
            'request_id': request_id,
            'success': True,
            'ttfa_ms': ttfa,
            'total_time_ms': total_time,
            'chunks': chunks,
            'audio_duration_s': audio_duration,
            'filename': filename
        }
        
    except Exception as e:
        return {
            'request_id': request_id,
            'success': False,
            'error': str(e)
        }

async def concurrent_test(num_concurrent: int, requests_per_worker: int = 2):
    """Test with multiple concurrent requests"""
    
    print(f"\n{'='*80}")
    print(f"🚀 Testing with {num_concurrent} concurrent workers")
    print(f"   {requests_per_worker} requests per worker = {num_concurrent * requests_per_worker} total")
    print(f"{'='*80}")
    
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks
        tasks = []
        request_id = 0
        
        for worker in range(num_concurrent):
            for req in range(requests_per_worker):
                text = TEST_PROMPTS[request_id % len(TEST_PROMPTS)]
                save_audio = (request_id < 3)  # Save first 3 audio files
                tasks.append(stream_tts_request(session, text, request_id, save_audio))
                request_id += 1
        
        # Execute concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.perf_counter() - start_time
        
        # Process results
        successful = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed = [r for r in results if isinstance(r, dict) and not r.get('success')]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        # Statistics
        if successful:
            ttfas = [r['ttfa_ms'] for r in successful if r.get('ttfa_ms')]
            total_times = [r['total_time_ms'] for r in successful]
            
            print(f"\n✅ Successful: {len(successful)}/{len(results)}")
            if ttfas:
                print(f"   TTFA: {statistics.mean(ttfas):.1f}ms avg, "
                      f"{min(ttfas):.1f}ms min, {max(ttfas):.1f}ms max, "
                      f"±{statistics.stdev(ttfas):.1f}ms std")
            print(f"   Total time: {statistics.mean(total_times):.1f}ms avg, "
                  f"{min(total_times):.1f}ms min, {max(total_times):.1f}ms max")
            print(f"   Throughput: {len(successful) / total_duration:.2f} req/s")
            
            # Show saved files
            saved_files = [r['filename'] for r in successful if r.get('filename')]
            if saved_files:
                print(f"   Saved audio: {', '.join(saved_files)}")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)}")
            for r in failed[:3]:
                print(f"   - Request {r['request_id']}: {r.get('error', 'Unknown error')}")
        
        if exceptions:
            print(f"\n💥 Exceptions: {len(exceptions)}")
            for exc in exceptions[:3]:
                print(f"   - {type(exc).__name__}: {exc}")
        
        return {
            'concurrency': num_concurrent,
            'total_requests': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'exceptions': len(exceptions),
            'duration_s': total_duration,
            'results': successful
        }

async def main():
    """Run benchmark suite"""
    
    print("="*80)
    print("🎯 FASTAPI TTS STREAMING BENCHMARK")
    print("="*80)
    print("Configuration:")
    print("  - Endpoint: http://localhost:8000/api/tts/stream")
    print("  - Backend: Orpheus TTS via llama-server")
    print("  - Target: <200ms TTFA, high throughput")
    print("="*80)
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{FASTAPI_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ FastAPI server is healthy")
                    print(f"   Status: {health.get('status')}")
                    print(f"   Client initialized: {health.get('client_initialized')}")
                else:
                    print(f"⚠️  Server responded with status {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to FastAPI server: {e}")
        print("   Make sure to start it with: python fastapi_server.py")
        return
    
    # Test different concurrency levels
    concurrency_levels = [1, 2, 4, 8, 12, 16]
    all_results = []
    
    for num_concurrent in concurrency_levels:
        try:
            result = await concurrent_test(num_concurrent, requests_per_worker=2)
            all_results.append(result)
            await asyncio.sleep(1)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted")
            break
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            continue
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Concurrency':<12} {'Requests':<10} {'Success':<10} {'TTFA (ms)':<15} {'Total (ms)':<15} {'Throughput':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        if result['successful'] > 0:
            ttfas = [r['ttfa_ms'] for r in result['results'] if r.get('ttfa_ms')]
            totals = [r['total_time_ms'] for r in result['results']]
            throughput = result['successful'] / result['duration_s']
            
            ttfa_str = f"{statistics.mean(ttfas):>6.1f} ± {statistics.stdev(ttfas):>4.1f}" if len(ttfas) > 1 else f"{statistics.mean(ttfas):>6.1f}" if ttfas else "N/A"
            total_str = f"{statistics.mean(totals):>6.1f} ± {statistics.stdev(totals):>4.1f}" if len(totals) > 1 else f"{statistics.mean(totals):>6.1f}"
            
            print(f"{result['concurrency']:<12} "
                  f"{result['total_requests']:<10} "
                  f"{result['successful']:<10} "
                  f"{ttfa_str:<15} "
                  f"{total_str:<15} "
                  f"{throughput:>6.2f} req/s")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
