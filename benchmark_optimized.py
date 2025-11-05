#!/usr/bin/env python3
"""
Benchmark script for optimized Orpheus TTS system
Tests concurrent performance with the ultra-fast streaming client
"""

import asyncio
import time
import statistics
from pathlib import Path
from ultra_fast_client import UltraFastOrpheusClient

# Test prompts of varying lengths
TEST_PROMPTS = [
    "Hello!",
    "How are you doing today?",
    "This is a test of the text-to-speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing concurrent audio generation with multiple requests.",
]

async def single_request(client: UltraFastOrpheusClient, prompt: str, request_id: int):
    """Make a single TTS request and measure timings"""
    start_time = time.perf_counter()
    ttfa = None
    total_time = None
    success = False
    error = None
    chunks = 0
    audio_duration = 0.0
    
    try:
        first_chunk = True
        async for audio_chunk, is_final in client.stream_generate(prompt):
            if first_chunk:
                ttfa = (time.perf_counter() - start_time) * 1000  # ms
                first_chunk = False
            chunks += 1
            audio_duration += len(audio_chunk) / 24000  # 24kHz
        
        total_time = (time.perf_counter() - start_time) * 1000  # ms
        success = True
        
    except Exception as e:
        error = str(e)
        total_time = (time.perf_counter() - start_time) * 1000
    
    return {
        'request_id': request_id,
        'prompt': prompt,
        'success': success,
        'ttfa_ms': ttfa,
        'total_time_ms': total_time,
        'chunks': chunks,
        'audio_duration_s': audio_duration,
        'error': error
    }

async def concurrent_test(num_concurrent: int, num_requests_per_worker: int = 2):
    """Test with multiple concurrent requests"""
    print(f"\n{'='*80}")
    print(f"🚀 Testing with {num_concurrent} concurrent workers")
    print(f"   {num_requests_per_worker} requests per worker = {num_concurrent * num_requests_per_worker} total requests")
    print(f"{'='*80}")
    
    async with UltraFastOrpheusClient() as client:
        # Create tasks for concurrent execution
        tasks = []
        request_id = 0
        for worker_id in range(num_concurrent):
            for req_num in range(num_requests_per_worker):
                prompt = TEST_PROMPTS[request_id % len(TEST_PROMPTS)]
                tasks.append(single_request(client, prompt, request_id))
                request_id += 1
        
        # Execute all concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.perf_counter() - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict) and r['success']]
        failed_results = [r for r in results if isinstance(r, dict) and not r['success']]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        # Calculate statistics
        if successful_results:
            ttfas = [r['ttfa_ms'] for r in successful_results]
            total_times = [r['total_time_ms'] for r in successful_results]
            
            print(f"\n✅ Successful: {len(successful_results)}/{len(results)}")
            print(f"   TTFA: {statistics.mean(ttfas):.1f}ms avg, {min(ttfas):.1f}ms min, {max(ttfas):.1f}ms max")
            print(f"   Total time: {statistics.mean(total_times):.1f}ms avg, {min(total_times):.1f}ms min, {max(total_times):.1f}ms max")
            print(f"   Throughput: {len(successful_results) / total_duration:.2f} req/s")
        
        if failed_results:
            print(f"\n❌ Failed: {len(failed_results)}")
            for r in failed_results[:3]:  # Show first 3 failures
                print(f"   - Request {r['request_id']}: {r['error']}")
        
        if exceptions:
            print(f"\n💥 Exceptions: {len(exceptions)}")
            for exc in exceptions[:3]:
                print(f"   - {type(exc).__name__}: {exc}")
        
        return {
            'num_concurrent': num_concurrent,
            'total_requests': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'exceptions': len(exceptions),
            'duration_s': total_duration,
            'results': successful_results
        }

async def main():
    """Run benchmark suite"""
    print("="*80)
    print("🎯 OPTIMIZED ORPHEUS TTS BENCHMARK")
    print("="*80)
    print("Configuration:")
    print("  - Server: 4 parallel slots, continuous batching")
    print("  - Client: Streaming + incremental SNAC decoding")
    print("  - Target: <200ms TTFA, <2s total latency")
    print("="*80)
    
    # Test different concurrency levels
    concurrency_levels = [1, 2, 4, 8, 12]
    all_results = []
    
    for num_concurrent in concurrency_levels:
        try:
            result = await concurrent_test(num_concurrent, num_requests_per_worker=2)
            all_results.append(result)
            await asyncio.sleep(1)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n\n❌ Error during benchmark: {e}")
            continue
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Concurrency':<12} {'Requests':<10} {'Success':<10} {'TTFA (ms)':<12} {'Total (ms)':<12} {'Throughput':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        if result['successful'] > 0:
            ttfas = [r['ttfa_ms'] for r in result['results']]
            totals = [r['total_time_ms'] for r in result['results']]
            throughput = result['successful'] / result['duration_s']
            
            print(f"{result['num_concurrent']:<12} "
                  f"{result['total_requests']:<10} "
                  f"{result['successful']:<10} "
                  f"{statistics.mean(ttfas):>6.1f} ± {statistics.stdev(ttfas) if len(ttfas) > 1 else 0:>4.1f}  "
                  f"{statistics.mean(totals):>6.1f} ± {statistics.stdev(totals) if len(totals) > 1 else 0:>4.1f}  "
                  f"{throughput:>6.2f} req/s")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
