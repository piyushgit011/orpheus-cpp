"""
Comprehensive Benchmark for Orpheus Streaming TTS with SNAC
Measures latency, throughput, and performance under various loads
"""
import asyncio
import time
import json
import os
from typing import List, Dict, Any
from orpheus_streaming_client import OrpheusStreamingClient, ConcurrencyTester


class BenchmarkRunner:
    """Comprehensive benchmark runner for Orpheus TTS"""
    
    def __init__(self, llama_server_url: str = "http://localhost:8090"):
        self.llama_server_url = llama_server_url
        self.output_dir = "benchmark_audio_outputs"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Audio outputs will be saved to: {self.output_dir}/")
        print()
        
    async def run_all_benchmarks(self):
        """Run all benchmark tests"""
        print("=" * 80)
        print("🔬 ORPHEUS STREAMING TTS BENCHMARK SUITE")
        print("=" * 80)
        print(f"Server: {self.llama_server_url}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        results = {}
        
        async with OrpheusStreamingClient(llama_server_url=self.llama_server_url) as client:
            tester = ConcurrencyTester(client)
            
            # Benchmark 1: Single Request Latency
            print("\n📊 BENCHMARK 1: Single Request Latency")
            print("-" * 80)
            results['single_request'] = await self._benchmark_single_requests(client)
            
            # Benchmark 2: Low Concurrency (2-3 users)
            print("\n📊 BENCHMARK 2: Low Concurrency (2-3 concurrent users)")
            print("-" * 80)
            results['low_concurrency'] = await self._benchmark_concurrency(tester, concurrent_users=3, num_requests=6)
            
            # Benchmark 3: Medium Concurrency (5 users)
            print("\n📊 BENCHMARK 3: Medium Concurrency (5 concurrent users)")
            print("-" * 80)
            results['medium_concurrency'] = await self._benchmark_concurrency(tester, concurrent_users=5, num_requests=10)
            
            # Benchmark 4: High Concurrency (10 users)
            print("\n📊 BENCHMARK 4: High Concurrency (10 concurrent users)")
            print("-" * 80)
            results['high_concurrency'] = await self._benchmark_concurrency(tester, concurrent_users=10, num_requests=20)
            
            # Benchmark 5: Sustained Load (15 users)
            print("\n📊 BENCHMARK 5: Sustained Load (15 concurrent users)")
            print("-" * 80)
            results['sustained_load'] = await self._benchmark_concurrency(tester, concurrent_users=15, num_requests=30)
            
            # Benchmark 6: Text Length Variation
            print("\n📊 BENCHMARK 6: Text Length Variation")
            print("-" * 80)
            results['text_length'] = await self._benchmark_text_lengths(client)
        
        # Print summary
        self._print_summary(results)
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
        
        # Count audio files
        audio_count = sum(1 for root, dirs, files in os.walk(self.output_dir) for f in files if f.endswith('.wav'))
        print(f"🔊 Generated {audio_count} audio files in: {self.output_dir}/")
        print(f"   You can play them to verify audio quality!")
        
        return results
    
    async def _benchmark_single_requests(self, client: OrpheusStreamingClient) -> Dict[str, Any]:
        """Benchmark single requests with different texts"""
        test_texts = [
            "Hello!",
            "This is a short test.",
            "Testing the Orpheus streaming system with SNAC decoding.",
            "The quick brown fox jumps over the lazy dog in the summer afternoon.",
            "This is a longer text to test how the system performs with more content and extended generation."
        ]
        
        results = []
        
        for i, text in enumerate(test_texts):
            print(f"  Test {i+1}/{len(test_texts)}: {len(text)} chars...", end=" ")
            
            try:
                start = time.time()
                output_file = os.path.join(self.output_dir, f"single_request_{i+1}_{len(text)}chars.wav")
                sr, audio, stats = await client.tts(text, save_to_file=output_file)
                total_time = time.time() - start
                
                result = {
                    "text_length": len(text),
                    "total_latency_ms": total_time * 1000,
                    "time_to_first_chunk_ms": stats.get('time_to_first_chunk_ms'),
                    "chunk_count": stats.get('chunk_count'),
                    "audio_duration_seconds": stats.get('audio_duration_seconds'),
                    "audio_samples": len(audio),
                    "sample_rate": sr,
                    "realtime_factor": (total_time / stats['audio_duration_seconds']) if stats['audio_duration_seconds'] > 0 else None
                }
                results.append(result)
                
                print(f"✅ {result['total_latency_ms']:.0f}ms (RTF: {result['realtime_factor']:.2f}x)" if result['realtime_factor'] else f"✅ {result['total_latency_ms']:.0f}ms")
                
            except Exception as e:
                print(f"❌ {e}")
                results.append({"error": str(e), "text_length": len(text)})
        
        # Calculate statistics
        successful = [r for r in results if 'error' not in r]
        if successful:
            latencies = [r['total_latency_ms'] for r in successful]
            ttfc_list = [r['time_to_first_chunk_ms'] for r in successful if r['time_to_first_chunk_ms']]
            rtf_list = [r['realtime_factor'] for r in successful if r['realtime_factor']]
            
            stats = {
                "test_count": len(test_texts),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "latency_ms": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "median": sorted(latencies)[len(latencies) // 2]
                },
                "realtime_factor": {
                    "min": min(rtf_list) if rtf_list else None,
                    "max": max(rtf_list) if rtf_list else None,
                    "mean": sum(rtf_list) / len(rtf_list) if rtf_list else None
                }
            }
            
            if ttfc_list:
                stats["time_to_first_chunk_ms"] = {
                    "min": min(ttfc_list),
                    "max": max(ttfc_list),
                    "mean": sum(ttfc_list) / len(ttfc_list)
                }
            
            print(f"\n  📈 Statistics:")
            print(f"     Latency: {stats['latency_ms']['min']:.0f}ms - {stats['latency_ms']['max']:.0f}ms (avg: {stats['latency_ms']['mean']:.0f}ms)")
            if stats['realtime_factor']['mean']:
                print(f"     Realtime Factor: {stats['realtime_factor']['mean']:.2f}x")
            
            return stats
        
        return {"test_count": len(test_texts), "successful": 0, "failed": len(results)}
    
    async def _benchmark_concurrency(
        self, 
        tester: ConcurrencyTester, 
        concurrent_users: int, 
        num_requests: int
    ) -> Dict[str, Any]:
        """Benchmark with specific concurrency level"""
        
        # Generate test texts
        base_texts = [
            "Hello, testing concurrent requests.",
            "This is a concurrency benchmark test.",
            "Measuring system performance under load.",
            "How well does Orpheus handle multiple users?",
            "Streaming audio generation with SNAC decoding."
        ]
        
        # Repeat texts to reach num_requests
        test_texts = (base_texts * ((num_requests // len(base_texts)) + 1))[:num_requests]
        
        # Save a sample from concurrency test
        concurrency_dir = os.path.join(self.output_dir, f"concurrency_{concurrent_users}users")
        os.makedirs(concurrency_dir, exist_ok=True)
        
        print(f"  Running {num_requests} requests with {concurrent_users} concurrent users...")
        print(f"  Sample audio will be saved to: {concurrency_dir}/")
        
        try:
            results = await tester.test_concurrent(
                test_texts,
                concurrent_users=concurrent_users,
                save_dir=concurrency_dir
            )
            
            print(f"  ✅ Success rate: {results['successful_requests']}/{results['total_requests']}")
            print(f"  ⏱️  Total time: {results['total_test_time_seconds']:.2f}s")
            print(f"  🚀 Throughput: {results['requests_per_second']:.2f} req/s")
            
            if 'latency_stats' in results:
                ls = results['latency_stats']
                print(f"  📊 Latency: min={ls['min_ms']:.0f}ms, max={ls['max_ms']:.0f}ms, mean={ls['mean_ms']:.0f}ms")
            
            if 'time_to_first_chunk_stats' in results:
                ttfc = results['time_to_first_chunk_stats']
                print(f"  ⚡ TTFC: min={ttfc['min_ms']:.0f}ms, max={ttfc['max_ms']:.0f}ms, mean={ttfc['mean_ms']:.0f}ms")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            return {"error": str(e), "concurrent_users": concurrent_users, "num_requests": num_requests}
    
    async def _benchmark_text_lengths(self, client: OrpheusStreamingClient) -> Dict[str, Any]:
        """Benchmark performance with different text lengths"""
        
        test_cases = [
            ("very_short", "Hi!"),
            ("short", "This is a short test sentence."),
            ("medium", "This is a medium length text that should generate a reasonable amount of audio. " * 2),
            ("long", "This is a longer text for testing extended generation capabilities. " * 5),
            ("very_long", "Testing with a very long text to see how the system scales with content length. " * 10)
        ]
        
        results = {}
        
        for label, text in test_cases:
            print(f"  Testing {label} text ({len(text)} chars)...", end=" ")
            
            try:
                start = time.time()
                output_file = os.path.join(self.output_dir, f"text_length_{label}.wav")
                sr, audio, stats = await client.tts(text, save_to_file=output_file)
                total_time = time.time() - start
                
                rtf = (total_time / stats['audio_duration_seconds']) if stats['audio_duration_seconds'] > 0 else None
                
                results[label] = {
                    "text_length": len(text),
                    "total_latency_ms": total_time * 1000,
                    "time_to_first_chunk_ms": stats.get('time_to_first_chunk_ms'),
                    "audio_duration_seconds": stats.get('audio_duration_seconds'),
                    "realtime_factor": rtf,
                    "chunk_count": stats.get('chunk_count')
                }
                
                print(f"✅ {total_time*1000:.0f}ms (RTF: {rtf:.2f}x)" if rtf else f"✅ {total_time*1000:.0f}ms")
                
            except Exception as e:
                print(f"❌ {e}")
                results[label] = {"error": str(e), "text_length": len(text)}
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Single request performance
        if 'single_request' in results and 'latency_ms' in results['single_request']:
            sr = results['single_request']
            print(f"\n🎯 Single Request Performance:")
            print(f"   Latency: {sr['latency_ms']['mean']:.0f}ms (range: {sr['latency_ms']['min']:.0f}-{sr['latency_ms']['max']:.0f}ms)")
            if sr['realtime_factor']['mean']:
                print(f"   Realtime Factor: {sr['realtime_factor']['mean']:.2f}x")
        
        # Concurrency performance
        print(f"\n🔀 Concurrency Performance:")
        for test_name in ['low_concurrency', 'medium_concurrency', 'high_concurrency', 'sustained_load']:
            if test_name in results and 'requests_per_second' in results[test_name]:
                test = results[test_name]
                label = test_name.replace('_', ' ').title()
                print(f"   {label}:")
                print(f"      Throughput: {test['requests_per_second']:.2f} req/s")
                print(f"      Success Rate: {test['successful_requests']}/{test['total_requests']}")
                if 'latency_stats' in test:
                    print(f"      Avg Latency: {test['latency_stats']['mean_ms']:.0f}ms")
        
        # Text length performance
        if 'text_length' in results:
            print(f"\n📏 Text Length Performance:")
            for label, data in results['text_length'].items():
                if 'error' not in data:
                    rtf_str = f", RTF: {data['realtime_factor']:.2f}x" if data.get('realtime_factor') else ""
                    print(f"   {label.replace('_', ' ').title()}: {data['total_latency_ms']:.0f}ms{rtf_str}")
        
        print("\n" + "=" * 80)


async def main():
    """Run benchmark suite"""
    benchmark = BenchmarkRunner(llama_server_url="http://localhost:8090")
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
