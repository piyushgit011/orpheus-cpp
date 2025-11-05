"""
Comprehensive Orpheus TTS Benchmark
Tests various concurrency levels and measures detailed latency metrics
"""
import asyncio
import time
import json
import os
from typing import List, Dict, Any
from orpheus_client_fixed import OrpheusClient, ConcurrencyTester


class ComprehensiveBenchmark:
    """Run comprehensive benchmarks on Orpheus TTS system"""
    
    def __init__(self, server_url: str = "http://localhost:8090"):
        self.server_url = server_url
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    async def run_all_benchmarks(self):
        """Execute complete benchmark suite"""
        print("=" * 80)
        print("🔬 COMPREHENSIVE ORPHEUS TTS BENCHMARK")
        print("=" * 80)
        print(f"Server: {self.server_url}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        results = {
            "metadata": {
                "server_url": self.server_url,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "test_name": "comprehensive_orpheus_benchmark"
            },
            "tests": {}
        }
        
        async with OrpheusClient(server_url=self.server_url) as client:
            tester = ConcurrencyTester(client)
            
            # Test 1: Single Request Latency (different text lengths)
            print("\n📊 BENCHMARK 1: Single Request Latency Analysis")
            print("-" * 80)
            results['tests']['single_request'] = await self._benchmark_single_requests(client)
            
            # Test 2: Low Concurrency (2 users)
            print("\n📊 BENCHMARK 2: Low Concurrency (2 concurrent users)")
            print("-" * 80)
            results['tests']['concurrency_2'] = await self._benchmark_concurrency(tester, 2, 6)
            
            # Test 3: Medium Concurrency (5 users)
            print("\n📊 BENCHMARK 3: Medium Concurrency (5 concurrent users)")
            print("-" * 80)
            results['tests']['concurrency_5'] = await self._benchmark_concurrency(tester, 5, 10)
            
            # Test 4: High Concurrency (10 users)
            print("\n📊 BENCHMARK 4: High Concurrency (10 concurrent users)")
            print("-" * 80)
            results['tests']['concurrency_10'] = await self._benchmark_concurrency(tester, 10, 20)
            
            # Test 5: Stress Test (15 users)
            print("\n📊 BENCHMARK 5: Stress Test (15 concurrent users)")
            print("-" * 80)
            results['tests']['concurrency_15'] = await self._benchmark_concurrency(tester, 15, 30)
            
            # Test 6: Maximum Stress (20 users)
            print("\n📊 BENCHMARK 6: Maximum Stress (20 concurrent users)")
            print("-" * 80)
            results['tests']['concurrency_20'] = await self._benchmark_concurrency(tester, 20, 40)
        
        # Print summary
        self._print_summary(results)
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"benchmark_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
        
        return results
    
    async def _benchmark_single_requests(self, client: OrpheusClient) -> Dict[str, Any]:
        """Test single requests with various text lengths"""
        test_cases = [
            ("very_short", "Hi!", 10),
            ("short", "This is a short test sentence.", 20),
            ("medium", "Testing the Orpheus TTS system with a medium length text to evaluate performance.", 40),
            ("long", "This is a longer text to test how the system performs with extended content. " * 2, 60),
            ("very_long", "Testing with very long text to see how latency scales with content length. " * 4, 100)
        ]
        
        results = []
        
        for label, text, expected_tokens in test_cases:
            print(f"  Testing {label} ({len(text)} chars, ~{expected_tokens} expected tokens)...", end=" ")
            
            try:
                start = time.time()
                audio_data, sample_rate, stats = await client.generate_speech(text, save_path=None)
                total_time = time.time() - start
                
                audio_duration = len(audio_data) / sample_rate
                rtf = total_time / audio_duration if audio_duration > 0 else None
                
                result = {
                    "label": label,
                    "text_length_chars": len(text),
                    "text_length_words": len(text.split()),
                    "total_latency_ms": total_time * 1000,
                    "audio_duration_seconds": audio_duration,
                    "audio_samples": len(audio_data),
                    "sample_rate": sample_rate,
                    "tokens_parsed": stats.get('tokens_parsed', 0),
                    "realtime_factor": rtf,
                    "tokens_per_second": stats.get('tokens_parsed', 0) / total_time if total_time > 0 else 0
                }
                results.append(result)
                
                print(f"✅ {total_time*1000:.0f}ms (RTF: {rtf:.2f}x, tokens: {stats.get('tokens_parsed', 0)})")
                
            except Exception as e:
                print(f"❌ {e}")
                results.append({
                    "label": label,
                    "error": str(e),
                    "text_length_chars": len(text)
                })
        
        # Calculate statistics
        successful = [r for r in results if 'error' not in r]
        if successful:
            latencies = [r['total_latency_ms'] for r in successful]
            rtfs = [r['realtime_factor'] for r in successful if r['realtime_factor']]
            tps = [r['tokens_per_second'] for r in successful]
            
            summary = {
                "test_count": len(test_cases),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "results": results,
                "statistics": {
                    "latency_ms": {
                        "min": min(latencies),
                        "max": max(latencies),
                        "mean": sum(latencies) / len(latencies),
                        "median": sorted(latencies)[len(latencies) // 2]
                    },
                    "realtime_factor": {
                        "min": min(rtfs) if rtfs else None,
                        "max": max(rtfs) if rtfs else None,
                        "mean": sum(rtfs) / len(rtfs) if rtfs else None
                    },
                    "tokens_per_second": {
                        "min": min(tps) if tps else None,
                        "max": max(tps) if tps else None,
                        "mean": sum(tps) / len(tps) if tps else None
                    }
                }
            }
            
            print(f"\n  📈 Summary Statistics:")
            print(f"     Latency: {summary['statistics']['latency_ms']['min']:.0f}-{summary['statistics']['latency_ms']['max']:.0f}ms (avg: {summary['statistics']['latency_ms']['mean']:.0f}ms)")
            if summary['statistics']['realtime_factor']['mean']:
                print(f"     RTF: {summary['statistics']['realtime_factor']['mean']:.2f}x")
            if summary['statistics']['tokens_per_second']['mean']:
                print(f"     Tokens/sec: {summary['statistics']['tokens_per_second']['mean']:.1f}")
            
            return summary
        
        return {"test_count": len(test_cases), "successful": 0, "failed": len(results)}
    
    async def _benchmark_concurrency(
        self, 
        tester: ConcurrencyTester, 
        concurrent_users: int,
        num_requests: int
    ) -> Dict[str, Any]:
        """Benchmark specific concurrency level"""
        
        base_texts = [
            "Hello, testing concurrent Orpheus requests.",
            "This is a concurrency benchmark test for TTS.",
            "Measuring system performance under concurrent load.",
            "How well does Orpheus handle multiple simultaneous users?",
            "Streaming audio generation with SNAC decoding under load."
        ]
        
        test_texts = (base_texts * ((num_requests // len(base_texts)) + 1))[:num_requests]
        
        # Create output directory for this test
        test_dir = os.path.join(self.output_dir, f"concurrency_{concurrent_users}_users")
        os.makedirs(test_dir, exist_ok=True)
        
        print(f"  Running {num_requests} requests with {concurrent_users} concurrent users...")
        print(f"  Sample audio will be saved to: {test_dir}/")
        
        try:
            results = await tester.test_concurrent(
                test_texts,
                concurrent_users=concurrent_users,
                save_dir=test_dir
            )
            
            print(f"  ✅ Success rate: {results['successful_requests']}/{results['total_requests']}")
            print(f"  ⏱️  Total time: {results['total_test_time_seconds']:.2f}s")
            print(f"  🚀 Throughput: {results['requests_per_second']:.2f} req/s")
            
            if 'latency_stats' in results:
                ls = results['latency_stats']
                print(f"  📊 Latency: min={ls['min_ms']:.0f}ms, max={ls['max_ms']:.0f}ms, mean={ls['mean_ms']:.0f}ms, median={ls['median_ms']:.0f}ms")
            
            # Calculate percentiles
            if results.get('successful_requests', 0) > 0 and 'latency_stats' in results:
                print(f"  📈 Concurrency efficiency: {(results['successful_requests'] / results['total_test_time_seconds']):.2f} completions/sec")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            return {
                "error": str(e),
                "concurrent_users": concurrent_users,
                "num_requests": num_requests
            }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary of all tests"""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Single request performance
        if 'single_request' in results['tests'] and 'statistics' in results['tests']['single_request']:
            sr = results['tests']['single_request']['statistics']
            print(f"\n🎯 Single Request Performance:")
            print(f"   Latency Range: {sr['latency_ms']['min']:.0f}ms - {sr['latency_ms']['max']:.0f}ms")
            print(f"   Average Latency: {sr['latency_ms']['mean']:.0f}ms")
            if sr['realtime_factor']['mean']:
                print(f"   Realtime Factor: {sr['realtime_factor']['mean']:.2f}x")
            if sr['tokens_per_second']['mean']:
                print(f"   Token Generation Speed: {sr['tokens_per_second']['mean']:.1f} tokens/sec")
        
        # Concurrency performance
        print(f"\n🔀 Concurrency Performance:")
        concurrency_tests = [
            ('concurrency_2', '2 Users'),
            ('concurrency_5', '5 Users'),
            ('concurrency_10', '10 Users'),
            ('concurrency_15', '15 Users'),
            ('concurrency_20', '20 Users')
        ]
        
        for test_key, label in concurrency_tests:
            if test_key in results['tests']:
                test = results['tests'][test_key]
                if 'requests_per_second' in test:
                    print(f"\n   {label}:")
                    print(f"      Success Rate: {test['successful_requests']}/{test['total_requests']}")
                    print(f"      Throughput: {test['requests_per_second']:.2f} req/s")
                    if 'latency_stats' in test:
                        ls = test['latency_stats']
                        print(f"      Latency: min={ls['min_ms']:.0f}ms, avg={ls['mean_ms']:.0f}ms, max={ls['max_ms']:.0f}ms")
                    efficiency = (test['successful_requests'] / test['total_test_time_seconds'])
                    print(f"      Efficiency: {efficiency:.2f} completions/sec")
        
        print("\n" + "=" * 80)
        print("💡 Interpretation:")
        print("   - RTF (Realtime Factor): Time to generate / Audio duration")
        print("     RTF < 1.0 means faster than realtime (good for streaming)")
        print("   - Lower latency and higher throughput are better")
        print("   - Watch for latency increases at higher concurrency")
        print("=" * 80)


async def main():
    """Run comprehensive benchmark"""
    benchmark = ComprehensiveBenchmark(server_url="http://localhost:8090")
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
