#!/usr/bin/env python3
"""
Comprehensive test with varying input lengths
Tests 20, 30, 50+ token inputs
"""

import asyncio
import time
from hyper_fast_client import HyperFastOrpheusClient
import numpy as np
from scipy.io import wavfile

# Test prompts with different token counts
TEST_PROMPTS = [
    # Short prompts (~20 tokens)
    ("Hello, how are you doing today? I hope you're having a wonderful day!", "short_20tok"),
    
    # Medium prompts (~30 tokens)
    ("This is a test of the text-to-speech system. We're checking the latency and performance with moderate length input text.", "medium_30tok"),
    
    # Long prompts (~50 tokens)
    ("The quick brown fox jumps over the lazy dog. Testing concurrent audio generation with multiple requests. "
     "This sentence contains enough words to reach approximately fifty tokens for comprehensive testing.", "long_50tok"),
    
    # Very long prompts (70+ tokens)
    ("In the realm of artificial intelligence and machine learning, text-to-speech synthesis has become increasingly sophisticated. "
     "Modern systems can generate highly natural sounding speech with minimal latency. This test evaluates performance "
     "with longer input sequences to understand how the system scales with increased token counts.", "verylong_70tok"),
]

async def test_prompt(client: HyperFastOrpheusClient, prompt: str, label: str):
    """Test a single prompt and return metrics"""
    print(f"\n{'='*80}")
    print(f"📝 Testing: {label}")
    print(f"   Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"   Prompt: {prompt}")
    print(f"{'='*80}")
    
    try:
        start_time = time.perf_counter()
        ttfa = None
        chunks = 0
        total_audio = []
        
        async for audio_chunk, info in client.stream_generate(
            prompt,
            temperature=0.2,
            max_tokens=512
        ):
            if ttfa is None:
                ttfa = info['ttfa_ms']
            chunks += 1
            total_audio.append(audio_chunk)
        
        total_time = (time.perf_counter() - start_time) * 1000
        full_audio = np.concatenate(total_audio) if total_audio else np.array([])
        audio_duration = len(full_audio) / 24000
        
        # Save audio
        filename = None
        if len(full_audio) > 0:
            audio_int16 = (full_audio * 32767).astype(np.int16)
            filename = f"test_{label}.wav"
            wavfile.write(filename, 24000, audio_int16)
        
        result = {
            'label': label,
            'success': True,
            'ttfa_ms': ttfa,
            'total_time_ms': total_time,
            'chunks': chunks,
            'audio_duration_s': audio_duration,
            'filename': filename,
            'error': None
        }
        
        # Print results
        target_symbol = "🎯" if ttfa and ttfa < 200 else ""
        print(f"\n✅ Success!")
        if ttfa is not None:
            print(f"   TTFA: {ttfa:.1f}ms {target_symbol}")
        else:
            print(f"   ⚠️  TTFA: N/A (no audio generated)")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Chunks: {chunks}")
        print(f"   Audio: {audio_duration:.2f}s ({len(full_audio)} samples)")
        if filename:
            print(f"   Saved: {filename}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return {
            'label': label,
            'success': False,
            'error': str(e)
        }

async def main():
    """Run comprehensive test suite"""
    
    print("="*80)
    print("🎯 COMPREHENSIVE ORPHEUS TTS TEST")
    print("="*80)
    print("Testing with varying input lengths: 20, 30, 50, 70+ tokens")
    print("Server: Ultra-optimized (50 slots, 1024 batch, 32 threads)")
    print("="*80)
    
    results = []
    
    async with HyperFastOrpheusClient() as client:
        for prompt, label in TEST_PROMPTS:
            result = await test_prompt(client, prompt, label)
            results.append(result)
            await asyncio.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"{'Test':<20} {'Success':<10} {'TTFA (ms)':<12} {'Total (ms)':<12} {'Audio (s)':<10}")
    print(f"{'-'*80}")
    
    for result in results:
        if result['success']:
            ttfa_str = f"{result['ttfa_ms']:.1f}" if result['ttfa_ms'] else "N/A"
            print(f"{result['label']:<20} {'✅':<10} {ttfa_str:<12} "
                  f"{result['total_time_ms']:<12.1f} {result['audio_duration_s']:<10.2f}")
        else:
            print(f"{result['label']:<20} {'❌':<10} {'Error':<12} {'-':<12} {'-':<10}")
            print(f"   Error: {result['error']}")
    
    print(f"{'='*80}\n")
    
    # Check for issues
    successful = [r for r in results if r['success']]
    if not successful:
        print("⚠️  WARNING: No successful generations!")
        print("   This may indicate a server configuration issue.")
        print("   Check server logs for errors.")
    elif all(r.get('ttfa_ms') is None for r in successful):
        print("⚠️  WARNING: No audio tokens generated!")
        print("   Server is responding but not producing audio tokens.")
        print("   Check:")
        print("   - Server is running the correct model")
        print("   - Context size is sufficient")
        print("   - Prompt format is correct")

if __name__ == "__main__":
    asyncio.run(main())
