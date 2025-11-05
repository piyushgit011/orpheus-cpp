"""
Orpheus Streaming TTS Client with SNAC Decoder
Connects to llama-server on port 8090 and decodes SNAC codes to audio
"""
import asyncio
import aiohttp
import numpy as np
import json
import time
import re
from typing import AsyncGenerator, Tuple, Dict, Any, List, Optional
from scipy.io.wavfile import write
import onnxruntime as ort
from huggingface_hub import hf_hub_download


class SNACDecoder:
    """Optimized SNAC decoder with anti-popping crossfade"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        
        # Download SNAC decoder model
        print("📥 Downloading SNAC decoder model...")
        repo_id = "onnx-community/snac_24khz-ONNX"
        snac_model_file = "decoder_model.onnx"
        snac_model_path = hf_hub_download(
            repo_id, subfolder="onnx", filename=snac_model_file
        )
        print(f"✅ SNAC model loaded from: {snac_model_path}")
        
        # Optimized ONNX session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            snac_model_path,
            sess_options=session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Crossfade buffer for smooth transitions
        self.overlap_samples = int(0.01 * sample_rate)  # 10ms overlap
        self.window_buffer = np.zeros(self.overlap_samples, dtype=np.float32)
        
        print(f"🔧 SNAC Decoder initialized with {self.session.get_providers()} providers")
    
    def decode_snac_codes(self, codes: List[np.ndarray]) -> np.ndarray:
        """
        Decode SNAC codes to audio with crossfade smoothing
        
        Args:
            codes: List of numpy arrays representing SNAC codes [code0, code1, code2, ...]
        
        Returns:
            Audio samples as int16 numpy array
        """
        try:
            # Prepare input dictionary for ONNX model
            input_dict = {}
            for i, inp in enumerate(self.session.get_inputs()):
                if i < len(codes):
                    input_dict[inp.name] = codes[i]
                else:
                    # Pad with zeros if not enough codes
                    input_dict[inp.name] = np.zeros((1, 1), dtype=np.int64)
            
            # Run ONNX inference
            audio_output = self.session.run(None, input_dict)[0]
            
            # Convert to audio samples
            if audio_output.ndim == 3:
                audio_samples = audio_output[0, 0, :]
            elif audio_output.ndim == 2:
                audio_samples = audio_output[0, :]
            else:
                audio_samples = audio_output.squeeze()
            
            # Convert to int16
            if audio_samples.dtype != np.int16:
                audio_samples = (audio_samples * 32767).astype(np.int16)
            
            # Apply crossfade to prevent popping
            audio_samples = self._apply_crossfade(audio_samples)
            
            return audio_samples
            
        except Exception as e:
            print(f"⚠️  SNAC decode error: {e}")
            return np.array([], dtype=np.int16)
    
    def _apply_crossfade(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply crossfade between chunks to prevent audio popping"""
        if len(audio_chunk) == 0:
            return audio_chunk
        
        fade_length = min(len(audio_chunk), len(self.window_buffer))
        
        if fade_length == 0:
            return audio_chunk
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
        
        # Apply crossfade
        result = audio_chunk.copy().astype(np.float32)
        if len(self.window_buffer) >= fade_length:
            result[:fade_length] = (
                result[:fade_length] * fade_in + 
                self.window_buffer[:fade_length] * fade_out
            )
        
        # Update window buffer
        if len(result) >= self.overlap_samples:
            self.window_buffer = result[-self.overlap_samples:].copy()
        
        return result.astype(np.int16)


class OrpheusStreamingClient:
    """Client for Orpheus llama-server with SNAC decoding"""
    
    def __init__(self, llama_server_url: str = "http://localhost:8090"):
        self.llama_server_url = llama_server_url.rstrip("/")
        self.snac_decoder = SNACDecoder()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _format_prompt(self, text: str, voice_id: str = "tara") -> str:
        """Format prompt for Orpheus TTS model"""
        # Correct Orpheus format from working code: <|audio|>voice: text<|eot_id|>
        return f"<|audio|>{voice_id}: {text}<|eot_id|>"
    
    def _parse_snac_codes(self, text: str) -> List[np.ndarray]:
        """
        Parse SNAC codes from model output
        Expected format: <|audio_0|> <|audio_1|> ... or [S_xxx] tokens
        """
        codes = []
        
        # Pattern 1: <|audio_N|> format
        audio_pattern = r'<\|audio_(\d+)\|>'
        matches = re.findall(audio_pattern, text)
        
        if matches:
            # Convert to numpy arrays
            for match in matches:
                code_value = int(match)
                codes.append(np.array([[code_value]], dtype=np.int64))
        
        # Pattern 2: [S_xxx] format (alternative SNAC token format)
        if not codes:
            snac_pattern = r'\[S_(\d+)\]'
            matches = re.findall(snac_pattern, text)
            if matches:
                for match in matches:
                    code_value = int(match)
                    codes.append(np.array([[code_value]], dtype=np.int64))
        
        # Pattern 3: Simple number sequences (fallback)
        if not codes:
            # Try to extract numbers that could be SNAC codes
            number_pattern = r'(\d+)'
            matches = re.findall(number_pattern, text)
            if matches:
                for match in matches[:100]:  # Limit to reasonable number
                    code_value = int(match)
                    if 0 <= code_value < 5000:  # SNAC codes are typically in this range
                        codes.append(np.array([[code_value]], dtype=np.int64))
        
        return codes
    
    async def stream_tts(
        self,
        text: str,
        voice_id: str = "tara",
        temperature: float = 0.8,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> AsyncGenerator[Tuple[int, np.ndarray, Dict[str, Any]], None]:
        """
        Stream TTS from Orpheus llama-server with SNAC decoding
        
        Yields:
            Tuple of (sample_rate, audio_chunk, metadata)
        """
        prompt = self._format_prompt(text, voice_id)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "cache_prompt": True
        }
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        audio_chunk_count = 0
        
        # Buffer for accumulating tokens before decoding
        token_buffer = ""
        min_tokens_for_decode = 50  # Decode every 50 tokens for lower latency
        
        try:
            async with self.session.post(
                f"{self.llama_server_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Server error {response.status}: {error_text}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line or not line.startswith('data: '):
                        continue
                    
                    # Remove 'data: ' prefix
                    json_str = line[6:]
                    
                    if json_str == '[DONE]':
                        # Decode any remaining tokens
                        if token_buffer:
                            codes = self._parse_snac_codes(token_buffer)
                            if codes:
                                audio_samples = self.snac_decoder.decode_snac_codes(codes)
                                if len(audio_samples) > 0:
                                    audio_chunk_count += 1
                                    yield (
                                        self.snac_decoder.sample_rate,
                                        audio_samples,
                                        {
                                            "chunk_index": audio_chunk_count,
                                            "is_final": True,
                                            "elapsed_ms": (time.time() - start_time) * 1000
                                        }
                                    )
                        break
                    
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                    
                    # Track first token timing
                    if first_token_time is None and 'content' in data:
                        first_token_time = time.time()
                    
                    # Accumulate tokens
                    if 'content' in data:
                        token_buffer += data['content']
                        token_count += 1
                        
                        # Decode when we have enough tokens (lower latency streaming)
                        if token_count % min_tokens_for_decode == 0:
                            codes = self._parse_snac_codes(token_buffer)
                            if codes:
                                audio_samples = self.snac_decoder.decode_snac_codes(codes)
                                if len(audio_samples) > 0:
                                    audio_chunk_count += 1
                                    
                                    metadata = {
                                        "chunk_index": audio_chunk_count,
                                        "tokens_processed": token_count,
                                        "is_final": False,
                                        "elapsed_ms": (time.time() - start_time) * 1000,
                                        "time_to_first_token_ms": (first_token_time - start_time) * 1000 if first_token_time else None
                                    }
                                    
                                    yield (
                                        self.snac_decoder.sample_rate,
                                        audio_samples,
                                        metadata
                                    )
                                    
                                    # Clear buffer after successful decode
                                    token_buffer = ""
        
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            raise
    
    async def tts(
        self,
        text: str,
        voice_id: str = "tara",
        save_to_file: Optional[str] = None
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """
        Non-streaming TTS - collects all audio chunks
        
        Returns:
            Tuple of (sample_rate, full_audio_array, stats)
        """
        all_audio = []
        sample_rate = 24000
        
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        
        async for sr, audio_chunk, metadata in self.stream_tts(text, voice_id):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            
            all_audio.append(audio_chunk)
            sample_rate = sr
            chunk_count += 1
        
        total_time = time.time() - start_time
        
        # Concatenate all audio
        if all_audio:
            full_audio = np.concatenate(all_audio)
        else:
            full_audio = np.array([], dtype=np.int16)
        
        stats = {
            "total_time_ms": total_time * 1000,
            "time_to_first_chunk_ms": (first_chunk_time - start_time) * 1000 if first_chunk_time else None,
            "chunk_count": chunk_count,
            "audio_duration_seconds": len(full_audio) / sample_rate if len(full_audio) > 0 else 0,
            "sample_rate": sample_rate
        }
        
        # Save to file if requested
        if save_to_file and len(full_audio) > 0:
            write(save_to_file, sample_rate, full_audio)
            print(f"💾 Audio saved to: {save_to_file}")
        
        return sample_rate, full_audio, stats


# Concurrency testing
class ConcurrencyTester:
    """Test concurrent TTS requests with latency measurements"""
    
    def __init__(self, client: OrpheusStreamingClient):
        self.client = client
    
    async def single_request(
        self,
        text: str,
        request_id: int,
        voice_id: str = "tara",
        save_to_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a single TTS request and measure latency"""
        start_time = time.time()
        
        try:
            sample_rate, audio, stats = await self.client.tts(text, voice_id, save_to_file=save_to_file)
            
            total_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "text": text,
                "total_latency_ms": total_time * 1000,
                "time_to_first_chunk_ms": stats.get("time_to_first_chunk_ms"),
                "chunk_count": stats.get("chunk_count"),
                "audio_duration_seconds": stats.get("audio_duration_seconds"),
                "audio_samples": len(audio),
                "sample_rate": sample_rate
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "text": text,
                "error": str(e),
                "total_latency_ms": (time.time() - start_time) * 1000
            }
    
    async def test_concurrent(
        self,
        texts: List[str],
        concurrent_users: int = 5,
        voice_id: str = "tara",
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test multiple concurrent requests and measure performance
        
        Args:
            texts: List of text strings to synthesize
            concurrent_users: Number of concurrent requests
            voice_id: Voice to use
            save_dir: Optional directory to save sample audio files
        
        Returns:
            Performance statistics
        """
        print(f"\n🧪 Testing {concurrent_users} concurrent requests...")
        print(f"📝 Total texts: {len(texts)}")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def limited_request(text: str, req_id: int):
            async with semaphore:
                # Save first 3 requests as samples
                save_file = None
                if save_dir and req_id < 3:
                    import os
                    save_file = os.path.join(save_dir, f"sample_{req_id+1}.wav")
                return await self.single_request(text, req_id, voice_id, save_to_file=save_file)
        
        # Execute all requests
        start_time = time.time()
        tasks = [limited_request(text, i) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        # Calculate statistics
        successful = [r for r in processed_results if r.get("success")]
        failed = [r for r in processed_results if not r.get("success")]
        
        latencies = [r["total_latency_ms"] for r in successful]
        ttfc = [r["time_to_first_chunk_ms"] for r in successful if r.get("time_to_first_chunk_ms")]
        
        stats = {
            "test_type": "concurrent_streaming",
            "concurrent_users": concurrent_users,
            "total_requests": len(texts),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_test_time_seconds": total_time,
            "requests_per_second": len(texts) / total_time if total_time > 0 else 0
        }
        
        if latencies:
            stats["latency_stats"] = {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "mean_ms": sum(latencies) / len(latencies),
                "median_ms": sorted(latencies)[len(latencies) // 2]
            }
        
        if ttfc:
            stats["time_to_first_chunk_stats"] = {
                "min_ms": min(ttfc),
                "max_ms": max(ttfc),
                "mean_ms": sum(ttfc) / len(ttfc),
                "median_ms": sorted(ttfc)[len(ttfc) // 2]
            }
        
        if failed:
            stats["errors"] = [r.get("error") for r in failed]
        
        return stats


async def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 Orpheus Streaming TTS Client with SNAC Decoder")
    print("=" * 60)
    
    # Test texts of varying lengths
    test_texts = [
        "Hello, world!",
        "This is a test of the streaming system.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing concurrent audio generation.",
        "How well does this system perform under load?"
    ]
    
    async with OrpheusStreamingClient(llama_server_url="http://localhost:8090") as client:
        
        # Test 1: Single streaming request
        print("\n" + "=" * 60)
        print("🧪 Test 1: Single Streaming Request")
        print("=" * 60)
        print("📝 Text: 'Hello, this is a streaming test with SNAC decoding.'")
        
        try:
            sr, audio, stats = await client.tts(
                "Hello, this is a streaming test with SNAC decoding.",
                save_to_file="test_streaming_snac.wav"
            )
            
            print(f"✅ Success!")
            ttfc = stats.get('time_to_first_chunk_ms')
            if ttfc is not None:
                print(f"   Time to first chunk: {ttfc:.1f}ms")
            else:
                print(f"   Time to first chunk: N/A")
            print(f"   Total time: {stats['total_time_ms']:.1f}ms")
            print(f"   Chunks: {stats['chunk_count']}")
            print(f"   Audio duration: {stats['audio_duration_seconds']:.2f}s")
            print(f"   Sample rate: {stats['sample_rate']}Hz")
            print(f"   Audio samples: {len(audio)}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        # Test 2: Concurrency test with 3 concurrent users
        print("\n" + "=" * 60)
        print("🧪 Test 2: Concurrency Test (3 users)")
        print("=" * 60)
        
        tester = ConcurrencyTester(client)
        
        try:
            results = await tester.test_concurrent(
                test_texts[:3],
                concurrent_users=3
            )
            
            print(f"\n📊 Results:")
            print(f"   Success rate: {results['successful_requests']}/{results['total_requests']}")
            print(f"   Total time: {results['total_test_time_seconds']:.2f}s")
            print(f"   Requests/sec: {results['requests_per_second']:.2f}")
            
            if 'latency_stats' in results:
                ls = results['latency_stats']
                print(f"   Latency - Min: {ls['min_ms']:.1f}ms, Max: {ls['max_ms']:.1f}ms, Mean: {ls['mean_ms']:.1f}ms")
            
            if 'time_to_first_chunk_stats' in results:
                ttfc = results['time_to_first_chunk_stats']
                print(f"   TTFC - Min: {ttfc['min_ms']:.1f}ms, Max: {ttfc['max_ms']:.1f}ms, Mean: {ttfc['mean_ms']:.1f}ms")
            
            if results.get('errors'):
                print(f"   ⚠️  Errors: {results['errors'][:3]}")
        
        except Exception as e:
            print(f"❌ Concurrency test failed: {e}")
        
        # Test 3: Higher concurrency (5 users)
        print("\n" + "=" * 60)
        print("🧪 Test 3: Higher Concurrency (5 users)")
        print("=" * 60)
        
        try:
            results = await tester.test_concurrent(
                test_texts,
                concurrent_users=5
            )
            
            print(f"\n📊 Results:")
            print(f"   Success rate: {results['successful_requests']}/{results['total_requests']}")
            print(f"   Total time: {results['total_test_time_seconds']:.2f}s")
            print(f"   Requests/sec: {results['requests_per_second']:.2f}")
            
            if 'latency_stats' in results:
                ls = results['latency_stats']
                print(f"   Latency - Min: {ls['min_ms']:.1f}ms, Max: {ls['max_ms']:.1f}ms, Mean: {ls['mean_ms']:.1f}ms")
            
            if 'time_to_first_chunk_stats' in results:
                ttfc = results['time_to_first_chunk_stats']
                print(f"   TTFC - Min: {ttfc['min_ms']:.1f}ms, Max: {ttfc['max_ms']:.1f}ms, Mean: {ttfc['mean_ms']:.1f}ms")
        
        except Exception as e:
            print(f"❌ Higher concurrency test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\n🔊 Audio files generated:")
    print("   - test_streaming_snac.wav (single streaming test)")
    print("\n💡 Play the audio files to verify they sound correct!")


if __name__ == "__main__":
    asyncio.run(main())
