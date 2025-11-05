"""
Fixed Orpheus Client - Using correct token parsing from working Gradio code
Connects to llama-server (port 8090) or LM Studio and generates audio with SNAC
"""
import asyncio
import aiohttp
import numpy as np
import json
import time
import re
import os
from typing import AsyncGenerator, Tuple, Dict, Any, List, Optional
from scipy.io.wavfile import write
import torch
from snac import SNAC


# Constants from working code
ORPHEUS_MIN_ID = 10
ORPHEUS_TOKENS_PER_LAYER = 4096
ORPHEUS_N_LAYERS = 7
ORPHEUS_MAX_ID = ORPHEUS_MIN_ID + (ORPHEUS_N_LAYERS * ORPHEUS_TOKENS_PER_LAYER)


class OrpheusClient:
    """Client for Orpheus TTS using llama-server/LM Studio"""
    
    def __init__(self, server_url: str = "http://localhost:8090"):
        self.server_url = server_url.rstrip("/")
        self.session = None
        
        # Load SNAC model
        print("📥 Loading SNAC vocoder model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(device)
        self.snac_model.eval()
        print(f"✅ SNAC model loaded on {device}")
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def parse_gguf_codes(self, response_text: str) -> List[int]:
        """
        Parse SNAC codes from GGUF model output
        Based on working Gradio code - looks for <custom_token_N> format
        """
        absolute_ids = []
        matches = re.findall(r"<custom_token_(\d+)>", response_text)
        
        if not matches:
            return []
        
        for number_str in matches:
            try:
                token_id = int(number_str)
                if ORPHEUS_MIN_ID <= token_id < ORPHEUS_MAX_ID:
                    absolute_ids.append(token_id)
            except ValueError:
                continue
        
        print(f"  - Parsed {len(absolute_ids)} valid audio token IDs")
        return absolute_ids
    
    def redistribute_codes(self, absolute_code_list: List[int]) -> Optional[np.ndarray]:
        """
        Convert Orpheus tokens to SNAC format and decode to audio
        Based on working Gradio code
        """
        if not absolute_code_list:
            return None
        
        snac_device = next(self.snac_model.parameters()).device
        layer_1, layer_2, layer_3 = [], [], []
        num_tokens = len(absolute_code_list)
        num_groups = num_tokens // ORPHEUS_N_LAYERS
        
        if num_groups == 0:
            return None
        
        print(f"  - Processing {num_groups} groups of {ORPHEUS_N_LAYERS} codes for SNAC...")
        
        for i in range(num_groups):
            base_idx = i * ORPHEUS_N_LAYERS
            if base_idx + ORPHEUS_N_LAYERS > num_tokens:
                break
            
            group_codes = absolute_code_list[base_idx:base_idx + ORPHEUS_N_LAYERS]
            processed_group = [None] * ORPHEUS_N_LAYERS
            valid_group = True
            
            for j, token_id in enumerate(group_codes):
                if not (ORPHEUS_MIN_ID <= token_id < ORPHEUS_MAX_ID):
                    valid_group = False
                    break
                
                layer_index = (token_id - ORPHEUS_MIN_ID) // ORPHEUS_TOKENS_PER_LAYER
                code_index = (token_id - ORPHEUS_MIN_ID) % ORPHEUS_TOKENS_PER_LAYER
                
                if layer_index != j:
                    valid_group = False
                    break
                
                processed_group[j] = code_index
            
            if not valid_group:
                continue
            
            try:
                # Map to SNAC layers as per working code
                layer_1.append(processed_group[0])
                layer_2.append(processed_group[1])
                layer_3.append(processed_group[2])
                layer_3.append(processed_group[3])
                layer_2.append(processed_group[4])
                layer_3.append(processed_group[5])
                layer_3.append(processed_group[6])
            except (IndexError, TypeError):
                continue
        
        try:
            if not layer_1 or not layer_2 or not layer_3:
                return None
            
            print(f"  - Final SNAC layer sizes: L1={len(layer_1)}, L2={len(layer_2)}, L3={len(layer_3)}")
            
            codes = [
                torch.tensor(layer_1, device=snac_device, dtype=torch.long).unsqueeze(0),
                torch.tensor(layer_2, device=snac_device, dtype=torch.long).unsqueeze(0),
                torch.tensor(layer_3, device=snac_device, dtype=torch.long).unsqueeze(0)
            ]
            
            with torch.no_grad():
                audio_hat = self.snac_model.decode(codes)
            
            return audio_hat.detach().squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"Error during SNAC decoding: {e}")
            return None
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "tara",
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        Generate speech using Orpheus TTS
        Based on working Gradio code
        """
        if not text.strip():
            return None
        
        print(f"Generating speech for: '{text[:50]}...'")
        start_time = time.time()
        
        # Correct prompt format from working code
        prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "stop": ["<|eot_id|>", "<|audio|>"],
            "stream": stream
        }
        
        print(f"  - Sending to {self.server_url}/completion")
        
        try:
            async with self.session.post(
                f"{self.server_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: Server returned {response.status}: {error_text}")
                    return None
                
                response_json = await response.json()
                print(f"  - Response received in {time.time() - start_time:.2f}s")
                
                # Extract generated text
                if "content" in response_json:
                    raw_text = response_json["content"].strip()
                elif "completion" in response_json:
                    raw_text = response_json["completion"].strip()
                else:
                    print(f"Unexpected response format: {response_json}")
                    return None
                
                if not raw_text:
                    print("Error: Empty response from server")
                    return None
                
                print(f"  - Raw response: {raw_text[:200]}...")
                
                # Parse SNAC codes
                absolute_ids = self.parse_gguf_codes(raw_text)
                if not absolute_ids:
                    print("Error: No valid audio codes parsed")
                    return None
                
                # Decode to audio
                audio_samples = self.redistribute_codes(absolute_ids)
                if audio_samples is None:
                    print("Error: Failed to decode audio")
                    return None
                
                print(f"  - Generated audio shape: {audio_samples.shape}")
                print(f"  - Total time: {time.time() - start_time:.2f}s")
                
                return (24000, audio_samples)
        
        except Exception as e:
            print(f"Error during TTS generation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def tts(
        self,
        text: str,
        voice_id: str = "tara",
        save_to_file: Optional[str] = None
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """
        Non-streaming TTS with stats
        """
        start_time = time.time()
        
        result = await self.generate_speech(text, voice=voice_id)
        
        if result is None:
            # Return empty audio
            return 24000, np.array([], dtype=np.float32), {
                "total_time_ms": (time.time() - start_time) * 1000,
                "error": "Generation failed"
            }
        
        sample_rate, audio = result
        total_time = time.time() - start_time
        
        stats = {
            "total_time_ms": total_time * 1000,
            "audio_duration_seconds": len(audio) / sample_rate,
            "sample_rate": sample_rate,
            "chunk_count": 1
        }
        
        # Save if requested
        if save_to_file and len(audio) > 0:
            # Convert to int16
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    max_val = np.max(np.abs(audio))
                    if max_val > 1e-6:
                        audio_int16 = np.int16(audio / max_val * 32767)
                    else:
                        audio_int16 = np.zeros_like(audio, dtype=np.int16)
                else:
                    audio_int16 = audio.astype(np.int16)
            else:
                audio_int16 = audio
            
            write(save_to_file, sample_rate, audio_int16)
            print(f"💾 Audio saved to: {save_to_file}")
        
        return sample_rate, audio, stats


# Concurrency testing
class ConcurrencyTester:
    """Test concurrent TTS requests"""
    
    def __init__(self, client: OrpheusClient):
        self.client = client
    
    async def single_request(
        self,
        text: str,
        request_id: int,
        voice_id: str = "tara",
        save_to_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute single TTS request with timing"""
        start_time = time.time()
        
        try:
            sample_rate, audio, stats = await self.client.tts(text, voice_id, save_to_file=save_to_file)
            
            total_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "text": text,
                "total_latency_ms": total_time * 1000,
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
        """Test concurrent requests"""
        print(f"\n🧪 Testing {concurrent_users} concurrent requests...")
        print(f"📝 Total texts: {len(texts)}")
        
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def limited_request(text: str, req_id: int):
            async with semaphore:
                save_file = None
                if save_dir and req_id < 3:
                    save_file = os.path.join(save_dir, f"sample_{req_id+1}.wav")
                return await self.single_request(text, req_id, voice_id, save_to_file=save_file)
        
        start_time = time.time()
        tasks = [limited_request(text, i) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"success": False, "error": str(result)})
            else:
                processed_results.append(result)
        
        successful = [r for r in processed_results if r.get("success")]
        failed = [r for r in processed_results if not r.get("success")]
        
        latencies = [r["total_latency_ms"] for r in successful]
        
        stats = {
            "test_type": "concurrent",
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
        
        if failed:
            stats["errors"] = [r.get("error") for r in failed]
        
        return stats


async def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 Orpheus TTS Client (Fixed)")
    print("=" * 60)
    
    test_texts = [
        "Hello, world!",
        "This is a test of the Orpheus system.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    async with OrpheusClient(server_url="http://localhost:8090") as client:
        
        # Test 1: Single request
        print("\n" + "=" * 60)
        print("🧪 Test 1: Single Request")
        print("=" * 60)
        
        try:
            sr, audio, stats = await client.tts(
                "Hello, this is a test of the fixed Orpheus client.",
                save_to_file="test_fixed_orpheus.wav"
            )
            
            print(f"✅ Success!")
            print(f"   Total time: {stats['total_time_ms']:.1f}ms")
            print(f"   Audio duration: {stats['audio_duration_seconds']:.2f}s")
            print(f"   Sample rate: {stats['sample_rate']}Hz")
            print(f"   Audio samples: {len(audio)}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        # Test 2: Concurrency
        print("\n" + "=" * 60)
        print("🧪 Test 2: Concurrency (3 users)")
        print("=" * 60)
        
        tester = ConcurrencyTester(client)
        os.makedirs("benchmark_audio_outputs", exist_ok=True)
        
        try:
            results = await tester.test_concurrent(
                test_texts,
                concurrent_users=3,
                save_dir="benchmark_audio_outputs"
            )
            
            print(f"\n📊 Results:")
            print(f"   Success rate: {results['successful_requests']}/{results['total_requests']}")
            print(f"   Total time: {results['total_test_time_seconds']:.2f}s")
            print(f"   Requests/sec: {results['requests_per_second']:.2f}")
            
            if 'latency_stats' in results:
                ls = results['latency_stats']
                print(f"   Latency - Min: {ls['min_ms']:.1f}ms, Max: {ls['max_ms']:.1f}ms, Mean: {ls['mean_ms']:.1f}ms")
        
        except Exception as e:
            print(f"❌ Concurrency test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Tests completed!")
    print("🔊 Check generated .wav files to verify audio quality")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
