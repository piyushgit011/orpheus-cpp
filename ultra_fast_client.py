"""
Ultra-Optimized Streaming Orpheus TTS Client
Target: 200ms time-to-first-audio (TTFA)

Key optimizations:
1. Streaming token decoding (decode every 28 tokens = 4 frames)
2. Connection pooling and keep-alive
3. Reduced max_tokens (512 instead of 2048)
4. Aggressive timeout settings
5. Incremental audio generation
"""
import asyncio
import aiohttp
import numpy as np
import json
import time
import re
from typing import AsyncGenerator, Tuple, Dict, Any, Optional
from scipy.io.wavfile import write
import torch


class OptimizedSNACDecoder:
    """Optimized SNAC decoder with incremental processing"""
    
    def __init__(self):
        print("📥 Loading SNAC decoder...")
        try:
            from snac import SNAC
            self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
            
            # Check CUDA availability properly
            self.device = "cpu"  # Default to CPU
            if torch.cuda.is_available():
                try:
                    self.model = self.model.to("cuda")
                    self.device = "cuda"
                except Exception as e:
                    print(f"⚠️  CUDA failed, using CPU: {e}")
                    self.model = self.model.to("cpu")
            else:
                self.model = self.model.to("cpu")
            
            self.model.eval()
            print(f"✅ SNAC loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load SNAC: {e}")
            raise
    
    def decode_incremental(self, token_ids: list) -> Optional[np.ndarray]:
        """
        Decode SNAC tokens incrementally
        Process as soon as we have 28 tokens (4 frames)
        """
        if len(token_ids) < 28:
            return None
        
        # Process complete frames only
        num_frames = len(token_ids) // 7
        tokens_to_process = num_frames * 7
        frame_tokens = token_ids[:tokens_to_process]
        
        try:
            # Redistribute tokens to SNAC layers
            layer_1, layer_2, layer_3 = [], [], []
            
            for i in range(num_frames):
                base_idx = i * 7
                layer_1.append(frame_tokens[base_idx])
                layer_2.extend([frame_tokens[base_idx + 1], frame_tokens[base_idx + 4]])
                layer_3.extend([
                    frame_tokens[base_idx + 2],
                    frame_tokens[base_idx + 3],
                    frame_tokens[base_idx + 5],
                    frame_tokens[base_idx + 6]
                ])
            
            # Create tensors
            codes = [
                torch.tensor(layer_1, device=self.device, dtype=torch.long).unsqueeze(0),
                torch.tensor(layer_2, device=self.device, dtype=torch.long).unsqueeze(0),
                torch.tensor(layer_3, device=self.device, dtype=torch.long).unsqueeze(0)
            ]
            
            # Decode to audio
            with torch.no_grad():
                audio_tensor = self.model.decode(codes)
            
            audio_np = audio_tensor.detach().squeeze().cpu().numpy()
            return audio_np
            
        except Exception as e:
            print(f"⚠️  Incremental decode error: {e}")
            return None


class UltraFastOrpheusClient:
    """Ultra-optimized Orpheus client targeting 200ms TTFA"""
    
    def __init__(self, server_url: str = "http://localhost:8090"):
        self.server_url = server_url
        self.snac_decoder = OptimizedSNACDecoder()
        self.session = None
        
        # Token parsing constants
        self.ORPHEUS_MIN_ID = 10
        self.ORPHEUS_TOKENS_PER_LAYER = 4096
        self.ORPHEUS_N_LAYERS = 7
        self.ORPHEUS_MAX_ID = self.ORPHEUS_MIN_ID + (self.ORPHEUS_N_LAYERS * self.ORPHEUS_TOKENS_PER_LAYER)
    
    async def __aenter__(self):
        """Setup optimized HTTP session"""
        # Aggressive connection settings for minimum latency
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=False,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(
            total=60,
            connect=2,  # Fast connect timeout
            sock_read=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=30"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def parse_token_id(self, token_str: str) -> Optional[int]:
        """Parse SNAC token ID from <custom_token_N> format"""
        match = re.search(r'<custom_token_(\d+)>', token_str)
        if match:
            token_id = int(match.group(1))
            if self.ORPHEUS_MIN_ID <= token_id < self.ORPHEUS_MAX_ID:
                # Convert absolute ID to relative code
                layer_index = (token_id - self.ORPHEUS_MIN_ID) // self.ORPHEUS_TOKENS_PER_LAYER
                code_value = (token_id - self.ORPHEUS_MIN_ID) % self.ORPHEUS_TOKENS_PER_LAYER
                return code_value
        return None
    
    async def stream_generate(
        self,
        text: str,
        voice: str = "tara",
        temperature: float = 0.3,  # Lower for faster generation
        max_tokens: int = 512  # Reduced from 2048
    ) -> AsyncGenerator[Tuple[np.ndarray, Dict[str, Any]], None]:
        """
        Stream audio generation with incremental SNAC decoding
        Yields audio chunks as soon as 28 tokens are available
        """
        
        # Optimized prompt format
        prompt = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["<|eot_id|>", "<|audio|>"],
            "cache_prompt": True  # Enable KV cache
        }
        
        start_time = time.time()
        first_token_time = None
        first_audio_time = None
        
        token_buffer = []
        tokens_decoded = 0
        chunk_count = 0
        
        try:
            async with self.session.post(
                f"{self.server_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Server error: {response.status} - {error_text[:200]}")
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str or not line_str.startswith('data: '):
                        continue
                    
                    json_str = line_str[6:]
                    
                    if json_str == '[DONE]':
                        # Process remaining tokens
                        if len(token_buffer) >= 28:
                            audio_chunk = self.snac_decoder.decode_incremental(token_buffer)
                            if audio_chunk is not None:
                                chunk_count += 1
                                yield audio_chunk, {
                                    "chunk_index": chunk_count,
                                    "is_final": True,
                                    "tokens_in_chunk": len(token_buffer),
                                    "elapsed_ms": (time.time() - start_time) * 1000
                                }
                        break
                    
                    try:
                        data = json.loads(json_str)
                        content = data.get('content', '')
                        
                        if not content:
                            continue
                        
                        # Mark first token time
                        if first_token_time is None:
                            first_token_time = time.time()
                        
                        # Parse token
                        token_id = self.parse_token_id(content)
                        if token_id is not None:
                            token_buffer.append(token_id)
                            
                            # Decode incrementally every 28 tokens (4 frames)
                            if len(token_buffer) >= 28:
                                # Take 28 tokens for decoding
                                tokens_to_decode = token_buffer[:28]
                                token_buffer = token_buffer[28:]
                                
                                audio_chunk = self.snac_decoder.decode_incremental(tokens_to_decode)
                                
                                if audio_chunk is not None:
                                    chunk_count += 1
                                    tokens_decoded += 28
                                    
                                    if first_audio_time is None:
                                        first_audio_time = time.time()
                                        ttfa = (first_audio_time - start_time) * 1000
                                        print(f"⚡ TTFA: {ttfa:.1f}ms")
                                    
                                    metadata = {
                                        "chunk_index": chunk_count,
                                        "is_final": False,
                                        "tokens_in_chunk": 28,
                                        "tokens_decoded_total": tokens_decoded,
                                        "elapsed_ms": (time.time() - start_time) * 1000,
                                        "ttfa_ms": (first_audio_time - start_time) * 1000 if first_audio_time else None
                                    }
                                    
                                    yield audio_chunk, metadata
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            raise
    
    async def generate_speech_fast(
        self,
        text: str,
        voice: str = "tara",
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Generate speech with streaming, return complete audio
        """
        all_chunks = []
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        
        async for audio_chunk, metadata in self.stream_generate(text, voice):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            
            all_chunks.append(audio_chunk)
            chunk_count += 1
        
        total_time = time.time() - start_time
        
        # Concatenate all audio
        if all_chunks:
            full_audio = np.concatenate(all_chunks)
        else:
            full_audio = np.array([], dtype=np.float32)
        
        # Convert to int16
        if full_audio.dtype != np.int16:
            max_val = np.max(np.abs(full_audio))
            if max_val > 1e-6:
                full_audio = np.int16(full_audio / max_val * 32767)
            else:
                full_audio = np.zeros_like(full_audio, dtype=np.int16)
        
        sample_rate = 24000
        
        stats = {
            "total_time_ms": total_time * 1000,
            "ttfa_ms": (first_chunk_time - start_time) * 1000 if first_chunk_time else None,
            "chunk_count": chunk_count,
            "audio_duration_seconds": len(full_audio) / sample_rate,
            "audio_samples": len(full_audio)
        }
        
        # Save if requested
        if save_path and len(full_audio) > 0:
            write(save_path, sample_rate, full_audio)
        
        return full_audio, sample_rate, stats


async def test_optimized_client():
    """Test the optimized client"""
    print("=" * 80)
    print("🚀 ULTRA-FAST ORPHEUS TTS CLIENT TEST")
    print("=" * 80)
    print("Target: 200ms Time-to-First-Audio (TTFA)")
    print("=" * 80)
    
    test_phrases = [
        ("Hello!", "test_fast_1.wav"),
        ("This is a quick test.", "test_fast_2.wav"),
        ("Yes, I know how Notion works - it's a workspace tool that combines notes, databases, wikis, and project management with a flexible block-based editor.", "test_fast_3.wav")
    ]
    
    async with UltraFastOrpheusClient() as client:
        
        for i, (text, filename) in enumerate(test_phrases, 1):
            print(f"\n📝 Test {i}/{len(test_phrases)}: '{text}'")
            print("-" * 80)
            
            try:
                audio, sr, stats = await client.generate_speech_fast(text, save_path=filename)
                
                print(f"✅ Success!")
                print(f"   TTFA: {stats['ttfa_ms']:.1f}ms" + (" 🎯" if stats['ttfa_ms'] < 500 else ""))
                print(f"   Total time: {stats['total_time_ms']:.1f}ms")
                print(f"   Chunks: {stats['chunk_count']}")
                print(f"   Audio: {stats['audio_duration_seconds']:.2f}s ({stats['audio_samples']} samples)")
                print(f"   Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_optimized_client())
