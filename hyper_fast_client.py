#!/usr/bin/env python3
"""
Hyper-optimized Orpheus TTS client with aggressive performance tuning
Target: <150ms TTFA, 20+ req/s throughput
"""

import asyncio
import aiohttp
import numpy as np
import re
import time
from typing import AsyncGenerator, Optional, Tuple, Dict, Any
from snac import SNAC
import torch
from scipy.io import wavfile

class HyperFastSNACDecoder:
    """Ultra-optimized SNAC decoder with minimal overhead"""
    
    def __init__(self):
        print("📥 Loading SNAC decoder...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        # Force CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            print(f"✅ SNAC loaded on cuda")
        else:
            self.device = torch.device("cpu")
            print(f"⚠️  SNAC loaded on cpu")
        
        # Pre-allocate tensors for minimal GC overhead
        self.layer_buffers = [
            torch.zeros((1, 1), dtype=torch.long, device=self.device),
            torch.zeros((1, 2), dtype=torch.long, device=self.device),
            torch.zeros((1, 4), dtype=torch.long, device=self.device)
        ]
    
    @torch.inference_mode()  # Faster than no_grad
    def decode_minimal(self, codes_list: list) -> np.ndarray:
        """
        Decode with minimal buffer (14 tokens = 2 frames = ~40ms audio)
        Faster TTFA at cost of more frequent decoding
        """
        if len(codes_list) < 14:  # Need at least 2 frames
            return np.array([], dtype=np.float32)
        
        # Take exactly 14 tokens (2 frames)
        codes_batch = codes_list[:14]
        
        # Redistribute to SNAC layers (7 tokens per frame)
        codes_l1 = []
        codes_l2 = []
        codes_l3 = []
        
        for i in range(0, len(codes_batch), 7):
            frame = codes_batch[i:i+7]
            if len(frame) == 7:
                codes_l1.append(frame[0])
                codes_l2.extend([frame[1], frame[4]])
                codes_l3.extend([frame[2], frame[3], frame[5], frame[6]])
        
        # Update buffers in-place (avoid allocation)
        n_frames = len(codes_l1)
        if n_frames > 0:
            self.layer_buffers[0].resize_(1, n_frames)
            self.layer_buffers[1].resize_(1, n_frames * 2)
            self.layer_buffers[2].resize_(1, n_frames * 4)
            
            self.layer_buffers[0][0] = torch.tensor(codes_l1, device=self.device)
            self.layer_buffers[1][0] = torch.tensor(codes_l2, device=self.device)
            self.layer_buffers[2][0] = torch.tensor(codes_l3, device=self.device)
            
            # Decode
            audio_hat = self.model.decode(self.layer_buffers)
            return audio_hat.squeeze().cpu().numpy()
        
        return np.array([], dtype=np.float32)

class HyperFastOrpheusClient:
    """Hyper-optimized client with aggressive performance tuning"""
    
    BASE_URL = "http://localhost:8090"
    ORPHEUS_MIN_ID = 10
    ORPHEUS_MAX_ID = 28687
    ORPHEUS_TOKENS_PER_LAYER = 4096
    
    def __init__(self):
        self.session = None
        self.decoder = None
    
    async def __aenter__(self):
        # Aggressive timeout and connection settings
        timeout = aiohttp.ClientTimeout(
            total=60,
            connect=2,
            sock_read=10
        )
        
        # Connection pooling with high limits
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=60"
            }
        )
        
        # Lazy-load decoder (only when first needed)
        if self.decoder is None:
            self.decoder = HyperFastSNACDecoder()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def parse_token_id(self, token_str: str) -> Optional[int]:
        """Fast token parsing with compiled regex"""
        match = re.search(r'<custom_token_(\d+)>', token_str)
        if match:
            token_id = int(match.group(1))
            if self.ORPHEUS_MIN_ID <= token_id < self.ORPHEUS_MAX_ID:
                code_value = (token_id - self.ORPHEUS_MIN_ID) % self.ORPHEUS_TOKENS_PER_LAYER
                return code_value
        return None
    
    async def stream_generate(
        self,
        text: str,
        voice: str = "tara",
        temperature: float = 0.2,  # Lower for faster sampling
        max_tokens: int = 384  # Reduced from 512
    ) -> AsyncGenerator[Tuple[np.ndarray, Dict[str, Any]], None]:
        """
        Stream audio with minimal buffering (14 tokens = 2 frames)
        Optimized for lowest TTFA
        """
        
        # Minimal prompt
        prompt = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.85,  # Slightly lower for faster sampling
            "top_k": 30,    # Reduced from 40
            "repeat_penalty": 1.05,  # Reduced from 1.1
            "stream": True,
            "stop": ["<|eot_id|>", "<|audio|>"],
            "cache_prompt": True,
            "n_predict": max_tokens  # Explicit prediction limit
        }
        
        codes_buffer = []
        total_audio = []
        ttfa = None
        start_time = time.perf_counter()
        chunk_count = 0
        
        try:
            async with self.session.post(
                f"{self.BASE_URL}/completion",
                json=payload
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line.startswith('data: '):
                        continue
                    
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        import json
                        data = json.loads(data_str)
                        content = data.get('content', '')
                        
                        if not content:
                            continue
                        
                        # Parse tokens
                        for token_match in re.finditer(r'<custom_token_(\d+)>', content):
                            code = self.parse_token_id(token_match.group(0))
                            if code is not None:
                                codes_buffer.append(code)
                        
                        # Decode every 14 tokens (2 frames = ~40ms audio)
                        while len(codes_buffer) >= 14:
                            audio_chunk = self.decoder.decode_minimal(codes_buffer)
                            
                            if len(audio_chunk) > 0:
                                if ttfa is None:
                                    ttfa = (time.perf_counter() - start_time) * 1000
                                    print(f"⚡ TTFA: {ttfa:.1f}ms")
                                
                                total_audio.append(audio_chunk)
                                chunk_count += 1
                                
                                yield audio_chunk, {
                                    'is_final': False,
                                    'ttfa_ms': ttfa,
                                    'chunk_num': chunk_count
                                }
                            
                            # Remove processed tokens
                            codes_buffer = codes_buffer[14:]
                    
                    except json.JSONDecodeError:
                        continue
                
                # Process remaining tokens (if any)
                if len(codes_buffer) >= 14:
                    audio_chunk = self.decoder.decode_minimal(codes_buffer)
                    if len(audio_chunk) > 0:
                        total_audio.append(audio_chunk)
                        chunk_count += 1
                        
                        yield audio_chunk, {
                            'is_final': True,
                            'ttfa_ms': ttfa,
                            'chunk_num': chunk_count,
                            'total_audio_s': sum(len(a) for a in total_audio) / 24000
                        }
        
        except Exception as e:
            raise Exception(f"Streaming error: {e}")

async def test_hyper_fast():
    """Test hyper-optimized client"""
    
    test_prompts = [
        "Hello!",
        "This is a quick test.",
        "Testing hyper-fast TTS generation with aggressive optimizations."
    ]
    
    print("="*80)
    print("⚡ HYPER-FAST ORPHEUS TTS CLIENT TEST")
    print("="*80)
    print("Target: <150ms TTFA, minimal latency")
    print("="*80)
    
    async with HyperFastOrpheusClient() as client:
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}/{len(test_prompts)}: '{prompt}'")
            print("-"*80)
            
            try:
                start_time = time.perf_counter()
                ttfa = None
                chunks = 0
                total_audio = []
                
                async for audio_chunk, info in client.stream_generate(prompt):
                    if ttfa is None:
                        ttfa = info['ttfa_ms']
                    chunks += 1
                    total_audio.append(audio_chunk)
                
                total_time = (time.perf_counter() - start_time) * 1000
                full_audio = np.concatenate(total_audio) if total_audio else np.array([])
                audio_duration = len(full_audio) / 24000
                
                # Save audio
                if len(full_audio) > 0:
                    audio_int16 = (full_audio * 32767).astype(np.int16)
                    filename = f"test_hyper_{i}.wav"
                    wavfile.write(filename, 24000, audio_int16)
                
                # Check if target met
                target_symbol = "🎯" if ttfa and ttfa < 150 else ""
                
                print(f"✅ Success!")
                if ttfa is not None:
                    print(f"   TTFA: {ttfa:.1f}ms {target_symbol}")
                else:
                    print(f"   TTFA: N/A (no tokens generated)")
                print(f"   Total time: {total_time:.1f}ms")
                print(f"   Chunks: {chunks}")
                print(f"   Audio: {audio_duration:.2f}s ({len(full_audio)} samples)")
                if len(full_audio) > 0:
                    print(f"   Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
    
    print("\n" + "="*80)
    print("✅ Tests completed!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_hyper_fast())
