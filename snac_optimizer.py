"""
Advanced SNAC Codec Optimization Module
Implements state-of-the-art SNAC streaming techniques
"""
import numpy as np
import asyncio
from typing import Generator, AsyncGenerator, List, Tuple
import onnxruntime as ort

class SNACStreamOptimizer:
    """
    Advanced SNAC optimizer based on research findings:
    - Sliding window modification for no popping
    - Segment-wise decoding for low latency
    - Depth-first flattening for real-time streaming
    """
    
    def __init__(self, model_path: str, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.segment_duration_ms = 100  # ~100ms segments as per research
        self.overlap_samples = int(0.01 * sample_rate)  # 10ms overlap
        
        # Optimized ONNX session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.add_session_config_entry('session.disable_prepacking', '0')
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Sliding window buffer
        self.window_buffer = np.zeros(self.overlap_samples, dtype=np.float32)
        
    def apply_crossfade(self, 
                       audio_chunk: np.ndarray, 
                       fade_length: int = None) -> np.ndarray:
        """Apply crossfade between chunks to prevent popping"""
        if fade_length is None:
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
        self.window_buffer = result[-self.overlap_samples:].copy()
        
        return result.astype(audio_chunk.dtype)
    
    def segment_wise_decode(self, 
                           token_chunks: List[np.ndarray]) -> Generator[np.ndarray, None, None]:
        """
        Segment-wise decoding for low-latency streaming
        Based on SNAC's hierarchical structure flattening
        """
        for chunk in token_chunks:
            # Process segment (approximately 100ms as per research)
            try:
                # Run SNAC decoding
                input_dict = {
                    inp.name: chunk[i] if i < len(chunk) else np.array([0])
                    for i, inp in enumerate(self.session.get_inputs())
                }
                
                audio_output = self.session.run(None, input_dict)[0]
                
                # Convert to audio samples
                audio_samples = audio_output.squeeze()
                if audio_samples.dtype != np.int16:
                    audio_samples = (audio_samples * 32767).astype(np.int16)
                
                # Apply crossfade to prevent popping
                smoothed_audio = self.apply_crossfade(audio_samples)
                
                yield smoothed_audio
                
            except Exception as e:
                print(f"⚠️ SNAC decode error: {e}")
                continue
    
    async def async_segment_decode(self, 
                                  token_stream: AsyncGenerator[List[np.ndarray], None]) -> AsyncGenerator[np.ndarray, None]:
        """Async version for integration with FastAPI"""
        async for token_chunks in token_stream:
            for audio_chunk in self.segment_wise_decode(token_chunks):
                yield audio_chunk