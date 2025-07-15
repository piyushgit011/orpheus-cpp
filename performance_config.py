"""
Performance Configuration based on research findings
"""
import os
import onnxruntime as ort

class PerformanceConfig:
    """Optimized configuration based on SNAC and FastAPI research"""
    
    @staticmethod
    def configure_onnx_for_snac():
        """Configure ONNX Runtime for optimal SNAC performance"""
        # Based on research findings for ONNX optimization
        return {
            'providers': [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Better performance
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ],
            'session_options': {
                'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                'execution_mode': ort.ExecutionMode.ORT_PARALLEL,
                'intra_op_num_threads': min(16, os.cpu_count() or 8),
                'inter_op_num_threads': min(16, os.cpu_count() or 8),
                'enable_cpu_mem_arena': True,
                'enable_mem_pattern': True,
                'enable_mem_reuse': True,
            },
            'config_entries': {
                'session.intra_op.allow_spinning': '1',
                'session.inter_op.allow_spinning': '1',
                'session.force_spinning_stop': '0',
                'session.disable_prepacking': '0',
            }
        }
    
    @staticmethod
    def get_streaming_config():
        """Optimal streaming configuration for SNAC"""
        return {
            'chunk_size': 4096,  # Optimal for 24kHz audio
            'buffer_size': 48000,  # 2 seconds
            'fade_duration_ms': 10,  # Based on research
            'segment_duration_ms': 100,  # SNAC segment size
            'pre_buffer_duration': 0.1,  # Reduced latency
        }
    
    @staticmethod
    def get_fastapi_config():
        """FastAPI optimization configuration"""
        return {
            'workers': 1,  # Single worker for GPU models
            'threads': min(32, (os.cpu_count() or 1) + 4),
            'max_requests': 100,
            'timeout': 300,
            'keepalive': 60,
        }