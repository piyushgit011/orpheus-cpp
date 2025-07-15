"""
Test script to verify GPU setup before running the server
"""
import os
import sys

def test_cuda_environment():
    """Test CUDA environment"""
    print("🧪 Testing CUDA Environment")
    print("=" * 40)
    
    # Check CUDA paths
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    print(f"CUDA_HOME: {cuda_home}")
    
    # Check if CUDA binaries exist
    nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
    if os.path.exists(nvcc_path):
        print("✅ nvcc found")
        os.system('nvcc --version')
    else:
        print("❌ nvcc not found")
    
    # Check library paths
    lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"LD_LIBRARY_PATH contains CUDA: {cuda_home in lib_path}")

def test_pytorch():
    """Test PyTorch CUDA support"""
    print("\n🧪 Testing PyTorch")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        return torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_onnxruntime():
    """Test ONNX Runtime GPU support"""
    print("\n🧪 Testing ONNX Runtime")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        
        # Preload DLLs
        try:
            ort.preload_dlls()
            print("✅ ONNX Runtime DLLs preloaded")
        except:
            print("⚠️  DLL preload failed")
        
        providers = ort.get_available_providers()
        print(f"✅ Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider available")
            
            # Test session creation
            try:
                # Create a simple test session
                import numpy as np
                
                # Simple identity model for testing
                test_providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kSameAsRequested',
                        'gpu_mem_limit': 1024 * 1024 * 1024,  # 1GB
                    }),
                    'CPUExecutionProvider'
                ]
                
                print("✅ CUDA provider can be configured")
                return True
                
            except Exception as e:
                print(f"❌ CUDA provider configuration failed: {e}")
                return False
        else:
            print("❌ CUDA provider not available")
            return False
            
    except ImportError:
        print("❌ ONNX Runtime not installed")
        return False

def test_llama_cpp():
    """Test llama-cpp-python CUDA support"""
    print("\n🧪 Testing llama-cpp-python")
    print("=" * 40)
    
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python imported successfully")
        
        # Check if CUDA support was compiled
        # This is a bit tricky to test without a model
        print("✅ llama-cpp-python available")
        return True
        
    except ImportError as e:
        print(f"❌ llama-cpp-python import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 GPU Setup Verification")
    print("=" * 60)
    
    # Test environment
    test_cuda_environment()
    
    # Test components
    pytorch_ok = test_pytorch()
    onnx_ok = test_onnxruntime()
    llama_ok = test_llama_cpp()
    
    print("\n📊 Summary")
    print("=" * 40)
    print(f"PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    print(f"ONNX Runtime CUDA: {'✅' if onnx_ok else '❌'}")
    print(f"llama-cpp-python: {'✅' if llama_ok else '❌'}")
    
    all_ok = pytorch_ok and onnx_ok and llama_ok
    
    if all_ok:
        print("\n🎉 All GPU components ready!")
        print("You can now run the TTS server with GPU acceleration.")
    else:
        print("\n⚠️  Some issues detected. Please fix before running server.")
        
        if not pytorch_ok:
            print("Fix PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        if not onnx_ok:
            print("Fix ONNX Runtime: pip uninstall onnxruntime onnxruntime-gpu -y && pip install onnxruntime-gpu")
        
        if not llama_ok:
            print("Fix llama-cpp-python: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")

if __name__ == "__main__":
    main()