#!/usr/bin/env python3
"""
Test the /api/tts endpoint with the exact text from the curl command
"""

import requests
import json
import base64
import numpy as np
from scipy.io import wavfile
import time

FASTAPI_URL = "http://localhost:9100"

def test_tts_endpoint():
    """Test /api/tts with long text"""
    
    text = "If you compiled llama.cpp yourself, Flash-Attention support is a build-time option. The parse error you hit is before that matters, but if you later see \"flash attention not available,\" rebuild with CUDA + FA enabled."
    
    print("="*80)
    print("🎯 Testing /api/tts endpoint")
    print("="*80)
    print(f"Text: {text}")
    print(f"Length: {len(text)} characters")
    print("="*80)
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.3,
        "max_tokens": 256,
        "stream": False
    }
    
    print("\n📤 Sending request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/tts",
            json=payload,
            timeout=60
        )
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Decode and save audio
            audio_bytes = base64.b64decode(data['audio_base64'])
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            filename = "tts_long_text.wav"
            wavfile.write(filename, 24000, audio_array)
            
            print("\n✅ Success!")
            print(f"   Duration: {data['duration_seconds']:.2f}s")
            print(f"   Sample rate: {data['sample_rate']} Hz")
            print(f"   Processing time: {data['processing_time_ms']:.1f}ms")
            print(f"   TTFA: {data.get('ttfa_ms', 'N/A')}ms")
            print(f"   Total request time: {total_time:.1f}ms")
            print(f"   Audio samples: {len(audio_array)}")
            print(f"   Saved to: {filename}")
            
        elif response.status_code == 400:
            error = response.json()
            print(f"\n❌ Bad Request:")
            print(f"   {json.dumps(error, indent=2)}")
            
        elif response.status_code == 500:
            error = response.json()
            print(f"\n❌ Server Error:")
            print(f"   {json.dumps(error, indent=2)}")
            
        else:
            print(f"\n❌ Unexpected status: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timeout after 60 seconds")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_tts_endpoint()
