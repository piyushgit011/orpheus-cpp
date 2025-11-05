#!/usr/bin/env python3
"""
Simple test script for FastAPI TTS WAV endpoint
Downloads WAV files directly
"""

import requests
import time

FASTAPI_URL = "http://localhost:9100"

def test_wav_endpoint(text: str, filename: str = None):
    """Test the WAV endpoint and save the file"""
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": False
    }
    
    print(f"\n📝 Generating TTS for: '{text}'")
    print(f"   Request: POST {FASTAPI_URL}/api/tts/wav")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/tts/wav",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"   {response.text}")
            return
        
        # Get timing info from headers
        processing_time = response.headers.get('X-Processing-Time-Ms', 'N/A')
        ttfa = response.headers.get('X-TTFA-Ms', 'N/A')
        duration = response.headers.get('X-Audio-Duration-Seconds', 'N/A')
        
        # Save WAV file
        if filename is None:
            filename = f"tts_output_{int(time.time())}.wav"
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"✅ Success!")
        print(f"   TTFA: {ttfa}ms")
        print(f"   Processing time: {processing_time}ms")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Audio duration: {duration}s")
        print(f"   File size: {len(response.content)} bytes")
        print(f"   Saved to: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("="*80)
    print("🎯 FastAPI TTS WAV Endpoint Test")
    print("="*80)
    
    # Check server health
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server is healthy")
            print(f"   Status: {health.get('status')}")
            print(f"   Client initialized: {health.get('client_initialized')}")
        else:
            print(f"⚠️  Server health check returned {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure FastAPI server is running on {FASTAPI_URL}")
        return
    
    # Test cases
    test_cases = [
        ("Hello, world!", "test_hello.wav"),
        ("How are you doing today?", "test_question.wav"),
        ("The quick brown fox jumps over the lazy dog.", "test_pangram.wav"),
        ("This is a longer test to check performance with more text.", "test_longer.wav"),
    ]
    
    for text, filename in test_cases:
        test_wav_endpoint(text, filename)
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)

if __name__ == "__main__":
    main()
