"""
Debug script to see what llama-server is returning
"""
import asyncio
import aiohttp
import json


async def test_server_response():
    """Test what the server returns"""
    
    prompt = "<|im_start|>system\nYou are a text-to-speech assistant. Speak with voice: tara<|im_end|>\n<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "prompt": prompt,
        "temperature": 0.8,
        "max_tokens": 200,
        "top_p": 0.95,
        "top_k": 40,
        "stream": True,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "cache_prompt": True
    }
    
    print("🔍 Testing llama-server response...")
    print(f"Prompt: {prompt}")
    print("\n" + "=" * 80)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8090/completion",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            print(f"Status: {response.status}")
            print("=" * 80)
            
            token_count = 0
            full_response = ""
            
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                
                if not line_str or not line_str.startswith('data: '):
                    continue
                
                json_str = line_str[6:]
                
                if json_str == '[DONE]':
                    print("\n[DONE]")
                    break
                
                try:
                    data = json.loads(json_str)
                    if 'content' in data:
                        content = data['content']
                        full_response += content
                        token_count += 1
                        
                        # Print first 10 tokens with details
                        if token_count <= 10:
                            print(f"Token {token_count}: {repr(content)}")
                        elif token_count == 11:
                            print("... (showing first 10 tokens)")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON error: {e}")
                    print(f"Line: {line_str}")
            
            print("\n" + "=" * 80)
            print(f"Total tokens: {token_count}")
            print(f"Full response length: {len(full_response)} chars")
            print("\n" + "=" * 80)
            print("Full response:")
            print(full_response[:500])
            if len(full_response) > 500:
                print(f"\n... ({len(full_response) - 500} more chars)")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_server_response())
