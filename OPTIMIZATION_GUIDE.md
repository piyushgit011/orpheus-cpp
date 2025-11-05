# Orpheus TTS Optimization Guide
## Target: 200ms Time-to-First-Audio (TTFA)

## Current Performance
- **TTFA**: 4,000-22,000ms (4-22 seconds)
- **Target**: 200ms TTFA
- **Required Improvement**: 20-110x faster

## Optimization Strategy

### 1. Server-Side Optimizations

#### Restart llama-server with optimized parameters:
```bash
./start_optimized_server.sh
```

**Key Changes:**
- `--parallel 4` - Enable 4 concurrent slots (was 1)
- `--cont-batching` - Continuous batching for better throughput
- `-c 2048` - Reduced context (was 4096) for faster processing
- `-b 512` - Larger batch size for better GPU utilization
- `--flash-attn` - Flash attention for faster inference
- `--no-mmap` - Keep model in RAM for faster access
- `--mlock` - Lock memory pages
- `--defrag-thold 0.1` - KV cache defragmentation

**Expected Impact:** 2-3x faster token generation

### 2. Client-Side Optimizations

#### Use the ultra-fast streaming client:
```bash
python ultra_fast_client.py
```

**Key Features:**
- **Incremental SNAC decoding**: Decode every 28 tokens (4 frames) instead of waiting for all tokens
- **Streaming processing**: Start audio generation immediately
- **Reduced max_tokens**: 512 instead of 2048
- **Lower temperature**: 0.3 instead of 0.8 for faster, more deterministic generation
- **Connection pooling**: Reuse HTTP connections with keep-alive
- **Aggressive timeouts**: Fast-fail on slow responses

**Expected Impact:** 5-10x faster TTFA

### 3. Prompt Optimization

**Shorter prompts = faster generation:**
```python
# Old: verbose
prompt = "<|im_start|>system\nYou are a TTS assistant...<|im_end|>..."

# New: minimal
prompt = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
```

**Expected Impact:** 1.5-2x faster

### 4. Token Budget Optimization

**Analysis of token requirements:**
- Short phrase ("Hello"): ~150 tokens
- Medium sentence: ~300 tokens
- Long sentence: ~500 tokens

**Recommendation:** Use dynamic max_tokens based on text length:
```python
# Estimate tokens needed
estimated_tokens = len(text.split()) * 50  # ~50 tokens per word
max_tokens = min(512, max(150, estimated_tokens))
```

**Expected Impact:** 1.5-2x faster for short phrases

### 5. SNAC Decoding Optimization

**Current**: Wait for all tokens, then decode once  
**Optimized**: Decode incrementally every 28 tokens

```python
# Every 28 tokens = 4 frames = ~85ms of audio
if len(token_buffer) >= 28:
    audio_chunk = decode_incremental(token_buffer[:28])
    yield audio_chunk  # Stream immediately
```

**Expected Impact:** 
- TTFA: 10-20x faster (first audio in ~200-500ms)
- Total time: 2-3x faster

### 6. GPU Optimization

**Ensure GPU is fully utilized:**
```bash
# Check GPU usage
nvidia-smi -l 1
```

**Should see:**
- GPU utilization: 90-100% during generation
- Memory usage: ~3-4GB
- Temperature: <80°C

### 7. Network Optimization

**Minimize network overhead:**
- Use localhost connection (no network latency)
- Enable HTTP keep-alive
- Use connection pooling
- Disable request/response compression for lower latency

### 8. Concurrent Request Optimization

**For multiple users:**
- llama-server with `--parallel 4` can handle 4 concurrent requests
- Each request gets its own slot
- No queuing delay for first 4 requests

**Expected throughput:** 4-5 requests/second

## Expected Results After Optimization

### Single Request Performance
| Metric | Before | After (Target) | Improvement |
|--------|--------|----------------|-------------|
| TTFA | 4,000ms | 200-500ms | 8-20x |
| Total Time | 5,000ms | 1,000-2,000ms | 2.5-5x |
| Tokens/sec | ~60 | ~150-200 | 2.5-3x |

### Concurrent Performance (5 users)
| Metric | Before | After (Target) | Improvement |
|--------|--------|----------------|-------------|
| Avg Latency | 5,105ms | 1,500-2,500ms | 2-3x |
| Throughput | 0.95 req/s | 4-5 req/s | 4-5x |

## Implementation Steps

### Step 1: Stop current server
```bash
# Find and kill current llama-server
pkill -f llama-server
```

### Step 2: Start optimized server
```bash
./start_optimized_server.sh
```

Wait for: `"HTTP server listening"` message

### Step 3: Test with ultra-fast client
```bash
python ultra_fast_client.py
```

### Step 4: Benchmark improvements
```bash
python comprehensive_benchmark.py
```

## Monitoring & Tuning

### Check server performance:
```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Check server metrics
curl http://localhost:8090/metrics

# View active slots
curl http://localhost:8090/slots
```

### Tuning knobs (in order of impact):
1. **max_tokens**: Lower = faster (try 256, 384, 512)
2. **temperature**: Lower = faster (try 0.2, 0.3, 0.4)
3. **parallel slots**: More = better concurrency (try 4, 8, 16)
4. **batch size**: Larger = better GPU util (try 256, 512, 1024)
5. **context size**: Smaller = faster (try 1024, 2048)

## Troubleshooting

### Issue: Still slow after optimization
**Check:**
- Is GPU actually being used? (`nvidia-smi`)
- Are all GPU layers offloaded? (check server startup logs)
- Is batch size appropriate? (should be 256-512)
- Any CPU throttling? (`htop`)

### Issue: "Server disconnected" errors
**Solutions:**
- Increase `--timeout` in llama-server
- Add retry logic in client
- Reduce concurrent requests
- Check server logs for OOM errors

### Issue: Audio quality degraded
**Check:**
- Temperature not too low (<0.2 can sound robotic)
- Incremental decoding buffer aligned (28 tokens)
- SNAC model loaded correctly
- No token ID parsing errors

## Advanced: Multi-Instance Deployment

For production with >20 concurrent users:

```bash
# Start 4 instances on different ports
./llama-server -m model.gguf --port 8090 --parallel 4 &
./llama-server -m model.gguf --port 8091 --parallel 4 &
./llama-server -m model.gguf --port 8092 --parallel 4 &
./llama-server -m model.gguf --port 8093 --parallel 4 &

# Use nginx or HAProxy for load balancing
```

**Expected:** 16-20 requests/second

## Summary

With all optimizations:
- **TTFA**: 200-500ms ✅
- **Total latency**: 1,000-2,000ms (2-5x improvement)
- **Throughput**: 4-5 req/s (4-5x improvement)
- **Concurrent users**: 4-5 simultaneous (4-5x improvement)

The **streaming approach** is key - users hear audio within 200-500ms while generation continues in background!
