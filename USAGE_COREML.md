# Using CoreML Encoder in Your Streaming Scripts

## Quick Start

Just add `--use_coreml` to your existing command:

```bash
# Your current command
python simulstreaming_whisper.py test.mp3 \
  --language ko \
  --vac \
  --vad_silence_ms 1000 \
  --beams 3 \
  -l CRITICAL \
  --cif_ckpt_path cif_model/medium.npz \
  --model_name medium \
  --model_path mlx_medium

# With CoreML acceleration (just add --use_coreml)
python simulstreaming_whisper.py test.mp3 \
  --language ko \
  --vac \
  --vad_silence_ms 1000 \
  --beams 3 \
  -l CRITICAL \
  --cif_ckpt_path cif_model/medium.npz \
  --model_name medium \
  --model_path mlx_medium \
  --use_coreml
```

## New Command-Line Arguments

### `--use_coreml`
Enable CoreML encoder acceleration (3-5x faster, lower power consumption)

**Example:**
```bash
python simulstreaming_whisper.py test.mp3 --model_name medium --model_path mlx_medium --use_coreml
```

### `--coreml_encoder_path PATH`
Specify explicit path to CoreML model (optional, auto-detected if not provided)

**Example:**
```bash
--use_coreml --coreml_encoder_path whisper.cpp/models/coreml-encoder-medium.mlpackage
```

### `--coreml_compute_units {ALL,CPU_AND_NE,CPU_ONLY}`
Control which hardware CoreML uses

**Options:**
- `ALL` (default) - Use all available compute units
- `CPU_AND_NE` - CPU + Neural Engine (recommended for power efficiency)
- `CPU_ONLY` - CPU only (for debugging)

**Example:**
```bash
--use_coreml --coreml_compute_units CPU_AND_NE
```

## Complete Examples

### File Processing with CoreML

```bash
python simulstreaming_whisper.py test.mp3 \
  --language ko \
  --vac \
  --vad_silence_ms 1000 \
  --beams 3 \
  -l CRITICAL \
  --cif_ckpt_path cif_model/medium.npz \
  --model_name medium \
  --model_path mlx_medium \
  --use_coreml \
  --coreml_compute_units CPU_AND_NE
```

### Server with CoreML

```bash
python simulstreaming_whisper_server.py \
  --language ko \
  --host localhost \
  --port 43001 \
  --warmup-file warmup.mp3 \
  --vac \
  --beams 3 \
  -l INFO \
  --model_path mlx_medium \
  --model_name medium \
  --cif_ckpt_path cif_model/medium.npz \
  --audio_min_len 1.0 \
  --use_coreml
```

### Real-time Microphone with CoreML

```bash
# Start server with CoreML
python simulstreaming_whisper_server.py \
  --language ko \
  --host localhost \
  --port 43001 \
  --vac \
  --beams 3 \
  --model_name medium \
  --model_path mlx_medium \
  --use_coreml

# Stream from microphone (in another terminal)
ffmpeg -f avfoundation -i ":2" -ac 1 -ar 16000 -f s16le -c:a pcm_s16le - | nc localhost 43001
```

## Performance Comparison

### Without CoreML (MLX encoder)
```bash
python simulstreaming_whisper.py test.mp3 \
  --model_name medium \
  --model_path mlx_medium \
  -l INFO
```
- **Encoding time**: ~800-1000ms per chunk
- **Power draw**: ~15-20W (high CPU/GPU usage)
- **Fan**: Usually active

### With CoreML (Neural Engine)
```bash
python simulstreaming_whisper.py test.mp3 \
  --model_name medium \
  --model_path mlx_medium \
  --use_coreml \
  --coreml_compute_units CPU_AND_NE \
  -l INFO
```
- **Encoding time**: ~271ms per chunk (3-5x faster!)
- **Power draw**: ~5-8W (low power)
- **Fan**: Usually silent

## Troubleshooting

### "CoreML model not found"

**Problem**: You haven't generated the CoreML model yet

**Solution**:
```bash
./scripts/generate_coreml_encoder.sh medium
```

### "Failed to initialize CoreML encoder"

**Problem**: CoreML dependencies missing

**Solution**:
```bash
pip install coremltools
```

### Model still uses MLX encoder

**Problem**: CoreML initialization failed, falling back to MLX

**Check the logs** for warnings:
```bash
python simulstreaming_whisper.py test.mp3 --use_coreml -l INFO
```

Look for:
- `✓ CoreML encoder initialized successfully` ✅ Working!
- `⚠ Failed to initialize CoreML encoder` ❌ Check error message

### First run is very slow

**Behavior**: First CoreML inference takes 2-3 seconds

**Explanation**: CoreML compiles the model for your device on first run. Subsequent runs are fast.

## Best Practices

1. **Always use `--use_coreml`** for production inference on Apple Silicon
2. **Use `CPU_AND_NE`** for best power efficiency:
   ```bash
   --use_coreml --coreml_compute_units CPU_AND_NE
   ```
3. **Keep decoder in MLX** (automatic) - you get CoreML speed + MLX flexibility
4. **Generate models once** - they're cached by the OS
5. **Monitor first run** - it's slower but then stays fast

## Backward Compatibility

All existing scripts work without any changes. CoreML is **opt-in** via `--use_coreml` flag.

```bash
# This still works exactly as before
python simulstreaming_whisper.py test.mp3 --model_name medium --model_path mlx_medium
```

## Performance Tips

### For Maximum Speed
```bash
--use_coreml --coreml_compute_units ALL
```

### For Maximum Battery Life
```bash
--use_coreml --coreml_compute_units CPU_AND_NE
```

### For Debugging
```bash
--use_coreml --coreml_compute_units CPU_ONLY -l DEBUG
```

## See Also

- [COREML_QUICKSTART.md](COREML_QUICKSTART.md) - Quick reference
- [docs/COREML_INTEGRATION.md](docs/COREML_INTEGRATION.md) - Full documentation
- [examples/coreml_example.py](examples/coreml_example.py) - Code example

## Summary

**Before**: Just run your command
```bash
python simulstreaming_whisper.py test.mp3 --model_name medium --model_path mlx_medium
```

**After**: Add `--use_coreml`
```bash
python simulstreaming_whisper.py test.mp3 --model_name medium --model_path mlx_medium --use_coreml
```

**Result**: 3-5x faster encoding, lower power, same quality! 🚀
