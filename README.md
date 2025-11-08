<h1 align="center">Lightning-SimulWhisper</h1>

<p align="center">
<img src="https://github.com/altalt-org/Lightning-SimulWhisper/raw/main/tests/timing_visualizations/comprehensive_summary.png" alt="WhisperLiveKit Demo" width="730">
</p>

<div align="center">
  <img src="typewriter-subtitle.svg" width="800" height="40" alt="Real-time, Local Speech-to-Text on Apple Silicon Devices">
</div>


<div align="center">
  <img src="warning-callout.svg" width="800" height="80" alt="Project warning">
</div>

The fastest, most power efficient real-time local transcriptions on your apple silicon devices ✨

Zero pytorch dependencies ⛔

15x speedup on encoding, 18x speedup on decoding ⚡

Lightning-SimulWhisper implements Whisper model for simultaneous transcription using **MLX** (Apple's machine learning framework) and **CoreML** for optimal performance on Apple Silicon devices. It uses the AlignAtt policy for streaming speech recognition.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=altalt-org/Lightning-SimulWhisper&type=date&legend=top-left)](https://www.star-history.com/#altalt-org/Lightning-SimulWhisper&type=date&legend=top-left)

## Performance Results

Using [the original SimulStreaming project](https://github.com/ufal/SimulStreaming) I could barely run the `base` model in real time. Now, I can run `medium` and even `large-v3-turbo` models in real time on my M2 Macbook Pro.

The MLX-only version consumes way too much power, so using the CoreML encoder is recommended.


- **CoreML Encoder**: While the encoder speedup is dramatic (up to 18x faster), the overall inference time improvement is more modest because the decoder still runs on MLX
- **MLX Decoder**: MLX provides up to 15x decoder speedup compared to PyTorch implementations, demonstrating excellent Apple Silicon optimization
- **Power Efficiency**: CoreML acceleration uses significantly less power than MLX-only implementations, though exact power measurements weren't captured in this benchmark
- **Decoder Performance**: MLX decoder performance remains consistent across implementations, showing the stability of the MLX framework
- **Speed Gains**: You can achieve up to **18x encoder speed increase** and **15x decoder speed increase** with optimal CoreML configuration

*Note: I have no idea on how to benchmark power consumption for a specific process. Any contributions or suggestions for accurate power measurement on Apple Silicon would be greatly appreciated!*

## Key Features

- **MLX Implementation**: Native Apple Silicon optimization with MLX framework (up to 15x decoder speedup)
- **CoreML Encoder**: Up to 18x faster encoding using Apple's Neural Engine
- **AlignAtt Policy**: State-of-the-art simultaneous decoding strategy
- **Multiple Model Support**: tiny, base, small, medium, large-v1, large-v2, large-v3
- **Beam Search**: Configurable beam search decoding
- **Real-time Streaming**: Both file simulation and live microphone input
- **Power Efficient**: Low power consumption with CoreML acceleration

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### CoreML Acceleration (Recommended)

For optimal performance on Apple Silicon, install CoreML dependencies:

```bash
pip install coremltools ane_transformers
```

### Generate CoreML Models

Generate CoreML encoder models for faster inference:

```bash
# Clone whisper.cpp for CoreML model generation
git clone https://github.com/ggml-org/whisper.cpp.git

# Generate CoreML encoder for your preferred model
./scripts/generate_coreml_encoder.sh base.en
```

Available models: `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`

### Lighter Installation

For minimal installation, remove `torchaudio` from `requirements.txt`. This disables the Silero VAD controller (`--vac` option).


## Usage

### Quick Start with CoreML (Recommended)

```bash
# Basic usage with CoreML acceleration
python3 simulstreaming_whisper.py jfk.wav \
  --model_name base \
  --model_path mlx_base \
  --use_coreml \
  --language en \
  --log-level CRITICAL

# With beam search and CIF model
python3 simulstreaming_whisper.py jfk.wav \
  --model_name medium \
  --model_path mlx_medium \
  --use_coreml \
  --beams 3 \
  --cif_ckpt_path cif_model/medium.npz \
  --language en \
  --log-level CRITICAL
```

### Real-time Simulation from Audio File

```bash
usage: simulstreaming_whisper.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--lan LAN] [--vac] [--vac-chunk-size VAC_CHUNK_SIZE] [--vad]
                                 [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--model_path MODEL_PATH] [--model_name MODEL_NAME] [--beams BEAMS] [--decoder DECODER] [--audio_max_len AUDIO_MAX_LEN]
                                 [--audio_min_len AUDIO_MIN_LEN] [--frame_threshold FRAME_THRESHOLD] [--cif_ckpt_path CIF_CKPT_PATH] [--never_fire | --no-never_fire]
                                 [--init_prompt INIT_PROMPT] [--static_init_prompt STATIC_INIT_PROMPT] [--max_context_tokens MAX_CONTEXT_TOKENS] [--start_at START_AT] [--comp_unaware]
                                 [--use_coreml] [--coreml_encoder_path COREML_ENCODER_PATH] [--coreml_compute_units {ALL,CPU_AND_NE,CPU_ONLY}]
                                 audio_path

CoreML Options:
  --use_coreml          Enable CoreML encoder acceleration (up to 18x faster, lower power)
  --coreml_encoder_path COREML_ENCODER_PATH
                        Path to CoreML encoder .mlpackage directory (auto-detected if not provided)
  --coreml_compute_units {ALL,CPU_AND_NE,CPU_ONLY}
                        CoreML compute units: ALL (default), CPU_AND_NE (recommended), CPU_ONLY

Model Options:
  --model_path MODEL_PATH
                        Path to MLX model directory or HuggingFace repo
  --model_name MODEL_NAME
                        Model name: tiny, base.en, small, medium, large-v1, large-v2, large-v3
  --beams BEAMS, -b BEAMS
                        Number of beams for beam search decoding (1 = greedy)
  --decoder DECODER     Override automatic decoder selection

Audio Processing:
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds
  --audio_max_len AUDIO_MAX_LEN
                        Max length of audio buffer in seconds (default: 30.0)
  --audio_min_len AUDIO_MIN_LEN
                        Skip processing if audio buffer is shorter than this length
  --frame_threshold FRAME_THRESHOLD
                        AlignAtt threshold in frames (default: 4)

Language:
  --lan LAN, --language LAN
                        Source language code (en, de, cs, etc.) or 'auto' for detection

CIF Model (End-of-Word Detection):
  --cif_ckpt_path CIF_CKPT_PATH
                        Path to CIF model checkpoint for word boundary detection
  --never_fire, --no-never_fire
                        Override CIF model behavior (default: False)

Context and Prompts:
  --init_prompt INIT_PROMPT
                        Initial prompt for the model (in target language)
  --static_init_prompt STATIC_INIT_PROMPT
                        Static prompt that doesn't scroll (terminology, etc.)
  --max_context_tokens MAX_CONTEXT_TOKENS
                        Maximum context tokens (default: model's max)

Simulation Options:
  --start_at START_AT   Start processing audio at this time
  --comp_unaware        Computationally unaware simulation
```

### Examples

```bash
# Basic MLX implementation
python simulstreaming_whisper.py test.mp3 \
  --language ko \
  --vac \
  --vad_silence_ms 1000 \
  --beams 3 \
  -l CRITICAL \
  --cif_ckpt_path cif_model/medium.npz \
  --model_name medium \
  --model_path mlx_medium

# With CoreML encoder acceleration (up to 18x faster, lower power)
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

# CoreML with Neural Engine (best power efficiency)
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

## Architecture

Lightning-SimulWhisper uses a hybrid architecture combining MLX and CoreML:

```
Audio Input (16kHz mono)
    ↓
Mel Spectrogram (MLX)
    ↓
┌─────────────────────┐
│  CoreML Encoder     │ ← Apple Neural Engine (up to 18x faster)
│  (whisper.cpp)      │
└─────────────────────┘
    ↓
Encoder Features (convert to MLX)
    ↓
┌─────────────────────┐
│  MLX Decoder        │ ← Full control, beam search, AlignAtt
│  (Simul-Whisper)    │
└─────────────────────┘
    ↓
Transcription Output
```

**Key Components:**
- **MLX Framework**: Apple's optimized ML framework for Apple Silicon (up to 15x decoder speedup)
- **CoreML Encoder**: Neural Engine acceleration for the encoder (up to 18x speedup, most compute-intensive part)
- **MLX Decoder**: Flexible decoding with AlignAtt policy, beam search, and streaming
- **AlignAtt Policy**: State-of-the-art simultaneous decoding strategy

