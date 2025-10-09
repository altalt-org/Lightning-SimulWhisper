#!/usr/bin/env python3
"""
Test script for CoreML encoder integration

This script tests the CoreML encoder functionality and compares
performance with the MLX encoder.
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx

from simul_whisper.coreml_encoder import CoreMLEncoder, check_coreml_availability


def test_coreml_availability():
    """Test if CoreML is available and properly configured"""
    print("=" * 60)
    print("CoreML Availability Check")
    print("=" * 60)

    info = check_coreml_availability()
    for key, value in info.items():
        print(f"  {key}: {value}")

    if not info.get("coreml_available"):
        print("\n❌ CoreML is not available")
        print("Install coremltools: pip install coremltools")
        return False

    if not info.get("is_apple_silicon"):
        print("\n⚠️  Warning: Not running on Apple Silicon")
        print("CoreML will work but may not use Neural Engine")

    print("\n✅ CoreML is available\n")
    return True


def test_encoder_initialization(model_name="base.en"):
    """Test encoder initialization"""
    print("=" * 60)
    print(f"Testing CoreML Encoder Initialization ({model_name})")
    print("=" * 60)

    try:
        encoder = CoreMLEncoder.from_model_name(model_name)
        print(f"✅ Successfully loaded encoder from: {encoder.model_path}")
        return encoder
    except FileNotFoundError as e:
        print(f"\n❌ Model not found: {e}")
        print(f"\nTo generate the model, run:")
        print(f"  ./scripts/generate_coreml_encoder.sh {model_name}")
        return None
    except Exception as e:
        print(f"\n❌ Failed to initialize encoder: {e}")
        return None


def test_encoder_inference(encoder, n_runs=5):
    """Test encoder inference with timing"""
    print("\n" + "=" * 60)
    print("Testing Encoder Inference")
    print("=" * 60)

    # Create dummy mel spectrogram (80 mels, 3000 frames)
    mel = mx.random.normal(shape=(80, 3000)).astype(mx.float32)
    print(f"Input shape: {mel.shape}")
    print(f"Input dtype: {mel.dtype}")

    # Warm-up run (CoreML compiles on first run)
    print("\nWarm-up run (compiling for your device)...")
    try:
        _ = encoder(mel)
        print("✅ Warm-up complete")
    except Exception as e:
        print(f"❌ Warm-up failed: {e}")
        return None

    # Timed runs
    print(f"\nRunning {n_runs} inference passes...")
    times = []
    for i in range(n_runs):
        t_start = time.time()
        output = encoder(mel)
        t_end = time.time()

        elapsed = (t_end - t_start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}ms")

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print("\n" + "-" * 60)
    print("Performance Statistics:")
    print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print("-" * 60)

    return avg_time


def compare_with_mlx(model_name="base.en"):
    """Compare CoreML and MLX encoder performance"""
    print("\n" + "=" * 60)
    print("Comparing CoreML vs MLX Encoder")
    print("=" * 60)

    try:
        from simul_whisper.mlx_whisper import load_models

        # Load MLX model
        print("\nLoading MLX model...")
        model_path = f"mlx-community/whisper-{model_name}-mlx"
        model = load_models.load_model(
            path_or_hf_repo=model_path,
            dtype=mx.float16,
            model_name=model_name
        )
        print("✅ MLX model loaded")

        # Create test input
        mel = mx.random.normal(shape=(1, 80, 3000)).astype(mx.float16)

        # Warm-up MLX
        print("\nMLX warm-up...")
        _ = model.encoder(mel)
        mx.eval(_)

        # Time MLX encoder
        print("\nTiming MLX encoder (5 runs)...")
        mlx_times = []
        for i in range(5):
            t_start = time.time()
            output = model.encoder(mel)
            mx.eval(output)
            t_end = time.time()

            elapsed = (t_end - t_start) * 1000
            mlx_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}ms")

        mlx_avg = np.mean(mlx_times)
        print(f"\nMLX Average: {mlx_avg:.2f}ms")

        # Get CoreML time (from previous test)
        print("\nTo see CoreML performance, run test_encoder_inference first")

    except Exception as e:
        print(f"❌ Failed to load MLX model: {e}")
        print("This comparison requires the MLX model to be available")


def main():
    parser = argparse.ArgumentParser(
        description="Test CoreML encoder integration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",
        help="Model name (default: base.en)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with MLX encoder performance"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of inference runs (default: 5)"
    )

    args = parser.parse_args()

    # Test 1: Availability
    if not test_coreml_availability():
        return 1

    # Test 2: Initialization
    encoder = test_encoder_initialization(args.model)
    if encoder is None:
        return 1

    # Test 3: Inference
    avg_time = test_encoder_inference(encoder, n_runs=args.runs)
    if avg_time is None:
        return 1

    # Test 4: Comparison (optional)
    if args.compare:
        compare_with_mlx(args.model)

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Try different compute units:")
    print("   encoder = CoreMLEncoder.from_model_name('base.en', compute_units='CPU_AND_NE')")
    print("\n2. Integrate into your pipeline:")
    print("   cfg = AlignAttConfig(..., use_coreml_encoder=True)")
    print("   whisper = PaddedAlignAttWhisper(cfg)")
    print("\n3. Run real audio inference:")
    print("   python your_script.py --use-coreml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
