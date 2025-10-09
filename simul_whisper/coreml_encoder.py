"""
CoreML Encoder Wrapper for Whisper

This module provides a Python interface to use CoreML-compiled Whisper encoder models
for faster and more power-efficient audio encoding on Apple Silicon.

Compatible with whisper.cpp generated CoreML models.
"""

import os
from pathlib import Path
from typing import Union
import numpy as np

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: coremltools not available. CoreML encoder will not work.")

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class CoreMLEncoder:
    """
    Wrapper for CoreML Whisper encoder model.

    This class provides a drop-in replacement for the MLX AudioEncoder,
    using Apple's CoreML framework for accelerated inference on Neural Engine.

    Args:
        model_path: Path to the .mlmodelc directory (compiled CoreML model)
        compute_units: Which compute units to use ('ALL', 'CPU_AND_NE', 'CPU_ONLY')
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        compute_units: str = "ALL"
    ):
        if not COREML_AVAILABLE:
            raise RuntimeError(
                "coremltools is not installed. "
                "Install with: pip install coremltools"
            )

        self.model_path = Path(model_path)

        # Handle both .mlmodelc and .mlpackage formats
        # The whisper.cpp script generates .mlpackage which needs to be used
        if not self.model_path.exists():
            # Check if there's a .mlpackage version
            if str(self.model_path).endswith('.mlmodelc'):
                mlpackage_path = Path(str(self.model_path).replace('.mlmodelc', '.mlpackage'))
                if mlpackage_path.exists():
                    print(f"Note: Using .mlpackage instead of .mlmodelc")
                    self.model_path = mlpackage_path
                else:
                    raise FileNotFoundError(
                        f"CoreML model not found at {model_path}. "
                        f"Generate it using: ./scripts/generate_coreml_encoder.sh"
                    )
            else:
                raise FileNotFoundError(
                    f"CoreML model not found at {model_path}. "
                    f"Generate it using: ./scripts/generate_coreml_encoder.sh"
                )

        # If .mlmodelc path was given but .mlpackage exists, prefer .mlpackage
        if str(self.model_path).endswith('.mlmodelc'):
            mlpackage_path = Path(str(self.model_path).replace('.mlmodelc', '.mlpackage'))
            if mlpackage_path.exists():
                print(f"Note: Found .mlpackage version, using that instead: {mlpackage_path.name}")
                self.model_path = mlpackage_path

        # Map compute units string to CoreML enum
        compute_units_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,  # CPU and Neural Engine
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY
        }

        compute_unit = compute_units_map.get(compute_units.upper(), ct.ComputeUnit.ALL)

        # Load the CoreML model
        print(f"Loading CoreML encoder from {self.model_path}")
        try:
            self.model = ct.models.MLModel(
                str(self.model_path),
                compute_units=compute_unit
            )
            print("CoreML encoder loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CoreML model from {self.model_path}: {e}\n"
                f"Try regenerating: ./scripts/generate_coreml_encoder.sh"
            )

        # Get model input/output specs
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name

    def __call__(self, mel: Union[mx.array, np.ndarray]) -> mx.array:
        """
        Encode mel spectrogram using CoreML.

        Args:
            mel: Mel spectrogram with shape (batch_size, n_mels, n_ctx)
                 Can be MLX array or numpy array

        Returns:
            MLX array with encoded features, shape (batch_size, n_ctx//2, n_state)
        """
        # Convert MLX array to numpy if needed
        if MLX_AVAILABLE and isinstance(mel, mx.array):
            mel_np = np.array(mel)
        else:
            mel_np = np.asarray(mel)

        # Ensure correct dtype (float32)
        mel_np = mel_np.astype(np.float32)

        # If input is 2D, add batch dimension
        if mel_np.ndim == 2:
            mel_np = mel_np[np.newaxis, ...]

        # Input should be (batch, n_mels, n_ctx) = (1, 80, 3000)
        # Run CoreML inference
        input_dict = {self.input_name: mel_np}
        output = self.model.predict(input_dict)

        # Extract output array
        output_array = output[self.output_name]

        # Convert back to MLX array if MLX is available
        if MLX_AVAILABLE:
            return mx.array(output_array)
        else:
            return output_array

    @staticmethod
    def find_model_path(model_name: str, search_dirs: list = None) -> Path:
        """
        Find CoreML model path by model name.

        Searches in common locations for ggml-{model_name}-encoder.mlmodelc

        Args:
            model_name: Name of the model (e.g., 'base.en', 'small', 'medium')
            search_dirs: Additional directories to search

        Returns:
            Path to the .mlmodelc directory

        Raises:
            FileNotFoundError if model not found
        """
        if search_dirs is None:
            search_dirs = []

        # Default search locations
        default_dirs = [
            Path.cwd() / "models",
            Path.cwd() / "whisper.cpp" / "models",
            Path(__file__).parent.parent / "models",
            Path(__file__).parent.parent / "whisper.cpp" / "models",
        ]

        all_dirs = default_dirs + [Path(d) for d in search_dirs]

        # Search for both .mlpackage (preferred) and .mlmodelc formats
        model_patterns = [
            f"coreml-encoder-{model_name}.mlpackage",  # New whisper.cpp format
            f"ggml-{model_name}-encoder.mlpackage",     # Alternative naming
            f"ggml-{model_name}-encoder.mlmodelc",      # Fallback to .mlmodelc
        ]

        for directory in all_dirs:
            for pattern in model_patterns:
                model_path = directory / pattern
                if model_path.exists():
                    return model_path

        raise FileNotFoundError(
            f"CoreML model for '{model_name}' not found in search directories.\n"
            f"Searched for: {model_patterns}\n"
            f"In directories: {[str(d) for d in all_dirs]}\n"
            f"Generate the model using: ./scripts/generate_coreml_encoder.sh {model_name}"
        )

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        search_dirs: list = None,
        compute_units: str = "ALL"
    ) -> "CoreMLEncoder":
        """
        Create CoreML encoder from model name.

        Convenience method that searches for the model and creates an encoder.

        Args:
            model_name: Name of the model (e.g., 'base.en', 'small')
            search_dirs: Additional directories to search for the model
            compute_units: Which compute units to use

        Returns:
            CoreMLEncoder instance
        """
        model_path = cls.find_model_path(model_name, search_dirs)
        return cls(model_path, compute_units)


def check_coreml_availability() -> dict:
    """
    Check if CoreML is available and return system info.

    Returns:
        Dictionary with availability status and system information
    """
    info = {
        "coreml_available": COREML_AVAILABLE,
        "mlx_available": MLX_AVAILABLE,
    }

    if COREML_AVAILABLE:
        try:
            import platform
            info["platform"] = platform.system()
            info["machine"] = platform.machine()
            info["coreml_version"] = ct.__version__

            # Check if running on Apple Silicon
            info["is_apple_silicon"] = (
                platform.system() == "Darwin" and
                platform.machine() == "arm64"
            )
        except Exception as e:
            info["error"] = str(e)

    return info


if __name__ == "__main__":
    # Test CoreML availability
    info = check_coreml_availability()
    print("CoreML Availability Check:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Example usage (if model exists)
    try:
        encoder = CoreMLEncoder.from_model_name("base.en")
        print(f"\nSuccessfully loaded CoreML encoder from: {encoder.model_path}")
    except FileNotFoundError as e:
        print(f"\n{e}")
