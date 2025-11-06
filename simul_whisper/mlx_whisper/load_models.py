# Copyright © 2023 Apple Inc.

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper

logger = logging.getLogger(__name__)


def _extract_model_size(path_or_hf_repo: str, model_name: str) -> str:
    """
    Extract model size from path_or_hf_repo or use model_name.
    
    Handles inputs like:
    - "mlx_base" -> "base"
    - "mlx_small" -> "small"
    - "mlx_large-v3" -> "large-v3"
    - "base" -> "base"
    - Full HF repo names are returned as-is
    """
    # If it's already a full HuggingFace repo path (contains "/"), return as-is
    if "/" in path_or_hf_repo:
        return path_or_hf_repo
    
    # Try to extract size from path_or_hf_repo (e.g., "mlx_base" -> "base")
    if path_or_hf_repo.startswith("mlx_"):
        size = path_or_hf_repo.replace("mlx_", "", 1)
        return size
    
    # If path_or_hf_repo looks like a simple model name, use it
    # Otherwise fall back to model_name parameter
    if path_or_hf_repo in ["tiny", "base", "small", "medium", "large", 
                           "large-v1", "large-v2", "large-v3"]:
        return path_or_hf_repo
    
    # Fall back to model_name
    return model_name


def _format_hf_repo_name(model_size: str) -> str:
    """Format model size to HuggingFace repository name."""
    return f"mlx-community/whisper-{model_size}-mlx"


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
    model_name: str = "small",
) -> whisper.Whisper:
    # First, check if it's a local path
    model_path = Path(path_or_hf_repo)
    
    if model_path.exists():
        logger.info(f"Loading model from local path: {model_path}")
    else:
        # Not a local path, determine the HuggingFace repo name
        model_size = _extract_model_size(path_or_hf_repo, model_name)
        
        # If it's not already a full repo path, format it
        if "/" not in model_size:
            hf_repo = _format_hf_repo_name(model_size)
            logger.info(
                f"Model path '{path_or_hf_repo}' not found locally. "
                f"Attempting to download from HuggingFace: {hf_repo}"
            )
            logger.info(
                "Note: To use a custom model, download it to a local directory "
                "and provide the full path."
            )
        else:
            hf_repo = model_size
            logger.info(f"Downloading model from HuggingFace: {hf_repo}")
        
        try:
            model_path = Path(snapshot_download(repo_id=hf_repo))
            logger.info(f"Successfully downloaded model to: {model_path}")
        except Exception as e:
            error_msg = (
                f"Failed to download model from HuggingFace: {hf_repo}\n"
                f"Error: {e}\n\n"
                f"To use a custom model:\n"
                f"1. Download the model to a local directory "
                f"(must contain config.json and weights.safetensors or weights.npz)\n"
                f"2. Provide the full path to the directory as --model_path\n"
                f"3. Example: --model_path /path/to/my/model"
            )
            logger.error(error_msg)
            raise

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    alignment_heads = whisper._ALIGNMENT_HEADS.get(model_name, whisper._ALIGNMENT_HEADS["small"])
    model.set_alignment_heads(alignment_heads)


    weights = tree_unflatten(list(weights.items()))
    model.update(weights)


    mx.eval(model.parameters())
    return model
