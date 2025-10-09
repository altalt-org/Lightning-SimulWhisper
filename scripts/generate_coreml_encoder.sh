#!/bin/bash
# Script to generate CoreML encoder models using whisper.cpp
# This creates the .mlmodelc files needed for CoreML acceleration

set -e

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./generate_coreml_encoder.sh <model-name>"
    echo ""
    echo "Available models:"
    echo "  - tiny.en, tiny"
    echo "  - base.en, base"
    echo "  - small.en, small"
    echo "  - medium.en, medium"
    echo "  - large-v1, large-v2, large-v3"
    echo "  - large-v3-turbo (or turbo)"
    echo ""
    echo "Example: ./generate_coreml_encoder.sh base.en"
    exit 1
fi

MODEL_NAME=$1

# Check if whisper.cpp exists
if [ ! -d "whisper.cpp" ]; then
    echo "Error: whisper.cpp directory not found."
    echo "Please clone whisper.cpp first:"
    echo "  git clone https://github.com/ggml-org/whisper.cpp.git"
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models

echo "=================================================="
echo "Generating CoreML encoder for model: $MODEL_NAME"
echo "=================================================="

cd whisper.cpp/models

# Check if Python environment is ready
echo ""
echo "Checking Python dependencies..."
python3 -c "import ane_transformers" 2>/dev/null || {
    echo "Installing ane_transformers..."
    pip install ane_transformers
}

python3 -c "import openai-whisper" 2>/dev/null || {
    echo "Installing openai-whisper..."
    pip install openai-whisper
}

python3 -c "import coremltools" 2>/dev/null || {
    echo "Installing coremltools..."
    pip install coremltools
}

echo ""
echo "Generating CoreML model..."
./generate-coreml-model.sh $MODEL_NAME

echo ""
echo "=================================================="
echo "CoreML model generated successfully!"
echo "=================================================="
echo ""
echo "Generated models:"
if [ -d "coreml-encoder-${MODEL_NAME}.mlpackage" ]; then
    echo "  ✓ coreml-encoder-${MODEL_NAME}.mlpackage (ready to use)"
    ls -lh coreml-encoder-${MODEL_NAME}.mlpackage/ | head -5
fi
if [ -d "ggml-${MODEL_NAME}-encoder.mlmodelc" ]; then
    echo "  ✓ ggml-${MODEL_NAME}-encoder.mlmodelc"
fi

echo ""
echo "To use this model in your Python code:"
echo ""
echo "from simul_whisper import PaddedAlignAttWhisper"
echo "from simul_whisper.config import AlignAttConfig"
echo ""
echo "cfg = AlignAttConfig("
echo "    model_path='mlx-community/whisper-${MODEL_NAME}-mlx',"
echo "    model_name='${MODEL_NAME}',"
echo "    use_coreml_encoder=True,  # Enable CoreML - will auto-find the model"
echo "    language='en',  # Set your language"
echo ")"
echo ""
echo "whisper = PaddedAlignAttWhisper(cfg)"
echo ""
echo "The CoreML encoder will be automatically detected and used!"
echo ""
