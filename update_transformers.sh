#!/bin/bash
# Update transformers with our modified files
# Run this script after modifying files in src/openpi/models_pytorch/transformers_replace/

set -e

VENV_PATH=".venv/lib/python3.11/site-packages/transformers"
SOURCE_PATH="src/openpi/models_pytorch/transformers_replace"

echo "Updating transformers with modified files..."

# Copy Gemma files
echo "  → Copying Gemma files..."
cp -r "${SOURCE_PATH}/models/gemma/"* "${VENV_PATH}/models/gemma/"

# Copy SigLIP check file if it exists
if [ -d "${SOURCE_PATH}/models/siglip" ]; then
    echo "  → Copying SigLIP files..."
    cp -r "${SOURCE_PATH}/models/siglip/"* "${VENV_PATH}/models/siglip/"
fi

echo "✅ Transformers updated successfully!"
echo ""
echo "Verification:"
if grep -q "from openpi.models_pytorch.lora import LoRALinear" "${VENV_PATH}/models/gemma/modeling_gemma.py"; then
    echo "  ✅ LoRA imports found in modeling_gemma.py"
else
    echo "  ❌ LoRA imports NOT found in modeling_gemma.py"
fi

if grep -q "_get_layer_weight" "${VENV_PATH}/models/gemma/modeling_gemma.py"; then
    echo "  ✅ _get_layer_weight helper found"
else
    echo "  ❌ _get_layer_weight helper NOT found"
fi

echo ""
echo "Done! You can now run training or inference."







