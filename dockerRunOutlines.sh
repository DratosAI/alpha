docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    -p 8000:8000 \
    --network=host \
    outlinesdev/outlines \
    --model="NousResearch/Meta-Llama-3-8B-Instruct"