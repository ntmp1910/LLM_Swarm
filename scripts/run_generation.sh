#!/bin/bash

# Set environment variables
export API_KEY="your-gpt-oss-api-key-here"
export HF_TOKEN="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"
export WANDB_API_KEY="bcbfaf91ff376b01bee9a7bafebbae3037991ae1"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the generation script
python generate_synthetic_gpt_oss.py \
    --prompts_dataset "your/prompts/dataset" \
    --prompt_column "prompt" \
    --max_samples 1000 \
    --checkpoint_path "./synthetic_data" \
    --checkpoint_interval 100 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --repo_id "your-username/synthetic_data_gpt_oss" \
    --wandb_username "your-wandb-username" \
    --config_file "config/gpt_oss_config.yaml"