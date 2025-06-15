uv run audioinsight-server \
    --backend faster-whisper --model large-v3-turbo \
    --llm_inference \
    --base_llm "openai/gpt-4.1-mini" \
    --fast_llm "openai/gpt-4.1-nano" \

# --backend openai-api --model whisper-1 \
