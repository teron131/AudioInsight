uv run audioinsight-server \
    --backend faster-whisper --model base \
    # --backend openai-api --model whisper-1 \
    --llm_inference \
    --base_llm "openai/gpt-4.1-mini" \
    --fast_llm "openai/gpt-4.1-nano" \
