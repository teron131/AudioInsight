# LLM Summarization Feature

This document describes the LLM-based transcription summarization feature that automatically generates summaries of transcribed conversations after periods of inactivity.

## Overview

The LLM summarization feature monitors transcription activity and automatically generates intelligent summaries when:
- No new transcription has been received for a configurable time period (default: 5 seconds)
- The accumulated text meets a minimum length requirement (default: 50 characters)

## Features

- **Automatic Triggering**: Summaries are generated automatically after periods of inactivity
- **Speaker-Aware**: Incorporates speaker diarization information when available
- **Structured Output**: Provides summaries with key points, speaker count, and confidence scores
- **Configurable**: Customizable idle time, text length thresholds, and LLM model
- **Real-time Integration**: Seamlessly integrates with the existing transcription pipeline
- **Statistics Tracking**: Monitors performance and usage statistics

## Installation

Install the required dependencies:

```bash
pip install whisperlivekit[llm]
```

Or install the dependencies manually:

```bash
pip install langchain langchain-openai python-dotenv
```

## Configuration

### Environment Variables

Set up your API key in a `.env` file or environment variable:

```bash
# For OpenRouter (recommended)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Or for OpenAI directly
OPENAI_API_KEY=your_openai_api_key_here
```

### Command Line Arguments

Enable LLM summarization when starting the server:

```bash
# Basic usage with default settings
whisperlivekit-server

# Custom configuration
whisperlivekit-server \
    --llm-model "gpt-4.1-mini" \
    --llm-trigger-time 5.0
```

### Available Arguments

- `--llm-summarization`: Enable LLM-based summarization (default: True)
- `--llm-model`: LLM model to use (default: "gpt-4.1-mini")
- `--llm-trigger-time`: Idle time in seconds before triggering summarization (default: 5.0)

## Usage

### Programmatic Usage

```python
from whisperlivekit import WhisperLiveKit
from whisperlivekit.audio_processor import AudioProcessor

# Initialize with LLM summarization enabled
kit = WhisperLiveKit(
    llm_summarization=True,
    llm_model="gpt-4.1-mini",
    llm_trigger_time=5.0,
)

# Create audio processor
processor = AudioProcessor()

# The summarizer will automatically monitor transcriptions
# and generate summaries after idle periods
```

### WebSocket API

When using the WebSocket API, summaries are included in the response:

```json
{
  "lines": [...],
  "buffer_transcription": "...",
  "buffer_diarization": "...",
  "summaries": [
    {
      "timestamp": 1234567890.123,
      "summary": "Discussion about quarterly results and planning for next quarter.",
      "key_points": [
        "Improve customer satisfaction scores",
        "Increase marketing budget",
        "Hire more developers"
      ],
      "speakers_mentioned": 2,
      "confidence": 0.95,
      "text_length": 245
    }
  ],
  "llm_stats": {
    "summaries_generated": 3,
    "total_text_summarized": 1250,
    "average_summary_time": 2.3
  }
}
```

## Response Format

### Summary Object

Each summary contains:

- `timestamp`: Unix timestamp when the summary was generated
- `summary`: Concise summary of the transcribed content
- `key_points`: List of main points or topics discussed
- `speakers_mentioned`: Number of speakers identified in the text
- `confidence`: AI confidence score (0.0 to 1.0) in the summary quality
- `text_length`: Length of the original transcribed text

### Statistics Object

The `llm_stats` object provides:

- `summaries_generated`: Total number of summaries created
- `total_text_summarized`: Total characters processed
- `average_summary_time`: Average time taken to generate summaries


## Testing

Test the feature independently:

```bash
python test_llm.py
```

This script will:
1. Check for API key availability
2. Create a test summarizer
3. Simulate transcription updates
4. Demonstrate automatic summarization
5. Show statistics and performance metrics

## Performance Considerations

- **Latency**: Summary generation typically takes 1-3 seconds
- **Cost**: Costs depend on the model and text length
- **Rate Limits**: Respects API rate limits with automatic retry logic
- **Memory**: Accumulated text is cleared after each summary

## Troubleshooting

### Common Issues

1. **No API Key**: Ensure `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is set
2. **Import Errors**: Install dependencies with `pip install whisperlivekit[llm]`
3. **Rate Limiting**: Use a less aggressive `llm_trigger_time` setting
4. **Model Errors**: Verify the model name is correct for your API provider

### Debug Logging

Enable debug logging to see detailed information:

```python
import logging
logging.getLogger("whisperlivekit.llm").setLevel(logging.DEBUG)
```

## Examples

### Meeting Summarization

Perfect for summarizing meetings, interviews, or discussions:

```bash
whisperlivekit-server \
    --llm-trigger-time 10.0 \
    --diarization
```

### Real-time Lecture Notes

For educational content with shorter idle times:

```bash
whisperlivekit-server \
    --llm-trigger-time 3.0
```

### Long-form Content

For podcasts or long presentations:

```bash
whisperlivekit-server \
    --llm-trigger-time 15.0 \
    --llm-model "gpt-4o"
```

## Integration with Existing Features

The LLM summarization feature works seamlessly with:

- **Speaker Diarization**: Incorporates speaker information in summaries
- **Real-time Transcription**: Monitors live transcription streams
- **File Upload**: Processes uploaded audio files
- **WebSocket API**: Provides summaries through the existing API
- **Multiple Languages**: Works with any language supported by Whisper

## Future Enhancements

Planned improvements include:

- Custom prompt templates
- Summary export formats (PDF, Word, etc.)
- Integration with note-taking applications
- Sentiment analysis
- Action item extraction
- Meeting minutes generation 