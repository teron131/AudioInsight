# LLM Analysis Feature

This document describes the LLM-based transcription analysis feature that automatically generates analysis of transcribed conversations after periods of inactivity.

## Overview

The LLM analysis feature monitors transcription activity and automatically generates intelligent analysis when:
- A period of inactivity is detected (configurable idle time)
- A certain number of conversation turns occur
- Maximum text length is reached

The system uses OpenAI-compatible models to analyze transcription content and extract:
- Concise summary of the conversation
- Key points and insights
- Context and important information

## Features

- **Automatic Triggering**: Analysis is generated based on configurable triggers
- **Real-time Monitoring**: Continuously monitors transcription activity
- **Callback System**: Supports custom callbacks for handling analysis results
- **Statistics Tracking**: Comprehensive metrics on analysis generation
- **Speaker Awareness**: Handles multi-speaker conversations with diarization
- **Duplicate Prevention**: Avoids generating duplicate analysis for the same content

## Configuration

### Command Line Arguments

Enable LLM analysis when starting the server:

```bash
python -m audioinsight.server \
    --llm-analysis \
    --llm-model "gpt-4.1-mini" \
    --llm-trigger-time 5.0 \
    --llm-conversation-trigger 2
```

### Available Options

- `--llm-analysis`: Enable LLM-based analysis (default: True)
- `--llm-model`: LLM model to use (default: gpt-4.1-mini)  
- `--llm-trigger-time`: Idle time in seconds before triggering analysis (default: 5.0)
- `--llm-conversation-trigger`: Number of speaker turns before triggering analysis (default: 2)

## Programmatic Usage

### Basic Setup

```python
# Initialize with LLM analysis enabled
processor = AudioProcessor()
llm_analysis=True,

# The analyzer will automatically monitor transcriptions
# and generate analysis based on configured triggers
```

### Custom Callbacks

```python
async def my_analysis_callback(analysis_response, transcription_text):
    print(f"Analysis: {analysis_response.summary}")
    print(f"Key Points: {analysis_response.key_points}")
    
# Add custom callback
processor.llm.add_analysis_callback(my_analysis_callback)
```

### Manual Analysis

```python
# Force analysis generation
analysis = await processor.llm.force_analysis()
if analysis:
    print(f"Summary: {analysis.summary}")
    print(f"Key Points: {analysis.key_points}")
```

## Statistics

The system tracks comprehensive statistics:

```json
{
    "total_conversations": 15,
    "total_analyses_generated": 3,
    "total_text_analyzed": 1250,
    "monitoring_start_time": 1698765432.1,
    "last_analysis_time": 1698765467.5
}
```

### Available Metrics

- `total_conversations`: Number of conversation turns processed
- `total_analyses_generated`: Total number of analyses created
- `total_text_analyzed`: Total characters processed
- `monitoring_start_time`: When monitoring started
- `last_analysis_time`: Timestamp of last analysis

## Testing

### Basic Test

1. Start the server with LLM analysis enabled
2. Create a test analyzer
3. Send some transcription data
4. Demonstrate automatic analysis

```python
import asyncio
from audioinsight.llm import LLM, LLMTrigger

async def test_analysis():
    # Create analyzer with short trigger time for testing
    trigger_config = LLMTrigger(
        idle_time_seconds=2.0,
        conversation_trigger_count=3
    )
    
    llm = LLM(
        model_id="gpt-4.1-mini",
        trigger_config=trigger_config
    )
    
    # Add test callback
    async def print_analysis(response, text):
        print(f"Analysis generated!")
        print(f"Summary: {response.summary}")
        print(f"Key points: {response.key_points}")
    
    llm.add_analysis_callback(print_analysis)
    
    # Start monitoring
    await llm.start_monitoring()
    
    # Simulate conversation
    llm.update_transcription("Hello, how are you today?", "Speaker 1")
    await asyncio.sleep(1)
    llm.update_transcription("I'm doing well, thanks for asking!", "Speaker 2")
    await asyncio.sleep(3)  # Trigger idle analysis
    
    await llm.stop_monitoring()

# Run test
asyncio.run(test_analysis())
```

## Use Cases

### Meeting Analysis

Perfect for analyzing meetings, interviews, or discussions:

- **Key Decisions**: Extract important decisions made
- **Action Items**: Identify tasks and responsibilities
- **Discussion Topics**: Analyze main themes
- **Participant Insights**: Track different perspectives

### Customer Support

Analyze customer interactions:

- **Issue Summary**: Quick overview of customer problems
- **Resolution Steps**: Track troubleshooting progression
- **Customer Sentiment**: Understand emotional context
- **Follow-up Actions**: Identify next steps

## Integration

The LLM analysis feature works seamlessly with:

- **Real-time Transcription**: Automatic analysis of live conversations
- **Speaker Diarization**: Multi-speaker conversation understanding
- **WebSocket Streaming**: Real-time analysis delivery
- **REST API**: Programmatic access to analysis results

## Environment Setup

Ensure you have the required environment variables:

```bash
# For OpenRouter API
OPENROUTER_API_KEY=your_openrouter_key

# Alternative: Direct OpenAI API
OPENAI_API_KEY=your_openai_key
```

## Error Handling

The system includes robust error handling:

- **API Failures**: Graceful degradation when LLM is unavailable
- **Rate Limiting**: Automatic backoff for API rate limits
- **Invalid Responses**: Fallback mechanisms for malformed responses
- **Network Issues**: Retry logic for transient failures

## Performance Considerations

- **Text Length Limits**: Configurable maximum text length for analysis
- **Cooldown Periods**: Prevents excessive API calls
- **Duplicate Detection**: Avoids reprocessing identical content
- **Async Processing**: Non-blocking analysis generation

For optimal performance, adjust trigger parameters based on your use case and API rate limits. 