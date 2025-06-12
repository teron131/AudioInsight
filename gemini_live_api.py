import asyncio
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Updated imports based on user suggestion and common Gemini SDK patterns
from google import genai
from google.genai import types as gemini_types  # Using alias for clarity
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS configuration to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration for analysis system
ANALYSIS_INTERVAL_SECONDS = float(os.getenv("ANALYSIS_INTERVAL_SECONDS", "5.0"))
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gemini-2.5-flash-preview-05-20")
MIN_TRANSCRIPT_LENGTH = int(os.getenv("MIN_TRANSCRIPT_LENGTH", "50"))

# Configuration for transcription buffering
CHINESE_BUFFER_SIZE = int(os.getenv("CHINESE_BUFFER_SIZE", "120"))  # Much larger for Chinese
ENGLISH_BUFFER_SIZE = int(os.getenv("ENGLISH_BUFFER_SIZE", "80"))  # Increased for English
CHINESE_TIMEOUT_SECONDS = float(os.getenv("CHINESE_TIMEOUT_SECONDS", "6.0"))  # Longer timeout
ENGLISH_TIMEOUT_SECONDS = float(os.getenv("ENGLISH_TIMEOUT_SECONDS", "4.0"))  # Increased timeout

# Global session tracking
active_sessions: Dict[int, Dict[str, Any]] = {}


class TranscriptAnalysis(BaseModel):
    """Pydantic model for structured transcript analysis output."""

    summary: str


class TranscriptAnalyzer:
    """Separate analyzer using standard Gemini text generation API with structured output."""

    def __init__(self, api_key: str, model: str = ANALYSIS_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def analyze_transcript(self, transcript: str) -> Optional[str]:
        """Analyze transcript using Gemini text generation API with structured output."""
        if len(transcript.strip()) < MIN_TRANSCRIPT_LENGTH:
            return None

        try:
            response = self.client.models.generate_content(model=self.model, config={"response_mime_type": "application/json", "response_schema": TranscriptAnalysis, "system_instruction": "You are an expert audio content analyzer. Provide a concise summary of the key points and themes in the transcript."}, contents=[f"Analyze this transcript and provide a summary:\n\n{transcript}"])

            if response and response.text:
                return response.text.strip()
            return None

        except Exception as e:
            print(f"Analysis error: {e}")
            return None


# Serve index.html from the root path
@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.get("/api/analysis/config")
async def get_analysis_config():
    """Get current analysis configuration."""
    return {"analysis_interval_seconds": ANALYSIS_INTERVAL_SECONDS, "analysis_model": ANALYSIS_MODEL, "min_transcript_length": MIN_TRANSCRIPT_LENGTH, "chinese_buffer_size": CHINESE_BUFFER_SIZE, "english_buffer_size": ENGLISH_BUFFER_SIZE, "chinese_timeout_seconds": CHINESE_TIMEOUT_SECONDS, "english_timeout_seconds": ENGLISH_TIMEOUT_SECONDS}


@app.post("/api/analysis/config")
async def update_analysis_config(config: dict):
    """Update analysis configuration at runtime."""
    global ANALYSIS_INTERVAL_SECONDS, MIN_TRANSCRIPT_LENGTH, CHINESE_BUFFER_SIZE, ENGLISH_BUFFER_SIZE, CHINESE_TIMEOUT_SECONDS, ENGLISH_TIMEOUT_SECONDS

    if "analysis_interval_seconds" in config:
        ANALYSIS_INTERVAL_SECONDS = float(config["analysis_interval_seconds"])

    if "min_transcript_length" in config:
        MIN_TRANSCRIPT_LENGTH = int(config["min_transcript_length"])

    if "chinese_buffer_size" in config:
        CHINESE_BUFFER_SIZE = int(config["chinese_buffer_size"])

    if "english_buffer_size" in config:
        ENGLISH_BUFFER_SIZE = int(config["english_buffer_size"])

    if "chinese_timeout_seconds" in config:
        CHINESE_TIMEOUT_SECONDS = float(config["chinese_timeout_seconds"])

    if "english_timeout_seconds" in config:
        ENGLISH_TIMEOUT_SECONDS = float(config["english_timeout_seconds"])

    return {"status": "success", "message": "Configuration updated", "current_config": {"analysis_interval_seconds": ANALYSIS_INTERVAL_SECONDS, "analysis_model": ANALYSIS_MODEL, "min_transcript_length": MIN_TRANSCRIPT_LENGTH, "chinese_buffer_size": CHINESE_BUFFER_SIZE, "english_buffer_size": ENGLISH_BUFFER_SIZE, "chinese_timeout_seconds": CHINESE_TIMEOUT_SECONDS, "english_timeout_seconds": ENGLISH_TIMEOUT_SECONDS}}


@app.get("/debug/sessions")
async def get_active_sessions():
    """Debug endpoint to check active session status."""
    session_status = {}
    for session_id, session_data in active_sessions.items():
        session_status[session_id] = {
            "start_time": session_data["start_time"],
            "duration": time.time() - session_data["start_time"],
            "has_listener_task": session_data.get("listener_task") is not None,
            "listener_done": session_data.get("listener_task").done() if session_data.get("listener_task") else None,
            "has_gemini_session": session_data.get("gemini_session") is not None,
        }
    return {"active_sessions": len(active_sessions), "sessions": session_status}


@app.get("/debug/test-websocket")
async def test_websocket_connection():
    """Test endpoint to verify WebSocket connectivity."""
    return {"status": "WebSocket endpoint available at ws://localhost:8888/ws"}


GEMINI_API_KEY_ENV = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY")

# Global counter for WebSocket session IDs
websocket_session_counter = 0


async def cleanup_session(session_id: int):
    """Clean up session resources properly."""
    if session_id in active_sessions:
        session_data = active_sessions[session_id]

        # Cancel any running tasks
        if "listener_task" in session_data and session_data["listener_task"]:
            session_data["listener_task"].cancel()
            try:
                await session_data["listener_task"]
            except asyncio.CancelledError:
                pass

        # Cancel analysis task
        if "analysis_task" in session_data and session_data["analysis_task"]:
            session_data["analysis_task"].cancel()
            try:
                await session_data["analysis_task"]
            except asyncio.CancelledError:
                pass

        # Clean up Gemini session if it exists
        if "gemini_session" in session_data:
            try:
                # The session will be automatically closed by the context manager
                pass
            except Exception as e:
                print(f"[SessID: {session_id}] Error cleaning up Gemini session: {e}")

        # Remove from active sessions
        del active_sessions[session_id]
        print(f"[SessID: {session_id}] Session cleaned up successfully.")


async def periodic_analysis_task(session_id: int, analyzer: TranscriptAnalyzer, websocket: WebSocket):
    """Background task that analyzes accumulated transcript every N seconds."""
    print(f"[SessID: {session_id}] Starting periodic analysis task (interval: {ANALYSIS_INTERVAL_SECONDS}s)")

    # Track when transcription stops to end analysis
    no_transcript_cycles = 0
    MAX_IDLE_CYCLES = 3  # Stop after 3 cycles without new transcript

    while session_id in active_sessions:
        try:
            await asyncio.sleep(ANALYSIS_INTERVAL_SECONDS)

            if session_id not in active_sessions:
                break

            session_data = active_sessions[session_id]
            transcript_buffer = session_data.get("transcript_buffer", "")
            transcription_active = session_data.get("transcription_active", False)
            last_transcript_time = session_data.get("last_transcript_time", 0)

            # Check if transcription has been inactive for too long
            time_since_last_transcript = time.time() - last_transcript_time
            if not transcription_active and time_since_last_transcript > ANALYSIS_INTERVAL_SECONDS * 2:
                no_transcript_cycles += 1
                print(f"[SessID: {session_id}] No transcript activity for {time_since_last_transcript:.1f}s (cycle {no_transcript_cycles}/{MAX_IDLE_CYCLES})")

                if no_transcript_cycles >= MAX_IDLE_CYCLES:
                    print(f"[SessID: {session_id}] Stopping analysis - transcription appears to have ended")
                    break
            else:
                no_transcript_cycles = 0  # Reset counter if there's activity

            if transcript_buffer and len(transcript_buffer.strip()) >= MIN_TRANSCRIPT_LENGTH:
                print(f"[SessID: {session_id}] Running analysis on {len(transcript_buffer)} chars of transcript")

                analysis_result = await analyzer.analyze_transcript(transcript_buffer)

                if analysis_result:
                    try:
                        # Send structured analysis to frontend with update flag
                        await websocket.send_text(f"🤖 AI ANALYSIS UPDATE: {analysis_result}")
                        print(f"[SessID: {session_id}] Sent updated analysis result")

                        # Keep only recent transcript (last 1000 chars) to avoid memory issues
                        if len(transcript_buffer) > 1000:
                            session_data["transcript_buffer"] = transcript_buffer[-800:]

                    except Exception as e:
                        print(f"[SessID: {session_id}] Error sending analysis: {e}")
                        break

        except asyncio.CancelledError:
            print(f"[SessID: {session_id}] Analysis task cancelled")
            break
        except Exception as e:
            print(f"[SessID: {session_id}] Analysis task error: {e}")
            # Continue running despite errors

    print(f"[SessID: {session_id}] Periodic analysis task finished")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global websocket_session_counter
    websocket_session_counter += 1
    current_session_id = websocket_session_counter

    await websocket.accept()
    print(f"[SessID: {current_session_id}] WebSocket client connected.")

    # Initialize session tracking with transcript buffer
    active_sessions[current_session_id] = {"start_time": time.time(), "websocket": websocket, "listener_task": None, "gemini_session": None, "transcript_buffer": "", "analysis_task": None, "transcription_active": False, "last_transcript_time": time.time()}

    API_KEY_TO_USE = None
    if GOOGLE_API_KEY_ENV and GEMINI_API_KEY_ENV:
        print(f"[SessID: {current_session_id}] Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        API_KEY_TO_USE = GOOGLE_API_KEY_ENV
    elif GOOGLE_API_KEY_ENV:
        print(f"[SessID: {current_session_id}] Using GOOGLE_API_KEY.")
        API_KEY_TO_USE = GOOGLE_API_KEY_ENV
    elif GEMINI_API_KEY_ENV:
        print(f"[SessID: {current_session_id}] Using GEMINI_API_KEY.")
        API_KEY_TO_USE = GEMINI_API_KEY_ENV
    else:
        print(f"[SessID: {current_session_id}] Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set.")
        await websocket.send_text("Error: API Key is not configured on the server.")
        await websocket.close()
        await cleanup_session(current_session_id)
        print(f"[SessID: {current_session_id}] Closed WebSocket due to missing API Key.")
        return

    client = None
    try:
        print(f"[SessID: {current_session_id}] Instantiating genai.Client with API key.")
        client = genai.Client(api_key=API_KEY_TO_USE)
        print(f"[SessID: {current_session_id}] genai.Client instantiated successfully.")
    except AttributeError:
        error_msg = "ERROR: 'genai.Client' not found. Ensure SDK is updated (pip install -U google-generativeai)."
        print(f"[SessID: {current_session_id}] {error_msg}")
        await websocket.send_text(error_msg)
        await websocket.close()
        await cleanup_session(current_session_id)
        return
    except Exception as e:
        error_msg = f"Error initializing Gemini Client: {e}"
        print(f"[SessID: {current_session_id}] {error_msg}")
        await websocket.send_text(f"ERROR: {error_msg}")
        await websocket.close()
        await cleanup_session(current_session_id)
        return

    model_name = "gemini-2.0-flash-live-001"

    # Configure session for optimal transcription and analysis
    config = {
        "response_modalities": ["TEXT"],
        "input_audio_transcription": {},  # Enable high-quality native transcription
        "output_audio_transcription": {},  # Also transcribe any audio responses
        "system_instruction": """You are an intelligent audio analysis assistant. Provide helpful analysis and insights about the audio content you hear.

Focus on:
- Summarizing key points and themes
- Identifying speakers and their characteristics  
- Noting important context or background information
- Providing relevant insights about the content

Respond naturally when you have meaningful observations to share.""",
        "realtime_input_config": {
            "automatic_activity_detection": {
                "disabled": False,
                "start_of_speech_sensitivity": gemini_types.StartSensitivity.START_SENSITIVITY_HIGH,
                "end_of_speech_sensitivity": gemini_types.EndSensitivity.END_SENSITIVITY_HIGH,
                "prefix_padding_ms": 500,  # Increased for better Chinese capture
                "silence_duration_ms": 5000,  # Reduced for faster Chinese processing
            }
        },
    }

    # Initialize the separate analyzer for periodic analysis
    analyzer = TranscriptAnalyzer(API_KEY_TO_USE, ANALYSIS_MODEL)

    gemini_listener_task = None
    analysis_task = None

    try:
        print(f"[SessID: {current_session_id}] Attempting to connect to Gemini Live API (model: {model_name}).")
        async with client.aio.live.connect(model=model_name, config=config) as session:
            print(f"[SessID: {current_session_id}] Gemini Live API session started successfully.")

            # Store session reference for cleanup
            active_sessions[current_session_id]["gemini_session"] = session

            await websocket.send_text("INFO: Connected to Gemini Live API.")

            # Start the periodic analysis task
            analysis_task = asyncio.create_task(periodic_analysis_task(current_session_id, analyzer, websocket))
            active_sessions[current_session_id]["analysis_task"] = analysis_task
            print(f"[SessID: {current_session_id}] Started periodic analysis task")

            async def listen_gemini_and_send_to_client():
                print(f"[SessID: {current_session_id}] Listener task started.")

                # Enhanced buffering for better Chinese text grouping
                transcription_buffer = ""
                last_transcription_send = time.time()

                def is_chinese_text(text: str) -> bool:
                    """Check if text contains Chinese characters."""
                    return any("\u4e00" <= char <= "\u9fff" for char in text)

                def should_send_transcription(buffer: str, new_text: str, time_since_last: float) -> bool:
                    """Simplified logic - accumulate more text without punctuation filtering."""
                    if not buffer.strip():
                        return False

                    is_chinese = is_chinese_text(buffer)

                    if is_chinese:
                        # For Chinese: much larger buffers and longer timeouts
                        # Send only when we have substantial content or significant time passed
                        if len(buffer) >= CHINESE_BUFFER_SIZE:
                            return True
                        if time_since_last >= CHINESE_TIMEOUT_SECONDS:
                            return True
                    else:
                        # For English: moderate buffers
                        if len(buffer) >= ENGLISH_BUFFER_SIZE:
                            return True
                        if time_since_last >= ENGLISH_TIMEOUT_SECONDS:
                            return True

                    return False

                try:
                    async for response_message in session.receive():
                        # Check if WebSocket is still connected
                        if current_session_id not in active_sessions:
                            print(f"[SessID: {current_session_id}] Session no longer active, stopping listener.")
                            break

                        # Handle different message types
                        text_to_send = None

                        # 1. Check for direct text attribute (AI analysis)
                        if hasattr(response_message, "text") and response_message.text:
                            text_to_send = f"🤖 AI ANALYSIS: {response_message.text}"

                        # Check for setup completion
                        elif hasattr(response_message, "setup_complete") and response_message.setup_complete:
                            continue

                        # 2. Check for server content (transcription and analysis)
                        elif hasattr(response_message, "server_content") and response_message.server_content:
                            server_content = response_message.server_content

                            # Handle native input transcription with enhanced smart buffering
                            if hasattr(server_content, "input_transcription") and server_content.input_transcription:
                                transcription_text = server_content.input_transcription.text
                                if transcription_text and transcription_text.strip():
                                    # Accumulate transcription for better grouping
                                    transcription_buffer += transcription_text
                                    current_time = time.time()
                                    time_since_last = current_time - last_transcription_send

                                    # Update session buffer for periodic analysis
                                    if current_session_id in active_sessions:
                                        session_data = active_sessions[current_session_id]
                                        session_data["transcript_buffer"] += transcription_text + " "
                                        session_data["transcription_active"] = True
                                        session_data["last_transcript_time"] = time.time()

                                    # Use enhanced smart sending logic
                                    if should_send_transcription(transcription_buffer, transcription_text, time_since_last):
                                        text_to_send = f"🎤 TRANSCRIPTION: {transcription_buffer.strip()}"

                                        # More detailed logging for Chinese vs English
                                        is_chinese = is_chinese_text(transcription_buffer)
                                        lang_info = "Chinese" if is_chinese else "English"
                                        print(f"[SessID: {current_session_id}] {lang_info} transcription ({len(transcription_buffer)} chars): '{transcription_buffer.strip()}'")

                                        # Reset buffer
                                        transcription_buffer = ""
                                        last_transcription_send = current_time

                            # Handle output transcription (if model generates audio responses)
                            if hasattr(server_content, "output_transcription") and server_content.output_transcription:
                                output_text = server_content.output_transcription.text
                                if output_text and output_text.strip():
                                    print(f"[SessID: {current_session_id}] Model audio response transcribed: '{output_text.strip()}'")
                                    # Could send this too if needed for debugging

                            # Handle AI analysis responses
                            if not text_to_send and hasattr(server_content, "parts") and server_content.parts:
                                text_parts = []
                                try:
                                    for part_item in server_content.parts:
                                        if hasattr(part_item, "text") and part_item.text:
                                            text_parts.append(part_item.text)
                                except (TypeError, AttributeError) as e:
                                    print(f"[SessID: {current_session_id}] Error iterating parts: {e}")

                                if text_parts:
                                    analysis_text = "".join(text_parts)
                                    text_to_send = f"🤖 AI ANALYSIS: {analysis_text}"
                                    print(f"[SessID: {current_session_id}] AI analysis: {analysis_text[:100]}...")

                            # Handle model turns
                            if hasattr(server_content, "model_turn") and server_content.model_turn:
                                if hasattr(server_content.model_turn, "parts") and server_content.model_turn.parts:
                                    model_turn_text = []
                                    for part in server_content.model_turn.parts:
                                        if hasattr(part, "text") and part.text:
                                            model_turn_text.append(part.text)
                                    if model_turn_text:
                                        analysis_text = "".join(model_turn_text)
                                        text_to_send = f"🤖 AI ANALYSIS: {analysis_text}"
                                        print(f"[SessID: {current_session_id}] Model turn analysis: {analysis_text[:100]}...")

                        # Handle setup completion
                        if hasattr(response_message, "setup_complete") and response_message.setup_complete:
                            print(f"[SessID: {current_session_id}] Setup complete")
                            continue

                        # Send text if we found any
                        if text_to_send:
                            try:
                                await websocket.send_text(text_to_send)
                            except Exception as e:
                                print(f"[SessID: {current_session_id}] Failed to send text to WebSocket: {e}")
                                break

                        # Handle usage metadata
                        if hasattr(response_message, "usage_metadata") and response_message.usage_metadata:
                            usage = response_message.usage_metadata
                            print(f"[SessID: {current_session_id}] Gemini Usage - Tokens: {usage.total_token_count}")

                        # Handle errors
                        if hasattr(response_message, "error") and response_message.error is not None:
                            error_obj = response_message.error
                            error_msg_detail = error_obj.message if hasattr(error_obj, "message") else str(error_obj)
                            print(f"[SessID: {current_session_id}] Gemini session error: {error_msg_detail}")
                            try:
                                await websocket.send_text(f"GEMINI_ERROR: {error_msg_detail}")
                            except Exception:
                                pass

                except WebSocketDisconnect:
                    print(f"[SessID: {current_session_id}] WebSocket client disconnected while listening to Gemini.")
                except Exception as e:
                    print(f"[SessID: {current_session_id}] Error in listen_gemini_and_send_to_client: {e}")
                    import traceback

                    print(traceback.format_exc())
                finally:
                    # Send any remaining buffered transcription
                    if transcription_buffer.strip():
                        try:
                            await websocket.send_text(f"🎤 TRANSCRIPTION: {transcription_buffer.strip()}")
                            print(f"[SessID: {current_session_id}] Final transcription: '{transcription_buffer.strip()}'")
                        except Exception:
                            pass

                    # Mark transcription as inactive
                    if current_session_id in active_sessions:
                        active_sessions[current_session_id]["transcription_active"] = False
                    print(f"[SessID: {current_session_id}] Gemini listening task finished.")

            gemini_listener_task = asyncio.create_task(listen_gemini_and_send_to_client())
            active_sessions[current_session_id]["listener_task"] = gemini_listener_task

            try:
                # Send initial status message to client
                await websocket.send_text("INFO: Ready to receive audio data. Please start speaking.")
                print(f"[SessID: {current_session_id}] Ready to receive audio data.")

                audio_chunk_count = 0

                while current_session_id in active_sessions:
                    audio_data = await websocket.receive_bytes()
                    if not audio_data:
                        print(f"[SessID: {current_session_id}] Received empty audio data, client closing.")
                        break

                    # Send audio to Gemini Live API
                    await session.send_realtime_input(audio=gemini_types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000"))

                    audio_chunk_count += 1

                    # Log progress occasionally
                    if audio_chunk_count % 200 == 0:
                        print(f"[SessID: {current_session_id}] Processed {audio_chunk_count} audio chunks")

            except WebSocketDisconnect:
                print(f"[SessID: {current_session_id}] WebSocket client disconnected while receiving audio.")
            except Exception as e:
                print(f"[SessID: {current_session_id}] Error receiving audio from client: {e}")
                import traceback

                print(f"[SessID: {current_session_id}] Traceback: {traceback.format_exc()}")
            finally:
                print(f"[SessID: {current_session_id}] Audio receiving loop finished.")

                # Cancel tasks
                if gemini_listener_task and not gemini_listener_task.done():
                    print(f"[SessID: {current_session_id}] Cancelling Gemini listener task.")
                    gemini_listener_task.cancel()
                try:
                    if gemini_listener_task:
                        await gemini_listener_task
                except asyncio.CancelledError:
                    print(f"[SessID: {current_session_id}] Gemini listener task cancelled.")
                except Exception as e:
                    print(f"[SessID: {current_session_id}] Exception in Gemini listener task: {e}")

                if analysis_task and not analysis_task.done():
                    print(f"[SessID: {current_session_id}] Cancelling analysis task.")
                    analysis_task.cancel()
                try:
                    if analysis_task:
                        await analysis_task
                except asyncio.CancelledError:
                    print(f"[SessID: {current_session_id}] Analysis task cancelled.")
                except Exception as e:
                    print(f"[SessID: {current_session_id}] Exception in analysis task: {e}")

    except Exception as e:
        error_message = f"Failed to connect/interact with Gemini Live API: {e}"
        print(f"[SessID: {current_session_id}] {error_message}")
        try:
            if current_session_id in active_sessions:
                await websocket.send_text(f"ERROR: {str(error_message)}")
        except Exception:
            pass
    finally:
        print(f"[SessID: {current_session_id}] Exiting websocket_endpoint.")

        # Always clean up the session
        await cleanup_session(current_session_id)

        # Close WebSocket if still open
        try:
            if websocket.client_state.name not in ["DISCONNECTED", "CLOSED"]:
                await websocket.close()
        except Exception as e:
            print(f"[SessID: {current_session_id}] Error closing websocket: {e}")

        print(f"[SessID: {current_session_id}] WebSocket connection fully closed.")


if __name__ == "__main__":
    if not (GEMINI_API_KEY_ENV or GOOGLE_API_KEY_ENV):
        print("FATAL: API Key (GEMINI_API_KEY or GOOGLE_API_KEY) is not set.")
    else:
        print("Starting Gemini Live Audio Transcription & Analysis Server")
        print("==========================================================")
        print(f"Server: http://localhost:8888")
        print(f"WebSocket: ws://localhost:8888/ws")
        print(f"Live Model: gemini-2.0-flash-live-001")
        print(f"Analysis Model: {ANALYSIS_MODEL}")
        print(f"Analysis Interval: {ANALYSIS_INTERVAL_SECONDS}s")
        print(f"Min Transcript Length: {MIN_TRANSCRIPT_LENGTH} chars")
        print("==========================================================")
        print("SIMPLIFIED BUFFERING - Raw text accumulation without punctuation filtering")
        print(f"Chinese Buffer: {CHINESE_BUFFER_SIZE} chars | Timeout: {CHINESE_TIMEOUT_SECONDS}s")
        print(f"English Buffer: {ENGLISH_BUFFER_SIZE} chars | Timeout: {ENGLISH_TIMEOUT_SECONDS}s")
        print("==========================================================")
        print("Open http://localhost:8888 in your browser to start!")
        uvicorn.run(app, host="0.0.0.0", port=8888)
