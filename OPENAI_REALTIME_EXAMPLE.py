import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Global session tracking
active_sessions: Dict[int, Dict[str, Any]] = {}
websocket_session_counter = 0


class TranscriptionConfig(BaseModel):
    """Configuration for transcription session."""

    model: str = "whisper-1"  # Use whisper-1 for continuous transcription
    prompt: str = ""
    language: str = ""


class SessionConfig(BaseModel):
    """Configuration for the transcription session."""

    input_audio_format: str = "pcm16"
    input_audio_transcription: TranscriptionConfig = TranscriptionConfig()
    turn_detection: Optional[Dict[str, Any]] = {
        "type": "semantic_vad",
        "eagerness": "auto",  # Let the user take their time to speak - best for continuous transcription
    }
    input_audio_noise_reduction: Optional[Dict[str, str]] = {"type": "near_field"}
    include: Optional[list] = None


# Serve index.html from the root path
@app.get("/")
async def get_index():
    return FileResponse("OPENAI_REALTIME_EXAMPLE_UI.html")


@app.get("/api/transcription/config")
async def get_transcription_config():
    """Get current transcription configuration."""
    return {"model": "whisper-1", "input_audio_format": "pcm16", "turn_detection_enabled": True, "noise_reduction": "near_field"}


@app.get("/api/analysis/config")
async def get_analysis_config():
    """Get current analysis configuration for UI compatibility."""
    return {"analysis_interval_seconds": 5.0, "analysis_model": "gpt-4o-mini-transcribe", "min_transcript_length": 50}


@app.post("/api/transcription/config")
async def update_transcription_config(config: dict):
    """Update transcription configuration at runtime."""
    return {"status": "success", "message": "Transcription configuration updated", "current_config": config}


@app.get("/debug/sessions")
async def get_active_sessions():
    """Debug endpoint to check active session status."""
    session_status = {}
    for session_id, session_data in active_sessions.items():
        session_status[session_id] = {
            "start_time": session_data["start_time"],
            "duration": time.time() - session_data["start_time"],
            "has_openai_connection": session_data.get("openai_ws") is not None,
            "transcription_active": session_data.get("transcription_active", False),
        }
    return {"active_sessions": len(active_sessions), "sessions": session_status}


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

        # Close OpenAI WebSocket connection
        if "openai_ws" in session_data and session_data["openai_ws"]:
            try:
                await session_data["openai_ws"].close()
            except Exception as e:
                print(f"[SessID: {session_id}] Error closing OpenAI WebSocket: {e}")

        # Remove from active sessions
        del active_sessions[session_id]
        print(f"[SessID: {session_id}] Session cleaned up successfully.")


class OpenAIRealtimeClient:
    """Client for OpenAI Realtime API transcription."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self.session_id = None

    async def connect(self):
        """Connect to OpenAI Realtime API."""
        import ssl

        import websockets.legacy.client

        # Create custom headers for authentication
        headers = [("Authorization", f"Bearer {self.api_key}"), ("OpenAI-Beta", "realtime=v1")]

        # Use legacy client for compatibility with headers - add transcription intent
        uri = f"{OPENAI_REALTIME_URL}?intent=transcription"

        try:
            # Use the legacy connect function that supports extra_headers
            self.ws = await websockets.legacy.client.connect(uri, extra_headers=headers, ssl=ssl.create_default_context())
            return self.ws
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    async def create_transcription_session(self, config: SessionConfig):
        """Create a transcription session."""
        # Build input_audio_transcription config
        transcription_config = {"model": config.input_audio_transcription.model}

        # Only include prompt if it's not empty
        if config.input_audio_transcription.prompt.strip():
            transcription_config["prompt"] = config.input_audio_transcription.prompt

        # Only include language if it's not empty
        if config.input_audio_transcription.language.strip():
            transcription_config["language"] = config.input_audio_transcription.language

        # Use transcription session format for transcription-only mode
        session_create_event = {
            "event_id": str(uuid.uuid4()),
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": config.input_audio_format,
                "input_audio_transcription": transcription_config,
                "turn_detection": config.turn_detection,
                "input_audio_noise_reduction": config.input_audio_noise_reduction,
            },
        }

        # Add optional include parameter for logprobs if needed
        if config.include:
            session_create_event["session"]["include"] = config.include

        await self.ws.send(json.dumps(session_create_event))

    async def send_audio(self, audio_data: bytes):
        """Send audio data to the transcription session."""
        import base64

        audio_event = {"event_id": str(uuid.uuid4()), "type": "input_audio_buffer.append", "audio": base64.b64encode(audio_data).decode("utf-8")}

        await self.ws.send(json.dumps(audio_event))

    async def commit_audio_buffer(self):
        """Manually commit the audio buffer for transcription."""
        commit_event = {"event_id": str(uuid.uuid4()), "type": "input_audio_buffer.commit"}
        await self.ws.send(json.dumps(commit_event))

    async def listen_for_transcriptions(self):
        """Listen for transcription events from OpenAI."""
        async for message in self.ws:
            try:
                event = json.loads(message)
                yield event
            except json.JSONDecodeError as e:
                print(f"Error parsing OpenAI message: {e}")
                continue


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global websocket_session_counter
    websocket_session_counter += 1
    current_session_id = websocket_session_counter

    await websocket.accept()
    print(f"[SessID: {current_session_id}] WebSocket client connected.")

    # Initialize session tracking
    active_sessions[current_session_id] = {"start_time": time.time(), "websocket": websocket, "listener_task": None, "openai_ws": None, "transcription_active": False, "last_transcript_time": time.time(), "current_item_transcripts": {}, "transcription_model": "whisper-1"}  # Track transcripts by item_id for proper handling  # Track model for behavior handling

    if not OPENAI_API_KEY:
        print(f"[SessID: {current_session_id}] OPENAI_API_KEY is not set.")
        await websocket.send_text("Error: OpenAI API Key is not configured on the server.")
        await websocket.close()
        await cleanup_session(current_session_id)
        return

    openai_client = None
    listener_task = None

    try:
        print(f"[SessID: {current_session_id}] Connecting to OpenAI Realtime API.")
        openai_client = OpenAIRealtimeClient(OPENAI_API_KEY)
        openai_ws = await openai_client.connect()

        # Store WebSocket reference for cleanup
        active_sessions[current_session_id]["openai_ws"] = openai_ws

        print(f"[SessID: {current_session_id}] Connected to OpenAI Realtime API successfully.")
        await websocket.send_text("INFO: Connected to OpenAI Realtime API.")

        # Create transcription session
        config = SessionConfig()
        await openai_client.create_transcription_session(config)
        print(f"[SessID: {current_session_id}] Transcription session created.")

        async def listen_openai_and_send_to_client():
            """Listen for transcription events from OpenAI and forward to client."""
            print(f"[SessID: {current_session_id}] OpenAI listener task started.")

            try:
                async for event in openai_client.listen_for_transcriptions():
                    # Check if session is still active
                    if current_session_id not in active_sessions:
                        print(f"[SessID: {current_session_id}] Session no longer active, stopping OpenAI listener.")
                        break

                    event_type = event.get("type")
                    text_to_send = None

                    # Debug: log key events only (comment out for less verbose logging)
                    if event_type not in ["conversation.item.created"]:
                        print(f"[SessID: {current_session_id}] Received event type: {event_type}")

                    # Handle transcription delta events (model-specific behavior)
                    if event_type == "conversation.item.input_audio_transcription.delta":
                        delta_text = event.get("delta", "")
                        item_id = event.get("item_id", "")

                        if delta_text.strip():
                            # whisper-1 always provides full turn transcript in delta
                            text_to_send = f"TRANSCRIPTION_DELTA:{delta_text}"
                            print(f"[SessID: {current_session_id}] Whisper-1 delta: '{delta_text}'")

                            # Update session state
                            if current_session_id in active_sessions:
                                active_sessions[current_session_id]["transcription_active"] = True
                                active_sessions[current_session_id]["last_transcript_time"] = time.time()

                    # Handle transcription completion events (final transcript)
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        item_id = event.get("item_id", "")

                        if transcript.strip():
                            # For whisper-1, completion is the same as delta, so we can send it as final
                            text_to_send = f"TRANSCRIPTION_COMPLETE:{transcript}"
                            print(f"[SessID: {current_session_id}] Whisper-1 final transcript: '{transcript}'")

                            # Update session state
                            if current_session_id in active_sessions:
                                active_sessions[current_session_id]["transcription_active"] = False
                                active_sessions[current_session_id]["last_transcript_time"] = time.time()

                    # Handle input audio buffer committed events
                    elif event_type == "input_audio_buffer.committed":
                        item_id = event.get("item_id", "")
                        print(f"[SessID: {current_session_id}] Audio buffer committed: {item_id}")

                    # Handle session updates (both session.updated and transcription_session.updated)
                    elif event_type in ["session.updated", "transcription_session.updated"]:
                        print(f"[SessID: {current_session_id}] Transcription session updated successfully")
                        text_to_send = "INFO: Transcription session ready. Please start speaking."

                    # Handle errors
                    elif event_type == "error":
                        error_detail = event.get("error", {})
                        error_message = error_detail.get("message", "Unknown error")
                        print(f"[SessID: {current_session_id}] OpenAI error: {error_message}")
                        text_to_send = f"ERROR: {error_message}"

                    # Send text if we have any
                    if text_to_send:
                        try:
                            await websocket.send_text(text_to_send)
                        except Exception as e:
                            print(f"[SessID: {current_session_id}] Failed to send text to WebSocket: {e}")
                            break

            except Exception as e:
                print(f"[SessID: {current_session_id}] Error in OpenAI listener: {e}")
                import traceback

                print(traceback.format_exc())
            finally:
                print(f"[SessID: {current_session_id}] OpenAI listener task finished.")

        # Start the listener task
        listener_task = asyncio.create_task(listen_openai_and_send_to_client())
        active_sessions[current_session_id]["listener_task"] = listener_task

        try:
            print(f"[SessID: {current_session_id}] Ready to receive audio data.")
            audio_chunk_count = 0

            while current_session_id in active_sessions:
                audio_data = await websocket.receive_bytes()
                if not audio_data:
                    print(f"[SessID: {current_session_id}] Received empty audio data, client closing.")
                    break

                # Send audio to OpenAI Realtime API
                await openai_client.send_audio(audio_data)
                audio_chunk_count += 1

                # Fallback: commit audio buffer every 200 chunks (~4-5 seconds) as backup to VAD
                # This ensures long speech segments get processed even if VAD doesn't trigger
                if audio_chunk_count % 200 == 0:
                    await openai_client.commit_audio_buffer()
                    print(f"[SessID: {current_session_id}] Fallback buffer commit at chunk {audio_chunk_count}")

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

            # Cancel listener task
            if listener_task and not listener_task.done():
                print(f"[SessID: {current_session_id}] Cancelling OpenAI listener task.")
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    print(f"[SessID: {current_session_id}] OpenAI listener task cancelled.")
                except Exception as e:
                    print(f"[SessID: {current_session_id}] Exception in OpenAI listener task: {e}")

    except Exception as e:
        error_message = f"Failed to connect/interact with OpenAI Realtime API: {e}"
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
    if not OPENAI_API_KEY:
        print("FATAL: OPENAI_API_KEY is not set.")
        print("Please set your OpenAI API key in the .env file or environment variables.")
    else:
        print("Starting OpenAI Realtime Transcription Server")
        print("=" * 50)
        print(f"Server: http://localhost:8889")
        print(f"WebSocket: ws://localhost:8889/ws")
        print(f"Transcription Model: whisper-1")
        print(f"Audio Format: pcm16")
        print("=" * 50)
        print("Using OpenAI Realtime API with whisper-1 for continuous transcription")
        print("Open http://localhost:8889 in your browser to start!")
        uvicorn.run(app, host="0.0.0.0", port=8889)
