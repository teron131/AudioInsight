import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from whisperlivekit import WhisperLiveKit
from whisperlivekit.audio_processor import AudioProcessor

# Initialize components
app = FastAPI()
kit = WhisperLiveKit(model="large-v3-turbo", diarization=False)


# Serve the web interface
@app.get("/")
async def get():
    return HTMLResponse(kit.web_interface())  # Use the built-in web interface


# Process WebSocket connections
async def handle_websocket_results(websocket, results_generator):
    async for response in results_generator:
        await websocket.send_json(response)


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor()
    await websocket.accept()
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_task.cancel()
