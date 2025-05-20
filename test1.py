import asyncio
import subprocess

from whisperlivekit.audio_processor import AudioProcessor


async def transcribe_file(path):
    ap = AudioProcessor()
    results = await ap.create_tasks()

    # start a background task to print captions as they arrive
    async def print_results():
        async for resp in results:
            lines = resp.get("lines", [])
            caption = " ".join([line["text"] for line in lines])
            buffer = resp.get("buffer_transcription", "")
            full_caption = (caption + " " + buffer).strip()
            print(full_caption)

    consumer_task = asyncio.create_task(print_results())

    # spawn ffmpeg to re-encode your file as WebM/Opus @16 kHz mono
    ff = subprocess.Popen(
        ["ffmpeg", "-i", path, "-f", "webm", "-c:a", "libopus", "-ar", "16000", "-ac", "1", "pipe:1"],
        stdout=subprocess.PIPE,
    )

    # pump ffmpeg's stdout into process_audio
    while True:
        chunk = ff.stdout.read(4096)
        if not chunk:
            break
        await ap.process_audio(chunk)

    # send the "end of stream" signal
    await ap.process_audio(b"")

    # wait for the printing task to finish
    await consumer_task


asyncio.run(transcribe_file("audio/c-IX1061gw0_1-17.mp3"))
