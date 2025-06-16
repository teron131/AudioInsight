import asyncio
import re
import threading
from typing import Any, List, Tuple

import numpy as np
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import AudioSource, MicrophoneAudioSource
from pyannote.core import Annotation
from rx.core import Observer

from ..logging_config import get_logger
from ..timed_objects import SpeakerSegment

logger = get_logger(__name__)


class DiarizationObserver(Observer):
    """Observer that logs all data emitted by the diarization pipeline and stores speaker segments."""

    def __init__(self):
        self.speaker_segments = []
        self.processed_time = 0
        self.segment_lock = threading.Lock()

    def on_next(self, value: Tuple[Annotation, Any]):
        annotation, audio = value

        logger.debug("\n--- New Diarization Result ---")

        duration = audio.extent.end - audio.extent.start
        logger.debug(f"Audio segment: {audio.extent.start:.2f}s - {audio.extent.end:.2f}s (duration: {duration:.2f}s)")
        logger.debug(f"Audio shape: {audio.data.shape}")

        with self.segment_lock:
            if audio.extent.end > self.processed_time:
                self.processed_time = audio.extent.end
            if annotation and len(annotation._labels) > 0:
                logger.debug("\nSpeaker segments:")
                for speaker, label in annotation._labels.items():
                    for start, end in zip(label.segments_boundaries_[:-1], label.segments_boundaries_[1:]):
                        print(f"  {speaker}: {start:.2f}s-{end:.2f}s")
                        self.speaker_segments.append(SpeakerSegment(speaker=speaker, start=start, end=end))
            else:
                logger.debug("\nNo speakers detected in this segment")

    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()

    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.speaker_segments = [segment for segment in self.speaker_segments if current_time - segment.end < older_than]

    def on_error(self, error):
        """Handle an error in the stream."""
        logger.debug(f"Error in diarization stream: {error}")

    def on_completed(self):
        """Handle the completion of the stream."""
        logger.debug("Diarization stream completed")


class WebSocketAudioSource(AudioSource):
    """
    Custom AudioSource that blocks in read() until close() is called.
    Use push_audio() to inject PCM chunks.
    """

    def __init__(self, uri: str = "websocket", sample_rate: int = 16000):
        super().__init__(uri, sample_rate)
        self._closed = False
        self._close_event = threading.Event()

    def read(self):
        self._close_event.wait()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stream.on_completed()
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        if not self._closed:
            new_audio = np.expand_dims(chunk, axis=0)
            logger.debug("Add new chunk with shape:", new_audio.shape)
            self.stream.on_next(new_audio)


class DiartDiarization:
    def __init__(self, sample_rate: int = 16000, config: SpeakerDiarizationConfig = None, use_microphone: bool = False):
        self.pipeline = SpeakerDiarization(config=config)
        self.observer = DiarizationObserver()

        if use_microphone:
            self.source = MicrophoneAudioSource()
            self.custom_source = None
        else:
            self.custom_source = WebSocketAudioSource(uri="websocket_source", sample_rate=sample_rate)
            self.source = self.custom_source

        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        Only used when working with WebSocketAudioSource.
        """
        if self.custom_source:
            self.custom_source.push_audio(pcm_array)
        self.observer.clear_old_segments()
        return self.observer.get_segments()

    def close(self):
        """Close the audio source."""
        if self.custom_source:
            self.custom_source.close()

    def assign_speakers_to_tokens(self, end_attributed_speaker, tokens: list) -> float:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        Uses the segments collected by the observer.
        """
        segments = self.observer.get_segments()

        # Create speaker mapping to ensure first speaker is always labeled as 0 (UI will display as "Speaker 1")
        speaker_mapping = {}
        next_speaker_id = 0  # Start from 0, not 1, since UI adds +1 for display

        # Sort segments by start time to process chronologically
        sorted_segments = sorted(segments, key=lambda s: s.start)

        # Debug logging: Show all detected speakers
        unique_speakers = set()
        for segment in sorted_segments:
            original_speaker = segment.speaker
            unique_speakers.add(original_speaker)

            # Map speakers consistently - first detected gets ID 0 (UI will show as "Speaker 1")
            if original_speaker not in speaker_mapping:
                speaker_mapping[original_speaker] = next_speaker_id
                next_speaker_id += 1

        logger.debug(f"🎤 Speaker assignment: {len(segments)} segments, {len(unique_speakers)} unique speakers")
        if speaker_mapping:
            logger.debug(f"🎤 Speaker mapping: {speaker_mapping}")

        if len(sorted_segments) > 0:
            logger.debug(f"🎤 Recent segments:")
            for segment in sorted_segments[-3:]:  # Show last 3 segments
                mapped_id = speaker_mapping.get(segment.speaker, 0)
                ui_display_id = mapped_id + 1  # What UI will display
                logger.debug(f"   {segment.speaker} -> ID {mapped_id} (UI: Speaker {ui_display_id}): {segment.start:.2f}s-{segment.end:.2f}s")

        tokens_updated = 0
        for token in tokens:
            original_speaker = token.speaker
            for segment in sorted_segments:
                if not (segment.end <= token.start or segment.start >= token.end):
                    # Use consistent speaker mapping starting from 0 (UI will add +1 for display)
                    new_speaker = speaker_mapping.get(segment.speaker, 0)
                    if token.speaker != new_speaker:
                        token.speaker = new_speaker
                        tokens_updated += 1
                        ui_display_id = new_speaker + 1  # What UI will display
                        logger.debug(f"🔄 Updated token '{token.text}' ({token.start:.2f}s) from speaker {original_speaker} to {new_speaker} (UI: Speaker {ui_display_id})")
                    end_attributed_speaker = max(token.end, end_attributed_speaker)
                    break  # Found a match, move to next token

        if tokens_updated > 0:
            logger.debug(f"🎯 Updated {tokens_updated} tokens with speaker assignments")

        return end_attributed_speaker
