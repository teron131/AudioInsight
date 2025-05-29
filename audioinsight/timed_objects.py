from dataclasses import dataclass
from typing import Optional


@dataclass
class TimedText:
    start: Optional[float]
    end: Optional[float]
    text: Optional[str] = ""
    speaker: Optional[int] = 0
    probability: Optional[float] = None
    is_dummy: Optional[bool] = False


@dataclass
class ASRToken(TimedText):
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        # Early return for zero offset to avoid unnecessary object creation
        if offset == 0:
            return self

        # Handle None values efficiently without additional computation
        start = self.start + offset if self.start is not None else None
        end = self.end + offset if self.end is not None else None
        return ASRToken(start, end, self.text, self.speaker, self.probability, self.is_dummy)


@dataclass
class Sentence(TimedText):
    pass


@dataclass
class Transcript(TimedText):
    pass


@dataclass
class SpeakerSegment(TimedText):
    pass
