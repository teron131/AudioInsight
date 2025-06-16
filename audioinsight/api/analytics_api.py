import time
from pathlib import Path

import librosa
import numpy as np
from fastapi import APIRouter

from ..audioinsight_server import error_response, success_response
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.get("/api/analytics/usage")
async def get_usage_analytics():
    """Get usage analytics and statistics."""
    try:
        analytics = {"session": {"current_session_active": app.kit is not None, "uptime": time.time() - getattr(app.kit, "created_at", time.time()) if app.kit else 0}, "timestamp": time.time()}

        return success_response("Usage analytics retrieved", {"analytics": analytics})
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}")
        return error_response(f"Error getting analytics: {str(e)}")


@router.post("/api/audio/analyze")
async def analyze_audio_quality(file_path: str):
    """Analyze audio quality and provide recommendations."""
    try:
        audio_path = Path(file_path)
        if not audio_path.exists():
            return error_response("Audio file not found")

        # Load audio file
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = len(y) / sr

        # Basic audio analysis
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zcr)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroids)

        # Basic noise estimation (simplified)
        noise_estimate = np.std(y[: int(0.1 * sr)])  # First 100ms as noise reference

        # Calculate SNR estimate
        signal_power = np.mean(y**2)
        noise_power = noise_estimate**2
        snr_estimate = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

        # Generate recommendations
        recommendations = []

        if avg_rms < 0.01:
            recommendations.append("Audio level is very low. Consider increasing input gain.")
        elif avg_rms > 0.8:
            recommendations.append("Audio level is very high. Risk of clipping.")

        if snr_estimate < 10:
            recommendations.append("High noise level detected. Consider noise reduction.")

        if sr < 16000:
            recommendations.append("Sample rate below 16kHz may affect transcription quality.")

        analysis = {
            "file_info": {
                "filename": audio_path.name,
                "duration": duration,
                "sample_rate": sr,
                "channels": 1,  # librosa loads as mono by default
            },
            "quality_metrics": {
                "average_rms": float(avg_rms),
                "average_zcr": float(avg_zcr),
                "average_spectral_centroid": float(avg_spectral_centroid),
                "estimated_snr": float(snr_estimate),
                "noise_estimate": float(noise_estimate),
            },
            "recommendations": recommendations,
            "overall_quality": ("good" if snr_estimate > 15 and 0.01 < avg_rms < 0.8 else "fair" if snr_estimate > 5 else "poor"),
        }

        return success_response("Audio quality analysis completed", {"analysis": analysis})

    except Exception as e:
        logger.error(f"Error analyzing audio quality: {e}")
        return error_response(f"Error analyzing audio: {str(e)}")
