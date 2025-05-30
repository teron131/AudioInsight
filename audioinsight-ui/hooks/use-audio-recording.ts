import { useCallback, useRef, useState } from 'react';

export interface AudioRecordingState {
  isRecording: boolean;
  error: string | null;
  audioStream: MediaStream | null;
}

export interface UseAudioRecordingReturn {
  isRecording: boolean;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  sendAudioData: (data: ArrayBuffer) => void;
}

export function useAudioRecording(
  onAudioData: (data: ArrayBuffer) => void,
  onError: (error: string) => void,
): UseAudioRecordingReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      
      audioStreamRef.current = stream;
      
      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });
      
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          // Convert blob to ArrayBuffer and send
          event.data.arrayBuffer().then((arrayBuffer) => {
            onAudioData(arrayBuffer);
          });
        }
      };
      
      mediaRecorder.onerror = (event) => {
        const error = 'Recording error occurred';
        setError(error);
        onError(error);
        stopRecording();
      };
      
      // Start recording with frequent data chunks
      mediaRecorder.start(100); // Send data every 100ms
      setIsRecording(true);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start recording';
      setError(errorMessage);
      onError(errorMessage);
    }
  }, [onAudioData, onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }
    
    setIsRecording(false);
    setError(null);
  }, [isRecording]);

  const sendAudioData = useCallback((data: ArrayBuffer) => {
    onAudioData(data);
  }, [onAudioData]);

  return {
    isRecording,
    error,
    startRecording,
    stopRecording,
    sendAudioData,
  };
} 