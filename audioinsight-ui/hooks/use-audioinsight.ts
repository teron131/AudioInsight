import { AudioInsightAPI, ExportRequest } from '@/lib/api';
import { AudioInsightWebSocket, TranscriptData, WebSocketMessage } from '@/lib/websocket';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAudioRecording } from './use-audio-recording';
import { useToast } from './use-toast';

export interface Analysis {
  summary?: string;
  key_points?: string[];
  keywords?: string[];
  summaries?: Array<{
    summary: string;
    key_points: string[];
  }>;
}

export interface UseAudioInsightReturn {
  // Connection state
  isConnected: boolean;
  
  // Recording state
  isRecording: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  
  // File upload
  isProcessingFile: boolean;
  uploadFile: (file: File) => Promise<void>;
  
  // Transcript data
  transcriptData: TranscriptData | null;
  
  // Analysis data
  analysis: Analysis | null;
  
  // Export functionality
  exportTranscript: (format: 'txt' | 'srt' | 'vtt' | 'json') => Promise<void>;
  
  // Utility functions
  clearSession: () => void;
  
  // System health
  systemHealth: string;
}

export function useAudioInsight(): UseAudioInsightReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessingFile, setIsProcessingFile] = useState(false);
  const [transcriptData, setTranscriptData] = useState<TranscriptData | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [systemHealth, setSystemHealth] = useState<string>('unknown');
  
  const { toast } = useToast();
  const websocketRef = useRef<AudioInsightWebSocket | null>(null);
  const apiRef = useRef<AudioInsightAPI>(new AudioInsightAPI());

  // Audio recording handlers
  const handleAudioData = useCallback((data: ArrayBuffer) => {
    if (websocketRef.current) {
      websocketRef.current.sendAudioData(data);
    }
  }, []);

  const handleRecordingError = useCallback((error: string) => {
    toast({
      title: "Recording Error",
      description: error,
      variant: "destructive",
    });
  }, [toast]);

  // Audio recording hook
  const {
    isRecording,
    startRecording: startAudioRecording,
    stopRecording: stopAudioRecording,
  } = useAudioRecording(handleAudioData, handleRecordingError);

  // WebSocket message handler
  const handleWebSocketMessage = useCallback((data: WebSocketMessage) => {
    try {
      // Check if this is a completion or finalization signal
      const isCompleting = data.type === 'ready_to_stop' || 
                           data.status === 'completed' || 
                           data.type === 'final' ||
                           data.type === 'completion' ||
                           (data.final === true);
      
      // Handle completion signals immediately for processing state
      if (data.type === 'ready_to_stop') {
        setIsProcessingFile(false);
      }
      
      // Handle transcript data
      if (data.type === 'transcription' || data.lines) {
        const {
          lines = [],
          buffer_transcription = "",
          buffer_diarization = "",
          remaining_time_transcription,
          remaining_time_diarization,
          diarization_enabled = false
        } = data;
        
        // Update transcript data normally first
        setTranscriptData({
          lines,
          buffer_transcription,
          buffer_diarization,
          remaining_time_transcription,
          remaining_time_diarization,
          diarization_enabled,
          timestamp: Date.now(),
          isFinalizing: isCompleting,
        });
      }
      
      // Handle completion signals - commit any remaining buffer text
      if (isCompleting) {
        // For ready_to_stop specifically, handle immediately
        if (data.type === 'ready_to_stop') {
          setTranscriptData(prev => {
            if (prev && (prev.buffer_transcription || prev.buffer_diarization)) {
              const finalBufferText = prev.buffer_diarization || prev.buffer_transcription || '';
              if (finalBufferText.trim() && prev.lines.length > 0) {
                // Append buffer text to the last existing line instead of creating a new one
                const updatedLines = [...prev.lines];
                const lastLineIndex = updatedLines.length - 1;
                updatedLines[lastLineIndex] = {
                  ...updatedLines[lastLineIndex],
                  text: (updatedLines[lastLineIndex].text || '') + finalBufferText.trim(),
                };
                
                return {
                  ...prev,
                  lines: updatedLines,
                  buffer_transcription: '',
                  buffer_diarization: '',
                  isFinalizing: false,
                  timestamp: Date.now(),
                };
              }
            }
            return prev ? { ...prev, isFinalizing: false, buffer_transcription: '', buffer_diarization: '' } : prev;
          });
          
          toast({
            title: "Success", 
            description: "Processing completed",
          });
        } else {
          // Use a small delay for other completion signals
          setTimeout(() => {
            setTranscriptData(prev => {
              if (prev && (prev.buffer_transcription || prev.buffer_diarization)) {
                const finalBufferText = prev.buffer_diarization || prev.buffer_transcription || '';
                if (finalBufferText.trim() && prev.lines.length > 0) {
                  // Append buffer text to the last existing line instead of creating a new one
                  const updatedLines = [...prev.lines];
                  const lastLineIndex = updatedLines.length - 1;
                  updatedLines[lastLineIndex] = {
                    ...updatedLines[lastLineIndex],
                    text: (updatedLines[lastLineIndex].text || '') + finalBufferText.trim(),
                  };
                  
                  return {
                    ...prev,
                    lines: updatedLines,
                    buffer_transcription: '',
                    buffer_diarization: '',
                    isFinalizing: false,
                    timestamp: Date.now(),
                  };
                }
              }
              return prev ? { ...prev, isFinalizing: false, buffer_transcription: '', buffer_diarization: '' } : prev;
            });
          }, 100); // Small delay to ensure proper order
          
          if (isRecording) {
            toast({
              title: "Success",
              description: "Live transcription completed",
            });
          }
        }
      }
      
      // Handle analysis results
      if (data.analysis || data.summary || data.key_points || data.keywords || data.summaries) {
        const analysisData: Analysis = {};
        
        // Handle summaries array from LLM
        if (data.summaries && data.summaries.length > 0) {
          const latestSummary = data.summaries[data.summaries.length - 1];
          analysisData.summary = latestSummary.summary;
          analysisData.key_points = latestSummary.key_points;
        }
        
        // Handle direct format
        if (data.summary) analysisData.summary = data.summary;
        if (data.key_points) analysisData.key_points = data.key_points;
        if (data.keywords) analysisData.keywords = data.keywords;
        if (data.summaries) analysisData.summaries = data.summaries;
        
        // Handle nested analysis object
        if (data.analysis) {
          Object.assign(analysisData, data.analysis);
        }
        
        setAnalysis(analysisData);
        
        // Store analysis in transcript data for export
        setTranscriptData(prev => prev ? {
          ...prev,
          analysis: analysisData
        } : null);
      }
      
      // Handle errors
      if (data.type === 'error' || data.error) {
        const errorMsg = data.error || data.message || 'Unknown error occurred';
        toast({
          title: "Error",
          description: errorMsg,
          variant: "destructive",
        });
        console.error('WebSocket error:', data);
      }

    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      toast({
        title: "Error",
        description: "Error processing response",
        variant: "destructive",
      });
    }
  }, [isRecording, isProcessingFile, toast]);

  // WebSocket error handler
  const handleWebSocketError = useCallback((error: string) => {
    toast({
      title: "Connection Error",
      description: error,
      variant: "destructive",
    });
  }, [toast]);

  // WebSocket status change handler
  const handleStatusChange = useCallback((connected: boolean) => {
    setIsConnected(connected);
    if (connected) {
      toast({
        title: "Connected",
        description: "Connected to AudioInsight server",
      });
    }
  }, [toast]);

  // Initialize WebSocket
  const initializeWebSocket = useCallback(async () => {
    if (!websocketRef.current) {
      websocketRef.current = new AudioInsightWebSocket(
        handleWebSocketMessage,
        handleWebSocketError,
        handleStatusChange
      );
    }
    
    if (!websocketRef.current.getConnectionState()) {
      await websocketRef.current.connect();
    }
  }, [handleWebSocketMessage, handleWebSocketError, handleStatusChange]);

  // Start recording function
  const startRecording = useCallback(async () => {
    try {
      await initializeWebSocket();
      clearSession();
      await startAudioRecording();
      toast({
        title: "Recording Started",
        description: "Speak now - live transcription active",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start recording';
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    }
  }, [initializeWebSocket, startAudioRecording, toast]);

  // Stop recording function
  const stopRecording = useCallback(() => {
    stopAudioRecording();
    toast({
      title: "Recording Stopped",
      description: "Processing final transcription...",
    });
  }, [stopAudioRecording, toast]);

  // File upload function
  const uploadFile = useCallback(async (file: File) => {
    try {
      setIsProcessingFile(true);
      clearSession();
      
      toast({
        title: "Uploading",
        description: `Processing ${file.name}...`,
      });

      // Upload file first
      const uploadResult = await apiRef.current.uploadFile(file);
      
      // Initialize WebSocket connection
      await initializeWebSocket();
      
      // Send file processing message via WebSocket
      const fileMessage = {
        type: 'file_upload',
        file_path: uploadResult.file_path,
        duration: uploadResult.duration,
        filename: uploadResult.filename,
      };

      if (websocketRef.current) {
        websocketRef.current.sendFileUploadMessage(fileMessage);
      }
      
      toast({
        title: "Processing",
        description: `Processing ${file.name} in real-time...`,
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'File upload failed';
      toast({
        title: "Upload Error",
        description: errorMessage,
        variant: "destructive",
      });
      setIsProcessingFile(false);
    }
  }, [initializeWebSocket, toast]);

  // Export transcript function
  const exportTranscript = useCallback(async (format: 'txt' | 'srt' | 'vtt' | 'json') => {
    if (!transcriptData || !transcriptData.lines || transcriptData.lines.length === 0) {
      toast({
        title: "No Data",
        description: "No transcript data available for export",
        variant: "destructive",
      });
      return;
    }

    try {
      const exportData: ExportRequest = {
        lines: transcriptData.lines,
        analysis: transcriptData.analysis,
      };

      const result = await apiRef.current.exportTranscript(exportData, format);
      
      if (result.status === 'success') {
        const mimeType = format === 'json' ? 'application/json' : 'text/plain';
        apiRef.current.downloadFile(result.content, result.filename, mimeType);
        
        toast({
          title: "Export Success",
          description: `Transcript exported as ${format.toUpperCase()}`,
        });
      } else {
        throw new Error(result.message || 'Export failed');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Export failed';
      toast({
        title: "Export Error",
        description: errorMessage,
        variant: "destructive",
      });
    }
  }, [transcriptData, toast]);

  // Clear session function
  const clearSession = useCallback(() => {
    setTranscriptData(null);
    setAnalysis(null);
  }, []);

  // Check system health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiRef.current.checkSystemHealth();
        setSystemHealth(health.status);
      } catch (error) {
        setSystemHealth('error');
      }
    };
    
    checkHealth();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (websocketRef.current) {
        websocketRef.current.disconnect();
      }
      if (isRecording) {
        stopAudioRecording();
      }
      // Cleanup session
      apiRef.current.cleanupSession().catch(console.warn);
    };
  }, [isRecording, stopAudioRecording]);

  return {
    isConnected,
    isRecording,
    startRecording,
    stopRecording,
    isProcessingFile,
    uploadFile,
    transcriptData,
    analysis,
    exportTranscript,
    clearSession,
    systemHealth,
  };
} 