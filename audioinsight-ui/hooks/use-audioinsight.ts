import { AudioInsightAPI, ExportRequest, ProcessingParameters } from '@/lib/api';
import { AudioInsightWebSocket, TranscriptData, WebSocketMessage } from '@/lib/websocket';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAudioRecording } from './use-audio-recording';
import { useToast } from './use-toast';

export interface AnalysisData {
  key_points?: string[];
  keywords?: string[];
  response_suggestions?: string[];
  action_plan?: string[];
  analyses?: Array<{
    key_points: string[];
    response_suggestions?: string[];
    action_plan?: string[];
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
  analysis: AnalysisData | null;
  
  // Export functionality
  exportTranscript: (format: 'txt' | 'srt' | 'vtt' | 'json') => Promise<void>;
  
  // Utility functions
  clearSession: () => void;
  
  // System health
  systemHealth: string;

  // Settings from processing parameters
  diarizationEnabled: boolean;
  showLagInfo: boolean;
  
  // Settings loading state
  settingsLoading: boolean;
}

export function useAudioInsight(): UseAudioInsightReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessingFile, setIsProcessingFile] = useState(false);
  const [transcriptData, setTranscriptData] = useState<TranscriptData | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [systemHealth, setSystemHealth] = useState<string>('unknown');
  const [processingParams, setProcessingParams] = useState<ProcessingParameters | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(true);
  
  const { toast } = useToast();
  const websocketRef = useRef<AudioInsightWebSocket | null>(null);
  const apiRef = useRef<AudioInsightAPI>(new AudioInsightAPI());

  // Derived values from processing parameters
  const diarizationEnabled = processingParams?.diarization ?? false;
  const showLagInfo = processingParams?.show_lag_info ?? false;

  // Load processing parameters
  const loadProcessingParameters = useCallback(async () => {
    try {
      setSettingsLoading(true);
      const params = await apiRef.current.getProcessingParameters();
      setProcessingParams(params);
    } catch (error) {
      console.error('Failed to load processing parameters:', error);
    } finally {
      setSettingsLoading(false);
    }
      }, []);

  // Load processing parameters on mount
  useEffect(() => {
    loadProcessingParameters();
  }, [loadProcessingParameters]);

  const checkHealth = useCallback(async () => {
    try {
      const health = await apiRef.current.checkSystemHealth();
      setSystemHealth(health.status);
    } catch (error) {
      setSystemHealth('error');
    }
  }, []);

  const handleAudioData = useCallback((data: ArrayBuffer) => {
    if (websocketRef.current && websocketRef.current.getConnectionState()) {
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

  const audioCtx = useAudioRecording(handleAudioData, handleRecordingError);
  
  const handleWebSocketMessage = useCallback((data: WebSocketMessage) => {
    try {
      const isCompleting = data.type === 'ready_to_stop' || 
                           data.status === 'completed' || 
                           data.type === 'final' ||
                           data.type === 'completion' ||
                           (data.final === true);
      
      if (data.type === 'ready_to_stop') {
        setIsProcessingFile(false);
      }
      
      if (data.type === 'transcription' || data.lines) {
        const {
          lines = [],
          buffer_transcription = "",
          buffer_diarization = "",
          remaining_time_transcription,
          remaining_time_diarization,
          diarization_enabled = false
        } = data;
        
        setTranscriptData((prevData) => ({
          ...(prevData || {}), // Ensure prevData is not null
          lines,
          buffer_transcription,
          buffer_diarization,
          remaining_time_transcription,
          remaining_time_diarization,
          diarization_enabled, // This reflects backend status, not the toggle directly
          timestamp: Date.now(),
          isFinalizing: isCompleting,
          analysis: prevData?.analysis, // Preserve existing analysis
        }));
      }
      
      if (isCompleting) {
        if (data.type === 'ready_to_stop') {
          setTranscriptData(prev => {
            if (prev && (prev.buffer_transcription || prev.buffer_diarization)) {
              const finalBufferText = prev.buffer_diarization || prev.buffer_transcription || '';
              if (finalBufferText.trim() && prev.lines.length > 0) {
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
          toast({ title: "Success", description: "Processing completed" });
        } else {
          setTimeout(() => {
            setTranscriptData(prev => {
              if (prev && (prev.buffer_transcription || prev.buffer_diarization)) {
                const finalBufferText = prev.buffer_diarization || prev.buffer_transcription || '';
                if (finalBufferText.trim() && prev.lines.length > 0) {
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
          }, 100);
          if (audioCtx.isRecording) { // Check audioCtx.isRecording
            toast({ title: "Success", description: "Live transcription completed" });
          }
        }
      }
      
      if (data.key_points || data.keywords || data.analyses || data.response_suggestions || data.action_plan) {
        const currentAnalysisData: AnalysisData = {};
        if (data.analyses && data.analyses.length > 0) {
          const latestAnalysis = data.analyses[data.analyses.length - 1];
          currentAnalysisData.key_points = latestAnalysis.key_points;
          currentAnalysisData.response_suggestions = latestAnalysis.response_suggestions;
          currentAnalysisData.action_plan = latestAnalysis.action_plan;
        }
        if (data.key_points) currentAnalysisData.key_points = data.key_points;
        if (data.keywords) currentAnalysisData.keywords = data.keywords;
        if (data.response_suggestions) currentAnalysisData.response_suggestions = data.response_suggestions;
        if (data.action_plan) currentAnalysisData.action_plan = data.action_plan;
        if (data.analyses) currentAnalysisData.analyses = data.analyses;
        if (data.analysis) {
          Object.assign(currentAnalysisData, data.analysis);
        }
        setAnalysis(currentAnalysisData);
        setTranscriptData(prev => prev ? { ...prev, analysis: currentAnalysisData } : null);
      }
      
      if (data.type === 'error' || data.error) {
        const errorMsg = data.error || data.message || 'Unknown error occurred';
        toast({ title: "Error", description: errorMsg, variant: "destructive" });
        console.error('WebSocket error:', data);
      }

    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      toast({ title: "Error", description: "Error processing response", variant: "destructive" });
    }
  }, [audioCtx.isRecording, toast]);

  const handleWebSocketError = useCallback((error: string) => {
    toast({ title: "Connection Error", description: error, variant: "destructive" });
  }, [toast]);

  const handleStatusChange = useCallback((connected: boolean) => {
    setIsConnected(connected);
    if (connected) {
      toast({ title: "Connected", description: "Connected to AudioInsight server" });
    } else {
      // toast({ title: "Disconnected", description: "Disconnected from AudioInsight server", variant: "default" });
    }
  }, [toast]);

  const initializeWebSocket = useCallback(async (diarizationSetting: boolean) => {
    // First, check if backend is ready
    try {
      const baseUrl = typeof window !== 'undefined' 
        ? `${window.location.protocol}//${window.location.hostname}:8080`
        : '';
      const healthResponse = await fetch(`${baseUrl}/health`);
      const healthData = await healthResponse.json();
      
      if (!healthData.backend_ready) {
        console.log("Backend not ready, waiting...");
        // Wait up to 30 seconds for backend to be ready
        let attempts = 0;
        const maxAttempts = 30; // 30 seconds with 1 second intervals
        let currentHealthData = healthData;
        
        while (!currentHealthData.backend_ready && attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          const retryResponse = await fetch(`${baseUrl}/health`);
          currentHealthData = await retryResponse.json();
          if (currentHealthData.backend_ready) {
            console.log("Backend is now ready!");
            break;
          }
          attempts++;
        }
        
        if (attempts >= maxAttempts) {
          throw new Error("Backend failed to become ready within 30 seconds");
        }
      }
    } catch (error) {
      console.warn("Could not check backend readiness, proceeding anyway:", error);
      // Continue with connection attempt even if health check fails
    }

    if (websocketRef.current && websocketRef.current.getConnectionState()) {
      // Check if diarization setting is different from the one used for current connection
      // This requires AudioInsightWebSocket to expose its currentDiarizationSetting or a way to check it.
      // For now, we assume if it's connected, we might need to reconnect if diarization changes.
      // A more robust way would be for websocket.connect to handle this internally or expose the setting.
      // currentDiarizationSetting is private, so we can't directly access it here.
      // We will rely on the setDiarizationEnabled to disconnect first if settings change.
    }

    if (!websocketRef.current) {
      websocketRef.current = new AudioInsightWebSocket(
        handleWebSocketMessage,
        handleWebSocketError,
        handleStatusChange
      );
    }
    
    // Disconnect if connected and diarization setting will change
    // This is a simplified check; ideally, AudioInsightWebSocket would expose its current setting
    if(websocketRef.current.getConnectionState() && websocketRef.current.getCurrentDiarizationSetting() !== diarizationSetting){
        websocketRef.current.disconnect();
    }


    if (!websocketRef.current.getConnectionState()) {
        try {
            await websocketRef.current.connect(diarizationSetting);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'WebSocket connection failed';
            toast({ title: "Connection Error", description: errorMessage, variant: "destructive" });
            throw error; 
        }
    }
  }, [handleWebSocketMessage, handleWebSocketError, handleStatusChange, toast]);
  


  const startRecording = useCallback(async () => {
    try {
      await initializeWebSocket(diarizationEnabled);
      await audioCtx.startRecording();
      toast({ title: "Recording Started", description: "Speak now - live transcription active" });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start recording';
      toast({ title: "Error", description: errorMessage, variant: "destructive" });
    }
  }, [initializeWebSocket, audioCtx, toast, diarizationEnabled]);

  const stopRecording = useCallback(() => {
    audioCtx.stopRecording();
    toast({ title: "Recording Stopped", description: "Finalizing transcription..." });
  }, [audioCtx, toast]);

  const uploadFile = useCallback(async (file: File) => {
    try {
      await initializeWebSocket(diarizationEnabled); 
      
      setIsProcessingFile(true);
      setTranscriptData(null); 
      setAnalysis(null);    

      const uploadResult = await apiRef.current.uploadFile(file);
      
      if (websocketRef.current && websocketRef.current.getConnectionState()) {
        websocketRef.current.sendFileUploadMessage({
          type: 'file_upload',
          file_path: uploadResult.file_path,
          duration: uploadResult.duration,
          filename: uploadResult.filename,
        });
        toast({ title: "Processing", description: `Processing ${file.name}...` });
      } else {
        throw new Error("WebSocket not connected for file processing.");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'File upload failed';
      toast({ title: "Upload Error", description: errorMessage, variant: "destructive" });
      setIsProcessingFile(false);
    }
  }, [initializeWebSocket, toast, diarizationEnabled]);

  const exportTranscript = useCallback(async (format: 'txt' | 'srt' | 'vtt' | 'json') => {
    if (!transcriptData || !transcriptData.lines || transcriptData.lines.length === 0) {
      toast({ title: "No Data", description: "No transcript data available for export", variant: "destructive" });
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
        toast({ title: "Export Success", description: `Transcript exported as ${format.toUpperCase()}` });
      } else {
        throw new Error(result.message || 'Export failed');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Export failed';
      toast({ title: "Export Error", description: errorMessage, variant: "destructive" });
    }
  }, [transcriptData, toast]);

  const clearSession = useCallback(async () => {
    try {
      // Disconnect WebSocket first
      if (websocketRef.current) {
        websocketRef.current.disconnect();
      }
      
      // Stop any ongoing recording
      if (audioCtx.isRecording) {
        audioCtx.stopRecording();
      }
      
      // Call backend to reset everything
      await apiRef.current.cleanupSession();
      
      // Clear all frontend state
      setTranscriptData(null);
      setAnalysis(null);
      setIsProcessingFile(false);
      
      // Wait a moment for backend cleanup to complete
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Reconnect WebSocket with current diarization setting
      await initializeWebSocket(diarizationEnabled);
      
      toast({ 
        title: "Session Reset", 
        description: "All data cleared and system reset to fresh state." 
      });
      
    } catch (error) {
      console.error("Error during session reset:", error);
      toast({ 
        title: "Reset Error", 
        description: "Failed to completely reset session. You may need to refresh the page.", 
        variant: "destructive" 
      });
    }
  }, [toast, audioCtx, diarizationEnabled, initializeWebSocket]);
  
  useEffect(() => {
    checkHealth();
    const intervalId = setInterval(checkHealth, 30000); 
    return () => clearInterval(intervalId);
  }, [checkHealth]);

  useEffect(() => {
    initializeWebSocket(diarizationEnabled).catch(error => {
        console.error("Initial WebSocket connection failed:", error);
    });
    
    return () => {
      if (websocketRef.current) {
        websocketRef.current.disconnect();
      }
      if (audioCtx.isRecording) { 
        audioCtx.stopRecording();
      }
      apiRef.current.cleanupSession().catch(console.warn);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps 
  }, [diarizationEnabled]); // Re-run if diarizationEnabled changes from the parent component (though it shouldn't directly)
                           // The primary way diarization change triggers re-connect is via setDiarizationEnabled -> initializeWebSocket

  return {
    isConnected,
    isRecording: audioCtx.isRecording,
    startRecording,
    stopRecording,
    isProcessingFile,
    uploadFile,
    transcriptData,
    analysis,
    exportTranscript,
    clearSession,
    systemHealth,
    diarizationEnabled,
    showLagInfo,
    settingsLoading,
  };
} 