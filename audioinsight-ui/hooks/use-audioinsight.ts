import { AudioInsightAPI, ProcessingParameters } from '@/lib/api';
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
  isConnecting: boolean;
  
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
  
  // Utility functions
  clearSession: () => void;
  
  // System health
  systemHealth: string;

  // Settings from processing parameters
  diarizationEnabled: boolean;
  showLagInfo: boolean;
  showSpeakers: boolean;
  
  // Settings loading state
  settingsLoading: boolean;
}

export function useAudioInsight(): UseAudioInsightReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(true); // Start as connecting
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
  const showSpeakers = processingParams?.show_speakers ?? false;

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
          diarization_enabled = false,
          transcript_parser
        } = data;
        
        setTranscriptData((prevData) => ({
          ...(prevData || {}), // Ensure prevData is not null
          lines,
          buffer_transcription,
          buffer_diarization,
          remaining_time_transcription,
          remaining_time_diarization,
          diarization_enabled, // This reflects backend status, not the toggle directly
          transcript_parser, // CHINESE PARSER FIX: Include parsed transcript data
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
    setIsConnecting(false); // Connection attempt completed
    if (connected) {
      toast({ title: "Connected", description: "Connected to AudioInsight server" });
    } else {
      // toast({ title: "Disconnected", description: "Disconnected from AudioInsight server", variant: "default" });
    }
  }, [toast]);

  const initializeWebSocket = useCallback(async (diarizationSetting: boolean) => {
    setIsConnecting(true); // Start connecting
    
    // First, check if backend is ready with improved error handling
    try {
      const baseUrl = typeof window !== 'undefined' 
        ? `${window.location.protocol}//${window.location.hostname}:8080`
        : '';
      
      console.log("Checking backend readiness...");
      
      // Use AbortController for proper timeout support
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const healthResponse = await fetch(`${baseUrl}/health`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (!healthResponse.ok) {
        throw new Error(`Backend health check failed: ${healthResponse.status} ${healthResponse.statusText}`);
      }
      
      const healthData = await healthResponse.json();
      console.log("Backend health status:", healthData);
      
      if (!healthData.backend_ready) {
        console.log("Backend not ready, waiting for initialization...");
        // Wait up to 15 seconds for backend to be ready (reduced from 30s)
        let attempts = 0;
        const maxAttempts = 15; // 15 seconds with 1 second intervals
        let currentHealthData = healthData;
        
        while (!currentHealthData.backend_ready && attempts < maxAttempts) {
          console.log(`Waiting for backend readiness... (${attempts + 1}/${maxAttempts})`);
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          try {
            const retryController = new AbortController();
            const retryTimeoutId = setTimeout(() => retryController.abort(), 3000); // 3s timeout for retries
            
            const retryResponse = await fetch(`${baseUrl}/health`, {
              signal: retryController.signal
            });
            clearTimeout(retryTimeoutId);
            
            if (retryResponse.ok) {
              currentHealthData = await retryResponse.json();
              if (currentHealthData.backend_ready) {
                console.log("Backend is now ready!");
                break;
              }
            }
          } catch (retryError) {
            console.warn(`Health check retry ${attempts + 1} failed:`, retryError);
          }
          
          attempts++;
        }
        
        if (attempts >= maxAttempts) {
          throw new Error("Backend failed to become ready within 15 seconds");
        }
      } else {
        console.log("Backend is ready!");
      }
    } catch (error) {
      console.warn("Backend readiness check failed, proceeding with connection attempt:", error);
      // Continue with connection attempt even if health check fails
      // This provides fallback behavior in case health endpoint has issues
    }

    // WebSocket connection logic
    if (websocketRef.current && websocketRef.current.getConnectionState()) {
      // Check if diarization setting is different from the one used for current connection
      console.log("WebSocket already connected, checking diarization setting compatibility...");
    }

    if (!websocketRef.current) {
      console.log("Creating new AudioInsightWebSocket instance...");
      websocketRef.current = new AudioInsightWebSocket(
        handleWebSocketMessage,
        handleWebSocketError,
        handleStatusChange
      );
    }
    
    // Disconnect if connected and diarization setting will change
    if(websocketRef.current.getConnectionState() && websocketRef.current.getCurrentDiarizationSetting() !== diarizationSetting){
        console.log("Diarization setting changed, disconnecting existing connection...");
        websocketRef.current.disconnect();
    }

    if (!websocketRef.current.getConnectionState()) {
        try {
            console.log(`Connecting WebSocket with diarization=${diarizationSetting}...`);
            await websocketRef.current.connect(diarizationSetting);
            console.log("WebSocket connected successfully!");
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'WebSocket connection failed';
            console.error("WebSocket connection error:", errorMessage);
            setIsConnecting(false); // Connection failed
            toast({ title: "Connection Error", description: errorMessage, variant: "destructive" });
            throw error; 
        }
    } else {
      console.log("WebSocket already connected, skipping connection attempt");
      setIsConnecting(false); // Already connected
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
    // Simple initial connection - backend is guaranteed to be ready before frontend starts
    const connectToReadyBackend = async () => {
      console.log("🔌 Connecting to ready backend...");
      
      try {
        await initializeWebSocket(diarizationEnabled);
        console.log("✅ Connected to backend successfully!");
      } catch (error) {
        console.error("❌ Failed to connect to ready backend:", error);
        // Show user-friendly message since backend should be ready
        toast({
          title: "Connection Failed",
          description: "Failed to connect to backend. Please try refreshing the page.",
          variant: "destructive"
        });
      }
    };
    
    console.log("🏁 Frontend started - connecting to ready backend");
    connectToReadyBackend();
    
    return () => {
      console.log("🧹 useEffect cleanup running");
      if (websocketRef.current) {
        websocketRef.current.disconnect();
      }
      if (audioCtx.isRecording) { 
        audioCtx.stopRecording();
      }
      apiRef.current.cleanupSession().catch(console.warn);
    };
  }, [diarizationEnabled, initializeWebSocket, toast]);

  return {
    isConnected,
    isConnecting,
    isRecording: audioCtx.isRecording,
    startRecording,
    stopRecording,
    isProcessingFile,
    uploadFile,
    transcriptData,
    analysis,
    clearSession,
    systemHealth,
    diarizationEnabled,
    showLagInfo,
    showSpeakers,
    settingsLoading,
  };
} 