export interface TranscriptLine {
  speaker: number;
  text: string;
  beg?: string;
  end?: string;
  confidence?: number;
}

export interface WebSocketMessage {
  type?: string;
  lines?: TranscriptLine[];
  buffer_transcription?: string;
  buffer_diarization?: string;
  remaining_time_transcription?: number;
  remaining_time_diarization?: number;
  diarization_enabled?: boolean;
  status?: string;
  error?: string;
  message?: string;
  final?: boolean;
  
  // Analysis data
  summary?: string;
  key_points?: string[];
  keywords?: string[];
  summaries?: Array<{
    summary: string;
    key_points: string[];
    response_suggestions?: string[];
    action_plan?: string[];
  }>;
  response_suggestions?: string[];
  action_plan?: string[];
  analysis?: {
    summary?: string;
    key_points?: string[];
    keywords?: string[];
    response_suggestions?: string[];
    action_plan?: string[];
    summaries?: Array<{
      summary: string;
      key_points: string[];
      response_suggestions?: string[];
      action_plan?: string[];
    }>;
  };

  // Adaptive frequency monitoring
  llm_adaptive_frequency?: {
    current_hz: number;
    cooldown_seconds: number;
    avg_processing_time: number;
    recent_times: number[];
  };
  parser_adaptive_frequency?: {
    current_hz: number;
    cooldown_seconds: number;
    avg_processing_time: number;
    recent_times: number[];
  };
}

export interface TranscriptData {
  lines: TranscriptLine[];
  buffer_transcription?: string;
  buffer_diarization?: string;
  remaining_time_transcription?: number;
  remaining_time_diarization?: number;
  diarization_enabled?: boolean;
  timestamp: number;
  isFinalizing?: boolean;
  analysis?: {
    summary?: string;
    key_points?: string[];
    keywords?: string[];
    response_suggestions?: string[];
    action_plan?: string[];
    summaries?: Array<{
      summary: string;
      key_points: string[];
      response_suggestions?: string[];
      action_plan?: string[];
    }>;
  };
}

export class AudioInsightWebSocket {
  private websocket: WebSocket | null = null;
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private currentDiarizationSetting = false;

  constructor(
    private onMessage: (data: WebSocketMessage) => void,
    private onError: (error: string) => void,
    private onStatusChange: (connected: boolean) => void,
  ) {}

  connect(diarizationEnabled: boolean): Promise<void> {
    this.currentDiarizationSetting = diarizationEnabled;
    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.hostname}:8080/asr?diarization=${diarizationEnabled}`;
        
        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.onStatusChange(true);
          resolve();
        };

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.onMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            this.onError('Error parsing server response');
          }
        };

        this.websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.onError('Connection error');
          reject(new Error('WebSocket connection failed'));
        };

        this.websocket.onclose = (event) => {
          console.log('WebSocket closed');
          this.isConnected = false;
          this.onStatusChange(false);
          
          // Don't reconnect if backend is not ready (code 1013)
          if (event && event.code === 1013) {
            console.log('Backend not ready, will not attempt reconnection');
            this.onError('Backend is starting up, please wait...');
            return;
          }
          
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            setTimeout(() => {
              this.reconnectAttempts++;
              console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
              this.connect(this.currentDiarizationSetting).catch(() => {
                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                  this.onError('Failed to reconnect to server');
                }
              });
            }, this.reconnectDelay * this.reconnectAttempts);
          }
        };

        setTimeout(() => {
          if (!this.isConnected) {
            this.websocket?.close();
            reject(new Error('Connection timeout'));
          }
        }, 10000);

      } catch (error) {
        reject(error);
      }
    });
  }

  sendAudioData(audioData: ArrayBuffer): void {
    if (this.isConnected && this.websocket) {
      this.websocket.send(audioData);
    }
  }

  sendFileUploadMessage(data: { type: string; file_path: string; duration: number; filename: string }): void {
    if (this.isConnected && this.websocket) {
      this.websocket.send(JSON.stringify(data));
    }
  }

  disconnect(): void {
    this.reconnectAttempts = this.maxReconnectAttempts;
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.isConnected = false;
    this.onStatusChange(false);
  }

  getConnectionState(): boolean {
    return this.isConnected;
  }

  getCurrentDiarizationSetting(): boolean {
    return this.currentDiarizationSetting;
  }
} 