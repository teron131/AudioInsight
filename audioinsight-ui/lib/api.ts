export interface FileUploadResponse {
  status: string;
  filename: string;
  file_path: string;
  duration: number;
  message: string;
}

export interface ExportRequest {
  lines: Array<{
    speaker: number;
    text: string;
    beg?: string;
    end?: string;
  }>;
  analysis?: {
    summary?: string;
    key_points?: string[];
    keywords?: string[];
  };
}

export interface ExportResponse {
  status: string;
  format: string;
  content: string;
  filename: string;
  message?: string;
}

export interface ModelStatus {
  asr: {
    loaded: boolean;
    model_name: string;
    backend: string;
    language: string;
    ready: boolean;
  };
  diarization: {
    loaded: boolean;
    enabled: boolean;
    ready: boolean;
  };
  llm: {
    fast_model: string;
    base_model: string;
    inference_enabled: boolean;
  };
}

export interface ProcessingParameters {
  // Server Configuration
  host: string;
  port: number;
  
  // Model Configuration  
  model: string;
  backend: string;
  language: string;
  task: string;
  model_cache_dir?: string;
  model_dir?: string;
  
  // Processing Configuration
  min_chunk_size: number;
  buffer_trimming: string;
  buffer_trimming_sec: number;
  vac_chunk_size: number;
  warmup_file?: string;
  
  // Feature Configuration (boolean toggles)
  transcription: boolean;
  diarization: boolean;
  vad_enabled: boolean;
  vac_enabled: boolean;
  confidence_validation: boolean;
  llm_inference: boolean;
  
  // LLM Configuration
  fast_llm: string;
  base_llm: string;
  llm_summary_interval: number;
  llm_new_text_trigger: number;
  parser_trigger_interval: number;
  parser_output_tokens: number;
}

export interface ConfigurationPreset {
  name: string;
  description: string;
  config: Record<string, any>;
}

export interface UploadedFile {
  filename: string;
  path: string;
  size: number;
  created: number;
  modified: number;
}

export interface LLMStatus {
  display_parser: {
    enabled: boolean;
    model: string;
    stats: any;
  };
  inference: {
    enabled: boolean;
    fast_model: string;
    base_model: string;
  };
}

export interface TranscriptParserStatus {
  enabled: boolean;
  stats: any;
  config: {
    model_id: string;
    output_tokens: number;
  };
  total_parsed: number;
  last_parsed_available: boolean;
}

export interface ParsedTranscript {
  original_text: string;
  parsed_text: string;
  segments: Array<{
    text: string;
    position: number;
    character_start: number;
    character_end: number;
    speaker?: number;
    timestamp_start?: number;
    timestamp_end?: number;
  }>;
  timestamps: Record<string, number>;
  speakers: Array<Record<string, any>>;
  parsing_time: number;
}

export interface BatchJob {
  batch_id: string;
  files: string[];
  total_files: number;
  status: string;
  created_at: number;
  config: Record<string, any>;
}

export interface BatchStatus {
  batch_id: string;
  status: string;
  processed_files: number;
  total_files: number;
  failed_files: number;
  progress_percent: number;
  started_at: number;
  completed_at: number;
  results: any[];
}

export interface AudioAnalysis {
  file_info: {
    filename: string;
    duration: number;
    sample_rate: number;
    channels: number;
  };
  quality_metrics: {
    average_rms: number;
    average_zcr: number;
    average_spectral_centroid: number;
    estimated_snr: number;
    noise_estimate: number;
  };
  recommendations: string[];
  overall_quality: 'good' | 'fair' | 'poor';
}

export class AudioInsightAPI {
  private baseUrl: string;

  constructor() {
    this.baseUrl = typeof window !== 'undefined' 
      ? `${window.location.protocol}//${window.location.hostname}:8001`
      : '';
  }

  // =============================================================================
  // Basic File Operations
  // =============================================================================

  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload-file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  async exportTranscript(data: ExportRequest, format: 'txt' | 'srt' | 'vtt' | 'json' = 'txt'): Promise<ExportResponse> {
    const response = await fetch(`${this.baseUrl}/api/export/transcript?format=${format}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Export failed: ${response.statusText}`);
    }

    return response.json();
  }

  // =============================================================================
  // Model Management APIs
  // =============================================================================

  async getModelsStatus(): Promise<ModelStatus> {
    const response = await fetch(`${this.baseUrl}/api/models/status`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get models status');
    }
    
    return data.models;
  }

  async reloadModels(modelType: 'all' | 'asr' | 'diarization' = 'all'): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/models/reload?model_type=${modelType}`, {
      method: 'POST',
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to reload models');
    }
    
    return data.reloaded;
  }

  async unloadModels(modelType: 'all' | 'asr' | 'diarization' = 'all'): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/models/unload?model_type=${modelType}`, {
      method: 'POST',
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to unload models');
    }
    
    return data.unloaded;
  }

  // =============================================================================
  // Processing Control APIs
  // =============================================================================

  async getProcessingParameters(): Promise<ProcessingParameters> {
    const response = await fetch(`${this.baseUrl}/api/processing/parameters`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get processing parameters');
    }
    
    return data.parameters;
  }

  async updateProcessingParameters(parameters: Partial<ProcessingParameters>): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/processing/parameters`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(parameters),
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to update processing parameters');
    }
    
    return data.updated_parameters;
  }

  // =============================================================================
  // Configuration Presets APIs
  // =============================================================================

  async getConfigurationPresets(): Promise<Record<string, ConfigurationPreset>> {
    const response = await fetch(`${this.baseUrl}/api/presets`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get configuration presets');
    }
    
    return data.presets;
  }

  async applyConfigurationPreset(presetName: string): Promise<{ preset: string; applied_config: Record<string, any> }> {
    const response = await fetch(`${this.baseUrl}/api/presets/${presetName}/apply`, {
      method: 'POST',
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to apply configuration preset');
    }
    
    return { preset: data.preset, applied_config: data.applied_config };
  }

  // =============================================================================
  // File Management APIs
  // =============================================================================

  async getUploadedFiles(): Promise<UploadedFile[]> {
    const response = await fetch(`${this.baseUrl}/api/files/uploaded`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get uploaded files');
    }
    
    return data.files;
  }

  async deleteUploadedFile(filePath: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/files/${encodeURIComponent(filePath)}`, {
      method: 'DELETE',
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to delete file');
    }
  }

  async cleanupOldFiles(maxAgeHours: number = 24): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/files/cleanup?max_age_hours=${maxAgeHours}`, {
      method: 'POST',
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to cleanup old files');
    }
    
    return data.deleted_files;
  }

  // =============================================================================
  // LLM Management APIs
  // =============================================================================

  async getLLMStatus(): Promise<LLMStatus> {
    const response = await fetch(`${this.baseUrl}/api/llm/status`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get LLM status');
    }
    
    return data.llm_status;
  }

  async testLLMConnection(modelId?: string): Promise<{ model: string; response: string; response_time: number }> {
    const url = modelId 
      ? `${this.baseUrl}/api/llm/test?model_id=${encodeURIComponent(modelId)}`
      : `${this.baseUrl}/api/llm/test`;
      
    const response = await fetch(url, { method: 'POST' });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'LLM connection test failed');
    }
    
    return { model: data.model, response: data.response, response_time: data.response_time };
  }

  // =============================================================================
  // Batch Processing APIs
  // =============================================================================

  async startBatchProcessing(filePaths: string[], processingConfig?: Record<string, any>): Promise<BatchJob> {
    const response = await fetch(`${this.baseUrl}/api/batch/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_paths: filePaths,
        processing_config: processingConfig,
      }),
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to start batch processing');
    }
    
    return data.batch;
  }

  async getBatchStatus(batchId: string): Promise<BatchStatus> {
    const response = await fetch(`${this.baseUrl}/api/batch/${batchId}/status`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get batch status');
    }
    
    return data.batch;
  }

  // =============================================================================
  // Audio Quality Analysis APIs
  // =============================================================================

  async analyzeAudioQuality(filePath: string): Promise<AudioAnalysis> {
    const response = await fetch(`${this.baseUrl}/api/audio/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file_path: filePath }),
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to analyze audio quality');
    }
    
    return data.analysis;
  }

  // =============================================================================
  // Session Management APIs
  // =============================================================================

  async cleanupSession(): Promise<void> {
    try {
      // Try new API first
      await fetch(`${this.baseUrl}/api/sessions/reset`, { method: 'POST' });
    } catch (error) {
      // Fallback to legacy API
      try {
        await fetch(`${this.baseUrl}/cleanup-session`, { method: 'POST' });
      } catch (legacyError) {
        console.warn('Session cleanup failed:', legacyError);
      }
    }
  }

  async checkSystemHealth(): Promise<{ status: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      
      if (!response.ok) {
        return { status: 'error' };
      }
      
      const data = await response.json();
      return { status: data.status || 'ok' };
    } catch (error) {
      console.warn('Health check failed:', error);
      return { status: 'error' };
    }
  }

  // =============================================================================
  // Transcript Parser APIs
  // =============================================================================

  async getTranscriptParserStatus(): Promise<TranscriptParserStatus> {
    const response = await fetch(`${this.baseUrl}/api/transcript-parser/status`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get transcript parser status');
    }
    
    return {
      enabled: data.enabled,
      stats: data.stats,
      config: data.config,
      total_parsed: data.total_parsed,
      last_parsed_available: data.last_parsed_available
    };
  }

  async enableTranscriptParser(enabled: boolean = true): Promise<{ enabled: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/transcript-parser/enable?enabled=${enabled}`, {
      method: 'POST',
    });
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to toggle transcript parser');
    }
    
    return { enabled: data.enabled, message: data.message };
  }

  async getParsedTranscripts(limit: number = 10): Promise<{ transcripts: ParsedTranscript[]; total_count: number; returned_count: number }> {
    const response = await fetch(`${this.baseUrl}/api/transcript-parser/transcripts?limit=${limit}`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get parsed transcripts');
    }
    
    return {
      transcripts: data.transcripts,
      total_count: data.total_count,
      returned_count: data.returned_count
    };
  }

  async getLatestParsedTranscript(): Promise<ParsedTranscript | null> {
    const response = await fetch(`${this.baseUrl}/api/transcript-parser/latest`);
    const data = await response.json();
    
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'Failed to get latest parsed transcript');
    }
    
    return data.transcript;
  }

  // =============================================================================
  // Utility Methods
  // =============================================================================

  downloadFile(content: string, filename: string, mimeType: string = 'text/plain'): void {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  formatDuration(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }

  formatTimestamp(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleString();
  }
} 