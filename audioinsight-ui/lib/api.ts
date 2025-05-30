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

export class AudioInsightAPI {
  private baseUrl: string;

  constructor() {
    this.baseUrl = typeof window !== 'undefined' 
      ? `${window.location.protocol}//${window.location.hostname}:8001`
      : '';
  }

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

  async checkSystemHealth(): Promise<{ status: string; [key: string]: any }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/system/health`);
      return response.json();
    } catch (error) {
      console.warn('Health check failed:', error);
      return { status: 'unknown' };
    }
  }

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
} 