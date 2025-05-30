"use client"

import { AnalysisPanel } from '@/components/analysis-panel';
import { ExportMenu } from '@/components/export-menu';
import { TranscriptDisplay } from '@/components/transcript-display';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { WaveformVisualization } from '@/components/waveform-visualization';
import { useAudioInsight } from '@/hooks/use-audioinsight';
import { Loader2, Mic, Square, Upload, Wifi, WifiOff } from 'lucide-react';
import { useRef } from 'react';

export default function AudioInsightPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const {
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
  } = useAudioInsight();

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await uploadFile(file);
      // Reset the input so the same file can be uploaded again
      event.target.value = '';
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const hasTranscriptData = transcriptData && transcriptData.lines.length > 0;

  return (
    <div className="container mx-auto px-6 py-8 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">
              Audio Transcription & Analysis
            </h1>
            <p className="text-muted-foreground">
              Real-time speech-to-text with AI-powered insights
            </p>
          </div>
          
          {/* Connection Status */}
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Badge variant="default" className="bg-green-600">
                <Wifi className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge variant="destructive">
                <WifiOff className="w-3 h-3 mr-1" />
                Disconnected
              </Badge>
            )}
            <Badge variant="outline">
              System: {systemHealth}
            </Badge>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center gap-4">
          {!isRecording ? (
            <Button 
              onClick={startRecording}
              disabled={isProcessingFile}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Mic className="w-4 h-4 mr-2" />
              Start Recording
            </Button>
          ) : (
            <Button 
              onClick={stopRecording}
              variant="destructive"
            >
              <Square className="w-4 h-4 mr-2" />
              Stop Recording
            </Button>
          )}

          <Button 
            onClick={handleUploadClick}
            variant="outline"
            disabled={isRecording || isProcessingFile}
          >
            {isProcessingFile ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Upload className="w-4 h-4 mr-2" />
            )}
            Upload Audio File
          </Button>

          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          <Button 
            onClick={clearSession}
            variant="outline"
            disabled={isRecording || isProcessingFile}
          >
            Clear Session
          </Button>

          <ExportMenu 
            onExport={exportTranscript}
            disabled={!hasTranscriptData}
          />
        </div>
      </div>

      {/* Waveform Visualization */}
      <WaveformVisualization 
        isActive={isRecording || isProcessingFile}
        className="mb-6"
      />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Transcript Section */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Live Transcript</span>
                {(isRecording || isProcessingFile) && (
                  <Badge variant="secondary">
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                    {isRecording ? 'Recording' : 'Processing'}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <TranscriptDisplay transcriptData={transcriptData} />
            </CardContent>
          </Card>
        </div>

        {/* Analysis Section */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>AI Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <AnalysisPanel analysis={analysis} />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Status Information */}
      {transcriptData && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {transcriptData.lines.length}
                </div>
                <div className="text-sm text-muted-foreground">
                  Transcript Segments
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {transcriptData.diarization_enabled ? 'Enabled' : 'Disabled'}
                </div>
                <div className="text-sm text-muted-foreground">
                  Speaker Diarization
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {analysis?.keywords?.length || 0}
                </div>
                <div className="text-sm text-muted-foreground">
                  Keywords Extracted
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
