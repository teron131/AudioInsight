"use client"

import { AnalysisPanel } from '@/components/analysis-panel';
import { ExportMenu } from '@/components/export-menu';
import { TranscriptDisplay } from '@/components/transcript-display';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useAudioInsight } from '@/hooks/use-audioinsight';
import { Loader2, Mic, Square, Upload } from 'lucide-react';
import { useRef, useState } from 'react';

export default function AudioInsightPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [diarizationEnabled, setDiarizationEnabled] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  
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
    setDiarizationEnabled: setDiarizationEnabledHook,
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

  const handleClearSession = async () => {
    setIsClearing(true);
    try {
      await clearSession();
    } finally {
      setIsClearing(false);
    }
  };

  const handleDiarizationChange = (enabled: boolean) => {
    setDiarizationEnabled(enabled);
    setDiarizationEnabledHook(enabled);
  };

  const hasTranscriptData = transcriptData && transcriptData.lines.length > 0;

  return (
    <div className="container mx-auto px-4 py-6">
      {/* Header Section */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            {/* Control Buttons */}
            <div className="flex items-center gap-4">
              {!isRecording ? (
                <Button 
                  onClick={startRecording}
                  disabled={isProcessingFile}
                  className="bg-blue-600 hover:bg-blue-700"
                  size="icon"
                >
                  <Mic className="w-4 h-4" />
                </Button>
              ) : (
                <Button 
                  onClick={stopRecording}
                  variant="destructive"
                  size="icon"
                >
                  <Square className="w-4 h-4" />
                </Button>
              )}

              <Button 
                onClick={handleUploadClick}
                variant="outline"
                disabled={isRecording || isProcessingFile}
                size="icon"
              >
                {isProcessingFile ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4" />
                )}
              </Button>

              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />

              <ExportMenu 
                onExport={exportTranscript}
                disabled={!hasTranscriptData}
                isIconOnly={true}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch 
                id="diarization-toggle"
                checked={diarizationEnabled}
                onCheckedChange={handleDiarizationChange}
                disabled={isRecording || isProcessingFile}
              />
              <Label htmlFor="diarization-toggle" className="text-base">Diarization</Label>
            </div>
          </div>
          
          <div>
            <Button 
              onClick={handleClearSession}
              variant="outline"
              size="sm"
              disabled={isRecording || isProcessingFile || isClearing}
              className="text-base"
            >
              {isClearing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Clearing...
                </>
              ) : (
                "Clear Session"
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Transcript Section */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between text-xl">
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
              <CardTitle className="text-xl">AI Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <AnalysisPanel analysis={analysis} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
