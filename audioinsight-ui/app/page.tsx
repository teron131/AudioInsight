"use client"

import { TranscriptDisplay } from '@/components/transcript-display';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useAudioInsight } from '@/hooks/use-audioinsight';
import { cn } from '@/lib/utils';
import { Loader2, Mic, Square, Upload } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';

export default function AudioInsightPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef<boolean>(false);
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
    clearSession,
    systemHealth,
    diarizationEnabled,
    showLagInfo,
    showSpeakers,
    settingsLoading,
  } = useAudioInsight();

  // Handle scroll detection for transcript container
  const handleTranscriptScroll = useCallback(() => {
    if (transcriptScrollRef.current) {
      const container = transcriptScrollRef.current;
      const isAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 10;
      userScrolledRef.current = !isAtBottom;
    }
  }, []);

  // Auto-scroll when transcript content updates
  const handleTranscriptContentUpdate = useCallback(() => {
    if (transcriptScrollRef.current && !userScrolledRef.current) {
      const container = transcriptScrollRef.current;
      
      // Use requestAnimationFrame to ensure DOM is fully updated
      requestAnimationFrame(() => {
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          });
        }
      });
    }
  }, []);

  // Reset scroll state when session is cleared or new session starts
  const handleClearSessionWithScrollReset = useCallback(async () => {
    setIsClearing(true);
    try {
      await clearSession();
      userScrolledRef.current = false; // Reset scroll state
    } finally {
      setIsClearing(false);
    }
  }, [clearSession]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      userScrolledRef.current = false; // Reset scroll state for new file
      await uploadFile(file);
      // Reset the input so the same file can be uploaded again
      event.target.value = '';
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };



  const handleStartRecording = async () => {
    userScrolledRef.current = false; // Reset scroll state for new recording
    await startRecording();
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
                  onClick={handleStartRecording}
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

            </div>
          </div>
          
          <div>
            <Button 
              onClick={handleClearSessionWithScrollReset}
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

      {/* Main Content Grid - AC/BC Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left Column - 2 rows */}
        <div className="lg:col-span-1 space-y-5">
          {/* Row 1: Transcript (A) */}
          <Card className="h-[320px]">
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
            <CardContent 
              ref={transcriptScrollRef}
              onScroll={handleTranscriptScroll}
              className="h-[calc(100%-5rem)] overflow-y-auto"
            >
              <TranscriptDisplay 
                transcriptData={transcriptData} 
                onContentUpdate={handleTranscriptContentUpdate}
                showLagInfo={showLagInfo}
                showSpeakers={showSpeakers}
              />
            </CardContent>
          </Card>

          {/* Row 2: Key Points (B) */}
          <Card className="h-[320px]">
            <CardHeader>
              <CardTitle className="text-xl">Key Points</CardTitle>
            </CardHeader>
            <CardContent className="h-[calc(100%-5rem)] overflow-y-auto">
              <div className="space-y-3">
                <div className={cn(
                  "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
                  "hover:border-muted-foreground",
                  analysis?.key_points?.length ? "text-foreground" : "text-muted-foreground"
                )}>
                  {analysis?.key_points?.length ? (
                    <ul className="space-y-2">
                      {analysis.key_points.map((point: string, index: number) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-blue-600 font-medium">•</span>
                          <span>{point}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <ul>
                      <li>Key points will appear here as the conversation is analyzed...</li>
                    </ul>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Response Suggestions & Action Plan (C) */}
        <div className="lg:col-span-1">
          <Card className="h-[660px]">
            <CardHeader>
              <CardTitle className="text-xl">Response & Actions</CardTitle>
            </CardHeader>
            <CardContent className="h-[calc(100%-5rem)] overflow-y-auto">
              <div className="space-y-6">
                {/* Response Suggestions Section */}
                <div className="space-y-3">
                  <h3 className="text-base font-semibold text-foreground">Response Suggestions</h3>
                  <div className={cn(
                    "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
                    "hover:border-muted-foreground",
                    analysis?.response_suggestions?.length ? "text-foreground" : "text-muted-foreground"
                  )}>
                    {analysis?.response_suggestions?.length ? (
                      <ul className="space-y-1">
                        {analysis.response_suggestions.map((suggestion: string, index: number) => (
                          <li key={index} className="flex items-start gap-2">
                            <span className="text-green-600 font-medium">💬</span>
                            <span>{suggestion}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <ul>
                        <li>Response suggestions will appear here</li>
                      </ul>
                    )}
                  </div>
                </div>

                {/* Action Plan Section */}
                <div className="space-y-3">
                  <h3 className="text-base font-semibold text-foreground">Action Plan</h3>
                  <div className={cn(
                    "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[80px] transition-all",
                    "hover:border-muted-foreground",
                    analysis?.action_plan?.length ? "text-foreground" : "text-muted-foreground"
                  )}>
                    {analysis?.action_plan?.length ? (
                      <ul className="space-y-1">
                        {analysis.action_plan.map((action: string, index: number) => (
                          <li key={index} className="flex items-start gap-2">
                            <span className="text-purple-600 font-medium">📋</span>
                            <span>{action}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <ul>
                        <li>Action recommendations will appear here</li>
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
