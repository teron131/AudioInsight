"use client";

import { cn } from '@/lib/utils';
import { TranscriptData } from '@/lib/websocket';
import { useLayoutEffect, useRef } from 'react';

interface TranscriptDisplayProps {
  transcriptData: TranscriptData | null;
  className?: string;
  onContentUpdate?: () => void; // Callback to trigger parent scroll
  showLagInfo?: boolean; // Whether to show lag information
  showSpeakers?: boolean; // Whether to show speaker labels
}

const speakerColors = {
  0: "border-l-blue-600",
  1: "border-l-green-600", 
  2: "border-l-purple-600",
  3: "border-l-orange-600",
  "-1": "border-l-blue-600",
  "-2": "border-l-gray-400",
} as const;

const getSpeakerColor = (speaker: number): string => {
  if (speaker === -2) return speakerColors["-2"];
  if (speaker === -1) return speakerColors["-1"];
  if (speaker >= 0 && speaker <= 3) return speakerColors[speaker as keyof typeof speakerColors];
  return speakerColors[0];
};

const getSpeakerLabel = (speaker: number): string => {
  if (speaker === -2) return "Silence";
  if (speaker === -1) return "Speaker 1";
  if (speaker >= 0) return `Speaker ${speaker + 1}`;
  return "Speaker 1";
};

export function TranscriptDisplay({ transcriptData, className, onContentUpdate, showLagInfo = true, showSpeakers = true }: TranscriptDisplayProps) {
  const lastContentRef = useRef<string>('');

  // Notify parent when content changes so it can handle scrolling
  useLayoutEffect(() => {
    if (!transcriptData) return;

    // Create content hash to detect actual content changes
    const currentContent = JSON.stringify({
      lines: transcriptData.lines,
      buffer_transcription: transcriptData.buffer_transcription,
      buffer_diarization: transcriptData.buffer_diarization
    });

    // Only notify if content actually changed
    if (currentContent !== lastContentRef.current) {
      lastContentRef.current = currentContent;
      onContentUpdate?.();
    }
  }, [transcriptData, onContentUpdate]);

  if (!transcriptData) {
    return (
      <div className={cn(
        "bg-secondary border border-border rounded-lg p-4 text-muted-foreground text-sm min-h-[200px] transition-all",
        "hover:border-muted-foreground",
        className
      )}>
        <div className="flex items-center justify-center h-full">
          Your audio transcript will appear here...
        </div>
      </div>
    );
  }

  const { 
    lines, 
    buffer_diarization, 
    buffer_transcription, 
    remaining_time_diarization, 
    remaining_time_transcription, 
    isFinalizing,
    diarization_enabled 
  } = transcriptData;

  return (
    <div 
      className={cn(
        "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[200px] transition-all",
        "hover:border-muted-foreground",
        className
      )}
    >
      {lines.length === 0 && !buffer_transcription && !buffer_diarization ? (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          Listening...
        </div>
      ) : (
        <div className="space-y-4">
          {lines.map((line, index) => {
            const isLastLine = index === lines.length - 1;
            let currentLineText = line.text || "";

            // The backend now sends the correct text (parsed or raw) in the line object.
            // We just need to append the buffer to the last line.
            if (isLastLine && !isFinalizing && (buffer_diarization || buffer_transcription)) {
              if (buffer_diarization) {
                currentLineText += `<span class="text-muted-foreground opacity-70 bg-secondary ml-1 px-1 rounded">${buffer_diarization}</span>`;
              }
              if (buffer_transcription) {
                currentLineText += `<span class="text-muted-foreground opacity-80 bg-secondary ml-1 px-1 rounded font-medium">${buffer_transcription}</span>`;
              }
            }

            return (
              <div key={index} className={cn("border-l-3 pl-3 pb-3 transition-all", getSpeakerColor(line.speaker))}>
                <div className="flex items-center flex-wrap gap-2 mb-2">
                  {showSpeakers && (
                    <span className="inline-flex items-center gap-1.5 px-2 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl text-xs font-semibold tracking-wide">
                      {getSpeakerLabel(line.speaker)}
                      {line.beg && line.end && <span className="text-xs opacity-90">{line.beg} - {line.end}</span>}
                    </span>
                  )}
                </div>
                <div
                  className="text-foreground leading-relaxed mt-2 pl-3 pt-2 border-l-0 rounded-bl-lg"
                  dangerouslySetInnerHTML={{ __html: currentLineText }}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
} 