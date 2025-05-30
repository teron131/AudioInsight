"use client";

import { cn } from '@/lib/utils';
import { TranscriptData } from '@/lib/websocket';
import { Loader2 } from 'lucide-react';

interface TranscriptDisplayProps {
  transcriptData: TranscriptData | null;
  className?: string;
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

export function TranscriptDisplay({ transcriptData, className }: TranscriptDisplayProps) {
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
    <div className={cn(
      "bg-secondary border border-border rounded-lg p-4 text-sm min-h-[200px] overflow-y-auto transition-all",
      "hover:border-muted-foreground",
      className
    )}>
      {lines.length === 0 && !buffer_transcription && !buffer_diarization ? (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          Listening...
        </div>
      ) : (
        <div className="space-y-4">
          {lines.map((line, index) => {
            const isLastLine = index === lines.length - 1;
            let currentLineText = line.text || "";
            
            // Only show buffer text for the last line if not finalizing
            if (isLastLine && !isFinalizing && (buffer_diarization || buffer_transcription)) {
              if (buffer_diarization) {
                currentLineText += `<span class="text-muted-foreground opacity-70 bg-secondary ml-1 px-1 rounded">${buffer_diarization}</span>`;
              }
              if (buffer_transcription) {
                currentLineText += `<span class="text-muted-foreground opacity-80 bg-secondary ml-1 px-1 rounded font-medium">${buffer_transcription}</span>`;
              }
            }

            return (
              <div key={index} className={cn(
                "border-l-3 pl-3 pb-3 transition-all",
                getSpeakerColor(line.speaker)
              )}>
                {/* Speaker info and time */}
                <div className="flex items-center flex-wrap gap-2 mb-2">
                  <span className="inline-flex items-center gap-1.5 px-2 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl text-xs font-semibold tracking-wide">
                    {getSpeakerLabel(line.speaker)}
                    {line.beg && line.end && (
                      <span className="text-xs opacity-90">
                        {line.beg} - {line.end}
                      </span>
                    )}
                  </span>
                  
                  {/* Processing indicators for last line */}
                  {isLastLine && !isFinalizing && (
                    <>
                      {remaining_time_transcription && remaining_time_transcription > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-secondary border rounded text-xs text-blue-600 font-medium">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          Transcription lag {remaining_time_transcription.toFixed(1)}s
                        </span>
                      )}
                      {diarization_enabled && buffer_diarization && remaining_time_diarization && remaining_time_diarization > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-secondary border rounded text-xs text-orange-600 font-medium">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          Diarization lag {remaining_time_diarization.toFixed(1)}s
                        </span>
                      )}
                    </>
                  )}
                </div>

                {/* Transcript text */}
                {line.speaker === -2 ? (
                  <div className="text-muted-foreground bg-secondary border rounded-full px-3 py-1 text-xs font-medium inline-block">
                    Silence
                    {line.beg && line.end && (
                      <span className="ml-1 opacity-70">
                        {line.beg} - {line.end}
                      </span>
                    )}
                  </div>
                ) : (
                  <div 
                    className="text-foreground leading-relaxed mt-2 pl-3 pt-2 border-l-0 rounded-bl-lg"
                    dangerouslySetInnerHTML={{ __html: currentLineText }}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
} 